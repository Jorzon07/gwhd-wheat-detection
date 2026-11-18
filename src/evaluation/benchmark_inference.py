"""Benchmark de tiempo de inferencia por imagen, con un protocolo ÚNICO para todos los modelos.

Cada medida es de extremo a extremo: imagen RGB uint8 de 1024x1024 en memoria -> (reescalado y normalización) -> transferencia
a la GPU -> red -> posprocesado (NMS o top-k) -> cajas en la CPU. Lote 1, precisión fp16 como en la evaluación, GPU
sincronizada antes y después de cada llamada, calentamiento descartado. Usa imágenes de VALIDACIÓN (la prueba ya se leyó una vez).
Los modelos federados y el de GAN tienen la misma arquitectura que RT-DETR ajustado: se miden una sola vez.
El ensamble e3 se estima como la suma de sus tres miembros más el tiempo de WBF en CPU (se ejecutan en serie).

Condiciones recomendadas: portátil ENCHUFADO, plan de energía "alto rendimiento", sin otros programas usando la GPU.
Uso:  python -m src.evaluation.benchmark_inference [--n 100] [--sweeps 2]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def timing_stats(ms) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(a.min()), "n": int(a.size)}


def bench_callable(fn, imgs: list, warmup: int = 15, sync=lambda: None) -> list:
    """Tiempos (ms) de fn(img) para cada imagen, tras `warmup` llamadas descartadas."""
    for img in imgs[:warmup]:
        fn(img)
        sync()
    out = []
    for img in imgs:
        sync()
        t0 = time.perf_counter()
        fn(img)
        sync()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def main() -> None:
    import torch
    from src.config import ROOT, load_config, resolve
    from src.data.samples import samples_from_config
    from src.data.transforms import apply_eval_transforms
    from src.evaluation.ensemble import fuse_image, load_npz_once
    from src.models.factory import build_detector

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="outputs/final/spec.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sweeps", type=int, default=3, help="barridos medidos (se añade uno previo de calentamiento, descartado)")
    ap.add_argument("--warmup", type=int, default=15)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Se necesita la GPU (CUDA).")
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8-sig"))
    models = {k: spec["models"][k] for k in ("rtdetr_tuned", "yolov8n", "frcnn", "retinanet", "detr")}

    cfg = load_config("configs/data.yaml")
    val = samples_from_config(cfg, "validation", train=False, image_size=1024)
    imgs = [val.get(i)[0] for i in range(a.n + a.warmup)]
    sync = torch.cuda.synchronize
    device = torch.device("cuda")

    def make_torch(entry):
        run = ROOT / entry["run"]
        tcfg = json.loads((run / "config.json").read_text(encoding="utf-8"))["train"]
        size, amp = int(tcfg["image_size"]), bool(tcfg.get("amp", True))
        model = build_detector(tcfg).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["model"])
        model.eval()

        @torch.no_grad()
        def fn(img):
            im, _ = apply_eval_transforms(img, np.zeros((0, 4), np.float32), size)
            x = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div_(255.0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model([x])[0]
            return out["boxes"].float().cpu().numpy(), out["scores"].float().cpu().numpy()
        return fn, sum(p.numel() for p in model.parameters()), model

    def make_yolo(entry):
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        run = ROOT / entry["run"]
        pa = json.loads((run / "summary.json").read_text(encoding="utf-8")).get("predict_args", {})
        prec = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
        kw = dict(imgsz=int(pa.get("imgsz", 800)), conf=float(pa.get("conf", 0.001)), iou=float(pa.get("iou", 0.7)),
                  max_det=int(pa.get("max_det", 300)), verbose=False, **prec)
        model = YOLO(str(run / "weights" / "best.pt"))

        def fn(img):
            r = model.predict(img, **kw)[0]
            return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return fn, sum(p.numel() for p in model.model.parameters()), model

    rows, per_sweep = {}, {}
    order = list(models)
    for s in range(a.sweeps + 1):                                   # s=0: barrido de calentamiento (reloj de GPU/CPU), descartado
        seq = order if s % 2 == 0 else order[::-1]                 # orden inverso en barridos alternos: deriva térmica
        for name in seq:
            e = models[name]
            fn, params, keep = (make_yolo if e["kind"] == "yolo" else make_torch)(e)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ms = bench_callable(fn, imgs[:a.n + a.warmup], a.warmup, sync)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20
            if s > 0:
                rows.setdefault(name, {"ms": [], "params": params, "peak": 0.0})
                rows[name]["ms"] += ms
                rows[name]["peak"] = max(rows[name]["peak"], peak)
                per_sweep.setdefault(name, []).append(float(np.median(ms)))
            tag = "calentamiento, descartado" if s == 0 else f"barrido {s}/{a.sweeps}"
            print(f"[{tag}] {name:14s} mediana {np.median(ms):6.1f} ms | p95 {np.percentile(ms, 95):6.1f} ms", flush=True)
            del fn, keep
            torch.cuda.empty_cache()
            time.sleep(5)                                           # deja enfriar un poco entre modelos

    # WBF en CPU, con las predicciones de validación ya guardadas de los miembros de e3
    members = spec["ensembles"]["e3"]["members"]
    preds = [load_npz_once(ROOT / spec["models"][m]["run"] / "val_predictions_best.npz") for m in members]
    names = sorted(preds[0])[:a.n]
    e3 = spec["ensembles"]["e3"]
    wbf = bench_callable(lambda n: fuse_image(([p[n][0] for p in preds], [p[n][1] for p in preds], e3["iou_thr"],
                                              e3["skip_thr"], None, "avg")), names, warmup=5)

    out = []
    for name, r in rows.items():
        st = timing_stats(r["ms"])
        out.append({"model": name, "params_M": round(r["params"] / 1e6, 1), **{k: round(v, 2) for k, v in st.items() if k != "n"},
                    "fps_median": round(1000.0 / st["median_ms"], 1), "peak_gpu_mib": round(r["peak"], 0),
                    "median_por_barrido": [round(x, 1) for x in per_sweep[name]], "n_medidas": st["n"],
                    "variacion_barridos_pct": round(100 * (max(per_sweep[name]) - min(per_sweep[name])) / st["median_ms"], 1)})
    st_w = timing_stats(wbf)
    e3_ms = sum(next(o["median_ms"] for o in out if o["model"] == m) for m in members) + st_w["median_ms"]
    out.append({"model": "e3 (estimado)", "params_M": round(sum(next(o["params_M"] for o in out if o["model"] == m) for m in members), 1),
                "median_ms": round(e3_ms, 2), "fps_median": round(1000.0 / e3_ms, 1),
                "nota": f"suma de medianas de {members} + WBF en CPU ({st_w['median_ms']:.1f} ms)"})
    dst = resolve("outputs/final")
    (dst / "benchmark.json").write_text(json.dumps({"results": out, "device": torch.cuda.get_device_name(0),
                                                    "torch": torch.__version__, "n_images": a.n, "sweeps": a.sweeps}, indent=2),
                                        encoding="utf-8")
    unstable = [o["model"] for o in out if o.get("variacion_barridos_pct", 0) > 15]
    if unstable:
        print(f"\nAVISO: las medianas de {unstable} varían más de un 15 % entre barridos: medida INESTABLE, no la uses sin repetirla "
              "(enchufa el portátil, plan de energía de alto rendimiento y cierra otros programas).")
    cols = ["model", "params_M", "median_ms", "p95_ms", "fps_median", "peak_gpu_mib", "median_por_barrido", "variacion_barridos_pct"]
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for o in out:
        md.append("| " + " | ".join(str(o.get(c, "")) for c in cols) + " |")
    (dst / "benchmark.md").write_text("\n".join(md) + f"\n\nGPU: {torch.cuda.get_device_name(0)}; lote 1; fp16; extremo a extremo.\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nGuardado en {dst / 'benchmark.md'}")


if __name__ == "__main__":
    main()
