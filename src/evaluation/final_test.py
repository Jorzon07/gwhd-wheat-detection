"""Evaluación FINAL en la partición de prueba. Se ejecuta UNA vez, con todas las decisiones ya tomadas en validación.

Nada se elige aquí: el `spec.json` fija qué modelos entran, de qué carpeta sale cada checkpoint y los parámetros de cada
ensamble (elegidos antes en validación). El umbral de confianza de cada modelo es su mejor F1 EN VALIDACIÓN (summary.json);
mAP no depende de ningún umbral.

Garantías:
  * Sin --confirm-final solo valida el spec y lista lo que haría (no lee ninguna imagen de prueba).
  * Las predicciones de cada modelo sobre prueba se generan una sola vez y se guardan en outputs/final/<modelo>/
    (con el SHA-256 del checkpoint en outputs/final/lock.json). Si ya existen se REUTILIZAN; nunca se vuelven a generar
    con otro checkpoint (error si el SHA cambió).
  * Ensambles: se fusionan las predicciones de prueba de sus miembros con los parámetros del spec.

spec.json:
{"reference": "rtdetr_tuned",
 "models": {"rtdetr_tuned": {"kind": "torch", "run": "outputs/runs/hv_..."},
            "yolov8n":      {"kind": "yolo",  "run": "outputs/runs/yolov8n_..."}},
 "ensembles": {"e3": {"members": ["rtdetr_tuned", "yolov8n"], "iou_thr": 0.55, "skip_thr": 0.05,
                      "weights": null, "conf_type": "avg", "val_summary": "outputs/ensembles/e3/summary.json"}}}

Uso:  python -m src.evaluation.final_test --spec outputs/final/spec.json [--confirm-final] [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/final")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_of(entry: dict, root: Path) -> Path:
    run = root / entry["run"] if not Path(entry["run"]).is_absolute() else Path(entry["run"])
    return run / ("weights/best.pt" if entry["kind"] == "yolo" else "best.pt")


def val_threshold(summary_path: Path) -> float:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(s["best_val_metrics"]["best_f1_threshold"])


def validate_spec(spec: dict, root: Path) -> list[str]:
    errs = []
    if spec.get("reference") not in spec.get("models", {}) and spec.get("reference") not in spec.get("ensembles", {}):
        errs.append("reference no está entre los modelos/ensambles")
    for n, e in spec.get("models", {}).items():
        if e.get("kind") not in ("torch", "yolo"):
            errs.append(f"{n}: kind debe ser torch o yolo")
            continue
        if not checkpoint_of(e, root).exists():
            errs.append(f"{n}: no existe {checkpoint_of(e, root)}")
        if not (Path(root / e["run"]) / "summary.json").exists():
            errs.append(f"{n}: falta summary.json (umbral de validación)")
    for n, e in spec.get("ensembles", {}).items():
        for m in e["members"]:
            if m not in spec["models"]:
                errs.append(f"{n}: miembro {m} no está en models")
        if not (root / e["val_summary"]).exists():
            errs.append(f"{n}: falta {e['val_summary']}")
        for k in ("iou_thr", "skip_thr"):
            if k not in e:
                errs.append(f"{n}: falta {k}")
    return errs


# ---------------------------------------------------------------- predicción sobre prueba ----------
def predict_torch(entry, root, split_csv_name="test"):
    import torch
    from src.config import load_config
    from src.data.dataset import WheatDataset, make_loader
    from src.data.samples import samples_from_config
    from src.models.factory import build_detector
    from src.training.engine import predict

    run = root / entry["run"]
    tcfg = json.loads((run / "config.json").read_text(encoding="utf-8"))["train"]
    cfg = load_config("configs/data.yaml")
    size, amp = int(tcfg["image_size"]), bool(tcfg.get("amp", True))
    samples = samples_from_config(cfg, split_csv_name, train=False, image_size=size)
    lc = cfg["phase3"]["loader"]
    loader = make_loader(WheatDataset(samples), int(tcfg.get("eval_batch_size", 4)), shuffle=False,
                         num_workers=int(lc["num_workers"]), pin_memory=bool(lc["pin_memory"]), seed=0)
    device = torch.device("cuda")
    model = build_detector(tcfg).to(device)
    ckpt = torch.load(checkpoint_of(entry, root), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    original = int(cfg["dataset"]["expected_image_size"][0])
    preds = predict(model, loader, samples.names, device, original / samples.image_size, amp)
    del model
    torch.cuda.empty_cache()
    return preds


def predict_yolo(entry, root, names, images_dir, image_ext):
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    run = root / entry["run"]
    pa = json.loads((run / "summary.json").read_text(encoding="utf-8")).get("predict_args", {})
    fp16 = bool(torch.cuda.is_available())
    prec = ({"quantize": 16} if fp16 else {}) if "quantize" in DEFAULT_CFG_DICT else {"half": fp16}
    kw = dict(imgsz=int(pa.get("imgsz", 800)), conf=float(pa.get("conf", 0.001)), iou=float(pa.get("iou", 0.7)),
              max_det=int(pa.get("max_det", 300)), verbose=False, **prec)
    model = YOLO(str(checkpoint_of(entry, root)))
    paths = [str(images_dir / (n if n.lower().endswith(image_ext) else n + image_ext)) for n in names]
    preds = {}
    for i in range(0, len(paths), 16):
        for n, r in zip(names[i:i + 16], model.predict(paths[i:i + 16], **kw)):
            preds[n] = (r.boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4),
                        r.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1))
    return preds


# ---------------------------------------------------------------- análisis ----------
def per_domain_map(preds, gts, domains, names, evaluate):
    rows = []
    for d in sorted(set(domains)):
        idx = [n for n, dd in zip(names, domains) if dd == d]
        r = evaluate({n: preds[n] for n in idx}, {n: gts[n] for n in idx})
        rows.append({"domain": d, "n_images": len(idx), "map50": r["map50"], "map50_95": r["map50_95"]})
    return pd.DataFrame(rows)


def main() -> None:
    from src.config import ROOT, load_config, resolve
    from src.data.annotations import parse_boxes_string
    from src.evaluation.bootstrap import bootstrap, match_all, metrics_from_matches, paired_differences, summarize
    from src.evaluation.compare_runs import df_to_md
    from src.evaluation.ensemble import fuse_all, load_npz_once
    from src.evaluation.metrics import evaluate
    from src.training.utils import save_predictions, to_builtin

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--confirm-final", action="store_true",
                    help="sin esta bandera NO se lee la partición de prueba: solo se valida el spec")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    root = ROOT
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8-sig"))
    errs = validate_spec(spec, root)
    print(f"Spec: {len(spec['models'])} modelos, {len(spec.get('ensembles', {}))} ensambles, referencia {spec['reference']}")
    if errs:
        raise SystemExit("Spec inválido:\n  " + "\n  ".join(errs))
    out = resolve(OUT)
    lock_path = out / "lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    for n, e in spec["models"].items():
        have = (out / n / "test_predictions.npz").exists()
        print(f"  {n:16s} {e['kind']:5s} {e['run']}  [{'predicciones ya generadas' if have else 'se predecirá'}]")
    for n, e in spec.get("ensembles", {}).items():
        print(f"  {n:16s} WBF {e['members']} iou_thr={e['iou_thr']} skip={e['skip_thr']} pesos={e.get('weights')}")
    if not a.confirm_final:
        print("\nDry-run: no se tocó la partición de prueba. Para ejecutar la evaluación final añade --confirm-final.")
        return

    cfg = load_config("configs/data.yaml")
    split_dir = resolve(cfg["phase3"]["split_dir"])
    test_meta = pd.read_csv(split_dir / "test.csv")
    names = sorted(test_meta["image_name"].astype(str))
    ann = pd.read_csv(split_dir / "annotations_clean.csv", dtype={"image_name": str}).set_index("image_name")
    gts = {n: parse_boxes_string(ann.at[n, "BoxesString"])[0].astype(np.float64) for n in names}
    domains = test_meta.set_index("image_name").loc[names, "domain"].astype(str).tolist()

    preds, thr = {}, {}
    for n, e in spec["models"].items():
        d = out / n
        d.mkdir(parents=True, exist_ok=True)
        sha = sha256_file(checkpoint_of(e, root))
        npz = d / "test_predictions.npz"
        if n in lock and lock[n]["sha256"] != sha:
            raise SystemExit(f"{n}: el checkpoint cambió desde que se predijo sobre prueba (lock.json). Se detiene.")
        if not npz.exists():
            print(f"Prediciendo prueba con {n} ...", flush=True)
            p = predict_torch(e, root) if e["kind"] == "torch" else predict_yolo(
                e, root, names, resolve(cfg["phase3"]["images_dir"]), cfg["dataset"]["image_ext"])
            if set(p) != set(names):
                raise SystemExit(f"{n}: predicciones para {len(p)} imágenes, se esperaban {len(names)}")
            save_predictions(npz, p)
            lock[n] = {"sha256": sha, "run": e["run"], "predicted": datetime.now().isoformat(timespec="seconds")}
            lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        preds[n] = load_npz_once(npz)
        thr[n] = val_threshold(root / e["run"] / "summary.json")
    for n, e in spec.get("ensembles", {}).items():
        fused = fuse_all([preds[m] for m in e["members"]], names, float(e["iou_thr"]), float(e["skip_thr"]),
                         e.get("weights"), e.get("conf_type", "avg"), a.workers)
        d = out / n
        d.mkdir(parents=True, exist_ok=True)
        save_predictions(d / "test_predictions.npz", fused)
        preds[n] = fused
        thr[n] = val_threshold(root / e["val_summary"])
        lock[n] = {"ensemble": e, "predicted": datetime.now().isoformat(timespec="seconds")}
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")

    # ---- métricas (umbral = mejor F1 de VALIDACIÓN) ----
    order = list(preds)
    rows, gl = [], [gts[n] for n in names]
    matched, points = {}, {}
    for m in order:
        r = evaluate(preds[m], gts, score_threshold=thr[m])
        matched[m] = match_all(gl, [preds[m][n] for n in names])
        points[m] = metrics_from_matches(matched[m], None, thr[m])
        rows.append({"model": m, "thr_val": thr[m], "map50": r["map50"], "map50_95": r["map50_95"], "map75": r["map75"],
                     "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})
        if abs(points[m]["map50_95"] - r["map50_95"]) > 1e-4:
            print(f"AVISO {m}: bootstrap {points[m]['map50_95']:.5f} vs evaluador {r['map50_95']:.5f}")
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_metrics.csv", index=False)
    md = [f"# Evaluación final en prueba ({len(names)} imágenes, {len(set(domains))} dominios)", "",
          "Umbral de confianza = mejor F1 en validación. Un solo entrenamiento por modelo.", "", df_to_md(table), ""]
    ref = spec["reference"]
    domarr = np.array(domains)
    for level in ("image", "domain"):
        draws = bootstrap(matched, thr, a.n_boot, 0, level, domarr if level == "domain" else None)
        s = summarize(draws, points)
        s.to_csv(out / f"test_ci_{level}.csv", index=False)
        diffs = pd.concat([paired_differences(draws, m, ref) for m in order if m != ref])
        diffs.to_csv(out / f"test_paired_{level}.csv", index=False)
        label = "imágenes" if level == "image" else f"dominios completos ({len(set(domains))})"
        md += [f"## IC 95 % por remuestreo de {label}", "", df_to_md(s[s.metric.isin(("map50", "map50_95", "f1"))]), "",
               f"Diferencias pareadas (modelo - {ref}):", "", df_to_md(diffs), ""]
    pdm = {m: per_domain_map(preds[m], gts, domains, names, evaluate).set_index("domain")["map50_95"] for m in order}
    dom = pd.DataFrame(pdm)
    dom.insert(0, "n_images", pd.Series(domains).value_counts().reindex(dom.index).to_numpy())
    dom.reset_index().to_csv(out / "test_per_domain.csv", index=False)
    md += ["## mAP50-95 por dominio de prueba", "", df_to_md(dom.reset_index()), ""]
    (out / "test_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:8]))
    print(f"\nInforme completo en {out / 'test_report.md'}")


if __name__ == "__main__":
    main()
