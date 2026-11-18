"""Sonda de recursos: pico de memoria y velocidad de entrenamiento/inferencia para cada (variante, tamaño, lote).

Usa los pesos preentrenados reales (se descargan una vez) para que las capas congeladas coincidan con el entrenamiento
real. Un lote que no cabe se registra como OOM y la sonda continúa.
Uso: python -m src.training.probe [--variants v1 v2] [--sizes 640 800 1024] [--batch-sizes 1 2 4] [--steps 8]
"""
from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime

import numpy as np
import torch

from src.config import load_config, resolve
from src.data.dataset import WheatDataset, make_loader
from src.data.samples import samples_from_config
from src.models.factory import build_detector, count_parameters
from src.training.utils import CsvLogger


def run_config(cfg, tcfg, variant, size, bs, steps, warmup, amp, device) -> dict:
    row = {"variant": variant, "image_size": size, "batch_size": bs, "status": "ok"}
    model = optimizer = loader = it = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_detector({**tcfg, "variant": variant, "image_size": size, "pretrained": True}).to(device).train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=1e-4, momentum=0.9)       # lr mínimo: solo se mide recursos
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        lc = cfg["phase3"]["loader"]
        loader = make_loader(WheatDataset(samples_from_config(cfg, "train", image_size=size)), bs, shuffle=True,
                             num_workers=int(lc["num_workers"]), pin_memory=bool(lc["pin_memory"]), seed=0)
        it = iter(loader)
        times, images = [], None
        for k in range(warmup + steps):
            images, targets = next(it)
            images = [im.to(device) for im in images]
            targets = [{a: b.to(device) for a, b in t.items()} for t in targets]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = sum(model(images, targets).values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if k >= warmup:
                times.append(time.perf_counter() - t0)
        row.update(step_s=round(float(np.mean(times)), 3), train_img_per_s=round(bs / float(np.mean(times)), 2),
                   train_peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                   train_peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2))
        model.eval()                                                      # inferencia con lote 1
        x, inf = images[0], []
        with torch.no_grad():
            for k in range(12):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    model([x])
                torch.cuda.synchronize()
                if k >= 2:
                    inf.append(time.perf_counter() - t0)
        row["inference_ms"] = round(1000 * float(np.median(inf)), 1)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    finally:
        del model, optimizer, loader, it
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--variants", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    ap.add_argument("--arch", choices=["frcnn", "retinanet", "detr", "rtdetr"], default=None, help="arquitectura a medir")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 800, 1024])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = dict(cfg["train"])
    if args.arch:
        tcfg["arch"] = args.arch
    variants = args.variants if tcfg.get("arch", "frcnn") == "frcnn" else ["v1"]   # RetinaNet: solo v1
    device = torch.device("cuda")
    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = CsvLogger(out_dir / f"probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB")
    for variant in variants:
        try:
            m = build_detector({**tcfg, "variant": variant, "image_size": 1024, "pretrained": True})
        except Exception as e:  # noqa: BLE001  (descarga de pesos: red, DNS, disco...)
            print(f"\nVariante {variant}: no se pudieron obtener los pesos ({type(e).__name__}: {e}). Se omite; "
                  f"comprueba la conexión y repite con --variants {variant}")
            continue
        tot, trn = count_parameters(m)
        print(f"\nVariante {variant}: {tot / 1e6:.1f} M parámetros ({trn / 1e6:.1f} M entrenables)")
        del m
        for size in args.sizes:
            for bs in args.batch_sizes:
                row = run_config(cfg, tcfg, variant, size, bs, args.steps, args.warmup, not args.no_amp, device)
                log.log(row)
                if row["status"] == "ok":
                    print(f"  tamaño {size:4d} lote {bs}: {row['step_s']:.2f} s/paso, {row['train_img_per_s']:.1f} img/s, "
                          f"pico {row['train_peak_alloc_gib']:.2f} GiB (reservado {row['train_peak_reserved_gib']:.2f}), "
                          f"inferencia {row['inference_ms']:.0f} ms", flush=True)
                else:
                    print(f"  tamaño {size:4d} lote {bs}: {row['status']}", flush=True)
    print("\nResultados en:", log.path)


if __name__ == "__main__":
    main()