"""Mide el rendimiento del DataLoader (imágenes/s) para elegir num_workers y batch_size en esta máquina.

Uso: python -m src.data.bench_loader [--split train] [--batches 60] [--warmup 12] [--workers 0 2 4 6 8]
"""
from __future__ import annotations

import argparse
import time


def main() -> None:
    from src.config import load_config
    from src.data.dataset import WheatDataset, make_loader, set_epoch
    from src.data.samples import samples_from_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--batches", type=int, default=60, help="lotes medidos tras el calentamiento")
    ap.add_argument("--warmup", type=int, default=12, help="lotes descartados (arranque de workers y prefetch)")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    lc = cfg["phase3"]["loader"]
    bs = args.batch_size or int(lc["batch_size"])
    ds = WheatDataset(samples_from_config(cfg, args.split, image_size=args.size))
    print(f"Partición {args.split}: {len(ds)} imágenes | tamaño {ds.samples.image_size} | batch_size {bs}")
    for nw in args.workers:
        loader = make_loader(ds, bs, shuffle=True, num_workers=nw, pin_memory=bool(lc["pin_memory"]), seed=0)
        set_epoch(loader, 0)
        t0, n_img, n_box = time.perf_counter(), 0, 0
        for i, (images, targets) in enumerate(loader):
            if i < args.warmup:                                   # el arranque de los workers no cuenta
                t0 = time.perf_counter()
                continue
            n_img += len(images)
            n_box += sum(len(t["boxes"]) for t in targets)
            if i >= args.warmup + args.batches:
                break
        dt = time.perf_counter() - t0
        print(f"  num_workers={nw}: {n_img / dt:6.1f} imágenes/s ({n_box / max(n_img, 1):.1f} cajas/imagen)")


if __name__ == "__main__":
    main()