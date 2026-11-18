"""Entrena YOLOv8n (Ultralytics, licencia AGPL-3.0) sobre GWHD y luego genera las predicciones de validación en el
formato común (para compare_runs / analyze_run). Solo usa train y validación.

Uso: python -m src.training.train_yolo [--epochs 50 | --hours 1.5] [--imgsz 800] [--batch 16]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/gwhd.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hours", type=float, default=None, help="presupuesto de tiempo; sustituye a --epochs")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split-dir", default="data/splits/location_seed42")
    ap.add_argument("--no-predict", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    import ultralytics

    project = Path("outputs/runs").resolve()
    name = f"yolov8n_{a.imgsz}_{time.strftime('%Y%m%d_%H%M%S')}"
    model = YOLO(a.weights)
    kw = dict(data=str(Path(a.data).resolve()), imgsz=a.imgsz, batch=a.batch, patience=a.patience, workers=a.workers,
              seed=a.seed, project=str(project), name=name, exist_ok=False, plots=False,
              flipud=0.5, fliplr=0.5, degrees=0.0)        # volteos vertical y horizontal, como el resto de modelos
    kw["time" if a.hours else "epochs"] = a.hours if a.hours else a.epochs
    if a.device is not None:
        kw["device"] = a.device
    t0 = time.time()
    model.train(**kw)
    run = project / name
    (run / "config.json").write_text(json.dumps({
        "split_dir": str(Path(a.split_dir)), "phase3": {"images_dir": "data/raw/gwhd_2021/images"},
        "model": "yolov8n", "ultralytics": ultralytics.__version__, "train_args": {k: str(v) for k, v in kw.items()},
        "wall_time_s": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    if not a.no_predict:
        from src.evaluation.yolo_predict import predict_run
        predict_run(run, a.data, Path(a.split_dir), a.imgsz, device=a.device)


if __name__ == "__main__":
    main()
