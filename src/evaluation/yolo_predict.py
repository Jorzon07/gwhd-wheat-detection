"""Predicciones de validación de un run de YOLO en el formato común + resumen con el evaluador del proyecto.

Escribe en <run>/: val_predictions_best.npz (names, offsets, boxes xyxy en píxeles originales, scores), summary.json
(best_val_metrics con el umbral de mejor F1 en validación) y, si no existe, config.json. Solo validación.
Uso: python -m src.evaluation.yolo_predict --run outputs/runs/yolov8n_800_...
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd


def predict_run(run: Path, data_yaml: str, split_dir: Path, imgsz: int = 800, conf: float = 0.001, iou: float = 0.7,
                max_det: int = 300, device=None, batch: int = 16) -> dict:
    import torch
    from ultralytics import YOLO
    from src.data.annotations import parse_boxes_string
    from src.evaluation.metrics import evaluate
    from src.training.utils import load_predictions

    run = Path(run)
    model = YOLO(str(run / "weights" / "best.pt"))
    val_list = Path(data_yaml).resolve().parent / "val.txt"
    paths = [p for p in val_list.read_text(encoding="utf-8").splitlines() if p.strip()]
    from ultralytics.cfg import DEFAULT_CFG_DICT
    fp16 = bool(torch.cuda.is_available())
    prec = ({"quantize": 16} if fp16 else {}) if "quantize" in DEFAULT_CFG_DICT else {"half": fp16}   # 'half' está obsoleto
    kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False, **prec,
              **({"device": device} if device is not None else {}))
    names, boxes, scores, off = [], [], [], [0]
    for i in range(0, len(paths), batch):
        for p, r in zip(paths[i:i + batch], model.predict(paths[i:i + batch], **kw)):
            b = r.boxes
            boxes.append(b.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4))
            scores.append(b.conf.cpu().numpy().astype(np.float32).reshape(-1))
            names.append(Path(p).name); off.append(off[-1] + len(scores[-1]))
    npz = run / "val_predictions_best.npz"
    np.savez(npz, names=np.array(names), offsets=np.array(off, dtype=np.int64),
             boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
             scores=np.concatenate(scores) if scores else np.zeros(0, np.float32))

    ann = pd.read_csv(Path(split_dir) / "annotations_clean.csv", dtype={"image_name": str}).set_index("image_name")
    gts = {n: parse_boxes_string(ann.at[n, "BoxesString"])[0].astype(np.float64) for n in names}
    res = evaluate(load_predictions(npz), gts)

    sample = paths[:110]                         # tiempo por imagen (lote 1, incluye preprocesado y NMS)
    ts = []
    for p in sample:
        t = time.perf_counter(); model.predict(p, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    ts = ts[10:]
    summary = {"run_dir": str(run), "smoke": False, "selection_metric": "fitness de Ultralytics (0.1 mAP50 + 0.9 mAP50-95)",
               "best_val_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.items()},
               "predict_args": {"imgsz": imgsz, "conf": conf, "iou": iou, "max_det": max_det},
               "inference_ms_mean": float(statistics.mean(ts)), "inference_ms_median": float(statistics.median(ts)),
               "inference_n_images": len(ts)}
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not (run / "config.json").exists():
        (run / "config.json").write_text(json.dumps({"split_dir": str(split_dir),
                                                     "phase3": {"images_dir": "data/raw/gwhd_2021/images"}}, indent=2),
                                         encoding="utf-8")
    m = summary["best_val_metrics"]
    print(f"YOLO val | mAP50 {m['map50']:.3f} | mAP50-95 {m['map50_95']:.3f} | mejor F1 {m.get('best_f1', float('nan')):.3f} "
          f"@ {m.get('best_f1_threshold', float('nan')):.3f} | {summary['inference_ms_median']:.1f} ms/imagen (mediana)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/yolo/gwhd.yaml")
    ap.add_argument("--split-dir", default="data/splits/location_seed42")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--iou", type=float, default=0.7, help="umbral de NMS (por defecto el de validación de Ultralytics)")
    a = ap.parse_args()
    predict_run(Path(a.run), a.data, Path(a.split_dir), a.imgsz, iou=a.iou)


if __name__ == "__main__":
    main()
