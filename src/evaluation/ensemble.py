"""Ensamble de detectores por fusión ponderada de cajas (WBF) sobre predicciones ya guardadas. Sin GPU.

Lee <run>/val_predictions_best.npz de cada miembro (cajas xyxy en el marco 1024), las fusiona con
`ensemble_boxes.weighted_boxes_fusion` y evalúa con el evaluador común. SOLO validación: el umbral de fusión se elige
aquí y después se congela. Escribe una carpeta con el mismo formato que un run (val_predictions_best.npz, summary.json,
config.json), así que sirve con compare_runs.

Uso:
  python -m src.evaluation.ensemble --name e3 --members tuned9=outputs/runs/hv_... yolo=outputs/runs/yolov8n_... \
      frcnn=outputs/runs/baseline_v1_... [--iou-thr 0.55 0.65] [--skip-thr 0.05] [--weights 1 1 1] [--workers 4]
Nota: WBF promedia puntuaciones de modelos con calibraciones distintas (sigmoide de RT-DETR, confianza de YOLO, etc.);
no se recalibran. `conf_type='avg'` penaliza las cajas que no encuentran todos los modelos.
"""
from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FRAME = 1024.0


def load_npz_once(path) -> dict:
    """Como utils.load_predictions, pero descomprime cada arreglo UNA vez (aquélla lo hace en cada imagen y cada
    trozo conserva el arreglo completo vivo: varios GB con 984 imágenes)."""
    z = np.load(path, allow_pickle=False)
    names, off, boxes, scores = [str(n) for n in z["names"]], z["offsets"], z["boxes"], z["scores"]
    return {n: (boxes[off[i]:off[i + 1]].copy(), scores[off[i]:off[i + 1]].copy()) for i, n in enumerate(names)}


def fuse_image(item):
    """item = (lista de cajas px por modelo, lista de puntuaciones, iou_thr, skip_thr, pesos, conf_type)."""
    from ensemble_boxes import weighted_boxes_fusion
    boxes_l, scores_l, iou_thr, skip_thr, weights, conf_type = item
    bl, sl, ll = [], [], []
    for b, s in zip(boxes_l, scores_l):
        b = np.clip(np.asarray(b, dtype=np.float64).reshape(-1, 4) / FRAME, 0.0, 1.0)
        bl.append(b)
        sl.append(np.asarray(s, dtype=np.float64).reshape(-1))
        ll.append(np.zeros(len(sl[-1])))
    if sum(len(s) for s in sl) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                 # avisa de cajas de área cero, que WBF descarta
        fb, fs, _ = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_thr,
                                          conf_type=conf_type)
    order = np.argsort(-fs, kind="stable")[:300]
    return fb[order] * FRAME, fs[order]


def fuse_all(preds_list: list, names: list, iou_thr: float, skip_thr: float, weights=None, conf_type: str = "avg",
             workers: int = 1) -> dict:
    """preds_list: [{nombre: (cajas, puntuaciones)}] por modelo. Devuelve {nombre: (cajas px, puntuaciones)}."""
    items = [([p[n][0] for p in preds_list], [p[n][1] for p in preds_list], iou_thr, skip_thr, weights, conf_type)
             for n in names]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(fuse_image, items, chunksize=16))
    else:
        res = [fuse_image(it) for it in items]
    return dict(zip(names, res))


def main() -> None:
    import pandas as pd
    from src.config import resolve
    from src.data.annotations import parse_boxes_string
    from src.evaluation.metrics import evaluate
    from src.training.utils import save_predictions, to_builtin

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="+", required=True, help="nombre=carpeta_del_run")
    ap.add_argument("--iou-thr", nargs="+", type=float, default=[0.55, 0.65])
    ap.add_argument("--skip-thr", type=float, default=0.05)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--conf-type", default="avg", choices=["avg", "max", "box_and_model_avg", "absent_model_aware_avg"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split-dir", default="data/splits/location_seed42")
    ap.add_argument("--out-root", default="outputs/ensembles")
    a = ap.parse_args()
    if a.weights is not None and len(a.weights) != len(a.members):
        raise SystemExit("--weights debe tener un valor por miembro")

    members = {}
    for it in a.members:
        n, _, p = it.partition("=")
        if not p:
            raise SystemExit(f"Formato incorrecto '{it}': use nombre=carpeta")
        members[n] = resolve(p)
    preds = {n: load_npz_once(p / "val_predictions_best.npz") for n, p in members.items()}
    names = sorted(next(iter(preds.values())))
    for n, pr in preds.items():
        if set(pr) != set(names):
            raise SystemExit(f"El miembro {n} no tiene las mismas imágenes de validación que el primero")

    split_dir = resolve(a.split_dir)
    ann = pd.read_csv(split_dir / "annotations_clean.csv", dtype={"image_name": str}).set_index("image_name")
    gts = {n: parse_boxes_string(ann.at[n, "BoxesString"])[0].astype(np.float64) for n in names}

    best, rows = None, []
    for thr in a.iou_thr:
        fused = fuse_all(list(preds.values()), names, thr, a.skip_thr, a.weights, a.conf_type, a.workers)
        res = evaluate(fused, gts)
        rows.append((thr, res))
        print(f"iou_thr {thr:.2f} | mAP50 {res['map50']:.4f} | mAP50-95 {res['map50_95']:.4f} | "
              f"P {res['precision']:.3f} R {res['recall']:.3f} F1 {res['f1']:.3f}", flush=True)
        if best is None or res["map50_95"] > best[1]["map50_95"]:
            best = (thr, res, fused)
    thr, res, fused = best
    out = resolve(a.out_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    save_predictions(out / "val_predictions_best.npz", fused)
    summary = {"run_dir": str(out), "smoke": False, "selection_metric": "map50_95 (iou_thr elegido en validación)",
               "best_val_metrics": to_builtin({k: v for k, v in res.items() if isinstance(v, (int, float, np.floating))}),
               "ensemble": {"members": {n: str(p) for n, p in members.items()}, "method": "WBF", "iou_thr": thr,
                            "iou_thr_grid": a.iou_thr, "skip_box_thr": a.skip_thr, "weights": a.weights,
                            "conf_type": a.conf_type}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps({"split_dir": str(split_dir), "phase3": {"images_dir": "data/raw/gwhd_2021/images"},
                                                 "ensemble": summary["ensemble"]}, indent=2), encoding="utf-8")
    print(f"\nElegido iou_thr {thr:.2f}: mAP50 {res['map50']:.4f} | mAP50-95 {res['map50_95']:.4f} | "
          f"mejor F1 {res['best_f1']:.3f} con umbral {res['best_f1_threshold']:.3f}\nGuardado en {out}")


if __name__ == "__main__":
    main()
