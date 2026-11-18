"""Evaluador común de detección de espigas. NumPy + pycocotools (sin PyTorch).

Todas las arquitecturas se evalúan con este mismo código, en el marco de coordenadas de la imagen ORIGINAL
(1024x1024): si un modelo trabaja a otra resolución, hay que reescalar sus cajas antes de evaluar.

* mAP@50 y mAP@50-95: COCOeval de pycocotools con una sola clase. Se fija maxDets=300 (el valor por defecto de COCO,
  100, es insuficiente: hay imágenes con hasta ~190 espigas y recortaría las detecciones válidas).
* Precision, recall y F1 a IoU=0.5 con un umbral de confianza: emparejamiento voraz por puntuación descendente
  (cada detección se asigna a la caja real libre con mayor IoU >= 0.5), el mismo criterio que COCOeval.
* Umbral de mejor F1: se obtiene barriendo todas las puntuaciones; debe calcularse SOLO con validación y aplicarse
  después a prueba.
"""
from __future__ import annotations

import contextlib
import io
from typing import Mapping

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = 300
CATEGORY_ID = 1


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU por pares entre cajas xyxy: (len(a), len(b))."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_image(det_boxes: np.ndarray, det_scores: np.ndarray, gt_boxes: np.ndarray, iou_thr: float = 0.5):
    """Emparejamiento voraz en una imagen. Devuelve (puntuaciones ordenadas desc., es_TP) por detección."""
    order = np.argsort(-np.asarray(det_scores, dtype=np.float64), kind="stable")
    scores = np.asarray(det_scores, dtype=np.float64)[order]
    boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)[order]
    is_tp = np.zeros(len(order), dtype=bool)
    if len(order) and len(gt_boxes):
        iou = box_iou_np(boxes, gt_boxes)
        taken = np.zeros(len(gt_boxes), dtype=bool)
        for i in range(len(order)):
            cand = np.where(taken, -1.0, iou[i])
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                taken[j] = True
                is_tp[i] = True
    return scores, is_tp


def _prepare(preds: Mapping[str, tuple], gts: Mapping[str, np.ndarray], max_dets: int):
    """Une predicciones y verdad terreno por nombre de imagen y se queda con las max_dets mejores por imagen."""
    missing = set(gts) - set(preds)
    if missing:
        raise ValueError(f"Faltan predicciones para {len(missing)} imágenes (p. ej. {sorted(missing)[:3]})")
    extra = set(preds) - set(gts)
    if extra:
        raise ValueError(f"Hay predicciones de imágenes sin verdad terreno (p. ej. {sorted(extra)[:3]})")
    names = sorted(gts)
    out = {}
    for n in names:
        boxes, scores = preds[n]
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(boxes) != len(scores):
            raise ValueError(f"{n}: {len(boxes)} cajas y {len(scores)} puntuaciones")
        if len(scores) > max_dets:
            top = np.argsort(-scores, kind="stable")[:max_dets]
            boxes, scores = boxes[top], scores[top]
        out[n] = (boxes, scores, np.asarray(gts[n], dtype=np.float64).reshape(-1, 4))
    return names, out


def _mean_valid(arr: np.ndarray) -> float:
    v = arr[arr > -1]                                           # COCO marca con -1 lo que no tiene datos
    return float(v.mean()) if v.size else float("nan")


def _coco_metrics(names, data, image_size, max_dets):
    """mAP y AR de COCOeval calculados desde sus tensores internos.

    No se usa `ev.stats`: el summarize() de pycocotools fija maxDets=100 al calcular AP@[.50:.95], de modo que con
    maxDets=[1, 10, 300] ese valor sale como -1 sin avisar.
    """
    nan = float("nan")
    keys = ("map50_95", "map50", "map75", "map_small", "map_medium", "map_large", "ar_max")
    w, h = image_size
    images = [{"id": i, "file_name": n, "width": w, "height": h} for i, n in enumerate(names)]
    anns, dets, aid = [], [], 1
    for i, n in enumerate(names):
        b, s, g = data[n]
        for x1, y1, x2, y2 in g:
            anns.append({"id": aid, "image_id": i, "category_id": CATEGORY_ID, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
            aid += 1
        for (x1, y1, x2, y2), sc in zip(b, s):
            dets.append({"image_id": i, "category_id": CATEGORY_ID, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(sc)})
    if not anns:
        return dict.fromkeys(keys, nan)
    if not dets:
        return dict.fromkeys(keys, 0.0)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):                      # pycocotools imprime mucho
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": CATEGORY_ID, "name": "wheat_head"}]}
        gt.createIndex()
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(range(len(names)))
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()
    m_idx = ev.params.maxDets.index(max_dets)
    labels = list(ev.params.areaRngLbl)                          # ['all', 'small', 'medium', 'large']
    prec = ev.eval["precision"]                                  # [IoU, recall, clase, área, maxDets]
    rec = ev.eval["recall"]                                      # [IoU, clase, área, maxDets]
    t50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    t75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    ap = lambda t, a: _mean_valid(prec[t, :, :, labels.index(a), m_idx])      # noqa: E731
    ap_all = lambda a: _mean_valid(prec[:, :, :, labels.index(a), m_idx])     # noqa: E731
    return {"map50_95": ap_all("all"), "map50": ap(t50, "all"), "map75": ap(t75, "all"),
            "map_small": ap_all("small"), "map_medium": ap_all("medium"), "map_large": ap_all("large"),
            "ar_max": _mean_valid(rec[:, :, labels.index("all"), m_idx])}


def evaluate(preds: Mapping[str, tuple], gts: Mapping[str, np.ndarray], image_size=(1024, 1024),
             score_threshold: float = 0.5, iou_threshold: float = 0.5, max_dets: int = MAX_DETS) -> dict:
    """Evalúa un conjunto de imágenes.

    preds[nombre] = (cajas Nx4 xyxy, puntuaciones N); gts[nombre] = cajas reales Mx4 xyxy. Todas las imágenes de
    `gts` deben estar en `preds` (con arrays vacíos si no hay detecciones).
    Devuelve un dict con mAP (COCO), precision/recall/F1 al umbral dado y el umbral de mejor F1.
    """
    names, data = _prepare(preds, gts, max_dets)
    n_gt = int(sum(len(data[n][2]) for n in names))
    n_det = int(sum(len(data[n][1]) for n in names))

    scores_all, tp_all = [], []
    for n in names:
        b, s, g = data[n]
        sc, tp = match_image(b, s, g, iou_threshold)
        scores_all.append(sc)
        tp_all.append(tp)
    scores = np.concatenate(scores_all) if scores_all else np.zeros(0)
    tp = np.concatenate(tp_all) if tp_all else np.zeros(0, dtype=bool)

    def prf(mask):
        t = int((tp & mask).sum())
        f = int((~tp & mask).sum())
        p = t / (t + f) if (t + f) else 0.0
        r = t / n_gt if n_gt else 0.0
        return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)

    p, r, f1 = prf(scores >= score_threshold)

    best = {"best_f1": 0.0, "best_f1_threshold": float("nan"), "best_f1_precision": 0.0, "best_f1_recall": 0.0}
    if len(scores) and n_gt:
        order = np.argsort(-scores, kind="stable")
        s_sorted, tp_sorted = scores[order], tp[order]
        ctp, cfp = np.cumsum(tp_sorted), np.cumsum(~tp_sorted)
        prec, rec = ctp / (ctp + cfp), ctp / n_gt
        f1s = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
        valid = np.r_[s_sorted[1:] != s_sorted[:-1], True]          # un solo candidato por puntuación repetida
        idx = np.where(valid)[0]
        k = idx[int(np.argmax(f1s[idx]))]
        best = {"best_f1": float(f1s[k]), "best_f1_threshold": float(s_sorted[k]),
                "best_f1_precision": float(prec[k]), "best_f1_recall": float(rec[k])}

    coco = _coco_metrics(names, data, image_size, max_dets)
    return {**coco, "precision": p, "recall": r, "f1": f1, "score_threshold": score_threshold,
            "iou_threshold": iou_threshold, **best, "n_images": len(names), "n_gt": n_gt, "n_dets": n_det,
            "max_dets": max_dets}


def evaluate_by_group(preds: Mapping[str, tuple], gts: Mapping[str, np.ndarray], groups: Mapping[str, str],
                      **kwargs) -> pd.DataFrame:
    """Métricas por grupo (p. ej. dominio o país). `groups[nombre_imagen] = etiqueta`."""
    rows = []
    for g in sorted(set(groups[n] for n in gts)):
        names = [n for n in gts if groups[n] == g]
        res = evaluate({n: preds[n] for n in names}, {n: gts[n] for n in names}, **kwargs)
        rows.append({"group": g, **res})
    return pd.DataFrame(rows)