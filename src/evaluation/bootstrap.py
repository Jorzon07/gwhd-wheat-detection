"""Bootstrap de intervalos de confianza para métricas de detección (una sola clase).

Se empareja UNA vez (10 umbrales de IoU) y luego se remuestrea: imágenes ("image") o
dominios completos ("domain"). Cada remuestreo se comparte entre modelos, de modo que las
diferencias pareadas son válidas. El AP reproduce COCOeval (maxDets=300) con cajas válidas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
REC_THRS = np.linspace(0.0, 1.0, 101)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


@dataclass
class MatchedModel:
    """Detecciones de todas las imágenes, ordenadas por puntuación descendente."""
    scores: np.ndarray      # (N,)
    tp: np.ndarray          # (T, N) bool
    img: np.ndarray         # (N,) índice de imagen de cada detección
    n_gt_img: np.ndarray    # (n_images,) espigas reales por imagen


def match_all(gts: list[np.ndarray], preds: list[tuple[np.ndarray, np.ndarray]],
              max_dets: int = 300) -> MatchedModel:
    """gts[i]: (n,4) xyxy; preds[i]: (boxes (m,4), scores (m,)). Mismo orden de imágenes."""
    if len(gts) != len(preds):
        raise ValueError("gts y preds deben tener la misma longitud")
    T = len(IOU_THRS)
    sc_all, tp_all, im_all = [], [], []
    for i, (g, (b, s)) in enumerate(zip(gts, preds)):
        g = np.asarray(g, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
        s = np.asarray(s, float).reshape(-1)
        o = np.argsort(-s, kind="mergesort")[:max_dets]
        b, s = b[o], s[o]
        tp = np.zeros((T, len(s)), bool)
        if len(s) and len(g):
            iou = _iou(b, g)
            for t, thr in enumerate(IOU_THRS):
                taken = np.zeros(len(g), bool)
                for d in range(len(s)):
                    c = np.where(taken, -1.0, iou[d])
                    j = int(np.argmax(c))
                    if c[j] >= min(thr, 1 - 1e-10):
                        taken[j] = True; tp[t, d] = True
        sc_all.append(s); tp_all.append(tp); im_all.append(np.full(len(s), i))
    sc = np.concatenate(sc_all) if sc_all else np.zeros(0)
    tp = np.concatenate(tp_all, axis=1) if tp_all else np.zeros((T, 0), bool)
    im = np.concatenate(im_all) if im_all else np.zeros(0, int)
    o = np.argsort(-sc, kind="mergesort")
    return MatchedModel(sc[o], tp[:, o], im[o], np.array([len(np.asarray(g).reshape(-1, 4)) for g in gts]))


def _ap_per_threshold(tp: np.ndarray, n_gt: float) -> np.ndarray:
    """tp ya ordenado por puntuación. Interpolación COCO de 101 puntos."""
    T = tp.shape[0]
    if n_gt == 0:
        return np.full(T, np.nan)
    if tp.shape[1] == 0:
        return np.zeros(T)
    ctp = np.cumsum(tp, axis=1, dtype=np.float64)
    cfp = np.cumsum(~tp, axis=1, dtype=np.float64)
    rc = ctp / n_gt
    pr = np.maximum.accumulate((ctp / (ctp + cfp))[:, ::-1], axis=1)[:, ::-1]
    ap = np.zeros(T)
    for t in range(T):
        inds = np.searchsorted(rc[t], REC_THRS, side="left")
        ok = inds < pr.shape[1]
        q = np.zeros(len(REC_THRS)); q[ok] = pr[t, inds[ok]]
        ap[t] = q.mean()
    return ap


def metrics_from_matches(m: MatchedModel, counts: np.ndarray | None = None,
                         score_threshold: float = 0.5) -> dict:
    """counts[i] = veces que la imagen i aparece en el remuestreo (None = todas una vez)."""
    if counts is None:
        counts = np.ones(len(m.n_gt_img), dtype=np.int64)
    w = counts[m.img]
    idx = np.repeat(np.arange(len(m.scores)), w)           # conserva el orden por puntuación
    tp = m.tp[:, idx]
    n_gt = float((counts * m.n_gt_img).sum())
    ap = _ap_per_threshold(tp, n_gt)
    keep = m.scores[idx] >= score_threshold
    ntp = float(tp[0, keep].sum()); nd = float(keep.sum())
    prec = ntp / nd if nd else 0.0
    rec = ntp / n_gt if n_gt else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"map50": ap[0], "map75": ap[5], "map50_95": float(np.mean(ap)),
            "precision": prec, "recall": rec, "f1": f1}


def _draw(rng, n_images: int, domains: np.ndarray | None, level: str) -> np.ndarray:
    if level == "image":
        return np.bincount(rng.integers(0, n_images, n_images), minlength=n_images)
    if level == "domain":
        if domains is None:
            raise ValueError("level='domain' requiere el vector de dominios")
        u, inv = np.unique(domains, return_inverse=True)
        dc = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        return dc[inv]
    raise ValueError("level debe ser 'image' o 'domain'")


def bootstrap(models: dict[str, MatchedModel], thresholds: dict[str, float] | None = None,
              n_boot: int = 1000, seed: int = 0, level: str = "image",
              domains: np.ndarray | None = None) -> pd.DataFrame:
    """Devuelve un DataFrame largo: (model, draw, métricas). Mismos remuestreos para todos."""
    names = list(models)
    n_img = len(models[names[0]].n_gt_img)
    if any(len(m.n_gt_img) != n_img for m in models.values()):
        raise ValueError("todos los modelos deben evaluarse sobre las mismas imágenes")
    if domains is not None and len(domains) != n_img:
        raise ValueError("domains debe tener una entrada por imagen")
    thresholds = thresholds or {}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        c = _draw(rng, n_img, domains, level)
        for n in names:
            r = metrics_from_matches(models[n], c, thresholds.get(n, 0.5))
            rows.append({"model": n, "draw": b, **r})
    return pd.DataFrame(rows)


def summarize(draws: pd.DataFrame, points: dict[str, dict], alpha: float = 0.05) -> pd.DataFrame:
    out = []
    for n, g in draws.groupby("model", sort=False):
        for k in ("map50", "map50_95", "map75", "precision", "recall", "f1"):
            v = g[k].dropna().to_numpy()
            lo, hi = np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out.append({"model": n, "metric": k, "point": points[n][k], "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def paired_differences(draws: pd.DataFrame, a: str, b: str, alpha: float = 0.05) -> pd.DataFrame:
    """Diferencia a - b por remuestreo. 'excluye_cero' = el IC no contiene 0."""
    da = draws[draws.model == a].set_index("draw"); db = draws[draws.model == b].set_index("draw")
    out = []
    for k in ("map50", "map50_95", "map75", "precision", "recall", "f1"):
        d = (da[k] - db[k]).dropna().to_numpy()
        lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out.append({"a": a, "b": b, "metric": k, "mean_diff": d.mean(), "lo": lo, "hi": hi,
                    "excluye_cero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(out)
