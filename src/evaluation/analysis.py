"""Análisis de los resultados de un detector en validación. Sin PyTorch.

Todo se calcula en el marco de la imagen ORIGINAL (1024 px), con las predicciones guardadas de la mejor época.
"""
from __future__ import annotations

from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.evaluation.metrics import box_iou_np, evaluate, match_image  # noqa: E402

SIZE_BINS = (0, 20, 40, 60, 100, np.inf)             # lado equivalente sqrt(ancho*alto) de la caja real, en px
DENSITY_BINS = (1, 21, 41, 61, 101, np.inf)          # espigas reales por imagen (las imágenes sin espigas van aparte)


def _bin_labels(bins, integer: bool = False) -> list[str]:
    """Etiquetas de intervalos [lo, hi). Con integer=True (conteos) se muestran inclusivos: 1-20, 21-40, ..., 101+."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isinf(hi):
            out.append(f"{int(lo)}+")
        else:
            out.append(f"{int(lo)}-{int(hi) - 1 if integer else int(hi)}")
    return out


def match_gt_flags(det_boxes, det_scores, gt_boxes, score_thr: float = 0.5, iou_thr: float = 0.5) -> np.ndarray:
    """Para cada caja real, si fue detectada (emparejamiento voraz por puntuación, solo detecciones >= score_thr)."""
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    matched = np.zeros(len(gt), dtype=bool)
    boxes = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(det_scores, dtype=np.float64).reshape(-1)
    keep = scores >= score_thr
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0 or len(gt) == 0:
        return matched
    order = np.argsort(-scores, kind="stable")
    iou = box_iou_np(boxes[order], gt)
    for i in range(len(order)):
        cand = np.where(matched, -1.0, iou[i])
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            matched[j] = True
    return matched


def per_image_table(preds: Mapping, gts: Mapping, score_thr: float, iou_thr: float = 0.5) -> pd.DataFrame:
    """TP/FP/FN, precision, recall y F1 de cada imagen al umbral dado."""
    rows = []
    for n in sorted(gts):
        b, s = preds[n]
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        keep = s >= score_thr
        _, tp = match_image(np.asarray(b).reshape(-1, 4)[keep], s[keep], gts[n], iou_thr)
        n_gt, n_det, t = len(gts[n]), int(keep.sum()), int(tp.sum())
        p = t / n_det if n_det else (1.0 if n_gt == 0 else 0.0)
        r = t / n_gt if n_gt else 1.0
        rows.append({"image_name": n, "n_gt": n_gt, "n_det": n_det, "tp": t, "fp": n_det - t, "fn": n_gt - t,
                     "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if (p + r) else 0.0})
    return pd.DataFrame(rows)


def recall_by_gt_size(preds: Mapping, gts: Mapping, score_thr: float, bins=SIZE_BINS) -> pd.DataFrame:
    """Recall por tamaño de la caja real (lado equivalente sqrt(w*h) en px del marco original)."""
    sizes, hit = [], []
    for n in sorted(gts):
        g = np.asarray(gts[n], dtype=np.float64).reshape(-1, 4)
        if len(g) == 0:
            continue
        b, s = preds[n]
        sizes.append(np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])))
        hit.append(match_gt_flags(b, s, g, score_thr))
    sizes = np.concatenate(sizes) if sizes else np.zeros(0)
    hit = np.concatenate(hit) if hit else np.zeros(0, dtype=bool)
    rows = []
    for lab, lo, hi in zip(_bin_labels(bins), bins[:-1], bins[1:]):
        m = (sizes >= lo) & (sizes < hi)
        rows.append({"size_px": lab, "n_gt": int(m.sum()), "recall": float(hit[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def recall_by_size_and_group(preds: Mapping, gts: Mapping, groups: Mapping[str, str], score_thr: float,
                             bins=SIZE_BINS) -> pd.DataFrame:
    """Recall por tamaño de caja real dentro de cada grupo (p. ej. dominio): separa el efecto del tamaño del del dominio."""
    parts = []
    for g in sorted(set(groups[n] for n in gts)):
        names = [n for n in gts if groups[n] == g]
        df = recall_by_gt_size({n: preds[n] for n in names}, {n: gts[n] for n in names}, score_thr, bins)
        df.insert(0, "group", g)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def metrics_by_density(preds: Mapping, gts: Mapping, score_thr: float, bins=DENSITY_BINS) -> pd.DataFrame:
    """Métricas por densidad de espigas reales por imagen. Las imágenes sin espigas se reportan aparte
    (solo tiene sentido contar sus falsos positivos)."""
    pit = per_image_table(preds, gts, score_thr).set_index("image_name")
    cols = ["gt_per_image", "n_images", "n_gt", "map50", "map50_95", "precision", "recall", "f1", "fp_per_image"]
    rows = []
    empty = [n for n in gts if len(gts[n]) == 0]
    if empty:
        rows.append(dict(zip(cols, ["0 (sin espigas)", len(empty), 0, np.nan, np.nan, np.nan, np.nan, np.nan,
                                    float(pit.loc[empty, "fp"].mean())])))
    counts = {n: len(gts[n]) for n in gts}
    for lab, lo, hi in zip(_bin_labels(bins, integer=True), bins[:-1], bins[1:]):
        names = [n for n, c in counts.items() if lo <= c < hi]
        if not names:
            rows.append(dict(zip(cols, [lab, 0, 0] + [np.nan] * 6)))
            continue
        r = evaluate({n: preds[n] for n in names}, {n: gts[n] for n in names}, score_threshold=score_thr)
        rows.append(dict(zip(cols, [lab, len(names), r["n_gt"], r["map50"], r["map50_95"], r["precision"],
                                    r["recall"], r["f1"], float(pit.loc[names, "fp"].mean())])))
    return pd.DataFrame(rows, columns=cols)


# ---------- figuras ----------
def curves_figure(metrics: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    ep = metrics["epoch"] + 1
    for col, lab in [("train_loss", "total"), ("train_loss_classifier", "clasificador"),
                     ("train_loss_box_reg", "regresión de cajas"), ("train_loss_objectness", "objectness RPN"),
                     ("train_loss_rpn_box_reg", "regresión RPN"),
                     ("train_classification", "clasificación (focal)"),
                     ("train_bbox_regression", "regresión de cajas (RetinaNet)")]:
        if col in metrics:
            ax[0, 0].plot(ep, metrics[col], marker="o", ms=3, label=lab, lw=2.2 if col == "train_loss" else 1.2)
    ax[0, 0].set_title("Pérdida de entrenamiento (media por época)"); ax[0, 0].set_xlabel("época"); ax[0, 0].legend(fontsize=7)
    ev = metrics.dropna(subset=["val_map50"]) if "val_map50" in metrics else metrics.iloc[0:0]
    for col, lab in [("val_map50", "mAP@50"), ("val_map50_95", "mAP@50-95")]:
        if col in ev:
            ax[0, 1].plot(ev["epoch"] + 1, ev[col], marker="o", label=lab)
    ax[0, 1].set_title("Validación: mAP"); ax[0, 1].set_xlabel("época"); ax[0, 1].legend()
    for col, lab in [("val_precision", "precision (umbral 0,5)"), ("val_recall", "recall (umbral 0,5)"),
                     ("val_f1", "F1 (umbral 0,5)"), ("val_best_f1", "mejor F1")]:
        if col in ev:
            ax[1, 0].plot(ev["epoch"] + 1, ev[col], marker="o", label=lab, ls="--" if col == "val_best_f1" else "-")
    ax[1, 0].set_title("Validación: precision, recall y F1"); ax[1, 0].set_xlabel("época")
    if ax[1, 0].get_legend_handles_labels()[0]:
        ax[1, 0].legend(fontsize=7)
    if "lr_end" in metrics:
        ax[1, 1].plot(ep, metrics["lr_end"], color="tab:green", marker="o", ms=3)
        ax[1, 1].set_ylabel("learning rate al final de la época")
    if "train_time_s" in metrics:
        a2 = ax[1, 1].twinx()
        a2.plot(ep, metrics["train_time_s"] / 60, color="tab:red", marker="s", ms=3)
        a2.set_ylabel("minutos de entrenamiento por época", color="tab:red")
    ax[1, 1].set_title("Learning rate y tiempo"); ax[1, 1].set_xlabel("época")
    for a in ax.ravel():
        a.xaxis.set_major_locator(MaxNLocator(integer=True))             # épocas enteras
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def groups_figure(by_domain: pd.DataFrame, by_country: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, max(3.5, 0.45 * max(len(by_domain), len(by_country)) + 1.5)))
    for a, df, title in ((ax[0], by_domain.sort_values("map50"), "mAP@50 por dominio"),
                         (ax[1], by_country.sort_values("map50"), "mAP@50 por país")):
        a.barh([f"{g} (n={n})" for g, n in zip(df["group"], df["n_images"])], df["map50"])
        a.set_xlim(0, 1); a.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def diagnostics_figure(size_df: pd.DataFrame, density_df: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(size_df["size_px"], size_df["recall"])
    for i, (r, n) in enumerate(zip(size_df["recall"], size_df["n_gt"])):
        ax[0].text(i, (0 if np.isnan(r) else r) + 0.01, f"n={n}", ha="center", fontsize=7)
    ax[0].set_ylim(0, 1.1); ax[0].set_title("Recall por tamaño de espiga (lado equivalente, px)")
    d = density_df[density_df["n_images"] > 0].dropna(subset=["f1"])
    ax[1].bar(d["gt_per_image"], d["f1"], label="F1")
    ax[1].plot(d["gt_per_image"], d["recall"], "o-", color="tab:red", label="recall")
    ax[1].set_ylim(0, 1.1); ax[1].set_title("F1 y recall por densidad (espigas por imagen)"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def examples_figure(names: list[str], preds: Mapping, gts: Mapping, per_image: pd.DataFrame,
                    image_path: Callable[[str], str], score_thr: float, path, title: str, group_of=None) -> None:
    """Imagen con cajas reales (verde) y detecciones >= umbral (rojo). Una columna por imagen."""
    n = len(names)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 8.6), squeeze=False)
    flat = axes.ravel()
    for a in flat:
        a.axis("off")
    idx = per_image.set_index("image_name")
    for a, name in zip(flat, names):
        a.imshow(Image.open(image_path(name)).convert("RGB"))
        for x1, y1, x2, y2 in np.asarray(gts[name]).reshape(-1, 4):
            a.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=0.9))
        b, s = preds[name]
        for (x1, y1, x2, y2), sc in zip(np.asarray(b).reshape(-1, 4), np.asarray(s)):
            if sc >= score_thr:
                a.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red",
                                              linewidth=0.9, linestyle="--"))
        r = idx.loc[name]
        g = f" | {group_of[name]}" if group_of else ""
        a.set_title(f"{name[:8]}{g}\nF1 {r['f1']:.2f} | GT {int(r['n_gt'])} det {int(r['n_det'])} "
                    f"(TP {int(r['tp'])} FP {int(r['fp'])} FN {int(r['fn'])})", fontsize=7)
    fig.suptitle(title + " — verde: real, rojo discontinuo: detección", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()