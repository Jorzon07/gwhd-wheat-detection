"""Analiza una ejecución de entrenamiento con las predicciones de VALIDACIÓN de la mejor época. Sin PyTorch.

Uso: python -m src.evaluation.analyze_run --run outputs/runs/<carpeta> [--threshold 0.5]
Por defecto el umbral de confianza es el de mejor F1 en validación. Escribe en <carpeta>/analysis/.
NO usa la partición de prueba.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve
from src.data.annotations import parse_boxes_string
from src.evaluation.analysis import (SIZE_BINS, _bin_labels, curves_figure, diagnostics_figure, examples_figure,
                                     groups_figure, metrics_by_density, per_image_table, recall_by_gt_size,
                                     recall_by_size_and_group)
from src.evaluation.metrics import evaluate, evaluate_by_group
from src.training.utils import load_predictions


def df_to_md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def analyze(run_dir, threshold: float | None = None, n_examples: int = 6, min_gt_for_worst: int = 5) -> dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    split_dir = Path(cfg["split_dir"])
    images_dir = resolve(cfg["phase3"]["images_dir"])
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    meta = pd.read_csv(split_dir / "validation.csv")
    ann = pd.read_csv(split_dir / "annotations_clean.csv", dtype={"image_name": str}).set_index("image_name")
    preds = load_predictions(run_dir / "val_predictions_best.npz")
    gts = {n: parse_boxes_string(ann.at[n, "BoxesString"])[0].astype(np.float64) for n in preds}
    domain, country = dict(zip(meta["image_name"], meta["domain"])), dict(zip(meta["image_name"], meta["country"]))

    base = evaluate(preds, gts)
    thr = float(base["best_f1_threshold"]) if threshold is None else float(threshold)
    res = evaluate(preds, gts, score_threshold=thr)

    def by(group_map, name):
        df = evaluate_by_group(preds, gts, group_map, score_threshold=thr)
        df = df[["group", "n_images", "n_gt", "map50", "map50_95", "precision", "recall", "f1"]]
        df.to_csv(out / f"val_by_{name}.csv", index=False)
        return df
    by_domain, by_country = by(domain, "domain"), by(country, "country")
    macro = {"macro_map50_domains": float(by_domain["map50"].mean()),
             "macro_map50_95_domains": float(by_domain["map50_95"].mean()),
             "macro_f1_domains": float(by_domain["f1"].mean())}
    (out / "val_metrics.json").write_text(json.dumps({**res, **macro, "threshold_used": thr,
                                                      "threshold_source": "mejor F1 en validación" if threshold is None
                                                      else "indicado por el usuario"}, indent=2), encoding="utf-8")
    size_dom = recall_by_size_and_group(preds, gts, domain, thr)
    size_dom.to_csv(out / "val_recall_by_size_and_domain.csv", index=False)
    cell = size_dom.apply(lambda r: "-" if r["n_gt"] == 0 else f"{r['recall']:.2f} (n={int(r['n_gt'])})", axis=1)
    size_dom_md = (size_dom.assign(cell=cell).pivot(index="group", columns="size_px", values="cell")
                   [_bin_labels(SIZE_BINS)].reset_index())
    size_df, dens_df = recall_by_gt_size(preds, gts, thr), metrics_by_density(preds, gts, thr)
    size_df.to_csv(out / "val_recall_by_size.csv", index=False)
    dens_df.to_csv(out / "val_by_density.csv", index=False)
    per_img = per_image_table(preds, gts, thr)
    per_img.assign(domain=per_img["image_name"].map(domain), country=per_img["image_name"].map(country)) \
        .to_csv(out / "val_per_image.csv", index=False)

    if (run_dir / "metrics.csv").exists():
        curves_figure(pd.read_csv(run_dir / "metrics.csv"), out / "curves.png")
    groups_figure(by_domain, by_country, out / "by_group.png")
    diagnostics_figure(size_df, dens_df, out / "diagnostics.png")
    path_of = lambda n: images_dir / (n if n.lower().endswith(".png") else n + ".png")        # noqa: E731
    ranked = per_img[per_img["n_gt"] >= min_gt_for_worst].sort_values(["f1", "image_name"])
    worst = ranked["image_name"].head(n_examples).tolist()
    best = ranked.sort_values(["f1", "n_gt"], ascending=[False, False])["image_name"].head(n_examples).tolist()
    if best:
        examples_figure(best, preds, gts, per_img, path_of, thr, out / "examples_best.png",
                        f"Mejores detecciones (F1, umbral {thr:.2f})", domain)
    if worst:
        examples_figure(worst, preds, gts, per_img, path_of, thr, out / "examples_worst.png",
                        f"Peores detecciones (F1, umbral {thr:.2f})", domain)

    md = f"""# Análisis de {run_dir.name} (validación, mejor época)

Umbral de confianza: **{thr:.3f}** ({'mejor F1 en validación' if threshold is None else 'indicado por el usuario'}).

## Global
mAP@50 {res['map50']:.3f} | mAP@50-95 {res['map50_95']:.3f} | precision {res['precision']:.3f} | recall {res['recall']:.3f} | F1 {res['f1']:.3f}
({res['n_images']} imágenes, {res['n_gt']} espigas reales)
Media simple de los dominios (cada dominio pesa igual): mAP@50 {macro['macro_map50_domains']:.3f} | mAP@50-95 {macro['macro_map50_95_domains']:.3f} | F1 {macro['macro_f1_domains']:.3f}

## Por dominio
{df_to_md(by_domain.sort_values('map50'))}

## Por país
{df_to_md(by_country.sort_values('map50'))}

## Recall por tamaño de espiga (lado equivalente en px, marco de 1024)
{df_to_md(size_df)}

## Recall por tamaño de espiga y dominio (recall (n espigas reales))
{df_to_md(size_dom_md)}

## Por densidad (espigas reales por imagen)
{df_to_md(dens_df)}

Notas: las métricas por grupo se calculan con las mismas predicciones; los grupos pequeños dan cifras ruidosas.
Las figuras (curves.png, by_group.png, diagnostics.png, examples_*.png) están en esta carpeta.
"""
    (out / "analysis.md").write_text(md, encoding="utf-8")
    return {"threshold": thr, "metrics": res, "out_dir": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    r = analyze(resolve(args.run), args.threshold, args.examples)
    m = r["metrics"]
    print(f"Umbral {r['threshold']:.3f} | mAP50 {m['map50']:.3f} | mAP50-95 {m['map50_95']:.3f} | "
          f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}\nAnálisis en: {r['out_dir']}")


if __name__ == "__main__":
    main()