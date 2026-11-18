"""Compara ejecuciones con intervalos de confianza bootstrap (VALIDACIÓN; nunca usa la partición de prueba).

Uso:
  python -m src.evaluation.compare_runs --runs frcnn=outputs/runs/baseline_... retinanet=outputs/runs/retinanet_... \
      [--n-boot 1000] [--seed 0] [--out outputs/compare/frcnn_vs_retinanet]
El primer modelo es la referencia de las diferencias pareadas (modelo_i - referencia).
Umbral de confianza de cada modelo: su mejor F1 en validación (summary.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.bootstrap import (bootstrap, match_all, metrics_from_matches, paired_differences,
                                      summarize)

METRICS = ("map50", "map50_95", "map75", "precision", "recall", "f1")


def load_run(run_dir: Path):
    z = np.load(run_dir / "val_predictions_best.npz", allow_pickle=False)
    names, off = [str(n) for n in z["names"]], z["offsets"]
    preds = {n: (z["boxes"][off[i]:off[i + 1]].astype(np.float64), z["scores"][off[i]:off[i + 1]].astype(np.float64))
             for i, n in enumerate(names)}
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return preds, float(summ["best_val_metrics"]["best_f1_threshold"]), json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8"))


def df_to_md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    from src.config import resolve
    from src.data.annotations import parse_boxes_string

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="nombre=carpeta_del_run (el primero es la referencia)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"Formato incorrecto '{item}': use nombre=carpeta")
        runs[name] = resolve(path)
    loaded = {n: load_run(p) for n, p in runs.items()}
    first = next(iter(loaded.values()))
    names = sorted(first[0])
    for n, (pr, _, _) in loaded.items():
        if set(pr) != set(names):
            raise SystemExit(f"El modelo {n} no tiene las mismas imágenes de validación que el primero")

    split_dir = Path(first[2]["split_dir"])
    meta = pd.read_csv(split_dir / "validation.csv").set_index("image_name")
    ann = pd.read_csv(split_dir / "annotations_clean.csv", dtype={"image_name": str}).set_index("image_name")
    gts = [parse_boxes_string(ann.at[n, "BoxesString"])[0].astype(np.float64) for n in names]
    domains = np.array([meta.at[n, "domain"] for n in names])

    matched, thresholds, points = {}, {}, {}
    for n, (pr, thr, _) in loaded.items():
        matched[n] = match_all(gts, [pr[i] for i in names])
        thresholds[n] = thr
        points[n] = metrics_from_matches(matched[n], None, thr)
        print(f"{n}: umbral {thr:.3f} | mAP50 {points[n]['map50']:.4f} | mAP50-95 {points[n]['map50_95']:.4f}")
        try:   # comprobación cruzada con el evaluador del proyecto
            from src.evaluation.metrics import evaluate
            from src.training.utils import load_predictions
            ref = evaluate(load_predictions(runs[n] / "val_predictions_best.npz"), dict(zip(names, gts)))
            if abs(ref["map50"] - points[n]["map50"]) > 1e-4 or abs(ref["map50_95"] - points[n]["map50_95"]) > 1e-4:
                print(f"   AVISO: difiere del evaluador del proyecto (map50 {ref['map50']:.6f}, "
                      f"map50_95 {ref['map50_95']:.6f}). No uses estos intervalos sin revisarlo.")
            else:
                print("   coincide con el evaluador del proyecto (±1e-6)")
        except Exception as e:  # noqa: BLE001
            print(f"   (comprobación cruzada omitida: {e})")

    out = resolve(args.out) if args.out else Path("outputs/compare") / "_vs_".join(runs)
    out.mkdir(parents=True, exist_ok=True)
    ref_name, md = next(iter(runs)), []
    for level in ("image", "domain"):
        draws = bootstrap(matched, thresholds, args.n_boot, args.seed, level, domains if level == "domain" else None)
        s = summarize(draws, points)
        s.to_csv(out / f"ci_{level}.csv", index=False)
        diffs = pd.concat([paired_differences(draws, n, ref_name) for n in runs if n != ref_name]) \
            if len(runs) > 1 else pd.DataFrame()
        diffs.to_csv(out / f"paired_{level}.csv", index=False)
        label = "imágenes" if level == "image" else f"dominios completos ({len(np.unique(domains))})"
        md += [f"## Remuestreo por {label} ({args.n_boot} réplicas, IC 95 % percentil)", "",
               df_to_md(s[s.metric.isin(("map50", "map50_95", "f1"))]), ""]
        if len(diffs):
            md += [f"Diferencias pareadas (modelo - {ref_name}):", "", df_to_md(diffs), ""]
    md += ["Notas: solo validación. El remuestreo por imagen subestima la incertidumbre porque las imágenes de un "
           "dominio están correlacionadas; el de dominios es más conservador pero con pocos dominios es muy ancho. "
           "Un solo entrenamiento por modelo: no incluye la variabilidad entre semillas (~0.01 en mAP50)."]
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)); print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
