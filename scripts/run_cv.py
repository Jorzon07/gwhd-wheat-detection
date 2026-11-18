"""Lanza la validación cruzada de RT-DETR (configuración ganadora de Hyperband) y agrega los resultados.

Uso: python scripts/run_cv.py [--epochs 6] [--k 3]        (reanudable: estado en outputs/cv/state.json)
Antes: python -m src.data.make_cv_splits
Solo usa pliegues de train+validación; nunca el test.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BEST = {"lr": "0.000246604", "grad_clip": "10", "backbone_lr_mult": "0.3"}      # s0_r0_c2 de Hyperband


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--cv-root", default="data/splits/cv")
    ap.add_argument("--out", default="outputs/cv")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.cv_root), Path(a.out)
    info = json.loads((root / "cv_info.json").read_text(encoding="utf-8"))
    info = [r for r in info if f"_k{a.k}_" in r["dir"]]
    if a.dry_run:
        for r in info:
            print(f"{r['variant']}_f{r['fold']}: train {r['n_train']} / val {r['n_val']}")
        return
    out.mkdir(parents=True, exist_ok=True); (out / "logs").mkdir(exist_ok=True)
    sf = out / "state.json"
    state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
    for r in info:
        key = f"{r['variant']}_f{r['fold']}"
        if key in state and state[key].get("value") is not None:
            continue
        cfg_path = Path(r["dir"]) / "config.yaml"
        cmd = [a.python, "-m", "src.training.train", "--config", str(cfg_path), "--arch", "rtdetr", "--image-size", "800",
               "--batch-size", "4", "--optimizer", "adamw", "--lr", BEST["lr"], "--grad-clip", BEST["grad_clip"],
               "--backbone-lr-mult", BEST["backbone_lr_mult"], "--epochs", str(a.epochs), "--name", f"cv_{key}"]
        print(f"[{time.strftime('%H:%M:%S')}] {key}: train {r['n_train']} / val {r['n_val']}", flush=True)
        t0 = time.time()
        log = out / "logs" / f"{key}.log"
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        txt = log.read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"^Carpeta:\s*(.+)$", txt, flags=re.M)
        res = {"run_dir": m[-1].strip() if m else None, "value": None}
        if res["run_dir"] and (Path(res["run_dir"]) / "summary.json").exists():
            run = Path(res["run_dir"])
            conf = json.loads((run / "config.json").read_text(encoding="utf-8"))
            used = str(conf.get("split_dir", "")).replace("\\", "/")
            if Path(used).resolve() != Path(r["dir"]).resolve():
                raise SystemExit(f"{key}: el entrenamiento NO usó el pliegue esperado ({used} != {r['dir']}). "
                                 "Se detiene para no mezclar resultados.")
            res["metrics"] = json.loads((run / "summary.json").read_text(encoding="utf-8"))["best_val_metrics"]
            res["value"] = res["metrics"]["map50_95"]
        res["minutes"] = round((time.time() - t0) / 60, 1)
        state[key] = res
        sf.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"    mAP50-95 = {res['value']} ({res['minutes']} min)", flush=True)

    rows, summ = [], {}
    for r in info:
        s = state.get(f"{r['variant']}_f{r['fold']}", {})
        if s.get("value") is None:
            continue
        m = s["metrics"]
        rows.append({"variante": r["variant"], "pliegue": r["fold"], "n_train": r["n_train"], "n_val": r["n_val"],
                     "ubic. compartidas": len(r["locations_in_both"]), "mAP50": f"{m['map50']:.3f}",
                     "mAP50-95": f"{m['map50_95']:.3f}", "F1": f"{m['best_f1']:.3f}"})
        summ.setdefault(r["variant"], []).append((m["map50"], m["map50_95"], m["best_f1"]))
    md = ["# Validación cruzada RT-DETR (configuración Hyperband, validación de cada pliegue)", "",
          md_table(rows, ["variante", "pliegue", "n_train", "n_val", "ubic. compartidas", "mAP50", "mAP50-95", "F1"]), ""]
    means = {}
    for v, vals in summ.items():
        a_ = np.array(vals); means[v] = a_.mean(0)
        md.append(f"**{v}** (n={len(vals)} pliegues): mAP50 {a_[:,0].mean():.3f} ± {a_[:,0].std(ddof=1) if len(vals)>1 else float('nan'):.3f} | "
                  f"mAP50-95 {a_[:,1].mean():.3f} ± {a_[:,1].std(ddof=1) if len(vals)>1 else float('nan'):.3f} | F1 {a_[:,2].mean():.3f}")
    if {"random", "grouped"} <= set(means):
        d = means["random"] - means["grouped"]
        md += ["", f"Inflación por fuga (aleatoria - agrupada): mAP50 {d[0]:+.3f} | mAP50-95 {d[1]:+.3f} | F1 {d[2]:+.3f}"]
    md += ["", "Nota: ± es la desviación típica entre pliegues (pocos pliegues: orientativa). Cada pliegue evalúa sobre sus "
               "propias imágenes, con eval cada 2 épocas; la mejor época se elige con el propio pliegue (sesgo optimista, igual en ambas variantes)."]
    (out / "cv_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
