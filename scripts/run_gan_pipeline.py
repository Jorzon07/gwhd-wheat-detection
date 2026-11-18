"""Cadena desatendida del aumento con GAN: (1) entrena el pix2pix, (2) genera las imágenes sintéticas y la partición
aumentada, (3) entrena RT-DETR con la receta de Hyperband sobre train real + sintéticas. Reanudable: cada etapa se
omite si ya terminó. Solo usa train y validación (nunca la partición de prueba).

Uso:  python scripts\\run_gan_pipeline.py [--minutes 75] [--n 1500] [--epochs 9] [--dry-run]
Registros en outputs/gan/logs/. Estado en outputs/gan/pipeline_state.json.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEST = {"lr": "0.000246604", "grad_clip": "10", "backbone_lr_mult": "0.3"}      # outputs/hyperband/rtdetr/best.json


def run(cmd: list, log: Path, dry: bool) -> str:
    print("   $ " + " ".join(cmd), flush=True)
    if dry:
        return ""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    txt = log.read_text(encoding="utf-8", errors="replace")
    if rc != 0:
        print("\n".join(txt.splitlines()[-25:]))
        raise SystemExit(f"Falló (código {rc}). Registro completo: {log}")
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=75.0, help="duración del entrenamiento del pix2pix")
    ap.add_argument("--n", type=int, default=1500, help="imágenes sintéticas")
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    py, out = sys.executable, ROOT / "outputs" / "gan"
    logs, sf = out / "logs", out / "pipeline_state.json"
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}

    print(f"[{time.strftime('%H:%M:%S')}] Etapa 1/3: pix2pix ({a.minutes:g} min)")
    cmd = [py, "-m", "src.gan.train_gan", "--minutes", str(a.minutes)]
    if (out / "pix2pix" / "last.pt").exists():
        cmd.append("--resume")                                   # si ya completó el tiempo, termina al instante
    run(cmd, logs / "1_pix2pix.log", a.dry_run)

    print(f"[{time.strftime('%H:%M:%S')}] Etapa 2/3: imágenes sintéticas ({a.n}) y partición gan_aug")
    if (ROOT / "data" / "splits" / "gan_aug" / "synthetic_info.json").exists():
        print("   ya hecha")
    else:
        run([py, "-m", "src.gan.make_synthetic", "--n", str(a.n)], logs / "2_synthetic.log", a.dry_run)

    print(f"[{time.strftime('%H:%M:%S')}] Etapa 3/3: RT-DETR (receta Hyperband) con train real + sintéticas")
    rd = state.get("detector_run")
    if rd and (Path(rd) / "summary.json").exists():
        print(f"   ya hecha: {rd}")
    else:
        txt = run([py, "-m", "src.training.train", "--config", "data/splits/gan_aug/config.yaml", "--arch", "rtdetr",
                   "--image-size", "800", "--batch-size", "4", "--optimizer", "adamw", "--lr", BEST["lr"],
                   "--grad-clip", BEST["grad_clip"], "--backbone-lr-mult", BEST["backbone_lr_mult"],
                   "--epochs", str(a.epochs), "--name", "gan_aug_rtdetr"], logs / "3_detector.log", a.dry_run)
        m = re.findall(r"^Carpeta:\s*(.+)$", txt, flags=re.M)
        if m:
            state["detector_run"] = m[-1].strip()
            sf.write_text(json.dumps(state, indent=2), encoding="utf-8")
            print("\n".join(l for l in txt.splitlines() if l.startswith(("Mejor época", "Resumen"))))
    if state.get("detector_run"):
        print("\nComparar con el control (mismo recetario, solo train real):\n  python -m src.evaluation.compare_runs --runs "
              f"tuned9=outputs\\runs\\hv_s0_r0_c2_20261117_022523 gan_aug={state['detector_run']}")


if __name__ == "__main__":
    main()
