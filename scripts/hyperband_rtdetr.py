"""Hyperband (versión reducida) para RT-DETR. Lanza `python -m src.training.train` como subproceso en cada prueba.

Recurso = épocas de entrenamiento desde cero (el calendario coseno se ajusta a las épocas de cada prueba).
Objetivo = mAP@[.5:.95] de VALIDACIÓN de la mejor época (no toca el test). Hiperparámetros: lr, backbone-lr-mult, grad-clip.
Reanudable: el estado queda en <out>/state.json; si se interrumpe, volver a ejecutar el mismo comando salta lo ya hecho.

Uso:  python scripts/hyperband_rtdetr.py --dry-run      # solo muestra el plan y el coste estimado
      python scripts/hyperband_rtdetr.py                # ejecuta (por defecto R=9, eta=3, 78 épocas)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_CFG = {"lr": 1e-4, "backbone_lr_mult": 0.1, "grad_clip": 0.1}      # receta usada en el entrenamiento de 20 épocas
MULTS, CLIPS = [0.03, 0.1, 0.3, 1.0], [0.1, 1.0, 10.0]


def sample_config(rng) -> dict:
    return {"lr": float(np.exp(rng.uniform(np.log(3e-5), np.log(4e-4)))),
            "backbone_lr_mult": float(rng.choice(MULTS)), "grad_clip": float(rng.choice(CLIPS))}


def make_plan(R: int, eta: int, seed: int) -> list[dict]:
    s_max = int(math.floor(math.log(R, eta) + 1e-9))
    B, plan = (s_max + 1) * R, []
    for s in range(s_max, -1, -1):
        n = int(math.ceil(B / R * eta ** s / (s + 1)))
        r = R * eta ** (-s)
        rng = np.random.default_rng([seed, s])
        cfgs = [sample_config(rng) for _ in range(n)]
        if s == s_max:
            cfgs[0] = dict(DEFAULT_CFG)                      # la receta por defecto compite con el mismo presupuesto
        rungs = []
        for i in range(s + 1):
            rungs.append({"n": int(math.floor(n * eta ** (-i))), "epochs": max(1, int(round(r * eta ** i)))})
        plan.append({"s": s, "configs": cfgs, "rungs": rungs})
    return plan


def total_epochs(plan) -> int:
    return sum(r["n"] * r["epochs"] for b in plan for r in b["rungs"])


def run_trial(key: str, cfg: dict, epochs: int, out: Path, python: str, extra: list[str]) -> dict:
    name = f"hv_{key}"
    cmd = [python, "-m", "src.training.train", "--arch", "rtdetr", "--image-size", "800", "--batch-size", "4",
           "--optimizer", "adamw", "--lr", f"{cfg['lr']:.6g}", "--grad-clip", f"{cfg['grad_clip']:g}",
           "--backbone-lr-mult", f"{cfg['backbone_lr_mult']:g}", "--epochs", str(epochs), "--name", name, *extra]
    (out / "logs").mkdir(exist_ok=True)
    t0 = time.time()
    with open(out / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    text = (out / "logs" / f"{name}.log").read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"^Carpeta:\s*(.+)$", text, flags=re.M)
    value, run_dir = -1.0, m[-1].strip() if m else None
    if run_dir and (Path(run_dir) / "summary.json").exists():
        try:
            value = float(json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))["best_val_metrics"]["map50_95"])
            if not math.isfinite(value):
                value = -1.0
        except Exception:                                      # noqa: BLE001
            value = -1.0
    return {"key": key, "cfg": cfg, "epochs": epochs, "value": value, "run_dir": run_dir,
            "returncode": proc.returncode, "minutes": round((time.time() - t0) / 60, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=9, help="épocas máximas por prueba")
    ap.add_argument("--eta", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/hyperband/rtdetr")
    ap.add_argument("--min-per-epoch", type=float, default=6.4, help="minutos por época (RT-DETR 800 px, medido)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("extra", nargs="*", help="argumentos extra para train.py (tras --)")
    a = ap.parse_args()

    plan = make_plan(a.R, a.eta, a.seed)
    ep = total_epochs(plan)
    print(f"Plan Hyperband (R={a.R}, eta={a.eta}): {sum(r['n'] for b in plan for r in b['rungs'])} pruebas, {ep} épocas "
          f"≈ {ep * a.min_per_epoch / 60:.1f} h")
    for b in plan:
        print(f"  rama s={b['s']}: " + " -> ".join(f"{r['n']} pruebas x {r['epochs']} ép" for r in b["rungs"]))
    if a.dry_run:
        for b in plan[:1]:
            print("  ejemplo de configuraciones de la 1.ª rama:")
            for c in b["configs"][:4]:
                print("   ", {k: round(v, 6) for k, v in c.items()})
        return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sf = out / "state.json"
    state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {"trials": {}}
    if state.get("plan_id", (a.R, a.eta, a.seed)) != [a.R, a.eta, a.seed] and state["trials"]:
        raise SystemExit("El estado existente es de otro plan (R, eta, semilla). Usa otra carpeta --out.")
    state["plan_id"] = [a.R, a.eta, a.seed]

    def save():
        tmp = sf.with_suffix(".tmp"); tmp.write_text(json.dumps(state, indent=2), encoding="utf-8"); os.replace(tmp, sf)

    for b in plan:
        alive = list(range(len(b["configs"])))
        for i, rung in enumerate(b["rungs"]):
            alive = alive[: rung["n"]]
            for j in alive:
                key = f"s{b['s']}_r{i}_c{j}"
                if key in state["trials"]:
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] {key}: {rung['epochs']} épocas, {b['configs'][j]}", flush=True)
                res = run_trial(key, b["configs"][j], rung["epochs"], out, a.python, a.extra)
                state["trials"][key] = res; save()
                print(f"    mAP50-95 val = {res['value']:.4f} ({res['minutes']} min, código {res['returncode']})", flush=True)
            keep = max(1, int(math.floor(len(alive) / a.eta)))
            alive = sorted(alive, key=lambda j: -state["trials"][f"s{b['s']}_r{i}_c{j}"]["value"])[:keep]

    rows = sorted(state["trials"].values(), key=lambda t: (-t["epochs"], -t["value"]))
    full = [t for t in rows if t["epochs"] == a.R and t["value"] >= 0]
    best = max(full, key=lambda t: t["value"]) if full else None
    md = ["# Hyperband RT-DETR (validación, mAP@[.5:.95] de la mejor época)", "",
          "| prueba | épocas | lr | backbone-lr-mult | grad-clip | mAP50-95 | min |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for t in rows:
        c = t["cfg"]
        md.append(f"| {t['key']} | {t['epochs']} | {c['lr']:.2e} | {c['backbone_lr_mult']:g} | {c['grad_clip']:g} | {t['value']:.4f} | {t['minutes']} |")
    if best:
        md += ["", f"Mejor configuración con presupuesto completo ({a.R} épocas): **{best['key']}** -> {best['cfg']} "
                   f"(mAP50-95 {best['value']:.4f})"]
        d = state["trials"].get("s2_r0_c0") if a.R == 9 else None
        (out / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    md += ["", "Nota: mAP50-95 con un solo entrenamiento por prueba; las diferencias menores de ~0.01 no son interpretables."]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md)); print(f"\nInforme en {out / 'report.md'}")


if __name__ == "__main__":
    main()
