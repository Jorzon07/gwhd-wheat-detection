"""Utilidades de entrenamiento SIN PyTorch (probables de forma aislada): ejecuciones, registros CSV, calendario de
learning rate, seguimiento del mejor modelo, checkpoints atómicos, predicciones y validación de reanudación."""
from __future__ import annotations

import csv
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ---------- ejecuciones ----------
def make_run_dir(base, name: str) -> Path:
    """Crea una carpeta nueva <base>/<name>_<fecha_hora>; nunca reutiliza ni sobrescribe una existente."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for k in range(1000):
        d = base / (f"{name}_{stamp}" + (f"_{k}" if k else ""))
        try:
            d.mkdir()                                   # sin exist_ok: falla si ya existe
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"No se pudo crear una carpeta de ejecución nueva en {base}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def environment_info() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        import torchvision
        info.update({"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda})
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


# ---------- registros ----------
class CsvLogger:
    """Registro CSV en memoria que reescribe el archivo de forma atómica; admite columnas nuevas y reanudación."""

    def __init__(self, path):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]

    def log(self, row: dict) -> None:
        self.rows.append(dict(row))
        cols: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)


# ---------- calendario de learning rate ----------
def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int,
          min_lr_ratio: float = 0.01, warmup_factor: float = 0.001) -> float:
    """Calentamiento lineal desde base_lr*warmup_factor y descenso coseno hasta base_lr*min_lr_ratio."""
    if step < warmup_steps:
        alpha = step / max(1, warmup_steps)
        return base_lr * (warmup_factor * (1 - alpha) + alpha)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------- seguimiento del mejor modelo ----------
class BestTracker:
    """Registra la mejor época según una métrica a maximizar y decide la parada anticipada (patience=0: nunca)."""

    def __init__(self, patience: int = 0):
        self.patience = int(patience)
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.bad_epochs = 0

    def update(self, epoch: int, value: float) -> bool:
        if value is None or not math.isfinite(value):           # NaN / inf nunca cuentan como mejora
            self.bad_epochs += 1
            return False
        if self.best is None or value > self.best + 1e-12:
            self.best, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs,
                "patience": self.patience}

    def load_state_dict(self, d: dict) -> None:
        self.best, self.best_epoch, self.bad_epochs = d["best"], d["best_epoch"], d["bad_epochs"]


# ---------- checkpoints ----------
def to_builtin(obj):
    """Convierte escalares/arrays de NumPy a tipos de Python (torch.load(weights_only=True) no admite NumPy)."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_builtin(v) for v in obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class CheckpointManager:
    """last.pt (para reanudar) y best.pt (mejor época). Escritura atómica: archivo temporal + os.replace.

    `save_fn(obj, path)` y `load_fn(path)` se inyectan (torch.save / torch.load en producción; pickle en las pruebas).
    """

    def __init__(self, run_dir, save_fn: Callable, load_fn: Callable):
        self.dir = Path(run_dir)
        self.save_fn, self.load_fn = save_fn, load_fn

    def _save(self, state: dict, name: str) -> Path:
        tmp = self.dir / (name + ".tmp")
        self.save_fn(state, tmp)
        os.replace(tmp, self.dir / name)
        return self.dir / name

    def save_last(self, state: dict) -> Path:
        return self._save(state, "last.pt")

    def save_best(self, state: dict) -> Path:
        return self._save(state, "best.pt")

    def has_last(self) -> bool:
        return (self.dir / "last.pt").exists()

    def load(self, name: str = "last.pt"):
        p = self.dir / name
        return self.load_fn(p) if p.exists() else None


def validate_resume(saved: dict, current: dict, keys: Iterable[str]) -> list[str]:
    """Lista las claves de configuración que difieren entre la ejecución guardada y la actual."""
    return [f"{k}: guardado={saved.get(k)!r} actual={current.get(k)!r}" for k in keys if saved.get(k) != current.get(k)]


# ---------- predicciones ----------
def rescale_boxes(boxes, scale: float) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4) * float(scale)


def save_predictions(path, preds: dict) -> None:
    """Guarda {nombre: (cajas Nx4, puntuaciones N)} en un .npz comprimido (se reutiliza en umbrales y ensembles)."""
    names = sorted(preds)
    counts = [len(preds[n][1]) for n in names]
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    boxes = np.concatenate([np.asarray(preds[n][0], dtype=np.float32).reshape(-1, 4) for n in names]) \
        if names else np.zeros((0, 4), np.float32)
    scores = np.concatenate([np.asarray(preds[n][1], dtype=np.float32).reshape(-1) for n in names]) \
        if names else np.zeros(0, np.float32)
    np.savez_compressed(path, names=np.array(names), offsets=offsets, boxes=boxes, scores=scores)


def load_predictions(path) -> dict:
    z = np.load(path, allow_pickle=False)
    names, off = [str(n) for n in z["names"]], z["offsets"]
    return {n: (z["boxes"][off[i]:off[i + 1]], z["scores"][off[i]:off[i + 1]]) for i, n in enumerate(names)}