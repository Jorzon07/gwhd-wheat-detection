from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path_like: str | Path | None) -> Path | None:
    """Convierte rutas relativas del config en rutas absolutas desde la raíz del proyecto."""
    if path_like is None:
        return None
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p