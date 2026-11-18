"""Utilidades de lectura compartidas por la Fase 2."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

OFFICIAL_FILES = {"train": "competition_train.csv", "val": "competition_val.csv", "test": "competition_test.csv"}


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Detecta ',' o ';' mirando solo la cabecera (BoxesString contiene ';' en las filas)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def latest_complete_audit(audit_root: Path) -> Path:
    runs = sorted(d for d in audit_root.iterdir()
                  if d.is_dir() and not d.name.endswith("_partial") and not d.name.startswith("followup_"))
    if not runs:
        raise SystemExit(f"No hay auditorías completas en {audit_root}")
    return runs[-1]