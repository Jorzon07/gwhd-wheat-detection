"""Lectura y validación del formato BoxesString de GWHD.

Formato declarado en Zenodo: cajas [x_min, y_min, x_max, y_max]; coordenadas separadas por
espacio y cajas por ';'. Sin cajas: "no_box". El parser también acepta comas por robustez,
y la auditoría confirma con datos reales si el orden x_min/y_min/x_max/y_max es el correcto.
"""
from __future__ import annotations

import math

import numpy as np

NO_BOX = "no_box"


def parse_boxes_string(value) -> tuple[np.ndarray, list[str]]:
    """Devuelve (array Nx4 float32, lista de errores). No lanza excepciones."""
    empty = np.zeros((0, 4), dtype=np.float32)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return empty, ["missing_value"]
    text = str(value).strip()
    if text == "" or text.lower() == NO_BOX:
        return empty, []

    boxes: list[list[float]] = []
    errors: list[str] = []
    for i, chunk in enumerate(text.split(";")):
        chunk = chunk.strip()
        if not chunk:          # p. ej. ';' final
            continue
        parts = chunk.replace(",", " ").split()
        if len(parts) != 4:
            errors.append(f"box_{i}:expected_4_values_got_{len(parts)}")
            continue
        try:
            boxes.append([float(p) for p in parts])
        except ValueError:
            errors.append(f"box_{i}:non_numeric")
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return arr, errors


def boxes_to_string(boxes: np.ndarray) -> str:
    if len(boxes) == 0:
        return NO_BOX
    return ";".join(" ".join(f"{v:g}" for v in b) for b in boxes.tolist())


def validate_boxes(boxes: np.ndarray, width: float, height: float,
                   min_side: float = 2.0, tol: float = 0.0) -> list[tuple[int, str, float]]:
    """Devuelve [(índice_caja, problema, magnitud)].

    Problemas: non_finite, non_positive_size, out_of_bounds, tiny_box.
    """
    issues: list[tuple[int, str, float]] = []
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            issues.append((i, "non_finite", 0.0))
            continue
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            issues.append((i, "non_positive_size", min(w, h)))
            continue
        overflow = max(-x1, -y1, x2 - width, y2 - height, 0.0)
        if overflow > tol:
            issues.append((i, "out_of_bounds", overflow))
        if min(w, h) < min_side:
            issues.append((i, "tiny_box", min(w, h)))
    return issues