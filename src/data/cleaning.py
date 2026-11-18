"""Reglas de limpieza de cajas aprobadas en la Fase 1."""
from __future__ import annotations

import numpy as np


def clean_boxes(boxes: np.ndarray, width: float, height: float, min_side: float = 2.0):
    """Recorta a [0, width] x [0, height] y descarta las cajas cuyo lado recortado sea < min_side.

    Devuelve (cajas_limpias Nx4 float32, eventos). Cada evento es (índice_original, acción, detalle),
    con acción en {"clipped", "dropped_outside", "dropped_tiny", "dropped_non_finite"}.
    """
    kept, events = [], []
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=float).tolist()):
        if not all(np.isfinite([x1, y1, x2, y2])):
            events.append((i, "dropped_non_finite", ""))
            continue
        cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
        cx2, cy2 = min(x2, float(width)), min(y2, float(height))
        w, h = cx2 - cx1, cy2 - cy1
        if w < min_side or h < min_side:
            action = "dropped_outside" if (w <= 0 or h <= 0) else "dropped_tiny"
            events.append((i, action, f"w={w:.1f},h={h:.1f}"))
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            events.append((i, "clipped", f"{x1:g} {y1:g} {x2:g} {y2:g} -> {cx1:g} {cy1:g} {cx2:g} {cy2:g}"))
        kept.append([cx1, cy1, cx2, cy2])
    return np.asarray(kept, dtype=np.float32).reshape(-1, 4), events