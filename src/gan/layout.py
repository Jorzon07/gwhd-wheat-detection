"""Mapas de disposición (layout) de cajas para la GAN condicional. NumPy puro (sin PyTorch).

Un layout es un arreglo (2, H, W) float32 en {-1, +1}: canal 0 = interior de las cajas, canal 1 = contorno de 1 px
(separa espigas que se tocan). Las transformaciones geométricas existen en dos versiones, sobre arreglos y sobre cajas,
y las pruebas comprueban que coinciden.
"""
from __future__ import annotations

import random

import numpy as np


def render_layout(boxes, height: int, width: int | None = None) -> np.ndarray:
    width = int(width or height)
    m = np.full((2, int(height), width), -1.0, dtype=np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        a, b = int(np.floor(max(x1, 0))), int(np.ceil(min(x2, width)))
        c, d = int(np.floor(max(y1, 0))), int(np.ceil(min(y2, height)))
        if b <= a or d <= c:
            continue
        m[0, c:d, a:b] = 1.0
        m[1, c, a:b] = 1.0
        m[1, d - 1, a:b] = 1.0
        m[1, c:d, a] = 1.0
        m[1, c:d, b - 1] = 1.0
    return m


def crop_boxes(boxes, x0: int, y0: int, w: int, h: int, min_vis: float = 0.5, min_side: float = 2.0) -> np.ndarray:
    """Cajas (xyxy) dentro de la ventana [x0, x0+w) x [y0, y0+h), en coordenadas de la ventana. Se descartan las que
    conservan menos de `min_vis` de su área o quedan con un lado menor que `min_side`."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    shifted = b - np.array([x0, y0, x0, y0], dtype=np.float64)
    hi = np.array([w, h, w, h], dtype=np.float64)
    cl = np.minimum(np.maximum(shifted, 0.0), hi)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    carea = (cl[:, 2] - cl[:, 0]) * (cl[:, 3] - cl[:, 1])
    keep = (area > 0) & (carea >= min_vis * area) & ((cl[:, 2] - cl[:, 0]) >= min_side) & ((cl[:, 3] - cl[:, 1]) >= min_side)
    return cl[keep].astype(np.float32)


def apply_array(a: np.ndarray, flip: bool, k: int) -> np.ndarray:
    """Volteo horizontal opcional y k giros de 90° antihorarios sobre los dos últimos ejes."""
    if flip:
        a = a[..., ::-1]
    a = np.rot90(a, k % 4, axes=(-2, -1))
    return np.ascontiguousarray(a)


def transform_boxes(boxes, size: int, flip: bool, k: int) -> np.ndarray:
    """Mismo efecto que `apply_array` sobre cajas xyxy de un cuadrado de lado `size`."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if flip:
        b = np.stack([size - b[:, 2], b[:, 1], size - b[:, 0], b[:, 3]], axis=1)
    for _ in range(k % 4):
        b = np.stack([b[:, 1], size - b[:, 2], b[:, 3], size - b[:, 0]], axis=1)
    return b.astype(np.float32)


def compose_canvas(pool: list, rng: random.Random, canvas: int = 1024, quad: int = 512, source: int = 1024,
                   min_vis: float = 0.5) -> np.ndarray:
    """Cajas de un lienzo canvas x canvas formado por (canvas/quad)^2 recortes aleatorios de `quad` px tomados de
    distintas imágenes de `pool` (lista de arreglos de cajas en el marco `source`), con volteo y giro aleatorios."""
    n = canvas // quad
    out = []
    for qy in range(n):
        for qx in range(n):
            boxes = pool[rng.randrange(len(pool))]
            x0, y0 = rng.randint(0, source - quad), rng.randint(0, source - quad)
            b = crop_boxes(boxes, x0, y0, quad, quad, min_vis)
            b = transform_boxes(b, quad, rng.random() < 0.5, rng.randrange(4))
            if len(b):
                out.append(b + np.array([qx * quad, qy * quad, qx * quad, qy * quad], dtype=np.float32))
    return np.concatenate(out) if out else np.zeros((0, 4), np.float32)
