"""Aumentos y remuestreo de imágenes con cajas. NumPy puro (sin PyTorch), deterministas dado un Generator.

Convención: imagen uint8 (alto, ancho, 3); cajas float32 (N, 4) en píxeles [x_min, y_min, x_max, y_max].
Los aumentos solo se usan en entrenamiento; validación y prueba solo se redimensionan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AugConfig:
    hflip: float = 0.5            # probabilidad de volteo horizontal
    vflip: float = 0.5            # probabilidad de volteo vertical (imágenes cenitales: válido)
    rot90: float = 0.5            # probabilidad de rotar 90/180/270 grados (k uniforme en {1, 2, 3})
    brightness: float = 0.2       # factor uniforme en [1-b, 1+b]
    contrast: float = 0.2         # factor uniforme en [1-c, 1+c]
    scale_crop_p: float = 0.5     # probabilidad de recorte cuadrado que luego se reescala (efecto de zoom)
    crop_min_frac: float = 0.6    # lado del recorte: fracción uniforme en [crop_min_frac, 1] del lado menor
    min_visibility: float = 0.5   # fracción mínima del área de una caja que debe seguir visible tras recortar
    min_box_side: float = 2.0     # lado mínimo (px) de una caja tras recortar


def clip_filter(boxes: np.ndarray, width: float, height: float, min_side: float,
                orig_area: np.ndarray | None = None, min_visibility: float = 0.0):
    """Recorta cajas a la imagen y descarta las que quedan por debajo de min_side o de min_visibility.

    Devuelve (cajas, máscara booleana de las conservadas respecto al array de entrada).
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32), np.zeros(0, dtype=bool)
    b = boxes.astype(np.float32).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    keep = (w >= min_side) & (h >= min_side)
    if orig_area is not None and min_visibility > 0:
        keep &= (w * h) >= min_visibility * orig_area
    return b[keep], keep


def hflip(img: np.ndarray, boxes: np.ndarray):
    w = img.shape[1]
    out = boxes.copy()
    out[:, [0, 2]] = w - boxes[:, [2, 0]]
    return img[:, ::-1], out


def vflip(img: np.ndarray, boxes: np.ndarray):
    h = img.shape[0]
    out = boxes.copy()
    out[:, [1, 3]] = h - boxes[:, [3, 1]]
    return img[::-1], out


def rot90(img: np.ndarray, boxes: np.ndarray, k: int):
    """Rota k*90 grados en sentido antihorario (igual que np.rot90) transformando las cajas."""
    k %= 4
    out = boxes.astype(np.float32).copy()
    h, w = img.shape[:2]
    for _ in range(k):
        # un giro de 90° antihorario lleva (x, y) -> (y, W - x), con W el ancho antes del giro;
        # después del giro ancho y alto se intercambian
        x1, y1, x2, y2 = out[:, 0].copy(), out[:, 1].copy(), out[:, 2].copy(), out[:, 3].copy()
        out[:, 0], out[:, 2] = y1, y2
        out[:, 1], out[:, 3] = w - x2, w - x1
        h, w = w, h
    return np.rot90(img, k), out


def adjust_brightness_contrast(img: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    f = img.astype(np.float32) * brightness
    mean = f.mean()
    f = (f - mean) * contrast + mean
    return np.clip(f, 0, 255).astype(np.uint8)


def resize_image_and_boxes(img: np.ndarray, boxes: np.ndarray, size: int):
    """Redimensiona a size x size escalando las cajas con los mismos factores por eje."""
    h, w = img.shape[:2]
    if (h, w) == (size, size):
        return img, boxes.astype(np.float32)
    out = np.array(Image.fromarray(np.ascontiguousarray(img)).resize((size, size), Image.BILINEAR))  # copia escribible
    b = boxes.astype(np.float32).copy()
    b[:, [0, 2]] *= size / w
    b[:, [1, 3]] *= size / h
    return out, b


def random_scale_crop(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator, cfg: AugConfig,
                      attempts: int = 5):
    """Recorte cuadrado aleatorio. Si la imagen tenía cajas y el recorte las elimina todas, se reintenta;
    tras `attempts` fallos se devuelve la imagen sin recortar (así el aumento no vacía imágenes sistemáticamente)."""
    h, w = img.shape[:2]
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.zeros(0)
    for _ in range(attempts):
        side = int(round(rng.uniform(cfg.crop_min_frac, 1.0) * min(h, w)))
        x0 = int(rng.integers(0, w - side + 1))
        y0 = int(rng.integers(0, h - side + 1))
        shifted = boxes - np.array([x0, y0, x0, y0], dtype=np.float32) if len(boxes) else boxes
        # área visible respecto a la original: se mide con la caja recortada a la ventana
        clipped, keep = clip_filter(shifted, side, side, cfg.min_box_side, area, cfg.min_visibility)
        if len(boxes) == 0 or len(clipped) > 0:
            return img[y0:y0 + side, x0:x0 + side], clipped
    return img, boxes.astype(np.float32)


def apply_train_transforms(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator,
                           cfg: AugConfig, out_size: int):
    boxes = boxes.astype(np.float32)
    if rng.random() < cfg.scale_crop_p:
        img, boxes = random_scale_crop(img, boxes, rng, cfg)
    if rng.random() < cfg.hflip:
        img, boxes = hflip(img, boxes)
    if rng.random() < cfg.vflip:
        img, boxes = vflip(img, boxes)
    if rng.random() < cfg.rot90:
        img, boxes = rot90(img, boxes, int(rng.integers(1, 4)))
    b = 1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
    c = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
    img = adjust_brightness_contrast(np.ascontiguousarray(img), b, c)
    img, boxes = resize_image_and_boxes(img, boxes, out_size)
    boxes, _ = clip_filter(boxes, out_size, out_size, cfg.min_box_side)
    return img, boxes


def apply_eval_transforms(img: np.ndarray, boxes: np.ndarray, out_size: int):
    img, boxes = resize_image_and_boxes(img, boxes, out_size)
    return img, boxes