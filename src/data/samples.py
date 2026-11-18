"""Lectura bajo demanda de imágenes y cajas limpias para una partición. NumPy puro (sin PyTorch).

Las imágenes se decodifican en cada acceso (nunca se mantiene el dataset completo en memoria). Opcionalmente hay una
caché LRU acotada de imágenes decodificadas (`cache_items`, 0 = desactivada); con varios workers cada proceso tiene
la suya.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.annotations import parse_boxes_string
from src.data.transforms import AugConfig, apply_eval_transforms, apply_train_transforms

SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}


class WheatSamples:
    def __init__(self, split_csv, ann_csv, images_dir, image_size: int = 1024, train: bool = False,
                 aug: AugConfig | None = None, seed: int = 0, cache_items: int = 0, image_ext: str = ".png",
                 check_files: bool = True):
        self.meta = pd.read_csv(split_csv)
        if "image_name" not in self.meta.columns:
            raise ValueError(f"{split_csv}: falta la columna image_name")
        self.names: list[str] = self.meta["image_name"].astype(str).tolist()
        self.images_dir = Path(images_dir)
        self.image_ext = image_ext
        self.image_size = int(image_size)
        self.train = bool(train)
        self.aug = (aug or AugConfig()) if train else None
        self.seed = int(seed)
        self.epoch = 0
        self.cache_items = int(cache_items)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        ann = pd.read_csv(ann_csv, dtype={"image_name": str})
        ann = ann[ann["image_name"].isin(set(self.names))]
        self.boxes: dict[str, np.ndarray] = {}
        for name, bstr in zip(ann["image_name"], ann["BoxesString"]):
            b, errs = parse_boxes_string(bstr)
            if errs:
                raise ValueError(f"{name}: anotación inválida {errs}")
            self.boxes[name] = b
        missing = [n for n in self.names if n not in self.boxes]
        if missing:
            raise ValueError(f"{len(missing)} imágenes de {Path(split_csv).name} sin anotación en {Path(ann_csv).name} "
                             f"(p. ej. {missing[:3]})")
        if check_files:
            absent = [n for n in self.names if not self.path(n).is_file()]
            if absent:
                raise FileNotFoundError(f"{len(absent)} imágenes no existen en {self.images_dir} (p. ej. {absent[:3]})")

    def path(self, name: str) -> Path:
        return self.images_dir / (name if name.lower().endswith(self.image_ext) else name + self.image_ext)

    def __len__(self) -> int:
        return len(self.names)

    def set_epoch(self, epoch: int) -> None:
        """Cambia la semilla de los aumentos: (semilla, época, índice) determina cada muestra."""
        self.epoch = int(epoch)

    def column(self, col: str) -> dict[str, str]:
        """Mapa nombre de imagen -> valor de una columna de la partición (p. ej. 'domain' o 'country')."""
        return dict(zip(self.names, self.meta[col].astype(str)))

    def _load(self, idx: int) -> np.ndarray:
        if self.cache_items and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx].copy()                          # el llamante nunca comparte memoria con la caché
        with Image.open(self.path(self.names[idx])) as im:
            arr = np.array(im.convert("RGB"))                       # np.array copia: escribible
        if self.cache_items:
            self._cache[idx] = arr.copy()
            while len(self._cache) > self.cache_items:
                self._cache.popitem(last=False)
        return arr

    def get(self, idx: int):
        """Devuelve (imagen uint8 HxWx3 contigua, cajas float32 Nx4, nombre)."""
        name = self.names[idx]
        img, boxes = self._load(idx), self.boxes[name]
        if self.train:
            rng = np.random.default_rng([self.seed, self.epoch, idx])
            img, boxes = apply_train_transforms(img, boxes, rng, self.aug, self.image_size)
        else:
            img, boxes = apply_eval_transforms(img, boxes, self.image_size)
        return np.ascontiguousarray(img), np.ascontiguousarray(boxes, dtype=np.float32).reshape(-1, 4), name


def samples_from_config(cfg: dict, split: str, train: bool | None = None, image_size: int | None = None) -> WheatSamples:
    """Construye las muestras de 'train', 'validation' o 'test' con la configuración phase3. Por defecto solo
    'train' usa aumentos."""
    from src.config import resolve
    p3 = cfg["phase3"]
    split_dir = resolve(p3["split_dir"])
    return WheatSamples(
        split_dir / SPLIT_FILES[split], split_dir / "annotations_clean.csv", resolve(p3["images_dir"]),
        image_size=int(image_size or p3["image_size"]), train=(split == "train") if train is None else train,
        aug=AugConfig(**p3.get("aug", {})), seed=int(p3.get("seed", 0)),
        cache_items=int(p3.get("loader", {}).get("cache_items", 0)), image_ext=cfg["dataset"]["image_ext"])