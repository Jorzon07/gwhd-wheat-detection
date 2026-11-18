"""Envoltura de PyTorch sobre WheatSamples: Dataset en formato torchvision-detection y DataLoader reproducible.

Formato de cada muestra: imagen float32 (3, H, W) en [0, 1] y un dict con boxes (N, 4 xyxy float32), labels (N, int64,
siempre 1 = wheat_head; 0 queda reservado para el fondo en torchvision), image_id, area e iscrowd. Las imágenes sin
espigas devuelven tensores vacíos (0, 4) y (0,), que torchvision admite.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.samples import WheatSamples


class WheatDataset(Dataset):
    """`indices` restringe el dataset a un subconjunto; el `image_id` de cada muestra sigue siendo su índice original."""

    def __init__(self, samples: WheatSamples, indices=None):
        self.samples = samples
        self.indices = list(range(len(samples))) if indices is None else [int(i) for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.samples.set_epoch(epoch)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, boxes, _ = self.samples.get(idx)
        image = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous().float().div_(255.0)
        b = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4)
        target = {"boxes": b,
                  "labels": torch.ones((len(b),), dtype=torch.int64),
                  "image_id": torch.tensor([idx], dtype=torch.int64),
                  "area": (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]),
                  "iscrowd": torch.zeros((len(b),), dtype=torch.int64)}
        return image, target

def collate_fn(batch):
    """Los lotes de detección son tuplas de imágenes y de dicts: las cajas varían de tamaño por imagen."""
    return tuple(zip(*batch))


def make_loader(dataset: WheatDataset, batch_size: int, shuffle: bool, num_workers: int = 0,
                pin_memory: bool = False, seed: int = 0, drop_last: bool = False) -> DataLoader:
    """DataLoader con barajado reproducible. Los workers se recrean cada época (persistent_workers=False), de modo que
    el `epoch` de los aumentos fijado con `set_epoch` llega siempre a los workers. En Windows los scripts que lo usen
    con num_workers > 0 deben protegerse con `if __name__ == "__main__":`."""
    g = torch.Generator()
    g.manual_seed(seed)
    kw = {"prefetch_factor": 2, "persistent_workers": False} if num_workers > 0 else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=pin_memory, collate_fn=collate_fn, generator=g, drop_last=drop_last, **kw)


def set_epoch(loader: DataLoader, epoch: int, seed: int = 0) -> None:
    """Llamar al inicio de cada época de entrenamiento: cambia aumentos y orden de barajado, de forma reproducible."""
    loader.dataset.set_epoch(epoch)
    if loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)