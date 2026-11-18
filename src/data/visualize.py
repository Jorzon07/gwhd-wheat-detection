"""Figura de control visual: la misma imagen sin aumentos (arriba) y con aumentos de entrenamiento (abajo).

Uso: python -m src.data.visualize [--split train] [--n 6] [--epoch 0] [--size 1024]
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _draw(ax, img, boxes, title):
    ax.imshow(img)
    for x1, y1, x2, y2 in boxes.tolist():
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=0.8))
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def save_samples_figure(eval_samples, train_samples, indices, path, epoch: int = 0) -> None:
    train_samples.set_epoch(epoch)
    fig, axes = plt.subplots(2, len(indices), figsize=(3.3 * len(indices), 6.8), squeeze=False)
    for col, idx in enumerate(indices):
        img, boxes, name = eval_samples.get(idx)
        _draw(axes[0][col], img, boxes, f"{name[:10]} | sin aumentos | {len(boxes)} cajas")
        img, boxes, _ = train_samples.get(idx)
        _draw(axes[1][col], img, boxes, f"época {epoch} | aumentada | {len(boxes)} cajas")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main() -> None:
    from src.config import load_config, resolve
    from src.data.samples import samples_from_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--size", type=int, default=None, help="tamaño de salida (por defecto, phase3.image_size)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ev = samples_from_config(cfg, args.split, train=False, image_size=args.size)
    tr = samples_from_config(cfg, args.split, train=True, image_size=args.size)
    idx = np.random.default_rng(0).choice(len(ev), size=min(args.n, len(ev)), replace=False).tolist()
    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"phase3_samples_{args.split}_epoch{args.epoch}.png"
    save_samples_figure(ev, tr, idx, path, args.epoch)
    print("Figura guardada en:", path)


if __name__ == "__main__":
    main()