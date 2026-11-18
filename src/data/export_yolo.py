"""Exporta TRAIN y VALIDACIÓN al formato de Ultralytics (YOLO). NO lee ni enlaza la partición de prueba.

Uso: python -m src.data.export_yolo [--split-dir data/splits/location_seed42] [--images-dir data/raw/gwhd_2021/images]
Crea data/yolo/{images,labels}, train.txt, val.txt y gwhd.yaml. Las imágenes se enlazan (hard link en NTFS, sin ocupar
espacio); si no se puede, se copian. Las etiquetas salen de annotations_clean.csv (ya limpias, cajas recortadas).
Ultralytics deduce la etiqueta de cada imagen cambiando la última carpeta 'images' por 'labels'.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.annotations import parse_boxes_string


def yolo_lines(boxes, w: int, h: int) -> list[str]:
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(float(x1), w)), max(0.0, min(float(x2), w))
        y1, y2 = max(0.0, min(float(y1), h)), max(0.0, min(float(y2), h))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "existente"
    try:
        os.link(src, dst)
        return "enlace"
    except OSError:
        shutil.copy2(src, dst)
        return "copia"


def export(split_dir: Path, images_dir: Path, out: Path) -> dict:
    ann = pd.read_csv(split_dir / "annotations_clean.csv", dtype={"image_name": str}).set_index("image_name")
    sets = {}
    for key, fname in (("train", "train.csv"), ("val", "validation.csv")):
        f = split_dir / fname
        if not f.exists():
            raise SystemExit(f"No existe {f}. CSV disponibles: {[p.name for p in split_dir.glob('*.csv')]}")
        sets[key] = pd.read_csv(f, dtype={"image_name": str})["image_name"].tolist()
    if set(sets["train"]) & set(sets["val"]):
        raise SystemExit("Fuga: hay imágenes en train y validación a la vez")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    modes, n_boxes, empty = {}, {"train": 0, "val": 0}, 0
    for key, names in sets.items():
        paths = []
        for n in names:
            name = n if n.lower().endswith(".png") else n + ".png"
            src = images_dir / name
            if not src.exists():
                raise SystemExit(f"Falta la imagen {src}")
            m = link_or_copy(src, out / "images" / name)
            modes[m] = modes.get(m, 0) + 1
            with Image.open(src) as im:
                w, h = im.size
            lines = yolo_lines(parse_boxes_string(ann.at[n, "BoxesString"])[0], w, h)
            (out / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                     encoding="utf-8")
            n_boxes[key] += len(lines); empty += not lines
            paths.append(str((out / "images" / name).resolve()))
        (out / f"{key}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    absout = str(out.resolve()).replace("\\", "/")
    (out / "gwhd.yaml").write_text(
        f"path: '{absout}'\ntrain: '{absout}/train.txt'\nval: '{absout}/val.txt'\nnames:\n  0: wheat\n", encoding="utf-8")
    info = {"train_images": len(sets["train"]), "val_images": len(sets["val"]), "train_boxes": n_boxes["train"],
            "val_boxes": n_boxes["val"], "imagenes_sin_cajas": empty, "modo_de_archivos": modes,
            "split_dir": str(split_dir), "test_tocado": False}
    (out / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data/splits/location_seed42")
    ap.add_argument("--images-dir", default="data/raw/gwhd_2021/images")
    ap.add_argument("--out", default="data/yolo")
    a = ap.parse_args()
    print(json.dumps(export(Path(a.split_dir), Path(a.images_dir), Path(a.out)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
