"""Genera imágenes sintéticas etiquetadas con el pix2pix entrenado y construye la partición de entrenamiento aumentada.

Para cada imagen sintética: se compone un layout de cajas a partir de recortes (con volteo/giro) de varias imágenes de
TRAIN, el generador lo convierte en imagen y las cajas del layout son sus etiquetas. Salida:
  data/gan/images_all/            enlaces a las imágenes reales de train y validación + las sintéticas (la de prueba NO)
  data/splits/<split-name>/       train.csv (train real + sintéticas), validation.csv (idéntico), annotations_clean.csv
                                  (solo train+validación+sintéticas), config.yaml (copia de configs/data.yaml apuntando aquí)
Nunca se lee test.csv ni ninguna imagen de prueba.

Uso:  python -m src.gan.make_synthetic --n 1500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw

from src.config import load_config, resolve
from src.data.annotations import boxes_to_string
from src.data.samples import samples_from_config
from src.gan.layout import compose_canvas, render_layout
from src.gan.models import UNetGenerator, generation_mode


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--generator", default="outputs/gan/pix2pix/generator.pt")
    ap.add_argument("--split-name", default="gan_aug")
    ap.add_argument("--images-out", default="data/gan/images_all")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=1024)
    ap.add_argument("--quad", type=int, default=512)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    p3 = cfg["phase3"]
    base = resolve(p3["split_dir"])
    images_dir, ext = resolve(p3["images_dir"]), cfg["dataset"]["image_ext"]
    out_split, out_imgs = resolve(f"data/splits/{a.split_name}"), resolve(a.images_out)
    if (out_split / "synthetic_info.json").exists():
        raise SystemExit(f"{out_split} ya existe (synthetic_info.json). Bórralo si quieres regenerar.")
    out_split.mkdir(parents=True, exist_ok=True)
    out_imgs.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(base / "train.csv")
    val = pd.read_csv(base / "validation.csv")
    samples = samples_from_config(cfg, "train", train=False, image_size=1024)
    pool = [samples.boxes[n] for n in samples.names]
    real_mean = float(np.mean([len(b) for b in pool]))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.no_amp and torch.cuda.is_bf16_supported()
    ck = torch.load(resolve(a.generator), map_location="cpu", weights_only=True)
    G = UNetGenerator(2, 3, int(ck["ngf"]), int(ck["num_downs"])).to(dev)
    G.load_state_dict(ck["G"])
    generation_mode(G)
    if a.canvas % (2 ** int(ck["num_downs"])):
        raise SystemExit(f"--canvas debe ser múltiplo de {2 ** int(ck['num_downs'])}")

    rng = random.Random(a.seed)
    rows, ann_rows, nb = [], [], []
    sheet = []
    for i in range(a.n):
        boxes = compose_canvas(pool, rng, a.canvas, a.quad, 1024)
        cond = torch.from_numpy(render_layout(boxes, a.canvas)).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            img = G(cond)[0].float().clamp(-1, 1)
        arr = ((img.cpu().numpy() + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
        name = f"synth_{i:05d}{ext}"
        Image.fromarray(arr).save(out_imgs / name)
        nb.append(len(boxes))
        rows.append({"image_name": name, "split": "train", "group": "synthetic", "domain": "synthetic",
                     "country": "synthetic", "location": "synthetic", "development_stage": "synthetic",
                     "n_boxes": len(boxes), "official_competition_split": "synthetic", "md5": ""})
        ann_rows.append({"image_name": name, "BoxesString": boxes_to_string(np.round(boxes, 1))})
        if i < 4:
            im = Image.fromarray(arr).convert("RGB")
            dr = ImageDraw.Draw(im)
            for x1, y1, x2, y2 in boxes:
                dr.rectangle([x1, y1, x2, y2], outline=(255, 255, 0))
            sheet.append(np.asarray(im.resize((512, 512))))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{a.n} imágenes sintéticas", flush=True)

    for n in list(train["image_name"]) + list(val["image_name"]):
        link_or_copy(images_dir / n, out_imgs / n)
    syn = pd.DataFrame(rows)
    pd.concat([train, syn], ignore_index=True).to_csv(out_split / "train.csv", index=False)
    val.to_csv(out_split / "validation.csv", index=False)
    ann = pd.read_csv(base / "annotations_clean.csv", dtype={"image_name": str})
    keep = ann[ann["image_name"].isin(set(train["image_name"]) | set(val["image_name"]))]
    pd.concat([keep, pd.DataFrame(ann_rows)], ignore_index=True).to_csv(out_split / "annotations_clean.csv", index=False)
    c = yaml.safe_load((resolve(a.config)).read_text(encoding="utf-8"))
    c["phase3"]["split_dir"] = f"data/splits/{a.split_name}"
    c["phase3"]["images_dir"] = a.images_out
    (out_split / "config.yaml").write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
    if sheet:
        Image.fromarray(np.concatenate(sheet, axis=1)).save(out_split / "synthetic_examples.png")
    info = {"n_synthetic": a.n, "n_real_train": len(train), "n_val": len(val), "mean_boxes_synthetic": float(np.mean(nb)),
            "mean_boxes_real_train": real_mean, "generator": str(a.generator), "canvas": a.canvas, "quad": a.quad,
            "generator_iterations": int(ck.get("it", -1)), "seed": a.seed}
    (out_split / "synthetic_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nListo: {a.n} imágenes sintéticas (cajas por imagen: {np.mean(nb):.1f} sintéticas vs {real_mean:.1f} reales).\n"
          f"Partición en {out_split}; ejemplos con cajas en {out_split / 'synthetic_examples.png'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
