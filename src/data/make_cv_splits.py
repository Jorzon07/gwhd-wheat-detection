"""Particiones de validación cruzada sobre TRAIN+VALIDACIÓN (el test no se lee ni se copia).

Dos variantes, mismos tamaños aproximados (k pliegues):
  random   : pliegues por imagen al azar (las imágenes de una misma sesión caen en train y validación: hay fuga)
  grouped  : pliegues por 'location' (ninguna ubicación aparece en train y validación a la vez)
Por cada pliegue crea data/splits/cv/<variante>_k<k>_f<i>/{train,validation}.csv + annotations_clean.csv y un YAML de
configuración (copia de configs/data.yaml con phase3.split_dir apuntando al pliegue, sin parada anticipada, eval cada 2 épocas).

Uso: python -m src.data.make_cv_splits [--k 3] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def random_folds(n: int, k: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    fold = np.empty(n, dtype=int)
    fold[perm] = np.arange(n) % k
    return fold


def grouped_folds(groups: pd.Series, k: int) -> np.ndarray:
    sizes = groups.value_counts()
    load, assign = [0] * k, {}
    for g, s in sorted(sizes.items(), key=lambda t: (-t[1], str(t[0]))):      # los grupos grandes primero, al pliegue más vacío
        f = int(np.argmin(load)); assign[g] = f; load[f] += int(s)
    return groups.map(assign).to_numpy()


def build(split_dir: Path, out_root: Path, base_cfg: Path, k: int, seed: int) -> list[dict]:
    pool = pd.concat([pd.read_csv(split_dir / "train.csv"), pd.read_csv(split_dir / "validation.csv")], ignore_index=True)
    assert pool["image_name"].is_unique, "imágenes repetidas entre train y validación"
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    info = []
    for variant in ("random", "grouped"):
        fold_of = random_folds(len(pool), k, seed) if variant == "random" else grouped_folds(pool["location"], k)
        for i in range(k):
            d = out_root / f"{variant}_k{k}_f{i}"
            d.mkdir(parents=True, exist_ok=True)
            va, tr = pool[fold_of == i].copy(), pool[fold_of != i].copy()
            tr["split"], va["split"] = "train", "validation"
            tr.to_csv(d / "train.csv", index=False); va.to_csv(d / "validation.csv", index=False)
            shutil.copy2(split_dir / "annotations_clean.csv", d / "annotations_clean.csv")
            c = json.loads(json.dumps(cfg))
            c["phase3"]["split_dir"] = d.as_posix()
            c["train"]["patience"] = 0          # sin parada anticipada: todos los pliegues entrenan lo mismo
            c["train"]["eval_every"] = 2
            (d / "config.yaml").write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
            shared = sorted(set(tr["location"]) & set(va["location"]))
            row = {"variant": variant, "fold": i, "dir": d.as_posix(), "n_train": len(tr), "n_val": len(va),
                   "val_locations": sorted(set(va["location"])), "val_countries": sorted(set(va["country"])),
                   "locations_in_both": shared, "val_boxes": int(va["n_boxes"].sum())}
            assert not (set(tr["image_name"]) & set(va["image_name"])), "fuga directa de imágenes"
            if variant == "grouped":
                assert not shared, f"ubicaciones compartidas en {d}: {shared}"
            info.append(row)
    (out_root / "cv_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data/splits/location_seed42")
    ap.add_argument("--out", default="data/splits/cv")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    for r in build(Path(a.split_dir), Path(a.out), Path(a.config), a.k, a.seed):
        print(f"{r['variant']:8s} f{r['fold']}: train {r['n_train']:5d} | val {r['n_val']:5d} ({r['val_boxes']} espigas) | "
              f"ubicaciones en val: {len(r['val_locations'])} | compartidas con train: {len(r['locations_in_both'])} | "
              f"países en val: {len(r['val_countries'])}")


if __name__ == "__main__":
    main()
