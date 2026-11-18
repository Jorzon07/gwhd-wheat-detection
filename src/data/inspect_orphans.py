"""Prueba la hipótesis: los archivos huérfanos (en disco, sin fila en los CSV) son las imágenes
a las que correspondían las filas con image_name repetido. SOLO LECTURA.

Salidas en outputs/audit/followup_orphans_<fecha>/ : orphan_nearest.csv, name_vs_orphan_hamming.csv,
repeated_vs_orphans.png. Uso: python -m src.data.inspect_orphans [--audit-run CARPETA]
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data.annotations import parse_boxes_string  # noqa: E402
from src.data.inspect_findings import SPLITS, draw, img_path, read_csv_auto  # noqa: E402


def ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--audit-run", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ext = cfg["dataset"]["image_ext"].lower()
    root = resolve(cfg["paths"]["extract_dir"])
    images_dir = root / "images"
    audit_root = resolve(cfg["paths"]["outputs_dir"]) / "audit"
    if args.audit_run:
        run_dir = audit_root / args.audit_run
    else:
        runs = sorted(d for d in audit_root.iterdir() if d.is_dir() and not d.name.endswith("_partial")
                      and not d.name.startswith("followup_"))
        if not runs:
            raise SystemExit("No hay auditorías completas en " + str(audit_root))
        run_dir = runs[-1]
    print("Usando la auditoría:", run_dir.name)

    ann = pd.concat([read_csv_auto(root / f).assign(competition_split=s) for s, f in SPLITS.items()],
                    ignore_index=True)
    ann["n_boxes"] = [len(parse_boxes_string(v)[0]) for v in ann["BoxesString"]]

    on_disk = {p.name for p in images_dir.iterdir() if p.suffix.lower() == ext}
    expected = {(n if n.lower().endswith(ext) else n + ext) for n in ann["image_name"]}
    orphans = sorted(on_disk - expected)
    rep_names = sorted(ann.loc[ann.duplicated("image_name", keep=False), "image_name"].unique())
    print(f"Huérfanos: {len(orphans)} | nombres repetidos: {len(rep_names)}")
    if not orphans or not rep_names:
        raise SystemExit("Nada que comparar.")

    tbl_path = run_dir / "image_table.csv"
    tbl = pd.read_csv(tbl_path, dtype={"phash": str, "md5": str})
    for col in ("image_name", "md5", "phash", "domain"):
        if col not in tbl.columns:
            raise SystemExit(f"Falta la columna {col} en {tbl_path}. Columnas: {list(tbl.columns)}")
    tbl = tbl[tbl["phash"].fillna("").str.len() > 0].reset_index(drop=True)
    tbl["h"] = [int(x, 16) for x in tbl["phash"]]
    by_name = tbl.set_index("image_name")

    out = audit_root / ("followup_orphans_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)

    # hashes de los huérfanos y vecinos más cercanos
    orphan_h, nearest_rows = {}, []
    for o in orphans:
        data = (images_dir / o).read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(images_dir / o) as im:
            h = int(str(imagehash.phash(im.convert("RGB"))), 16)
        orphan_h[o] = h
        d = sorted(((ham(h, x), n, dom, m == md5) for x, n, dom, m in
                    zip(tbl["h"], tbl["image_name"], tbl["domain"], tbl["md5"])), key=lambda t: t[0])[:3]
        for rank, (dist, n, dom, same) in enumerate(d, 1):
            nearest_rows.append({"orphan": o, "rank": rank, "nearest": n, "domain": dom, "hamming": dist, "same_md5": same})
    nearest = pd.DataFrame(nearest_rows)
    nearest.to_csv(out / "orphan_nearest.csv", index=False)
    print("\nVecinos más cercanos de cada huérfano (pHash):")
    print(nearest.assign(orphan=nearest["orphan"].str[:12], nearest=nearest["nearest"].str[:12]).to_string(index=False))

    # distancia entre cada nombre repetido y cada huérfano
    mat = pd.DataFrame({o[:12]: [ham(int(by_name.loc[r, "phash"], 16), orphan_h[o]) for r in rep_names]
                        for o in orphans}, index=[r[:12] for r in rep_names])
    mat.to_csv(out / "name_vs_orphan_hamming.csv")
    print("\nDistancia pHash: filas = nombres repetidos, columnas = huérfanos")
    print(mat.to_string())

    # figura: cada nombre repetido con sus dos juegos de cajas, sobre el archivo nombrado y sobre su huérfano más parecido
    fig, axes = plt.subplots(len(rep_names), 4, figsize=(16, 4 * len(rep_names)), squeeze=False)
    for r_i, r in enumerate(rep_names):
        rows = ann[ann["image_name"] == r].reset_index(drop=True)
        rh = int(by_name.loc[r, "phash"], 16)
        best = min(orphans, key=lambda o: ham(rh, orphan_h[o]))
        panels = []
        for target_name, target_path in [("archivo nombrado", img_path(images_dir, r, ext)),
                                         (f"huérfano {best[:8]}", images_dir / best)]:
            for k in range(min(2, len(rows))):
                panels.append((target_name, target_path, rows.loc[k]))
        for ax, (tname, tpath, row) in zip(axes[r_i], panels):
            boxes, _ = parse_boxes_string(row["BoxesString"])
            draw(ax, tpath, boxes, f"{r[:8]} | {tname}\nfila: {row['domain']} ({row['n_boxes']} cajas)")
    plt.tight_layout()
    plt.savefig(out / "repeated_vs_orphans.png", dpi=90)
    plt.close()
    print("\nSalidas en:", out)


if __name__ == "__main__":
    main()