"""Seguimiento de la auditoría completa. SOLO LECTURA sobre los datos.

Escribe en outputs/audit/followup_<fecha>/ :
  repeated_name_rows.csv, orphan_files.csv, empty_images_by_domain.csv,
  near_duplicate_detail.csv, near_duplicates_side_by_side.png, out_of_bounds_images.png
Uso: python -m src.data.inspect_findings [--audit-run NOMBRE_CARPETA]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data.annotations import parse_boxes_string  # noqa: E402

SPLITS = {"train": "competition_train.csv", "val": "competition_val.csv", "test": "competition_test.csv"}


def read_csv_auto(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def img_path(images_dir: Path, name: str, ext: str) -> Path:
    return images_dir / (name if name.lower().endswith(ext) else name + ext)


def draw(ax, path: Path, boxes, title: str, highlight=(), pad=0):
    ax.imshow(Image.open(path).convert("RGB"))
    for i, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        color = "cyan" if i in highlight else "red"
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.0))
    if pad:
        ax.set_xlim(-pad, 1024 + pad)
        ax.set_ylim(1024 + pad, -pad)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--audit-run", default=None, help="carpeta dentro de outputs/audit (por defecto, la última completa)")
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
            raise SystemExit("No hay ejecuciones completas de auditoría en " + str(audit_root))
        run_dir = runs[-1]
    print("Usando la auditoría:", run_dir.name)

    out = audit_root / ("followup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)

    frames = []
    for split, fname in SPLITS.items():
        df = read_csv_auto(root / fname)
        df["competition_split"] = split
        frames.append(df)
    ann = pd.concat(frames, ignore_index=True)
    ann["n_boxes"] = [len(parse_boxes_string(v)[0]) for v in ann["BoxesString"]]
    uniq = ann.drop_duplicates("image_name", keep="first").set_index("image_name")

    # 1) nombres repetidos en los CSV
    rep = ann[ann.duplicated("image_name", keep=False)].sort_values("image_name").copy()
    if len(rep):
        rep["same_boxes_in_all_rows"] = rep.groupby("image_name")["BoxesString"].transform(lambda s: s.nunique() == 1)
    rep.drop(columns=["BoxesString"]).to_csv(out / "repeated_name_rows.csv", index=False)
    print(f"\n[1] Filas con image_name repetido: {len(rep)}")
    print(rep.drop(columns=["BoxesString"]).to_string(index=False) if len(rep) else "  (ninguna)")

    # 2) archivos en disco sin fila en los CSV
    on_disk = {p.name for p in images_dir.iterdir() if p.suffix.lower() == ext}
    expected = {(n if n.lower().endswith(ext) else n + ext) for n in ann["image_name"]}
    orphans = sorted(on_disk - expected)
    missing = sorted(expected - on_disk)
    pd.DataFrame({"orphan_file": orphans}).to_csv(out / "orphan_files.csv", index=False)
    print(f"\n[2] Archivos en disco: {len(on_disk)} | huérfanos: {len(orphans)} | filas sin archivo: {len(missing)}")
    for o in orphans:
        print("   huérfano:", o)

    # 3) imágenes sin espigas por dominio
    g = uniq.groupby("domain").agg(n_images=("n_boxes", "size"), n_empty=("n_boxes", lambda s: int((s == 0).sum())))
    g["pct_empty"] = (100 * g["n_empty"] / g["n_images"]).round(1)
    g = g[g["n_empty"] > 0].sort_values("n_empty", ascending=False)
    g.to_csv(out / "empty_images_by_domain.csv")
    print(f"\n[3] Imágenes sin espigas por dominio (total {int((uniq['n_boxes'] == 0).sum())}):")
    print(g.to_string())

    # 4) pares duplicados / casi duplicados
    pairs_path = run_dir / "near_duplicate_pairs.csv"
    try:
        pairs = pd.read_csv(pairs_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pairs = pd.DataFrame()
    if len(pairs):
        rows = []
        for p in pairs.itertuples(index=False):
            a, b = uniq.loc[p.image_a], uniq.loc[p.image_b]
            rows.append({"image_a": p.image_a, "image_b": p.image_b, "hamming": p.hamming, "same_md5": p.same_md5,
                         "domain_a": a["domain"], "domain_b": b["domain"],
                         "n_boxes_a": a["n_boxes"], "n_boxes_b": b["n_boxes"],
                         "identical_annotations": a["BoxesString"] == b["BoxesString"]})
        detail = pd.DataFrame(rows)
        detail.to_csv(out / "near_duplicate_detail.csv", index=False)
        print("\n[4] Pares duplicados/casi duplicados:")
        print(detail.drop(columns=["image_a", "image_b"]).to_string(index=False))
        fig, axes = plt.subplots(len(detail), 2, figsize=(8, 4 * len(detail)), squeeze=False)
        for r, p in enumerate(detail.itertuples(index=False)):
            for c, (name, dom, nb) in enumerate([(p.image_a, p.domain_a, p.n_boxes_a),
                                                 (p.image_b, p.domain_b, p.n_boxes_b)]):
                boxes, _ = parse_boxes_string(uniq.loc[name, "BoxesString"])
                draw(axes[r][c], img_path(images_dir, name, ext), boxes,
                     f"{dom} | {nb} cajas | h={p.hamming} md5={p.same_md5}")
        plt.tight_layout()
        plt.savefig(out / "near_duplicates_side_by_side.png", dpi=110)
        plt.close()
    else:
        print("\n[4] No hay pares en near_duplicate_pairs.csv")

    # 5) cajas fuera de límites
    inv_path = run_dir / "invalid_boxes.csv"
    try:
        inv = pd.read_csv(inv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        inv = pd.DataFrame(columns=["image_name", "domain", "box_index", "issue", "magnitude"])
    oob = inv[inv["issue"] == "out_of_bounds"]
    names = list(oob["image_name"].drop_duplicates())
    print(f"\n[5] Imágenes con cajas fuera de límites: {len(names)}")
    if names:
        print(oob.groupby(["domain", "image_name"]).agg(n=("box_index", "size"),
                                                          max_overflow_px=("magnitude", "max")).to_string())
        fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.5), squeeze=False)
        for ax, name in zip(axes[0], names):
            boxes, _ = parse_boxes_string(uniq.loc[name, "BoxesString"])
            idx = set(oob.loc[oob["image_name"] == name, "box_index"])
            draw(ax, img_path(images_dir, name, ext), boxes,
                 f"{uniq.loc[name, 'domain']} | {len(idx)} fuera (cian)", idx, pad=80)
        plt.tight_layout()
        plt.savefig(out / "out_of_bounds_images.png", dpi=110)
        plt.close()

    print("\nSalidas en:", out)


if __name__ == "__main__":
    main()