"""Auditoría de GWHD 2021. Uso: python -m src.data.audit [--limit N] [--no-cache]"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import metadata
from pathlib import Path

import imagehash
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data.annotations import parse_boxes_string, validate_boxes  # noqa: E402
from src.data.duplicates import near_duplicate_pairs  # noqa: E402

SPLIT_FILES = {"train": "competition_train.csv", "val": "competition_val.csv", "test": "competition_test.csv"}
REQUIRED_COLS = ["image_name", "BoxesString", "domain"]


# ---------- descubrimiento de archivos ----------
def read_csv_auto(path: Path) -> pd.DataFrame:
    """Detecta ',' o ';' mirando solo la cabecera (BoxesString contiene ';' en las filas)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")

def scan_tree(root: Path):
    csvs: dict[str, list[Path]] = {}
    image_dirs: list[Path] = []
    for dp, dn, fn in os.walk(root):
        image_dirs += [Path(dp) / d for d in dn if d.lower() == "images"]
        for f in fn:
            if f.lower().endswith(".csv"):
                csvs.setdefault(f.lower(), []).append(Path(dp) / f)
    return csvs, image_dirs


def load_annotations(csvs: dict[str, list[Path]]):
    frames, sources = [], {}
    for split, fname in SPLIT_FILES.items():
        hits = csvs.get(fname.lower(), [])
        if len(hits) != 1:
            raise SystemExit(f"Se esperaba exactamente 1 archivo {fname}; encontrados: {hits}")
        df = read_csv_auto(hits[0])
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{fname}: faltan columnas {missing}. Columnas reales: {list(df.columns)}")
        extra = [c for c in df.columns if c not in REQUIRED_COLS]
        sources[split] = {"path": str(hits[0]), "rows": len(df), "extra_columns": extra}
        df = df[REQUIRED_COLS].copy()
        df["competition_split"] = split
        frames.append(df)
    ann = pd.concat(frames, ignore_index=True)
    ann["domain"] = ann["domain"].fillna("unknown")
    return ann, sources


# ---------- inspección de imágenes ----------
def inspect_image(job):
    name, path = job
    rec = {"image_name": name, "file_exists": path.is_file(), "file_size": np.nan, "md5": "",
           "width": np.nan, "height": np.nan, "mode": "", "phash": "", "error": ""}
    if not rec["file_exists"]:
        rec["error"] = "missing_file"
        return rec
    try:
        data = path.read_bytes()
        rec["file_size"] = len(data)
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.load()                                   # fuerza la decodificación completa
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            rec["phash"] = str(imagehash.phash(im.convert("RGB")))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def inspect_images(jobs, workers: int, cache_path: Path, use_cache: bool) -> pd.DataFrame:
    cached = {}
    if use_cache and cache_path.exists():
        c = pd.read_csv(cache_path, dtype={"phash": str, "md5": str}).fillna({"error": "", "phash": "", "md5": ""})
        cached = {r["image_name"]: r for r in c.to_dict("records")}
    records, todo = [], []
    for name, path in jobs:
        c = cached.get(name)
        if c and c["error"] == "" and path.is_file() and int(c["file_size"]) == path.stat().st_size:
            records.append(c)
        else:
            todo.append((name, path))
    print(f"Imágenes: {len(jobs)} | desde caché: {len(records)} | a procesar: {len(todo)}")
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            records += list(tqdm(ex.map(inspect_image, todo), total=len(todo), desc="inspeccionando"))
    df = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ---------- análisis de cajas ----------
def analyse_boxes(table: pd.DataFrame, min_side: float):
    n_boxes, perr, issues, ws, hs = [], [], [], [], []
    for row in table.itertuples(index=False):
        boxes, errs = parse_boxes_string(row.BoxesString)
        n_boxes.append(len(boxes))
        perr.append("|".join(errs))
        if len(boxes):
            ws.append(boxes[:, 2] - boxes[:, 0])
            hs.append(boxes[:, 3] - boxes[:, 1])
            if pd.notna(row.width) and pd.notna(row.height):
                for idx, issue, mag in validate_boxes(boxes, float(row.width), float(row.height), min_side):
                    issues.append({"image_name": row.image_name, "domain": row.domain,
                                   "box_index": idx, "issue": issue, "magnitude": round(mag, 3)})
    w = np.concatenate(ws) if ws else np.array([])
    h = np.concatenate(hs) if hs else np.array([])
    return n_boxes, perr, pd.DataFrame(issues, columns=["image_name", "domain", "box_index", "issue", "magnitude"]), w, h


# ---------- figuras ----------
def fig_hist_boxes(table, out):
    plt.figure(figsize=(8, 4))
    plt.hist(table["n_boxes"], bins=60)
    plt.xlabel("espigas por imagen"); plt.ylabel("imágenes"); plt.title("Distribución de espigas por imagen")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def fig_domains(dom, out):
    d = dom.sort_values("n_images")
    plt.figure(figsize=(8, max(4, 0.25 * len(d))))
    plt.barh(d["domain"].astype(str), d["n_images"])
    plt.xlabel("imágenes"); plt.title("Imágenes por dominio"); plt.tight_layout()
    plt.savefig(out, dpi=120); plt.close()


def fig_box_sizes(w, h, out):
    if len(w) == 0:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(w, bins=80); ax[0].set_title("ancho de caja (px)")
    ax[1].hist(h, bins=80); ax[1].set_title("alto de caja (px)")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()


def fig_samples(table, images_dir, ext, n, seed, out):
    ok = table[(table["file_exists"] == True) & (table["error"] == "")]  # noqa: E712
    if ok.empty:
        return
    sample = ok.sample(n=min(n, len(ok)), random_state=seed)
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, sample.itertuples(index=False)):
        fname = r.image_name if r.image_name.lower().endswith(ext) else r.image_name + ext
        ax.imshow(Image.open(images_dir / fname).convert("RGB"))
        boxes, _ = parse_boxes_string(r.BoxesString)
        for x1, y1, x2, y2 in boxes.tolist():
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=0.8))
        ax.set_title(f"{r.image_name[:18]} | {r.domain} | {r.n_boxes}", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=100); plt.close()


# ---------- informe ----------
def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(vacío)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def lib_versions():
    out = {}
    for p in ["numpy", "pandas", "pillow", "imagehash", "matplotlib", "pyyaml", "requests", "tqdm"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "no instalado"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None, help="auditar solo N imágenes al azar (prueba rápida)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg, ds = cfg["audit"], cfg["dataset"]
    ext = ds["image_ext"].lower()
    root = resolve(cfg["paths"]["extract_dir"])
    if not root.exists():
        raise SystemExit(f"No existe {root}. Ejecuta primero: python -m src.data.download")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_partial" if args.limit else "")
    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "audit" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    csvs, image_dirs = scan_tree(root)
    images_dir = resolve(cfg["paths"]["images_dir"])
    if images_dir is None:
        if len(image_dirs) != 1:
            raise SystemExit(f"Se esperaba 1 carpeta 'images'; encontradas: {image_dirs}. "
                             f"Indica paths.images_dir en configs/data.yaml.")
        images_dir = image_dirs[0]
    print("Carpeta de imágenes:", images_dir)

    ann, sources = load_annotations(csvs)
    n_rows_total = len(ann)
    name_counts = ann["image_name"].value_counts()
    dup_names = name_counts[name_counts > 1].index
    cross_csv_dups = ann[ann["image_name"].isin(dup_names)].groupby("image_name")["competition_split"].nunique()
    n_name_in_multiple_splits = int((cross_csv_dups > 1).sum())
    table = ann.drop_duplicates("image_name", keep="first").reset_index(drop=True)

    if args.limit:
        table = table.sample(n=min(args.limit, len(table)), random_state=acfg["seed"]).reset_index(drop=True)

    jobs = []
    for name in table["image_name"]:
        fname = name if name.lower().endswith(ext) else name + ext
        jobs.append((name, images_dir / fname))
    cache_path = resolve(cfg["paths"]["processed_dir"]) / "image_inventory_cache.csv"
    insp = inspect_images(jobs, acfg["num_workers"], cache_path, use_cache=not args.no_cache)
    table = table.merge(insp, on="image_name", how="left")

    n_boxes, perr, issues, bw, bh = analyse_boxes(table, acfg["tiny_box_min_side_px"])
    table["n_boxes"] = n_boxes
    table["parse_errors"] = perr

    # orphan files (solo en ejecución completa)
    orphan = None
    if not args.limit:
        on_disk = {p.name for p in images_dir.iterdir() if p.suffix.lower() == ext}
        expected = {(n if n.lower().endswith(ext) else n + ext) for n in ann["image_name"]}
        orphan = len(on_disk - expected)

    # duplicados
    ok_hash = table[table["phash"].astype(str).str.len() > 0].reset_index(drop=True)
    md5_groups = (ok_hash.groupby("md5")
              .agg(count=("image_name", "count"),
                   image_names=("image_name", lambda s: ";".join(s))))
    md5_groups = md5_groups[md5_groups["count"] > 1].reset_index()
    hashes = np.array([int(h, 16) for h in ok_hash["phash"]], dtype=np.uint64)
    pairs = near_duplicate_pairs(hashes, acfg["phash_hamming_threshold"]) if len(hashes) else []
    pair_rows = [{
        "image_a": ok_hash.at[i, "image_name"], "image_b": ok_hash.at[j, "image_name"], "hamming": d,
        "same_md5": ok_hash.at[i, "md5"] == ok_hash.at[j, "md5"],
        "domain_a": ok_hash.at[i, "domain"], "domain_b": ok_hash.at[j, "domain"],
        "split_a": ok_hash.at[i, "competition_split"], "split_b": ok_hash.at[j, "competition_split"],
        "cross_competition_split": ok_hash.at[i, "competition_split"] != ok_hash.at[j, "competition_split"],
    } for i, j, d in pairs]
    pairs_df = pd.DataFrame(pair_rows)

    # resúmenes por dominio / split
    dom = (table.groupby("domain").agg(n_images=("image_name", "count"), n_boxes=("n_boxes", "sum"),
                                       mean_boxes=("n_boxes", "mean"), median_boxes=("n_boxes", "median"))
           .round(2).reset_index())
    dom_split = pd.crosstab(table["domain"], table["competition_split"]).reset_index()
    split_sum = (table.groupby("competition_split").agg(n_images=("image_name", "count"), n_boxes=("n_boxes", "sum"),
                                                        mean_boxes=("n_boxes", "mean")).round(2).reset_index())
    size_counts = (table.assign(size=table["width"].astype("Int64").astype(str) + "x" + table["height"].astype("Int64").astype(str))
                   ["size"].value_counts().rename_axis("size").reset_index(name="n_images"))

        # metadatos
    meta_hits = csvs.get("metadata_dataset.csv", []) or csvs.get("metadata.csv", [])
    meta_info = {"found": len(meta_hits)}
    if len(meta_hits) == 1:
        meta = read_csv_auto(meta_hits[0])
        meta = meta.apply(lambda c: c.str.strip() if c.dtype == object else c)
        meta_info.update({"path": str(meta_hits[0]), "shape": list(meta.shape), "columns": list(meta.columns)})
        meta.to_csv(out_dir / "metadata_copy.csv", index=False)
        if "name" in meta.columns:
            meta["domain_key"] = meta["name"].astype(str).str.strip().str.lower()
            dom["domain_key"] = dom["domain"].astype(str).str.strip().str.lower()
            in_data, in_meta = set(dom["domain_key"]), set(meta["domain_key"])
            meta_info["domains_in_data_not_in_metadata"] = sorted(in_data - in_meta)
            if not args.limit:
                meta_info["metadata_not_in_data"] = sorted(in_meta - in_data)
            dom = (dom.merge(meta.drop(columns=["name"]), on="domain_key", how="left")
                      .drop(columns="domain_key"))

    # métricas del resumen
    exp_w, exp_h = ds["expected_image_size"]
    unexpected_size = int(((table["width"] != exp_w) | (table["height"] != exp_h)).sum()) if table["width"].notna().any() else 0
    n_missing = int((~table["file_exists"].fillna(False).astype(bool)).sum())
    n_unreadable = int(((table["error"].fillna("") != "") & (table["error"] != "missing_file")).sum())
    n_parse_err = int((table["parse_errors"] != "").sum())
    issue_counts = issues["issue"].value_counts().to_dict() if len(issues) else {}
    n_cross_pairs = int(pairs_df["cross_competition_split"].sum()) if len(pairs_df) else 0

    critical = {
        "filas_CSV_sin_archivo_de_imagen": n_missing,
        "imagenes_ilegibles_o_corruptas": n_unreadable,
        "imagenes_con_errores_de_parseo_de_cajas": n_parse_err,
        "cajas_no_finitas_o_tamano_no_positivo": issue_counts.get("non_finite", 0) + issue_counts.get("non_positive_size", 0),
        "image_name_en_varios_CSV_de_splits_distintos": n_name_in_multiple_splits,
    }
    warnings = {
        "cajas_fuera_de_limites (recortables en Fase 2)": issue_counts.get("out_of_bounds", 0),
        "cajas_diminutas": issue_counts.get("tiny_box", 0),
        f"imagenes_con_tamano_distinto_de_{exp_w}x{exp_h}": unexpected_size,
        "grupos_duplicados_exactos_md5": len(md5_groups),
        "pares_casi_duplicados_pHash": len(pairs_df),
        "pares_casi_duplicados_entre_splits_oficiales": n_cross_pairs,
        "imagenes_sin_espigas": int((table["n_boxes"] == 0).sum()),
        "archivos_huerfanos_en_disco": "n/d (ejecución parcial)" if orphan is None else orphan,
    }
    summary = [("run_id", run_id), ("partial_run", bool(args.limit)), ("filas_CSV_total", n_rows_total),
               ("imagenes_unicas_auditadas", len(table)), ("instancias_total", int(table["n_boxes"].sum())),
               ("n_dominios", table["domain"].nunique())]
    summary += [(f"CRITICO:{k}", v) for k, v in critical.items()] + [(f"AVISO:{k}", v) for k, v in warnings.items()]
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(out_dir / "audit_summary.csv", index=False)

    # salidas CSV
    table.drop(columns=["BoxesString"]).to_csv(out_dir / "image_table.csv", index=False)
    issues.to_csv(out_dir / "invalid_boxes.csv", index=False)
    md5_groups.to_csv(out_dir / "exact_duplicate_groups.csv", index=False)
    pairs_df.to_csv(out_dir / "near_duplicate_pairs.csv", index=False)
    dom.to_csv(out_dir / "domain_summary.csv", index=False)
    dom_split.to_csv(out_dir / "domain_by_competition_split.csv", index=False)
    split_sum.to_csv(out_dir / "competition_split_summary.csv", index=False)
    size_counts.to_csv(out_dir / "image_size_counts.csv", index=False)
    n_stats = table["n_boxes"].describe().round(2).to_frame("n_boxes").T
    n_stats.to_csv(out_dir / "boxes_per_image_stats.csv")
    if len(bw):
        pd.DataFrame({"w": bw, "h": bh, "area": bw * bh}).describe().round(2).T.to_csv(out_dir / "box_size_stats.csv")

    # figuras
    fig_hist_boxes(table, out_dir / "hist_boxes_per_image.png")
    fig_domains(dom, out_dir / "images_per_domain.png")
    fig_box_sizes(bw, bh, out_dir / "box_sizes.png")
    fig_samples(table, images_dir, ext, acfg["n_sample_images"], acfg["seed"], out_dir / "samples_with_boxes.png")

    # configuración y versiones
    (out_dir / "run_config.json").write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"), "python": sys.version, "platform": platform.platform(),
        "libraries": lib_versions(), "config": cfg, "csv_sources": sources, "limit": args.limit,
    }, indent=2, default=str))

    # informe Markdown
    blocked = any(v for v in critical.values())
    verdict = ("BLOQUEADO: hay errores críticos sin resolver." if blocked
               else "Sin errores críticos detectados en esta ejecución.")
    if args.limit:
        verdict += " (EJECUCIÓN PARCIAL: no sustituye a la auditoría completa.)"
    md = f"""# Informe de auditoría GWHD 2021 ({run_id})

**Veredicto:** {verdict}

## Errores críticos (deben ser 0 para continuar)
{df_to_md(pd.DataFrame(list(critical.items()), columns=['comprobación', 'valor']))}

## Avisos (a decidir en la Fase 2)
{df_to_md(pd.DataFrame([(k, v) for k, v in warnings.items()], columns=['comprobación', 'valor']))}

## Totales
- Filas en los 3 CSV de la competición: {n_rows_total}
- Imágenes únicas auditadas: {len(table)}
- Instancias (espigas) en esas imágenes: {int(table['n_boxes'].sum())}
- Dominios: {table['domain'].nunique()}

## Por split oficial de la competición (no es nuestra partición 70/15/15)
{df_to_md(split_sum)}

## Espigas por imagen
{df_to_md(n_stats.reset_index(names='variable'))}

## Tamaños de imagen
{df_to_md(size_counts)}

## Dominios
{df_to_md(dom)}

## Metadata.csv
- Archivos encontrados: {meta_info['found']}
- Ruta: {meta_info.get('path', 'no encontrado')}
- Dimensiones: {meta_info.get('shape', 'n/d')}
- Columnas: {meta_info.get('columns', 'n/d')}
"""
    report_path = out_dir / "report.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"\nInforme guardado en: {report_path}")
    print(f"Veredicto: {verdict}")


if __name__ == "__main__":
    main()
