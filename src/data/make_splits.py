"""Fase 2: limpieza (reglas aprobadas en la Fase 1) + partición agrupada reproducible.

Uso:  python -m src.data.make_splits [--overwrite]
Salida: data/splits/<group_key>_seed<seed>/  (no se sobrescribe sin --overwrite)
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from importlib import metadata as md

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config, resolve  # noqa: E402
from src.data.annotations import boxes_to_string, parse_boxes_string  # noqa: E402
from src.data.cleaning import clean_boxes  # noqa: E402
from src.data.io_utils import OFFICIAL_FILES, latest_complete_audit, read_csv_auto, sha256_of  # noqa: E402
from src.data.split import (SETS, assign_groups, composition_summary, location_overlap, merge_groups,  # noqa: E402
                            unseen_countries, verify_composition, verify_splits)

GROUP_KEYS = ("domain", "location", "country")
# columnas que no pueden compartirse entre conjuntos según la clave de agrupación (agrupar por una clave más
# gruesa implica también que las más finas queden disjuntas)
DISJOINT = {"domain": ("domain",), "location": ("location", "domain"), "country": ("country", "location", "domain")}


def build_clean_table(cfg: dict, run_dir, root, images_dir, min_side: float):
    """Aplica las reglas de la Fase 1. Devuelve (tabla limpia, dict de cajas limpias, cuarentena, eventos de cajas)."""
    ext = cfg["dataset"]["image_ext"].lower()
    exp_w, exp_h = cfg["dataset"]["expected_image_size"]

    ann = pd.concat([read_csv_auto(root / f).assign(competition_split=s) for s, f in OFFICIAL_FILES.items()],
                    ignore_index=True)
    quarantine: list[dict] = []

    # 1) nombres repetidos: se conservan solo si todas las filas son idénticas (mismo dominio y mismas cajas)
    keep_rows = []
    for name, grp in ann.groupby("image_name", sort=True):
        if len(grp) == 1:
            keep_rows.append(grp.iloc[0])
        elif grp["BoxesString"].nunique() == 1 and grp["domain"].nunique() == 1:
            keep_rows.append(grp.iloc[0])
        else:
            quarantine.append({"image_name": name, "reason": "repeated_name_conflicting_rows",
                               "detail": f"{len(grp)} filas; dominios={sorted(grp['domain'].astype(str).unique())}"})
    uniq = pd.DataFrame(keep_rows).reset_index(drop=True)

    # 2) archivos huérfanos (informativo: no tienen fila, nunca entran en las tablas)
    if images_dir.is_dir():
        on_disk = {p.name for p in images_dir.iterdir() if p.suffix.lower() == ext}
        expected = {(n if n.lower().endswith(ext) else n + ext) for n in ann["image_name"]}
        for o in sorted(on_disk - expected):
            quarantine.append({"image_name": o, "reason": "orphan_file_no_annotation", "detail": "sin fila en los CSV"})

    # 3) hashes y tamaños de la auditoría
    tbl = pd.read_csv(run_dir / "image_table.csv", dtype={"md5": str, "phash": str})
    need = ["image_name", "md5", "phash", "width", "height"]
    miss_cols = [c for c in need if c not in tbl.columns]
    if miss_cols:
        raise SystemExit(f"image_table.csv no tiene las columnas {miss_cols}")
    uniq = uniq.merge(tbl[need], on="image_name", how="left", validate="one_to_one")
    if uniq["md5"].isna().any():
        raise SystemExit(f"{int(uniq['md5'].isna().sum())} imágenes sin md5 en la auditoría; repite la auditoría completa.")
    if ((uniq["width"] != exp_w) | (uniq["height"] != exp_h)).any():
        raise SystemExit("Hay imágenes con tamaño distinto del esperado; la limpieza de cajas asume el tamaño del dataset.")

    # 4) duplicados exactos (mismo md5)
    drop_names: set[str] = set()
    for md5, grp in uniq.groupby("md5"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("image_name")
        if grp["BoxesString"].nunique() == 1:
            for n in grp["image_name"].iloc[1:]:
                quarantine.append({"image_name": n, "reason": "exact_duplicate_removed",
                                   "detail": f"copia de {grp['image_name'].iloc[0][:12]}"})
                drop_names.add(n)
        else:
            for n in grp["image_name"]:
                quarantine.append({"image_name": n, "reason": "exact_duplicate_conflicting_labels",
                                   "detail": f"md5={md5[:8]}; cajas por copia={[len(parse_boxes_string(v)[0]) for v in grp['BoxesString']]}"})
                drop_names.add(n)
    uniq = uniq[~uniq["image_name"].isin(drop_names)].reset_index(drop=True)

    # 5) metadatos por dominio
    meta = read_csv_auto(root / "metadata_dataset.csv")
    for col in meta.columns:
        meta[col] = meta[col].astype(str).str.strip()
    meta["_k"] = meta["name"].str.lower()
    uniq["_k"] = uniq["domain"].astype(str).str.strip().str.lower()
    uniq = uniq.merge(meta.drop(columns=["name"]), on="_k", how="left")
    if uniq["country"].isna().any():
        bad = sorted(uniq.loc[uniq["country"].isna(), "domain"].unique())
        raise SystemExit(f"Dominios sin metadatos: {bad}")
    uniq = uniq.drop(columns=["_k"])

    # 6) limpieza de cajas
    clean_strings, n_raw, n_clean, events = [], [], [], []
    boxes_by_image: dict[str, np.ndarray] = {}
    for r in uniq.itertuples(index=False):
        raw, errs = parse_boxes_string(r.BoxesString)
        if errs:
            raise SystemExit(f"Errores de parseo en {r.image_name}: {errs}")
        cleaned, ev = clean_boxes(raw, r.width, r.height, min_side)
        for idx, action, detail in ev:
            events.append({"image_name": r.image_name, "domain": r.domain, "box_index": idx,
                           "action": action, "detail": detail})
        boxes_by_image[r.image_name] = cleaned
        clean_strings.append(boxes_to_string(cleaned))
        n_raw.append(len(raw))
        n_clean.append(len(cleaned))
    uniq["BoxesString_clean"] = clean_strings
    uniq["n_boxes_raw"] = n_raw
    uniq["n_boxes"] = n_clean
    uniq = uniq.rename(columns={"competition_split": "official_competition_split"})
    return (uniq, boxes_by_image, pd.DataFrame(quarantine, columns=["image_name", "reason", "detail"]),
            pd.DataFrame(events, columns=["image_name", "domain", "box_index", "action", "detail"]))


def df_to_md(df: pd.DataFrame) -> str:
    cols = [df.index.name or "index"] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for idx in df.index:
        lines.append("| " + " | ".join([str(idx)] + [str(df.at[idx, c]) for c in df.columns]) + " |")
    return "\n".join(lines)


def lib_versions() -> dict:
    out = {}
    for p in ("numpy", "pandas", "scipy", "matplotlib", "pyyaml"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "no instalado"
    return out


def make_figure(clean: pd.DataFrame, boxes_by_image: dict, targets: dict, path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    tot_i, tot_b = len(clean), clean["n_boxes"].sum()
    x = np.arange(len(SETS))
    img_share = [(clean["split"] == s).sum() / tot_i for s in SETS]
    box_share = [clean.loc[clean["split"] == s, "n_boxes"].sum() / tot_b for s in SETS]
    ax[0, 0].bar(x - 0.2, img_share, 0.4, label="imágenes")
    ax[0, 0].bar(x + 0.2, box_share, 0.4, label="espigas")
    for i, s in enumerate(SETS):
        ax[0, 0].hlines(targets[s], i - 0.4, i + 0.4, colors="k", linestyles="--")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(SETS); ax[0, 0].set_title("Proporción por conjunto (--- objetivo)")
    ax[0, 0].legend()
    for s in SETS:
        ax[0, 1].hist(clean.loc[clean["split"] == s, "n_boxes"], bins=50, histtype="step", density=True, label=s)
    ax[0, 1].set_title("Espigas por imagen (densidad)"); ax[0, 1].legend()
    comp = pd.crosstab(clean["split"], clean["country"]).reindex(list(SETS))
    comp.plot(kind="bar", stacked=True, ax=ax[1, 0], legend=True, width=0.7, colormap="tab20")
    ax[1, 0].set_title("Imágenes por país"); ax[1, 0].legend(fontsize=6, ncol=2); ax[1, 0].tick_params(axis="x", rotation=0)
    split_of = dict(zip(clean["image_name"], clean["split"]))
    for s in SETS:
        ws = [b[:, 2] - b[:, 0] for n, b in boxes_by_image.items() if len(b) and split_of[n] == s]
        if ws:
            ax[1, 1].hist(np.concatenate(ws), bins=60, histtype="step", density=True, label=s)
    ax[1, 1].set_title("Ancho de caja (px)"); ax[1, 1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p2 = cfg["phase2"]
    seed, key, targets = int(p2["seed"]), p2["group_key"], p2["targets"]
    if key not in GROUP_KEYS:
        raise SystemExit(f"group_key debe ser uno de {GROUP_KEYS}")
    root = resolve(cfg["paths"]["extract_dir"])
    audit_root = resolve(cfg["paths"]["outputs_dir"]) / "audit"
    run_dir = (audit_root / p2["audit_run"]) if p2.get("audit_run") else latest_complete_audit(audit_root)
    out_dir = resolve(p2["output_dir"]) / f"{key}_seed{seed}"
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} ya existe. Usa --overwrite solo si quieres reemplazarla.")
        shutil.rmtree(out_dir)
    print("Auditoría de referencia:", run_dir.name)

    clean, boxes_by_image, quarantine, events = build_clean_table(
        cfg, run_dir, root, root / "images", float(p2["min_box_side_px"]))
    print(f"Imágenes tras limpieza: {len(clean)} | en cuarentena/excluidas: {len(quarantine)} | eventos de cajas: {len(events)}")

    # enlaces por casi duplicados (pHash) entre imágenes que siguen en la tabla
    link_pairs: list[tuple[str, str]] = []
    pairs_path = run_dir / "near_duplicate_pairs.csv"
    if pairs_path.exists() and pairs_path.stat().st_size > 0:
        pairs = pd.read_csv(pairs_path)
        present = set(clean["image_name"])
        link_pairs = [(a, b) for a, b in zip(pairs["image_a"], pairs["image_b"]) if a in present and b in present]
    clean["group_raw"] = clean[key].astype(str)
    gl = dict(zip(clean["image_name"], clean["group_raw"]))
    label = merge_groups(sorted(clean["group_raw"].unique()), [(gl[a], gl[b]) for a, b in link_pairs])
    clean["group"] = clean["group_raw"].map(label)

    sizes = clean.groupby("group").agg(n_images=("image_name", "size"), n_boxes=("n_boxes", "sum"))
    domain_sizes = clean.groupby(["group", "domain"]).size().rename("n_images").reset_index()
    country_sizes = clean.groupby(["group", "country"]).size().rename("n_images").reset_index()
    share = p2.get("max_domain_share")
    n_ctry = int(p2.get("min_countries") or 0)
    min_ci = int(p2.get("min_country_images") or 1)
    unseen_cap = p2.get("max_unseen_country_share")
    assigned, solver = assign_groups(sizes, targets, seed, float(p2["solver_time_limit_s"]),
                                    float(p2.get("tolerance_pp", 0.1)), domain_sizes, country_sizes,
                                    None if share is None else float(share), n_ctry, min_ci,
                                    None if unseen_cap is None else float(unseen_cap))
    clean["split"] = clean["group"].map(assigned)
    print(f"Grupos: {len(sizes)} | solver: estado={solver['solver_status']} "
          f"tolerancia usada={solver['tolerance_used_pp']} pp | desv.max imágenes={solver['max_dev_images']*100:.3f} pp, "
          f"cajas={solver['max_dev_boxes']*100:.3f} pp")

    cols = ["image_name", "split", "group", "domain", "country", "location", "development_stage",
            "n_boxes", "official_competition_split", "md5"]
    tables = {s: clean.loc[clean["split"] == s, cols].sort_values("image_name").reset_index(drop=True) for s in SETS}
    errors = verify_splits(tables, link_pairs, ("image_name", "group") + DISJOINT[key])
    errors += verify_composition(clean, None if share is None else float(share), n_ctry, min_ci,
                                 None if unseen_cap is None else float(unseen_cap))
    if errors:
        raise SystemExit("Comprobaciones anti-fuga FALLIDAS:\n  " + "\n  ".join(errors))
    print("Comprobaciones anti-fuga: OK")

    out_dir.mkdir(parents=True)
    file_names = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    for s in SETS:
        tables[s].to_csv(out_dir / file_names[s], index=False)
    clean[["image_name", "BoxesString_clean", "n_boxes_raw", "n_boxes"]].rename(
        columns={"BoxesString_clean": "BoxesString"}).to_csv(out_dir / "annotations_clean.csv", index=False)
    quarantine.to_csv(out_dir / "quarantine.csv", index=False)
    events.to_csv(out_dir / "box_changes.csv", index=False)
    pd.DataFrame(link_pairs, columns=["image_a", "image_b"]).to_csv(out_dir / "near_duplicate_links.csv", index=False)

    # distribuciones
    tot_i, tot_b = len(clean), int(clean["n_boxes"].sum())
    per_set = clean.groupby("split").agg(n_images=("image_name", "size"), n_instances=("n_boxes", "sum"),
                                         mean_boxes=("n_boxes", "mean"), n_groups=("group", "nunique")).reindex(list(SETS))
    per_set["pct_images"] = (100 * per_set["n_images"] / tot_i).round(2)
    per_set["pct_instances"] = (100 * per_set["n_instances"] / tot_b).round(2)
    per_set["target_pct"] = [100 * targets[s] for s in SETS]
    per_set.round(2).to_csv(out_dir / "distribution_sets.csv")
    pd.DataFrame({"split": list(SETS), "class": "wheat_head",
                  "n_instances": [int(per_set.at[s, "n_instances"]) for s in SETS]}).to_csv(out_dir / "distribution_classes.csv", index=False)
    clean.groupby("split")["n_boxes"].describe().reindex(list(SETS)).round(2).to_csv(out_dir / "distribution_boxes_per_image.csv")
    rows = []
    for s in SETS:
        arrs = [boxes_by_image[n] for n in clean.loc[clean["split"] == s, "image_name"] if len(boxes_by_image[n])]
        b = np.concatenate(arrs)
        rows.append({"split": s, "width_mean": (b[:, 2] - b[:, 0]).mean(), "width_median": np.median(b[:, 2] - b[:, 0]),
                     "height_mean": (b[:, 3] - b[:, 1]).mean(), "height_median": np.median(b[:, 3] - b[:, 1]),
                     "area_median": np.median((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))})
    pd.DataFrame(rows).round(2).to_csv(out_dir / "distribution_box_sizes.csv", index=False)
    clean.groupby(["split", "country", "domain"]).agg(n_images=("image_name", "size"), n_instances=("n_boxes", "sum")) \
        .reset_index().to_csv(out_dir / "distribution_metadata.csv", index=False)
    sizes.assign(split=assigned).reset_index().to_csv(out_dir / "group_assignment.csv", index=False)

    # diagnóstico de la limitación de agrupar por dominio: ubicaciones repartidas entre conjuntos
    loc_n = clean.groupby(["location", "split"]).size().unstack(fill_value=0)
    split_locs = loc_n[(loc_n > 0).sum(axis=1) > 1]
    split_locs.assign(n_total=split_locs.sum(axis=1)).reset_index().to_csv(out_dir / "locations_split_across_sets.csv", index=False)
    imgs_in_split_locs = int(split_locs.to_numpy().sum())

    comp = composition_summary(clean, min_ci)
    comp.to_csv(out_dir / "composition_per_set.csv", index=False)
    unseen = unseen_countries(clean, 1)
    unseen.to_csv(out_dir / "unseen_countries.csv", index=False)
    overlap = location_overlap(clean)
    make_figure(clean, boxes_by_image, targets, out_dir / "split_overview.png")

    # informe y configuración
    emptied = int(((clean["n_boxes_raw"] > 0) & (clean["n_boxes"] == 0)).sum())
    ev_counts = events["action"].value_counts().to_dict()
    q_counts = quarantine["reason"].value_counts().to_dict()
    if key == "location":
        limitation_text = (
            f"- **Agrupación por ubicación.** Ninguna ubicación aparece en más de un conjunto (verificado). Consecuencia: "
            f"{unseen.at[0, 'pct_of_set']} % de las imágenes de validación y {unseen.at[1, 'pct_of_set']} % de las de test "
            "proceden de países sin imágenes en train, de modo que las métricas mezclan generalización a ubicaciones nuevas "
            "y a países nuevos. Es una estimación más exigente, no comparable con la de una partición por dominio.\n"
            "- La metadata asigna a ETHZ_1 la ubicación 'Usask' (distinta de 'Saskatchewan', de Usask_1); se trató como "
            "una ubicación propia sin verificar si es un error de la metadata.")
    else:
        limitation_text = (
            f"- **Agrupación por {key}.** {len(split_locs)} ubicaciones tienen imágenes en más de un conjunto "
            f"({imgs_in_split_locs} imágenes); detalle en `locations_split_across_sets.csv`. Sesiones del mismo sitio pueden "
            "compartir campo, variedades o sensor, lo que podría hacer optimistas las métricas de test. No se ha medido ese efecto.")
    md_text = f"""# Informe de partición ({key}, semilla {seed})

Auditoría de referencia: `{run_dir.name}`. Generado: {datetime.now().isoformat(timespec='seconds')}.

## Conjuntos
{df_to_md(per_set.round(2))}

Objetivo: {targets}. Desviación máxima lograda: {solver['max_dev_images']*100:.3f} pp (imágenes), {solver['max_dev_boxes']*100:.3f} pp (espigas); tolerancia pedida {solver['tolerance_requested_pp']} pp, usada {solver['tolerance_used_pp']} pp.
Estado del solver: {solver['solver_status']} ({solver['solver_message']}).

## Limpieza
- Imágenes en la tabla limpia: {len(clean)}
- Cuarentena / exclusiones: {q_counts}
- Eventos de cajas: {ev_counts}
- Imágenes que quedaron sin cajas por la limpieza (antes tenían): {emptied}
- Imágenes sin espigas (incluye negativos originales): {int((clean['n_boxes'] == 0).sum())}

## Composición
{df_to_md(comp.drop(columns=['countries']).set_index('split'))}

Países por conjunto (imágenes): train: {'; '.join(f'{c}:{n}' for c, n in clean[clean['split'] == 'train'].groupby('country').size().sort_values(ascending=False).items())}
validation: {comp.set_index('split').at['validation', 'countries']}
test: {comp.set_index('split').at['test', 'countries']}

Imágenes de val/test en países **sin ninguna imagen en train**:
{df_to_md(unseen.set_index('split'))}

Imágenes de val/test en una ubicación que también tiene imágenes en train: {overlap}

Restricciones activas: max_domain_share={share}, min_countries={n_ctry} (con >= {min_ci} imágenes), max_unseen_country_share={unseen_cap}.

## Limitaciones conocidas
{limitation_text}
- Esta partición no es la oficial de la competición GWHD 2021 (que separa por dominio con otras proporciones): los resultados no son comparables con su clasificación.
- Las proporciones se miden sobre imágenes y espigas a la vez; la asignación es una de varias particiones válidas dentro de la tolerancia, elegida por la semilla.
"""
    (out_dir / "split_report.md").write_text(md_text, encoding="utf-8")

    outputs_hash = {p.name: sha256_of(p) for p in sorted(out_dir.glob("*.csv"))}
    inputs_hash = {f: sha256_of(root / f) for f in list(OFFICIAL_FILES.values()) + ["metadata_dataset.csv"]}
    (out_dir / "run_config.json").write_text(json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"), "seed": seed, "group_key": key, "targets": targets,
        "audit_run": run_dir.name, "phase2_config": p2, "solver": solver, "python": sys.version,
        "platform": platform.platform(), "libraries": lib_versions(),
        "input_csv_sha256": inputs_hash, "output_csv_sha256": outputs_hash,
        "counts": {"images_clean": len(clean), "quarantine_rows": len(quarantine), "box_events": len(events)},
    }, indent=2, default=str))

    print("\n" + per_set.round(2).to_string())
    print(f"\nCuarentena: {q_counts}\nEventos de cajas: {ev_counts}\nImágenes vaciadas por la limpieza: {emptied}")
    print(f"Ubicaciones repartidas entre conjuntos: {len(split_locs)} ({imgs_in_split_locs} imágenes) | "
          f"val/test en ubicaciones de train: {overlap}")
    print("\n" + comp.drop(columns=["countries"]).to_string(index=False))
    print("\nImágenes de países sin imágenes en train:\n" + unseen.to_string(index=False))
    print("Salidas en:", out_dir)


if __name__ == "__main__":
    main()