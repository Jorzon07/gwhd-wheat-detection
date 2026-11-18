"""Descarga con reanudación, verificación md5, extracción idempotente y resumen de estructura."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import load_config, resolve

CHUNK = 1024 * 1024
MIN_FREE_GB = 25


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f, tqdm(total=path.stat().st_size, unit="B", unit_scale=True, desc="md5") as bar:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    pos = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:
            print("El archivo parcial ya parece completo; se verificará su md5.")
        else:
            r.raise_for_status()
            resumed = pos > 0 and r.status_code == 206
            if pos and not resumed:
                pos = 0                      # el servidor ignoró Range: empezar de nuevo
            total = int(r.headers.get("Content-Length", 0)) + pos
            with open(part, "ab" if resumed else "wb") as f, tqdm(
                total=total or None, initial=pos, unit="B", unit_scale=True, desc="descarga"
            ) as bar:
                for chunk in r.iter_content(CHUNK):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    part.replace(dest)


def summarize_dir(root: Path) -> None:
    if not root.exists():
        print(f"{root} no existe todavía.")
        return
    ext_count: Counter = Counter()
    csvs: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            ext_count[p.suffix.lower()] += 1
            if p.suffix.lower() == ".csv":
                csvs.append(p)
    print(f"\nEstructura de {root}:")
    print("  Subcarpetas de primer nivel:", sorted(d.name for d in root.iterdir() if d.is_dir()))
    print("  Archivos por extensión     :", dict(ext_count))
    for c in sorted(csvs):
        print(f"  CSV: {c.relative_to(root)} ({c.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--summary-only", action="store_true", help="solo mostrar la estructura ya extraída")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dataset"]
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    zip_path = resolve(cfg["paths"]["zip_path"])
    extract_dir = resolve(cfg["paths"]["extract_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        summarize_dir(extract_dir)
        return

    free_gb = shutil.disk_usage(raw_dir).free / 1e9
    if free_gb < MIN_FREE_GB and not zip_path.exists():
        raise SystemExit(f"Espacio libre insuficiente: {free_gb:.1f} GB (se piden >= {MIN_FREE_GB} GB).")

    if zip_path.exists():
        print(f"Ya existe {zip_path.name}; no se descarga de nuevo.")
    else:
        print("Descargando desde Zenodo...")
        download(ds["zenodo_url"], zip_path)

    md5 = md5_of_file(zip_path)
    if md5 != ds["md5"]:
        raise SystemExit(f"md5 NO coincide: obtenido {md5}, esperado {ds['md5']}. "
                         f"Borra {zip_path} y vuelve a ejecutar.")
    print("md5 verificado.")

    manifest_path = raw_dir / "download_manifest.json"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    first_seen = now
    if manifest_path.exists():
        first_seen = json.loads(manifest_path.read_text()).get("first_seen_utc", now)
    manifest_path.write_text(json.dumps({
        "dataset": ds["name"], "record": ds["zenodo_record"], "url": ds["zenodo_url"],
        "license": ds["license"], "md5_expected": ds["md5"], "md5_computed": md5,
        "size_bytes": zip_path.stat().st_size, "first_seen_utc": first_seen, "last_verified_utc": now,
    }, indent=2))

    marker = extract_dir / ".extracted_ok"
    if marker.exists() and marker.read_text().strip() == md5:
        print("Ya extraído; no se extrae de nuevo.")
    else:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            for m in tqdm(members, desc="extrayendo"):
                zf.extract(m, extract_dir)
        marker.write_text(md5)

    summarize_dir(extract_dir)


if __name__ == "__main__":
    main()