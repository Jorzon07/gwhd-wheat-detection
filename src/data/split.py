"""Asignación de grupos a train/validation/test, control de composición y comprobaciones anti-fuga."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SETS = ("train", "validation", "test")
EVAL = (1, 2)                      # índices de validation y test dentro de SETS


def merge_groups(groups: list[str], links: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Union-find: une grupos enlazados (p. ej. por imágenes casi duplicadas entre grupos distintos).

    Devuelve {grupo -> etiqueta}. Un grupo no enlazado conserva su nombre; los enlazados reciben
    el nombre de todos sus miembros ordenados y unidos por '+'.
    """
    parent = {g: g for g in groups}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in links:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, list[str]] = {}
    for g in groups:
        members.setdefault(find(g), []).append(g)
    return {g: "+".join(sorted(members[find(g)])) for g in groups}


def assign_groups(sizes: pd.DataFrame, targets: dict[str, float], seed: int,
                  time_limit_s: float = 60.0, tolerance_pp: float = 0.1,
                  domain_sizes: pd.DataFrame | None = None, country_sizes: pd.DataFrame | None = None,
                  max_domain_share: float | None = None, min_countries: int = 0, min_country_images: int = 1,
                  max_unseen_country_share: float | None = None):
    """Asigna cada grupo a un único conjunto con proporciones dentro de una tolerancia explícita.

    `sizes`: índice = grupo; columnas n_images y n_boxes. La proporción de imágenes Y la de espigas de cada
    conjunto debe quedar a menos de `tolerance_pp` puntos porcentuales del objetivo. Restricciones opcionales
    de composición, exactas y lineales, aplicadas a validation y test:
      * `max_domain_share`: ningún dominio puede superar esa fracción de las imágenes del conjunto
        (requiere `domain_sizes` con columnas group, domain, n_images).
      * `min_countries`: al menos ese número de países con >= `min_country_images` imágenes en el conjunto
        (requiere `country_sizes` con columnas group, country, n_images).
      * `max_unseen_country_share`: como máximo esa fracción de las imágenes del conjunto puede venir de países
        sin ninguna imagen en train (requiere `country_sizes`).
    Entre las asignaciones válidas se elige la que minimiza un coste aleatorio fijado por la semilla: misma
    semilla => misma partición. Si no hay solución, la tolerancia se duplica hasta 5 pp y se informa del valor
    usado. Las desviaciones devueltas se calculan sobre la asignación final.
    """
    t = np.array([targets[s] for s in SETS], dtype=float)
    if abs(t.sum() - 1.0) > 1e-9:
        raise ValueError(f"Los objetivos deben sumar 1, suman {t.sum()}")
    groups = list(sizes.index)
    gidx = {g: i for i, g in enumerate(groups)}
    G, S, K = len(groups), len(SETS), len(EVAL)
    n_img = sizes["n_images"].to_numpy(float)
    n_box = sizes["n_boxes"].to_numpy(float)

    use_share = max_domain_share is not None and domain_sizes is not None
    use_ctry = min_countries > 0 and country_sizes is not None
    use_unseen = max_unseen_country_share is not None and country_sizes is not None
    pairs = pd.DataFrame(columns=["group", "country", "n_images"])
    if use_ctry or use_unseen:
        pairs = country_sizes.groupby(["group", "country"], as_index=False)["n_images"].sum()
        pairs = pairs[pairs["n_images"] > 0].reset_index(drop=True)
    countries = sorted(pairs["country"].unique())
    cidx = {c: i for i, c in enumerate(countries)}
    C, P = len(countries), len(pairs)
    z0 = G * S                       # z[c,k]: país presente en validation/test (binaria)
    q0 = z0 + C * K                  # q[c]: país sin imágenes en train (continua, cota inferior)
    v0 = q0 + C                      # v[p,k]: par (grupo,país) en val/test con país no visto (continua)
    nv = v0 + P * K + 2
    tI, tB = nv - 2, nv - 1

    rows, lo, hi = [], [], []

    def add(r, l, h):
        rows.append(r); lo.append(l); hi.append(h)

    for g in range(G):                                   # cada grupo en exactamente un conjunto
        r = np.zeros(nv); r[g * S:(g + 1) * S] = 1
        add(r, 1.0, 1.0)
    for s in range(S):                                   # |proporción - objetivo| <= tolerancia
        for vec, tcol in ((n_img, tI), (n_box, tB)):
            share = np.zeros(nv); share[s:G * S:S] = vec / vec.sum()
            up = share.copy(); up[tcol] = -1.0; add(up, -np.inf, t[s])
            dn = share.copy(); dn[tcol] = 1.0; add(dn, t[s], np.inf)
    if use_share:                                        # n_dominio * a[g,s] <= cuota * (imágenes del conjunto s)
        for d in domain_sizes.itertuples(index=False):
            for s in EVAL:
                r = np.zeros(nv); r[s:G * S:S] = -max_domain_share * n_img
                r[gidx[d.group] * S + s] += float(d.n_images)
                add(r, -np.inf, 0.0)
    if use_ctry:                                         # z[c,s]=1 solo si el país tiene >= min_country_images en s
        for c, ci in cidx.items():
            sub = pairs[pairs["country"] == c]
            for k, s in enumerate(EVAL):
                r = np.zeros(nv)
                for grp, n in zip(sub["group"], sub["n_images"]):
                    r[gidx[grp] * S + s] += float(n)
                r[z0 + ci * K + k] = -float(min_country_images)
                add(r, 0.0, np.inf)
        for k in range(K):
            r = np.zeros(nv); r[[z0 + ci * K + k for ci in range(C)]] = 1.0
            add(r, float(min_countries), np.inf)
    if use_unseen:
        for c, ci in cidx.items():                       # q[c] >= 1 - (nº de grupos del país asignados a train)
            r = np.zeros(nv); r[q0 + ci] = 1.0
            for grp in pairs.loc[pairs["country"] == c, "group"]:
                r[gidx[grp] * S] += 1.0
            add(r, 1.0, np.inf)
        for p, row in enumerate(pairs.itertuples(index=False)):      # v >= a[g,s] + q[c] - 1
            for k, s in enumerate(EVAL):
                r = np.zeros(nv); r[v0 + p * K + k] = 1.0
                r[gidx[row.group] * S + s] = -1.0; r[q0 + cidx[row.country]] = -1.0
                add(r, -1.0, np.inf)
        for k, s in enumerate(EVAL):                     # imágenes de países no vistos <= cuota * imágenes del conjunto
            r = np.zeros(nv); r[s:G * S:S] = -max_unseen_country_share * n_img
            for p, n in enumerate(pairs["n_images"]):
                r[v0 + p * K + k] = float(n)
            add(r, -np.inf, 0.0)

    constraints = LinearConstraint(np.array(rows), lo, hi)
    integrality = np.r_[np.ones(G * S + C * K), np.zeros(C + P * K + 2)]
    cost = np.r_[np.random.default_rng(seed).random(G * S), np.zeros(nv - G * S)]

    tol, res = tolerance_pp / 100.0, None
    while tol <= 0.05 + 1e-12:
        res = milp(cost, constraints=constraints, integrality=integrality,
                   bounds=Bounds(np.zeros(nv), np.r_[np.ones(v0 + P * K), tol, tol]),
                   options={"time_limit": float(time_limit_s)})
        if res.x is not None:
            break
        tol *= 2
    if res is None or res.x is None:
        raise RuntimeError("No hay partición válida con estas restricciones, ni siquiera con 5 pp de tolerancia. "
                           "Relaja max_domain_share, min_countries, min_country_images o max_unseen_country_share.")
    a = np.rint(res.x[:G * S]).reshape(G, S).astype(int)
    if not (a.sum(axis=1) == 1).all():
        raise RuntimeError("Asignación inválida: algún grupo no está en exactamente un conjunto.")
    assigned = pd.Series([SETS[i] for i in a.argmax(axis=1)], index=sizes.index, name="split")
    if set(assigned) != set(SETS):
        raise RuntimeError(f"Algún conjunto quedó vacío: {assigned.value_counts().to_dict()}")
    sh_img = np.array([n_img[(assigned == s).to_numpy()].sum() for s in SETS]) / n_img.sum()
    sh_box = np.array([n_box[(assigned == s).to_numpy()].sum() for s in SETS]) / n_box.sum()
    info = {"solver_status": int(res.status), "solver_message": str(res.message),
            "tolerance_requested_pp": float(tolerance_pp), "tolerance_used_pp": round(tol * 100, 4),
            "max_dev_images": float(np.abs(sh_img - t).max()), "max_dev_boxes": float(np.abs(sh_box - t).max()),
            "max_domain_share": max_domain_share, "min_countries": min_countries,
            "min_country_images": min_country_images, "max_unseen_country_share": max_unseen_country_share}
    return assigned, info


def composition_summary(df: pd.DataFrame, min_country_images: int = 1) -> pd.DataFrame:
    """Composición por conjunto. `df` necesita split, domain, country, location."""
    rows = []
    for s in SETS:
        d = df[df["split"] == s]
        by_dom, by_ctry = d.groupby("domain").size(), d.groupby("country").size()
        rows.append({"split": s, "n_images": len(d), "n_domains": int(by_dom.size),
                     "n_locations": int(d["location"].nunique()),
                     "n_countries": int((by_ctry >= min_country_images).sum()),
                     "largest_domain": by_dom.idxmax(), "largest_domain_share_pct": round(100 * by_dom.max() / len(d), 1),
                     "countries": "; ".join(f"{c}:{n}" for c, n in by_ctry.sort_values(ascending=False).items())})
    return pd.DataFrame(rows)


def location_overlap(df: pd.DataFrame) -> dict[str, int]:
    """Imágenes de validation/test cuya ubicación también tiene imágenes en train."""
    train_locs = set(df.loc[df["split"] == "train", "location"])
    return {s: int(((df["split"] == s) & df["location"].isin(train_locs)).sum()) for s in ("validation", "test")}


def unseen_countries(df: pd.DataFrame, min_images: int = 1) -> pd.DataFrame:
    """Países de validation/test (con >= min_images) que no tienen ninguna imagen en train."""
    in_train = set(df.loc[df["split"] == "train", "country"])
    out = []
    for s in ("validation", "test"):
        d = df[df["split"] == s]
        by = d.groupby("country").size()
        un = by[(by >= min_images) & (~by.index.isin(in_train))]
        out.append({"split": s, "unseen_countries": "; ".join(f"{c}:{n}" for c, n in un.items()) or "-",
                    "n_images_unseen": int(un.sum()), "pct_of_set": round(100 * un.sum() / len(d), 1)})
    return pd.DataFrame(out)


def verify_composition(df: pd.DataFrame, max_domain_share: float | None, min_countries: int,
                       min_country_images: int, max_unseen_country_share: float | None = None) -> list[str]:
    """Comprueba sobre la asignación final las restricciones de composición. Lista vacía = correcto."""
    errors = []
    comp = composition_summary(df, min_country_images).set_index("split")
    for s in ("validation", "test"):
        if max_domain_share is not None and comp.at[s, "largest_domain_share_pct"] > 100 * max_domain_share + 1e-9:
            errors.append(f"{s}: el dominio {comp.at[s, 'largest_domain']} supera el {100 * max_domain_share:.0f} % "
                          f"({comp.at[s, 'largest_domain_share_pct']} %)")
        if min_countries and comp.at[s, "n_countries"] < min_countries:
            errors.append(f"{s}: solo {comp.at[s, 'n_countries']} países con >= {min_country_images} imágenes "
                          f"(mínimo {min_countries})")
    if max_unseen_country_share is not None:
        un = unseen_countries(df, 1).set_index("split")
        for s in ("validation", "test"):
            if un.at[s, "pct_of_set"] > 100 * max_unseen_country_share + 1e-9:
                errors.append(f"{s}: {un.at[s, 'pct_of_set']} % de imágenes de países sin imágenes en train "
                              f"(máximo {100 * max_unseen_country_share:.0f} %)")
    return errors


def verify_splits(tables: dict[str, pd.DataFrame], link_pairs: Iterable[tuple[str, str]] = (),
                  disjoint_cols: tuple[str, ...] = ("image_name", "group")) -> list[str]:
    """Comprueba ausencia de fugas. Devuelve la lista de errores (vacía = correcto).

    Las columnas de `disjoint_cols` (y md5, si todas las tablas la tienen) no pueden compartir valores entre
    conjuntos. `link_pairs`: pares de imágenes casi duplicadas que no pueden estar en conjuntos distintos.
    """
    errors: list[str] = []
    for name, df in tables.items():
        for col in disjoint_cols:
            if col not in df.columns:
                errors.append(f"{name}: falta la columna {col}")
        if "image_name" in df.columns and df["image_name"].duplicated().any():
            errors.append(f"{name}: image_name repetido dentro del conjunto")
    if errors:
        return errors
    use_md5 = all("md5" in df.columns for df in tables.values())
    for (na, a), (nb, b) in combinations(tables.items(), 2):
        for col in tuple(disjoint_cols) + (("md5",) if use_md5 else ()):
            common = set(a[col]) & set(b[col])
            if common:
                errors.append(f"solapamiento de {col} entre {na} y {nb}: {len(common)} (p. ej. {sorted(common)[:3]})")
    where = {n: s for s, df in tables.items() for n in df["image_name"]}
    for x, y in link_pairs:
        if x in where and y in where and where[x] != where[y]:
            errors.append(f"par casi duplicado separado: {x[:12]} ({where[x]}) / {y[:12]} ({where[y]})")
    return errors