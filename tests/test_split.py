import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ROOT
from src.data.annotations import parse_boxes_string
from src.data.make_splits import DISJOINT
from src.data.split import (SETS, assign_groups, composition_summary, location_overlap, merge_groups,
                            unseen_countries, verify_composition, verify_splits)

TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def synthetic_sizes(n=30, seed=0):
    rng = np.random.default_rng(seed)
    n_img = rng.integers(15, 300, n)
    return pd.DataFrame({"n_images": n_img, "n_boxes": (n_img * rng.uniform(10, 70, n)).astype(int)},
                        index=[f"g{i}" for i in range(n)])


def synthetic_world(seed=0):
    """24 ubicaciones, 6 países, 1-2 dominios por ubicación."""
    rng = np.random.default_rng(seed)
    rows = []
    for gi in range(24):
        for d in range(int(rng.integers(1, 3))):
            rows.append(dict(location=f"L{gi}", domain=f"L{gi}_{d}", country=f"C{gi % 6}",
                             n_images=int(rng.integers(20, 120))))
    w = pd.DataFrame(rows)
    w["n_boxes"] = (w["n_images"] * rng.uniform(20, 60, len(w))).astype(int)
    return w


def test_assign_groups_one_set_per_group_and_within_tolerance():
    sizes = synthetic_sizes()
    assigned, info = assign_groups(sizes, TARGETS, seed=1, time_limit_s=30, tolerance_pp=1.0)
    assert set(assigned) == set(SETS) and len(assigned) == len(sizes)
    tol = info["tolerance_used_pp"] / 100 + 1e-9
    assert info["max_dev_images"] <= tol and info["max_dev_boxes"] <= tol


def test_assign_groups_reproducible_with_same_seed():
    sizes = synthetic_sizes()
    a1, _ = assign_groups(sizes, TARGETS, seed=7, time_limit_s=30, tolerance_pp=1.0)
    a2, _ = assign_groups(sizes, TARGETS, seed=7, time_limit_s=30, tolerance_pp=1.0)
    assert a1.equals(a2)


def test_composition_constraints_are_enforced():
    w = synthetic_world()
    sizes = w.groupby("location").agg(n_images=("n_images", "sum"), n_boxes=("n_boxes", "sum"))
    dsz = w.rename(columns={"location": "group"})[["group", "domain", "n_images"]]
    csz = w.rename(columns={"location": "group"}).groupby(["group", "country"], as_index=False)["n_images"].sum()
    assigned, info = assign_groups(sizes, TARGETS, seed=3, time_limit_s=30, tolerance_pp=2.0, domain_sizes=dsz,
                                   country_sizes=csz, max_domain_share=0.35, min_countries=3,
                                   min_country_images=10, max_unseen_country_share=0.6)
    img = w.loc[w.index.repeat(w["n_images"])].reset_index(drop=True)
    img["split"] = img["location"].map(assigned)
    assert verify_composition(img, 0.35, 3, 10, 0.6) == []
    assert set(img["split"]) == set(SETS)
    assert location_overlap(img) == {"validation": 0, "test": 0}      # una ubicación = un conjunto


def test_verify_composition_detects_violations():
    df = pd.DataFrame({"split": ["train"] * 10 + ["validation"] * 4 + ["test"] * 4,
                       "domain": ["a"] * 10 + ["b"] * 4 + ["c"] * 4, "country": ["X"] * 10 + ["Y"] * 4 + ["Z"] * 4,
                       "location": ["l1"] * 10 + ["l2"] * 4 + ["l3"] * 4})
    errs = verify_composition(df, 0.5, 2, 1, 0.5)
    assert any("supera" in e for e in errs) and any("países" in e for e in errs) and any("sin imágenes en train" in e for e in errs)
    assert verify_composition(df, None, 0, 1, None) == []


def test_unseen_countries_and_composition_summary():
    df = pd.DataFrame({"split": ["train", "train", "validation", "validation", "test", "test"],
                       "domain": list("abcdef"), "country": ["X", "Y", "X", "Z", "Y", "Y"],
                       "location": list("pqrstu")})
    un = unseen_countries(df).set_index("split")
    assert un.at["validation", "n_images_unseen"] == 1 and un.at["test", "n_images_unseen"] == 0
    assert composition_summary(df).set_index("split").at["test", "n_countries"] == 1


def test_merge_groups():
    m = merge_groups(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("X", "A")])
    assert m == {"A": "A+B+C", "B": "A+B+C", "C": "A+B+C", "D": "D"}


def _tables():
    mk = lambda names, grp: pd.DataFrame({"image_name": names, "group": grp, "md5": [f"m{n}" for n in names],
                                          "location": grp})
    return {"train": mk(["a", "b"], ["g1", "g1"]), "validation": mk(["c"], ["g2"]), "test": mk(["d"], ["g3"])}


def test_verify_clean_ok():
    assert verify_splits(_tables(), [("a", "b")]) == []


def test_verify_detects_image_group_md5_overlap_and_split_pairs():
    t = _tables(); t["test"].loc[0, "image_name"] = "a"
    assert any("image_name" in e for e in verify_splits(t))
    t = _tables(); t["test"].loc[0, "group"] = "g1"
    assert any("group" in e for e in verify_splits(t))
    t = _tables(); t["test"].loc[0, "md5"] = "ma"
    assert any("md5" in e for e in verify_splits(t))
    assert any("par casi duplicado" in e for e in verify_splits(_tables(), [("a", "c")]))


def test_verify_detects_extra_disjoint_columns():
    t = _tables(); t["test"].loc[0, "location"] = "g1"
    assert verify_splits(t) == []                                           # por defecto no se mira 'location'
    assert any("location" in e for e in verify_splits(t, (), ("image_name", "group", "location")))


def _real_split_dirs():
    return [p.parent for p in (ROOT / "data" / "splits").glob("*/train.csv")]


@pytest.mark.skipif(not _real_split_dirs(), reason="Aún no se ha generado ninguna partición real")
@pytest.mark.parametrize("d", _real_split_dirs(), ids=lambda p: p.name)
def test_real_splits_have_no_leakage_and_valid_boxes(d: Path):
    key = d.name.rsplit("_seed", 1)[0]
    tables = {"train": pd.read_csv(d / "train.csv"), "validation": pd.read_csv(d / "validation.csv"),
              "test": pd.read_csv(d / "test.csv")}
    links = pd.read_csv(d / "near_duplicate_links.csv")
    assert verify_splits(tables, list(zip(links["image_a"], links["image_b"])),
                         ("image_name", "group") + DISJOINT[key]) == []
    cfg = json.loads((d / "run_config.json").read_text())["solver"]
    allrows = pd.concat([df.assign(split=s) for s, df in tables.items()], ignore_index=True)
    assert verify_composition(allrows, cfg.get("max_domain_share"), int(cfg.get("min_countries") or 0),
                              int(cfg.get("min_country_images") or 1), cfg.get("max_unseen_country_share")) == []
    ann = pd.read_csv(d / "annotations_clean.csv")
    assert set(ann["image_name"]) == set(allrows["image_name"])
    n_by_name = dict(zip(ann["image_name"], ann["n_boxes"]))
    for df in tables.values():
        assert all(n_by_name[n] == k for n, k in zip(df["image_name"], df["n_boxes"]))
    for s in ann["BoxesString"]:
        b, errs = parse_boxes_string(s)
        assert errs == []
        if len(b):
            assert b.min() >= 0 and b[:, [0, 2]].max() <= 1024 and b[:, [1, 3]].max() <= 1024
            assert ((b[:, 2] - b[:, 0]) >= 2).all() and ((b[:, 3] - b[:, 1]) >= 2).all()