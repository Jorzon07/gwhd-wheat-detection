import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.evaluation.analysis import (match_gt_flags, metrics_by_density, per_image_table, recall_by_gt_size, recall_by_size_and_group)
from src.evaluation.analyze_run import analyze
from src.training.utils import save_predictions


def box(x, y, w=40, h=40):
    return [x, y, x + w, y + h]


def test_match_gt_flags_threshold_and_greedy():
    gt = np.array([box(0, 0), box(100, 0), box(200, 0)], float)
    dets = np.array([box(1, 1), box(101, 0), box(500, 500)], float)
    scores = np.array([0.9, 0.3, 0.8])
    assert match_gt_flags(dets, scores, gt, score_thr=0.5).tolist() == [True, False, False]    # 2ª bajo el umbral
    assert match_gt_flags(dets, scores, gt, score_thr=0.2).tolist() == [True, True, False]
    assert match_gt_flags(np.zeros((0, 4)), np.zeros(0), gt).tolist() == [False] * 3
    assert match_gt_flags(dets, scores, np.zeros((0, 4))).shape == (0,)


def test_per_image_table_counts():
    gts = {"a": np.array([box(0, 0), box(100, 0)], float), "b": np.zeros((0, 4))}
    preds = {"a": (np.array([box(0, 0), box(300, 300)], float), np.array([0.9, 0.8])),
             "b": (np.array([box(0, 0)], float), np.array([0.9]))}
    t = per_image_table(preds, gts, 0.5).set_index("image_name")
    assert (t.at["a", "tp"], t.at["a", "fp"], t.at["a", "fn"]) == (1, 1, 1)
    assert t.at["a", "precision"] == 0.5 and t.at["a", "recall"] == 0.5 and t.at["a", "f1"] == 0.5
    assert (t.at["b", "fp"], t.at["b", "n_gt"]) == (1, 0) and t.at["b", "precision"] == 0.0 and t.at["b", "recall"] == 1.0


def test_recall_by_gt_size_bins():
    small, big = box(0, 0, 10, 10), box(100, 0, 90, 90)                 # lados equivalentes 10 y 90
    gts = {"a": np.array([small, big], float)}
    preds = {"a": (np.array([big], float), np.array([0.9]))}            # solo se detecta la grande
    df = recall_by_gt_size(preds, gts, 0.5).set_index("size_px")
    assert df.at["0-20", "n_gt"] == 1 and df.at["0-20", "recall"] == 0.0
    assert df.at["60-100", "n_gt"] == 1 and df.at["60-100", "recall"] == 1.0
    assert np.isnan(df.at["20-40", "recall"]) and df.at["20-40", "n_gt"] == 0


def test_recall_by_size_and_group_separates_domains():
    gts = {"a": np.array([box(0, 0, 10, 10), box(100, 0, 90, 90)], float), "b": np.array([box(0, 0, 10, 10)], float)}
    preds = {"a": (np.array([box(100, 0, 90, 90)], float), np.array([0.9])),       # G1: solo detecta la grande
             "b": (np.array([box(0, 0, 10, 10)], float), np.array([0.9]))}          # G2: detecta la pequeña
    r = recall_by_size_and_group(preds, gts, {"a": "G1", "b": "G2"}, 0.5).set_index(["group", "size_px"])
    assert r.loc[("G1", "0-20"), "recall"] == 0.0 and r.loc[("G2", "0-20"), "recall"] == 1.0
    assert r.loc[("G1", "60-100"), "recall"] == 1.0 and r.loc[("G1", "60-100"), "n_gt"] == 1
    assert r.loc[("G2", "60-100"), "n_gt"] == 0


def test_metrics_by_density_rows():
    gts = {"e": np.zeros((0, 4)), "a": np.array([box(0, 0)], float),
           "b": np.array([box(i * 50, 0, 40, 40) for i in range(30)], float)}
    preds = {"e": (np.array([box(0, 0), box(60, 60)], float), np.array([0.9, 0.8])),
             "a": (np.array([box(0, 0)], float), np.array([0.9])),
             "b": (gts["b"][:15], np.full(15, 0.9))}
    df = metrics_by_density(preds, gts, 0.5).set_index("gt_per_image")
    assert df.at["0 (sin espigas)", "fp_per_image"] == 2.0 and df.at["0 (sin espigas)", "n_images"] == 1
    assert df.at["1-20", "n_images"] == 1 and df.at["1-20", "recall"] == 1.0
    assert df.at["21-40", "n_images"] == 1 and df.at["21-40", "recall"] == 0.5 and df.at["21-40", "fp_per_image"] == 0.0
    assert df.at["101+", "n_images"] == 0 and np.isnan(df.at["101+", "f1"])


@pytest.fixture()
def synthetic_run(tmp_path):
    rng = np.random.default_rng(0)
    images, split = tmp_path / "images", tmp_path / "split"
    images.mkdir(); split.mkdir()
    names = [f"v{i}.png" for i in range(12)]
    gts, rows = {}, []
    for i, n in enumerate(names):
        k = 0 if i == 0 else int(rng.integers(3, 12))
        xy = rng.uniform(0, 900, (k, 2)); g = np.hstack([xy, xy + rng.uniform(25, 90, (k, 2))])
        gts[n] = g
        Image.fromarray(rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)).save(images / n)
        rows.append((n, f"d{i % 3}", f"c{i % 2}"))
    pd.DataFrame(rows, columns=["image_name", "domain", "country"]).to_csv(split / "validation.csv", index=False)
    pd.DataFrame({"image_name": names, "BoxesString": ["no_box" if len(gts[n]) == 0 else
                  ";".join(" ".join(f"{v:g}" for v in b) for b in gts[n]) for n in names]}).to_csv(
        split / "annotations_clean.csv", index=False)
    preds = {}
    for n in names:
        g = gts[n]
        keep = rng.random(len(g)) < 0.8
        fp_xy = rng.uniform(0, 900, (2, 2)); fp = np.hstack([fp_xy, fp_xy + 40])
        preds[n] = (np.vstack([g[keep] + rng.normal(0, 3, (int(keep.sum()), 4)), fp]),
                    np.r_[rng.uniform(0.5, 1, int(keep.sum())), rng.uniform(0, 0.4, 2)])
    run = tmp_path / "run"
    run.mkdir()
    save_predictions(run / "val_predictions_best.npz", preds)
    (run / "config.json").write_text(json.dumps({"split_dir": str(split), "phase3": {"images_dir": str(images)}}))
    pd.DataFrame({"epoch": [0, 1, 2], "train_loss": [3, 2, 1.5], "train_loss_classifier": [1, .8, .6],
                  "val_map50": [np.nan, 0.5, 0.7], "val_map50_95": [np.nan, 0.2, 0.3], "val_precision": [np.nan, .6, .7],
                  "val_recall": [np.nan, .5, .6], "val_f1": [np.nan, .55, .65], "val_best_f1": [np.nan, .6, .7],
                  "lr_end": [.005, .003, .001], "train_time_s": [600, 610, 605]}).to_csv(run / "metrics.csv", index=False)
    return run


def test_analyze_end_to_end_writes_all_outputs(synthetic_run):
    r = analyze(synthetic_run, n_examples=4)
    out = synthetic_run / "analysis"
    for f in ["val_metrics.json", "val_by_domain.csv", "val_by_country.csv", "val_recall_by_size.csv",
              "val_by_density.csv", "val_per_image.csv", "curves.png", "by_group.png", "diagnostics.png",
              "examples_best.png", "examples_worst.png", "analysis.md", "val_recall_by_size_and_domain.csv"]:
        assert (out / f).exists() and (out / f).stat().st_size > 0, f
    assert 0.4 <= r["threshold"] <= 1.0                                   # umbral de mejor F1 entre las puntuaciones
    by_domain = pd.read_csv(out / "val_by_domain.csv")
    assert set(by_domain["group"]) == {"d0", "d1", "d2"} and by_domain["n_images"].sum() == 12
    assert "mAP@50" in (out / "analysis.md").read_text(encoding="utf-8")
    meta = json.loads((out / "val_metrics.json").read_text())
    assert meta["threshold_source"] == "mejor F1 en validación"
    assert meta["macro_map50_domains"] == pytest.approx(by_domain["map50"].mean())
    assert "Recall por tamaño de espiga y dominio" in (out / "analysis.md").read_text(encoding="utf-8")


def test_analyze_respects_explicit_threshold_and_is_repeatable(synthetic_run):
    a = analyze(synthetic_run, threshold=0.45)
    b = analyze(synthetic_run, threshold=0.45)
    assert a["threshold"] == 0.45 and a["metrics"]["f1"] == b["metrics"]["f1"]
    assert json.loads((synthetic_run / "analysis" / "val_metrics.json").read_text())["threshold_source"] == "indicado por el usuario"

def test_curves_figure_handles_retinanet_loss_columns(tmp_path):
    from src.evaluation.analysis import curves_figure
    df = pd.DataFrame({"epoch": [0, 1], "train_loss": [1.0, 0.8], "train_classification": [0.6, 0.5],
                       "train_bbox_regression": [0.4, 0.3], "val_map50": [0.5, 0.6], "val_map50_95": [0.2, 0.3],
                       "lr_end": [0.002, 0.001], "train_time_s": [500, 505]})
    out = tmp_path / "c.png"
    curves_figure(df, out)
    assert out.exists() and out.stat().st_size > 1000
