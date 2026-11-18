import contextlib, io
import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.evaluation.bootstrap import (IOU_THRS, bootstrap, match_all, metrics_from_matches,
                                      paired_differences)


def synth(seed=0, n_images=24, max_gt=40, noise=8.0, hit=0.8, score_lo=0.4):
    rng = np.random.default_rng(seed); gts, preds = [], []
    for _ in range(n_images):
        n = int(rng.integers(0, max_gt)); xy = rng.uniform(0, 900, (n, 2))
        g = np.hstack([xy, xy + rng.uniform(20, 120, (n, 2))]); keep = rng.random(n) < hit
        k = int(keep.sum()); b = g[keep] + rng.normal(0, noise, (k, 4))
        # cajas siempre válidas (COCO ignora las de área negativa)
        b[:, 2] = np.maximum(b[:, 2], b[:, 0] + 2); b[:, 3] = np.maximum(b[:, 3], b[:, 1] + 2)
        m = int(rng.integers(0, 15)); fxy = rng.uniform(0, 900, (m, 2))
        fp = np.hstack([fxy, fxy + rng.uniform(20, 120, (m, 2))])
        gts.append(g)
        preds.append((np.vstack([b, fp]), np.r_[rng.uniform(score_lo, 1.0, k), rng.uniform(0, 0.7, m)]))
    return gts, preds


def cocoeval(gts, preds):
    images = [{"id": i, "width": 1024, "height": 1024} for i in range(len(gts))]
    anns, dets, aid = [], [], 1
    for i, (g, (b, s)) in enumerate(zip(gts, preds)):
        for x1, y1, x2, y2 in g:
            anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0}); aid += 1
        for (x1, y1, x2, y2), sc in zip(b, s):
            dets.append({"image_id": i, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "w"}]}
        gt.createIndex(); ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.maxDets = [1, 10, 300]; ev.evaluate(); ev.accumulate()
    p = ev.eval["precision"]  # T,R,K,A,M
    return p[0, :, 0, 0, 2].mean(), np.mean([p[t, :, 0, 0, 2].mean() for t in range(len(IOU_THRS))])


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_map_reproduces_cocoeval_exactly(seed):
    gts, preds = synth(seed)
    r = metrics_from_matches(match_all(gts, preds))
    a50, a = cocoeval(gts, preds)
    assert r["map50"] == pytest.approx(a50, abs=1e-9)
    assert r["map50_95"] == pytest.approx(a, abs=1e-9)


def test_maxdets_truncation_matches_cocoeval():
    gts, preds = synth(1, max_gt=15)
    big = []
    for b, s in preds:
        b2 = np.vstack([b] * 30); s2 = np.concatenate([s] * 30) + np.random.default_rng(0).uniform(0, 1e-6, len(s) * 30)
        big.append((b2, s2))
    r = metrics_from_matches(match_all(gts, big, max_dets=300))
    assert r["map50"] == pytest.approx(cocoeval(gts, big)[0], abs=1e-9)


def test_perfect_empty_and_no_gt():
    gts = [np.array([[0, 0, 50, 50.], [100, 100, 160, 170]])] * 3
    perfect = [(g, np.array([0.9, 0.8])) for g in gts]
    r = metrics_from_matches(match_all(gts, perfect)); assert r["map50_95"] == pytest.approx(1.0)
    r = metrics_from_matches(match_all(gts, [(np.zeros((0, 4)), np.zeros(0))] * 3)); assert r["map50"] == 0.0
    r = metrics_from_matches(match_all([np.zeros((0, 4))] * 2, [(np.zeros((0, 4)), np.zeros(0))] * 2))
    assert np.isnan(r["map50"])


def test_determinism_and_identical_models_zero_diff():
    gts, preds = synth(2); m = match_all(gts, preds)
    d1 = bootstrap({"a": m, "b": m}, n_boot=30, seed=5); d2 = bootstrap({"a": m, "b": m}, n_boot=30, seed=5)
    assert d1.equals(d2)
    pdiff = paired_differences(d1, "a", "b"); assert (pdiff.mean_diff.abs() < 1e-12).all()
    assert not pdiff.excluye_cero.any()


def test_point_inside_ci_and_better_model_excludes_zero():
    gts, good = synth(4, n_images=60, noise=3, hit=0.95)
    _, bad = synth(4, n_images=60, noise=25, hit=0.5)
    mg, mb = match_all(gts, good), match_all(gts, bad)
    d = bootstrap({"good": mg, "bad": mb}, n_boot=200, seed=1)
    g = d[d.model == "good"].map50
    assert g.quantile(0.025) <= metrics_from_matches(mg)["map50"] <= g.quantile(0.975)
    row = paired_differences(d, "good", "bad").query("metric=='map50'").iloc[0]
    assert row.excluye_cero and row.lo > 0


def test_domain_bootstrap_is_wider_when_domains_differ():
    # dominios con dificultad distinta: remuestrear dominios debe dar IC más ancho
    gts, preds, dom = [], [], []
    for k, (noise, hit) in enumerate([(3, 0.95), (12, 0.7), (25, 0.45), (6, 0.85)]):
        g, p = synth(10 + k, n_images=30, noise=noise, hit=hit); gts += g; preds += p; dom += [k] * 30
    m = match_all(gts, preds); dom = np.array(dom)
    wi = bootstrap({"m": m}, n_boot=300, seed=0, level="image").map50
    wd = bootstrap({"m": m}, n_boot=300, seed=0, level="domain", domains=dom).map50
    assert wd.std() > wi.std()


def test_input_validation():
    gts, preds = synth(0, n_images=5); m = match_all(gts, preds)
    with pytest.raises(ValueError): bootstrap({"m": m}, level="domain", n_boot=2)
    with pytest.raises(ValueError): bootstrap({"m": m}, level="x", n_boot=2)
    with pytest.raises(ValueError): match_all(gts, preds[:-1])
    m2 = match_all(*[x[:3] for x in (gts, preds)])
    with pytest.raises(ValueError): bootstrap({"a": m, "b": m2}, n_boot=2)
