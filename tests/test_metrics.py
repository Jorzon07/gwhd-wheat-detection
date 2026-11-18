import numpy as np
import pytest

from src.evaluation.metrics import box_iou_np, evaluate, evaluate_by_group, match_image

E = (np.zeros((0, 4)), np.zeros(0))


def grid_boxes(n, size=40, step=60, per_row=15):
    return np.array([[(i % per_row) * step, (i // per_row) * step,
                      (i % per_row) * step + size, (i // per_row) * step + size] for i in range(n)], dtype=float)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]], float)
    assert box_iou_np(a, a)[0, 0] == pytest.approx(1.0)
    assert box_iou_np(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0
    assert box_iou_np(a, np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(50 / 150)
    assert box_iou_np(np.zeros((0, 4)), a).shape == (0, 1)


def test_perfect_predictions():
    gts = {"a": grid_boxes(5), "b": grid_boxes(3)}
    preds = {k: (v, np.full(len(v), 0.9)) for k, v in gts.items()}
    r = evaluate(preds, gts)
    assert r["map50"] == pytest.approx(1.0) and r["map50_95"] == pytest.approx(1.0)
    assert (r["precision"], r["recall"], r["f1"], r["best_f1"]) == (1.0, 1.0, 1.0, 1.0)


def test_no_predictions():
    gts = {"a": grid_boxes(4)}
    r = evaluate({"a": E}, gts)
    assert r["map50"] == 0.0 and r["recall"] == 0.0 and r["precision"] == 0.0 and r["n_dets"] == 0


def test_half_recall_gives_half_ap_and_full_precision():
    gts = {"a": grid_boxes(4)}
    r = evaluate({"a": (gts["a"][:2], np.array([0.9, 0.8]))}, gts)
    assert r["recall"] == 0.5 and r["precision"] == 1.0
    assert r["map50"] == pytest.approx(51 / 101, abs=1e-6)               # interpolación COCO de 101 puntos


def test_shifted_boxes_match_only_low_iou_thresholds():
    g = grid_boxes(6, size=100, step=150, per_row=6)
    shifted = g + np.array([20, 0, 20, 0])                              # IoU = 80/120 = 0.667
    r = evaluate({"a": (shifted, np.full(6, 0.9))}, {"a": g})
    assert r["map50"] == pytest.approx(1.0)
    assert r["map50_95"] == pytest.approx(0.4, abs=1e-6)                 # coincide en IoU 0.50, 0.55, 0.60, 0.65


def test_more_than_100_boxes_per_image_are_not_truncated():
    g = grid_boxes(150)
    r = evaluate({"a": (g, np.full(150, 0.9))}, {"a": g})
    assert r["map50"] == pytest.approx(1.0) and r["recall"] == 1.0 and r["n_dets"] == 150


def test_false_positive_on_image_without_ground_truth():
    gts = {"a": grid_boxes(1), "b": np.zeros((0, 4))}
    preds = {"a": (gts["a"], np.array([0.9])), "b": (np.array([[0, 0, 30, 30.0]]), np.array([0.95]))}
    r = evaluate(preds, gts)
    assert r["precision"] == 0.5 and r["recall"] == 1.0
    assert r["map50"] == pytest.approx(0.5, abs=1e-6)


def test_best_f1_threshold_separates_true_from_false_positives():
    g = grid_boxes(2)
    far = grid_boxes(2) + 500
    boxes = np.vstack([g, far])
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    r = evaluate({"a": (boxes, scores)}, {"a": g}, score_threshold=0.1)
    assert r["precision"] == 0.5 and r["recall"] == 1.0 and r["f1"] == pytest.approx(2 / 3)
    assert r["best_f1"] == pytest.approx(1.0) and r["best_f1_threshold"] == pytest.approx(0.8)


def test_duplicate_detection_counts_as_false_positive():
    g = grid_boxes(1)
    r = evaluate({"a": (np.vstack([g, g]), np.array([0.9, 0.8]))}, {"a": g})
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_match_is_greedy_by_score_not_by_input_order():
    g = np.array([[0, 0, 10, 10]], float)
    boxes = np.array([[0, 0, 10, 9], [0, 0, 10, 10]], float)            # la segunda es mejor, pero ambas superan 0.5
    scores, tp = match_image(boxes, np.array([0.5, 0.9]), g)
    assert scores.tolist() == [0.9, 0.5] and tp.tolist() == [True, False]


def test_input_validation():
    gts = {"a": grid_boxes(1), "b": grid_boxes(1)}
    with pytest.raises(ValueError, match="Faltan predicciones"):
        evaluate({"a": E}, gts)
    with pytest.raises(ValueError, match="sin verdad terreno"):
        evaluate({"a": E, "b": E, "c": E}, gts)
    with pytest.raises(ValueError, match="puntuaciones"):
        evaluate({"a": (grid_boxes(2), np.zeros(1))}, {"a": grid_boxes(2)})


def test_max_dets_truncation_keeps_best_scores():
    g = grid_boxes(1)
    junk = grid_boxes(400) + 2000
    boxes = np.vstack([g, junk])
    scores = np.r_[0.99, np.full(400, 0.1)]
    r = evaluate({"a": (boxes, scores)}, {"a": g}, max_dets=300)
    assert r["n_dets"] == 300 and r["recall"] == 1.0


def test_evaluate_by_group():
    gts = {"a": grid_boxes(3), "b": grid_boxes(3), "c": grid_boxes(3)}
    preds = {"a": (gts["a"], np.full(3, 0.9)), "b": (gts["b"], np.full(3, 0.9)), "c": E}
    df = evaluate_by_group(preds, gts, {"a": "G1", "b": "G1", "c": "G2"}).set_index("group")
    assert df.at["G1", "map50"] == pytest.approx(1.0) and df.at["G2", "map50"] == 0.0
    assert df.at["G1", "n_images"] == 2 and df.at["G2", "n_images"] == 1


def test_no_ground_truth_at_all_gives_nan_map():
    r = evaluate({"a": E}, {"a": np.zeros((0, 4))})
    assert np.isnan(r["map50"]) and r["recall"] == 0.0


def _noisy_dataset(seed=0, n_images=12):
    rng = np.random.default_rng(seed)
    gts, preds = {}, {}
    for i in range(n_images):
        n = int(rng.integers(0, 25))
        xy = rng.uniform(0, 900, (n, 2))
        wh = rng.uniform(20, 120, (n, 2))
        g = np.hstack([xy, xy + wh])
        jitter = g + rng.normal(0, 8, g.shape)                           # detecciones buenas pero imprecisas
        keep = rng.random(n) < 0.8                                       # algunas espigas no se detectan
        fp_xy = rng.uniform(0, 900, (int(rng.integers(0, 12)), 2))
        fp = np.hstack([fp_xy, fp_xy + rng.uniform(20, 120, fp_xy.shape)])
        boxes = np.vstack([jitter[keep], fp]) if (keep.any() or len(fp)) else np.zeros((0, 4))
        scores = np.r_[rng.uniform(0.4, 1.0, int(keep.sum())), rng.uniform(0.0, 0.7, len(fp))]
        gts[f"img{i}"] = g
        preds[f"img{i}"] = (boxes, scores)
    return gts, preds


def test_map_matches_stock_cocoeval_on_noisy_data():
    """Con maxDets=100 el summarize() estándar es válido: mis valores deben coincidir con él."""
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gts, preds = _noisy_dataset()
    mine = evaluate(preds, gts, max_dets=100)
    names = sorted(gts)
    images = [{"id": i, "width": 1024, "height": 1024} for i in range(len(names))]
    anns, dets, aid = [], [], 1
    for i, n in enumerate(names):
        for x1, y1, x2, y2 in gts[n]:
            anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            aid += 1
        for (x1, y1, x2, y2), s in zip(*preds[n]):
            dets.append({"image_id": i, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "w"}]}
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    assert mine["map50_95"] == pytest.approx(s[0], abs=1e-9)
    assert mine["map50"] == pytest.approx(s[1], abs=1e-9)
    assert mine["map75"] == pytest.approx(s[2], abs=1e-9)
    assert mine["ar_max"] == pytest.approx(s[8], abs=1e-9)
    assert 0.2 < mine["map50"] < 0.95                                     # los datos son realmente ruidosos


def test_best_f1_equals_brute_force_threshold_scan():
    gts, preds = _noisy_dataset(seed=5)
    r = evaluate(preds, gts)
    all_scores = np.unique(np.concatenate([p[1] for p in preds.values()]))
    f1s = [evaluate(preds, gts, score_threshold=float(t))["f1"] for t in all_scores]
    assert r["best_f1"] == pytest.approx(max(f1s), abs=1e-12)
    at_best = evaluate(preds, gts, score_threshold=r["best_f1_threshold"])
    assert at_best["f1"] == pytest.approx(r["best_f1"], abs=1e-12)
    assert at_best["precision"] == pytest.approx(r["best_f1_precision"]) and at_best["recall"] == pytest.approx(r["best_f1_recall"])