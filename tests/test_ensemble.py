import numpy as np
import pytest

pytest.importorskip("ensemble_boxes")
from src.evaluation.ensemble import fuse_all, fuse_image


def _gt(n=20, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 900, size=(n, 2))
    wh = rng.uniform(30, 90, size=(n, 2))
    return np.hstack([xy, xy + wh])


def test_identical_members_return_same_boxes_and_scores():
    g = _gt()
    s = np.linspace(0.9, 0.5, len(g))
    b, sc = fuse_image(([g, g], [s, s], 0.55, 0.0, None, "avg"))
    assert len(b) == len(g)
    order = np.argsort(-s)
    assert np.allclose(b, g[order], atol=1e-3) and np.allclose(sc, s[order], atol=1e-6)


def test_box_seen_by_one_model_is_penalised_with_avg():
    g = _gt(5)
    s = np.full(5, 0.8)
    b, sc = fuse_image(([g, g[:0]], [s, s[:0]], 0.55, 0.0, None, "avg"))
    assert len(b) == 5 and np.allclose(sc, 0.4)             # 0.8 * 1/2: avg penaliza al modelo ausente


def test_fusion_improves_localisation_of_noisy_members():
    g = _gt(30, 1)
    rng = np.random.default_rng(3)
    a = g + rng.normal(0, 2, g.shape)
    b2 = g + rng.normal(0, 2, g.shape)
    s = np.full(len(g), 0.9)
    fb, _ = fuse_image(([a, b2], [s, s], 0.55, 0.0, None, "avg"))
    assert len(fb) == len(g)                                 # todas las cajas se agrupan
    err = lambda x: np.abs(x[np.argsort(x[:, 0] * 1e4 + x[:, 1])] - g[np.argsort(g[:, 0] * 1e4 + g[:, 1])]).mean()   # noqa: E731
    assert err(fb) < err(a)


def test_empty_inputs_and_parallel_equals_serial():
    b, s = fuse_image(([np.zeros((0, 4)), np.zeros((0, 4))], [np.zeros(0), np.zeros(0)], 0.55, 0.0, None, "avg"))
    assert len(b) == 0 and len(s) == 0
    g = _gt(10)
    preds = [{f"im{i}": (g + i, np.full(10, 0.7)) for i in range(6)}, {f"im{i}": (g + 2 * i, np.full(10, 0.6)) for i in range(6)}]
    names = sorted(preds[0])
    ser = fuse_all(preds, names, 0.55, 0.05, workers=1)
    par = fuse_all(preds, names, 0.55, 0.05, workers=2)
    for n in names:
        assert np.allclose(ser[n][0], par[n][0]) and np.allclose(ser[n][1], par[n][1])
