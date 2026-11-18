import random

import numpy as np
import pytest

from src.gan.layout import apply_array, compose_canvas, crop_boxes, render_layout, transform_boxes


def _int_boxes(n=6, size=64, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.integers(2, size - 22, (n, 2))
    wh = rng.integers(6, 18, (n, 2))
    return np.hstack([xy, xy + wh]).astype(np.float32)


def test_render_values_and_shape():
    m = render_layout(np.array([[2, 3, 6, 8]]), 16)
    assert m.shape == (2, 16, 16) and set(np.unique(m)) == {-1.0, 1.0}
    assert (m[0] == 1).sum() == (6 - 2) * (8 - 3)                      # interior
    assert m[1, 3, 2:6].min() == 1 and m[1, 7, 2:6].min() == 1         # contorno superior e inferior
    assert render_layout(np.zeros((0, 4)), 8).max() == -1.0


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_box_transform_matches_array_transform(flip, k):
    b = _int_boxes()
    a = apply_array(render_layout(b, 64), flip, k)
    c = render_layout(transform_boxes(b, 64, flip, k), 64)
    assert np.array_equal(a, c)


def test_crop_boxes_visibility_and_coordinates():
    boxes = np.array([[10, 10, 30, 30], [90, 10, 110, 30], [0, 0, 4, 4]], dtype=np.float32)
    out = crop_boxes(boxes, 0, 0, 100, 100, min_vis=0.5)
    assert len(out) == 3 and np.allclose(out[1], [90, 10, 100, 30])    # recortada, 50 % visible: se conserva
    out = crop_boxes(boxes, 20, 0, 100, 100, min_vis=0.6)
    assert len(out) == 1                                               # x=10..30 queda con 50 % (fuera); 0..4 sale de la ventana
    assert np.allclose(out[0], [70, 10, 90, 30])
    assert crop_boxes(boxes, 500, 500, 100, 100).shape == (0, 4)


def test_compose_canvas_inside_and_reproducible():
    pool = [_int_boxes(30, 1024, s) * 1.0 for s in range(5)]
    a = compose_canvas(pool, random.Random(1), 256, 128, 1024)
    b = compose_canvas(pool, random.Random(1), 256, 128, 1024)
    assert np.array_equal(a, b)
    assert len(a) > 0 and a.min() >= 0 and a.max() <= 256
    assert (a[:, 2] > a[:, 0]).all() and (a[:, 3] > a[:, 1]).all()


def test_models_forward_shapes_and_modes():
    torch = pytest.importorskip("torch")
    from src.gan.models import PatchDiscriminator, UNetGenerator, generation_mode, init_weights
    G = UNetGenerator(2, 3, ngf=4, num_downs=5)
    D = PatchDiscriminator(5, ndf=4)
    G.apply(init_weights); D.apply(init_weights)
    x = torch.randn(2, 2, 64, 64)
    y = G(x)
    assert y.shape == (2, 3, 64, 64) and float(y.abs().max()) <= 1.0
    assert D(x, y).shape[0] == 2
    # totalmente convolucional: otro tamaño múltiplo de 2**5
    assert G(torch.randn(1, 2, 128, 96)).shape == (1, 3, 128, 96)
    generation_mode(G)
    assert not G.downs[1].net[-1].training and any(m.training for m in G.modules() if isinstance(m, torch.nn.Dropout))
    # el gradiente llega hasta la entrada del generador (skips bien conectados)
    G.train()
    xx = torch.randn(2, 2, 64, 64, requires_grad=True)
    G(xx).sum().backward()
    assert xx.grad is not None and float(xx.grad.abs().sum()) > 0
