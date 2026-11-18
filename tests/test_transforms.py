import numpy as np
import pytest

from src.data.transforms import (AugConfig, adjust_brightness_contrast, apply_eval_transforms,
                                 apply_train_transforms, clip_filter, hflip, random_scale_crop,
                                 resize_image_and_boxes, rot90, vflip)


def mask_of(boxes, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes.astype(int):
        m[y1:y2, x1:x2] = 1
    return m


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


BOXES = np.array([[3, 5, 20, 12], [30, 2, 41, 30]], dtype=np.float32)     # imagen no cuadrada 40 x 60 (alto x ancho)
H, W = 40, 60


@pytest.mark.parametrize("fn,name", [(hflip, "hflip"), (vflip, "vflip")])
def test_flips_match_mask_method(fn, name):
    img = np.zeros((H, W, 3), np.uint8)
    _, out = fn(img, BOXES)
    for box_in, box_out in zip(BOXES, out):
        m = mask_of(box_in[None], H, W)
        m2 = m[:, ::-1] if name == "hflip" else m[::-1]
        assert np.array_equal(bbox_of(m2), box_out)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_rot90_matches_mask_method_on_non_square_image(k):
    img = np.zeros((H, W, 3), np.uint8)
    new_img, out = rot90(img, BOXES, k)
    assert new_img.shape[:2] == ((W, H) if k % 2 else (H, W))
    for box_in, box_out in zip(BOXES, out):
        m2 = np.rot90(mask_of(box_in[None], H, W), k)
        assert np.array_equal(bbox_of(m2), box_out), (k, box_in, box_out)


def test_resize_scales_boxes_per_axis():
    img = np.zeros((40, 60, 3), np.uint8)
    out_img, b = resize_image_and_boxes(img, BOXES, 30)
    assert out_img.shape == (30, 30, 3)
    assert np.allclose(b[0], [3 * 0.5, 5 * 0.75, 20 * 0.5, 12 * 0.75])


def test_resize_noop_when_same_size():
    img = np.zeros((32, 32, 3), np.uint8)
    out_img, b = resize_image_and_boxes(img, np.array([[1, 2, 3, 4]], np.float32), 32)
    assert out_img is img and b.tolist() == [[1, 2, 3, 4]]


def test_clip_filter_rules():
    boxes = np.array([[-5, 0, 10, 10], [50, 50, 51, 80], [10, 10, 30, 30], [0, 0, 100, 100]], np.float32)
    area = np.array([15 * 10, 30, 400, 10000], np.float32)
    out, keep = clip_filter(boxes, 60, 60, 2.0, area, 0.5)
    # caja 0: recortada a 10x10=100 >= 0.5*150 -> se conserva; caja 1: queda 1 px de ancho -> fuera;
    # caja 2: intacta; caja 3: recortada a 60x60=3600 < 0.5*10000 -> fuera por poca visibilidad
    assert keep.tolist() == [True, False, True, False]
    assert out.tolist() == [[0, 0, 10, 10], [10, 10, 30, 30]]


def test_scale_crop_boxes_inside_and_visible_enough():
    rng = np.random.default_rng(0)
    cfg = AugConfig()
    img = np.zeros((200, 200, 3), np.uint8)
    boxes = np.array([[10, 10, 60, 60], [120, 100, 190, 180], [90, 90, 110, 110]], np.float32)
    for _ in range(50):
        out_img, out = random_scale_crop(img, boxes, rng, cfg)
        assert out_img.shape[0] == out_img.shape[1] <= 200
        s = out_img.shape[0]
        assert (out >= 0).all() and (out[:, [0, 2]] <= s).all() and (out[:, [1, 3]] <= s).all()
        assert ((out[:, 2] - out[:, 0]) >= cfg.min_box_side).all()


def test_scale_crop_never_empties_an_image_that_had_boxes():
    cfg = AugConfig(crop_min_frac=0.3)
    img = np.zeros((200, 200, 3), np.uint8)
    boxes = np.array([[2, 2, 20, 20]], np.float32)                  # un solo objeto en una esquina
    for seed in range(60):
        _, out = random_scale_crop(img, boxes, np.random.default_rng(seed), cfg)
        assert len(out) >= 1


def test_empty_boxes_pass_through_everything():
    img = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    empty = np.zeros((0, 4), np.float32)
    for seed in range(20):
        o_img, o_b = apply_train_transforms(img, empty, np.random.default_rng(seed), AugConfig(), 48)
        assert o_img.shape == (48, 48, 3) and o_b.shape == (0, 4)
    e_img, e_b = apply_eval_transforms(img, empty, 32)
    assert e_img.shape == (32, 32, 3) and e_b.shape == (0, 4)


def test_brightness_contrast():
    img = np.full((8, 8, 3), 100, np.uint8)
    assert np.array_equal(adjust_brightness_contrast(img, 1.0, 1.0), img)
    assert adjust_brightness_contrast(img, 1.2, 1.0).mean() > 100
    assert adjust_brightness_contrast(img, 5.0, 1.0).max() == 255            # satura sin desbordar


def test_train_transforms_deterministic_given_rng_and_boxes_valid():
    img = np.random.default_rng(3).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 40], [60, 70, 120, 110], [100, 5, 127, 30]], np.float32)
    a_img, a_b = apply_train_transforms(img, boxes, np.random.default_rng([42, 0, 7]), AugConfig(), 96)
    b_img, b_b = apply_train_transforms(img, boxes, np.random.default_rng([42, 0, 7]), AugConfig(), 96)
    assert np.array_equal(a_img, b_img) and np.array_equal(a_b, b_b)
    for seed in range(100):
        o_img, o_b = apply_train_transforms(img, boxes, np.random.default_rng(seed), AugConfig(), 96)
        assert o_img.shape == (96, 96, 3) and o_img.dtype == np.uint8
        assert np.isfinite(o_b).all() and (o_b >= 0).all() and (o_b <= 96).all()
        assert ((o_b[:, 2] - o_b[:, 0]) >= 2).all() and ((o_b[:, 3] - o_b[:, 1]) >= 2).all()


def test_geometric_transforms_preserve_box_count_without_crop():
    cfg = AugConfig(scale_crop_p=0.0)
    img = np.zeros((128, 128, 3), np.uint8)
    boxes = np.array([[10, 10, 50, 40], [60, 70, 120, 110]], np.float32)
    for seed in range(50):
        _, o_b = apply_train_transforms(img, boxes, np.random.default_rng(seed), cfg, 128)
        assert len(o_b) == 2


def test_resize_returns_writable_array():
    """np.asarray(PIL) es de solo lectura: provocaba avisos de PyTorch en cada worker."""
    out_img, _ = resize_image_and_boxes(np.zeros((40, 60, 3), np.uint8), BOXES, 30)
    assert out_img.flags.writeable