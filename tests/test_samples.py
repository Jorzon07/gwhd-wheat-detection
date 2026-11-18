import numpy as np
import pandas as pd
import pytest

from src.data.samples import WheatSamples
from src.data.transforms import AugConfig


def make(fx, **kw):
    return WheatSamples(fx["split_csv"], fx["ann_csv"], fx["images_dir"], **kw)


def test_length_names_and_eval_mode_is_identity_at_native_size(synthetic_split):
    s = make(synthetic_split, image_size=96, train=False)
    assert len(s) == 6 and s.names == synthetic_split["names"]
    img, boxes, name = s.get(0)
    assert img.shape == (96, 96, 3) and img.dtype == np.uint8 and img.flags["C_CONTIGUOUS"]
    assert boxes.tolist() == [[10, 10, 40, 40], [50, 50, 90, 80]] and name == "img0.png"


def test_eval_mode_resizes_boxes_and_keeps_empty_images(synthetic_split):
    s = make(synthetic_split, image_size=48, train=False)
    img, boxes, _ = s.get(0)
    assert img.shape == (48, 48, 3) and np.allclose(boxes, [[5, 5, 20, 20], [25, 25, 45, 40]])
    img, boxes, _ = s.get(2)                                   # imagen sin espigas
    assert boxes.shape == (0, 4) and boxes.dtype == np.float32


def test_eval_mode_is_deterministic_and_never_augments(synthetic_split):
    s = make(synthetic_split, image_size=96, train=False)
    s.set_epoch(3)
    a, b = s.get(1)[0], s.get(1)[0]
    assert np.array_equal(a, b)


def test_train_mode_deterministic_per_seed_epoch_index_and_varies_across_epochs(synthetic_split):
    s1 = make(synthetic_split, image_size=64, train=True, seed=11)
    s2 = make(synthetic_split, image_size=64, train=True, seed=11)
    for i in range(6):
        a, b = s1.get(i), s2.get(i)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    s2.set_epoch(1)
    assert any(not np.array_equal(s1.get(i)[0], s2.get(i)[0]) for i in range(6))
    s3 = make(synthetic_split, image_size=64, train=True, seed=12)
    assert any(not np.array_equal(s1.get(i)[0], s3.get(i)[0]) for i in range(6))


def test_train_boxes_always_valid(synthetic_split):
    s = make(synthetic_split, image_size=64, train=True, seed=1, aug=AugConfig(scale_crop_p=1.0))
    for epoch in range(15):
        s.set_epoch(epoch)
        for i in range(len(s)):
            _, b, _ = s.get(i)
            assert (b >= 0).all() and (b <= 64).all() and ((b[:, 2] - b[:, 0]) >= 2).all()


def test_missing_annotation_and_missing_file_are_detected(synthetic_split, tmp_path):
    ann = pd.read_csv(synthetic_split["ann_csv"])
    ann.iloc[:-1].to_csv(tmp_path / "short.csv", index=False)
    with pytest.raises(ValueError, match="sin anotación"):
        WheatSamples(synthetic_split["split_csv"], tmp_path / "short.csv", synthetic_split["images_dir"])
    (synthetic_split["images_dir"] / "img3.png").unlink()
    with pytest.raises(FileNotFoundError, match="no existen"):
        make(synthetic_split)


def test_invalid_annotation_string_is_rejected(synthetic_split, tmp_path):
    ann = pd.read_csv(synthetic_split["ann_csv"])
    ann.loc[0, "BoxesString"] = "1 2 3"
    ann.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="anotación inválida"):
        WheatSamples(synthetic_split["split_csv"], tmp_path / "bad.csv", synthetic_split["images_dir"])


def test_lru_cache_is_bounded_and_returns_same_pixels(synthetic_split):
    s = make(synthetic_split, image_size=96, train=False, cache_items=2)
    first = s.get(0)[0].copy()
    for i in (1, 2, 3):
        s.get(i)
    assert len(s._cache) == 2 and 0 not in s._cache
    assert np.array_equal(s.get(0)[0], first)
    nocache = make(synthetic_split, image_size=96, train=False, cache_items=0)
    nocache.get(0)
    assert len(nocache._cache) == 0


def test_column_lookup(synthetic_split):
    s = make(synthetic_split)
    assert s.column("domain")["img4.png"] == "d1"


def test_visualize_saves_figure(synthetic_split, tmp_path):
    from src.data.visualize import save_samples_figure
    ev = make(synthetic_split, image_size=64, train=False)
    tr = make(synthetic_split, image_size=64, train=True, seed=1)
    out = tmp_path / "fig.png"
    save_samples_figure(ev, tr, [0, 2, 3], out, epoch=2)
    assert out.exists() and out.stat().st_size > 1000


def test_samples_from_config_reads_phase3_block(synthetic_split, tmp_path):
    import shutil

    from src.data.samples import samples_from_config
    split_dir = tmp_path / "splitdir"
    split_dir.mkdir()
    for f in ("train.csv", "annotations_clean.csv"):
        shutil.copy(synthetic_split["split_csv"].parent / f, split_dir / f)
    cfg = {"dataset": {"image_ext": ".png"},
           "phase3": {"seed": 3, "split_dir": str(split_dir), "images_dir": str(synthetic_split["images_dir"]),
                      "image_size": 32, "aug": {"hflip": 1.0}, "loader": {"cache_items": 1}}}
    tr = samples_from_config(cfg, "train")
    assert tr.train and tr.image_size == 32 and tr.aug.hflip == 1.0 and tr.cache_items == 1 and tr.seed == 3
    assert not samples_from_config(cfg, "train", train=False).train


@pytest.mark.parametrize("size", [1024, 640])
def test_real_train_samples_match_clean_annotations(size):
    """Datos reales: sin aumentos, el número de cajas coincide con annotations_clean.csv y todas caen dentro."""
    from src.config import load_config, resolve
    p3 = load_config()["phase3"]
    split_dir, images_dir = resolve(p3["split_dir"]), resolve(p3["images_dir"])
    if not (split_dir / "train.csv").exists():
        pytest.skip("Aún no existe la partición")
    s = WheatSamples(split_dir / "train.csv", split_dir / "annotations_clean.csv", images_dir, image_size=size,
                     train=False, check_files=False)
    if not s.path(s.names[0]).is_file():
        pytest.skip("Faltan las imágenes del dataset")
    ann = pd.read_csv(split_dir / "annotations_clean.csv").set_index("image_name")
    for i in range(8):
        img, boxes, name = s.get(i)
        assert img.shape == (size, size, 3)
        assert len(boxes) == ann.at[name, "n_boxes"]
        assert (boxes >= 0).all() and (boxes <= size).all()

def test_returned_arrays_are_writable_and_never_alias_the_cache(synthetic_split):
    s = make(synthetic_split, image_size=64, train=False, cache_items=2)           # 96 -> 64: camino con PIL
    assert s.get(0)[0].flags.writeable
    native = make(synthetic_split, image_size=96, train=False, cache_items=2)
    first = native.get(0)[0]
    assert first.flags.writeable
    reference = first.copy()
    first[:] = 0                                                                    # el llamante modifica su copia
    assert np.array_equal(native.get(0)[0], reference)                              # la caché no se corrompe
    hit = native.get(0)[0]
    hit[:] = 7
    assert np.array_equal(native.get(0)[0], reference)