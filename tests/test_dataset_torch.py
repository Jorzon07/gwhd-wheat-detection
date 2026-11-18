import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import WheatDataset, collate_fn, make_loader, set_epoch  # noqa: E402
from src.data.samples import WheatSamples  # noqa: E402


def build(fx, train=False, size=64, seed=0):
    return WheatDataset(WheatSamples(fx["split_csv"], fx["ann_csv"], fx["images_dir"], image_size=size,
                                     train=train, seed=seed))


def test_item_format(synthetic_split):
    image, target = build(synthetic_split)[0]
    assert image.dtype == torch.float32 and image.shape == (3, 64, 64)
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert target["boxes"].dtype == torch.float32 and target["boxes"].shape == (2, 4)
    assert target["labels"].dtype == torch.int64 and (target["labels"] == 1).all()
    assert target["image_id"].item() == 0 and target["area"].shape == (2,) and target["iscrowd"].shape == (2,)


def test_empty_image_gives_empty_tensors(synthetic_split):
    _, target = build(synthetic_split)[2]
    assert target["boxes"].shape == (0, 4) and target["labels"].shape == (0,) and target["area"].shape == (0,)


def test_train_items_convert_without_stride_errors(synthetic_split):
    ds = build(synthetic_split, train=True)
    for epoch in range(10):                                   # los volteos producen vistas con pasos negativos
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            image, target = ds[i]
            assert image.shape == (3, 64, 64) and target["boxes"].shape[1] == 4


def test_loader_batches_and_collate(synthetic_split):
    loader = make_loader(build(synthetic_split), batch_size=4, shuffle=False, num_workers=0)
    batches = list(loader)
    assert [len(b[0]) for b in batches] == [4, 2]
    images, targets = batches[0]
    assert isinstance(images, tuple) and isinstance(targets, tuple) and len(targets) == 4
    assert collate_fn([(1, 2), (3, 4)]) == ((1, 3), (2, 4))


def test_shuffle_reproducible_for_same_seed_and_epoch(synthetic_split):
    def order(seed, epoch):
        loader = make_loader(build(synthetic_split, train=True), batch_size=2, shuffle=True, seed=seed)
        set_epoch(loader, epoch, seed)
        return [int(t["image_id"]) for _, ts in loader for t in ts]
    assert order(5, 0) == order(5, 0)
    assert order(5, 3) == order(5, 3)
    assert any(order(5, e) != order(5, 0) for e in range(1, 6))
    

def test_subset_indices_keep_original_image_ids(synthetic_split):
    full = build(synthetic_split)
    sub = WheatDataset(full.samples, indices=[4, 1])
    assert len(sub) == 2
    assert [int(sub[i][1]["image_id"]) for i in range(2)] == [4, 1]
    assert sub[0][1]["boxes"].shape == full[4][1]["boxes"].shape