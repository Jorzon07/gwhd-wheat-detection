import pytest

torch = pytest.importorskip("torch")

from src.training.engine import NonFiniteLossError, train_one_epoch  # noqa: E402


class TinyDetector(torch.nn.Module):
    def __init__(self, nan_on=()):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.nan_on, self.calls = set(nan_on), 0

    def forward(self, images, targets=None):
        x = torch.stack([im.mean(dim=(1, 2))[:2] for im in images])
        loss = self.lin(x).pow(2).mean()
        if self.calls in self.nan_on:
            loss = loss * float("nan")
        self.calls += 1
        return {"l": loss}


def loader(n_batches=4, bs=2):
    batch = (tuple(torch.rand(3, 4, 4) for _ in range(bs)), tuple({"boxes": torch.zeros((0, 4))} for _ in range(bs)))
    return [batch] * n_batches


def run(model, optimizer, lds, accum=2, grad_clip=1.0, **kw):
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return train_one_epoch(model, lds, optimizer, scaler, torch.device("cpu"), lambda s: 0.01, 0, grad_clip, accum,
                           False, **kw)


def test_one_epoch_counts_steps_with_accumulation_and_applies_lr_scale_per_group():
    m = TinyDetector()
    opt = torch.optim.SGD([{"params": [m.lin.weight], "lr_scale": 1.0}, {"params": [m.lin.bias], "lr_scale": 0.1}], lr=1.0)
    out = run(m, opt, loader(4), accum=2)
    assert out["step"] == 2 and out["skipped_batches"] == 0 and out["loss"] == out["loss"]       # 4 lotes / 2 = 2 pasos
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
    assert opt.param_groups[1]["lr"] == pytest.approx(0.001)                                      # lr * lr_scale


def test_groups_without_lr_scale_keep_working():
    m = TinyDetector()
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    run(m, opt, loader(2), accum=1)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


def test_the_last_partial_accumulation_group_still_steps():
    m = TinyDetector()
    out = run(m, torch.optim.SGD(m.parameters(), lr=0.1), loader(3), accum=2)
    assert out["step"] == 2                                                                       # 2 lotes + 1 sobrante


def test_non_finite_loss_batches_are_skipped_and_counted():
    m = TinyDetector(nan_on={1})
    out = run(m, torch.optim.SGD(m.parameters(), lr=0.1), loader(4), accum=1)
    assert out["skipped_batches"] == 1 and out["step"] == 3 and all(torch.isfinite(p).all() for p in m.parameters())


def test_too_many_consecutive_non_finite_losses_abort_training():
    m = TinyDetector(nan_on=set(range(10)))
    with pytest.raises(NonFiniteLossError):
        run(m, torch.optim.SGD(m.parameters(), lr=0.1), loader(8), accum=1, max_consecutive_bad=2)