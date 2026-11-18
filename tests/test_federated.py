import pytest

torch = pytest.importorskip("torch")
from src.training.federated import (Accumulator, count_params, fedavg, partition_clients, round_lr, run_round,
                                    select_clients)


def _state(v, steps=5):
    return {"w": torch.full((3,), float(v)), "b": torch.tensor([float(v)]), "bn.num_batches_tracked": torch.tensor(steps)}


def test_fedavg_weighted_mean_and_int_kept():
    g = _state(0, 7)
    out = fedavg(g, [_state(1), _state(4)], [1, 2])
    assert torch.allclose(out["w"], torch.full((3,), 3.0))              # (1*1 + 2*4) / 3
    assert int(out["bn.num_batches_tracked"]) == 7                       # entero: se conserva el global
    assert out["w"].dtype == torch.float32


def test_fedavg_equal_weights_is_mean_and_does_not_mutate():
    g = _state(0)
    before = {k: v.clone() for k, v in g.items()}
    out = fedavg(g, [_state(2), _state(4), _state(6)], [5, 5, 5])
    assert torch.allclose(out["b"], torch.tensor([4.0]))
    assert all(torch.equal(g[k], before[k]) for k in g)


def test_half_precision_keeps_dtype():
    g = {"w": torch.zeros(2, dtype=torch.float16)}
    out = fedavg(g, [{"w": torch.ones(2, dtype=torch.float16)}], [3])
    assert out["w"].dtype == torch.float16 and torch.allclose(out["w"].float(), torch.ones(2))


def test_accumulator_rejects_bad_input():
    acc = Accumulator(_state(0))
    with pytest.raises(ValueError):
        acc.add(_state(1), 0)
    with pytest.raises(ValueError):
        acc.result()


@pytest.mark.parametrize("mode", ["domain", "iid", "single"])
def test_partition_covers_every_image_once(mode):
    doms = ["a"] * 5 + ["b"] * 3 + ["c"] * 2
    cl = partition_clients(doms, mode, seed=1)
    flat = sorted(i for v in cl.values() for i in v)
    assert flat == list(range(10))


def test_partition_domain_pure_and_iid_same_sizes_but_mixed():
    doms = ["a"] * 50 + ["b"] * 30
    d = partition_clients(doms, "domain")
    assert {k: len(v) for k, v in d.items()} == {"a": 50, "b": 30}
    assert all(doms[i] == k for k, v in d.items() for i in v)
    iid = partition_clients(doms, "iid", seed=3)
    assert sorted(len(v) for v in iid.values()) == [30, 50]
    assert any(doms[i] != "a" for i in iid["iid_a"])                     # mezclado
    assert iid == partition_clients(doms, "iid", seed=3)                 # reproducible


def test_select_clients_deterministic_and_all():
    names = [f"c{i}" for i in range(10)]
    assert select_clients(names, 0, 0, 1) == sorted(names)
    a = select_clients(names, 4, 2, 7)
    assert a == select_clients(names, 4, 2, 7) and len(set(a)) == 4


def test_round_lr_decreases():
    lrs = [round_lr(r, 9, 1e-3, 0.01) for r in range(9)]
    assert lrs[0] == pytest.approx(1e-3) and all(x > y for x, y in zip(lrs, lrs[1:]))


def test_run_round_with_stub_training_and_failure():
    model = torch.nn.Linear(2, 1)
    g = {k: v.detach().clone() for k, v in model.state_dict().items()}

    class Boom(RuntimeError):
        pass

    def train_fn(name):
        if name == "bad":
            raise Boom("pérdida no finita")
        val = {"a": 1.0, "b": 3.0}[name]
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(val)
        return {"loss": 0.5}

    new, stats = run_round(model, g, [("a", 10), ("b", 30), ("bad", 5)], train_fn, skip_exc=(Boom,))
    assert [s["ok"] for s in stats] == [True, True, False]
    assert torch.allclose(new["weight"], torch.full((1, 2), 2.5))        # (10*1 + 30*3)/40 = 2.5; "bad" no cuenta
    assert count_params(new) == 3
    with pytest.raises(RuntimeError):
        run_round(model, g, [("bad", 5)], train_fn, skip_exc=(Boom,))
