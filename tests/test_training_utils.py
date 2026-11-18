import pickle

import numpy as np
import pytest

from src.training.utils import (BestTracker, CheckpointManager, CsvLogger, format_eta, load_predictions, lr_at,
                                make_run_dir, rescale_boxes, save_predictions, to_builtin, validate_resume)


def test_run_dirs_are_never_reused(tmp_path):
    dirs = [make_run_dir(tmp_path, "base") for _ in range(5)]
    assert len({d.name for d in dirs}) == 5 and all(d.is_dir() for d in dirs)
    (dirs[0] / "keep.txt").write_text("x")
    make_run_dir(tmp_path, "base")
    assert (dirs[0] / "keep.txt").read_text() == "x"


def test_lr_schedule_shape():
    kw = dict(total_steps=1000, base_lr=0.01, warmup_steps=100, min_lr_ratio=0.01, warmup_factor=0.001)
    assert lr_at(0, **kw) == pytest.approx(0.01 * 0.001)
    assert lr_at(100, **kw) == pytest.approx(0.01)                        # fin del calentamiento = lr base
    assert lr_at(1000, **kw) == pytest.approx(0.01 * 0.01)                # fin = lr base * min_lr_ratio
    assert lr_at(550, **kw) == pytest.approx(0.01 * (0.01 + 0.99 * 0.5))  # punto medio del coseno
    warm = [lr_at(s, **kw) for s in range(0, 101)]
    assert all(b >= a for a, b in zip(warm, warm[1:]))                    # sube durante el calentamiento
    after = [lr_at(s, **kw) for s in range(100, 1001)]
    assert all(b <= a + 1e-15 for a, b in zip(after, after[1:]))          # baja después
    assert lr_at(5000, **kw) == pytest.approx(0.0001)                     # no se sale del rango


def test_lr_schedule_without_warmup():
    assert lr_at(0, total_steps=10, base_lr=0.1, warmup_steps=0) == pytest.approx(0.1)


def test_csv_logger_appends_resumes_and_handles_new_columns(tmp_path):
    p = tmp_path / "m.csv"
    lg = CsvLogger(p)
    lg.log({"epoch": 1, "loss": 0.5})
    lg.log({"epoch": 2, "loss": 0.4, "map50": 0.7})                       # columna nueva a mitad
    again = CsvLogger(p)                                                  # reanudación: conserva lo anterior
    again.log({"epoch": 3, "loss": 0.3, "map50": 0.8})
    rows = CsvLogger(p).rows
    assert [r["epoch"] for r in rows] == ["1", "2", "3"] and rows[0]["map50"] == "" and rows[2]["map50"] == "0.8"
    assert not (tmp_path / "m.csv.tmp").exists()


def test_best_tracker_and_early_stopping():
    t = BestTracker(patience=2)
    assert t.update(1, 0.5) and t.update(2, 0.6)
    assert not t.update(3, 0.6) and not t.should_stop()                   # empate no es mejora
    assert not t.update(4, 0.55) and t.should_stop()
    assert (t.best, t.best_epoch) == (0.6, 2)
    t2 = BestTracker(patience=2)
    t2.update(1, 0.5)
    assert not t2.update(2, float("nan")) and not t2.update(3, None)       # NaN y None nunca mejoran
    assert t2.best == 0.5 and t2.should_stop()
    t3 = BestTracker(patience=0)
    for e in range(10):
        t3.update(e, 0.1)
    assert not t3.should_stop()                                           # patience=0: nunca para
    t4 = BestTracker(); t4.load_state_dict(t.state_dict())
    assert (t4.best, t4.best_epoch, t4.bad_epochs) == (0.6, 2, 2)


def _mgr(tmp_path):
    def save(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return CheckpointManager(tmp_path, save, load)


def test_checkpoint_manager_roundtrip_and_atomicity(tmp_path):
    m = _mgr(tmp_path)
    assert not m.has_last() and m.load() is None
    m.save_last({"epoch": 1}); m.save_last({"epoch": 2}); m.save_best({"epoch": 1})
    assert m.load()["epoch"] == 2 and m.load("best.pt")["epoch"] == 1
    assert not list(tmp_path.glob("*.tmp"))                               # sin temporales residuales


def test_checkpoint_failure_leaves_previous_file_intact(tmp_path):
    good = _mgr(tmp_path)
    good.save_last({"epoch": 1})

    def broken(obj, path):
        open(path, "wb").write(b"parcial")
        raise OSError("disco lleno")
    with pytest.raises(OSError):
        CheckpointManager(tmp_path, broken, good.load_fn).save_last({"epoch": 2})
    assert good.load()["epoch"] == 1                                      # last.pt anterior sigue válido


def test_to_builtin_converts_numpy_types():
    out = to_builtin({"a": np.float64(1.5), "b": [np.int64(3), np.array([1, 2])], "c": (np.float32(2.0),)})
    assert out == {"a": 1.5, "b": [3, [1, 2]], "c": (2.0,)}
    assert type(out["a"]) is float and type(out["b"][0]) is int


def test_validate_resume_reports_differences():
    assert validate_resume({"lr": 0.1, "size": 640}, {"lr": 0.1, "size": 640}, ["lr", "size"]) == []
    diff = validate_resume({"lr": 0.1, "size": 640}, {"lr": 0.2, "size": 640}, ["lr", "size"])
    assert len(diff) == 1 and "lr" in diff[0]


def test_predictions_roundtrip_including_empty_images(tmp_path):
    preds = {"b": (np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float), np.array([0.9, 0.4])),
             "a": (np.zeros((0, 4)), np.zeros(0)), "c": (np.array([[10, 10, 20, 20.0]]), np.array([0.7]))}
    p = tmp_path / "p.npz"
    save_predictions(p, preds)
    back = load_predictions(p)
    assert set(back) == {"a", "b", "c"} and back["a"][0].shape == (0, 4)
    assert np.allclose(back["b"][0], preds["b"][0]) and np.allclose(back["b"][1], preds["b"][1])
    assert np.allclose(back["c"][0], preds["c"][0])
    save_predictions(tmp_path / "e.npz", {})
    assert load_predictions(tmp_path / "e.npz") == {}


def test_rescale_boxes_and_eta():
    assert np.allclose(rescale_boxes([[10, 20, 30, 40]], 1.6), [[16, 32, 48, 64]])
    assert rescale_boxes(np.zeros((0, 4)), 2.0).shape == (0, 4)
    assert format_eta(65) == "1m05s" and format_eta(3725) == "1h02m05s" and format_eta(-3) == "0m00s"