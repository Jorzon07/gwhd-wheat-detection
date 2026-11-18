import time

import numpy as np

from src.evaluation.benchmark_inference import bench_callable, timing_stats


def test_timing_stats_values():
    st = timing_stats([1.0, 2.0, 3.0, 4.0, 100.0])
    assert st["median_ms"] == 3.0 and st["min_ms"] == 1.0 and st["n"] == 5
    assert st["p95_ms"] > st["median_ms"] and abs(st["mean_ms"] - 22.0) < 1e-9


def test_bench_callable_discards_warmup_and_syncs():
    calls, syncs = [], []

    def fn(x):
        calls.append(x)
        time.sleep(0.002)

    out = bench_callable(fn, list(range(10)), warmup=3, sync=lambda: syncs.append(1))
    assert len(out) == 10 and len(calls) == 13                       # 3 de calentamiento + 10 medidas
    assert np.median(out) >= 1.5 and all(t > 0 for t in out)
    assert len(syncs) == 3 + 2 * 10                                  # antes y después de cada medida
