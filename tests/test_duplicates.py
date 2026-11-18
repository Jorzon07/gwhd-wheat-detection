import numpy as np

from src.data.duplicates import near_duplicate_pairs


def test_near_duplicates():
    h = np.array([0, 1, 2**64 - 1], dtype=np.uint64)
    assert near_duplicate_pairs(h, threshold=1) == [(0, 1, 1)]
    assert near_duplicate_pairs(h, threshold=0) == []