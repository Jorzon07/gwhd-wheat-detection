from __future__ import annotations

import numpy as np

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int, chunk: int = 512) -> list[tuple[int, int, int]]:
    """Pares (i, j, distancia_hamming) con i < j y distancia <= threshold.

    `hashes` es un array uint64 (p. ej. pHash de 64 bits). Se procesa por bloques para limitar memoria.
    """
    h = np.ascontiguousarray(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, len(h), chunk):
        block = h[start:start + chunk]
        x = np.bitwise_xor(block[:, None], h[None, :])                 # (c, n) uint64
        bits = x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
        dist = _POPCOUNT[bits].sum(axis=-1)
        rows, cols = np.nonzero(dist <= threshold)
        for a, b in zip(rows.tolist(), cols.tolist()):
            i = start + a
            if b > i:
                pairs.append((i, b, int(dist[a, b])))
    return pairs