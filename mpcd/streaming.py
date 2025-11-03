from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def stream_step(r: np.ndarray, v: np.ndarray, dt: float, box: np.ndarray) -> None:
    """Ballistic streaming with periodic boundaries (in-place)."""

    n = r.shape[0]
    for i in prange(n):
        r[i, 0] += v[i, 0] * dt
        r[i, 1] += v[i, 1] * dt
        r[i, 2] += v[i, 2] * dt

        # periodic wrap
        for k in range(3):
            L = box[k]
            x = r[i, k]
            # fast modulo for positive L
            x -= np.floor(x / L) * L
            r[i, k] = x


