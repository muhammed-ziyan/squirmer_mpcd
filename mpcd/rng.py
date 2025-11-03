from __future__ import annotations

import numpy as np


def seed_all(seed: int) -> None:
    """Seed global NumPy RNG for reproducibility.

    Note: Numba's np.random uses NumPy's RNG state per thread/process in most
    environments. Calling this at program start yields reproducible runs in
    practice for single-threaded kernels. For strict control, pre-generate
    random numbers/axes outside njit and pass them as arrays.
    """

    np.random.seed(seed)


def random_unit_vectors(n: int) -> np.ndarray:
    """Generate n random unit vectors in 3D (Gaussian -> normalize)."""

    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    v /= norms[:, None]
    return v


