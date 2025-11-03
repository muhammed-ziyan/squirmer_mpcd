from __future__ import annotations

import numpy as np


def allocate_particles(num_particles: int, dtype: str = "float32"):
    """Allocate SoA arrays for particle positions and velocities.

    Returns (r, v, m), where r and v are (N,3) arrays and m is scalar mass.
    """

    dt = np.float32 if dtype == "float32" else np.float64
    r = np.zeros((num_particles, 3), dtype=dt)
    v = np.zeros((num_particles, 3), dtype=dt)
    return r, v


