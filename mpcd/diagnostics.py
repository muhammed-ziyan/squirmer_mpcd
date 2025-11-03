from __future__ import annotations

import numpy as np


def temperature(v: np.ndarray, mass: float) -> float:
    """Return instantaneous temperature estimate from velocities (k_B = 1)."""

    n = v.shape[0]
    if n == 0:
        return 0.0
    ke = 0.5 * mass * np.sum(v * v)
    dof = 3 * n
    return 2.0 * ke / dof


def mean_speed(v: np.ndarray) -> float:
    s = np.sqrt(np.sum(v * v, axis=1))
    return float(np.mean(s))


def total_momentum(v: np.ndarray, mass: float) -> np.ndarray:
    return mass * np.sum(v, axis=0)


