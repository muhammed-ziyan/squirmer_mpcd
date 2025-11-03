from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SquirmerState:
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    orientation: np.ndarray  # (3,) unit vector
    omega: np.ndarray  # (3,)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def slip_velocity_on_surface(normal: np.ndarray, orientation: np.ndarray, B1: float, B2: float) -> np.ndarray:
    """Return tangential slip vector at the surface point with outward normal.

    The slip is defined as u_s = (B1 sinθ + B2 sinθ cosθ) e_θ, where θ is the
    angle between orientation (swimming axis) and normal. e_θ lies in the
    tangent plane and points along decreasing θ.
    """

    o = normalize(orientation)
    n = normalize(normal)

    cos_theta = np.clip(np.dot(o, n), -1.0, 1.0)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

    # e_theta direction: component of o orthogonal to n, normalized, with sign
    t = o - cos_theta * n
    t_norm = np.linalg.norm(t)
    if t_norm > 0:
        e_theta = t / t_norm
    else:
        # at the pole, direction arbitrary in tangent plane; choose any
        e_theta = np.array([1.0, 0.0, 0.0], dtype=n.dtype)
        # project to tangent plane just in case
        e_theta -= np.dot(e_theta, n) * n
        en = np.linalg.norm(e_theta)
        if en > 0:
            e_theta /= en

    amp = B1 * sin_theta + B2 * sin_theta * cos_theta
    return amp * e_theta


