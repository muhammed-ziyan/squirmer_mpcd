from __future__ import annotations

import numpy as np


def inside_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    d = points - center[None, :]
    return np.sum(d * d, axis=1) < radius * radius


def closest_point_on_sphere(point: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    v = point - center
    n = np.linalg.norm(v)
    if n == 0.0:
        return center + np.array([radius, 0.0, 0.0], dtype=point.dtype)
    return center + v * (radius / n)


def surface_normal(point_on_sphere: np.ndarray, center: np.ndarray) -> np.ndarray:
    v = point_on_sphere - center
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=point_on_sphere.dtype)
    return v / n


