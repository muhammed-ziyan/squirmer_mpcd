from __future__ import annotations

import numpy as np


def grid_shape(box: np.ndarray, cell_size: float) -> np.ndarray:
    """Return integer grid shape (nx, ny, nz)."""

    return np.floor(box / cell_size).astype(np.int64)


def wrap_positions(r: np.ndarray, box: np.ndarray) -> None:
    """In-place periodic wrapping of positions into [0, L) in each dimension."""

    for k in range(3):
        L = box[k]
        r[:, k] -= np.floor(r[:, k] / L) * L


def random_shift(cell_size: float) -> np.ndarray:
    """Random grid shift in [-0.5 a0, 0.5 a0)."""

    return (np.random.rand(3) - 0.5) * cell_size


def positions_to_cell_indices(r: np.ndarray, box: np.ndarray, cell_size: float, shift: np.ndarray) -> np.ndarray:
    """Compute cell indices for each position given a random shift.

    Returns array of shape (N, 3) with integer indices.
    """

    s = r + shift[None, :]
    s = s - np.floor(s / box) * box
    idx = np.floor(s / cell_size).astype(np.int64)
    return idx


