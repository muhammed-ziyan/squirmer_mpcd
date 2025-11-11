from __future__ import annotations

import numpy as np


class GhostBuffer:
    """Reusable buffer for ghost (phantom) particles.

    Positions and velocities are stored in preallocated arrays and only the
    prefix [0:num_active) is considered valid at a given step.
    """

    def __init__(self, capacity: int, dtype: str = "float32") -> None:
        dt = np.float32 if dtype == "float32" else np.float64
        self.r = np.zeros((capacity, 3), dtype=dt)
        self.v = np.zeros((capacity, 3), dtype=dt)
        self.num_active = 0

    def reset(self) -> None:
        self.num_active = 0


def estimate_cell_center(cid: int, nx: int, ny: int, cell_size: float, box: np.ndarray) -> np.ndarray:
    ix = cid % nx
    iy = (cid // nx) % ny
    iz = cid // (nx * ny)
    c = np.array([
        (ix + 0.5) * cell_size,
        (iy + 0.5) * cell_size,
        (iz + 0.5) * cell_size,
    ], dtype=box.dtype)
    # already within [0, L)
    return c


def prepare_ghosts_per_cell(
    box: np.ndarray,
    cell_size: float,
    center: np.ndarray,
    radius: float,
    orientation: np.ndarray,
    B1: float,
    B2: float,
    n0: float,
    wall_v: np.ndarray | None = None,
    wall_omega: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ghost counts and mean velocities per cell.

    - Ghosts only for cells intersecting the sphere.
    - Mean velocity = rigid wall velocity (assumed 0 here, handled in coupling)
      plus tangential slip at the surface point nearest the cell center.
    - Returns (counts: (n_cells,), mu: (n_cells,3)).
    """

    from .domain import grid_shape
    from .squirmer import slip_velocity_on_surface
    from .geometry import closest_point_on_sphere, surface_normal

    nx, ny, nz = grid_shape(box, cell_size)
    n_cells = int(nx * ny * nz)
    counts = np.zeros(n_cells, dtype=np.int64)
    mu = np.zeros((n_cells, 3), dtype=box.dtype)

    if wall_v is None:
        wall_v = np.zeros(3, dtype=box.dtype)
    if wall_omega is None:
        wall_omega = np.zeros(3, dtype=box.dtype)

    # volume fraction crude estimate via center distance heuristic
    cell_radius = np.sqrt(3.0) * 0.5 * cell_size
    for cid in range(n_cells):
        c = estimate_cell_center(cid, nx, ny, cell_size, box)
        d = np.linalg.norm(c - center)
        intersects = (d - cell_radius) < radius and (d + cell_radius) > radius
        inside = d + cell_radius <= radius
        outside = d - cell_radius >= radius
        # Only apply ghosts in the near-surface shell to enforce tangential slip
        if outside or inside:
            continue
        if not intersects:
            continue

        # approximate shell volume fraction via overlap proxy
        frac = max(0.0, min(1.0, (radius + cell_radius - d) / (2 * cell_radius)))
        if frac <= 0.0:
            continue

        lam = n0 * frac
        # Poisson sample; for determinism one could use np.random.poisson
        n_g = np.random.poisson(lam)
        if n_g <= 0:
            continue

        # Surface-based mean slip: project cell center on sphere
        ps = closest_point_on_sphere(c, center, radius)
        n_hat = surface_normal(ps, center)
        us = slip_velocity_on_surface(n_hat, orientation, B1, B2)
        # add rigid-body wall velocity at surface point
        rx = ps - center
        uw = np.array([
            wall_v[0] + wall_omega[1] * rx[2] - wall_omega[2] * rx[1],
            wall_v[1] + wall_omega[2] * rx[0] - wall_omega[0] * rx[2],
            wall_v[2] + wall_omega[0] * rx[1] - wall_omega[1] * rx[0],
        ], dtype=box.dtype)
        counts[cid] = n_g
        mu[cid, :] = uw + us

    return counts, mu


def prepare_ghosts_per_cell_into(
    box: np.ndarray,
    cell_size: float,
    center: np.ndarray,
    radius: float,
    orientation: np.ndarray,
    B1: float,
    B2: float,
    n0: float,
    wall_v: np.ndarray,
    wall_omega: np.ndarray,
    counts_out: np.ndarray,
    mu_out: np.ndarray,
) -> None:
    """In-place variant to reuse buffers for counts and mu arrays."""

    from .domain import grid_shape
    from .squirmer import slip_velocity_on_surface
    from .geometry import closest_point_on_sphere, surface_normal

    nx, ny, nz = grid_shape(box, cell_size)
    n_cells = int(nx * ny * nz)
    counts_out[:] = 0
    mu_out[:] = 0
    cell_radius = np.sqrt(3.0) * 0.5 * cell_size
    for cid in range(n_cells):
        c = estimate_cell_center(cid, nx, ny, cell_size, box)
        d = np.linalg.norm(c - center)
        intersects = (d - cell_radius) < radius and (d + cell_radius) > radius
        inside = d + cell_radius <= radius
        outside = d - cell_radius >= radius
        # Only apply ghosts in the near-surface shell (intersecting cells)
        if outside or inside:
            continue
        if not intersects:
            continue
        frac = max(0.0, min(1.0, (radius + cell_radius - d) / (2 * cell_radius)))
        if frac <= 0.0:
            continue
        lam = n0 * frac
        n_g = np.random.poisson(lam)
        if n_g <= 0:
            continue
        ps = closest_point_on_sphere(c, center, radius)
        n_hat = surface_normal(ps, center)
        us = slip_velocity_on_surface(n_hat, orientation, B1, B2)
        rx = ps - center
        uw = np.array([
            wall_v[0] + wall_omega[1] * rx[2] - wall_omega[2] * rx[1],
            wall_v[1] + wall_omega[2] * rx[0] - wall_omega[0] * rx[2],
            wall_v[2] + wall_omega[0] * rx[1] - wall_omega[1] * rx[0],
        ], dtype=box.dtype)
        counts_out[cid] = n_g
        mu_out[cid, :] = uw + us


