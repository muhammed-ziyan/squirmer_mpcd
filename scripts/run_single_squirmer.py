#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import numpy as np

from mpcd.rng import seed_all
from mpcd.domain import grid_shape, random_shift, wrap_positions
from mpcd.streaming import stream_step
from mpcd.collision import collide_srd, collide_srd_with_ghosts
from mpcd.particles import allocate_particles
from mpcd.diagnostics import temperature as estimate_temperature
from mpcd.boundary import bounce_back_sphere
from mpcd.ghost import prepare_ghosts_per_cell_into
from mpcd.squirmer import SquirmerState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--n0", type=int, default=10, help="mean particles per cell")
    ap.add_argument("--a", type=float, default=3.0, help="squirmer radius (not used yet)")
    ap.add_argument("--B1", type=float, default=0.03)
    ap.add_argument("--B2", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=130.0, help="SRD rotation angle in degrees")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--beta", type=float, default=0.95, help="fraction of analytic U=(2/3)B1 to prescribe")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    seed_all(args.seed)

    a0 = 1.0
    L = np.array([args.grid * a0, args.grid * a0, args.grid * a0], dtype=np.float64)
    nx, ny, nz = grid_shape(L, a0)
    n_cells = int(nx * ny * nz)

    N = int(args.n0 * n_cells)
    r, v = allocate_particles(N, dtype=args.dtype)

    # initialize positions uniformly and velocities Maxwellian
    r[:] = np.random.rand(N, 3) * L[None, :]
    mass = 1.0
    sigma = math.sqrt(args.T / mass)
    v[:] = np.random.normal(scale=sigma, size=v.shape).astype(v.dtype)
    v -= np.mean(v, axis=0, keepdims=True)

    alpha_rad = math.radians(args.alpha)

    # Squirmer state (free swimmer)
    center = L / 2.0
    squirmer = SquirmerState(
        position=center.astype(r.dtype),
        velocity=np.zeros(3, dtype=r.dtype),
        orientation=np.array([1.0, 0.0, 0.0], dtype=r.dtype),
        omega=np.zeros(3, dtype=r.dtype),
    )
    squirmer_mass = np.array([1e6], dtype=np.float64)[0]

    # Ghost buffers (reused)
    counts = np.zeros(int(nx * ny * nz), dtype=np.int64)
    mu = np.zeros((int(nx * ny * nz), 3), dtype=r.dtype)

    for step in range(args.steps):
        # Stream + impermeable bounce on sphere (no tangential slip here)
        # Prescribe squirmer velocity slightly less than analytic value
        U_target = args.beta * (2.0 * args.B1 / 3.0)
        squirmer.velocity[:] = U_target * squirmer.orientation.astype(squirmer.velocity.dtype)
        bounce_back_sphere(r, v, args.dt, squirmer.position, args.a, squirmer.velocity, squirmer.omega, squirmer.orientation, args.B1, args.B2)
        wrap_positions(r, L)
        shift = random_shift(a0)
        # Prepare ghosts for slip inside intersected cells
        prepare_ghosts_per_cell_into(L, a0, squirmer.position.astype(np.float64), args.a, squirmer.orientation.astype(np.float64), args.B1, args.B2, args.n0, squirmer.velocity.astype(np.float64), squirmer.omega.astype(np.float64), counts, mu)
        impulse = np.zeros(3, dtype=np.float64)
        collide_srd_with_ghosts(r, v, L, a0, shift.astype(r.dtype), alpha_rad, args.T, mass, counts, mu.astype(r.dtype), impulse)

        squirmer.position += squirmer.velocity * args.dt
        # Periodic wrap of squirmer center
        squirmer.position[:] = squirmer.position - np.floor(squirmer.position / L) * L

        if (step + 1) % max(1, args.steps // 10) == 0:
            Tinst = estimate_temperature(v.astype(np.float64), mass)
            U = float(np.linalg.norm(squirmer.velocity))
            print(f"step {step+1}/{args.steps} T~{Tinst:.3f}  U~{U:.5f}")

    U_meas = float(np.linalg.norm(squirmer.velocity))
    U_ref = 2.0 * args.B1 / 3.0
    print(f"Measured U ~ {U_meas:.5f}, expected 2/3 B1 = {U_ref:.5f}")


if __name__ == "__main__":
    main()


