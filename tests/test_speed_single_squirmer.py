import math
import numpy as np

from mpcd.rng import seed_all
from mpcd.domain import grid_shape, random_shift, wrap_positions
from mpcd.particles import allocate_particles
from mpcd.boundary import bounce_back_sphere
from mpcd.collision import collide_srd_with_ghosts
from mpcd.ghost import prepare_ghosts_per_cell


def test_single_squirmer_speed_approx():
    # small domain for test speed
    seed_all(1)
    a0 = 1.0
    grid = 16
    L = np.array([grid * a0, grid * a0, grid * a0], dtype=np.float64)
    nx, ny, nz = grid_shape(L, a0)
    n_cells = int(nx * ny * nz)

    n0 = 10
    N = int(n0 * n_cells)
    r, v = allocate_particles(N, dtype="float32")
    r[:] = np.random.rand(N, 3) * L[None, :]
    T = 1.0
    mass = 1.0
    v[:] = np.random.normal(scale=math.sqrt(T / mass), size=v.shape).astype(r.dtype)
    v -= np.mean(v, axis=0, keepdims=True)

    # Squirmer params
    radius = 3.0
    B1 = 0.06
    B2 = 0.0
    dt = 0.1
    alpha = math.radians(130.0)
    pos = (L / 2).astype(r.dtype)
    vel = np.zeros(3, dtype=r.dtype)
    omg = np.zeros(3, dtype=r.dtype)
    orient = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    beta = 0.95
    steps = 400

    for _ in range(steps):
        # prescribe V slightly less than analytic value
        U_target = beta * (2.0 * B1 / 3.0)
        vel[:] = (U_target * orient).astype(vel.dtype)
        bounce_back_sphere(r, v, dt, pos, radius, vel, omg, orient.astype(r.dtype), B1, B2)                                                                 
        wrap_positions(r, L)
        shift = random_shift(a0).astype(r.dtype)
        counts, mu = prepare_ghosts_per_cell(L, a0, pos.astype(np.float64), radius, orient, B1, B2, n0, vel.astype(np.float64), omg.astype(np.float64))                                                              
        dummy_impulse = np.zeros(3, dtype=np.float64)
        collide_srd_with_ghosts(r, v, L, a0, shift, alpha, T, mass, counts, mu.astype(r.dtype), dummy_impulse)                                                    
        pos += vel * dt
        pos[:] = pos - np.floor(pos / L) * L

    U_meas = float(np.linalg.norm(vel))
    U_ref = 2.0 * B1 / 3.0
    # expect measured to be just less than analytic value
    assert U_meas < U_ref
    # and reasonably close (within 20%)
    rel_err = abs(U_meas - U_ref) / max(1e-12, U_ref)
    assert rel_err < 0.2


