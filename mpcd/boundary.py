from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _normalize3(x: float, y: float, z: float):
    n = np.sqrt(x * x + y * y + z * z)
    if n == 0.0:
        return 1.0, 0.0, 0.0
    return x / n, y / n, z / n


@njit(cache=True)
def _slip_velocity(normal: np.ndarray, orientation: np.ndarray, B1: float, B2: float):
    # Compute tangential unit e_theta and amplitude B1 sinθ + B2 sinθ cosθ
    nx, ny, nz = normal[0], normal[1], normal[2]
    ox, oy, oz = orientation[0], orientation[1], orientation[2]
    # normalize inputs (defensive)
    ox, oy, oz = _normalize3(ox, oy, oz)
    nx, ny, nz = _normalize3(nx, ny, nz)
    cos_th = ox * nx + oy * ny + oz * nz
    if cos_th > 1.0:
        cos_th = 1.0
    if cos_th < -1.0:
        cos_th = -1.0
    sin_th = np.sqrt(max(0.0, 1.0 - cos_th * cos_th))
    # e_theta direction: component of o orthogonal to n
    tx = ox - cos_th * nx
    ty = oy - cos_th * ny
    tz = oz - cos_th * nz
    tx, ty, tz = _normalize3(tx, ty, tz)
    amp = B1 * sin_th + B2 * sin_th * cos_th
    return np.array([amp * tx, amp * ty, amp * tz], dtype=normal.dtype)


@njit(parallel=True, cache=True)
def bounce_back_sphere(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    center: np.ndarray,
    radius: float,
    squirmer_v: np.ndarray,
    squirmer_omega: np.ndarray,
    orientation: np.ndarray,
    B1: float,
    B2: float,
) -> None:
    """Impermeable bounce-back on a moving sphere with rigid-body wall velocity.

    Enforces no-penetration by reflecting the normal component of the relative
    velocity v_rel = v - (V + Ω × (x_s - R)) at the surface. The tangential
    component is left unchanged here; tangential slip is imposed statistically
    via ghost particles during collision.
    """

    n_particles = r.shape[0]
    eps = 1e-7
    for i in prange(n_particles):
        # Predict position
        x_new0 = r[i, 0] + v[i, 0] * dt
        x_new1 = r[i, 1] + v[i, 1] * dt
        x_new2 = r[i, 2] + v[i, 2] * dt

        dx0 = x_new0 - center[0]
        dx1 = x_new1 - center[1]
        dx2 = x_new2 - center[2]
        d2 = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
        if d2 >= radius * radius:
            # no collision
            r[i, 0] = x_new0
            r[i, 1] = x_new1
            r[i, 2] = x_new2
            continue

        # Compute surface normal at projection
        dnorm = np.sqrt(d2)
        if dnorm == 0.0:
            nx, ny, nz = 1.0, 0.0, 0.0
        else:
            nx, ny, nz = dx0 / dnorm, dx1 / dnorm, dx2 / dnorm

        # Surface point
        xs0 = center[0] + nx * (radius + eps)
        xs1 = center[1] + ny * (radius + eps)
        xs2 = center[2] + nz * (radius + eps)

        # Rigid-body wall velocity at surface
        rx0 = xs0 - center[0]
        rx1 = xs1 - center[1]
        rx2 = xs2 - center[2]
        wx, wy, wz = squirmer_omega[0], squirmer_omega[1], squirmer_omega[2]
        # Ω × r
        cx0 = wy * rx2 - wz * rx1
        cx1 = wz * rx0 - wx * rx2
        cx2 = wx * rx1 - wy * rx0
        uw0 = squirmer_v[0] + cx0
        uw1 = squirmer_v[1] + cx1
        uw2 = squirmer_v[2] + cx2

        # Relative velocity and reflection of normal component
        vrel0 = v[i, 0] - uw0
        vrel1 = v[i, 1] - uw1
        vrel2 = v[i, 2] - uw2
        vdotn = vrel0 * nx + vrel1 * ny + vrel2 * nz
        # reflect normal if penetrating
        if vdotn < 0.0:
            vrel0 = vrel0 - 2.0 * vdotn * nx
            vrel1 = vrel1 - 2.0 * vdotn * ny
            vrel2 = vrel2 - 2.0 * vdotn * nz

        # Impose tangential slip: set tangential part of relative velocity to u_slip
        n_vec = np.array([nx, ny, nz], dtype=r.dtype)
        us = _slip_velocity(n_vec, orientation, B1, B2)
        # Remove tangential component from vrel, then add desired us
        vdotn2 = vrel0 * nx + vrel1 * ny + vrel2 * nz
        vtan0 = vrel0 - vdotn2 * nx
        vtan1 = vrel1 - vdotn2 * ny
        vtan2 = vrel2 - vdotn2 * nz
        # set tangential to slip
        vrel0 = vdotn2 * nx + us[0]
        vrel1 = vdotn2 * ny + us[1]
        vrel2 = vdotn2 * nz + us[2]

        # Update velocity and place at surface
        v[i, 0] = uw0 + vrel0
        v[i, 1] = uw1 + vrel1
        v[i, 2] = uw2 + vrel2
        r[i, 0] = xs0
        r[i, 1] = xs1
        r[i, 2] = xs2


