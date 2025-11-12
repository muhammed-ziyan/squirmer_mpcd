from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _linear_cell_id(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return ix + nx * (iy + ny * iz)


@njit(cache=True)
def _compute_cell_ids(r: np.ndarray, box: np.ndarray, cell_size: float, shift: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    n = r.shape[0]
    ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        x = r[i, 0] + shift[0]
        y = r[i, 1] + shift[1]
        z = r[i, 2] + shift[2]

        # wrap
        x -= np.floor(x / box[0]) * box[0]
        y -= np.floor(y / box[1]) * box[1]
        z -= np.floor(z / box[2]) * box[2]

        ix = int(np.floor(x / cell_size))
        iy = int(np.floor(y / cell_size))
        iz = int(np.floor(z / cell_size))

        # guard
        if ix == nx:
            ix = 0
        if iy == ny:
            iy = 0
        if iz == nz:
            iz = 0

        ids[i] = _linear_cell_id(ix, iy, iz, nx, ny)
    return ids


@njit(cache=True)
def _prefix_sum(counts: np.ndarray) -> np.ndarray:
    offs = np.empty_like(counts)
    s = 0
    for i in range(counts.size):
        offs[i] = s
        s += counts[i]
    return offs


@njit(cache=True)
def _rodrigues_rotate(v: np.ndarray, kx: float, ky: float, kz: float, ca: float, sa: float) -> None:
    # In-place rotate vectors in v by axis k and angle alpha
    for i in range(v.shape[0]):
        x, y, z = v[i, 0], v[i, 1], v[i, 2]
        # v_parallel = (v·k) k
        dot = x * kx + y * ky + z * kz
        vx = x * ca + sa * (ky * z - kz * y) + (1.0 - ca) * dot * kx
        vy = y * ca + sa * (kz * x - kx * z) + (1.0 - ca) * dot * ky
        vz = z * ca + sa * (kx * y - ky * x) + (1.0 - ca) * dot * kz
        v[i, 0], v[i, 1], v[i, 2] = vx, vy, vz


@njit(parallel=True, cache=True)
def collide_srd(
    r: np.ndarray,
    v: np.ndarray,
    box: np.ndarray,
    cell_size: float,
    shift: np.ndarray,
    alpha: float,
    temperature: float,
    mass: float,
) -> None:
    """SRD-a collision step with random axis rotation and simple MB rescale.

    - r, v modified in place
    - shift is the random grid shift used for binning
    - temperature used to rescale relative velocities to match MB target per cell
    """

    nx = int(np.floor(box[0] / cell_size))
    ny = int(np.floor(box[1] / cell_size))
    nz = int(np.floor(box[2] / cell_size))
    n_cells = nx * ny * nz
    n = r.shape[0]

    ids = _compute_cell_ids(r, box, cell_size, shift, nx, ny, nz)

    counts = np.zeros(n_cells, dtype=np.int64)
    for i in range(n):
        counts[ids[i]] += 1

    offsets = _prefix_sum(counts)
    # copy for placing
    place = offsets.copy()

    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        cid = ids[i]
        j = place[cid]
        order[j] = i
        place[cid] = j + 1

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # Process each cell independently
    for cid in prange(n_cells):
        start = offsets[cid]
        end = start + counts[cid]
        m = end - start
        if m <= 1:
            continue

        # Compute center of mass velocity
        vcmx = 0.0
        vcmy = 0.0
        vcmz = 0.0
        for t in range(start, end):
            i = order[t]
            vcmx += v[i, 0]
            vcmy += v[i, 1]
            vcmz += v[i, 2]
        invm = 1.0 / m
        vcmx *= invm
        vcmy *= invm
        vcmz *= invm

        # Subtract CM
        # Store rel velocities in a temporary buffer (stack alloc friendly)
        # Numba cannot allocate dynamic 2D arrays easily; operate in-place by
        # subtracting, rotating, then adding back.

        # Random axis per cell
        kx = np.random.normal()
        ky = np.random.normal()
        kz = np.random.normal()
        kn = np.sqrt(kx * kx + ky * ky + kz * kz)
        if kn == 0.0:
            kx, ky, kz = 1.0, 0.0, 0.0
        else:
            kx /= kn
            ky /= kn
            kz /= kn

        # First pass: subtract CM
        for t in range(start, end):
            i = order[t]
            v[i, 0] -= vcmx
            v[i, 1] -= vcmy
            v[i, 2] -= vcmz

        # Rotate in-place
        for t in range(start, end):
            i = order[t]
            x = v[i, 0]
            y = v[i, 1]
            z = v[i, 2]
            dot = x * kx + y * ky + z * kz
            vx = x * ca + sa * (ky * z - kz * y) + (1.0 - ca) * dot * kx
            vy = y * ca + sa * (kz * x - kx * z) + (1.0 - ca) * dot * ky
            vz = z * ca + sa * (kx * y - ky * x) + (1.0 - ca) * dot * kz
            v[i, 0], v[i, 1], v[i, 2] = vx, vy, vz

        # Optional MB rescale to target temperature per cell
        # Compute current relative kinetic energy
        ke = 0.0
        for t in range(start, end):
            i = order[t]
            ke += v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2]
        # degrees of freedom: 3*(m-1)
        dof = 3 * (m - 1)
        if dof > 0 and ke > 0.0:
            target_ke = dof * temperature / mass
            scale = np.sqrt(target_ke / ke)
            for t in range(start, end):
                i = order[t]
                v[i, 0] *= scale
                v[i, 1] *= scale
                v[i, 2] *= scale

        # Add CM back
        for t in range(start, end):
            i = order[t]
            v[i, 0] += vcmx
            v[i, 1] += vcmy
            v[i, 2] += vcmz


@njit(parallel=True, cache=True)
def collide_srd_with_ghosts(
    r: np.ndarray,
    v: np.ndarray,
    box: np.ndarray,
    cell_size: float,
    shift: np.ndarray,
    alpha: float,
    temperature: float,
    mass: float,
    ghost_counts: np.ndarray,
    ghost_mu: np.ndarray,
    impulse_out: np.ndarray,  # shape (3,), accumulated impulse on squirmer
    arm: np.ndarray | None = None,
    torque_out: np.ndarray | None = None,   # shape (3,), accumulated torque on squirmer
) -> None:
    """SRD collision including ghost particles per cell to impose slip.

    Real particles are rotated relative to the combined center-of-mass of
    (real + ghosts). Ghosts are not modified. The change in total momentum of
    real particles is accumulated as impulse on the squirmer (negative sign
    by convention).
    """

    nx = int(np.floor(box[0] / cell_size))
    ny = int(np.floor(box[1] / cell_size))
    nz = int(np.floor(box[2] / cell_size))
    n_cells = nx * ny * nz
    n = r.shape[0]

    ids = _compute_cell_ids(r, box, cell_size, shift, nx, ny, nz)

    counts = np.zeros(n_cells, dtype=np.int64)
    for i in range(n):
        counts[ids[i]] += 1

    offsets = _prefix_sum(counts)
    place = offsets.copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        cid = ids[i]
        j = place[cid]
        order[j] = i
        place[cid] = j + 1

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # reset impulse and torque
    impulse_out[0] = 0.0
    impulse_out[1] = 0.0
    impulse_out[2] = 0.0
    if torque_out is not None:
        torque_out[0] = 0.0
        torque_out[1] = 0.0
        torque_out[2] = 0.0

    for cid in prange(n_cells):
        start = offsets[cid]
        end = start + counts[cid]
        m_real = end - start
        m_ghost = int(ghost_counts[cid])
        if m_real + m_ghost <= 1:
            continue

        # total momentum before
        px0 = 0.0
        py0 = 0.0
        pz0 = 0.0
        for t in range(start, end):
            i = order[t]
            px0 += mass * v[i, 0]
            py0 += mass * v[i, 1]
            pz0 += mass * v[i, 2]

        # sample ghost velocities explicitly
        gx_mu = ghost_mu[cid, 0]
        gy_mu = ghost_mu[cid, 1]
        gz_mu = ghost_mu[cid, 2]
        sigma = np.sqrt(temperature / mass)
        gpx = 0.0
        gpy = 0.0
        gpz = 0.0
        # allocate local arrays for ghosts
        # Note: numba allows dynamic allocation inside njit; scope is per-iteration
        if m_ghost > 0:
            gvel = np.empty((m_ghost, 3), dtype=v.dtype)
            for j in range(m_ghost):
                # Maxwellian around mean slip
                gvx = gx_mu + sigma * np.random.normal()
                gvy = gy_mu + sigma * np.random.normal()
                gvz = gz_mu + sigma * np.random.normal()
                gvel[j, 0] = gvx
                gvel[j, 1] = gvy
                gvel[j, 2] = gvz
                gpx += mass * gvx
                gpy += mass * gvy
                gpz += mass * gvz
        else:
            # dummy to satisfy typing; not used when m_ghost==0
            gvel = np.empty((0, 3), dtype=v.dtype)

        total_mass = mass * (m_real + m_ghost)

        # v_cm from combined momentum
        vcmx = (px0 + gpx) / total_mass
        vcmy = (py0 + gpy) / total_mass
        vcmz = (pz0 + gpz) / total_mass

        # Random axis
        kx = np.random.normal()
        ky = np.random.normal()
        kz = np.random.normal()
        kn = np.sqrt(kx * kx + ky * ky + kz * kz)
        if kn == 0.0:
            kx, ky, kz = 1.0, 0.0, 0.0
        else:
            kx /= kn
            ky /= kn
            kz /= kn

        # subtract CM
        for t in range(start, end):
            i = order[t]
            v[i, 0] -= vcmx
            v[i, 1] -= vcmy
            v[i, 2] -= vcmz

        # rotate real relative velocities
        for t in range(start, end):
            i = order[t]
            x = v[i, 0]
            y = v[i, 1]
            z = v[i, 2]
            dot = x * kx + y * ky + z * kz
            vx = x * ca + sa * (ky * z - kz * y) + (1.0 - ca) * dot * kx
            vy = y * ca + sa * (kz * x - kx * z) + (1.0 - ca) * dot * ky
            vz = z * ca + sa * (kx * y - ky * x) + (1.0 - ca) * dot * kz
            v[i, 0], v[i, 1], v[i, 2] = vx, vy, vz

        # rotate ghost relative velocities (discard after use)
        for j in range(m_ghost):
            x = gvel[j, 0] - vcmx
            y = gvel[j, 1] - vcmy
            z = gvel[j, 2] - vcmz
            dot = x * kx + y * ky + z * kz
            vx = x * ca + sa * (ky * z - kz * y) + (1.0 - ca) * dot * kx
            vy = y * ca + sa * (kz * x - kx * z) + (1.0 - ca) * dot * ky
            vz = z * ca + sa * (kx * y - ky * x) + (1.0 - ca) * dot * kz
            gvel[j, 0] = vx + vcmx
            gvel[j, 1] = vy + vcmy
            gvel[j, 2] = vz + vcmz

        # no additional thermostat when ghosts present (to avoid bias)

        # add CM back
        for t in range(start, end):
            i = order[t]
            v[i, 0] += vcmx
            v[i, 1] += vcmy
            v[i, 2] += vcmz

        # compute impulse on squirmer from change in real momentum
        px1 = 0.0
        py1 = 0.0
        pz1 = 0.0
        for t in range(start, end):
            i = order[t]
            px1 += mass * v[i, 0]
            py1 += mass * v[i, 1]
            pz1 += mass * v[i, 2]
        # Δp_real = p1 - p0; impulse on squirmer = -Δp_real
        dpx = (px0 - px1)
        dpy = (py0 - py1)
        dpz = (pz0 - pz1)
        impulse_out[0] += dpx
        impulse_out[1] += dpy
        impulse_out[2] += dpz

        # Torque contribution using per-cell lever arm (surface point minus center)
        if arm is not None and torque_out is not None:
            rx = arm[cid, 0]
            ry = arm[cid, 1]
            rz = arm[cid, 2]
            tx = ry * dpz - rz * dpy
            ty = rz * dpx - rx * dpz
            tz = rx * dpy - ry * dpx
            torque_out[0] += tx
            torque_out[1] += ty
            torque_out[2] += tz


