from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SquirmerState:
    position: np.ndarray  # (3,)
    velocity: np.ndarray  # (3,)
    orientation: np.ndarray  # (3,) unit vector
    omega: np.ndarray  # (3,)
    # Azimuthal slip amplitude (C1) and swirl axis (lab-frame unit vector)
    C1: float = 0.0
    swirl_axis: np.ndarray | None = None


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def _tangent_frames(axis: np.ndarray, rhat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (e_phi, e_theta, sin_theta, cos_theta) for spherical tangent basis
    defined with respect to 'axis' at surface normal 'rhat'.

    e_phi = normalized(axis × rhat), e_theta = e_phi × rhat.
    """
    a = normalize(axis)
    n = normalize(rhat)
    ephi = np.cross(a, n)
    nrm = float(np.linalg.norm(ephi))
    if nrm < 1e-14:
        # degenerate (axis parallel to normal)
        return np.zeros(3, dtype=n.dtype), np.zeros(3, dtype=n.dtype), 0.0, 1.0
    ephi = (ephi / nrm).astype(n.dtype)
    etheta = np.cross(ephi, n).astype(n.dtype)
    cos_t = float(np.clip(np.dot(a, n), -1.0, 1.0))
    sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t * cos_t)))
    return ephi, etheta, sin_t, cos_t


def slip_velocity_on_surface(
    normal: np.ndarray,
    orientation: np.ndarray,
    B1: float,
    B2: float,
    C1: float = 0.0,
    swirl_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Return tangential slip vector at the surface point with outward normal.

    The slip is defined as u_s = (B1 sinθ + B2 sinθ cosθ) e_θ, where θ is the
    angle between orientation (swimming axis) and normal. e_θ lies in the tangent
    plane and points along decreasing θ.

    Additionally, when C1 != 0, add an azimuthal swirl term around 'swirl_axis':
    u_φ = C1 sin(θ_n) e_φ^(n), where θ_n is angle between swirl_axis and normal.
    """

    n_hat = normalize(normal)

    # Classic polar (theta) slip with respect to propulsion axis
    _, e_theta_p, sin_theta_p, cos_theta_p = _tangent_frames(orientation, n_hat)
    u_theta = (B1 * sin_theta_p + B2 * sin_theta_p * cos_theta_p) * e_theta_p

    # Azimuthal (phi) slip around a potentially misaligned swirl axis
    u_phi: np.ndarray
    if C1 != 0.0 and swirl_axis is not None:
        e_phi_n, _, sin_theta_n, _ = _tangent_frames(swirl_axis, n_hat)
        u_phi = (C1 * sin_theta_n) * e_phi_n
    else:
        # preserve dtype
        u_phi = np.zeros(3, dtype=n_hat.dtype)

    u = (u_theta + u_phi).astype(n_hat.dtype)
    # Remove any tiny normal component due to numerics
    u -= np.dot(u, n_hat) * n_hat
    return u


