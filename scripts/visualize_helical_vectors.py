#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpcd.squirmer import slip_velocity_on_surface, normalize


def fibonacci_sphere(samples: int) -> np.ndarray:
    """Return approximately uniform unit vectors on the sphere."""
    samples = max(4, samples)
    indices = np.arange(samples, dtype=np.float64) + 0.5
    phi = math.pi * (1.0 + math.sqrt(5.0))  # golden angle
    theta = np.arccos(1.0 - 2.0 * indices / samples)
    az = phi * indices
    x = np.sin(theta) * np.cos(az)
    y = np.sin(theta) * np.sin(az)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1).astype(np.float64)


def build_swirl_axis(misalign_deg: float, dtype=np.float64) -> np.ndarray:
    """Swirl axis tilted in x–z plane by misalignment degrees."""
    phi = math.radians(misalign_deg)
    axis = np.array([math.sin(phi), 0.0, math.cos(phi)], dtype=dtype)
    return normalize(axis.astype(dtype))


def plot_vectors(
    normals: np.ndarray,
    radius: float,
    scaled_vectors: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Render sphere surface and surface slip vectors."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot sphere surface for context
    u = np.linspace(0.0, 2.0 * math.pi, 60)
    v = np.linspace(0.0, math.pi, 30)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        xs,
        ys,
        zs,
        rstride=1,
        cstride=1,
        facecolors=plt.cm.Greys_r((zs / radius + 1.0) / 2.0),
        shade=True,
        linewidth=0.0,
        antialiased=False,
    )

    # Place vectors on surface
    origins = radius * normals
    if len(origins) > 0:
        ax.quiver(
            origins[:, 0],
            origins[:, 1],
            origins[:, 2],
            scaled_vectors[:, 0],
            scaled_vectors[:, 1],
            scaled_vectors[:, 2],
            length=1.0,
            normalize=False,
            color="#0072B2",
            linewidth=1.0,
        )

    lim = radius * 1.3
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Helical slip velocity field on squirmer surface")
    ax.view_init(elev=25, azim=35)
    fig.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"Saved visualization to {out_path.resolve()}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static visualization of helical slip velocities on a squirmer."
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Squirmer radius")
    parser.add_argument("--B1", type=float, default=0.03, help="Polar slip amplitude")
    parser.add_argument("--B2", type=float, default=0.0, help="Second squirming mode")
    parser.add_argument("--C1", type=float, default=0.02, help="Azimuthal slip amplitude")
    parser.add_argument(
        "--misalign-deg",
        type=float,
        default=20.0,
        help="Angle between swirl axis and propulsion axis (deg)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of surface sample points for quiver plot",
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=0.4,
        help="Max arrow length as fraction of radius",
    )
    parser.add_argument(
        "--show-both",
        action="store_true",
        help="Render both hemispheres (otherwise only the front hemisphere is shown)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional path to save the figure (PNG). When omitted, only display.",
    )
    args = parser.parse_args()

    orientation = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    swirl_axis = build_swirl_axis(args.misalign_deg, dtype=np.float64)

    normals = fibonacci_sphere(args.samples)
    slips = np.array(
        [
            slip_velocity_on_surface(
                normal,
                orientation,
                args.B1,
                args.B2,
                args.C1,
                swirl_axis,
            )
            for normal in normals
        ],
        dtype=np.float64,
    )

    if not args.show_both:
        mask = np.dot(normals, orientation) >= 0.0
        normals = normals[mask]
        slips = slips[mask]

    max_slip = float(np.max(np.linalg.norm(slips, axis=1)))
    if max_slip > 0.0:
        arrow_scale = args.vector_scale * args.radius / max_slip
    else:
        arrow_scale = 0.0
    scaled_vectors = slips * arrow_scale

    save_path = args.save if args.save else None
    plot_vectors(normals, args.radius, scaled_vectors, save_path)


if __name__ == "__main__":
    main()

