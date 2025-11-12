#!/usr/bin/env python
from __future__ import annotations

import sys
import argparse
import math
from pathlib import Path
import numpy as np

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpcd.rng import seed_all
from mpcd.domain import grid_shape, random_shift, wrap_positions
from mpcd.streaming import stream_step
from mpcd.collision import collide_srd, collide_srd_with_ghosts
from mpcd.particles import allocate_particles
from mpcd.diagnostics import temperature as estimate_temperature
from mpcd.boundary import bounce_back_sphere
from mpcd.ghost import prepare_ghosts_per_cell_into_with_arm
from mpcd.squirmer import SquirmerState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--n0", type=int, default=10, help="mean particles per cell")
    ap.add_argument("--a", type=float, default=3.0, help="squirmer radius (not used yet)")
    ap.add_argument("--B1", type=float, default=0.03)
    ap.add_argument("--B2", type=float, default=0.0)
    ap.add_argument("--C1", type=float, default=0.0, help="Azimuthal slip amplitude")
    ap.add_argument("--misalign-deg", type=float, default=20.0, help="Angle between swirl and propulsion axes (deg)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=130.0, help="SRD rotation angle in degrees")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--beta", type=float, default=0.95, help="fraction of analytic U=(2/3)B1 to prescribe")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--live", action="store_true", help="enable live plotting during simulation")
    ap.add_argument("--plot-interval", type=int, default=500, help="live plot update interval (steps)")
    ap.add_argument("--three-d", action="store_true", help="add a 3D trajectory panel")
    args = ap.parse_args()

    print("=" * 70)
    print("MPCD Squirmer Simulation - Starting")
    print("=" * 70)
    print(f"Simulation parameters:")
    print(f"  Grid size: {args.grid}^3")
    print(f"  Particles per cell (n0): {args.n0}")
    print(f"  Squirmer radius (a): {args.a}")
    print(f"  B1 parameter: {args.B1}")
    print(f"  B2 parameter: {args.B2}")
    print(f"  Time step (dt): {args.dt}")
    print(f"  SRD rotation angle: {args.alpha}°")
    print(f"  Temperature: {args.T}")
    print(f"  Total steps: {args.steps}")
    print(f"  Random seed: {args.seed}")
    print(f"  Data type: {args.dtype}")
    print("=" * 70)

    seed_all(args.seed)
    print("✓ Random number generator seeded")

    a0 = 1.0
    L = np.array([args.grid * a0, args.grid * a0, args.grid * a0], dtype=np.float64)
    nx, ny, nz = grid_shape(L, a0)
    n_cells = int(nx * ny * nz)

    N = int(args.n0 * n_cells)
    print(f"✓ Initializing {N:,} particles in {n_cells:,} cells ({nx}x{ny}x{nz})")
    r, v = allocate_particles(N, dtype=args.dtype)

    # initialize positions uniformly and velocities Maxwellian
    r[:] = np.random.rand(N, 3) * L[None, :]
    mass = 1.0
    sigma = math.sqrt(args.T / mass)
    v[:] = np.random.normal(scale=sigma, size=v.shape).astype(v.dtype)
    v -= np.mean(v, axis=0, keepdims=True)
    print(f"✓ Particles initialized with Maxwellian velocities")

    alpha_rad = math.radians(args.alpha)

    # Squirmer state (free swimmer)
    center = L / 2.0
    squirmer = SquirmerState(
        position=center.astype(r.dtype),
        velocity=np.zeros(3, dtype=r.dtype),
        orientation=np.array([0.0, 0.0, 1.0], dtype=r.dtype),
        omega=np.zeros(3, dtype=r.dtype),
        C1=float(args.C1),
        swirl_axis=None,
    )
    # Initialize swirl axis misaligned from orientation in x–z plane (rotate around y)
    phi = math.radians(args.misalign_deg)
    swirl_axis0 = np.array([math.sin(phi), 0.0, math.cos(phi)], dtype=r.dtype)
    squirmer.swirl_axis = swirl_axis0 / np.linalg.norm(swirl_axis0)
    # Displaced-fluid mass approximation: rho ~ 1, each SRD particle mass=mass, number density ~ n0 per cell of size a0^3
    squirmer_mass = (4.0 / 3.0) * math.pi * (args.a ** 3) * args.n0 * mass
    # Optional: start close to theoretical speed along orientation
    if args.beta is not None and args.beta > 0.0:
        U_ref = (2.0 * args.B1 / 3.0)
        squirmer.velocity = (args.beta * U_ref) * squirmer.orientation.astype(r.dtype)
    print(f"✓ Squirmer initialized at center: {center}")

    # Ghost buffers (reused)
    counts = np.zeros(int(nx * ny * nz), dtype=np.int64)
    mu = np.zeros((int(nx * ny * nz), 3), dtype=r.dtype)
    arm = np.zeros((int(nx * ny * nz), 3), dtype=np.float64)

    # history for visualization
    pos_hist = np.zeros((args.steps, 3), dtype=np.float64)
    speed_hist = np.zeros(args.steps, dtype=np.float64)

    # Helix verification helpers
    def _unwrap_traj(pos: np.ndarray, Lbox: np.ndarray) -> np.ndarray:
        d = np.diff(pos, axis=0)
        shift = -np.round(d / Lbox) * Lbox
        # keep small jumps unchanged
        for k in range(3):
            mask = np.abs(d[:, k]) < (Lbox[k] / 2.0)
            shift[mask, k] = 0.0
        return np.vstack([pos[0], pos[0] + np.cumsum(d + shift, axis=0)])

    def _pca_axis(seg: np.ndarray) -> np.ndarray:
        X = seg - np.mean(seg, axis=0, keepdims=True)
        cov = (X.T @ X) / max(1, X.shape[0] - 1)
        w, V = np.linalg.eigh(cov)
        a = V[:, int(np.argmax(w))]
        nrm = float(np.linalg.norm(a))
        return a / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _circle_fit(xy: np.ndarray) -> tuple[float, np.ndarray, float]:
        x = xy[:, 0]; y = xy[:, 1]
        x_ = x - np.mean(x); y_ = y - np.mean(y)
        Z = np.vstack([x_ * x_ + y_ * y_, x_, y_, np.ones_like(x_)]).T
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        a, b, c, d = Vt[-1]
        if abs(a) < 1e-14:
            return float("nan"), np.array([float("nan"), float("nan")]), float("nan")
        cx = -b / (2.0 * a); cy = -c / (2.0 * a)
        R = float(np.sqrt(max(1e-24, (b * b + c * c - 4.0 * a * d)) / (4.0 * a * a)))
        resid = float(np.sqrt(np.mean((np.sqrt((x_ - cx) ** 2 + (y_ - cy) ** 2) - R) ** 2)))
        return R, np.array([cx + np.mean(x), cy + np.mean(y)], dtype=np.float64), resid

    def _curvature_torsion(seg: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if seg.shape[0] < 7:
            return np.array([]), np.array([])
        v = (seg[2:] - seg[:-2]) / (2.0 * dt)  # length N-2
        a = (seg[2:] - 2.0 * seg[1:-1] + seg[:-2]) / (dt * dt)  # length N-2
        # jerk aligned with midpoints of v,a
        j_mid = (a[2:] - a[:-2]) / (2.0 * dt)  # length N-4
        v_mid = v[1:-1]  # length N-4
        a_mid = a[1:-1]  # length N-4
        cross = np.cross(v_mid, a_mid)
        vnorm3 = (np.linalg.norm(v_mid, axis=1) ** 3 + 1e-24)
        kappa = np.linalg.norm(cross, axis=1) / vnorm3
        num = np.abs(np.einsum("ij,ij->i", cross, j_mid))
        denom = (np.linalg.norm(cross, axis=1) ** 2 + 1e-24)
        tau = num / denom
        return kappa, tau

    def _helix_metrics(pos_so_far: np.ndarray, Lbox: np.ndarray, dt: float) -> tuple[float, float, float, float, float, float]:
        unwrapped = _unwrap_traj(pos_so_far.astype(np.float64), Lbox.astype(np.float64))
        start = unwrapped.shape[0] // 2
        seg = unwrapped[start:, :]
        if seg.shape[0] < 16:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        axis = _pca_axis(seg[1:] - seg[:-1])
        # plane basis
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(u, axis))) > 0.9:
            u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = u - float(np.dot(u, axis)) * axis
        u /= (np.linalg.norm(u) + 1e-24)
        v = np.cross(axis, u)
        xy = np.c_[seg @ u, seg @ v]
        R_fit, _, resid = _circle_fit(xy)
        kappa, tau = _curvature_torsion(seg, dt)
        if kappa.size == 0 or tau.size == 0:
            return R_fit, resid, float("nan"), float("nan"), float("nan"), float("nan")
        denom = (kappa * kappa + tau * tau + 1e-24)
        R_geo_series = kappa / denom
        P_geo_series = 2.0 * math.pi * (tau / denom)
        R_geo = float(np.median(R_geo_series))
        P_geo = float(np.median(P_geo_series))
        cv_R = float(np.std(R_geo_series) / (R_geo + 1e-24))
        cv_P = float(np.std(P_geo_series) / (P_geo + 1e-24))
        return R_fit, resid, R_geo, P_geo, cv_R, cv_P

    # Optional live visualization setup
    live = bool(args.live)
    plot_interval = max(1, int(args.plot_interval))
    if live:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            # Build figure manually to allow mixed 2D/3D axes
            if args.three_d:
                fig = plt.figure(figsize=(14, 4))
                ax_traj2d = fig.add_subplot(1, 3, 1)
                ax_speed = fig.add_subplot(1, 3, 2)
                ax_traj3d = fig.add_subplot(1, 3, 3, projection='3d')
            else:
                fig = plt.figure(figsize=(10, 4))
                ax_traj2d = fig.add_subplot(1, 2, 1)
                ax_speed = fig.add_subplot(1, 2, 2)
                ax_traj3d = None

            traj_line, = ax_traj2d.plot([], [], lw=1.5)
            ax_traj2d.set_xlabel('x')
            ax_traj2d.set_ylabel('y')
            ax_traj2d.set_title('Squirmer trajectory (xy)')
            ax_traj2d.axis('equal')

            speed_line, = ax_speed.plot([], [], lw=1.0)
            ax_speed.set_xlabel('step')
            ax_speed.set_ylabel('speed |U|')
            ax_speed.set_title('Squirmer speed')

            if ax_traj3d is not None:
                traj3d_line, = ax_traj3d.plot([], [], [], lw=1.0)
                ax_traj3d.set_xlabel('x')
                ax_traj3d.set_ylabel('y')
                ax_traj3d.set_zlabel('z')
                ax_traj3d.set_title('Squirmer trajectory (3D)')
            else:
                traj3d_line = None

            fig.tight_layout()
        except Exception as e:
            print(f"⚠ Live plotting disabled: {e}")
            live = False

    print("\n" + "=" * 70)
    print("Starting simulation loop...")
    print("=" * 70 + "\n")

    for step in range(args.steps):
        # Stream + impermeable bounce on sphere (no tangential slip here)
        imp_bounce = np.zeros(3, dtype=np.float64)
        tau_bounce = np.zeros(3, dtype=np.float64)
        bounce_back_sphere(
            r, v, args.dt,
            squirmer.position, args.a,
            squirmer.velocity, squirmer.omega,
            mass,
            imp_bounce,
            tau_bounce,
        )
        wrap_positions(r, L)
        shift = random_shift(a0)
        # Prepare ghosts for slip inside intersected cells
        prepare_ghosts_per_cell_into_with_arm(
            L, a0,
            squirmer.position.astype(np.float64), args.a,
            squirmer.orientation.astype(np.float64),
            args.B1, args.B2, float(args.C1), squirmer.swirl_axis.astype(np.float64), args.n0,
            squirmer.velocity.astype(np.float64),
            squirmer.omega.astype(np.float64),
            counts, mu, arm,
        )
        imp_coll = np.zeros(3, dtype=np.float64)
        tau_coll = np.zeros(3, dtype=np.float64)
        collide_srd_with_ghosts(
            r, v, L, a0,
            shift.astype(r.dtype),
            alpha_rad, args.T, mass,
            counts, mu.astype(r.dtype),
            imp_coll, arm.astype(np.float64), tau_coll,
        )

        # Update squirmer velocity and position from total impulse
        total_imp = imp_bounce + imp_coll
        squirmer.velocity += (total_imp / squirmer_mass).astype(squirmer.velocity.dtype)
        squirmer.position += squirmer.velocity * args.dt
        # Periodic wrap of squirmer center
        squirmer.position[:] = squirmer.position - np.floor(squirmer.position / L) * L

        # Update angular dynamics from total torque
        total_tau = (tau_bounce + tau_coll).astype(np.float64)
        I = (2.0 / 5.0) * squirmer_mass * (args.a ** 2)
        if I > 0.0:
            squirmer.omega = (squirmer.omega + (total_tau / I) * args.dt).astype(r.dtype)
        # Integrate orientation and swirl axis with ω × a
        def _advance_axis(ax: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
            ax_new = ax + dt * np.cross(omega.astype(ax.dtype), ax)
            nrm = float(np.linalg.norm(ax_new))
            if nrm > 0.0:
                ax_new = ax_new / nrm
            return ax_new.astype(ax.dtype)
        squirmer.orientation = _advance_axis(squirmer.orientation, squirmer.omega, args.dt)
        squirmer.swirl_axis = _advance_axis(squirmer.swirl_axis, squirmer.omega, args.dt)

        # log history
        pos_hist[step, :] = squirmer.position.astype(np.float64)
        speed_hist[step] = float(np.linalg.norm(squirmer.velocity))

        # Progress reporting: every 10% milestones and every 100 steps
        report_interval = max(1, args.steps // 10)
        percentage = 100.0 * (step + 1) / args.steps
        is_milestone = (step + 1) % report_interval == 0
        is_minor_update = (step + 1) % 100 == 0
        
        if is_milestone or is_minor_update:
            Tinst = estimate_temperature(v.astype(np.float64), mass)
            U = float(np.linalg.norm(squirmer.velocity))
            pos = squirmer.position.astype(np.float64)
            if is_milestone:
                # Major milestone (every 10%)
                print(f"[{percentage:5.1f}%] Step {step+1}/{args.steps} | "
                      f"Temperature: {Tinst:.4f} | Speed: {U:.6f} | "
                      f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                # Helix diagnostics
                omega_mag = float(np.linalg.norm(squirmer.omega.astype(np.float64)))
                if omega_mag > 1e-8:
                    p_lab = squirmer.orientation.astype(np.float64)
                    U_par = float(np.dot(squirmer.velocity.astype(np.float64), p_lab))
                    U_perp = float(np.linalg.norm(squirmer.velocity.astype(np.float64) - U_par * p_lab))
                    R_helix = U_perp / omega_mag
                    P_helix = 2.0 * math.pi * U_par / omega_mag
                    print(f"    [helix] R≈{R_helix:.3g}, pitch≈{P_helix:.3g}, |ω|≈{omega_mag:.3g}")
                # Helix verification (geometry-based)
                R_fit, resid, R_geo, P_geo, cv_R, cv_P = _helix_metrics(pos_hist[: step + 1], L, float(args.dt))
                if np.isfinite(R_fit) and np.isfinite(resid):
                    print(f"    [helix-verify] circle-fit R={R_fit:.3g}, rms-resid={resid:.3g}")
                if np.isfinite(R_geo) and np.isfinite(P_geo):
                    print(f"    [helix-verify] geometric R≈{R_geo:.3g}, pitch≈{P_geo:.3g}, CV(R)≈{cv_R:.2f}, CV(P)≈{cv_P:.2f}")
                # Persist metrics
                try:
                    results_dir = Path("results")
                    results_dir.mkdir(parents=True, exist_ok=True)
                    with open(results_dir / "helix_metrics.txt", "a", encoding="utf-8") as f:
                        f.write(f"step={step+1} pct={percentage:.1f} Rfit={R_fit} resid={resid} Rgeo={R_geo} Pgeo={P_geo} cvR={cv_R} cvP={cv_P}\n")
                except Exception as _e_metrics:
                    pass
            elif is_minor_update:
                # Minor update (every 100 steps, but not a milestone)
                print(f"[{percentage:5.1f}%] Step {step+1}/{args.steps} | Speed: {U:.6f}")

        # Live plot update
        if live and (((step + 1) % plot_interval) == 0 or (step + 1) == args.steps):
            try:
                # Update 2D trajectory and speed
                traj_line.set_data(pos_hist[: step + 1, 0], pos_hist[: step + 1, 1])
                speed_line.set_data(np.arange(step + 1), speed_hist[: step + 1])
                ax_traj2d.relim(); ax_traj2d.autoscale_view()
                ax_speed.relim(); ax_speed.autoscale_view()

                # Update 3D trajectory if enabled
                if args.three_d and traj3d_line is not None:
                    xs = pos_hist[: step + 1, 0]
                    ys = pos_hist[: step + 1, 1]
                    zs = pos_hist[: step + 1, 2]
                    traj3d_line.set_data(xs, ys)
                    traj3d_line.set_3d_properties(zs)
                    # Auto scale 3D axes
                    ax_traj3d.auto_scale_xyz(xs, ys, zs)

                plt.pause(0.001)
            except Exception as e:
                print(f"⚠ Live plotting update failed: {e}")
                live = False

    print("\n" + "=" * 70)
    print("Simulation completed!")
    print("=" * 70)
    
    U_meas = float(np.linalg.norm(squirmer.velocity))
    U_avg = float(np.mean(speed_hist))
    U_ref = 2.0 * args.B1 / 3.0
    error = abs(U_meas - U_ref)
    error_pct = 100.0 * error / U_ref if U_ref > 0 else 0.0
    error_avg = abs(U_avg - U_ref)
    error_avg_pct = 100.0 * error_avg / U_ref if U_ref > 0 else 0.0
    
    print(f"\nResults:")
    print(f"  Final speed (U): {U_meas:.6f}")
    print(f"  Average speed (U_avg): {U_avg:.6f}")
    print(f"  Expected speed (2/3 B1): {U_ref:.6f}")
    print(f"  Error (final): {error:.6f} ({error_pct:.2f}%)")
    print(f"  Error (average): {error_avg:.6f} ({error_avg_pct:.2f}%)")
    
    if error_avg_pct < 5.0:
        print("  ✓ Excellent agreement with theory!")
    elif error_avg_pct < 10.0:
        print("  ✓ Good agreement with theory")
    else:
        print("  ⚠ Significant deviation from theory")

    # Visualization: trajectory and speed
    print("\nGenerating visualization...")
    try:
        import matplotlib.pyplot as plt
        if not live:
            # Build figure manually to allow optional 3D panel
            if args.three_d:
                fig = plt.figure(figsize=(14, 4))
                ax_traj2d = fig.add_subplot(1, 3, 1)
                ax_speed = fig.add_subplot(1, 3, 2)
                ax_traj3d = fig.add_subplot(1, 3, 3, projection='3d')
            else:
                fig = plt.figure(figsize=(10, 4))
                ax_traj2d = fig.add_subplot(1, 2, 1)
                ax_speed = fig.add_subplot(1, 2, 2)
                ax_traj3d = None

            # Trajectory projection (xy)
            ax_traj2d.plot(pos_hist[:, 0], pos_hist[:, 1], lw=1.5)
            ax_traj2d.set_xlabel('x')
            ax_traj2d.set_ylabel('y')
            ax_traj2d.set_title('Squirmer trajectory (xy)')
            ax_traj2d.axis('equal')

            # Speed over time
            ax_speed.plot(speed_hist)
            ax_speed.set_xlabel('step')
            ax_speed.set_ylabel('speed |U|')
            ax_speed.set_title('Squirmer speed')

            # 3D trajectory
            if ax_traj3d is not None:
                ax_traj3d.plot(pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2], lw=1.0)
                ax_traj3d.set_xlabel('x')
                ax_traj3d.set_ylabel('y')
                ax_traj3d.set_zlabel('z')
                ax_traj3d.set_title('Squirmer trajectory (3D)')

            fig.tight_layout()
        else:
            # Bring live figure to blocking state
            plt.ioff()
        print("✓ Plot generated and displayed")
        plt.show()
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
    
    print("\n" + "=" * 70)
    print("Simulation finished successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()


