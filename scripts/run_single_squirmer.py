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
        orientation=np.array([1.0, 0.0, 0.0], dtype=r.dtype),
        omega=np.zeros(3, dtype=r.dtype),
    )
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

    # history for visualization
    pos_hist = np.zeros((args.steps, 3), dtype=np.float64)
    speed_hist = np.zeros(args.steps, dtype=np.float64)

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
        bounce_back_sphere(
            r, v, args.dt,
            squirmer.position, args.a,
            squirmer.velocity, squirmer.omega,
            mass,
            imp_bounce,
        )
        wrap_positions(r, L)
        shift = random_shift(a0)
        # Prepare ghosts for slip inside intersected cells
        prepare_ghosts_per_cell_into(
            L, a0,
            squirmer.position.astype(np.float64), args.a,
            squirmer.orientation.astype(np.float64),
            args.B1, args.B2, args.n0,
            squirmer.velocity.astype(np.float64),
            squirmer.omega.astype(np.float64),
            counts, mu,
        )
        imp_coll = np.zeros(3, dtype=np.float64)
        collide_srd_with_ghosts(
            r, v, L, a0,
            shift.astype(r.dtype),
            alpha_rad, args.T, mass,
            counts, mu.astype(r.dtype),
            imp_coll,
        )

        # Update squirmer velocity and position from total impulse
        total_imp = imp_bounce + imp_coll
        squirmer.velocity += (total_imp / squirmer_mass).astype(squirmer.velocity.dtype)
        squirmer.position += squirmer.velocity * args.dt
        # Periodic wrap of squirmer center
        squirmer.position[:] = squirmer.position - np.floor(squirmer.position / L) * L

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


