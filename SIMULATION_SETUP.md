# MPCD Squirmer Simulation - Full Setup Documentation

## Overview

This project implements a **Multi-Particle Collision Dynamics (MPCD/SRD)** simulation with a **spherical squirmer** boundary condition. The simulation uses a ghost/phantom particle technique to enforce prescribed tangential slip at the squirmer surface, validating the analytical swimming speed relationship U = 2/3 B1 in 3D.

## Project Structure

```
squirmer_mpcd/
├── mpcd/                    # Core simulation modules
│   ├── __init__.py         # Package initialization
│   ├── boundary.py         # Boundary conditions (bounce-back on sphere)
│   ├── collision.py        # SRD collision operations
│   ├── diagnostics.py      # Temperature estimation and diagnostics
│   ├── domain.py           # Grid and domain utilities
│   ├── geometry.py         # Geometric calculations
│   ├── ghost.py            # Ghost particle preparation for slip
│   ├── particles.py        # Particle allocation and management
│   ├── rng.py              # Random number generation
│   ├── squirmer.py         # Squirmer state and slip velocity
│   ├── streaming.py        # Particle streaming step
│   └── types.py            # Type definitions
├── scripts/
│   └── run_single_squirmer.py  # Main simulation script
├── tests/
│   └── test_speed_single_squirmer.py  # Unit tests
├── results/                # Simulation output directory
├── requirements.txt        # Python dependencies
└── README.md              # Basic project information
```

## Dependencies

The simulation requires:
- **numpy** (>=1.24) - Numerical computations
- **numba** (>=0.59) - JIT compilation for performance
- **matplotlib** (>=3.8) - Visualization

Install dependencies:
```bash
pip install -r requirements.txt
```

## Simulation Algorithm

The simulation follows a standard MPCD/SRD algorithm with squirmer coupling:

### 1. **Initialization**
   - Create a cubic domain of size `L = grid × a0` (where `a0 = 1.0` is the cell size)
   - Initialize `N = n0 × (grid³)` particles uniformly in the domain
   - Assign Maxwellian velocities with temperature `T` and zero net momentum
   - Place squirmer at domain center with initial velocity (optional, controlled by `--beta`)

### 2. **Time Step Loop** (for each step):
   
   **a. Streaming with Bounce-Back**
   - Stream all particles by `v × dt`
   - Apply impermeable bounce-back boundary condition on the squirmer sphere
   - Track impulse on squirmer from bounce-back collisions
   - Wrap positions periodically

   **b. Random Grid Shift**
   - Apply random shift to collision grid (prevents grid artifacts)

   **c. Ghost Particle Preparation**
   - For each cell intersecting the squirmer:
     - Compute prescribed tangential slip velocity based on squirmer orientation
     - Prepare ghost particles with appropriate velocities to enforce slip
     - Store ghost particle counts and mean velocities per cell

   **d. SRD Collision**
   - Perform stochastic rotation dynamics (SRD) collision in each cell
   - Include ghost particles in collision step to enforce slip boundary condition
   - Track impulse on squirmer from collision step

   **e. Squirmer Update**
   - Update squirmer velocity: `v_new = v_old + (total_impulse / mass)`
   - Update squirmer position: `r_new = r_old + v × dt`
   - Apply periodic boundary conditions to squirmer position

### 3. **Output**
   - Track squirmer position and speed history
   - Compute final and average swimming speeds
   - Compare with theoretical prediction: `U_theory = 2/3 × B1`
   - Generate trajectory and speed plots

## Main Simulation Script

### Usage

```bash
python scripts/run_single_squirmer.py [OPTIONS]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--grid` | int | 32 | Grid size (creates grid³ cells) |
| `--n0` | int | 10 | Mean particles per cell |
| `--a` | float | 3.0 | Squirmer radius |
| `--B1` | float | 0.03 | First squirmer mode (controls swimming speed) |
| `--B2` | float | 0.0 | Second squirmer mode (controls chirality) |
| `--dt` | float | 0.1 | Time step |
| `--alpha` | float | 130.0 | SRD rotation angle (degrees) |
| `--T` | float | 1.0 | Temperature |
| `--steps` | int | 1000 | Number of simulation steps |
| `--beta` | float | 0.95 | Fraction of theoretical speed to prescribe initially |
| `--dtype` | str | "float32" | Data type (float32 or float64) |
| `--seed` | int | 1234 | Random seed |
| `--live` | flag | False | Enable live plotting during simulation |
| `--plot-interval` | int | 500 | Live plot update interval (steps) |
| `--three-d` | flag | False | Add 3D trajectory panel to plots |

### Example Runs

**Basic simulation:**
```bash
python scripts/run_single_squirmer.py --grid 32 --n0 20 --a 4 --B1 0.03 --steps 3000
```

**With live visualization:**
```bash
python scripts/run_single_squirmer.py --grid 32 --n0 20 --a 4 --B1 0.03 --steps 3000 --live --plot-interval 200
```

**High-resolution simulation:**
```bash
python scripts/run_single_squirmer.py --grid 48 --n0 10 --a 3.0 --B1 0.03 --steps 5000 --dtype float64
```

**With 3D trajectory visualization:**
```bash
python scripts/run_single_squirmer.py --grid 32 --n0 20 --a 4 --B1 0.03 --steps 3000 --live --three-d
```

## Key Components

### Squirmer State (`mpcd/squirmer.py`)
- **SquirmerState**: Dataclass storing position, velocity, orientation, and angular velocity
- **slip_velocity_on_surface()**: Computes tangential slip velocity based on squirmer parameters (B1, B2) and surface geometry

### Boundary Conditions (`mpcd/boundary.py`)
- **bounce_back_sphere()**: Implements impermeable bounce-back on spherical squirmer surface
- Returns impulse on squirmer from particle collisions

### Ghost Particles (`mpcd/ghost.py`)
- **prepare_ghosts_per_cell_into()**: Prepares ghost particles in cells intersecting the squirmer
- Ghost particles enforce the prescribed tangential slip velocity at the squirmer surface

### Collision (`mpcd/collision.py`)
- **collide_srd_with_ghosts()**: Performs SRD collision including ghost particles
- Rotates particle velocities around cell center-of-mass velocity
- Returns impulse on squirmer from collision step

### Streaming (`mpcd/streaming.py`)
- **stream_step()**: Streams particles by their velocities

### Domain (`mpcd/domain.py`)
- **grid_shape()**: Computes grid dimensions
- **random_shift()**: Generates random grid shift
- **wrap_positions()**: Applies periodic boundary conditions

## Physical Parameters

### Squirmer Swimming Speed
The theoretical swimming speed for a squirmer is:
```
U_theory = (2/3) × B1
```

The simulation validates this relationship by comparing measured speeds with the theoretical prediction.

### Squirmer Mass
The squirmer mass is approximated as the mass of displaced fluid:
```
m_squirmer = (4π/3) × a³ × n0 × m_particle
```
where `m_particle = 1.0` is the mass of each MPCD particle.

### SRD Collision
- Rotation angle: `α = 130°` (default)
- Collision occurs in cells of size `a0³ = 1.0`
- Random grid shift prevents grid artifacts

## Output and Diagnostics

### Console Output
- Progress updates every 10% of simulation and every 100 steps
- Reports: temperature, squirmer speed, and position
- Final summary with:
  - Final speed (U)
  - Average speed (U_avg)
  - Expected speed (2/3 B1)
  - Error percentages

### Visualization
- **2D trajectory plot**: XY projection of squirmer path
- **Speed plot**: Squirmer speed vs. time step
- **3D trajectory plot** (optional): Full 3D trajectory visualization

### Performance Notes
- Uses Numba JIT compilation for performance-critical functions
- Preallocates buffers to avoid memory leaks
- Supports both float32 and float64 precision
- Start with small grids (e.g., 32³) for development, scale up for production runs

## Typical Results

For a well-converged simulation with `B1 = 0.03`:
- Expected speed: `U = 0.02`
- Typical measured speeds: `U ≈ 0.015 - 0.025` (depending on grid resolution and parameters)
- Error typically: < 10% for good parameter choices

## Notes

- The simulation uses **periodic boundary conditions** for the domain
- The squirmer is a **free swimmer** (not fixed in space)
- Ghost particles are used to enforce **tangential slip** at the squirmer surface
- The implementation emphasizes **preallocation and buffer reuse** to avoid memory leaks
- For development, use smaller grids (32³); for production, scale up to 48³ or larger

## References

This implementation is based on the MPCD/SRD method for simulating squirmer motion in a fluid, validating the analytical swimming speed relationship for spherical squirmers.

