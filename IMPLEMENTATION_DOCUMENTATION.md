# MPCD Squirmer Simulation - Complete Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Simulation Algorithm](#simulation-algorithm)
6. [Physical Model](#physical-model)
7. [Key Algorithms](#key-algorithms)
8. [Usage Guide](#usage-guide)
9. [Performance Optimizations](#performance-optimizations)

---

## Overview

This project implements a **Multi-Particle Collision Dynamics (MPCD/SRD)** simulation with a **spherical squirmer** boundary condition. The simulation uses a ghost/phantom particle technique to enforce prescribed tangential slip at the squirmer surface, enabling the study of self-propelled swimmers in a fluid medium.

### Key Features
- **3D MPCD/SRD fluid simulation** with periodic boundary conditions
- **Spherical squirmer** with configurable slip velocity modes (B1, B2, C1)
- **Chiral swimming** support via misaligned azimuthal slip (C1) and swirl axis
- **Ghost particle method** for enforcing tangential slip boundary conditions
- **Numba JIT compilation** for high-performance numerical computations
- **Real-time visualization** capabilities with trajectory tracking

### Theoretical Validation
The simulation validates the analytical swimming speed relationship:
```
U_theory = (2/3) × B1
```

---

## Architecture

### Project Structure
```
squirmer_mpcd/
├── mpcd/                          # Core simulation modules
│   ├── __init__.py                # Package initialization
│   ├── boundary.py               # Bounce-back boundary conditions
│   ├── collision.py              # SRD collision operations
│   ├── diagnostics.py            # Temperature and diagnostics
│   ├── domain.py                 # Grid and domain utilities
│   ├── geometry.py               # Geometric calculations
│   ├── ghost.py                  # Ghost particle preparation
│   ├── particles.py              # Particle allocation
│   ├── rng.py                    # Random number generation
│   ├── squirmer.py               # Squirmer state and slip velocity
│   ├── streaming.py              # Particle streaming step
│   └── types.py                  # Type definitions
├── scripts/
│   └── run_single_squirmer.py    # Main simulation script
├── tests/
│   └── test_speed_single_squirmer.py
├── results/                      # Simulation outputs
└── requirements.txt
```

### Data Flow
```
Initialization
    ↓
Time Step Loop:
    1. Streaming + Bounce-back
    2. Random Grid Shift
    3. Ghost Particle Preparation
    4. SRD Collision (with ghosts)
    5. Squirmer Update
    ↓
Output & Visualization
```

---

## Core Components

### 1. Squirmer (`mpcd/squirmer.py`)

**Purpose**: Defines squirmer state and computes slip velocity on the surface.

**Key Classes/Functions**:
- `SquirmerState`: Dataclass storing:
  - `position`: Center of mass position (3D)
  - `velocity`: Translational velocity (3D)
  - `orientation`: Propulsion axis unit vector (3D)
  - `omega`: Angular velocity (3D)
  - `C1`: Azimuthal slip amplitude
  - `swirl_axis`: Swirl axis unit vector (for chiral motion)

- `slip_velocity_on_surface()`: Computes tangential slip velocity
  - **Input**: Surface normal, orientation, B1, B2, C1, swirl_axis
  - **Output**: Tangential slip vector
  - **Formula**:
    - Polar slip: `u_θ = (B1 sin(θ) + B2 sin(θ)cos(θ)) e_θ`
    - Azimuthal slip: `u_φ = C1 sin(θ_n) e_φ` (when C1 ≠ 0)
    - Total: `u_slip = u_θ + u_φ` (projected to tangent plane)

**Implementation Details**:
- Uses tangent frame construction to avoid singularities at poles
- Handles misalignment between propulsion axis and swirl axis
- Ensures slip is strictly tangential (removes any normal component)

### 2. Boundary Conditions (`mpcd/boundary.py`)

**Purpose**: Enforces impermeable bounce-back on the squirmer sphere.

**Key Function**: `bounce_back_sphere()`
- **Algorithm**:
  1. Predict particle positions after streaming: `r_new = r_old + v × dt`
  2. Check if particle penetrates sphere: `|r_new - center| < radius`
  3. If penetrating:
     - Compute surface normal at contact point
     - Calculate rigid-body wall velocity: `u_wall = V + Ω × (r_surface - center)`
     - Reflect normal component of relative velocity: `v_rel' = v_rel - 2(v_rel·n)n`
     - Update particle velocity: `v_new = u_wall + v_rel'`
     - Place particle at surface: `r_new = center + radius × n`
  4. Accumulate impulse and torque on squirmer

**Key Features**:
- Parallelized with Numba `prange`
- Tracks impulse: `ΔP = -m(v_new - v_old)`
- Tracks torque: `τ = (r_surface - center) × ΔP`
- Preserves tangential component (slip handled by ghosts)

### 3. Ghost Particles (`mpcd/ghost.py`)

**Purpose**: Prepares virtual particles to enforce tangential slip boundary conditions.

**Key Function**: `prepare_ghosts_per_cell_into_with_arm()`
- **Algorithm**:
  1. For each cell in the collision grid:
     - Check if cell intersects squirmer sphere
     - If intersecting:
       - Compute volume fraction of cell inside sphere
       - Sample Poisson number of ghosts: `n_g ~ Poisson(λ = n0 × fraction)`
       - Project cell center onto sphere surface
       - Compute surface normal at projection point
       - Calculate slip velocity at that point
       - Set ghost mean velocity: `μ = u_wall + u_slip`
       - Store lever arm: `arm = r_surface - center`
  2. Store ghost counts and mean velocities per cell

**Implementation Details**:
- Only creates ghosts in cells intersecting the sphere (near-surface shell)
- Ghost velocities are Maxwellian-distributed around mean slip velocity
- Reuses preallocated buffers to avoid memory allocation
- Lever arm stored for torque computation

### 4. Collision (`mpcd/collision.py`)

**Purpose**: Performs Stochastic Rotation Dynamics (SRD) collision step.

**Key Functions**:

#### `collide_srd()`: Standard SRD collision
1. Bin particles into cells using random grid shift
2. For each cell:
   - Compute center-of-mass velocity
   - Subtract CM from all velocities
   - Rotate relative velocities by random axis and angle α
   - Rescale to match target temperature (Maxwell-Boltzmann)
   - Add CM back

#### `collide_srd_with_ghosts()`: SRD with ghost particles
1. Bin real particles into cells
2. For each cell with ghosts:
   - Sample ghost velocities from Maxwellian around mean slip
   - Compute combined CM (real + ghosts)
   - Rotate real particle relative velocities
   - Compute impulse on squirmer: `ΔP = -(p_after - p_before)`
   - Compute torque: `τ = arm × ΔP`

**SRD Parameters**:
- Rotation angle: `α = 130°` (default, can be 110-130°)
- Random axis per cell (prevents grid artifacts)
- Temperature rescaling maintains thermal equilibrium

### 5. Streaming (`mpcd/streaming.py`)

**Purpose**: Advances particle positions ballistically.

**Key Function**: `stream_step()`
- Updates positions: `r_new = r_old + v × dt`
- Applies periodic boundary conditions: `r = r - floor(r/L) × L`
- Parallelized with Numba

### 6. Domain (`mpcd/domain.py`)

**Purpose**: Grid and domain management utilities.

**Key Functions**:
- `grid_shape()`: Computes grid dimensions from box size
- `wrap_positions()`: Periodic wrapping
- `random_shift()`: Random grid shift for SRD (prevents grid artifacts)

### 7. Geometry (`mpcd/geometry.py`)

**Purpose**: Geometric calculations for sphere interactions.

**Key Functions**:
- `closest_point_on_sphere()`: Projects point onto sphere surface
- `surface_normal()`: Computes outward normal at surface point
- `inside_sphere()`: Checks if points are inside sphere

### 8. Diagnostics (`mpcd/diagnostics.py`)

**Purpose**: Monitoring and diagnostics.

**Key Functions**:
- `temperature()`: Estimates instantaneous temperature from velocities
- `mean_speed()`: Computes mean particle speed
- `total_momentum()`: Computes total system momentum

---

## Implementation Details

### Squirmer Slip Velocity Computation

The slip velocity is computed using a tangent frame construction:

```python
def slip_velocity_on_surface(normal, orientation, B1, B2, C1, swirl_axis):
    # Polar slip (standard squirmer)
    e_theta = orientation - (orientation·normal) × normal  # tangent component
    e_theta = normalize(e_theta)
    sin_theta = sqrt(1 - (orientation·normal)²)
    u_theta = (B1 × sin_theta + B2 × sin_theta × cos_theta) × e_theta
    
    # Azimuthal slip (chiral term)
    if C1 != 0 and swirl_axis is not None:
        e_phi = normalize(swirl_axis × normal)  # azimuthal direction
        sin_theta_n = sqrt(1 - (swirl_axis·normal)²)
        u_phi = C1 × sin_theta_n × e_phi
    else:
        u_phi = 0
    
    # Total slip (projected to tangent plane)
    u_slip = u_theta + u_phi
    u_slip = u_slip - (u_slip·normal) × normal  # remove normal component
    return u_slip
```

### Ghost Particle Method

The ghost particle method enforces slip by:
1. **Creating virtual particles** in cells intersecting the squirmer
2. **Setting their mean velocity** to match the desired wall velocity (rigid-body + slip)
3. **Including them in SRD collisions** to statistically enforce the boundary condition
4. **Tracking momentum exchange** to update squirmer dynamics

**Advantages**:
- Excellent no-slip quality (including tangential)
- Easy to thermostat
- Statistically consistent with MPCD method

### Bounce-Back Algorithm

The bounce-back enforces impermeability:
1. **Predictive collision detection**: Check if `r + v×dt` penetrates sphere
2. **Normal reflection**: Reflect normal component of relative velocity
3. **Tangential preservation**: Keep tangential component unchanged (slip handled by ghosts)
4. **Momentum conservation**: Apply equal-and-opposite impulse to squirmer

### SRD Collision with Ghosts

The collision step:
1. **Bins particles** into cells (with random shift)
2. **Samples ghost velocities** from Maxwellian around mean slip
3. **Computes combined CM** of real + ghost particles
4. **Rotates real velocities** around combined CM
5. **Tracks momentum change** to compute impulse on squirmer

---

## Simulation Algorithm

### Initialization Phase

1. **Domain Setup**:
   - Create cubic domain: `L = grid × a0` (where `a0 = 1.0`)
   - Compute grid dimensions: `(nx, ny, nz) = floor(L / a0)`
   - Total cells: `n_cells = nx × ny × nz`

2. **Particle Initialization**:
   - Number of particles: `N = n0 × n_cells`
   - Positions: Uniform random in `[0, L)³`
   - Velocities: Maxwellian with temperature `T`, zero net momentum

3. **Squirmer Initialization**:
   - Position: Domain center
   - Velocity: Optional initial velocity (fraction `β` of theoretical speed)
   - Orientation: Initial axis (e.g., `[0, 0, 1]`)
   - Swirl axis: Misaligned from orientation by angle `α` (for chiral motion)

### Time Step Loop

For each time step `dt`:

#### Step 1: Streaming with Bounce-Back
```python
bounce_back_sphere(r, v, dt, squirmer.position, radius, 
                   squirmer.velocity, squirmer.omega, mass,
                   imp_bounce, tau_bounce)
wrap_positions(r, L)
```
- Stream particles: `r = r + v × dt`
- Apply bounce-back on squirmer surface
- Accumulate impulse and torque from collisions
- Wrap positions periodically

#### Step 2: Random Grid Shift
```python
shift = random_shift(a0)
```
- Random shift in `[-0.5a0, 0.5a0)` per dimension
- Prevents grid artifacts in SRD

#### Step 3: Ghost Particle Preparation
```python
prepare_ghosts_per_cell_into_with_arm(
    L, a0, squirmer.position, radius,
    squirmer.orientation, B1, B2, C1, swirl_axis, n0,
    squirmer.velocity, squirmer.omega,
    counts, mu, arm
)
```
- Identify cells intersecting squirmer
- Compute slip velocity at surface points
- Sample ghost particle counts and mean velocities
- Store lever arms for torque computation

#### Step 4: SRD Collision
```python
collide_srd_with_ghosts(
    r, v, L, a0, shift, alpha_rad, T, mass,
    counts, mu, imp_coll, arm, tau_coll
)
```
- Bin particles into cells
- Perform SRD collision including ghosts
- Accumulate impulse and torque on squirmer

#### Step 5: Squirmer Update
```python
# Update velocity
total_imp = imp_bounce + imp_coll
squirmer.velocity += (total_imp / squirmer_mass) × dt

# Update position
squirmer.position += squirmer.velocity × dt
squirmer.position = wrap(squirmer.position, L)

# Update angular velocity
total_tau = tau_bounce + tau_coll
I = (2/5) × squirmer_mass × radius²
squirmer.omega += (total_tau / I) × dt

# Update orientation and swirl axis
squirmer.orientation = advance_axis(squirmer.orientation, omega, dt)
squirmer.swirl_axis = advance_axis(squirmer.swirl_axis, omega, dt)
```

### Output Phase

- Track position and speed history
- Compute diagnostics (temperature, momentum)
- Generate visualizations (trajectory, speed plots)
- Compare with theoretical predictions

---

## Physical Model

### Squirmer Swimming Speed

For a spherical squirmer with slip mode B1:
```
U_theory = (2/3) × B1
```

This is validated by comparing measured speeds with theory.

### Squirmer Mass

Approximated as displaced fluid mass:
```
m_squirmer = (4π/3) × a³ × n0 × m_particle
```
where `m_particle = 1.0` is the MPCD particle mass.

### Chiral Motion

When `C1 ≠ 0` and swirl axis is misaligned from propulsion axis:
- Body spins about internal axis
- Produces helical trajectory
- Helix radius: `R ≈ U_perp / |ω|`
- Helix pitch: `P ≈ 2π × U_par / |ω|`

### SRD Fluid Properties

- **Cell size**: `a0 = 1.0` (reduced units)
- **Mean free path**: `λ = h × sqrt(k_B T / m)`
- **Viscosity**: Depends on `n0`, `α`, `h`
- **Reynolds number**: `Re = ρ U a / η` (should be << 1)
- **Mach number**: `Ma = U / c_s` (should be << 0.1)

---

## Key Algorithms

### Tangent Frame Construction

To compute slip velocity without singularities:
```python
def _tangent_frames(axis, rhat):
    # axis: reference axis (orientation or swirl_axis)
    # rhat: surface normal
    
    a = normalize(axis)
    n = normalize(rhat)
    
    # Azimuthal direction: e_phi = normalize(axis × normal)
    ephi = cross(a, n)
    if norm(ephi) < epsilon:
        # Degenerate case (axis || normal)
        return zeros(3), zeros(3), 0.0, 1.0
    ephi = normalize(ephi)
    
    # Polar direction: e_theta = e_phi × normal
    etheta = cross(ephi, n)
    
    # Angles
    cos_theta = dot(a, n)
    sin_theta = sqrt(1 - cos_theta²)
    
    return ephi, etheta, sin_theta, cos_theta
```

### Ghost Particle Sampling

For each intersecting cell:
```python
# Volume fraction estimate
frac = max(0, min(1, (radius + cell_radius - d) / (2 × cell_radius)))

# Poisson sampling
lambda = n0 × frac
n_ghosts = Poisson(lambda)

# Mean velocity
surface_point = closest_point_on_sphere(cell_center, center, radius)
normal = surface_normal(surface_point, center)
slip = slip_velocity_on_surface(normal, orientation, B1, B2, C1, swirl_axis)
wall_velocity = V + Ω × (surface_point - center)
ghost_mean_velocity = wall_velocity + slip
```

### SRD Rotation

Rodrigues rotation formula:
```python
def rodrigues_rotate(v, axis, angle):
    # v: velocity vector
    # axis: rotation axis (normalized)
    # angle: rotation angle
    
    cos_a = cos(angle)
    sin_a = sin(angle)
    
    v_parallel = dot(v, axis) × axis
    v_perp = v - v_parallel
    
    # Rotate perpendicular component
    v_rotated = cos_a × v_perp + sin_a × cross(axis, v_perp) + v_parallel
    
    return v_rotated
```

---

## Usage Guide

### Basic Simulation

```bash
python scripts/run_single_squirmer.py \
    --grid 32 \
    --n0 10 \
    --a 3.0 \
    --B1 0.03 \
    --steps 1000
```

### Chiral Squirmer (Helical Motion)

```bash
python scripts/run_single_squirmer.py \
    --grid 48 \
    --n0 10 \
    --a 3.0 \
    --B1 0.03 \
    --C1 0.02 \
    --misalign-deg 20.0 \
    --steps 5000
```

### With Live Visualization

```bash
python scripts/run_single_squirmer.py \
    --grid 32 \
    --n0 20 \
    --a 4.0 \
    --B1 0.03 \
    --steps 3000 \
    --live \
    --plot-interval 200 \
    --three-d
```

### High-Resolution Run

```bash
python scripts/run_single_squirmer.py \
    --grid 64 \
    --n0 10 \
    --a 4.0 \
    --B1 0.03 \
    --steps 10000 \
    --dtype float64
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--grid`` | int | 32 | Grid size (creates grid³ cells) |
| `--n0` | int | 10 | Mean particles per cell |
| `--a` | float | 3.0 | Squirmer radius |
| `--B1` | float | 0.03 | First squirmer mode (swimming speed) |
| `--B2` | float | 0.0 | Second squirmer mode (pusher/puller) |
| `--C1` | float | 0.0 | Azimuthal slip amplitude (chirality) |
| `--misalign-deg` | float | 20.0 | Angle between swirl and propulsion axes (degrees) |
| `--dt` | float | 0.1 | Time step |
| `--alpha` | float | 130.0 | SRD rotation angle (degrees) |
| `--T` | float | 1.0 | Temperature |
| `--steps` | int | 1000 | Number of simulation steps |
| `--beta` | float | 0.95 | Fraction of theoretical speed for initial velocity |
| `--dtype` | str | "float32" | Data type (float32 or float64) |
| `--seed` | int | 1234 | Random seed |
| `--live` | flag | False | Enable live plotting |
| `--plot-interval` | int | 500 | Live plot update interval (steps) |
| `--three-d` | flag | False | Add 3D trajectory panel |

---

## Performance Optimizations

### Numba JIT Compilation
- All performance-critical functions use `@njit` decorator
- Parallel loops with `prange` for multi-threading
- Cache enabled for faster startup after first run

### Memory Management
- Preallocated buffers for ghost particles (reused each step)
- Structure-of-Arrays (SoA) layout for particles
- In-place operations where possible

### Algorithmic Optimizations
- Cell-based binning with prefix sums
- Early termination for empty cells
- Vectorized operations with NumPy

### Typical Performance
- **Small grid (32³)**: ~1000 steps/second
- **Medium grid (48³)**: ~300 steps/second
- **Large grid (64³)**: ~100 steps/second

(Performance depends on hardware, number of particles, and squirmer size)

---

## Validation and Testing

### Theoretical Validation
- Swimming speed: `U_measured ≈ (2/3) × B1` (typically < 10% error)
- Momentum conservation: Total momentum should be constant (within numerical precision)
- Temperature: Should remain near target `T` (with small fluctuations)

### Diagnostics
- Instantaneous temperature monitoring
- Squirmer speed tracking
- Position history for trajectory analysis
- Helix metrics (for chiral swimmers): radius, pitch, curvature, torsion

### Common Issues and Solutions

1. **Speed too low/high**: Adjust `B1` or check grid resolution
2. **Temperature drift**: Check thermostat in collision step
3. **Momentum leaks**: Verify impulse/torque bookkeeping
4. **Grid artifacts**: Ensure random shift is applied
5. **Poor slip enforcement**: Increase `n0` or check ghost particle preparation

---

## References

This implementation is based on:
- MPCD/SRD method for fluid simulation
- Squirmer model for self-propelled swimmers
- Ghost particle method for boundary conditions
- Chiral swimming via misaligned slip modes

For theoretical background, see:
- Lighthill (1952) - Squirmer model
- Malevanets & Kapral (1999) - SRD method
- Zöttl & Stark (2016) - Squirmer dynamics

---

## Conclusion

This simulation provides a complete, validated implementation of MPCD squirmer dynamics with support for:
- Standard squirmer swimming (B1, B2 modes)
- Chiral swimming (C1 mode with misaligned axes)
- High-performance computation (Numba JIT)
- Comprehensive diagnostics and visualization

The modular architecture makes it easy to extend with additional features such as multiple squirmers, external fields, or different boundary geometries.

