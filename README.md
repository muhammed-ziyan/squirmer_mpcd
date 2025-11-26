# MPCD Squirmer Simulation

A high-performance Python/Numba implementation of Multi-Particle Collision Dynamics (MPCD/SRD) with spherical squirmer boundary conditions. This simulation uses ghost/phantom particles to enforce prescribed tangential slip velocities, enabling the study of self-propelled swimmers and chiral motion in fluid media.

## Features

- **3D MPCD/SRD Fluid Simulation**: Efficient implementation using Numba JIT compilation
- **Spherical Squirmer**: Configurable slip velocity modes (B1, B2, C1) for various swimming behaviors
- **Chiral Swimming**: Support for helical trajectories via misaligned azimuthal slip (C1) and swirl axis
- **Ghost Particle Method**: Advanced boundary condition enforcement using phantom particles
- **Periodic Boundary Conditions**: Suitable for bulk fluid simulations
- **Real-time Visualization**: Live trajectory tracking and speed monitoring
- **Theoretical Validation**: Validates analytical swimming speed relationship U = (2/3) × B1

## Requirements

- Python 3.8+
- NumPy >= 1.24
- Numba >= 0.59
- Matplotlib >= 3.8

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd squirmer_mpcd
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Single Squirmer Simulation

Run a simple squirmer simulation with default parameters:

```bash
python scripts/run_single_squirmer.py --grid 48 --n0 10 --a 3.0 --B1 0.03 --steps 5000
```

### Chiral Swimming (Helical Trajectory)

Simulate a squirmer with chiral motion:

```bash
python scripts/run_single_squirmer.py --grid 48 --n0 10 --a 3.0 --B1 0.03 --C1 0.01 --misalign-deg 20 --steps 5000 --three-d
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--grid` | Grid size (cubic) | 32 |
| `--n0` | Mean particles per cell | 10 |
| `--a` | Squirmer radius | 3.0 |
| `--B1` | First squirmer mode (propulsion) | 0.03 |
| `--B2` | Second squirmer mode (pusher/puller) | 0.0 |
| `--C1` | Azimuthal slip amplitude (chiral) | 0.0 |
| `--misalign-deg` | Angle between swirl and propulsion axes (degrees) | 20.0 |
| `--dt` | Time step | 0.1 |
| `--alpha` | SRD rotation angle (degrees) | 130.0 |
| `--T` | Temperature | 1.0 |
| `--steps` | Number of simulation steps | 1000 |
| `--seed` | Random seed | 1234 |
| `--live` | Enable live plotting | False |
| `--three-d` | Include 3D trajectory visualization | False |

## Project Structure

```
squirmer_mpcd/
├── mpcd/                    # Core simulation modules
│   ├── boundary.py         # Bounce-back boundary conditions
│   ├── collision.py       # SRD collision operations
│   ├── diagnostics.py     # Temperature and diagnostics
│   ├── domain.py          # Grid and domain utilities
│   ├── geometry.py        # Geometric calculations
│   ├── ghost.py           # Ghost particle preparation
│   ├── particles.py       # Particle allocation
│   ├── rng.py             # Random number generation
│   ├── squirmer.py        # Squirmer state and dynamics
│   ├── streaming.py       # Particle streaming step
│   └── types.py           # Type definitions
├── scripts/                # Simulation scripts
│   └── run_single_squirmer.py
├── tests/                  # Test suite
│   └── test_speed_single_squirmer.py
├── results/                # Simulation outputs and figures
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── IMPLEMENTATION_DOCUMENTATION.md  # Detailed implementation guide
├── chiral_squirmer_mpcd_guide.md    # Chiral swimming guide
└── SIMULATION_SETUP.md     # Simulation setup documentation
```

## Usage Examples

### Example 1: Standard Squirmer

```bash
python scripts/run_single_squirmer.py \
    --grid 64 \
    --n0 10 \
    --a 3.0 \
    --B1 0.03 \
    --steps 10000
```

### Example 2: Pusher Squirmer (B2 > 0)

```bash
python scripts/run_single_squirmer.py \
    --grid 48 \
    --n0 10 \
    --a 3.0 \
    --B1 0.03 \
    --B2 0.015 \
    --steps 5000
```

### Example 3: Chiral Swimmer with Live Visualization

```bash
python scripts/run_single_squirmer.py \
    --grid 48 \
    --n0 10 \
    --a 3.0 \
    --B1 0.03 \
    --C1 0.01 \
    --misalign-deg 25 \
    --steps 5000 \
    --live \
    --three-d
```

## Performance Tips

- **Start Small**: Begin with small grids (e.g., 32³) for development and testing
- **Memory Management**: The code emphasizes preallocation and buffer reuse to avoid memory leaks
- **Grid Size**: Scale up grid sizes gradually based on computational resources
- **Numba JIT**: First run will be slower due to JIT compilation; subsequent runs are optimized

## Documentation

- **[Implementation Documentation](IMPLEMENTATION_DOCUMENTATION.md)](IMPLEMENTATION_DOCUMENTATION.md)**: Complete technical documentation of the implementation
- **[Chiral Squirmer Guide](chiral_squirmer_mpcd_guide.md)**: Detailed guide for chiral swimming simulations
- **[Simulation Setup](SIMULATION_SETUP.md)**: Setup and configuration guide

## Theory

The simulation implements the squirmer model with surface slip velocity:

**Standard Squirmer Terms:**
- B1: First mode (propulsion)
- B2: Second mode (pusher/puller behavior)

**Chiral Term:**
- C1: Azimuthal slip amplitude (enables helical motion)

**Theoretical Swimming Speed:**
```
U_theory = (2/3) × B1
```

The simulation validates this relationship and supports additional modes for complex swimming behaviors.

## Testing

Run the test suite to verify the implementation:

```bash
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Uses periodic boundary conditions
- Emphasizes preallocation and buffer reuse to avoid memory leaks
- Start with small grids for development (e.g., 32³) and scale up
- The simulation is optimized for 3D bulk fluid behavior

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

[Add citation information here]
