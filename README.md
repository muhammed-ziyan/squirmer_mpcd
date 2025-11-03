# MPCD Squirmer (3D, Numba)

A Python/Numba implementation of multi-particle collision dynamics (SRD/MPCD)
with a spherical squirmer boundary using ghost/phantom particles to impose
prescribed tangential slip. Validates the analytical swimming speed
U = 2/3 B1 in 3D.

## Install

```bash
pip install -r requirements.txt
```

## Run single-squirmer demo

```bash
python scripts/run_single_squirmer.py --grid 48 --n0 10 --a 3.0 --B1 0.03 --steps 5000
```

## Notes
- Uses periodic boundaries.
- Emphasizes preallocation and buffer reuse to avoid memory leaks.
- Start with small grids for development (e.g., 32^3) and scale up.

