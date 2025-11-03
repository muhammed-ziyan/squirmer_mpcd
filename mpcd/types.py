from dataclasses import dataclass
from typing import Tuple


@dataclass
class DomainParams:
    box: Tuple[float, float, float]
    cell_size: float


@dataclass
class FluidParams:
    mean_particles_per_cell: int
    mass: float
    temperature: float
    rotation_angle_deg: float
    dt: float
    dtype: str = "float32"  # "float32" or "float64"


@dataclass
class SquirmerParams:
    radius: float
    B1: float
    B2: float = 0.0
    mass: float = 1e6  # large to reduce jitter
    moment_inertia_scalar: float = 1e6


