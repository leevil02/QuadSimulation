"""Set-based geometry. Drones and obstacles are represented as sets"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Ball:
    """All points within radius of center"""
    center: np.ndarray
    radius: float

    def distance_to(self, other: "Ball") -> float:
        """Gap between two balls. Negative means they overlap (collision)."""
        return float(np.linalg.norm(self.center - other.center) - self.radius - other.radius)

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.linalg.norm(point - self.center) <= self.radius)


@dataclass
class Ellipsoid:
    """{x : (x-c)' P^-1 (x-c) <= scale^2}."""
    center: np.ndarray
    shape: np.ndarray      # PD matrix P
    scale: float = 2.0     # ~ 2-sigma

    def boundary_2d(self, n: int = 60) -> np.ndarray:
        """Points on the boundary, for plotting (2D only)."""
        theta = np.linspace(0, 2 * np.pi, n)
        circle = np.vstack([np.cos(theta), np.sin(theta)])
        L = np.linalg.cholesky(self.shape)
        return (self.center[:, None] + self.scale * L @ circle).T
