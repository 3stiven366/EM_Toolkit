"""
Hilo recto infinito.

Corriente I a lo largo de una recta infinita. Solución analítica: el campo es
circular alrededor del eje, con magnitud

    B = μ₀ I / (2π s)      (s = distancia perpendicular al eje)

y dirección dada por la regla de la mano derecha (î_dirección × ŝ_radial).

Sobreescribe `magnetic_field` con la fórmula cerrada: un hilo infinito no
admite discretización en segmentos finitos.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import mu0
from .path import CurrentPath


class InfiniteWire(CurrentPath):
    """Hilo recto infinito con corriente `current` a lo largo de `direction`.

    Args:
        current: corriente I [A]. El signo invierte el sentido.
        position: un punto sobre el eje del hilo [m]. Por defecto, el origen.
        direction: dirección de la corriente. Por defecto +z.
        r_min: softening de la distancia al eje [m].
    """

    def __init__(
        self,
        current: float,
        position: np.ndarray | None = None,
        direction: np.ndarray | None = None,
        r_min: float | None = None,
    ) -> None:
        super().__init__(current, position=position, r_min=r_min)
        direction = (
            np.array([0.0, 0.0, 1.0]) if direction is None
            else np.asarray(direction, float)
        )
        if np.linalg.norm(direction) < 1e-12:
            raise ValueError("direction no puede ser el vector nulo")
        self.direction = direction / np.linalg.norm(direction)

    def path(self, t: np.ndarray) -> np.ndarray:
        """Recta a lo largo de `direction`, con t interpretado como coordenada axial."""
        t = np.asarray(t, dtype=float)
        return self.position + t[:, None] * self.direction

    # Sobreescribe el motor genérico con la solución analítica ---------------

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Campo B circular alrededor del eje [T]. (N,3) -> (N,3)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.position                        # (N,3)
        along = (rel @ self.direction)[:, None] * self.direction
        perp = rel - along                                  # componente perpendicular
        s = np.linalg.norm(perp, axis=1)                    # (N,)
        s_eff = np.sqrt(s**2 + self._r_min() ** 2)

        magnitude = mu0 * self.current / (2 * np.pi * s_eff)   # (N,)

        # Dirección azimutal: direction × ŝ_radial (mano derecha)
        with np.errstate(invalid="ignore", divide="ignore"):
            s_hat = perp / s[:, None]
        s_hat = np.nan_to_num(s_hat)                        # sobre el eje -> 0
        phi_hat = np.cross(self.direction, s_hat)           # (N,3), unitario
        return magnitude[:, None] * phi_hat

    def __repr__(self) -> str:
        return (
            f"InfiniteWire(current={self.current:g}, "
            f"direction={self.direction.tolist()})"
        )


__all__ = ["InfiniteWire"]
