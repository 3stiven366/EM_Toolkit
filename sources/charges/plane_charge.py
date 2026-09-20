"""
Plano infinito cargado uniformemente.

Plano infinito con densidad superficial σ. El campo es uniforme, perpendicular
al plano y apunta hacia afuera en ambos lados (hacia adentro si σ < 0):

    E = σ / (2ε₀)     (magnitud constante, independiente de la distancia)

El potencial crece linealmente con la distancia al plano y diverge en el
infinito, así que se mide respecto al propio plano (V = 0 sobre él):

    V(d) = -σ/(2ε₀) · |d|     (d = distancia con signo al plano)

Solución analítica pura: no hay discretización.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import eps0
from ...core.base import PhysicalObject


class PlaneCharge(PhysicalObject):
    """Plano infinito con densidad superficial `sigma`.

    El plano pasa por `position` y su normal es `normal`. El campo en cada lado
    apunta alejándose del plano (para σ > 0).

    Args:
        sigma: densidad superficial de carga σ [C/m²].
        position: un punto del plano [m]. Por defecto, el origen.
        normal: vector normal al plano. Por defecto +z.
    """

    def __init__(
        self,
        sigma: float,
        position: np.ndarray | None = None,
        normal: np.ndarray | None = None,
    ) -> None:
        super().__init__(position=position)
        self.sigma = float(sigma)

        normal = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float)
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("normal no puede ser el vector nulo")
        self.normal = normal / np.linalg.norm(normal)

    def _signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Distancia con signo de cada punto al plano (proyección sobre normal)."""
        return (points - self.position) @ self.normal      # (N,)

    # -- Protocolo ChargeSource ---------------------------------------------

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo E uniforme [V/m]. (N,3) -> (N,3).

        Apunta en +normal en el lado positivo y en -normal en el negativo.
        Sobre el plano (d = 0) se devuelve cero por convención.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        d = self._signed_distance(points)                  # (N,)
        magnitude = self.sigma / (2 * eps0)
        side = np.sign(d)                                   # +1, -1 o 0 sobre el plano
        return magnitude * side[:, None] * self.normal[None, :]

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial V [V], con V = 0 sobre el plano. (N,3) -> (N,)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        d = self._signed_distance(points)
        return -self.sigma / (2 * eps0) * np.abs(d)

    def total_charge(self) -> float:
        """Carga total: infinita (plano infinito)."""
        return np.inf if self.sigma != 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"PlaneCharge(sigma={self.sigma:g}, "
            f"normal={self.normal.tolist()}, position={self.position.tolist()})"
        )


__all__ = ["PlaneCharge"]
