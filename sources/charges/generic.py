"""
Distribución de carga arbitraria.

Comodín para geometrías sin clase propia. Se construye de dos maneras:

- Explícita: se dan directamente las posiciones y cargas de un conjunto de
  cargas puntuales.
- Por densidad: se da una función ρ(puntos) [C/m³] y una región; la región se
  discretiza en una malla y cada celda aporta ρ·dV.

En ambos casos el campo y el potencial salen por superposición (coulomb).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...core.base import PhysicalObject
from ...physics import coulomb


class GenericCharge(PhysicalObject):
    """Conjunto arbitrario de cargas puntuales.

    Construir con posiciones y cargas explícitas, o usar el constructor de clase
    `from_density` para muestrear una función de densidad sobre un volumen.

    Args:
        positions: (M, 3) posiciones de las cargas [m].
        charges: (M,) valores de carga [C].
        r_min: softening [m].
    """

    def __init__(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        r_min: float | None = None,
    ) -> None:
        positions = np.atleast_2d(np.asarray(positions, dtype=float))
        charges = np.asarray(charges, dtype=float)
        if positions.shape[0] != charges.shape[0]:
            raise ValueError(
                f"positions ({positions.shape[0]}) y charges "
                f"({charges.shape[0]}) deben tener el mismo número de elementos"
            )
        # Centro geométrico como posición del objeto
        super().__init__(position=positions.mean(axis=0))
        self._sources = positions
        self._charges = charges
        self.r_min = r_min

    def _r_min(self) -> float:
        from ...config.defaults import FIELD_DEFAULTS
        return FIELD_DEFAULTS["r_min"] if self.r_min is None else self.r_min

    @classmethod
    def from_density(
        cls,
        density: Callable[[np.ndarray], np.ndarray],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        n: int = 20,
        r_min: float | None = None,
    ) -> "GenericCharge":
        """Construye la distribución muestreando ρ sobre una malla del volumen.

        Cada celda de la malla aporta una carga puntual dq = ρ(centro)·dV.
        Las celdas con carga nula se descartan.

        Args:
            density: función (K,3) -> (K,) que da ρ [C/m³] en cada punto.
            x_range, y_range, z_range: límites del volumen a muestrear.
            n: número de celdas por eje (la malla tiene n³ celdas).
        """
        xs = np.linspace(*x_range, n)
        ys = np.linspace(*y_range, n)
        zs = np.linspace(*z_range, n)
        dV = (
            (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])
            if n > 1 else 0.0
        )
        X, Y, Z = np.meshgrid(xs, ys, zs)
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

        rho = np.asarray(density(pts), dtype=float)
        dq = rho * dV

        nonzero = dq != 0.0
        if not np.any(nonzero):
            raise ValueError("La densidad es nula en toda la región muestreada")
        return cls(pts[nonzero], dq[nonzero], r_min=r_min)

    # -- Protocolo ChargeSource ---------------------------------------------

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo E [V/m]. (N,3) -> (N,3)."""
        return coulomb.electric_field(
            points, self._sources, self._charges, r_min=self._r_min()
        )

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial V [V]. (N,3) -> (N,)."""
        return coulomb.potential(
            points, self._sources, self._charges, r_min=self._r_min()
        )

    def total_charge(self) -> float:
        """Carga total [C]."""
        return float(np.sum(self._charges))

    @property
    def n_charges(self) -> int:
        """Número de cargas puntuales que componen la distribución."""
        return self._sources.shape[0]

    def __repr__(self) -> str:
        return (
            f"GenericCharge(n_charges={self.n_charges}, "
            f"total_charge={self.total_charge():g})"
        )


__all__ = ["GenericCharge"]
