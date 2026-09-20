"""
Anillo cargado uniformemente.

Anillo de radio R con carga total Q repartida uniformemente. El campo y el
potencial se obtienen discretizando el anillo en cargas puntuales y superponiendo
(vía coulomb), lo que da resultado exacto dentro, fuera y sobre el eje.

En el eje existe solución analítica cerrada, útil para validación:
    E_z(z) = k Q z / (R² + z²)^{3/2}
    V(z)   = k Q / (R² + z²)^{1/2}

El anillo se define en el plano perpendicular a `axis`, centrado en `position`.
"""

from __future__ import annotations

import numpy as np

from ...core.base import PhysicalObject
from ...core.frame import Frame
from ...physics import coulomb


class RingCharge(PhysicalObject):
    """Anillo de radio `radius` con carga total `charge`.

    Args:
        charge: carga total Q [C].
        radius: radio del anillo R [m], > 0.
        position: centro del anillo [m]. Por defecto, el origen.
        axis: normal al plano del anillo. Por defecto +z.
        n_segments: número de cargas puntuales para discretizar.
        r_min: softening [m].
    """

    def __init__(
        self,
        charge: float,
        radius: float,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        n_segments: int = 200,
        r_min: float | None = None,
    ) -> None:
        super().__init__(position=position)
        if radius <= 0:
            raise ValueError(f"radius debe ser positivo, recibido {radius}")

        self.charge = float(charge)
        self.radius = float(radius)
        self.n_segments = int(n_segments)
        self.r_min = r_min

        axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / np.linalg.norm(axis)

        self._build_discretization()

    def _r_min(self) -> float:
        from ...config.defaults import FIELD_DEFAULTS
        return FIELD_DEFAULTS["r_min"] if self.r_min is None else self.r_min

    def _build_discretization(self) -> None:
        """Coloca n cargas puntuales sobre el anillo, orientado según axis."""
        theta = np.linspace(0, 2 * np.pi, self.n_segments, endpoint=False)
        # Anillo en el plano xy local, luego rotado para que +z_local = axis
        local = np.stack(
            [self.radius * np.cos(theta),
             self.radius * np.sin(theta),
             np.zeros_like(theta)],
            axis=1,
        )
        frame = Frame.from_z_to(self.axis, origin=self.position)
        self._sources = frame.to_global(local)                     # (n, 3)
        self._charges = np.full(self.n_segments, self.charge / self.n_segments)

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
        return self.charge

    def __repr__(self) -> str:
        return (
            f"RingCharge(charge={self.charge:g}, radius={self.radius:g}, "
            f"position={self.position.tolist()})"
        )


__all__ = ["RingCharge"]
