"""
Carga puntual.

Primera fuente concreta: envuelve la física de `coulomb.py` en un objeto con
posición. Hereda de `PhysicalObject` (posicionamiento) y cumple el protocolo
`ChargeSource`, por lo que el `EMWorld` la reconoce y suma su campo sin conocer
su clase.

Patrón que seguirán las demás fuentes: estado físico mínimo + delegación a las
funciones puras de `physics/`.
"""

from __future__ import annotations

import numpy as np

from ...core.base import PhysicalObject
from ...physics import coulomb


class PointCharge(PhysicalObject):
    """Carga puntual de valor `q` en una posición del espacio.

    Args:
        q: valor de la carga [C]. Positiva o negativa.
        position: posición [m], forma (3,). Por defecto, el origen.
        r_min: softening de la singularidad [m]. Si es None, usa el valor por
            defecto del paquete.
    """

    def __init__(
        self,
        q: float,
        position: np.ndarray | None = None,
        r_min: float | None = None,
    ) -> None:
        super().__init__(position=position)
        self.q = float(q)
        self.r_min = r_min

    def _r_min(self) -> float:
        from ...config.defaults import FIELD_DEFAULTS
        return FIELD_DEFAULTS["r_min"] if self.r_min is None else self.r_min

    # -- Protocolo ChargeSource ---------------------------------------------

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo E en cada punto [V/m]. (N,3) -> (N,3)."""
        return coulomb.point_charge_field(
            points, self.position, self.q, r_min=self._r_min()
        )

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial V en cada punto [V]. (N,3) -> (N,)."""
        return coulomb.point_charge_potential(
            points, self.position, self.q, r_min=self._r_min()
        )

    def total_charge(self) -> float:
        """Carga total [C]."""
        return self.q

    @property
    def sign(self) -> int:
        """Signo de la carga: +1, -1 o 0."""
        return int(np.sign(self.q))

    def __repr__(self) -> str:
        return f"PointCharge(q={self.q:g}, position={self.position.tolist()})"


__all__ = ["PointCharge"]
