"""
Dipolo eléctrico.

Se modela como dos cargas puntuales reales (+q y -q) separadas una distancia d.
Esto da el campo exacto a cualquier distancia (no la aproximación de dipolo
ideal) y reutiliza `PointCharge`. El campo y el potencial se obtienen por
superposición de ambas cargas.

Convención: el momento dipolar p = q·d apunta de la carga negativa a la
positiva.
"""

from __future__ import annotations

import numpy as np

from ...core.base import PhysicalObject
from .point_charge import PointCharge


class Dipole(PhysicalObject):
    """Dipolo eléctrico de dos cargas ±q separadas por `separation`.

    Las cargas se colocan simétricamente respecto al centro `position`, a lo
    largo de `axis`: la positiva en +axis·d/2, la negativa en -axis·d/2.

    Args:
        q: magnitud de la carga [C], > 0.
        separation: distancia entre las cargas d [m], > 0.
        position: centro del dipolo [m]. Por defecto, el origen.
        axis: dirección de la positiva respecto al centro. Por defecto +x.
        r_min: softening de cada carga [m].
    """

    def __init__(
        self,
        q: float,
        separation: float,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        r_min: float | None = None,
    ) -> None:
        super().__init__(position=position)
        if q <= 0:
            raise ValueError(f"q debe ser positiva (la magnitud), recibido {q}")
        if separation <= 0:
            raise ValueError(f"separation debe ser positiva, recibido {separation}")

        self.q = float(q)
        self.separation = float(separation)
        axis = np.array([1.0, 0.0, 0.0]) if axis is None else np.asarray(axis, float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / norm

        offset = self.axis * (self.separation / 2.0)
        self.positive = PointCharge(+self.q, self.position + offset, r_min=r_min)
        self.negative = PointCharge(-self.q, self.position - offset, r_min=r_min)

    # -- Protocolo ChargeSource (por superposición) --------------------------

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo E del dipolo [V/m]. (N,3) -> (N,3)."""
        return self.positive.electric_field(points) + self.negative.electric_field(points)

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial V del dipolo [V]. (N,3) -> (N,)."""
        return self.positive.potential(points) + self.negative.potential(points)

    def total_charge(self) -> float:
        """Carga total: cero (dipolo neutro)."""
        return 0.0

    @property
    def moment(self) -> np.ndarray:
        """Momento dipolar p = q·d, de la negativa a la positiva [C·m]."""
        return self.q * self.separation * self.axis

    def __repr__(self) -> str:
        return (
            f"Dipole(q={self.q:g}, separation={self.separation:g}, "
            f"position={self.position.tolist()})"
        )


__all__ = ["Dipole"]
