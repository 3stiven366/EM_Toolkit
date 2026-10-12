"""
Esfera cargada: cascarón superficial o volumen uniforme.

Ambos casos tienen solución analítica exacta por la ley de Gauss, así que se
usan las fórmulas cerradas (sin discretización). La esfera es simétricamente
radial respecto a su centro.

Cascarón (toda la carga en la superficie, radio R):
    r < R:  E = 0                    (campo nulo dentro)
    r ≥ R:  E = k Q / r²             (como carga puntual)

Volumen uniforme (carga repartida en el volumen, radio R):
    r < R:  E = k Q r / R³           (crece linealmente con r)
    r ≥ R:  E = k Q / r²             (como carga puntual)

El potencial se obtiene integrando E y fijando V(∞) = 0.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import k_e
from ...core.base import PhysicalObject


class SphereCharge(PhysicalObject):
    """Esfera cargada de radio `radius` y carga total `charge`.

    Args:
        charge: carga total Q [C].
        radius: radio de la esfera R [m], > 0.
        position: centro de la esfera [m]. Por defecto, el origen.
        shell: si True, cascarón (carga en la superficie); si False, volumen
            uniforme.
    """

    def __init__(
        self,
        charge: float,
        radius: float,
        position: np.ndarray | None = None,
        shell: bool = False,
    ) -> None:
        super().__init__(position=position)
        if radius <= 0:
            raise ValueError(f"radius debe ser positivo, recibido {radius}")
        self.charge = float(charge)
        self.radius = float(radius)
        self.shell = shell

    # -- Protocolo ChargeSource ---------------------------------------------

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo E radial [V/m]. (N,3) -> (N,3)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.position                       # (N,3)
        r = np.linalg.norm(rel, axis=1)                    # (N,)

        magnitude = np.empty_like(r)
        outside = r >= self.radius
        inside = ~outside

        # Fuera: como carga puntual (ambos casos)
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude[outside] = k_e * self.charge / r[outside] ** 2
            if self.shell:
                magnitude[inside] = 0.0                     # cascarón: E=0 dentro
            else:
                magnitude[inside] = k_e * self.charge * r[inside] / self.radius**3

        # Dirección radial unitaria (r=0 -> campo nulo)
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = rel / r[:, None]
        direction = np.nan_to_num(direction)               # centro exacto -> 0
        return magnitude[:, None] * direction

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial V [V], con V(∞) = 0. (N,3) -> (N,)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        r = np.linalg.norm(points - self.position, axis=1)
        R = self.radius
        V = np.empty_like(r)
        outside = r >= R
        inside = ~outside

        V[outside] = k_e * self.charge / r[outside]
        if self.shell:
            # Dentro del cascarón el potencial es constante = valor en la superficie
            V[inside] = k_e * self.charge / R
        else:
            # Volumen uniforme: V(r<R) = kQ/(2R)·(3 - r²/R²)
            V[inside] = k_e * self.charge / (2 * R) * (3 - (r[inside] / R) ** 2)
        return V

    def total_charge(self) -> float:
        """Carga total [C]."""
        return self.charge

    def __repr__(self) -> str:
        tipo = "shell" if self.shell else "volume"
        return (
            f"SphereCharge(charge={self.charge:g}, radius={self.radius:g}, "
            f"{tipo}, position={self.position.tolist()})"
        )


__all__ = ["SphereCharge"]
