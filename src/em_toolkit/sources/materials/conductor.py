"""
Esfera conductora en un campo eléctrico uniforme externo.

Una esfera conductora de radio R colocada en un campo uniforme E₀ redistribuye
su carga hasta anular el campo en su interior. El resultado externo es el campo
aplicado más el de un dipolo inducido de momento

    p = 4π ε₀ R³ E₀

Campo total (origen en el centro de la esfera):
    r < R:  E = 0                         (interior del conductor)
    r ≥ R:  E = E₀ + E_dipolo(p)          (aplicado + respuesta inducida)

La superficie queda equipotencial. Esta es la solución analítica exacta del
problema de contorno; no hay ajuste numérico.

A diferencia de las fuentes de `charges/`, un conductor no genera campo por sí
solo: responde al campo externo `E0` que se le pasa. Por eso NO cumple el
protocolo `ChargeSource` de forma autónoma; expone `total_field` para componer
la escena.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import eps0
from ...core.base import PhysicalObject


class ConductingSphere(PhysicalObject):
    """Esfera conductora de radio `radius` en un campo uniforme `E0`.

    Args:
        radius: radio de la esfera R [m], > 0.
        E0: campo eléctrico uniforme externo aplicado [V/m], vector (3,).
        position: centro de la esfera [m]. Por defecto, el origen.
    """

    def __init__(
        self,
        radius: float,
        E0: np.ndarray,
        position: np.ndarray | None = None,
    ) -> None:
        super().__init__(position=position)
        if radius <= 0:
            raise ValueError(f"radius debe ser positivo, recibido {radius}")
        self.radius = float(radius)
        self.E0 = np.asarray(E0, dtype=float)
        if self.E0.shape != (3,):
            raise ValueError(f"E0 debe tener forma (3,), no {self.E0.shape}")

    @property
    def induced_moment(self) -> np.ndarray:
        """Momento dipolar inducido p = 4π ε₀ R³ E₀ [C·m]."""
        return 4 * np.pi * eps0 * self.radius**3 * self.E0

    def total_field(self, points: np.ndarray) -> np.ndarray:
        """Campo total (aplicado + inducido) [V/m]. (N,3) -> (N,3).

        Cero dentro de la esfera; E₀ más el dipolo inducido fuera.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.position                       # (N,3)
        r = np.linalg.norm(rel, axis=1)                    # (N,)

        out = np.zeros_like(points)
        outside = r >= self.radius
        if not np.any(outside):
            return out

        ro = rel[outside]
        rn = r[outside]
        p = self.induced_moment

        # Campo dipolar: E = [3(p·r̂)r̂ - p] / (4πε₀ r³)
        r_hat = ro / rn[:, None]
        p_dot = r_hat @ p                                   # (M,)
        E_dip = (3 * p_dot[:, None] * r_hat - p[None, :]) / (
            4 * np.pi * eps0 * rn[:, None] ** 3
        )
        out[outside] = self.E0[None, :] + E_dip
        return out

    def induced_field(self, points: np.ndarray) -> np.ndarray:
        """Solo el campo del dipolo inducido (sin E₀) [V/m]. (N,3) -> (N,3).

        Útil para visualizar por separado la respuesta del conductor.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.position
        r = np.linalg.norm(rel, axis=1)
        out = np.zeros_like(points)
        outside = r >= self.radius
        if not np.any(outside):
            return out
        ro, rn = rel[outside], r[outside]
        p = self.induced_moment
        r_hat = ro / rn[:, None]
        p_dot = r_hat @ p
        out[outside] = (3 * p_dot[:, None] * r_hat - p[None, :]) / (
            4 * np.pi * eps0 * rn[:, None] ** 3
        )
        return out

    def surface_charge_density(self, theta: np.ndarray) -> np.ndarray:
        """Densidad de carga inducida en la superficie σ(θ) = 3 ε₀ |E₀| cosθ.

        θ se mide desde la dirección de E₀. [C/m²].
        """
        theta = np.asarray(theta, dtype=float)
        return 3 * eps0 * np.linalg.norm(self.E0) * np.cos(theta)

    def __repr__(self) -> str:
        return (
            f"ConductingSphere(radius={self.radius:g}, "
            f"E0={self.E0.tolist()}, position={self.position.tolist()})"
        )


__all__ = ["ConductingSphere"]
