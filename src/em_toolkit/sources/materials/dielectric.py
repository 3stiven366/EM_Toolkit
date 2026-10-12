"""
Esfera dieléctrica en un campo eléctrico uniforme externo.

Una esfera dieléctrica lineal de constante κ (épsilon relativa) colocada en un
campo uniforme E₀ se polariza. A diferencia del conductor, el campo interior no
se anula: se reduce y permanece uniforme.

Solución analítica (origen en el centro de la esfera):
    r < R:  E = E₀ · 3 / (κ + 2)                       (uniforme, reducido)
    r ≥ R:  E = E₀ + E_dipolo(p)
            con  p = 4π ε₀ R³ (κ - 1)/(κ + 2) · E₀      (dipolo inducido)

Límites de consistencia:
    κ → 1  : sin material, E = E₀ en todas partes (p → 0).
    κ → ∞  : se recupera la esfera conductora (E_in → 0, p → 4πε₀R³E₀).

Como el conductor, un dieléctrico responde al campo externo `E0` que se le pasa;
expone `total_field` para componer la escena.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import eps0
from ...core.base import PhysicalObject


class DielectricSphere(PhysicalObject):
    """Esfera dieléctrica de radio `radius` y constante `kappa` en campo `E0`.

    Args:
        radius: radio de la esfera R [m], > 0.
        kappa: constante dieléctrica relativa κ, ≥ 1.
        E0: campo eléctrico uniforme externo aplicado [V/m], vector (3,).
        position: centro de la esfera [m]. Por defecto, el origen.
    """

    def __init__(
        self,
        radius: float,
        kappa: float,
        E0: np.ndarray,
        position: np.ndarray | None = None,
    ) -> None:
        super().__init__(position=position)
        if radius <= 0:
            raise ValueError(f"radius debe ser positivo, recibido {radius}")
        if kappa < 1:
            raise ValueError(f"kappa debe ser >= 1, recibido {kappa}")
        self.radius = float(radius)
        self.kappa = float(kappa)
        self.E0 = np.asarray(E0, dtype=float)
        if self.E0.shape != (3,):
            raise ValueError(f"E0 debe tener forma (3,), no {self.E0.shape}")

    @property
    def _factor(self) -> float:
        """Factor de respuesta (κ-1)/(κ+2) del dipolo inducido."""
        return (self.kappa - 1) / (self.kappa + 2)

    @property
    def interior_field(self) -> np.ndarray:
        """Campo uniforme dentro de la esfera E_in = E₀·3/(κ+2) [V/m]."""
        return self.E0 * 3.0 / (self.kappa + 2)

    @property
    def induced_moment(self) -> np.ndarray:
        """Momento dipolar inducido p = 4πε₀R³·(κ-1)/(κ+2)·E₀ [C·m]."""
        return 4 * np.pi * eps0 * self.radius**3 * self._factor * self.E0

    def total_field(self, points: np.ndarray) -> np.ndarray:
        """Campo total [V/m]. (N,3) -> (N,3).

        Interior uniforme reducido; exterior E₀ más dipolo inducido.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.position
        r = np.linalg.norm(rel, axis=1)

        out = np.empty_like(points)
        inside = r < self.radius
        outside = ~inside

        if np.any(inside):
            out[inside] = self.interior_field

        if np.any(outside):
            ro, rn = rel[outside], r[outside]
            p = self.induced_moment
            r_hat = ro / rn[:, None]
            p_dot = r_hat @ p
            E_dip = (3 * p_dot[:, None] * r_hat - p[None, :]) / (
                4 * np.pi * eps0 * rn[:, None] ** 3
            )
            out[outside] = self.E0[None, :] + E_dip
        return out

    def polarization(self) -> np.ndarray:
        """Polarización uniforme del material P = ε₀(κ-1)E_in [C/m²]."""
        return eps0 * (self.kappa - 1) * self.interior_field

    def __repr__(self) -> str:
        return (
            f"DielectricSphere(radius={self.radius:g}, kappa={self.kappa:g}, "
            f"E0={self.E0.tolist()}, position={self.position.tolist()})"
        )


__all__ = ["DielectricSphere"]
