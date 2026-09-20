"""
Línea cargada con densidad lineal uniforme λ.

Dos regímenes:

- Infinita: solución analítica. El campo es radial al eje,
      E = λ / (2π ε₀ s)      (s = distancia perpendicular al eje)
  El potencial de un hilo infinito diverge, así que se mide respecto a una
  distancia de referencia s_ref:
      V = -λ / (2π ε₀) · ln(s / s_ref)

- Finita: se discretiza en cargas puntuales y se superpone (usa coulomb).

Ambos casos comparten un eje definido por un punto (`position`) y una dirección
(`axis`).
"""

from __future__ import annotations

import numpy as np

from ...config.constants import eps0
from ...core.base import PhysicalObject
from ...physics import coulomb


class LineCharge(PhysicalObject):
    """Línea recta con densidad de carga lineal `lambda_`.

    Args:
        lambda_: densidad lineal de carga λ [C/m].
        position: un punto sobre el eje [m]. Para línea finita, su centro.
        axis: dirección del eje. Por defecto +z.
        length: longitud [m]. Requerida si `infinite=False`.
        infinite: si True, hilo infinito (solución analítica).
        s_ref: distancia de referencia del potencial [m] (solo caso infinito).
        n_segments: número de cargas puntuales para el caso finito.
        r_min: softening [m].
    """

    def __init__(
        self,
        lambda_: float,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        length: float | None = None,
        infinite: bool = False,
        s_ref: float = 1.0,
        n_segments: int = 200,
        r_min: float | None = None,
    ) -> None:
        super().__init__(position=position)
        self.lambda_ = float(lambda_)
        self.infinite = infinite
        self.s_ref = float(s_ref)
        self.n_segments = int(n_segments)
        self.r_min = r_min

        axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / norm

        if not infinite:
            if length is None or length <= 0:
                raise ValueError("length debe ser positiva para una línea finita")
            self.length = float(length)
            self._build_discretization()
        else:
            self.length = np.inf

    def _r_min(self) -> float:
        from ...config.defaults import FIELD_DEFAULTS
        return FIELD_DEFAULTS["r_min"] if self.r_min is None else self.r_min

    def _build_discretization(self) -> None:
        """Reparte la carga total en n cargas puntuales sobre el eje."""
        s = np.linspace(-self.length / 2, self.length / 2, self.n_segments)
        self._sources = self.position + s[:, None] * self.axis   # (n, 3)
        dq = self.lambda_ * (self.length / self.n_segments)
        self._charges = np.full(self.n_segments, dq)             # (n,)

    def _perp_distance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vector y distancia perpendicular de cada punto al eje infinito."""
        rel = points - self.position                  # (N,3)
        along = (rel @ self.axis)[:, None] * self.axis  # componente paralela
        perp = rel - along                            # componente perpendicular
        s = np.linalg.norm(perp, axis=1)              # (N,)
        return perp, s

    # -- Protocolo ChargeSource ---------------------------------------------

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo E [V/m]. (N,3) -> (N,3)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self.infinite:
            perp, s = self._perp_distance(points)
            s_eff = np.sqrt(s**2 + self._r_min() ** 2)
            magnitude = self.lambda_ / (2 * np.pi * eps0 * s_eff)  # (N,)
            with np.errstate(invalid="ignore", divide="ignore"):
                direction = perp / s[:, None]
            direction = np.nan_to_num(direction)   # sobre el eje: dir indefinida -> 0
            return magnitude[:, None] * direction
        return coulomb.electric_field(
            points, self._sources, self._charges, r_min=self._r_min()
        )

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial V [V]. (N,3) -> (N,)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self.infinite:
            _, s = self._perp_distance(points)
            s_eff = np.sqrt(s**2 + self._r_min() ** 2)
            return -self.lambda_ / (2 * np.pi * eps0) * np.log(s_eff / self.s_ref)
        return coulomb.potential(
            points, self._sources, self._charges, r_min=self._r_min()
        )

    def total_charge(self) -> float:
        """Carga total [C]. Infinita para el hilo infinito."""
        return np.inf if self.infinite else self.lambda_ * self.length

    def __repr__(self) -> str:
        tipo = "infinite" if self.infinite else f"length={self.length:g}"
        return f"LineCharge(lambda={self.lambda_:g}, {tipo})"


__all__ = ["LineCharge"]
