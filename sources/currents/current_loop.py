"""
Espira circular de corriente.

Anillo de radio R por el que circula una corriente I. Diseño híbrido:

- En el eje: solución analítica exacta,
      B_z(z) = μ₀ I R² / [2 (R² + z²)^{3/2}]
- Fuera del eje: motor numérico de Biot-Savart (`CurrentPath`).

La espira se define en el plano perpendicular a `axis`, centrada en `position`.
El sentido de I y la dirección de B siguen la regla de la mano derecha respecto
a `axis`. Es la base del solenoide (una pila de espiras).
"""

from __future__ import annotations

import numpy as np

from ...config.constants import mu0
from ...core.frame import Frame
from .path import CurrentPath


class CurrentLoop(CurrentPath):
    """Espira circular de radio `radius` con corriente `current`.

    Args:
        current: corriente I [A]. El signo invierte el sentido (y B).
        radius: radio de la espira R [m], > 0.
        position: centro de la espira [m]. Por defecto, el origen.
        axis: normal al plano de la espira (dirección de B en el centro para
            I > 0). Por defecto +z.
        n_segments: subdivisiones para el cálculo fuera del eje.
        r_min: softening [m].
    """

    def __init__(
        self,
        current: float,
        radius: float,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        n_segments: int = 400,
        r_min: float | None = None,
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius debe ser positivo, recibido {radius}")
        super().__init__(
            current,
            position=position,
            t_range=(0.0, 2 * np.pi),
            n_segments=n_segments,
            closed=True,
            r_min=r_min,
        )
        self.radius = float(radius)

        axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / np.linalg.norm(axis)
        self._frame = Frame.from_z_to(self.axis, origin=self.position)

    def path(self, t: np.ndarray) -> np.ndarray:
        """Círculo de radio R en el plano perpendicular a axis. t ∈ [0, 2π]."""
        t = np.asarray(t, dtype=float)
        local = np.stack(
            [self.radius * np.cos(t), self.radius * np.sin(t), np.zeros_like(t)],
            axis=-1,
        )
        return self._frame.to_global(local)

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Campo B [T]. Fórmula analítica en el eje; numérico fuera. (N,3)->(N,3)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        rel = points - self.position
        z = rel @ self.axis                                 # coord axial (N,)
        perp = rel - z[:, None] * self.axis
        s = np.linalg.norm(perp, axis=1)                    # dist. al eje (N,)

        on_axis = s < 1e-9
        result = np.empty_like(points)

        # En el eje: analítico
        if np.any(on_axis):
            za = z[on_axis]
            Bz = mu0 * self.current * self.radius**2 / (
                2 * (self.radius**2 + za**2) ** 1.5
            )
            result[on_axis] = Bz[:, None] * self.axis

        # Fuera del eje: numérico
        if np.any(~on_axis):
            result[~on_axis] = super().magnetic_field(points[~on_axis])

        return result

    @property
    def moment(self) -> np.ndarray:
        """Momento dipolar magnético m = I·A·n̂ = I·πR²·axis [A·m²]."""
        return self.current * np.pi * self.radius**2 * self.axis

    def __repr__(self) -> str:
        return (
            f"CurrentLoop(current={self.current:g}, radius={self.radius:g}, "
            f"position={self.position.tolist()})"
        )


__all__ = ["CurrentLoop"]
