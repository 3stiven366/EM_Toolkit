"""
Toroide: bobinado de N vueltas sobre un núcleo toroidal.

Un toroide es un solenoide curvado hasta cerrarse sobre sí mismo. El campo queda
confinado dentro del núcleo, es circunferencial (sigue el dónut) y su magnitud
decae con el radio:

    dentro:  B = μ₀ N I / (2π r)     (r = distancia al eje central del toroide)
    fuera:   B ≈ 0

El toroide se caracteriza por su radio mayor R (del centro al centro del tubo) y
su radio menor a (del tubo). Su `path(t)` es una hélice enrollada sobre el
círculo mayor; el campo sale del motor de Biot-Savart de `CurrentPath`.

El eje central del toroide es `axis`, centrado en `position`.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import mu0
from ...core.frame import Frame
from .path import CurrentPath


class Toroid(CurrentPath):
    """Toroide de `n_turns` vueltas sobre un núcleo de dónut.

    Args:
        current: corriente I [A]. El signo invierte el sentido (y B).
        major_radius: radio mayor R (eje central → centro del tubo) [m], > 0.
        minor_radius: radio menor a (radio del tubo) [m], 0 < a < R.
        n_turns: número de vueltas N.
        position: centro del toroide [m]. Por defecto, el origen.
        axis: eje central del toroide. Por defecto +z.
        segments_per_turn: subdivisiones por vuelta.
        r_min: softening [m].
    """

    def __init__(
        self,
        current: float,
        major_radius: float,
        minor_radius: float,
        n_turns: int,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        segments_per_turn: int = 40,
        r_min: float | None = None,
    ) -> None:
        if major_radius <= 0:
            raise ValueError(f"major_radius debe ser positivo, recibido {major_radius}")
        if not (0 < minor_radius < major_radius):
            raise ValueError("minor_radius debe cumplir 0 < a < R (major_radius)")
        if n_turns <= 0:
            raise ValueError(f"n_turns debe ser positivo, recibido {n_turns}")

        n_segments = int(segments_per_turn * n_turns)
        super().__init__(
            current,
            position=position,
            t_range=(0.0, 1.0),
            n_segments=n_segments,
            closed=True,
            r_min=r_min,
        )
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.n_turns = int(n_turns)

        axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / np.linalg.norm(axis)
        self._frame = Frame.from_z_to(self.axis, origin=self.position)

    def path(self, t: np.ndarray) -> np.ndarray:
        """Hélice sobre el círculo mayor. t ∈ [0,1] da una vuelta completa al dónut."""
        t = np.asarray(t, dtype=float)
        theta = 2 * np.pi * t                        # posición a lo largo del dónut
        phi = 2 * np.pi * self.n_turns * t           # enrollado alrededor del tubo

        R, a = self.major_radius, self.minor_radius
        rho = R + a * np.cos(phi)                     # distancia al eje central
        local = np.stack(
            [rho * np.cos(theta), rho * np.sin(theta), a * np.sin(phi)],
            axis=-1,
        )
        return self._frame.to_global(local)

    def __repr__(self) -> str:
        return (
            f"Toroid(current={self.current:g}, major_radius={self.major_radius:g}, "
            f"minor_radius={self.minor_radius:g}, n_turns={self.n_turns})"
        )


__all__ = ["Toroid"]
