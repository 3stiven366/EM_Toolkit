"""
Solenoide: hélice de N vueltas de corriente.

Modela un solenoide real como una hélice de radio R, longitud L y N vueltas,
recorrida por una corriente I. El campo se obtiene por el motor de Biot-Savart
de `CurrentPath` (la hélice es su `path(t)`), así que es válido dentro, fuera y
en los extremos, incluyendo los efectos de borde reales.

Referencia ideal (para validación), solenoide largo y denso:
    interior:  B = μ₀ n I     (axial y uniforme; n = N/L vueltas por metro)
    exterior:  B ≈ 0

El eje del solenoide es `axis`, centrado en `position`. Es la base de tu
ejemplo: un cilindro con N vueltas de cable encima.
"""

from __future__ import annotations

import numpy as np

from ...core.frame import Frame
from .path import CurrentPath


class Solenoid(CurrentPath):
    """Solenoide helicoidal de `n_turns` vueltas.

    Args:
        current: corriente I [A]. El signo invierte el sentido (y B).
        radius: radio del solenoide R [m], > 0.
        length: longitud L [m], > 0.
        n_turns: número de vueltas N.
        position: centro del solenoide [m]. Por defecto, el origen.
        axis: dirección del eje. Por defecto +z.
        segments_per_turn: subdivisiones por vuelta para la integración.
        r_min: softening [m].
    """

    def __init__(
        self,
        current: float,
        radius: float,
        length: float,
        n_turns: int,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        segments_per_turn: int = 60,
        r_min: float | None = None,
    ) -> None:
        if radius <= 0:
            raise ValueError(f"radius debe ser positivo, recibido {radius}")
        if length <= 0:
            raise ValueError(f"length debe ser positiva, recibido {length}")
        if n_turns <= 0:
            raise ValueError(f"n_turns debe ser positivo, recibido {n_turns}")

        n_segments = int(segments_per_turn * n_turns)
        super().__init__(
            current,
            position=position,
            t_range=(0.0, 1.0),
            n_segments=n_segments,
            closed=False,
            r_min=r_min,
        )
        self.radius = float(radius)
        self.length = float(length)
        self.n_turns = int(n_turns)

        axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / np.linalg.norm(axis)
        self._frame = Frame.from_z_to(self.axis, origin=self.position)

    @property
    def turns_per_length(self) -> float:
        """Densidad de vueltas n = N / L [vueltas/m]."""
        return self.n_turns / self.length

    def path(self, t: np.ndarray) -> np.ndarray:
        """Hélice: t ∈ [0,1] recorre las N vueltas de un extremo al otro."""
        t = np.asarray(t, dtype=float)
        phi = 2 * np.pi * self.n_turns * t              # ángulo azimutal
        z = self.length * (t - 0.5)                     # avance axial, centrado
        local = np.stack(
            [self.radius * np.cos(phi), self.radius * np.sin(phi), z],
            axis=-1,
        )
        return self._frame.to_global(local)

    def __repr__(self) -> str:
        return (
            f"Solenoid(current={self.current:g}, radius={self.radius:g}, "
            f"length={self.length:g}, n_turns={self.n_turns})"
        )


__all__ = ["Solenoid"]
