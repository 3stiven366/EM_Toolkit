"""
Segmento recto finito de corriente.

Corriente I por un segmento recto entre dos puntos. Es la pieza básica para
construir circuitos y caminos poligonales. El campo se obtiene por el motor de
Biot-Savart de `CurrentPath` (discretización).

Solución analítica de referencia (para validación):
    B = μ₀ I / (4π s) · (sinθ₂ - sinθ₁)
donde s es la distancia perpendicular al segmento y θ₁, θ₂ los ángulos desde
la perpendicular a cada extremo.
"""

from __future__ import annotations

import numpy as np

from .path import CurrentPath


class WireSegment(CurrentPath):
    """Segmento recto de corriente entre `start` y `end`.

    Args:
        current: corriente I [A]. El signo va de start a end (+) o al revés (-).
        start: punto inicial del segmento [m].
        end: punto final del segmento [m].
        n_segments: número de subsegmentos para la integración.
        r_min: softening [m].
    """

    def __init__(
        self,
        current: float,
        start: np.ndarray,
        end: np.ndarray,
        n_segments: int = 200,
        r_min: float | None = None,
    ) -> None:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        if np.linalg.norm(end - start) < 1e-12:
            raise ValueError("start y end no pueden coincidir")

        # Posición de referencia: punto medio del segmento
        super().__init__(
            current,
            position=0.5 * (start + end),
            t_range=(0.0, 1.0),
            n_segments=n_segments,
            closed=False,
            r_min=r_min,
        )
        self.start = start
        self.end = end

    def path(self, t: np.ndarray) -> np.ndarray:
        """Recta de start a end parametrizada por t ∈ [0, 1]."""
        t = np.asarray(t, dtype=float)
        return self.start + t[:, None] * (self.end - self.start)

    @property
    def length(self) -> float:
        """Longitud del segmento [m]."""
        return float(np.linalg.norm(self.end - self.start))

    def __repr__(self) -> str:
        return (
            f"WireSegment(current={self.current:g}, "
            f"start={self.start.tolist()}, end={self.end.tolist()})"
        )


__all__ = ["WireSegment"]
