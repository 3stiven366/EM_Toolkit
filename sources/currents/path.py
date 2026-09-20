"""
Camino de corriente: base de todas las fuentes magnéticas.

`CurrentPath` es el motor universal para el campo magnético. Una subclase solo
define su curva parametrizada `path(t)`; el campo B sale automáticamente por
integración de Biot-Savart. Las geometrías con solución analítica (hilo
infinito, espira en el eje) sobreescriben `magnetic_field` por exactitud y
velocidad.

Cumple el protocolo `CurrentSource`: tiene atributo `current` y método
`magnetic_field`, por lo que el `EMWorld` la reconoce como fuente de B.
"""

from __future__ import annotations

import numpy as np

from ...core.base import PhysicalObject
from ...physics import biot_savart


class CurrentPath(PhysicalObject):
    """Corriente que circula por un camino parametrizado.

    Las subclases implementan `path(t)`, que devuelve los puntos del camino en
    coordenadas globales para un array de parámetros `t`. El resto (campo B) se
    obtiene por Biot-Savart sin fórmula cerrada.

    Args:
        current: corriente I [A]. El signo indica el sentido de circulación.
        position: posición de referencia del objeto [m].
        t_range: intervalo (t0, t1) del parámetro del camino.
        n_segments: número de segmentos para discretizar el camino.
        closed: si el camino es un lazo cerrado.
        r_min: softening del denominador de Biot-Savart [m].
    """

    def __init__(
        self,
        current: float,
        position: np.ndarray | None = None,
        t_range: tuple[float, float] = (0.0, 1.0),
        n_segments: int = 400,
        closed: bool = False,
        r_min: float | None = None,
    ) -> None:
        super().__init__(position=position)
        self.current = float(current)
        self.t_range = t_range
        self.n_segments = int(n_segments)
        self.closed = closed
        self.r_min = r_min

    def _r_min(self) -> float:
        from ...config.defaults import FIELD_DEFAULTS
        return FIELD_DEFAULTS["r_min"] if self.r_min is None else self.r_min

    def path(self, t: np.ndarray) -> np.ndarray:
        """Puntos del camino en coordenadas globales. t:(K,) -> (K,3).

        Debe implementarlo cada subclase.
        """
        raise NotImplementedError(
            f"{type(self).__name__} debe implementar path(t)."
        )

    def path_points(self) -> np.ndarray:
        """Discretización del camino en vértices, según t_range y n_segments."""
        t = np.linspace(self.t_range[0], self.t_range[1], self.n_segments + 1)
        return np.asarray(self.path(t), dtype=float)

    # -- Protocolo CurrentSource --------------------------------------------

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Campo B por Biot-Savart sobre el camino discretizado. (N,3)->(N,3) [T]."""
        return biot_savart.biot_savart_discrete(
            points,
            self.path_points(),
            self.current,
            r_min=self._r_min(),
            closed=self.closed,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(current={self.current:g})"


__all__ = ["CurrentPath"]
