"""
Conversión entre unidades físicas (SI) y unidades de escena de Manim.

La física del paquete trabaja siempre en metros. Manim dibuja en su propio
sistema donde el ancho de pantalla es ~14 unidades. `UnitScale` traduce
posiciones y longitudes entre ambos mundos sin que la física conozca a Manim.

Solo se escalan magnitudes espaciales (posiciones, longitudes). Campos, cargas
y corrientes conservan sus unidades SI; su representación visual se ajusta en
la capa `fields/`, no aquí.
"""

from __future__ import annotations

import numpy as np


class UnitScale:
    """Escala lineal entre metros (SI) y unidades de escena de Manim.

    La relación es: u_manim = m_si / meters_per_unit

    donde `meters_per_unit` indica cuántos metros representa una unidad de
    Manim. Con el valor por defecto (1.0), un metro equivale a una unidad
    de escena.

    Args:
        meters_per_unit: Metros que representa cada unidad de Manim.
            Debe ser estrictamente positivo.

    Ejemplo:
        >>> scale = UnitScale(meters_per_unit=0.5)  # 1 u_manim = 0.5 m
        >>> scale.to_scene(1.0)                      # 1 m -> 2 u_manim
        2.0
        >>> scale.to_si(2.0)                         # 2 u_manim -> 1 m
        1.0
    """

    def __init__(self, meters_per_unit: float = 1.0) -> None:
        if meters_per_unit <= 0:
            raise ValueError(
                f"meters_per_unit debe ser positivo, recibido {meters_per_unit}"
            )
        self.meters_per_unit: float = float(meters_per_unit)

    @property
    def units_per_meter(self) -> float:
        """Factor inverso: unidades de Manim por metro."""
        return 1.0 / self.meters_per_unit

    def to_scene(self, si_value: np.ndarray | float) -> np.ndarray | float:
        """Convierte una magnitud espacial de metros a unidades de Manim.

        Acepta escalares o arrays de numpy de cualquier forma (p. ej. `(N, 3)`
        para conjuntos de posiciones). No modifica el argumento de entrada.
        """
        return np.asarray(si_value) / self.meters_per_unit if isinstance(
            si_value, np.ndarray
        ) else si_value / self.meters_per_unit

    def to_si(self, scene_value: np.ndarray | float) -> np.ndarray | float:
        """Convierte una magnitud espacial de unidades de Manim a metros."""
        return np.asarray(scene_value) * self.meters_per_unit if isinstance(
            scene_value, np.ndarray
        ) else scene_value * self.meters_per_unit

    def __repr__(self) -> str:
        return f"UnitScale(meters_per_unit={self.meters_per_unit})"


#: Escala por defecto del paquete: 1 metro = 1 unidad de Manim.
DEFAULT_SCALE = UnitScale(meters_per_unit=1.0)


__all__ = ["UnitScale", "DEFAULT_SCALE"]
