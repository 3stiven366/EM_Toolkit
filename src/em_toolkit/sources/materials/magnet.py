"""
Imán de barra como dipolo magnético.

Un imán permanente es, a efectos de su campo externo, equivalente a una espira
de corriente con el mismo momento dipolar magnético m. Este objeto modela el
imán mediante una `CurrentLoop` equivalente que cumple I·πR² = m, por lo que su
campo externo es el de un dipolo magnético correcto.

El momento m apunta de S a N. Los extremos `north_pole` y `south_pole` se
exponen para la representación visual.

Limitación: el campo cercano de un imán real no es idéntico al de una espira
(la magnetización está distribuida en el volumen). El modelo es fiel en el campo
externo / lejano, que es lo relevante para la mayoría de las animaciones.
"""

from __future__ import annotations

import numpy as np

from ...core.base import PhysicalObject
from ..currents.current_loop import CurrentLoop


class BarMagnet(PhysicalObject):
    """Imán de barra caracterizado por su momento magnético.

    Args:
        moment: magnitud del momento dipolar magnético m [A·m²], > 0.
        length: longitud física del imán [m], > 0 (solo geometría visual).
        position: centro del imán [m]. Por defecto, el origen.
        axis: dirección del momento (de S a N). Por defecto +z.
        loop_radius: radio de la espira equivalente [m]. Pequeño frente a las
            distancias de interés para un buen comportamiento dipolar.
        r_min: softening [m].
    """

    def __init__(
        self,
        moment: float,
        length: float,
        position: np.ndarray | None = None,
        axis: np.ndarray | None = None,
        loop_radius: float = 0.05,
        r_min: float | None = None,
    ) -> None:
        super().__init__(position=position)
        if moment <= 0:
            raise ValueError(f"moment debe ser positivo (magnitud), recibido {moment}")
        if length <= 0:
            raise ValueError(f"length debe ser positiva, recibido {length}")
        if loop_radius <= 0:
            raise ValueError(f"loop_radius debe ser positivo, recibido {loop_radius}")

        self.moment_magnitude = float(moment)
        self.length = float(length)
        self.loop_radius = float(loop_radius)

        axis = np.array([0.0, 0.0, 1.0]) if axis is None else np.asarray(axis, float)
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError("axis no puede ser el vector nulo")
        self.axis = axis / np.linalg.norm(axis)

        # Espira equivalente: I tal que m = I·πR²
        current = self.moment_magnitude / (np.pi * self.loop_radius**2)
        self._loop = CurrentLoop(
            current,
            radius=self.loop_radius,
            position=self.position,
            axis=self.axis,
            r_min=r_min,
        )

    @property
    def current(self) -> float:
        """Corriente de la espira equivalente (para el protocolo CurrentSource)."""
        return self._loop.current

    @property
    def moment(self) -> np.ndarray:
        """Momento dipolar magnético m = m·axis, de S a N [A·m²]."""
        return self.moment_magnitude * self.axis

    @property
    def north_pole(self) -> np.ndarray:
        """Posición del polo norte [m]."""
        return self.position + self.axis * (self.length / 2)

    @property
    def south_pole(self) -> np.ndarray:
        """Posición del polo sur [m]."""
        return self.position - self.axis * (self.length / 2)

    # -- Protocolo CurrentSource --------------------------------------------

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Campo B del imán (dipolo magnético) [T]. (N,3) -> (N,3)."""
        return self._loop.magnetic_field(points)

    def __repr__(self) -> str:
        return (
            f"BarMagnet(moment={self.moment_magnitude:g}, length={self.length:g}, "
            f"position={self.position.tolist()})"
        )


__all__ = ["BarMagnet"]
