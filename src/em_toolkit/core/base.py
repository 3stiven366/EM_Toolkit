"""
Base común de los objetos físicos con posición y orientación.

`PhysicalObject` integra un `Frame` y ofrece la interfaz para colocar y
orientar cualquier fuente en ℝ³. Las fuentes concretas (cargas, corrientes)
heredan de aquí para no reimplementar el posicionamiento.

Heredar de esta clase es opcional: el `EMWorld` reconoce las fuentes por sus
protocolos (tipado estructural), no por esta base. `PhysicalObject` es una
comodidad, no un requisito.
"""

from __future__ import annotations

import numpy as np

from .frame import Frame


class PhysicalObject:
    """Objeto ubicable y orientable en el espacio.

    Args:
        position: Posición del origen del objeto [m]. Forma (3,). Por defecto,
            el origen global.
        frame: Marco de referencia explícito. Si se da, tiene prioridad sobre
            `position` (que se ignora).
    """

    def __init__(
        self,
        position: np.ndarray | None = None,
        frame: Frame | None = None,
    ) -> None:
        if frame is not None:
            self.frame = frame
        else:
            self.frame = Frame(origin=position)

    @property
    def position(self) -> np.ndarray:
        """Posición del objeto en coordenadas globales [m]."""
        return self.frame.origin

    def move_to(self, position: np.ndarray) -> "PhysicalObject":
        """Traslada el objeto a una posición global, conservando su orientación."""
        self.frame = Frame(
            origin=np.asarray(position, dtype=float),
            rotation=self.frame.rotation,
        )
        return self

    def orient(self, frame: Frame) -> "PhysicalObject":
        """Reemplaza el marco completo (posición y orientación)."""
        self.frame = frame
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}(position={self.position.tolist()})"


__all__ = ["PhysicalObject"]
