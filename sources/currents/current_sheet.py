"""
Lámina de corriente infinita.

Plano infinito por el que fluye una corriente superficial de densidad K [A/m]
en una dirección contenida en el plano. Es el análogo magnético del plano
cargado: el campo es uniforme, paralelo a la lámina y de sentido opuesto a cada
lado.

    B = μ₀ K / 2     (magnitud constante a cada lado)

La dirección de B es perpendicular a la corriente y paralela a la lámina, dada
por B = ±(μ₀/2) K × n̂, con n̂ la normal del lado correspondiente (regla de la
mano derecha). Solución analítica: no hay discretización.
"""

from __future__ import annotations

import numpy as np

from ...config.constants import mu0
from ...core.base import PhysicalObject


class CurrentSheet(PhysicalObject):
    """Lámina de corriente infinita con densidad superficial `k`.

    Args:
        k: magnitud de la densidad superficial de corriente K [A/m], > 0.
        position: un punto del plano [m]. Por defecto, el origen.
        normal: normal al plano de la lámina. Por defecto +z.
        current_direction: dirección del flujo de corriente (en el plano).
            Por defecto +x. Se proyecta al plano si tiene componente normal.
    """

    def __init__(
        self,
        k: float,
        position: np.ndarray | None = None,
        normal: np.ndarray | None = None,
        current_direction: np.ndarray | None = None,
    ) -> None:
        super().__init__(position=position)
        if k <= 0:
            raise ValueError(f"k debe ser positiva (magnitud), recibido {k}")
        self.k = float(k)

        normal = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float)
        if np.linalg.norm(normal) < 1e-12:
            raise ValueError("normal no puede ser el vector nulo")
        self.normal = normal / np.linalg.norm(normal)

        cd = (
            np.array([1.0, 0.0, 0.0]) if current_direction is None
            else np.asarray(current_direction, float)
        )
        cd = cd - np.dot(cd, self.normal) * self.normal   # proyectar al plano
        if np.linalg.norm(cd) < 1e-12:
            raise ValueError("current_direction no puede ser paralela a normal")
        self.current_direction = cd / np.linalg.norm(cd)

        #: Vector densidad de corriente K = k · dirección [A/m].
        self.K = self.k * self.current_direction

    @property
    def current(self) -> float:
        """Magnitud de la densidad de corriente (para el protocolo CurrentSource)."""
        return self.k

    def _signed_side(self, points: np.ndarray) -> np.ndarray:
        """Signo del lado de cada punto respecto al plano (+1, -1, 0)."""
        return np.sign((points - self.position) @ self.normal)

    # -- Protocolo CurrentSource --------------------------------------------

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Campo B uniforme [T]. (N,3) -> (N,3).

        A cada lado B = (μ₀/2)·side·(K × n̂). Sobre la lámina (d = 0) se devuelve
        cero por convención.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        side = self._signed_side(points)                       # (N,)
        base = 0.5 * mu0 * np.cross(self.K, self.normal)       # (3,)
        return side[:, None] * base[None, :]

    def __repr__(self) -> str:
        return (
            f"CurrentSheet(k={self.k:g}, normal={self.normal.tolist()}, "
            f"current_direction={self.current_direction.tolist()})"
        )


__all__ = ["CurrentSheet"]
