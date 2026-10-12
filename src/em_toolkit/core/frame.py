"""
Marco de referencia local para orientar y posicionar objetos en ℝ³.

Un `Frame` guarda el origen local (en coordenadas globales) y una matriz de
rotación 3×3. Permite trasladar puntos y vectores entre el marco local del
objeto y el marco global del mundo.

Uso típico: un objeto (solenoide, dipolo) define su física en un marco local
cómodo —por ejemplo con el eje a lo largo de +z— y el `Frame` lo coloca y
orienta en cualquier posición del espacio. La distinción clave:

    - punto:  se rota y se traslada     (posición absoluta)
    - vector: solo se rota              (dirección, sin origen)
"""

from __future__ import annotations

import numpy as np

from .exceptions import PhysicsError

_EPS = 1e-12


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < _EPS:
        raise PhysicsError("No se puede normalizar un vector nulo.")
    return v / n


class Frame:
    """Marco local: rotación 3×3 más traslación en ℝ³.

    Args:
        origin: Origen local expresado en coordenadas globales [m]. Forma (3,).
        rotation: Matriz de rotación 3×3. Sus columnas son los ejes locales
            expresados en la base global. Por defecto, la identidad.
    """

    def __init__(
        self,
        origin: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ) -> None:
        self.origin = (
            np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
        )
        self.rotation = (
            np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        )
        if self.origin.shape != (3,):
            raise PhysicsError(f"origin debe tener forma (3,), no {self.origin.shape}")
        if self.rotation.shape != (3, 3):
            raise PhysicsError(
                f"rotation debe tener forma (3,3), no {self.rotation.shape}"
            )

    # -- Transformación de puntos (rota + traslada) --------------------------

    def to_global(self, local_points: np.ndarray) -> np.ndarray:
        """Lleva puntos del marco local al global. (...,3) -> (...,3)."""
        p = np.asarray(local_points, dtype=float)
        return p @ self.rotation.T + self.origin

    def to_local(self, global_points: np.ndarray) -> np.ndarray:
        """Lleva puntos del marco global al local. (...,3) -> (...,3)."""
        p = np.asarray(global_points, dtype=float)
        return (p - self.origin) @ self.rotation

    # -- Transformación de vectores (solo rota) ------------------------------

    def rotate_to_global(self, local_vectors: np.ndarray) -> np.ndarray:
        """Rota vectores del marco local al global, sin trasladar."""
        return np.asarray(local_vectors, dtype=float) @ self.rotation.T

    def rotate_to_local(self, global_vectors: np.ndarray) -> np.ndarray:
        """Rota vectores del marco global al local, sin trasladar."""
        return np.asarray(global_vectors, dtype=float) @ self.rotation

    # -- Constructores de conveniencia ---------------------------------------

    @classmethod
    def from_axis_angle(
        cls,
        axis: np.ndarray,
        angle: float,
        origin: np.ndarray | None = None,
    ) -> "Frame":
        """Frame rotado un `angle` (rad) alrededor de `axis` (fórmula de Rodrigues)."""
        u = _normalize(axis)
        ux, uy, uz = u
        c, s = np.cos(angle), np.sin(angle)
        k = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
        rot = np.eye(3) + s * k + (1 - c) * (k @ k)
        return cls(origin=origin, rotation=rot)

    @classmethod
    def from_z_to(
        cls,
        direction: np.ndarray,
        origin: np.ndarray | None = None,
    ) -> "Frame":
        """Frame cuyo eje local +z apunta hacia `direction`.

        Útil para objetos con simetría axial (solenoides, anillos) definidos
        con el eje en +z y luego orientados en cualquier dirección.
        """
        target = _normalize(direction)
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, target)
        s = np.linalg.norm(v)
        c = np.dot(z, target)

        if s < _EPS:  # ya alineado (c=+1) o antiparalelo (c=-1)
            rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
            return cls(origin=origin, rotation=rot)

        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + k + k @ k * ((1 - c) / (s * s))
        return cls(origin=origin, rotation=rot)

    def __repr__(self) -> str:
        return f"Frame(origin={self.origin.tolist()})"


#: Marco identidad: origen en cero, sin rotación.
IDENTITY = Frame()


__all__ = ["Frame", "IDENTITY"]
