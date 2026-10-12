"""
Fuerzas electromagnéticas.

Fuerza de Lorentz sobre cargas y fuerza por unidad de longitud entre corrientes.
Funciones puras y vectorizadas; las animaciones de trayectorias las usan para
integrar el movimiento de partículas.

Convención de firmas:
    q        : escalar [C]      ·  v, E, B : (3,) o (N,3)
    salida F : misma forma que la entrada [N]
"""

from __future__ import annotations

import numpy as np

from ..config.constants import mu0


def lorentz_force(
    charge: float,
    velocity: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Fuerza de Lorentz F = q(E + v × B). [N].

    Acepta un solo estado (3,) o un lote (N,3), siempre que v, E y B compartan
    forma. E o B pueden pasarse como cero escalar si no hay ese campo.
    """
    velocity = np.asarray(velocity, dtype=float)
    E = np.zeros_like(velocity) if np.isscalar(E) else np.asarray(E, dtype=float)
    B = np.zeros_like(velocity) if np.isscalar(B) else np.asarray(B, dtype=float)
    return charge * (E + np.cross(velocity, B))


def force_between_parallel_wires(
    current_1: float,
    current_2: float,
    separation: float,
) -> float:
    """Fuerza por unidad de longitud entre dos hilos rectos paralelos. [N/m].

        F/L = μ₀ I₁ I₂ / (2π d)

    El signo indica el carácter: positivo = atracción (corrientes en el mismo
    sentido), negativo = repulsión (sentidos opuestos).

    Args:
        current_1, current_2: corrientes [A] (con signo según sentido).
        separation: distancia entre los hilos [m], > 0.
    """
    if separation <= 0:
        raise ValueError(f"separation debe ser positiva, recibido {separation}")
    return mu0 * current_1 * current_2 / (2.0 * np.pi * separation)


__all__ = ["lorentz_force", "force_between_parallel_wires"]
