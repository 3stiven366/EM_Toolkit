"""
Muestreo del espacio: grillas, semillas e integración de líneas de campo.

Genera los puntos donde se evalúan los campos (grillas regulares 2D/3D),
las semillas desde donde nacen las líneas de campo, y traza esas líneas
integrando el campo con Runge-Kutta de 4º orden.

Toda coordenada está en unidades de escena (las que dibuja Manim); la
conversión desde/hacia metros la hace `config.units`, no este módulo.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def grid_2d(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    density: int,
    z: float = 0.0,
) -> np.ndarray:
    """Grilla regular en el plano z=cte. Devuelve (N, 3).

    Args:
        x_range, y_range: extremos (min, max) en cada eje.
        density: número de puntos por eje (la grilla tiene density² puntos).
        z: altura del plano.
    """
    xs = np.linspace(*x_range, density)
    ys = np.linspace(*y_range, density)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)
    return pts


def grid_3d(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    density: int,
) -> np.ndarray:
    """Grilla regular en un volumen. Devuelve (N, 3) con density³ puntos."""
    xs = np.linspace(*x_range, density)
    ys = np.linspace(*y_range, density)
    zs = np.linspace(*z_range, density)
    X, Y, Z = np.meshgrid(xs, ys, zs)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def ring_seeds(
    center: np.ndarray,
    radius: float,
    n: int,
    plane: str = "xy",
) -> np.ndarray:
    """Semillas equiespaciadas en un círculo alrededor de `center`. (n, 3).

    Útil para emitir líneas de campo desde una carga puntual de forma simétrica.

    Args:
        plane: plano del círculo, "xy", "xz" o "yz".
    """
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = radius * np.cos(theta), radius * np.sin(theta)
    zeros = np.zeros(n)
    if plane == "xy":
        offsets = np.stack([c, s, zeros], axis=1)
    elif plane == "xz":
        offsets = np.stack([c, zeros, s], axis=1)
    elif plane == "yz":
        offsets = np.stack([zeros, c, s], axis=1)
    else:
        raise ValueError(f"plane debe ser 'xy', 'xz' o 'yz', no {plane!r}")
    return center + offsets


def integrate_field_line(
    field_func: Callable[[np.ndarray], np.ndarray],
    seed: np.ndarray,
    step: float = 0.05,
    max_steps: int = 2000,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    min_field: float = 1e-12,
) -> np.ndarray:
    """Traza una línea de campo desde `seed` integrando con RK4.

    Sigue la dirección normalizada del campo (paso de arco constante), de modo
    que el espaciado de la línea no depende de la magnitud del campo.

    Args:
        field_func: función (M,3) -> (M,3) que evalúa el campo.
        seed: punto inicial (3,).
        step: paso de arco por iteración.
        max_steps: tope de puntos de la línea.
        bounds: (min_xyz, max_xyz) para detener la línea al salir de la región.
        min_field: si |campo| cae por debajo, la línea termina (punto crítico).

    Returns:
        (P, 3) puntos de la línea (P <= max_steps + 1).
    """

    def direction(p: np.ndarray) -> np.ndarray | None:
        f = field_func(p[None, :])[0]
        norm = np.linalg.norm(f)
        if norm < min_field:
            return None
        return f / norm

    points = [np.asarray(seed, dtype=float)]
    lo, hi = (None, None) if bounds is None else bounds

    for _ in range(max_steps):
        p = points[-1]
        k1 = direction(p)
        if k1 is None:
            break
        k2 = direction(p + 0.5 * step * k1)
        if k2 is None:
            break
        k3 = direction(p + 0.5 * step * k2)
        if k3 is None:
            break
        k4 = direction(p + step * k3)
        if k4 is None:
            break
        p_next = p + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if bounds is not None and (np.any(p_next < lo) or np.any(p_next > hi)):
            break
        points.append(p_next)

    return np.array(points)


__all__ = [
    "grid_2d",
    "grid_3d",
    "ring_seeds",
    "integrate_field_line",
]
