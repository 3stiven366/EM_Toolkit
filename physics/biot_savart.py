"""
Campo magnético por integración numérica de la ley de Biot-Savart.

    B(r) = (μ₀ I / 4π) ∮ dl × (r - r') / |r - r'|³

Motor universal: cualquier camino de corriente se discretiza en segmentos y se
suma su contribución. Una geometría nueva solo necesita entregar su curva
r'(t); el campo se obtiene aquí sin fórmula cerrada. Las geometrías con
solución analítica pueden sobreescribir por exactitud y velocidad.

Softening idéntico al de Coulomb: |r-r'| se regulariza con r_min para acotar
el campo justo sobre el conductor.

Convención de firmas:
    points   : (N, 3) puntos de evaluación [m]
    path_pts : (P, 3) vértices que discretizan el camino [m]
    current  : escalar [A]
    salida B : (N, 3) [T]
"""

from __future__ import annotations

import numpy as np

from ..config.constants import BIOT_SAVART_CONSTANT  # μ₀/4π
from ..config.defaults import FIELD_DEFAULTS


def biot_savart_discrete(
    points: np.ndarray,
    path_pts: np.ndarray,
    current: float,
    r_min: float = FIELD_DEFAULTS["r_min"],
    closed: bool = False,
) -> np.ndarray:
    """Campo B de un camino ya discretizado en vértices. (N,3) -> (N,3) [T].

    Cada segmento se representa por su vector dl y su punto medio. La suma de
    Riemann sobre los segmentos aproxima la integral de línea.

    Args:
        points:  (N, 3) puntos de evaluación.
        path_pts:(P, 3) vértices del camino en orden.
        current: corriente [A] (el signo indica el sentido).
        r_min:   softening del denominador [m].
        closed:  si True, une el último vértice con el primero (lazo cerrado).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    path_pts = np.asarray(path_pts, dtype=float)

    if closed:
        path_pts = np.vstack([path_pts, path_pts[0]])

    starts = path_pts[:-1]                       # (S, 3)
    ends = path_pts[1:]                          # (S, 3)
    dl = ends - starts                           # (S, 3)  vector de cada segmento
    midpoints = 0.5 * (starts + ends)            # (S, 3)  fuente efectiva

    # r - r' para cada punto (N) y cada segmento (S): (N, S, 3)
    r = points[:, None, :] - midpoints[None, :, :]
    dist2 = np.sum(r * r, axis=2) + r_min**2     # (N, S)
    inv = 1.0 / dist2**1.5                        # (N, S)  = 1/|r|³

    # dl × r  para cada (N, S): broadcasting del producto cruz
    cross = np.cross(dl[None, :, :], r)          # (N, S, 3)
    contrib = cross * inv[:, :, None]            # (N, S, 3)

    return BIOT_SAVART_CONSTANT * current * np.sum(contrib, axis=1)  # (N, 3)


def biot_savart_path(
    points: np.ndarray,
    path_func,
    current: float,
    t_range: tuple[float, float] = (0.0, 1.0),
    n_segments: int = 400,
    r_min: float = FIELD_DEFAULTS["r_min"],
    closed: bool = False,
) -> np.ndarray:
    """Campo B de un camino parametrizado r'(t). (N,3) -> (N,3) [T].

    Discretiza `path_func` en `n_segments` segmentos sobre `t_range` y delega
    en `biot_savart_discrete`.

    Args:
        path_func: función t -> (3,) o (K,3), evaluable de forma vectorizada
            sobre un array de parámetros t.
        t_range:   intervalo (t0, t1) del parámetro.
        n_segments:número de segmentos de discretización.
        closed:    si el camino es un lazo cerrado.
    """
    t = np.linspace(t_range[0], t_range[1], n_segments + 1)
    path_pts = np.asarray(path_func(t), dtype=float)

    # Aceptar path_func que devuelva (3, P) y normalizar a (P, 3)
    if path_pts.shape[0] == 3 and path_pts.shape[1] != 3:
        path_pts = path_pts.T

    return biot_savart_discrete(points, path_pts, current, r_min=r_min, closed=closed)


__all__ = ["biot_savart_discrete", "biot_savart_path"]
