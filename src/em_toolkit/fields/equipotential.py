"""
Curvas equipotenciales.

Traza curvas de potencial constante (V = cte). Son perpendiculares a las líneas
de campo en todo punto: para un sistema de cargas, rodean las fuentes.

El cálculo de contornos usa marching squares a través de matplotlib (solo como
motor numérico de curvas de nivel; no dibuja nada). Cada contorno se convierte
en una curva de Manim. El color codifica el valor del potencial.

Importa Manim solo a través de `_backend`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import defaults


def equipotential_levels(
    potential_func: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    levels,
    resolution: int = 300,
    z: float = 0.0,
) -> list[tuple[float, np.ndarray]]:
    """Calcula los contornos equipotenciales en una grilla (sin Manim).

    Args:
        potential_func: función (N,3) -> (N,) que evalúa V.
        x_range, y_range: límites de la región.
        levels: lista de valores de V para los que trazar contornos, o un entero
            (número de niveles automáticos entre percentiles robustos de V).
        resolution: puntos por eje de la grilla de evaluación.
        z: plano de evaluación.

    Returns:
        Lista de (nivel, curva), donde curva es un array (P,3) de puntos. Un
        mismo nivel puede producir varias curvas (varios contornos separados).
    """
    import matplotlib
    matplotlib.use("Agg")  # backend sin ventana, solo cálculo
    import matplotlib.pyplot as plt

    xs = np.linspace(*x_range, resolution)
    ys = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], axis=1)

    V = potential_func(pts).reshape(X.shape)

    # Niveles automáticos: percentiles robustos para evitar las singularidades
    if isinstance(levels, int):
        finite = V[np.isfinite(V)]
        lo, hi = np.percentile(finite, [5, 95])
        levels = np.linspace(lo, hi, levels)

    # matplotlib calcula los contornos; no se renderiza ninguna figura
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(X, Y, V, levels=levels)

    resultado = []
    # API robusta entre versiones de matplotlib (allsegs por nivel)
    niveles = cs.levels
    for nivel, segs in zip(niveles, cs.allsegs):
        for seg in segs:
            if seg.shape[0] < 2:
                continue
            curva = np.column_stack([seg, np.full(seg.shape[0], z)])
            resultado.append((float(nivel), curva))

    plt.close(fig)
    return resultado


def equipotential_2d(
    potential_func: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    levels=12,
    resolution: int = 300,
    colormap: str | None = None,
    stroke_width: float = 2.0,
):
    """Construye un VGroup de curvas equipotenciales en 2D.

    Args:
        potential_func: función (N,3) -> (N,) que evalúa V.
        x_range, y_range: límites de la región.
        levels: niveles de V (lista) o número de niveles automáticos.
        resolution: resolución de la grilla de cálculo.
        colormap: colormap para colorear por valor de V. Por defecto, el del paquete.
        stroke_width: grosor de las curvas.

    Returns:
        Un VGroup con las curvas equipotenciales.
    """
    from ..mobjects._backend import VGroup, VMobject, colormap_color

    cmap = defaults.FIELD_DEFAULTS["colormap"] if colormap is None else colormap

    contornos = equipotential_levels(
        potential_func, x_range, y_range, levels, resolution=resolution
    )
    if not contornos:
        return VGroup()

    niveles = np.array([n for n, _ in contornos])
    lo, hi = niveles.min(), niveles.max()
    span = hi - lo if hi - lo > 1e-12 else 1.0

    grupo = VGroup()
    for nivel, curva in contornos:
        t = (nivel - lo) / span
        c = VMobject(stroke_width=stroke_width)
        c.set_points_as_corners(curva)
        c.set_color(colormap_color(float(t), cmap))
        grupo.add(c)

    return grupo


__all__ = ["equipotential_2d", "equipotential_levels"]
