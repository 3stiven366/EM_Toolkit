"""
Campo vectorial 2D como flechas.

Toma una función de campo (p. ej. `carga.electric_field`), una grilla de puntos
y una estrategia de escalado, y produce el VGroup de flechas listo para añadir a
una escena. El largo lo decide la estrategia; el color codifica la magnitud real
mediante un colormap.

Reemplaza el bucle manual de flechas por una sola llamada. Importa Manim solo a
través de `_backend`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import defaults
from .scaling import Normalized, ScalingStrategy


def vector_field_2d(
    field_func: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    scaling: ScalingStrategy | None = None,
    colormap: str | None = None,
    stroke_width: float = 2.5,
    min_magnitude: float = 1e-30,
):
    """Construye un VGroup de flechas que representa un campo en 2D.

    Args:
        field_func: función (N,3) -> (N,3) que evalúa el campo en cada punto.
        points: (N,3) posiciones donde dibujar flechas (en unidades de escena).
        scaling: estrategia de largo de flecha. Por defecto, Normalized().
        colormap: nombre del colormap para el color por magnitud. Por defecto,
            el del paquete (viridis).
        stroke_width: grosor de las flechas.
        min_magnitude: magnitud por debajo de la cual la flecha se omite
            (evita flechas de dirección indefinida en campo nulo).

    Returns:
        Un VGroup con una flecha por cada punto de magnitud suficiente.
    """
    from ..mobjects._backend import Arrow, VGroup, colormap_color

    scaling = Normalized() if scaling is None else scaling
    cmap = defaults.FIELD_DEFAULTS["colormap"] if colormap is None else colormap

    points = np.atleast_2d(np.asarray(points, dtype=float))
    vectors = field_func(points)                       # (N,3)
    magnitudes = np.linalg.norm(vectors, axis=1)       # (N,)

    # Filtrar puntos con campo despreciable (dirección indefinida)
    mask = magnitudes > min_magnitude
    points = points[mask]
    vectors = vectors[mask]
    magnitudes = magnitudes[mask]

    if magnitudes.size == 0:
        return VGroup()

    largos = scaling.lengths(magnitudes)               # (M,)
    colores_t = scaling.normalize(magnitudes)          # (M,) en [0,1]
    direcciones = vectors / magnitudes[:, None]        # unitarios

    grupo = VGroup()
    for p, d, L, t in zip(points, direcciones, largos, colores_t):
        inicio = p
        fin = p + d * L
        flecha = Arrow(
            start=inicio,
            end=fin,
            buff=0,
            stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.35,
            color=colormap_color(t, cmap),
        )
        grupo.add(flecha)

    return grupo


__all__ = ["vector_field_2d"]
