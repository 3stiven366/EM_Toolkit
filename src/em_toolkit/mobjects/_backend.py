"""
Punto único de contacto con Manim.

Regla de oro de la capa visual: **ningún otro archivo del paquete importa
`manim` directamente**. Todos importan desde aquí. Así, si Manim cambia su API
o se quisiera otro backend, solo este archivo se toca.

Reexporta las primitivas de Manim que el toolkit usa y ofrece utilidades de
conversión (colores hex de `palette.py` -> colores de Manim).

Probado contra ManimCE 0.22.
"""

from __future__ import annotations

import numpy as np

# -- Reexport de primitivas de Manim ----------------------------------------
# Mobjects y geometría
from manim import (  # noqa: F401
    Arrow,
    Arrow3D,
    Circle,
    Cylinder,
    Dot,
    Dot3D,
    Line,
    Prism,
    Sphere,
    Surface,
    Text,
    VGroup,
    VMobject,
)

# Escenas
from manim import Scene, ThreeDScene  # noqa: F401

# Animaciones comunes
from manim import (  # noqa: F401
    Create,
    FadeIn,
    FadeOut,
    GrowArrow,
    Transform,
    Write,
)

# Utilidades de color y constantes de dirección
from manim import (  # noqa: F401
    DOWN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    UP,
    ManimColor,
    interpolate_color,
)


def hex_to_color(hex_str: str) -> ManimColor:
    """Convierte un color hex ('#RRGGBB' o '#RRGGBBAA') en un ManimColor.

    El canal alfa, si está presente, se ignora aquí: la opacidad se aplica al
    dibujar el mobject (parámetro `fill_opacity` / `stroke_opacity`), no en el
    color. Usa `hex_to_color_and_opacity` si necesitas ambos.
    """
    return ManimColor(hex_str[:7])


def hex_to_color_and_opacity(hex_str: str) -> tuple[ManimColor, float]:
    """Devuelve (color, opacidad) a partir de un hex '#RRGGBB' o '#RRGGBBAA'.

    Si no hay canal alfa, la opacidad es 1.0.
    """
    color = ManimColor(hex_str[:7])
    if len(hex_str) >= 9:
        alpha = int(hex_str[7:9], 16) / 255.0
    else:
        alpha = 1.0
    return color, alpha


def colormap_color(t: float, cmap: str = "viridis") -> ManimColor:
    """Color de un colormap de matplotlib evaluado en t ∈ [0, 1].

    Permite codificar magnitud de campo con gradientes suaves (viridis, plasma,
    etc.) en vez de una interpolación lineal entre dos colores.
    """
    import matplotlib

    t = float(np.clip(t, 0.0, 1.0))
    # matplotlib >= 3.9 usa matplotlib.colormaps[...]; versiones viejas,
    # matplotlib.cm.get_cmap(...). Se intenta la API nueva primero.
    try:
        colormap = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
    r, g, b, _ = colormap(t)
    return ManimColor.from_rgb((r, g, b))


__all__ = [
    # Mobjects
    "Arrow", "Arrow3D", "Circle", "Cylinder", "Dot", "Dot3D", "Line",
    "Prism", "Sphere", "Surface", "Text", "VGroup", "VMobject",
    # Escenas
    "Scene", "ThreeDScene",
    # Animaciones
    "Create", "FadeIn", "FadeOut", "GrowArrow", "Transform", "Write",
    # Direcciones y color
    "ORIGIN", "UP", "DOWN", "LEFT", "RIGHT", "IN", "OUT",
    "ManimColor", "interpolate_color",
    # Conversión de color
    "hex_to_color", "hex_to_color_and_opacity", "colormap_color",
]
