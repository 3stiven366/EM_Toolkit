"""
Representación visual 2D de cargas.

Convierte una carga del toolkit (`PointCharge`) en su mobject esquemático:
un disco de color según el signo, con el símbolo + / − y una etiqueta opcional.

Color por convención (palette): carga positiva en rojo, negativa en azul,
neutra en gris. Importa Manim solo a través de `_backend`.
"""

from __future__ import annotations

from ...config import defaults, palette
from ...sources.charges.point_charge import PointCharge
from .._backend import Dot, Text, VGroup, hex_to_color


def charge_color(sign: int):
    """Color de Manim según el signo de la carga (+1, -1, 0)."""
    if sign > 0:
        return hex_to_color(palette.POSITIVE)
    if sign < 0:
        return hex_to_color(palette.NEGATIVE)
    return hex_to_color(palette.NEUTRAL)


def charge_symbol(sign: int) -> str:
    """Símbolo textual del signo: '+', '−' o ''."""
    if sign > 0:
        return "+"
    if sign < 0:
        return "−"          # signo menos tipográfico (U+2212), no guion
    return ""


def charge_mobject(
    charge: PointCharge,
    radius: float | None = None,
    show_symbol: bool = True,
    label: str | None = None,
) -> VGroup:
    """Construye el mobject 2D de una carga puntual.

    Args:
        charge: la carga del toolkit a representar.
        radius: radio del disco en unidades de escena. Si None, usa el default.
        show_symbol: dibujar el símbolo + / − dentro del disco.
        label: texto opcional junto a la carga (p. ej. "q₁").

    Returns:
        Un VGroup posicionado en charge.position, con el disco, el símbolo y
        la etiqueta (los que apliquen).
    """
    r = defaults.CHARGE_DEFAULTS["radius"] if radius is None else radius
    sign = charge.sign
    color = charge_color(sign)

    # Disco totalmente opaco: tapa las líneas de campo que pasan por su centro.
    disco = Dot(point=charge.position, radius=r, color=color)
    disco.set_fill(color, opacity=1.0)
    disco.set_stroke(width=0)
    grupo = VGroup(disco)

    if show_symbol and sign != 0:
        # Símbolo blanco, grueso, dimensionado a ~1.2·radio para que se lea
        # bien dentro del disco (el disco tiene diámetro 2·r).
        simbolo = Text(
            charge_symbol(sign),
            color=hex_to_color("#FFFFFF"),
            weight="BOLD",
        )
        simbolo.set(width=r * 1.2)
        simbolo.move_to(charge.position)
        grupo.add(simbolo)

    if label:
        etiqueta = Text(label)
        etiqueta.set(height=defaults.CHARGE_DEFAULTS["label_font_size"] / 100)
        # Colocar la etiqueta arriba-derecha del disco
        etiqueta.next_to(disco, direction=_up_right(), buff=0.1)
        grupo.add(etiqueta)

    return grupo


def _up_right():
    """Dirección arriba-derecha para colocar etiquetas (evita import directo)."""
    from .._backend import RIGHT, UP
    return UP + RIGHT


__all__ = ["charge_mobject", "charge_color", "charge_symbol"]
