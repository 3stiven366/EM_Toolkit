"""Configuración global: constantes físicas, escalado, colores y valores por defecto."""

from . import constants, defaults, palette, units
from .units import DEFAULT_SCALE, UnitScale

__all__ = ["constants", "units", "palette", "defaults", "UnitScale", "DEFAULT_SCALE"]
