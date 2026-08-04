"""
Paleta de colores canónica para electromagnetismo.

Centraliza todos los colores del paquete para mantener consistencia visual
entre proyectos. Los valores son hex neutrales respecto al backend; la
conversión a colores de Manim ocurre en la capa `mobjects/`.
"""

from __future__ import annotations

# Campos
E_FIELD = "#3B82F6"      # campo eléctrico (azul)
B_FIELD = "#22C55E"      # campo magnético (verde)
CURRENT = "#EAB308"      # corriente (amarillo)

# Cargas
POSITIVE = "#EF4444"     # carga positiva (rojo)
NEGATIVE = "#3B82F6"     # carga negativa (azul)
NEUTRAL = "#9CA3AF"      # neutro / sin carga (gris)

# Elementos auxiliares
AUXILIARY = "#6B728080"  # superficies gaussianas, lazos amperianos (gris translúcido)
POLE_NORTH = "#EF4444"   # polo norte (rojo)
POLE_SOUTH = "#3B82F6"   # polo sur (azul)

# Circuitos
WIRE = "#D1D5DB"
RESISTOR = "#F97316"
CAPACITOR = "#06B6D4"
INDUCTOR = "#A855F7"
BATTERY = "#84CC16"

#: Colormap por defecto para codificar magnitud de campo.
DEFAULT_COLORMAP = "viridis"


__all__ = [
    "E_FIELD",
    "B_FIELD",
    "CURRENT",
    "POSITIVE",
    "NEGATIVE",
    "NEUTRAL",
    "AUXILIARY",
    "POLE_NORTH",
    "POLE_SOUTH",
    "WIRE",
    "RESISTOR",
    "CAPACITOR",
    "INDUCTOR",
    "BATTERY",
    "DEFAULT_COLORMAP",
]
