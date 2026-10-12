"""
Parámetros numéricos por defecto del paquete.

Centraliza los valores de configuración usados al muestrear y dibujar campos,
cargas y circuitos, evitando números mágicos dispersos. Cada consumidor puede
sobrescribirlos localmente; estos son solo el punto de partida.
"""

from __future__ import annotations

from . import palette

# Muestreo y visualización de campos
FIELD_DEFAULTS = {
    "density": 12,               # muestras por unidad de escena
    "r_min": 0.15,               # softening cerca de singularidades: |r| -> sqrt(r² + r_min²)
    "max_magnitude": None,       # recorte de magnitud (None = sin recorte)
    "colormap": palette.DEFAULT_COLORMAP,
    "stroke_width": 2.0,
    "arrow_tip_ratio": 0.25,     # tamaño de punta relativo a la longitud
}

# Líneas de campo (integración)
FIELD_LINE_DEFAULTS = {
    "step": 0.05,                # paso de integración en unidades de escena
    "max_steps": 2000,           # tope de pasos por línea
    "min_step": 1e-4,            # paso mínimo antes de detener (adaptativo)
    "seeds_per_charge": 12,      # líneas emitidas por unidad de carga
}

# Cargas
CHARGE_DEFAULTS = {
    "radius": 0.28,              # radio visual en unidades de escena
    "glow_ratio": 2.5,           # extensión del halo respecto al radio
    "label_font_size": 28,
}

# Elementos de circuito
CIRCUIT_DEFAULTS = {
    "wire_stroke": 3.0,
    "element_scale": 1.0,
    "label_font_size": 24,
}


__all__ = [
    "FIELD_DEFAULTS",
    "FIELD_LINE_DEFAULTS",
    "CHARGE_DEFAULTS",
    "CIRCUIT_DEFAULTS",
]
