"""Visualización de campos: escalado, campos vectoriales, líneas de campo y equipotenciales."""

from .equipotential import equipotential_2d, equipotential_levels
from .field_lines import charge_seeds, field_lines_2d
from .scaling import LinearClipped, Logarithmic, Normalized, PhysicalSI, ScalingStrategy
from .vector_field import vector_field_2d

__all__ = [
    "vector_field_2d",
    "field_lines_2d", "charge_seeds",
    "equipotential_2d", "equipotential_levels",
    "ScalingStrategy", "Normalized", "Logarithmic", "LinearClipped", "PhysicalSI",
]
