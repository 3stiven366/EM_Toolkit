"""
Constantes físicas fundamentales en unidades del SI (CODATA 2018).

Base de toda la física del paquete. ε₀ y k_e se derivan de μ₀ y c mediante
relaciones exactas para evitar inconsistencias. El escalado a unidades de
Manim ocurre en `units.py`, no aquí.
"""

from __future__ import annotations

import math

# Exactas por definición (SI 2019)
SPEED_OF_LIGHT: float = 299_792_458.0
ELEMENTARY_CHARGE: float = 1.602_176_634e-19

# Electromagnéticas (μ₀ medida; ε₀ y k_e derivadas)
VACUUM_PERMEABILITY: float = 1.256_637_062_12e-6
VACUUM_PERMITTIVITY: float = 1.0 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2)
COULOMB_CONSTANT: float = 1.0 / (4.0 * math.pi * VACUUM_PERMITTIVITY)
BIOT_SAVART_CONSTANT: float = VACUUM_PERMEABILITY / (4.0 * math.pi)

# Masas de partículas (CODATA 2018)
ELECTRON_MASS: float = 9.109_383_701_5e-31
PROTON_MASS: float = 1.672_621_923_69e-27

# Aliases cortos
eps0 = VACUUM_PERMITTIVITY
mu0 = VACUUM_PERMEABILITY
c = SPEED_OF_LIGHT
e = ELEMENTARY_CHARGE
k_e = COULOMB_CONSTANT
m_e = ELECTRON_MASS
m_p = PROTON_MASS


__all__ = [
    "SPEED_OF_LIGHT",
    "ELEMENTARY_CHARGE",
    "VACUUM_PERMEABILITY",
    "VACUUM_PERMITTIVITY",
    "COULOMB_CONSTANT",
    "BIOT_SAVART_CONSTANT",
    "ELECTRON_MASS",
    "PROTON_MASS",
    "eps0",
    "mu0",
    "c",
    "e",
    "k_e",
    "m_e",
    "m_p",
]
