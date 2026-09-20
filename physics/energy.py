"""
Energía del campo electromagnético y parámetros de elementos.

Densidades de energía de los campos E y B, y fórmulas cerradas de capacitancia
e inductancia para las configuraciones más comunes. Todo en SI.

    u_E = ½ ε₀ |E|²      [J/m³]
    u_B = |B|² / (2μ₀)   [J/m³]
"""

from __future__ import annotations

import numpy as np

from ..config.constants import eps0, mu0


def electric_energy_density(E: np.ndarray) -> np.ndarray:
    """Densidad de energía del campo eléctrico u_E = ½ε₀|E|². [J/m³].

    Acepta un vector (3,) o un lote (N,3); devuelve escalar o (N,).
    """
    E = np.atleast_2d(np.asarray(E, dtype=float))
    u = 0.5 * eps0 * np.sum(E * E, axis=1)
    return u[0] if u.shape == (1,) else u


def magnetic_energy_density(B: np.ndarray) -> np.ndarray:
    """Densidad de energía del campo magnético u_B = |B|²/2μ₀. [J/m³]."""
    B = np.atleast_2d(np.asarray(B, dtype=float))
    u = np.sum(B * B, axis=1) / (2.0 * mu0)
    return u[0] if u.shape == (1,) else u


def capacitor_energy(capacitance: float, voltage: float) -> float:
    """Energía almacenada en un capacitor U = ½CV². [J]."""
    return 0.5 * capacitance * voltage**2


def inductor_energy(inductance: float, current: float) -> float:
    """Energía almacenada en un inductor U = ½LI². [J]."""
    return 0.5 * inductance * current**2


def parallel_plate_capacitance(
    area: float,
    separation: float,
    kappa: float = 1.0,
) -> float:
    """Capacitancia de un capacitor de placas paralelas C = κε₀A/d. [F].

    Args:
        area:       área de una placa [m²].
        separation: distancia entre placas [m], > 0.
        kappa:      constante dieléctrica relativa (1 = vacío).
    """
    if separation <= 0:
        raise ValueError(f"separation debe ser positiva, recibido {separation}")
    return kappa * eps0 * area / separation


def solenoid_inductance(
    n_turns: int,
    area: float,
    length: float,
) -> float:
    """Inductancia de un solenoide largo L = μ₀N²A/ℓ. [H].

    Aproximación de solenoide ideal (largo >> radio).

    Args:
        n_turns: número total de vueltas N.
        area:    área de la sección transversal [m²].
        length:  longitud del solenoide [m], > 0.
    """
    if length <= 0:
        raise ValueError(f"length debe ser positiva, recibido {length}")
    return mu0 * n_turns**2 * area / length


__all__ = [
    "electric_energy_density",
    "magnetic_energy_density",
    "capacitor_energy",
    "inductor_energy",
    "parallel_plate_capacitance",
    "solenoid_inductance",
]
