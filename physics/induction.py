"""
Inducción electromagnética: flujo magnético y ley de Faraday.

    Φ = ∫∫ B · dA        (flujo a través de una superficie)
    fem = -N dΦ/dt        (ley de Faraday-Lenz)

El flujo se calcula numéricamente sumando B·n̂ sobre una malla de la superficie.
La fem se obtiene por diferencias finitas a partir de una serie temporal de
flujo. El signo negativo de Faraday codifica la ley de Lenz.
"""

from __future__ import annotations

import numpy as np


def magnetic_flux(
    B_at_points: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
) -> float:
    """Flujo Φ = Σ B·n̂ dA sobre una superficie discretizada. [Wb].

    Args:
        B_at_points: (K, 3) campo B en el centro de cada parche.
        normals:     (K, 3) normal unitaria de cada parche.
        areas:       (K,)   área de cada parche [m²].
    """
    B_at_points = np.atleast_2d(np.asarray(B_at_points, dtype=float))
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    areas = np.asarray(areas, dtype=float)
    return float(np.sum(np.sum(B_at_points * normals, axis=1) * areas))


def flux_uniform(
    B: np.ndarray,
    normal: np.ndarray,
    area: float,
) -> float:
    """Flujo a través de una superficie plana en campo uniforme. [Wb].

        Φ = B · n̂ A = B A cos θ
    """
    B = np.asarray(B, dtype=float)
    normal = np.asarray(normal, dtype=float)
    return float(np.dot(B, normal) * area)


def emf_from_flux(
    flux_series: np.ndarray,
    times: np.ndarray,
    n_turns: int = 1,
) -> np.ndarray:
    """fem inducida = -N dΦ/dt por diferencias finitas. [V].

    Devuelve un array del mismo largo que `flux_series`, usando gradiente
    centrado (bordes de un lado). El signo negativo es la ley de Lenz.

    Args:
        flux_series: (T,) flujo en cada instante [Wb].
        times:       (T,) instantes correspondientes [s].
        n_turns:     número de espiras N.
    """
    flux_series = np.asarray(flux_series, dtype=float)
    times = np.asarray(times, dtype=float)
    return -n_turns * np.gradient(flux_series, times)


def motional_emf(
    B: float,
    length: float,
    velocity: float,
) -> float:
    """fem de movimiento de una barra en campo uniforme. [V].

        fem = B L v     (B, L y v mutuamente perpendiculares)
    """
    return B * length * velocity


__all__ = [
    "magnetic_flux",
    "flux_uniform",
    "emf_from_flux",
    "motional_emf",
]
