"""
Campo eléctrico y potencial de cargas puntuales (ley de Coulomb).

Funciones puras y vectorizadas que operan sobre posiciones de fuente y puntos
de evaluación. Las clases de `sources/` las usan internamente; aquí no hay
estado ni objetos, solo física.

Softening de singularidades: en r = 0 el campo diverge. Se sustituye r por
r_eff = sqrt(r² + r_min²), lo que deja el resultado prácticamente exacto lejos
de la carga y acotado justo sobre ella. Con r_min = 0 se recupera la fórmula
sin regularizar.

Convención de firmas:
    points  : (N, 3) posiciones de evaluación [m]
    sources : (M, 3) posiciones de las cargas [m]
    charges : (M,)   valores de carga [C]
    salida E: (N, 3) [V/m]   ·   salida V: (N,) [V]
"""

from __future__ import annotations

import numpy as np

from ..config.constants import k_e
from ..config.defaults import FIELD_DEFAULTS


def point_charge_field(
    points: np.ndarray,
    source: np.ndarray,
    charge: float,
    r_min: float = FIELD_DEFAULTS["r_min"],
) -> np.ndarray:
    """Campo E de una sola carga puntual. (N,3) -> (N,3) [V/m]."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    r = points - np.asarray(source, dtype=float)          # (N,3)
    dist2 = np.sum(r * r, axis=1) + r_min**2               # (N,)
    inv = 1.0 / dist2**1.5                                 # (N,)  = 1/r_eff³
    return k_e * charge * r * inv[:, None]


def point_charge_potential(
    points: np.ndarray,
    source: np.ndarray,
    charge: float,
    r_min: float = FIELD_DEFAULTS["r_min"],
) -> np.ndarray:
    """Potencial V de una sola carga puntual. (N,3) -> (N,) [V]."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    r = points - np.asarray(source, dtype=float)
    dist = np.sqrt(np.sum(r * r, axis=1) + r_min**2)
    return k_e * charge / dist


def electric_field(
    points: np.ndarray,
    sources: np.ndarray,
    charges: np.ndarray,
    r_min: float = FIELD_DEFAULTS["r_min"],
) -> np.ndarray:
    """Campo E de un conjunto de cargas por superposición. (N,3) -> (N,3) [V/m].

    Args:
        points:  (N, 3) puntos de evaluación.
        sources: (M, 3) posiciones de las cargas.
        charges: (M,)   valores de carga.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    sources = np.atleast_2d(np.asarray(sources, dtype=float))
    charges = np.asarray(charges, dtype=float)

    # Diferencias punto-fuente: (N, M, 3)
    r = points[:, None, :] - sources[None, :, :]
    dist2 = np.sum(r * r, axis=2) + r_min**2              # (N, M)
    inv = 1.0 / dist2**1.5                                # (N, M)
    contrib = k_e * charges[None, :, None] * r * inv[:, :, None]  # (N, M, 3)
    return np.sum(contrib, axis=1)                        # (N, 3)


def potential(
    points: np.ndarray,
    sources: np.ndarray,
    charges: np.ndarray,
    r_min: float = FIELD_DEFAULTS["r_min"],
) -> np.ndarray:
    """Potencial V de un conjunto de cargas por superposición. (N,3) -> (N,) [V]."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    sources = np.atleast_2d(np.asarray(sources, dtype=float))
    charges = np.asarray(charges, dtype=float)

    r = points[:, None, :] - sources[None, :, :]
    dist = np.sqrt(np.sum(r * r, axis=2) + r_min**2)     # (N, M)
    return np.sum(k_e * charges[None, :] / dist, axis=1) # (N,)


__all__ = [
    "point_charge_field",
    "point_charge_potential",
    "electric_field",
    "potential",
]
