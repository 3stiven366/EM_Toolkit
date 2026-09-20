"""
Superposición de campos de múltiples fuentes.

El electromagnetismo clásico es lineal: el campo total es la suma de los campos
individuales. Estas funciones recorren una colección de fuentes, evalúan cada
una en los puntos dados y acumulan el resultado. El `EMWorld` las usa como su
motor interno.

Las fuentes se identifican por sus protocolos (`ChargeSource`, `CurrentSource`),
no por su clase concreta: cualquier objeto con los métodos correctos participa.

Convención de firmas:
    points   : (N, 3) puntos de evaluación [m]
    sources  : iterable de fuentes
    salida E,B: (N, 3)   ·   salida V: (N,)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..core.protocols import ChargeSource, CurrentSource


def total_electric_field(
    points: np.ndarray,
    sources: Iterable,
) -> np.ndarray:
    """Suma el campo E de todas las fuentes que son ChargeSource. (N,3)->(N,3)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    total = np.zeros_like(points)
    for src in sources:
        if isinstance(src, ChargeSource):
            total = total + src.electric_field(points)
    return total


def total_potential(
    points: np.ndarray,
    sources: Iterable,
) -> np.ndarray:
    """Suma el potencial V de todas las fuentes que son ChargeSource. (N,3)->(N,)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    total = np.zeros(points.shape[0])
    for src in sources:
        if isinstance(src, ChargeSource):
            total = total + src.potential(points)
    return total


def total_magnetic_field(
    points: np.ndarray,
    sources: Iterable,
) -> np.ndarray:
    """Suma el campo B de todas las fuentes que son CurrentSource. (N,3)->(N,3)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    total = np.zeros_like(points)
    for src in sources:
        if isinstance(src, CurrentSource):
            total = total + src.magnetic_field(points)
    return total


__all__ = [
    "total_electric_field",
    "total_potential",
    "total_magnetic_field",
]
