"""
Estrategias de escalado para visualización de campos vectoriales.

Un campo tiene magnitudes que pueden variar en muchos órdenes (enorme cerca de
una carga, diminuta lejos). Dibujar flechas con largo proporcional a la
magnitud real suele ser ilegible. Estas estrategias deciden el **largo** de
cada flecha a partir de su magnitud; el **color** siempre codifica la magnitud
real por separado (en vector_field.py).

Cada estrategia implementa `lengths(magnitudes) -> largos` sobre un array, y
`normalize(magnitudes) -> t in [0,1]` para el color. Son numéricas puras: no
importan Manim.
"""

from __future__ import annotations

import numpy as np


class ScalingStrategy:
    """Interfaz base. Las subclases definen cómo la magnitud se vuelve largo."""

    def lengths(self, magnitudes: np.ndarray) -> np.ndarray:
        """Largo de cada flecha (unidades de escena) a partir de la magnitud."""
        raise NotImplementedError

    def normalize(self, magnitudes: np.ndarray) -> np.ndarray:
        """Mapea magnitudes a t ∈ [0,1] para codificar color (log por defecto).

        El color es independiente del largo: incluso con largo uniforme, el
        color conserva la información de magnitud. Se usa escala logarítmica
        porque los campos abarcan muchos órdenes.
        """
        m = np.asarray(magnitudes, dtype=float)
        pos = m[m > 0]
        if pos.size == 0:
            return np.zeros_like(m)
        logm = np.log10(np.maximum(m, pos.min() * 1e-6))
        lo, hi = logm.min(), logm.max()
        if hi - lo < 1e-12:
            return np.full_like(m, 0.5)
        return (logm - lo) / (hi - lo)


class Normalized(ScalingStrategy):
    """Todas las flechas del mismo largo. Muestra la topología del campo.

    Es el modo pedagógico por defecto: la dirección se ve con claridad en todo
    el dominio, y la magnitud va solo en el color.
    """

    def __init__(self, length: float = 0.5) -> None:
        self.length = float(length)

    def lengths(self, magnitudes: np.ndarray) -> np.ndarray:
        m = np.asarray(magnitudes, dtype=float)
        return np.full_like(m, self.length)


class Logarithmic(ScalingStrategy):
    """Largo ∝ log(1 + m/m0). Comprime el rango dinámico.

    Las flechas crecen con la magnitud pero sin explotar cerca de las fuentes.
    """

    def __init__(self, min_length: float = 0.1, max_length: float = 0.7) -> None:
        self.min_length = float(min_length)
        self.max_length = float(max_length)

    def lengths(self, magnitudes: np.ndarray) -> np.ndarray:
        m = np.asarray(magnitudes, dtype=float)
        pos = m[m > 0]
        if pos.size == 0:
            return np.full_like(m, self.min_length)
        logm = np.log10(np.maximum(m, pos.min() * 1e-6))
        lo, hi = logm.min(), logm.max()
        if hi - lo < 1e-12:
            t = np.full_like(m, 0.5)
        else:
            t = (logm - lo) / (hi - lo)
        return self.min_length + t * (self.max_length - self.min_length)


class LinearClipped(ScalingStrategy):
    """Largo ∝ magnitud, saturado en `v_max`. Bueno para campos uniformes.

    Preserva las proporciones reales dentro del rango, útil cuando las
    magnitudes son comparables (capacitor, interior de un solenoide).
    """

    def __init__(self, v_max: float, max_length: float = 0.6) -> None:
        if v_max <= 0:
            raise ValueError(f"v_max debe ser positivo, recibido {v_max}")
        self.v_max = float(v_max)
        self.max_length = float(max_length)

    def lengths(self, magnitudes: np.ndarray) -> np.ndarray:
        m = np.asarray(magnitudes, dtype=float)
        t = np.clip(m / self.v_max, 0.0, 1.0)
        return t * self.max_length


class PhysicalSI(ScalingStrategy):
    """Largo = magnitud real × factor. Comparaciones cuantitativas.

    El largo es físicamente fiel (proporcional a la magnitud SI). Puede ser
    ilegible si el rango es amplio; pensado para cálculos y comparaciones
    numéricas, no para claridad topológica.
    """

    def __init__(self, scale: float) -> None:
        if scale <= 0:
            raise ValueError(f"scale debe ser positivo, recibido {scale}")
        self.scale = float(scale)

    def lengths(self, magnitudes: np.ndarray) -> np.ndarray:
        return np.asarray(magnitudes, dtype=float) * self.scale


__all__ = [
    "ScalingStrategy",
    "Normalized",
    "Logarithmic",
    "LinearClipped",
    "PhysicalSI",
]
