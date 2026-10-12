"""
Contratos estructurales de las fuentes físicas.

Definen la interfaz mínima que un objeto debe cumplir para participar en la
superposición de campos. El `EMWorld` opera sobre estos protocolos, no sobre
clases concretas: cualquier objeto con los métodos correctos es una fuente
válida, sin necesidad de heredar de una base común (tipado estructural).

Convención de firmas de campo:
    entrada  points : np.ndarray de forma (N, 3), posiciones en metros
    salida   vector : np.ndarray de forma (N, 3)   [E en V/m, B en T]
             escalar: np.ndarray de forma (N,)      [V en voltios]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ChargeSource(Protocol):
    """Fuente de campo eléctrico y potencial."""

    def electric_field(self, points: np.ndarray) -> np.ndarray:
        """Campo eléctrico E en cada punto [V/m]. (N,3) -> (N,3)."""
        ...

    def potential(self, points: np.ndarray) -> np.ndarray:
        """Potencial eléctrico V en cada punto [V]. (N,3) -> (N,)."""
        ...

    def total_charge(self) -> float:
        """Carga total de la fuente [C]."""
        ...


@runtime_checkable
class CurrentSource(Protocol):
    """Fuente de campo magnético a partir de una corriente."""

    current: float

    def magnetic_field(self, points: np.ndarray) -> np.ndarray:
        """Campo magnético B en cada punto [T]. (N,3) -> (N,3)."""
        ...


@runtime_checkable
class Renderable(Protocol):
    """Objeto capaz de producir su representación visual.

    El tipo de retorno es intencionalmente `object`: la capa `mobjects/`
    devuelve un Mobject de Manim, pero este módulo no importa Manim.
    """

    def build(self, dim: int, style: dict | None = None) -> object:
        """Construye el mobject correspondiente para la dimensión dada."""
        ...


__all__ = ["ChargeSource", "CurrentSource", "Renderable"]
