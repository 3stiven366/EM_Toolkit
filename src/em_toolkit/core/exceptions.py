"""
Excepciones del paquete.

Todas derivan de `EMToolkitError`, de modo que un consumidor puede capturar
cualquier error propio de la librería con un solo `except`.
"""

from __future__ import annotations


class EMToolkitError(Exception):
    """Base de todas las excepciones del paquete."""


class PhysicsError(EMToolkitError):
    """Configuración física inválida o cálculo mal definido."""


class UnsupportedDimensionError(EMToolkitError):
    """Se pidió una representación en una dimensión no soportada por el objeto."""

    def __init__(self, obj_name: str, dim: int) -> None:
        super().__init__(
            f"{obj_name} no tiene representación en {dim}D."
        )
        self.obj_name = obj_name
        self.dim = dim


class RegistryError(EMToolkitError):
    """No existe un builder visual registrado para la clase y dimensión dadas."""


__all__ = [
    "EMToolkitError",
    "PhysicsError",
    "UnsupportedDimensionError",
    "RegistryError",
]
