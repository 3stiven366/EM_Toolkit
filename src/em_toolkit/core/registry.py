"""
Registro de builders visuales por clase física y dimensión.

Desacopla el modelo físico de su representación. Una fuente (`PointCharge`)
no sabe dibujarse; un builder registrado se encarga de convertirla en mobject
para una dimensión concreta (2 o 3). El `EMWorld` llama a `build(obj, dim)`
sin conocer los detalles visuales.

La resolución respeta la herencia: si no hay builder para la clase exacta, se
busca en sus bases (MRO). Así un builder para una clase base cubre a sus
subclases mientras no se registre uno más específico.
"""

from __future__ import annotations

from typing import Callable

from .exceptions import RegistryError

# (clase, dim) -> función builder
_REGISTRY: dict[tuple[type, int], Callable] = {}


def register(cls: type, dim: int) -> Callable:
    """Decorador que asocia un builder a una clase física y una dimensión.

    Ejemplo:
        @register(PointCharge, dim=3)
        def _(charge, style):
            return Sphere(...)
    """
    if dim not in (2, 3):
        raise ValueError(f"dim debe ser 2 o 3, no {dim}")

    def decorator(builder: Callable) -> Callable:
        _REGISTRY[(cls, dim)] = builder
        return builder

    return decorator


def _resolve(cls: type, dim: int) -> Callable | None:
    """Busca el builder para `cls` en `dim`, subiendo por el MRO si hace falta."""
    for base in cls.__mro__:
        builder = _REGISTRY.get((base, dim))
        if builder is not None:
            return builder
    return None


def build(obj: object, dim: int, style: dict | None = None) -> object:
    """Construye el mobject de `obj` para la dimensión `dim`.

    Busca el builder registrado para el tipo de `obj` (o una de sus bases) y
    lo invoca. Lanza `RegistryError` si no hay ninguno.
    """
    builder = _resolve(type(obj), dim)
    if builder is None:
        raise RegistryError(
            f"No hay builder registrado para {type(obj).__name__} en {dim}D."
        )
    return builder(obj, style)


def is_registered(cls: type, dim: int) -> bool:
    """Indica si existe un builder resoluble para `cls` en `dim`."""
    return _resolve(cls, dim) is not None


def clear_registry() -> None:
    """Vacía el registro. Uso principal: aislar tests."""
    _REGISTRY.clear()


__all__ = ["register", "build", "is_registered", "clear_registry"]
