"""Infraestructura base: protocolos, marcos de referencia, objeto físico y registro visual."""

from . import base, exceptions, frame, protocols, registry
from .base import PhysicalObject
from .exceptions import (
    EMToolkitError,
    PhysicsError,
    RegistryError,
    UnsupportedDimensionError,
)
from .frame import IDENTITY, Frame
from .protocols import ChargeSource, CurrentSource, Renderable
from .registry import build, is_registered, register

__all__ = [
    "base", "exceptions", "frame", "protocols", "registry",
    "PhysicalObject",
    "Frame", "IDENTITY",
    "ChargeSource", "CurrentSource", "Renderable",
    "register", "build", "is_registered",
    "EMToolkitError", "PhysicsError", "RegistryError", "UnsupportedDimensionError",
]
