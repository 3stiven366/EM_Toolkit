"""Fuentes físicas: cargas, corrientes y materiales."""

from . import charges, currents, materials
from .charges import (
    Dipole, DiskCharge, GenericCharge, LineCharge,
    PlaneCharge, PointCharge, RingCharge, SphereCharge,
)
from .currents import (
    CurrentLoop, CurrentPath, CurrentSheet,
    InfiniteWire, Solenoid, Toroid, WireSegment,
)
from .materials import BarMagnet, ConductingSphere, DielectricSphere

__all__ = [
    "charges", "currents", "materials",
    # charges
    "PointCharge", "Dipole", "LineCharge", "RingCharge",
    "DiskCharge", "PlaneCharge", "SphereCharge", "GenericCharge",
    # currents
    "CurrentPath", "InfiniteWire", "WireSegment", "CurrentLoop",
    "Solenoid", "Toroid", "CurrentSheet",
    # materials
    "BarMagnet", "ConductingSphere", "DielectricSphere",
]
