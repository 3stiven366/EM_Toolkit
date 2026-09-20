"""Fuentes de carga: puntual, dipolo, línea, anillo, disco, plano, esfera y genérica."""

from .dipole import Dipole
from .disk_charge import DiskCharge
from .generic import GenericCharge
from .line_charge import LineCharge
from .plane_charge import PlaneCharge
from .point_charge import PointCharge
from .ring_charge import RingCharge
from .sphere_charge import SphereCharge

__all__ = [
    "PointCharge",
    "Dipole",
    "LineCharge",
    "RingCharge",
    "DiskCharge",
    "PlaneCharge",
    "SphereCharge",
    "GenericCharge",
]
