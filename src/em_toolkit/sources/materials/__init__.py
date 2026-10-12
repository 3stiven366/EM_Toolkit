"""Materiales: imán de barra, esfera conductora y esfera dieléctrica en campos externos."""

from .conductor import ConductingSphere
from .dielectric import DielectricSphere
from .magnet import BarMagnet

__all__ = ["BarMagnet", "ConductingSphere", "DielectricSphere"]
