"""Tests de core/ y config/ — infraestructura base."""

import math

import numpy as np
import pytest

from em_toolkit.config import constants as k
from em_toolkit.config.units import UnitScale
from em_toolkit.core import registry
from em_toolkit.core.exceptions import EMToolkitError, RegistryError
from em_toolkit.core.frame import Frame
from em_toolkit.core.protocols import ChargeSource, CurrentSource


# ----------------------------- config -------------------------------------

def test_constants_exact_relations():
    """c² = 1/(μ₀ε₀) y k_e = 1/(4πε₀)."""
    assert abs(k.c**2 - 1 / (k.mu0 * k.eps0)) / k.c**2 < 1e-12
    assert abs(k.k_e - 1 / (4 * math.pi * k.eps0)) < 1e-3


def test_unit_scale_roundtrip():
    s = UnitScale(meters_per_unit=0.5)
    assert np.isclose(s.to_si(s.to_scene(3.7)), 3.7)


def test_unit_scale_rejects_nonpositive():
    with pytest.raises(ValueError):
        UnitScale(meters_per_unit=0)


# ------------------------------ frame -------------------------------------

def test_frame_point_roundtrip():
    f = Frame.from_axis_angle([1, 1, 0], 0.7, origin=[2, -1, 3])
    pts = np.random.rand(10, 3)
    assert np.allclose(f.to_local(f.to_global(pts)), pts)


def test_frame_rotation_orthogonal():
    f = Frame.from_axis_angle([1, 1, 0], 0.7)
    assert np.allclose(f.rotation @ f.rotation.T, np.eye(3))
    assert np.isclose(np.linalg.det(f.rotation), 1.0)


def test_frame_from_z_to():
    """El eje +z local apunta a la dirección pedida."""
    target = np.array([1.0, 2.0, -1.0])
    f = Frame.from_z_to(target)
    assert np.allclose(
        f.rotate_to_global([0, 0, 1]), target / np.linalg.norm(target), atol=1e-9
    )


def test_frame_from_z_to_antiparallel():
    """Caso límite: dirección -z."""
    f = Frame.from_z_to([0, 0, -1])
    assert np.allclose(f.rotate_to_global([0, 0, 1]), [0, 0, -1], atol=1e-9)


# ----------------------------- registry -----------------------------------

def test_registry_dispatch_and_mro():
    registry.clear_registry()

    class Charge:
        pass

    class PointCharge(Charge):
        pass

    @registry.register(PointCharge, dim=3)
    def _(obj, style):
        return "sphere3d"

    @registry.register(Charge, dim=2)
    def _(obj, style):
        return "dot2d"

    pc = PointCharge()
    assert registry.build(pc, 3) == "sphere3d"
    assert registry.build(pc, 2) == "dot2d"  # vía MRO desde la base
    registry.clear_registry()


def test_registry_missing_raises():
    registry.clear_registry()

    class Thing:
        pass

    with pytest.raises(RegistryError):
        registry.build(Thing(), 3)


def test_registry_rejects_bad_dim():
    with pytest.raises(ValueError):
        @registry.register(object, dim=4)
        def _(o, s):
            return None


# --------------------------- exceptions -----------------------------------

def test_exception_hierarchy():
    assert issubclass(RegistryError, EMToolkitError)
