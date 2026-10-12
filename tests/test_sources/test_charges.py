"""Tests de sources/charges/ — validación contra solución analítica."""

import numpy as np
import pytest

from em_toolkit.config.constants import eps0, k_e
from em_toolkit.core.protocols import ChargeSource
from em_toolkit.sources.charges import (
    Dipole,
    DiskCharge,
    GenericCharge,
    LineCharge,
    PlaneCharge,
    PointCharge,
    RingCharge,
    SphereCharge,
)

Q = 1e-9


@pytest.mark.parametrize("cls,args", [
    (PointCharge, (Q,)),
    (Dipole, (Q, 0.2)),
    (LineCharge, (Q,)),
    (RingCharge, (Q, 0.5)),
    (DiskCharge, (Q, 1.0)),
    (PlaneCharge, (Q,)),
    (SphereCharge, (Q, 1.0)),
    (GenericCharge, ([[0, 0, 0]], [Q])),
])
def test_all_charges_implement_protocol(cls, args):
    """Toda fuente de carga cumple el protocolo ChargeSource."""
    if cls is LineCharge:
        obj = cls(*args, length=1.0)
    else:
        obj = cls(*args)
    assert isinstance(obj, ChargeSource)


# --------------------------- PointCharge ----------------------------------

def test_point_charge_field():
    q = PointCharge(Q, r_min=0.0)
    assert np.isclose(np.linalg.norm(q.electric_field([[1.0, 0, 0]])), k_e * Q)


def test_point_charge_move_shifts_field():
    q = PointCharge(Q, r_min=0.0).move_to([3, 0, 0])
    assert np.isclose(np.linalg.norm(q.electric_field([[4.0, 0, 0]])), k_e * Q)


# ----------------------------- Dipole -------------------------------------

def test_dipole_neutral():
    assert Dipole(Q, 0.2).total_charge() == 0.0


def test_dipole_bisector_zero_potential():
    d = Dipole(Q, 0.2, axis=[1, 0, 0], r_min=0.0)
    assert np.allclose(d.potential([[0, 1.0, 0], [0, 0, 2.0]]), 0.0, atol=1e-6)


# ---------------------------- RingCharge ----------------------------------

def test_ring_on_axis():
    """E_z en el eje = kQz/(R²+z²)^1.5."""
    R = 0.5
    ring = RingCharge(Q, R, n_segments=400, r_min=0.0)
    z = 1.0
    E = ring.electric_field([[0, 0, z]])
    assert np.isclose(E[0, 2], k_e * Q * z / (R**2 + z**2) ** 1.5, rtol=1e-3)


def test_ring_center_zero_field():
    ring = RingCharge(Q, 0.5, n_segments=400, r_min=0.0)
    assert np.allclose(ring.electric_field([[0, 0, 0]]), 0, atol=1e-12)


# ---------------------------- PlaneCharge ---------------------------------

def test_plane_uniform_field():
    sigma = Q
    plane = PlaneCharge(sigma)
    E0 = sigma / (2 * eps0)
    for z in [0.1, 1.0, 100.0]:
        assert np.isclose(np.linalg.norm(plane.electric_field([[0, 0, z]])), E0)


def test_plane_field_consistency_with_potential():
    """-dV/dd = E (consistencia campo-potencial)."""
    sigma = Q
    plane = PlaneCharge(sigma)
    V1 = plane.potential([[0, 0, 1.0]])[0]
    V2 = plane.potential([[0, 0, 2.0]])[0]
    assert np.isclose(-(V2 - V1), sigma / (2 * eps0))


# ---------------------------- SphereCharge --------------------------------

def test_shell_zero_inside():
    """Ley de Gauss: E=0 dentro de un cascarón."""
    shell = SphereCharge(Q, 1.0, shell=True)
    for r in [0.1, 0.5, 0.99]:
        assert np.allclose(shell.electric_field([[r, 0, 0]]), 0, atol=1e-15)


def test_shell_outside_like_point():
    """Fuera del cascarón = carga puntual."""
    shell = SphereCharge(Q, 1.0, shell=True)
    pc = PointCharge(Q, r_min=0.0)
    pts = np.array([[2.0, 0, 0], [0, 3.0, 0]])
    assert np.allclose(shell.electric_field(pts), pc.electric_field(pts))


def test_volume_linear_inside():
    """Dentro del volumen uniforme E=kQr/R³ (lineal)."""
    R = 1.0
    vol = SphereCharge(Q, R, shell=False)
    for r in [0.2, 0.5, 0.8]:
        E = vol.electric_field([[r, 0, 0]])
        assert np.isclose(np.linalg.norm(E), k_e * Q * r / R**3)


def test_volume_potential_center():
    """V en el centro del volumen = 3kQ/2R."""
    vol = SphereCharge(Q, 1.0, shell=False)
    assert np.isclose(vol.potential([[0, 0, 0]])[0], 3 * k_e * Q / 2)


# ---------------------------- LineCharge ----------------------------------

def test_infinite_line_field():
    """Hilo infinito: E = λ/(2πε₀s)."""
    line = LineCharge(Q, infinite=True, r_min=0.0)
    s = 0.5
    E = line.electric_field([[s, 0, 0]])
    assert np.isclose(np.linalg.norm(E), Q / (2 * np.pi * eps0 * s))


def test_finite_line_far_like_point():
    """Línea finita lejos = carga puntual Q=λL."""
    line = LineCharge(Q, length=2.0, n_segments=500, r_min=0.0)
    s = 50.0
    E = np.linalg.norm(line.electric_field([[s, 0, 0]]))
    assert abs(E - k_e * (Q * 2.0) / s**2) / (k_e * Q * 2.0 / s**2) < 0.02


# ---------------------------- GenericCharge -------------------------------

def test_generic_single_equals_point():
    g = GenericCharge([[0, 0, 0]], [2e-9], r_min=0.0)
    pc = PointCharge(2e-9, r_min=0.0)
    pts = np.array([[1.0, 0, 0], [0, 2.0, 0]])
    assert np.allclose(g.electric_field(pts), pc.electric_field(pts))
