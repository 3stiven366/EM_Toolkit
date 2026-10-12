"""Tests de sources/currents/ y sources/materials/ — validación analítica."""

import numpy as np
import pytest

from em_toolkit.config.constants import eps0, mu0
from em_toolkit.core.protocols import CurrentSource
from em_toolkit.sources.currents import (
    CurrentLoop,
    CurrentSheet,
    InfiniteWire,
    Solenoid,
    Toroid,
    WireSegment,
)
from em_toolkit.sources.materials import (
    BarMagnet,
    ConductingSphere,
    DielectricSphere,
)


@pytest.mark.parametrize("obj", [
    InfiniteWire(5.0),
    WireSegment(3.0, [0, 0, -1], [0, 0, 1]),
    CurrentLoop(3.0, 0.5),
    Solenoid(2.0, 0.2, 3.0, 50),
    Toroid(3.0, 1.0, 0.2, 100),
    CurrentSheet(10.0),
    BarMagnet(5.0, 0.3),
])
def test_all_currents_implement_protocol(obj):
    assert isinstance(obj, CurrentSource)


# --------------------------- InfiniteWire ---------------------------------

def test_infinite_wire_field():
    """B = μ₀I/2πs."""
    wire = InfiniteWire(5.0, r_min=0.0)
    s = 0.2
    B = wire.magnetic_field([[s, 0, 0]])
    assert np.isclose(np.linalg.norm(B), mu0 * 5.0 / (2 * np.pi * s))


def test_infinite_wire_right_hand_rule():
    """I en +z, punto en +x -> B en +y."""
    wire = InfiniteWire(5.0, r_min=0.0)
    B = wire.magnetic_field([[0.2, 0, 0]])
    assert B[0, 1] > 0 and abs(B[0, 0]) < 1e-12


# --------------------------- WireSegment ----------------------------------

def test_wire_segment_near_infinite_limit():
    """Un segmento visto de cerca (s<<L) ~ hilo infinito."""
    seg = WireSegment(4.0, [0, 0, -1.0], [0, 0, 1.0], n_segments=2000, r_min=0.0)
    s = 0.01
    B = np.linalg.norm(seg.magnetic_field([[s, 0, 0]]))
    assert abs(B - mu0 * 4.0 / (2 * np.pi * s)) / (mu0 * 4.0 / (2 * np.pi * s)) < 0.02


# ---------------------------- CurrentLoop ---------------------------------

def test_loop_center_analytic():
    """B en el centro = μ₀I/2R (vía rama analítica)."""
    loop = CurrentLoop(3.0, 0.5, r_min=0.0)
    assert np.isclose(loop.magnetic_field([[0, 0, 0]])[0, 2], mu0 * 3.0 / (2 * 0.5))


def test_loop_magnetic_moment():
    loop = CurrentLoop(3.0, 0.5)
    assert np.allclose(loop.moment, [0, 0, 3.0 * np.pi * 0.5**2])


def test_loop_axis_continuity():
    """El campo es continuo entre la rama analítica (eje) y la numérica."""
    loop = CurrentLoop(3.0, 0.5, n_segments=800, r_min=0.0)
    B_axis = loop.magnetic_field([[0, 0, 0.5]])[0, 2]
    B_near = loop.magnetic_field([[1e-4, 0, 0.5]])[0, 2]
    assert np.isclose(B_axis, B_near, rtol=1e-3)


# ----------------------------- Solenoid -----------------------------------

def test_solenoid_interior_field():
    """Interior ~ μ₀nI."""
    I, R, L, N = 2.0, 0.2, 8.0, 400
    sol = Solenoid(I, R, L, N, segments_per_turn=40, r_min=0.0)
    B_ideal = mu0 * (N / L) * I
    B = sol.magnetic_field([[0, 0, 0]])[0, 2]
    assert abs(B - B_ideal) / B_ideal < 0.03


def test_solenoid_edge_half_field():
    """En el extremo, B ~ mitad del centro (solenoide semi-infinito)."""
    I, R, L, N = 2.0, 0.2, 8.0, 400
    sol = Solenoid(I, R, L, N, segments_per_turn=40, r_min=0.0)
    B_center = sol.magnetic_field([[0, 0, 0]])[0, 2]
    B_edge = sol.magnetic_field([[0, 0, L / 2]])[0, 2]
    assert abs(B_edge / B_center - 0.5) < 0.1


def test_solenoid_exterior_small():
    """El campo exterior es mucho menor que el interior."""
    I, R, L, N = 2.0, 0.2, 8.0, 400
    sol = Solenoid(I, R, L, N, segments_per_turn=40, r_min=0.0)
    B_ideal = mu0 * (N / L) * I
    B_ext = np.linalg.norm(sol.magnetic_field([[3 * R, 0, 0]]))
    assert B_ext < 0.1 * B_ideal


# ------------------------------ Toroid ------------------------------------

def test_toroid_confined_field():
    """B dentro del núcleo = μ₀NI/2πr."""
    I, R, a, N = 3.0, 1.0, 0.2, 200
    tor = Toroid(I, R, a, N, segments_per_turn=30, r_min=0.0)
    B = np.linalg.norm(tor.magnetic_field([[R, 0, 0]]))
    assert abs(B - mu0 * N * I / (2 * np.pi * R)) / (mu0 * N * I / (2 * np.pi * R)) < 0.03


def test_toroid_zero_in_hole():
    """El campo es casi nulo en el agujero central."""
    I, R, a, N = 3.0, 1.0, 0.2, 200
    tor = Toroid(I, R, a, N, segments_per_turn=30, r_min=0.0)
    B_in = np.linalg.norm(tor.magnetic_field([[R, 0, 0]]))
    B_hole = np.linalg.norm(tor.magnetic_field([[0, 0, 0]]))
    assert B_hole < 0.05 * B_in


# --------------------------- CurrentSheet ---------------------------------

def test_sheet_uniform_field():
    """B = μ₀K/2 uniforme."""
    sheet = CurrentSheet(10.0, normal=[0, 0, 1], current_direction=[1, 0, 0])
    B_ideal = mu0 * 10.0 / 2
    for z in [0.1, 1.0, 50.0]:
        assert np.isclose(np.linalg.norm(sheet.magnetic_field([[0, 0, z]])), B_ideal)


def test_sheet_opposite_sides():
    sheet = CurrentSheet(10.0, normal=[0, 0, 1], current_direction=[1, 0, 0])
    Bp = sheet.magnetic_field([[0, 0, 1.0]])
    Bn = sheet.magnetic_field([[0, 0, -1.0]])
    assert np.allclose(Bp, -Bn)


# ----------------------------- BarMagnet ----------------------------------

def test_magnet_dipole_axis_field():
    """B en el eje = μ₀m/2πz³."""
    m = 5.0
    mag = BarMagnet(m, 0.3, loop_radius=0.02, r_min=0.0)
    z = 2.0
    B = mag.magnetic_field([[0, 0, z]])
    assert abs(B[0, 2] - mu0 * m / (2 * np.pi * z**3)) / (mu0 * m / (2 * np.pi * z**3)) < 0.01


def test_magnet_axis_equator_ratio():
    """B_eje/B_ecuador = 2 (firma del dipolo)."""
    mag = BarMagnet(5.0, 0.3, loop_radius=0.02, r_min=0.0)
    Be = abs(mag.magnetic_field([[0, 0, 2.0]])[0, 2])
    Beq = abs(mag.magnetic_field([[2.0, 0, 0]])[0, 2])
    assert np.isclose(Be / Beq, 2.0, rtol=1e-2)


# -------------------------- ConductingSphere ------------------------------

def test_conductor_zero_inside():
    sph = ConductingSphere(1.0, [0, 0, 1000.0])
    assert np.allclose(sph.total_field([[0, 0, 0], [0.5, 0, 0]]), 0, atol=1e-9)


def test_conductor_three_E0_at_pole():
    """E = 3E₀ en el polo (resultado clásico)."""
    E0 = 1000.0
    sph = ConductingSphere(1.0, [0, 0, E0])
    assert np.isclose(sph.total_field([[0, 0, 1.0]])[0, 2], 3 * E0, rtol=1e-3)


# -------------------------- DielectricSphere ------------------------------

def test_dielectric_interior_reduced():
    """E_in = E₀·3/(κ+2)."""
    E0 = np.array([0, 0, 1000.0])
    kappa = 4.0
    die = DielectricSphere(1.0, kappa, E0)
    assert np.allclose(die.interior_field, E0 * 3 / (kappa + 2))


def test_dielectric_limit_is_conductor():
    """κ→∞ reproduce la esfera conductora."""
    E0 = np.array([0, 0, 1000.0])
    die = DielectricSphere(1.0, 1e8, E0)
    cond = ConductingSphere(1.0, E0)
    pts = np.array([[2.0, 0, 0], [0, 0, 3.0]])
    assert np.allclose(die.total_field(pts), cond.total_field(pts), rtol=1e-5)


def test_dielectric_limit_no_material():
    """κ=1 no altera el campo."""
    E0 = np.array([0, 0, 1000.0])
    die = DielectricSphere(1.0, 1.0, E0)
    assert np.allclose(die.total_field([[0, 0, 0], [5, 0, 0]]), E0)
