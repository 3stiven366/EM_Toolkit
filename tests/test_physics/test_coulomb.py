"""Tests de physics/coulomb.py — validación contra solución analítica."""

import numpy as np
import pytest

from em_toolkit.config.constants import k_e
from em_toolkit.physics import coulomb


Q = 1e-9


def test_point_charge_magnitude():
    """|E| a 1 m = k_e q / r²."""
    E = coulomb.point_charge_field([[1.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    assert np.isclose(np.linalg.norm(E), k_e * Q)


def test_point_charge_radial_direction():
    """E apunta radialmente hacia afuera para carga positiva."""
    E = coulomb.point_charge_field([[1.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    assert np.allclose(E[0], [k_e * Q, 0, 0])


def test_inverse_square_law():
    """E cae como 1/r²: factor 4 al doblar la distancia."""
    E1 = coulomb.point_charge_field([[1.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    E2 = coulomb.point_charge_field([[2.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    assert np.isclose(np.linalg.norm(E1) / np.linalg.norm(E2), 4.0)


def test_potential():
    """V = k_e q / r."""
    V = coulomb.point_charge_potential([[1.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    assert np.isclose(V[0], k_e * Q)


def test_potential_inverse_r():
    """V cae como 1/r: factor 2 al doblar la distancia."""
    V1 = coulomb.point_charge_potential([[1.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    V2 = coulomb.point_charge_potential([[2.0, 0, 0]], [0, 0, 0], Q, r_min=0.0)
    assert np.isclose(V1[0] / V2[0], 2.0)


def test_negative_charge_points_inward():
    """El campo de una carga negativa apunta hacia ella."""
    E = coulomb.point_charge_field([[1.0, 0, 0]], [0, 0, 0], -Q, r_min=0.0)
    assert E[0, 0] < 0


def test_superposition_equals_manual_sum(grid_points):
    """La superposición es igual a la suma de contribuciones individuales."""
    srcs = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    qs = np.array([Q, -Q])
    E_sup = coulomb.electric_field(grid_points, srcs, qs, r_min=0.0)
    E_man = (
        coulomb.point_charge_field(grid_points, srcs[0], qs[0], r_min=0.0)
        + coulomb.point_charge_field(grid_points, srcs[1], qs[1], r_min=0.0)
    )
    assert np.allclose(E_sup, E_man)


def test_dipole_bisector_zero_potential():
    """En el plano medio de un dipolo, V = 0 por simetría."""
    srcs = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    qs = np.array([Q, -Q])
    pts = np.array([[1.0, 0.5, 0.3], [1.0, -1.0, 2.0]])
    V = coulomb.potential(pts, srcs, qs, r_min=0.0)
    assert np.allclose(V, 0.0, atol=1e-6)


def test_softening_finite_at_origin():
    """El softening acota el campo en r=0 en vez de dar infinito."""
    E = coulomb.point_charge_field([[0, 0, 0]], [0, 0, 0], Q)
    assert np.all(np.isfinite(E))


def test_softening_follows_theory():
    """El error del softening sigue err ≈ 1.5(r_min/d)²."""
    for d in [2.0, 5.0, 10.0]:
        soft = np.linalg.norm(coulomb.point_charge_field([[d, 0, 0]], [0, 0, 0], Q))
        hard = np.linalg.norm(
            coulomb.point_charge_field([[d, 0, 0]], [0, 0, 0], Q, r_min=0.0)
        )
        err = abs(soft - hard) / hard
        teo = 1.5 * (0.15 / d) ** 2
        assert abs(err - teo) / teo < 0.05


def test_output_shapes(grid_points):
    """Formas de salida: E es (N,3), V es (N,)."""
    srcs = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    qs = np.array([Q, -Q])
    assert coulomb.electric_field(grid_points, srcs, qs).shape == (20, 3)
    assert coulomb.potential(grid_points, srcs, qs).shape == (20,)
