"""Tests de physics/: forces, induction, energy, superposition, sampling."""

import numpy as np

from em_toolkit.config.constants import c, eps0, mu0
from em_toolkit.physics import (
    energy,
    forces,
    induction,
    sampling,
    superposition,
)


# ----------------------------- forces -------------------------------------

def test_lorentz_electric_only():
    assert np.allclose(forces.lorentz_force(2.0, [0, 0, 0], [1, 0, 0], 0), [2, 0, 0])


def test_lorentz_parallel_velocity_no_force():
    assert np.allclose(forces.lorentz_force(2.0, [0, 0, 1], 0, [0, 0, 1]), 0)


def test_lorentz_perpendicular():
    """v=+x, B=+z -> F = q(v×B) = q(-y)."""
    assert np.allclose(forces.lorentz_force(2.0, [1, 0, 0], 0, [0, 0, 1]), [0, -2, 0])


def test_magnetic_force_perpendicular_to_velocity():
    v = np.array([1.0, 2.0, 3.0])
    F = forces.lorentz_force(1.0, v, 0, [0.0, 1.0, -1.0])
    assert abs(np.dot(F, v)) < 1e-12


def test_parallel_wires_attract():
    assert forces.force_between_parallel_wires(3.0, 3.0, 0.1) > 0


def test_antiparallel_wires_repel():
    assert forces.force_between_parallel_wires(3.0, -3.0, 0.1) < 0


# ---------------------------- induction -----------------------------------

def test_flux_perpendicular():
    assert np.isclose(induction.flux_uniform([0, 0, 2.0], [0, 0, 1.0], 3.0), 6.0)


def test_flux_parallel_zero():
    assert np.isclose(induction.flux_uniform([2.0, 0, 0], [0, 0, 1.0], 3.0), 0.0)


def test_faraday_constant_rate():
    """Flujo lineal -> fem constante = -dΦ/dt."""
    t = np.linspace(0, 1, 50)
    emf = induction.emf_from_flux(3.0 * t, t)
    assert np.allclose(emf, -3.0, atol=1e-9)


def test_lenz_sign():
    """Flujo creciente -> fem negativa."""
    t = np.linspace(0, 1, 50)
    assert np.all(induction.emf_from_flux(3.0 * t, t) < 0)


def test_faraday_sinusoidal():
    """Φ=sin(ωt) -> fem=-ω cos(ωt)."""
    w = 2 * np.pi
    t = np.linspace(0, 1, 1000)
    emf = induction.emf_from_flux(np.sin(w * t), t)
    assert np.allclose(emf, -w * np.cos(w * t), atol=1e-2)


# ------------------------------ energy ------------------------------------

def test_electric_energy_density():
    assert np.isclose(energy.electric_energy_density([3.0, 4.0, 0]), 0.5 * eps0 * 25)


def test_magnetic_energy_density():
    assert np.isclose(energy.magnetic_energy_density([0, 0, 2.0]), 4 / (2 * mu0))


def test_em_wave_equipartition():
    """En una onda EM (E=cB), las densidades de energía coinciden."""
    B = 1e-6
    uE = energy.electric_energy_density([c * B, 0, 0])
    uB = energy.magnetic_energy_density([0, 0, B])
    assert np.isclose(uE, uB, rtol=1e-9)


def test_parallel_plate_capacitance():
    assert np.isclose(
        energy.parallel_plate_capacitance(0.01, 0.001), eps0 * 0.01 / 0.001
    )


def test_dielectric_increases_capacitance():
    base = energy.parallel_plate_capacitance(0.01, 0.001)
    with_k = energy.parallel_plate_capacitance(0.01, 0.001, kappa=4.0)
    assert np.isclose(with_k, 4 * base)


# --------------------------- superposition --------------------------------

class _FakeCharge:
    def __init__(self, v):
        self.v = v

    def electric_field(self, p):
        return np.full_like(np.atleast_2d(p), self.v, dtype=float)

    def potential(self, p):
        return np.full(len(np.atleast_2d(p)), self.v * 10.0)

    def total_charge(self):
        return self.v


class _FakeWire:
    def __init__(self, cur):
        self.current = cur

    def magnetic_field(self, p):
        return np.full_like(np.atleast_2d(p), self.current, dtype=float)


def test_superposition_filters_by_protocol():
    pts = np.zeros((5, 3))
    srcs = [_FakeCharge(2.0), _FakeCharge(3.0), _FakeWire(1.5)]
    assert np.allclose(superposition.total_electric_field(pts, srcs), 5.0)
    assert np.allclose(superposition.total_magnetic_field(pts, srcs), 1.5)


def test_superposition_empty():
    pts = np.zeros((5, 3))
    assert np.allclose(superposition.total_electric_field(pts, []), 0.0)


# ----------------------------- sampling -----------------------------------

def test_grid_2d_shape():
    assert sampling.grid_2d((-1, 1), (-1, 1), 5).shape == (25, 3)


def test_grid_3d_shape():
    assert sampling.grid_3d((-1, 1), (-1, 1), (-1, 1), 4).shape == (64, 3)


def test_ring_seeds_on_circle():
    s = sampling.ring_seeds([0, 0, 0], 2.0, 8, "xy")
    assert np.allclose(np.linalg.norm(s, axis=1), 2.0)


def test_field_line_constant_field():
    """En campo constante +x, la línea es recta con paso de arco = step."""
    line = sampling.integrate_field_line(
        lambda p: np.tile([1.0, 0, 0], (len(p), 1)), [0, 0, 0], step=0.1, max_steps=50
    )
    assert np.allclose(line[:, 1], 0) and np.allclose(line[:, 2], 0)
    assert np.isclose(line[-1, 0], 50 * 0.1, atol=1e-6)


def test_field_line_stops_at_null():
    """Campo nulo -> la línea termina de inmediato."""
    line = sampling.integrate_field_line(
        lambda p: np.zeros_like(p), [0, 0, 0], step=0.1, max_steps=100
    )
    assert len(line) == 1
