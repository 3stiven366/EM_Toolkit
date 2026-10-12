"""Tests de physics/biot_savart.py — validación contra solución analítica."""

import numpy as np

from em_toolkit.config.constants import mu0
from em_toolkit.physics import biot_savart


I = 3.0
R = 0.5


def _loop(t):
    return np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=-1)


def test_loop_center_field():
    """B en el centro de una espira = μ₀I/2R."""
    B = biot_savart.biot_savart_path(
        [[0, 0, 0]], _loop, I, t_range=(0, 2 * np.pi), n_segments=800, closed=True, r_min=0.0
    )
    assert np.isclose(np.linalg.norm(B), mu0 * I / (2 * R), rtol=1e-3)


def test_loop_center_direction():
    """B en +z para corriente antihoraria (mano derecha)."""
    B = biot_savart.biot_savart_path(
        [[0, 0, 0]], _loop, I, t_range=(0, 2 * np.pi), n_segments=800, closed=True, r_min=0.0
    )
    assert B[0, 2] > 0
    assert abs(B[0, 0]) < 1e-9 and abs(B[0, 1]) < 1e-9


def test_loop_on_axis():
    """B(z) en el eje = μ₀IR²/2(R²+z²)^1.5."""
    z = 0.8
    B = biot_savart.biot_savart_path(
        [[0, 0, z]], _loop, I, t_range=(0, 2 * np.pi), n_segments=800, closed=True, r_min=0.0
    )
    expected = mu0 * I * R**2 / (2 * (R**2 + z**2) ** 1.5)
    assert np.isclose(B[0, 2], expected, rtol=1e-3)


def test_linearity_in_current():
    """B es lineal en la corriente."""
    args = dict(t_range=(0, 2 * np.pi), n_segments=400, closed=True, r_min=0.0)
    B1 = biot_savart.biot_savart_path([[0, 0, 0.5]], _loop, 1.0, **args)
    B5 = biot_savart.biot_savart_path([[0, 0, 0.5]], _loop, 5.0, **args)
    assert np.allclose(B5, 5 * B1)


def test_sign_inverts_field():
    """Corriente negativa invierte el campo."""
    args = dict(t_range=(0, 2 * np.pi), n_segments=800, closed=True, r_min=0.0)
    B = biot_savart.biot_savart_path([[0, 0, 0]], _loop, I, **args)
    Bneg = biot_savart.biot_savart_path([[0, 0, 0]], _loop, -I, **args)
    assert np.allclose(Bneg, -B)


def test_convergence_with_segments():
    """El error decrece monótonamente al refinar la discretización."""
    exact = mu0 * I / (2 * R)
    errs = []
    for n in [50, 100, 400, 1600]:
        B = biot_savart.biot_savart_path(
            [[0, 0, 0]], _loop, I, t_range=(0, 2 * np.pi), n_segments=n, closed=True, r_min=0.0
        )
        errs.append(abs(np.linalg.norm(B) - exact) / exact)
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))


def test_infinite_wire_approximation():
    """Un hilo recto largo se aproxima a μ₀I/2πr."""
    L = 400.0

    def wire(t):
        return np.stack(
            [np.zeros_like(t), np.zeros_like(t), t * L / 2], axis=-1
        )

    d = 0.3
    B = biot_savart.biot_savart_path(
        [[d, 0, 0]], wire, I, t_range=(-1, 1), n_segments=4000, closed=False, r_min=0.0
    )
    expected = mu0 * I / (2 * np.pi * d)
    assert np.isclose(np.linalg.norm(B), expected, rtol=1e-2)
