"""Configuración compartida de pytest: fixtures y tolerancias comunes."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Generador aleatorio reproducible."""
    return np.random.default_rng(42)


@pytest.fixture
def grid_points(rng):
    """Conjunto de puntos de prueba (20, 3) en una región moderada."""
    return rng.uniform(-3, 3, size=(20, 3))
