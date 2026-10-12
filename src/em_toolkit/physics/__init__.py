"""Física pura del electromagnetismo: campos, fuerzas, inducción, energía y muestreo. Sin Manim."""

from . import (
    biot_savart,
    coulomb,
    energy,
    forces,
    induction,
    sampling,
    superposition,
)
from .biot_savart import biot_savart_discrete, biot_savart_path
from .coulomb import electric_field, point_charge_field, point_charge_potential, potential
from .energy import (
    capacitor_energy,
    electric_energy_density,
    inductor_energy,
    magnetic_energy_density,
    parallel_plate_capacitance,
    solenoid_inductance,
)
from .forces import force_between_parallel_wires, lorentz_force
from .induction import emf_from_flux, flux_uniform, magnetic_flux, motional_emf
from .sampling import grid_2d, grid_3d, integrate_field_line, ring_seeds
from .superposition import (
    total_electric_field,
    total_magnetic_field,
    total_potential,
)

__all__ = [
    "coulomb", "biot_savart", "superposition", "forces",
    "induction", "energy", "sampling",
    "point_charge_field", "point_charge_potential", "electric_field", "potential",
    "biot_savart_discrete", "biot_savart_path",
    "total_electric_field", "total_potential", "total_magnetic_field",
    "lorentz_force", "force_between_parallel_wires",
    "magnetic_flux", "flux_uniform", "emf_from_flux", "motional_emf",
    "electric_energy_density", "magnetic_energy_density",
    "capacitor_energy", "inductor_energy",
    "parallel_plate_capacitance", "solenoid_inductance",
    "grid_2d", "grid_3d", "ring_seeds", "integrate_field_line",
]
