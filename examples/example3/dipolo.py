"""
Escena de dipolo: flechas + líneas de campo.

Primer resultado "de divulgación": combina el campo vectorial (flechas) con las
líneas de campo continuas que nacen en la carga + y entran a la −.

Ejecutar:
    manim -pql dipolo.py Dipolo
"""

import numpy as np
from manim import Create, FadeIn, Scene

from em_toolkit.sources.charges import Dipole
from em_toolkit.mobjects.d2.charges import charge_mobject
from em_toolkit.fields import field_lines_2d, charge_seeds, Normalized, vector_field_2d


class Dipolo(Scene):
    def construct(self):
        # Dipolo: +q en +x, -q en -x, separados 4 unidades (más visible)
        dip = Dipole(q=1e-9, separation=4.0, axis=[1, 0, 0], r_min=0.25)

        region = (np.array([-6.5, -4.0, -1.0]), np.array([6.5, 4.0, 1.0]))

        # Líneas de campo: nacen en ambas cargas
        seeds = charge_seeds([dip.positive, dip.negative], n_per_charge=14, radius=0.35)
        lineas = field_lines_2d(
            dip.electric_field,
            seeds,
            bounds=region,
            step=0.04,
            max_steps=3000,
            stroke_width=2.5,
        )

        # Las dos cargas como mobjects del toolkit
        carga_pos = charge_mobject(dip.positive)
        carga_neg = charge_mobject(dip.negative)

        # Animación: primero las cargas, luego las líneas fluyendo
        self.play(FadeIn(carga_pos), FadeIn(carga_neg))
        self.play(Create(lineas), run_time=3.0)
        self.wait(1.5)
