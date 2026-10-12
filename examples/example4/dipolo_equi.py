"""
Dipolo: líneas de campo + equipotenciales.

Muestra la relación perpendicular entre ambas: las líneas de campo (que salen de
+ y entran a −) cruzan las equipotenciales (óvalos alrededor de cada carga)
siempre en ángulo recto.

Ejecutar:
    manim -pql dipolo_equi.py DipoloEqui
"""

import numpy as np
from manim import Create, FadeIn, Scene

from em_toolkit.sources.charges import Dipole
from em_toolkit.mobjects.d2.charges import charge_mobject
from em_toolkit.fields import charge_seeds, equipotential_2d, field_lines_2d


class DipoloEqui(Scene):
    def construct(self):
        dip = Dipole(q=1e-9, separation=4.0, axis=[1, 0, 0], r_min=0.25)
        region = (np.array([-6.5, -4.0, -1.0]), np.array([6.5, 4.0, 1.0]))

        # Equipotenciales (al fondo, más tenues)
        equis = equipotential_2d(
            dip.potential,
            x_range=(-6.5, 6.5),
            y_range=(-4.0, 4.0),
            levels=16,
            resolution=350,
            stroke_width=1.5,
        )

        # Líneas de campo (encima de las equipotenciales)
        seeds = charge_seeds([dip.positive, dip.negative], n_per_charge=14, radius=0.4)
        lineas = field_lines_2d(
            dip.electric_field, seeds, bounds=region,
            step=0.04, max_steps=3000, stroke_width=2.5,
        )

        carga_pos = charge_mobject(dip.positive)
        carga_neg = charge_mobject(dip.negative)

        # Orden: equipotenciales al fondo, líneas encima, cargas al frente
        self.play(Create(equis), run_time=2.5)
        self.play(Create(lineas), run_time=2.5)
        self.play(FadeIn(carga_pos), FadeIn(carga_neg))
        self.bring_to_front(carga_pos, carga_neg)
        self.wait(1.5)
