"""
Escena de prueba de la arquitectura formal.

Reproduce la carga puntual + campo de la escena ad-hoc, pero usando los
componentes del toolkit (charge_mobject + vector_field_2d) en vez de código
suelto. Si se ve igual que escena_prueba.py, la capa visual funciona.

Ejecutar:
    manim -pql escena_formal.py CargaFormal
"""

import numpy as np
from manim import Create, FadeIn, Scene

from em_toolkit.sources import PointCharge
from em_toolkit.mobjects.d2.charges import charge_mobject
from em_toolkit.fields.vector_field import vector_field_2d
from em_toolkit.fields.scaling import Normalized


class CargaFormal(Scene):
    def construct(self):
        # Física: una carga del toolkit
        carga = PointCharge(q=1e-9, position=[0, 0, 0], r_min=0.3)

        # Grilla de evaluación (plano z=0), sin el punto sobre la carga
        paso = 0.9
        coords = np.arange(-6, 6 + paso, paso)
        grid = np.array([[x, y, 0.0] for x in coords for y in coords])
        grid = grid[np.linalg.norm(grid, axis=1) > 1e-6]

        # Campo de flechas con la arquitectura formal
        flechas = vector_field_2d(
            carga.electric_field,
            grid,
            scaling=Normalized(length=0.55),
        )

        # La carga como mobject del toolkit
        carga_visual = charge_mobject(carga)

        # Animación
        self.play(FadeIn(carga_visual))
        self.play(Create(flechas), run_time=2.5)
        self.wait(1.5)
