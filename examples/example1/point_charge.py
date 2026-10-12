"""
Escena mínima de prueba: carga puntual y su campo eléctrico en 2D.

Propósito: confirmar que em_toolkit (física validada) + ManimCE funcionan
juntos de principio a fin, antes de construir la capa visual completa.

Ejecutar:
    manim -pql escena_prueba.py CargaPuntualCampo

    -p  reproduce al terminar
    -ql calidad baja (rápido para pruebas); usa -qh para alta calidad
"""

import numpy as np
from manim import (
    BLUE,
    RED,
    WHITE,
    Arrow,
    Create,
    Dot,
    FadeIn,
    Scene,
    Tex,
    VGroup,
    config,
)

from em_toolkit.sources import PointCharge


class CargaPuntualCampo(Scene):
    def construct(self):
        # --- Física: una carga puntual del toolkit --------------------------
        # q positiva en el origen. r_min evita que las flechas exploten cerca.
        carga = PointCharge(q=1e-9, position=[0, 0, 0], r_min=0.3)

        # --- Grilla de puntos donde evaluamos el campo (plano z=0) ----------
        paso = 0.9
        coords = np.arange(-6, 6 + paso, paso)
        puntos = np.array(
            [[x, y, 0.0] for x in coords for y in coords]
        )
        # Quitar el punto central (sobre la carga)
        puntos = puntos[np.linalg.norm(puntos, axis=1) > 1e-6]

        # --- Evaluar el campo con la física validada ------------------------
        E = carga.electric_field(puntos)              # (N, 3) en V/m
        magnitudes = np.linalg.norm(E, axis=1)

        # Normalización para color: 0..1 según magnitud (escala log para rango)
        log_mag = np.log10(magnitudes + 1e-30)
        lo, hi = log_mag.min(), log_mag.max()
        intensidad = (log_mag - lo) / (hi - lo + 1e-12)

        # --- Construir las flechas ------------------------------------------
        flechas = VGroup()
        largo_fijo = 0.55                              # largo uniforme (Normalized)
        for p, e_vec, t in zip(puntos, E, intensidad):
            direccion = e_vec / (np.linalg.norm(e_vec) + 1e-30)
            inicio = p
            fin = p + direccion * largo_fijo
            # Color: azul (débil) -> rojo (fuerte)
            color = _interp_color(BLUE, RED, t)
            flecha = Arrow(
                start=inicio,
                end=fin,
                buff=0,
                stroke_width=2.5,
                max_tip_length_to_length_ratio=0.35,
                color=color,
            )
            flechas.add(flecha)

        # --- La carga: punto rojo con signo + ------------------------------
        punto_carga = Dot(point=[0, 0, 0], radius=0.18, color=RED)
        signo = Tex("$+$", color=WHITE).scale(0.7).move_to([0, 0, 0])
        carga_visual = VGroup(punto_carga, signo)

        # --- Animación ------------------------------------------------------
        self.play(FadeIn(carga_visual))
        self.play(Create(flechas), run_time=2.5)
        self.wait(1.5)


def _interp_color(c1, c2, t):
    """Interpola linealmente entre dos colores de Manim (t en [0,1])."""
    from manim import interpolate_color
    return interpolate_color(c1, c2, float(np.clip(t, 0, 1)))
