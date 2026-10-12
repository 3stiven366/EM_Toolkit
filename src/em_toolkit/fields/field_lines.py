"""
Líneas de campo continuas.

Traza líneas de campo integrando el campo desde puntos semilla (RK4, vía
`physics.sampling.integrate_field_line`) y las convierte en curvas de Manim.
A diferencia de las flechas, muestran la trayectoria continua del campo: el
look clásico de los videos de divulgación.

El color de cada línea codifica la magnitud media del campo a lo largo de ella
(colormap). Importa Manim solo a través de `_backend`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import defaults
from ..physics.sampling import integrate_field_line, ring_seeds


def charge_seeds(
    charges,
    n_per_charge: int = 12,
    radius: float = 0.3,
    plane: str = "xy",
) -> list[tuple[np.ndarray, int]]:
    """Genera semillas alrededor de cada carga, con su sentido de trazado.

    Las líneas nacen en las cargas positivas (sentido +1) y en las negativas
    (sentido -1, se integran contra el campo para que entren a la carga).

    Args:
        charges: iterable de objetos con `.position` y `.sign`.
        n_per_charge: semillas por carga.
        radius: radio del anillo de semillas alrededor de cada carga.
        plane: plano del anillo ("xy", "xz", "yz").

    Returns:
        Lista de (semilla, sentido), con sentido +1 o -1.
    """
    seeds = []
    for q in charges:
        if q.sign == 0:
            continue
        ring = ring_seeds(q.position, radius, n_per_charge, plane=plane)
        sentido = 1 if q.sign > 0 else -1
        for s in ring:
            seeds.append((s, sentido))
    return seeds


def field_lines_2d(
    field_func: Callable[[np.ndarray], np.ndarray],
    seeds: list[tuple[np.ndarray, int]] | np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    step: float = 0.05,
    max_steps: int = 2000,
    colormap: str | None = None,
    stroke_width: float = 2.5,
    min_field: float = 1e-12,
):
    """Construye un VGroup de líneas de campo en 2D.

    Args:
        field_func: función (N,3) -> (N,3) que evalúa el campo.
        seeds: lista de (punto, sentido) con sentido ±1, o un array (M,3) de
            puntos (todos con sentido +1).
        bounds: (min_xyz, max_xyz) para recortar las líneas a la región visible.
        step: paso de arco de la integración.
        max_steps: tope de puntos por línea.
        colormap: colormap para el color por magnitud. Por defecto, el del paquete.
        stroke_width: grosor de las líneas.
        min_field: magnitud mínima antes de cortar la línea (punto crítico).

    Returns:
        Un VGroup con una curva por semilla (las degeneradas se descartan).
    """
    from ..mobjects._backend import VGroup, VMobject, colormap_color

    cmap = defaults.FIELD_DEFAULTS["colormap"] if colormap is None else colormap

    # Normalizar seeds a lista de (punto, sentido)
    seeds_norm = []
    for item in seeds:
        if isinstance(item, tuple):
            seeds_norm.append((np.asarray(item[0], dtype=float), int(item[1])))
        else:
            seeds_norm.append((np.asarray(item, dtype=float), 1))

    # Para colorear por magnitud, necesitamos el rango global del campo en las líneas
    trayectorias = []
    mags_medias = []
    for seed, sentido in seeds_norm:
        # El sentido -1 se logra integrando el campo invertido
        if sentido >= 0:
            ff = field_func
        else:
            ff = lambda p, _f=field_func: -_f(p)
        linea = integrate_field_line(
            ff, seed, step=step, max_steps=max_steps,
            bounds=bounds, min_field=min_field,
        )
        if linea.shape[0] < 2:
            continue                       # línea degenerada
        mag = np.linalg.norm(field_func(linea), axis=1)
        trayectorias.append(linea)
        mags_medias.append(float(np.mean(mag)))

    if not trayectorias:
        return VGroup()

    # Normalizar colores por el rango logarítmico de magnitudes medias
    mm = np.array(mags_medias)
    logm = np.log10(np.maximum(mm, mm[mm > 0].min() * 1e-6)) if np.any(mm > 0) else mm
    lo, hi = logm.min(), logm.max()
    t_colores = (logm - lo) / (hi - lo) if hi - lo > 1e-12 else np.full_like(mm, 0.5)

    grupo = VGroup()
    for linea, t in zip(trayectorias, t_colores):
        curva = VMobject(stroke_width=stroke_width)
        curva.set_points_as_corners(linea)
        curva.set_color(colormap_color(float(t), cmap))
        grupo.add(curva)

    return grupo


__all__ = ["field_lines_2d", "charge_seeds"]
