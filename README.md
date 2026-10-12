# em_toolkit

Librería reutilizable de objetos de electromagnetismo para animaciones con Manim.

Separa el **modelo físico** (puro NumPy, sin Manim) de la **representación visual**,
de modo que la física se puede verificar de forma independiente del render.

## Instalación

```bash
pip install -e .            # solo física (numpy, scipy)
pip install -e ".[viz]"     # + Manim para la capa visual
pip install -e ".[dev]"     # + pytest para las pruebas
```

## Pruebas

```bash
pytest                      # ejecuta toda la validación física
```

## Estado

| Capa | Estado |
|------|--------|
| config, core, physics | completo y probado |
| sources (cargas, corrientes, materiales) | completo y probado (18 clases) |
| fields, mobjects, circuits, systems, animations, scenes | en desarrollo (capa visual) |

Toda la capa física está validada contra soluciones analíticas conocidas.
