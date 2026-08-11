"""Figura de la Fase 1: el mapa de la cohorte ordenada por identificador.

Que demuestra, en una sola imagen y con los datos leidos del disco (finding 033):

1. Los dos conjuntos NO ocupan tramos separados de la base: sus identificadores
   aparecen entremezclados de principio a fin.
2. Los nueve pacientes con CPC 2 o 3 (los vecinos de la linea de corte) son TODOS
   del conjunto de desarrollo. Es consecuencia de haber elegido ese conjunto
   mirando el nivel exacto de CPC y el de prueba mirando solo Good/Poor.
3. El desenlace aparece agrupado por identificador en los 35 descargados. Se
   dibuja porque se observa; su causa en la base original NO se conoce y por eso
   no se explica en ningun sitio.

Reutiliza el estilo y la paleta de `make_figuras_fase0.py` para que la figura sea
indistinguible de las siete anteriores. Solo lee metadatos y dibuja: no toca el
pipeline, ni las caracteristicas, ni el modelo congelado.

Uso:
    cd development && uv run python scripts/make_figura_fase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402

from cohort_table import build_cohort, verify_against_finding_031  # noqa: E402
from make_figuras_fase0 import (  # noqa: E402
    BAND,
    BLUE,
    INK_2,
    MUTED,
    ORANGE,
    SURFACE,
    TEXT_WIDTH_IN,
    _despine,
    _save,
)

# Como se dibuja cada conjunto. El color distingue los dos conjuntos reales
# (azul/naranja, par validado para daltonismo); los excluidos van en gris y con
# marcador hueco porque no son una tercera categoria, sino un estado descartado.
# La FORMA duplica siempre la informacion del color: en una tesis impresa en
# blanco y negro la figura tiene que seguir leyendose.
GROUP_STYLE = {
    "desarrollo": {"label": "Desarrollo", "color": BLUE, "marker": "o", "filled": True},
    "held-out": {"label": "Prueba externa", "color": ORANGE, "marker": "^", "filled": True},
    "excluido": {"label": "Excluido", "color": MUTED, "marker": "o", "filled": False},
}

BORDERLINE_CPC = {2, 3}


def figura_mapa_cohorte(cohort) -> None:
    """Dibuja los 35 pacientes: posicion = orden de identificador, altura = nivel de CPC."""
    data = cohort.sort_values("patient").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.15))

    # Franja de fondo: los dos niveles vecinos a la linea de corte. Es el area
    # cuyo contenido constituye el hallazgo, asi que va detras de todo.
    ax.axhspan(1.5, 3.5, color=BAND, zorder=0)
    # La linea de corte entre desenlace favorable (CPC 1-2) y desfavorable (3-5).
    ax.axhline(2.5, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)

    for group, style in GROUP_STYLE.items():
        subset = data[data["group"] == group]
        if subset.empty:
            continue
        ax.scatter(
            subset.index,
            subset["cpc"],
            marker=style["marker"],
            s=34,
            facecolors=style["color"] if style["filled"] else SURFACE,
            edgecolors=style["color"],
            linewidths=1.1,
            label=style["label"],
            zorder=3,
        )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data["patient"], rotation=90, fontsize=5.4)
    ax.set_xlim(-0.9, len(data) - 0.1)

    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(5.6, 0.4)  # CPC 1 arriba: el mejor desenlace en la parte superior
    ax.set_ylabel("Nivel de la escala CPC")
    ax.set_xlabel("Pacientes, ordenados por identificador", labelpad=2)

    # Rotulo de la franja. Va en la altura del CPC 4 porque esa fila esta VACIA en
    # los 35 pacientes: es el unico espacio de la figura donde un texto no tapa
    # ningun dato. La franja sombreada queda justo encima, asi que se lee referida
    # a ella. (Meterlo dentro de la franja chocaba con 0371, 0406 y 0525.)
    ax.text(
        (len(data) - 1) / 2, 4.05,
        "La franja sombreada marca los casos frontera (CPC 2 y 3):\n"
        "los nueve pertenecen al conjunto de desarrollo",
        ha="center", va="center", fontsize=6.6, color=INK_2, linespacing=1.4,
    )
    ax.text(
        -0.5, 2.42, "línea de corte",
        ha="left", va="bottom", fontsize=6.2, color=INK_2, style="italic",
    )

    _despine(ax)
    ax.grid(axis="y", color=BAND, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.35,
              columnspacing=1.4, borderaxespad=0.0)

    _save(fig, "fig08_mapa_cohorte")


if __name__ == "__main__":
    cohort = build_cohort()

    # El disco manda: si dejara de reproducir el finding 031, no se dibuja nada.
    problems = verify_against_finding_031(cohort)
    if problems:
        print("ABORTING: disk no longer reproduces finding 031")
        for problem in problems:
            print(f"  MISMATCH: {problem}")
        raise SystemExit(1)

    # Comprobacion propia de esta figura: si los casos frontera dejaran de estar
    # todos en desarrollo, el rotulo de la figura seria falso. Mejor abortar.
    borderline = cohort[cohort["cpc"].isin(BORDERLINE_CPC)]
    misplaced = borderline[borderline["group"] != "desarrollo"]
    if not misplaced.empty:
        print("ABORTING: borderline cases are no longer all in development")
        print(misplaced[["patient", "group", "cpc"]].to_string(index=False))
        raise SystemExit(1)
    print(f"OK: {len(borderline)}/{len(borderline)} borderline cases in development.")

    figura_mapa_cohorte(cohort)
    print("Done.")
