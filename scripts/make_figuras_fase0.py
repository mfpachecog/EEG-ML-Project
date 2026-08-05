"""Genera las siete figuras de la Fase 0 (Etapa 2) para el capitulo de metodologia.

Las figuras estan especificadas una por una en `docs/tesis_latex/HOJA_DE_RUTA.md` §11.3
y todos sus datos vienen verificados del finding 031. Este script NO recalcula nada del
pipeline de la v1: solo lee metadatos ya verificados (via `cohort_table.py`) y dibuja.

Salida: PDF vectorial en `docs/tesis_latex/proyecto/figuras/`.

Uso, desde `development/`:
    PYTHONPATH=src uv run python scripts/make_figuras_fase0.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cohort_table import DATA_DIR, build_cohort, verify_against_finding_031

# Destino de las figuras: el repo de documentos, no el de codigo.
FIG_DIR = (
    Path(__file__).resolve().parents[2] / "docs" / "tesis_latex" / "proyecto" / "figuras"
)

# --- Paleta -------------------------------------------------------------------
# Slots categoricos 1-3 de la paleta de referencia, validados para daltonismo con
# `--pairs all` en modo claro (peor par deuteranopia dE 9.2, vision normal dE 24.0).
# Toda figura lleva ademas etiqueta directa o leyenda, de modo que el color nunca es
# el unico canal que transporta la informacion.
BLUE = "#2a78d6"      # desenlace favorable / conjunto de desarrollo
ORANGE = "#eb6834"    # desenlace desfavorable / conjunto de prueba externa
AQUA = "#1baf7a"      # tercer nivel cuando hace falta
INK = "#0b0b0b"       # texto principal
INK_2 = "#52514e"     # texto secundario
MUTED = "#898781"     # ejes y etiquetas menores
GRID = "#e1e0d9"      # rejilla, siempre por detras
SURFACE = "#ffffff"   # el papel de la tesis es blanco, no el gris del sistema web
BAND = "#eef2f7"      # sombreado de la franja 24-72 h

# Ancho util de la caja de texto de la tesis, en pulgadas (~15 cm).
TEXT_WIDTH_IN = 5.9

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
    }
)


def _despine(ax, keep=("left", "bottom")) -> None:
    """Quita los bordes que no aportan informacion, dejando solo los ejes utiles."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_linewidth(0.8)


def _save(fig, name: str) -> None:
    """Guarda la figura como PDF vectorial en la carpeta de figuras de la tesis."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out.relative_to(FIG_DIR.parents[3])}")


def _box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=7.5, textcolor=INK, weight="normal"):
    """Dibuja una caja redondeada con texto centrado, para los diagramas de flujo."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=textcolor, linespacing=1.45, fontweight=weight,
    )


def _arrow(ax, start, end, color=MUTED, style="-|>", linewidth=1.0):
    """Flecha fina entre dos puntos, para encadenar las cajas de un diagrama."""
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=9,
            color=color, linewidth=linewidth, shrinkA=1, shrinkB=1,
        )
    )


# =============================================================================
# FIGURA 1 — anatomia de un archivo de cabecera + jerarquia paciente/segmento/epoca
# =============================================================================

def figura_1_estructura_datos() -> None:
    """Cabecera .hea real anotada, junto al esquema paciente -> segmentos -> epocas.

    El archivo elegido es `0284_025_024_EEG.hea`: paciente 0284, segmento 25, hora
    absoluta 24. Se muestran solo algunos canales para que quepa; el resto se indica
    con puntos suspensivos.
    """
    hea_path = DATA_DIR / "0284" / "0284_025_024_EEG.hea"
    raw_lines = hea_path.read_text().splitlines()

    # Se conservan la primera linea (el resumen del registro), tres canales, una
    # elision y las tres lineas de comentario finales.
    shown = raw_lines[:4] + ["        ...  (19 canales en total)  ..."] + raw_lines[-3:]

    fig, (ax_hea, ax_tree) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 3.4), gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.10}
    )

    # --- Panel izquierdo: el archivo de texto, tal cual esta en disco -------------
    ax_hea.axis("off")
    ax_hea.set_title("a) Cabecera real de un segmento", loc="left", color=INK, pad=6)

    ax_hea.add_patch(
        FancyBboxPatch(
            (0.0, 0.28), 1.0, 0.60,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#f7f7f5", edgecolor=GRID, linewidth=1.0,
            transform=ax_hea.transAxes,
        )
    )

    # Las lineas del archivo se dibujan dentro de la caja gris; las anotaciones van
    # DEBAJO, no al lado, para que en ningun caso tapen el texto original.
    y = 0.845
    line_rows: dict[int, float] = {}
    for i, line in enumerate(shown):
        # Se recorta la linea para que no desborde la caja; el detalle relevante
        # (la unidad /nu y el nombre del canal) esta al inicio.
        text = line if len(line) <= 44 else line[:41] + "..."
        ax_hea.text(
            0.022, y, text, transform=ax_hea.transAxes,
            fontsize=4.8, family="monospace", color=INK if i == 0 else INK_2, va="top",
        )
        line_rows[i] = y
        y -= 0.069

    ax_hea.text(
        0.022, 0.245,
        f"archivo: {hea_path.name}",
        transform=ax_hea.transAxes, fontsize=5.4, color=MUTED, va="top", style="italic",
    )

    # Anotaciones de los tres campos que importan, apiladas bajo la caja.
    notes = [
        ("19 canales · 500 Hz · 229 500 muestras", BLUE),
        ("unidades sin calibrar de amplitud (/nu)", ORANGE),
        ("hora absoluta desde el paro: 24 h", AQUA),
    ]
    y_note = 0.155
    for text, color in notes:
        ax_hea.text(
            0.50, y_note, text, transform=ax_hea.transAxes, fontsize=5.8,
            color=color, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.30", facecolor=SURFACE, edgecolor=color, linewidth=0.8),
        )
        y_note -= 0.075

    # --- Panel derecho: la jerarquia de las tres unidades -----------------------
    ax_tree.axis("off")
    ax_tree.set_xlim(0, 1)
    ax_tree.set_ylim(0, 1)
    ax_tree.set_title("b) Cómo se organizan los datos", loc="left", color=INK, pad=6)

    _box(ax_tree, 0.02, 0.79, 0.96, 0.12,
         "PACIENTE  (una carpeta)", "#eaf2fd", BLUE, fontsize=6.8, weight="bold")
    _box(ax_tree, 0.02, 0.44, 0.96, 0.23,
         "SEGMENTOS\n(archivos por hora)\n"
         "un par .hea + .mat cada uno\n"
         "≈ 1 hora, no garantizada",
         "#fdf0ea", ORANGE, fontsize=6.2)
    _box(ax_tree, 0.02, 0.09, 0.96, 0.23,
         "ÉPOCAS\n(ventanas de 10 s)\n"
         "unidad de análisis del modelo\n"
         "resultan del preprocesamiento",
         "#e8f7f1", AQUA, fontsize=6.2)

    _arrow(ax_tree, (0.50, 0.79), (0.50, 0.675))
    _arrow(ax_tree, (0.50, 0.44), (0.50, 0.325))

    ax_tree.text(0.545, 0.733, "se divide en", fontsize=5.8, color=MUTED, ha="left", va="center")
    ax_tree.text(0.545, 0.383, "se divide en", fontsize=5.8, color=MUTED, ha="left", va="center")

    _save(fig, "fig01_estructura_datos")


# =============================================================================
# FIGURA 2 — la escala CPC y dónde cae realmente la línea de corte
# =============================================================================

def figura_2_escala_cpc() -> None:
    """La escala CPC de 1 a 5 con el corte entre el 2 y el 3.

    El punto de la figura es que la frontera NO separa consciencia de inconsciencia:
    el CPC 3 esta despierto y aun asi pertenece a la clase desfavorable. Por eso la
    columna 'consciente' se dibuja aparte y a proposito.
    """
    levels = [
        (1, "Buen desempeño cerebral", "Independiente para la vida diaria", True),
        (2, "Discapacidad moderada", "Quedan secuelas, sigue siendo independiente", True),
        (3, "Discapacidad severa", "Consciente, pero depende de otras personas", False),
        (4, "Síndrome de vigilia sin respuesta", "Abre los ojos, sin respuesta al entorno", False),
        (5, "Muerte", "Incluida la muerte cerebral", False),
    ]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))
    ax.set_xlim(0, 1)
    ax.axis("off")

    # Geometria: una banda izquierda para la etiqueta de clase, el bloque central con
    # las cajas y una columna derecha para la marca de consciencia. No se solapan.
    x_class, x_box, w_box, x_awake = 0.055, 0.135, 0.715, 0.895

    for i, (cpc, title, detail, favorable) in enumerate(levels):
        y = 4 - i
        color = BLUE if favorable else ORANGE
        face = "#eaf2fd" if favorable else "#fdf0ea"

        ax.add_patch(
            FancyBboxPatch(
                (x_box, y + 0.14), w_box, 0.72,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                facecolor=face, edgecolor=color, linewidth=1.0,
            )
        )
        # El numero del nivel, en su propio cuadro de color solido.
        ax.add_patch(
            FancyBboxPatch(
                (x_box + 0.014, y + 0.155), 0.058, 0.69,
                boxstyle="round,pad=0.004,rounding_size=0.015",
                facecolor=color, edgecolor=color, linewidth=0,
            )
        )
        ax.text(x_box + 0.043, y + 0.50, str(cpc), ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

        ax.text(x_box + 0.092, y + 0.63, title, ha="left", va="center", fontsize=7.4,
                color=INK, fontweight="bold")
        ax.text(x_box + 0.092, y + 0.36, detail, ha="left", va="center", fontsize=6.6, color=INK_2)

        # Marca de consciencia: informacion secundaria, deliberadamente separada del
        # color de clase para que se vea que las dos cosas no coinciden.
        awake = cpc in (1, 2, 3, 4)
        ax.text(
            x_awake, y + 0.50, "sí" if awake else "no",
            ha="center", va="center", fontsize=7.0,
            color=INK_2 if awake else MUTED, fontweight="bold" if awake else "normal",
        )

    # La linea de corte, entre el CPC 2 y el CPC 3.
    ax.plot([0.02, 0.98], [3.06, 3.06], color=INK, linewidth=1.6, linestyle=(0, (5, 2)))
    ax.text(
        0.50, 3.06, "  línea de corte: INDEPENDENCIA FUNCIONAL  ",
        ha="center", va="center", fontsize=7.0, color=INK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=SURFACE, edgecolor=INK, linewidth=1.0),
    )

    # Etiquetas de clase: verticales, en la banda izquierda, sin invadir las cajas.
    ax.text(x_class, 3.98, "FAVORABLE\nclase 1", ha="center", va="center", fontsize=6.4,
            color=BLUE, fontweight="bold", linespacing=1.4, rotation=90)
    ax.text(x_class, 1.50, "DESFAVORABLE\nclase 0", ha="center", va="center", fontsize=6.4,
            color=ORANGE, fontweight="bold", linespacing=1.4, rotation=90)

    ax.text(
        x_awake, 5.08, "¿está\ndespierto?", ha="center", va="center",
        fontsize=6.0, color=MUTED, fontweight="bold", linespacing=1.35,
    )
    ax.set_ylim(-0.05, 5.45)

    _save(fig, "fig02_escala_cpc")


# =============================================================================
# FIGURA 3 — el flujo de selección de pacientes (C24)
# =============================================================================

def figura_3_flujo_pacientes(cohort: pd.DataFrame) -> None:
    """Diagrama de flujo: 35 descargados -> 20 + 15 -> 3 excluidos -> 17 finales."""
    dev = cohort[cohort["group"] == "desarrollo"]
    held = cohort[cohort["group"] == "held-out"]
    excl = cohort[cohort["group"] == "excluido"].sort_values("patient")

    n_dev_fav = int((dev["outcome"] == "Good").sum())
    n_held_fav = int((held["outcome"] == "Good").sum())

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Nivel 1: lo descargado.
    _box(ax, 0.235, 0.895, 0.53, 0.085,
         f"{len(cohort)} pacientes descargados de I-CARE v2.1",
         "#f2f2ef", MUTED, fontsize=7.6, weight="bold")

    # Nivel 2: la division de diseno, hecha antes de mirar ninguna señal. La nota que
    # lo explica va CENTRADA y con una banda vertical propia, sin invadir las cajas.
    ax.text(0.50, 0.822, "división fijada antes de mirar la señal",
            ha="center", va="center", fontsize=6.2, color=MUTED)

    _box(ax, 0.025, 0.660, 0.45, 0.115,
         f"{len(dev) + len(excl)} para DESARROLLO\nbalance 10 / 10 elegido a mano",
         "#eaf2fd", BLUE, fontsize=7.0)
    _box(ax, 0.525, 0.660, 0.45, 0.115,
         f"{len(held)} para PRUEBA EXTERNA\n"
         f"{n_held_fav} favorables / {len(held) - n_held_fav} desfavorables",
         "#fdf0ea", ORANGE, fontsize=7.0)

    _arrow(ax, (0.40, 0.895), (0.22, 0.787))
    _arrow(ax, (0.60, 0.895), (0.78, 0.787))

    # Nivel 3: la exclusion por ventana, que solo afecta al conjunto de desarrollo.
    # La caja se dimensiona con holgura suficiente para las seis lineas de texto.
    motivos = "\n".join(
        f"{row.patient} (CPC {int(row.cpc)}) — termina en la hora {int(row.h_max)}"
        for row in excl.itertuples()
    )
    _box(ax, 0.500, 0.300, 0.475, 0.275,
         "3 excluidos: ningún segmento dentro\nde la ventana de 24 a 72 horas\n\n" + motivos,
         "#faf3f3", "#d03b3b", fontsize=6.4)

    _arrow(ax, (0.25, 0.660), (0.25, 0.578))
    _arrow(ax, (0.335, 0.438), (0.490, 0.438))
    ax.text(0.412, 0.462, "se retiran 3", ha="center", va="bottom",
            fontsize=6.2, color=MUTED)

    # Nivel 4: el conjunto final que se usa en todo el trabajo.
    _box(ax, 0.025, 0.300, 0.31, 0.275,
         f"{len(dev)} PACIENTES DE\nDESARROLLO\n\n"
         f"{n_dev_fav} favorables\n{len(dev) - n_dev_fav} desfavorables",
         "#eaf2fd", BLUE, fontsize=7.0)

    # Pie: dos notas que el diagrama por si solo no puede transmitir.
    ax.text(
        0.025, 0.205,
        "· El balance 10 / 10 del conjunto inicial fue una elección deliberada del autor durante la descarga.\n"
        "  Al excluir 3 pacientes (1 favorable y 2 desfavorables) quedó en 9 / 8.\n"
        "· De los 15 de prueba externa, el paciente 0356 no tiene ningún archivo de EEG en disco:\n"
        "  quedan 14 realmente utilizables.",
        ha="left", va="top", fontsize=6.2, color=INK_2, linespacing=1.6,
    )

    _save(fig, "fig03_flujo_pacientes")


# =============================================================================
# FIGURA 4 — distribución del CPC POR CONJUNTO (C24)
# =============================================================================

def figura_4_distribucion_cpc(cohort: pd.DataFrame) -> None:
    """CPC y desenlace binario, separados por conjunto.

    Va separada y no agregada a proposito: el hallazgo es que el conjunto de prueba
    externa no tiene ningun CPC 2 ni CPC 3 (los casos frontera), y en una grafica
    agregada esa ausencia desaparece.
    """
    groups = [
        ("desarrollo", "Desarrollo"),
        ("held-out", "Prueba externa"),
        ("excluido", "Excluidos"),
    ]
    cpc_levels = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH_IN, 2.6), sharey=True,
        gridspec_kw={"wspace": 0.16},
    )

    y_max = 9
    for ax, (key, title) in zip(axes, groups):
        subset = cohort[cohort["group"] == key]
        counts = [int((subset["cpc"] == level).sum()) for level in cpc_levels]

        # Sombreado de la zona frontera (CPC 2 y 3), por detras de las barras.
        ax.axvspan(1.5, 3.5, color=BAND, zorder=0)

        colors = [BLUE if level <= 2 else ORANGE for level in cpc_levels]
        bars = ax.bar(cpc_levels, counts, width=0.66, color=colors, zorder=3)

        # Etiqueta directa sobre cada barra: el color nunca va solo.
        for level, count, bar in zip(cpc_levels, counts, bars):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, count + 0.22, str(count),
                        ha="center", va="bottom", fontsize=7.0, color=INK, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.18, "0",
                        ha="center", va="bottom", fontsize=6.8, color=MUTED)

        n_frontera = int(subset["cpc"].isin([2, 3]).sum())
        ax.set_title(
            f"{title}  (n = {len(subset)})\n{n_frontera} de {len(subset)} en la frontera",
            color=INK, pad=5, fontsize=8,
        )
        ax.set_xticks(cpc_levels)
        ax.set_xlabel("nivel de la escala CPC")
        ax.set_ylim(0, y_max)
        ax.set_yticks(range(0, y_max + 1, 2))
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=1)
        ax.set_axisbelow(True)
        _despine(ax)

    axes[0].set_ylabel("número de pacientes")

    # Leyenda unica arriba, con el sombreado explicado.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
        plt.Rectangle((0, 0), 1, 1, color=BAND),
    ]
    labels = [
        "favorable (CPC 1–2)",
        "desfavorable (CPC 3–5)",
        "casos frontera (CPC 2 y 3)",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))

    _save(fig, "fig04_distribucion_cpc")


# =============================================================================
# FIGURA 5 — caracterización clínica por conjunto (C24)
# =============================================================================

def figura_5_caracterizacion_clinica(cohort: pd.DataFrame) -> None:
    """Edad, sexo y hospital de procedencia, separados por conjunto.

    Los dos hechos que la figura debe hacer evidentes son el predominio del hospital A
    en desarrollo y la aparicion del hospital D unicamente en la prueba externa.
    """
    dev = cohort[cohort["group"] == "desarrollo"]
    held = cohort[cohort["group"] == "held-out"]
    # Etiquetas cortas de dos lineas: los nombres completos se solapaban en los ejes
    # estrechos de los paneles a y b.
    sets = [("Desa-\nrrollo", dev, BLUE), ("Prueba\nexterna", held, ORANGE)]

    fig, (ax_age, ax_sex, ax_hosp) = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH_IN, 2.5), gridspec_kw={"wspace": 0.45, "width_ratios": [1, 0.68, 1.38]}
    )

    # --- a) Edad: un punto por paciente, mas la media -------------------------
    rng = np.random.default_rng(0)  # semilla fija: la dispersion vertical es estable
    for i, (label, subset, color) in enumerate(sets):
        ages = subset["age"].dropna().to_numpy()
        jitter = rng.uniform(-0.13, 0.13, size=len(ages))
        ax_age.scatter(np.full(len(ages), i) + jitter, ages, s=13, color=color,
                       alpha=0.75, linewidth=0.5, edgecolor=SURFACE, zorder=3)
        ax_age.plot([i - 0.26, i + 0.26], [ages.mean()] * 2, color=INK, linewidth=1.6, zorder=4)
        ax_age.text(i + 0.30, ages.mean(), f"{ages.mean():.0f}", fontsize=6.8,
                    color=INK, va="center", fontweight="bold")

    ax_age.set_xticks(range(len(sets)))
    ax_age.set_xticklabels([label for label, _, _ in sets])
    ax_age.set_xlim(-0.55, len(sets) - 0.35)
    ax_age.set_ylabel("edad (años)")
    ax_age.set_title("a) Edad", loc="left", color=INK, pad=5)
    ax_age.grid(axis="y", color=GRID, linewidth=0.7)
    ax_age.set_axisbelow(True)
    _despine(ax_age)

    # --- b) Sexo: proporcion de hombres, con el conteo escrito ----------------
    for i, (label, subset, color) in enumerate(sets):
        n_male = int((subset["sex"] == "Male").sum())
        pct = 100 * n_male / len(subset)
        ax_sex.bar(i, pct, width=0.5, color=color, zorder=3)
        ax_sex.text(i, pct + 2.5, f"{n_male} de {len(subset)}", ha="center", va="bottom",
                    fontsize=6.8, color=INK, fontweight="bold")

    ax_sex.set_xticks(range(len(sets)))
    ax_sex.set_xticklabels([label for label, _, _ in sets])
    ax_sex.set_xlim(-0.6, len(sets) - 0.4)
    ax_sex.set_ylim(0, 100)
    ax_sex.set_ylabel("hombres (%)")
    ax_sex.set_title("b) Sexo", loc="left", color=INK, pad=5)
    ax_sex.grid(axis="y", color=GRID, linewidth=0.7)
    ax_sex.set_axisbelow(True)
    _despine(ax_sex)

    # --- c) Hospital: barras agrupadas por conjunto ---------------------------
    hospitals = sorted(set(dev["hospital"]) | set(held["hospital"]))
    y_pos = np.arange(len(hospitals))
    height = 0.36

    for i, (label, subset, color) in enumerate(sets):
        counts = [int((subset["hospital"] == h).sum()) for h in hospitals]
        offset = (0.5 - i) * height
        ax_hosp.barh(y_pos + offset, counts, height=height * 0.92, color=color, zorder=3)
        for y, count in zip(y_pos + offset, counts):
            if count:
                ax_hosp.text(count + 0.22, y, str(count), va="center", fontsize=6.5, color=INK)

    ax_hosp.set_yticks(y_pos)
    ax_hosp.set_yticklabels([f"Hosp. {h}" for h in hospitals])
    ax_hosp.invert_yaxis()
    ax_hosp.set_xlabel("pacientes")
    ax_hosp.set_xlim(0, 13.5)
    ax_hosp.set_title("c) Hospital de procedencia", loc="left", color=INK, pad=5)
    ax_hosp.grid(axis="x", color=GRID, linewidth=0.7)
    ax_hosp.set_axisbelow(True)
    _despine(ax_hosp)

    # La anotacion del hospital D: es el hallazgo, y no puede depender de que el
    # lector compare dos barras por su cuenta. La leyenda se saca del panel (va como
    # leyenda comun de la figura) justamente para dejarle este hueco libre.
    if "D" in hospitals:
        idx = hospitals.index("D")
        ax_hosp.annotate(
            "el hospital D solo\naparece en prueba externa",
            xy=(3.2, idx - height / 2), xytext=(5.0, idx + 0.62),
            fontsize=6.0, color=ORANGE, fontweight="bold", linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8),
        )

    # Leyenda unica para los tres paneles: el color significa lo mismo en todos.
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in sets]
    fig.legend(
        handles, ["conjunto de desarrollo (17)", "conjunto de prueba externa (15)"],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.13),
    )

    _save(fig, "fig05_caracterizacion_clinica")


# =============================================================================
# FIGURA 6 — histograma de la hora máxima de registro (C29)
# =============================================================================

def figura_6_hora_maxima(cohort: pd.DataFrame) -> None:
    """Histograma de la hora maxima de registro, SOLO de los 20 iniciales.

    Los 15 pacientes de prueba externa quedan fuera a proposito: se descargaron en
    modo ligero (ADR-008, solo archivos EEG dentro de la ventana), de modo que su
    hora maxima esta truncada por el ancho de banda y no por el paciente. Incluirlos
    produciria una figura que miente.
    """
    initial_20 = cohort[cohort["group"].isin(["desarrollo", "excluido"])].copy()
    h_max = initial_20["h_max"].dropna().astype(int).to_numpy()

    # Se distinguen los excluidos dentro del mismo histograma: son los tres que caen
    # por debajo del piso de la ventana, y verlos ahi explica la exclusion sin texto.
    is_excluded = (initial_20["group"] == "excluido").to_numpy()
    h_max_kept = initial_20.loc[~is_excluded, "h_max"].dropna().astype(int).to_numpy()
    h_max_excl = initial_20.loc[is_excluded, "h_max"].dropna().astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.5))

    bins = np.arange(0, 192, 12)
    ax.axvspan(24, 72, color=BAND, zorder=0)
    ax.hist(
        [h_max_kept, h_max_excl], bins=bins, stacked=True,
        color=[BLUE, "#d03b3b"], edgecolor=SURFACE, linewidth=0.8, zorder=3,
        label=["conservados (17 de desarrollo)", "excluidos (3): no alcanzan la ventana"],
    )

    for x in (24, 72):
        ax.axvline(x, color=INK_2, linewidth=1.0, linestyle=(0, (4, 2)), zorder=4)

    ax.set_ylim(0, 7)
    ax.text(48, 6.75, "ventana 24–72 h", ha="center", va="top",
            fontsize=6.8, color=INK_2, fontweight="bold")

    ax.set_xlabel("hora máxima de registro (horas desde el paro cardíaco)")
    ax.set_ylabel("número de pacientes")
    ax.set_xlim(0, 186)
    ax.set_xticks(range(0, 181, 24))
    ax.set_yticks(range(0, 8, 2))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=6.4)
    _despine(ax)

    # Las dos medianas no coinciden y conviene decirlo: sobre los 17 que quedan es de
    # 72 h; sobre los 20 iniciales baja a 57 h porque los tres excluidos la arrastran.
    median_kept = int(np.median(h_max_kept))
    median_all = int(np.median(h_max))
    ax.set_title(
        f"Los {len(h_max)} pacientes descargados de forma completa\n"
        f"mediana = {median_kept} h sobre los 17 conservados · {median_all} h si se incluyen los 3 excluidos",
        loc="left", color=INK, pad=6, fontsize=7.6,
    )

    _save(fig, "fig06_hora_maxima")


# =============================================================================
# FIGURA 7 — cobertura de la ventana, paciente por paciente
# =============================================================================

def figura_7_cobertura_ventana(cohort: pd.DataFrame) -> None:
    """Una barra horizontal por paciente, de su hora minima a su hora maxima.

    En una sola imagen se ven la cobertura desigual de la ventana, los pacientes que
    apenas la rozan y por que fue necesario excluir tres. El conjunto de prueba
    externa se dibuja en un panel aparte porque su descarga reducida recorta el rango
    de forma artificial: mezclarlos en un mismo panel invitaria a compararlos.
    """
    initial_20 = cohort[cohort["group"].isin(["desarrollo", "excluido"])].copy()
    held = cohort[(cohort["group"] == "held-out") & cohort["h_min"].notna()].copy()

    # Orden por hora de inicio: hace visible de un vistazo quien entra tarde.
    initial_20 = initial_20.sort_values(["h_min", "h_max"], ascending=[False, False])
    held = held.sort_values(["h_min", "h_max"], ascending=[False, False])

    fig, (ax_dev, ax_held) = plt.subplots(
        2, 1, figsize=(TEXT_WIDTH_IN, 4.9), sharex=True,
        gridspec_kw={"height_ratios": [len(initial_20), len(held)], "hspace": 0.22},
    )

    def draw(ax, table, title):
        ax.axvspan(24, 72, color=BAND, zorder=0)
        for y, row in enumerate(table.itertuples()):
            excluded = row.group == "excluido"
            color = "#d03b3b" if excluded else (BLUE if row.outcome == "Good" else ORANGE)
            ax.barh(y, row.h_max - row.h_min, left=row.h_min, height=0.6,
                    color=color, zorder=3, alpha=0.95)
            label = f"{row.patient}"
            if excluded:
                label += " ✕"
            ax.text(-3, y, label, ha="right", va="center", fontsize=6.2,
                    color="#d03b3b" if excluded else INK_2,
                    fontweight="bold" if excluded else "normal")

        ax.set_yticks([])
        # Holgura arriba para que ni el titulo ni la etiqueta de la ventana toquen
        # la primera barra.
        ax.set_ylim(-1.7, len(table) - 0.2)
        ax.invert_yaxis()
        for x in (24, 72):
            ax.axvline(x, color=INK_2, linewidth=1.0, linestyle=(0, (4, 2)), zorder=4)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", color=INK, pad=5, fontsize=8)
        _despine(ax, keep=("bottom",))

    draw(ax_dev, initial_20, "a) Los 20 pacientes descargados de forma completa")
    draw(ax_held, held,
         "b) Prueba externa — el rango está recortado por una descarga reducida")

    ax_held.set_xlabel("horas desde el paro cardíaco")
    ax_held.set_xlim(0, 175)
    ax_held.set_xticks(range(0, 175, 24))

    ax_dev.text(48, -1.05, "ventana de interés  24–72 h", ha="center", va="center",
                fontsize=6.8, color=INK_2, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
        plt.Rectangle((0, 0), 1, 1, color="#d03b3b"),
    ]
    labels = ["desenlace favorable", "desenlace desfavorable", "excluido (✕): no alcanza la ventana"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.035))

    _save(fig, "fig07_cobertura_ventana")


# =============================================================================

def main() -> None:
    cohort = build_cohort()

    # Puerta de seguridad: si el disco ya no reproduce el finding 031, no se dibuja nada.
    problems = verify_against_finding_031(cohort)
    if problems:
        for problem in problems:
            print(f"  MISMATCH: {problem}")
        raise SystemExit("cohort does not match finding 031; refusing to draw figures")
    print("cohort verified against finding 031. Drawing figures:")

    figura_1_estructura_datos()
    figura_2_escala_cpc()
    figura_3_flujo_pacientes(cohort)
    figura_4_distribucion_cpc(cohort)
    figura_5_caracterizacion_clinica(cohort)
    figura_6_hora_maxima(cohort)
    figura_7_cobertura_ventana(cohort)
    print("done.")


if __name__ == "__main__":
    main()
