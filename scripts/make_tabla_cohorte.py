"""Genera la tabla LaTeX de los 35 pacientes descargados, leyendo el disco.

Por que existe: el capitulo de metodologia afirma que los identificadores de los
dos conjuntos estan ENTREMEZCLADOS y que los casos frontera (CPC 2 y 3) quedaron
todos en desarrollo (finding 033). Una tabla transcrita a mano volveria a ser una
afirmacion; esta se REGENERA desde los archivos de PhysioNet, igual que las
figuras de la Fase 0, de modo que si el disco cambiara la tabla cambiaria con el.

Reutiliza `cohort_table.build_cohort()` e invoca su verificacion contra el finding
031 ANTES de escribir nada: si el disco dejara de reproducir las cifras publicadas,
el script aborta en vez de emitir una tabla equivocada.

No toca el pipeline de la v1: solo lee metadatos y nombres de archivos.

Uso:
    cd development && uv run python scripts/make_tabla_cohorte.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cohort_table import build_cohort, verify_against_finding_031  # noqa: E402

# Destino: la carpeta de tablas del proyecto LaTeX, hermana de `figuras/`.
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "tesis_latex"
    / "proyecto"
    / "tablas"
    / "tabla_cohorte.tex"
)

# Traducciones al español: el destino es el documento de tesis, no el código.
GROUP_LABEL = {
    "desarrollo": "Desarrollo",
    "held-out": "Prueba externa",
    "excluido": "Excluido",
}
OUTCOME_LABEL = {"Good": "Favorable", "Poor": "Desfavorable"}
SEX_LABEL = {"Male": "H", "Female": "M"}

# Niveles vecinos a la linea de corte del CPC. Se resaltan en negrita porque el
# reparto de estos nueve pacientes es justamente lo que la tabla debe evidenciar.
BORDERLINE_CPC = {2, 3}


def _fmt_cpc(value) -> str:
    """Formatea el nivel de CPC, en negrita si es un caso frontera (CPC 2 o 3)."""
    if value != value:  # NaN
        return "--"
    level = int(value)
    return f"\\textbf{{{level}}}" if level in BORDERLINE_CPC else str(level)


def _fmt_int(value) -> str:
    """Formatea un entero que puede venir ausente."""
    return "--" if value != value or value is None else str(int(value))


def build_table(cohort) -> str:
    """Arma el cuerpo LaTeX completo de la tabla, ordenada por identificador."""
    rows = []
    for _, r in cohort.sort_values("patient").iterrows():
        rows.append(
            " & ".join(
                [
                    r["patient"],
                    GROUP_LABEL.get(r["group"], r["group"]),
                    str(r["hospital"]),
                    _fmt_int(r["age"]),
                    SEX_LABEL.get(r["sex"], "--"),
                    OUTCOME_LABEL.get(r["outcome"], str(r["outcome"])),
                    _fmt_cpc(r["cpc"]),
                    _fmt_int(r["n_segments_in_window"]),
                ]
            )
            + r" \\"
        )

    caption = (
        "Los 35 pacientes descargados, ordenados por identificador. La tabla permite "
        "comprobar dos afirmaciones del texto. La primera es que los conjuntos de "
        "desarrollo y de prueba externa no ocupan tramos separados de la base de datos, "
        "sino que sus identificadores aparecen entremezclados a lo largo de todo el rango. "
        "La segunda es que los nueve pacientes cuyo nivel de CPC es 2 o 3, resaltados en "
        "negrita y vecinos de la línea de corte entre desenlace favorable y desfavorable, "
        "pertenecen todos al conjunto de desarrollo. La última columna indica cuántos "
        "segmentos de cada paciente caen dentro de la ventana de 24 a 72 horas; en los tres "
        "pacientes excluidos ese número es cero. En los pacientes de prueba externa esa "
        "cantidad está limitada por la descarga reducida y no por la duración real del registro."
    )
    short_caption = "Los 35 pacientes descargados y su reparto entre conjuntos"

    header = (
        "Paciente & Conjunto & Hospital & Edad & Sexo & Desenlace & CPC & Segmentos \\\\"
        "\n"
        " & & & & & & & en ventana \\\\"
    )

    return f"""% ADVERTENCIA: archivo GENERADO. No editar a mano.
% Lo produce development/scripts/make_tabla_cohorte.py leyendo los datos de PhysioNet.
% Para actualizarlo: cd development && uv run python scripts/make_tabla_cohorte.py
\\begin{{footnotesize}}
\\begin{{longtable}}{{llccccccr}}
\\caption[{short_caption}]{{{caption}}}\\label{{tab:cohorte-completa}}\\\\
\\toprule
{header}
\\midrule
\\endfirsthead
\\multicolumn{{8}}{{l}}{{\\footnotesize\\itshape Tabla \\thetable{{}} (continuación)}}\\\\
\\toprule
{header}
\\midrule
\\endhead
\\midrule
\\multicolumn{{8}}{{r}}{{\\footnotesize\\itshape Continúa en la página siguiente}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
""" + "\n".join(rows) + """
\\end{longtable}
\\end{footnotesize}
"""


if __name__ == "__main__":
    cohort = build_cohort()

    # El disco manda: si dejara de reproducir el finding 031, no se emite tabla.
    problems = verify_against_finding_031(cohort)
    if problems:
        print("ABORTING: disk no longer reproduces finding 031")
        for problem in problems:
            print(f"  MISMATCH: {problem}")
        raise SystemExit(1)
    print("OK: disk reproduces finding 031; building table.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_table(cohort), encoding="utf-8")

    n_borderline_dev = len(
        cohort[(cohort["cpc"].isin(BORDERLINE_CPC)) & (cohort["group"] == "desarrollo")]
    )
    n_borderline_total = len(cohort[cohort["cpc"].isin(BORDERLINE_CPC)])
    print(f"Rows written: {len(cohort)}")
    print(f"Borderline (CPC 2-3): {n_borderline_dev}/{n_borderline_total} in development")
    print(f"Written to: {OUTPUT_PATH}")
