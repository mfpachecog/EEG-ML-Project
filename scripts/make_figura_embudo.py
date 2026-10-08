"""Figura del embudo de señal: cuánta señal de un paciente llega realmente al modelo.

Responde a la corrección del autor del 2026-10-06 (bloque D): el texto describía las
reducciones (59 segmentos -> 16 -> 2000 épocas -> 200) en tres secciones distintas y
nunca juntas, así que el lector no podía dimensionar que al modelo llega ~1 % de la
ventana. La figura las pone en una sola unidad (épocas de 10 s) y en un solo dibujo.

Como el resto del código de figuras, MIDE en lugar de copiar cifras: lee las cabeceras
de los segmentos del paciente 0284, re-ejecuta su preprocesamiento (no escribe nada en
disco) y aborta antes de dibujar si no reproduce el finding 020.

Uso, desde development/:  uv run python scripts/make_figura_embudo.py   (~1 min)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wfdb  # noqa: E402

import preprocessing as pp  # noqa: E402
from config import DATA_DIR  # noqa: E402
from make_figuras_fase0 import (  # noqa: E402
    BAND, BLUE, GRID, INK, MUTED, ORANGE, TEXT_WIDTH_IN, _despine, _save,
)

PATIENT = "0284"
EPOCH_SECONDS = 10
MODEL_EPOCHS = 200  # épocas por paciente que ve el modelo (modeling.DEFAULT_EPOCHS_PER_PATIENT)


def medir_embudo(pid: str = PATIENT) -> dict:
    """Mide cada escalón del embudo para un paciente, desde disco."""
    patient_dir = DATA_DIR / pid
    records = pp.in_window_eeg_records(patient_dir)
    seconds = np.array([
        (h := wfdb.rdheader(str(patient_dir / r))).sig_len / h.fs for r in records
    ])
    # Las épocas se cortan dentro de cada segmento: el sobrante final (< 10 s) de cada
    # archivo se pierde, por eso se suma segmento a segmento y no sobre las horas totales.
    possible_epochs = int((seconds // EPOCH_SECONDS).sum())

    # Mismo cálculo que preprocess_patient: segmentos justos para el tope, con margen x2.
    n_seg = min(len(records), int(np.ceil(
        pp._SEGMENT_SAFETY * pp.MAX_EPOCHS_PER_PATIENT / pp._EPOCHS_PER_SEGMENT_EST
    )))
    chosen = pp._strided_indices(len(records), n_seg)

    mne.set_log_level("ERROR")
    epochs = pp.preprocess_patient(pid, verbose=False)
    model_idx = np.unique(np.linspace(0, len(epochs) - 1, MODEL_EPOCHS).round().astype(int))

    return {
        "n_segments": len(records),
        "hours": seconds.sum() / 3600,
        "possible": possible_epochs,
        "chosen": chosen,
        "processed": int((seconds[chosen] // EPOCH_SECONDS).sum()),
        "saved": len(epochs),
        "model": len(model_idx),
    }


def figura_16_embudo(m: dict) -> None:
    fig, (ax_strip, ax_fun) = plt.subplots(
        2, 1, figsize=(TEXT_WIDTH_IN, 3.7), gridspec_kw={"height_ratios": [0.55, 2.2], "hspace": 0.55}
    )

    # --- a) Los 59 segmentos y los 16 elegidos ----------------------------------
    n = m["n_segments"]
    chosen = set(m["chosen"].tolist())
    for i in range(n):
        ax_strip.add_patch(plt.Rectangle(
            (i, 0), 0.82, 1, facecolor=BLUE if i in chosen else BAND,
            edgecolor=BLUE if i in chosen else GRID, linewidth=0.6,
        ))
    for i in m["chosen"]:
        ax_strip.text(i + 0.41, -0.35, str(i + 1), ha="center", va="top", fontsize=5.2, color=BLUE)
    ax_strip.set_xlim(-0.5, n + 0.5)
    ax_strip.set_ylim(-1.6, 1.1)
    ax_strip.axis("off")
    ax_strip.set_title(
        f"a) Paciente {PATIENT}: los {len(chosen)} segmentos procesados (en azul) de los {n} de la ventana",
        loc="left", color=INK, fontsize=7.5, pad=4,
    )
    ax_strip.text(0, -1.05, "← inicio de la ventana (24 h)", fontsize=5.6, color=MUTED, va="top")
    ax_strip.text(n, -1.05, "fin de la ventana (72 h) →", fontsize=5.6, color=MUTED, va="top", ha="right")

    # --- b) El embudo, todo en épocas de 10 s ------------------------------------
    steps = [
        ("Señal disponible en la ventana\n(59 segmentos)", m["possible"], BAND),
        ("Segmentos procesados (16)", m["processed"], BLUE),
        ("Épocas limpias (paso 6)", m["clean"], BLUE),
        ("Épocas guardadas (tope)", m["saved"], BLUE),
        ("Épocas que ve el modelo", m["model"], ORANGE),
    ]
    total = m["possible"]
    y = np.arange(len(steps))[::-1]
    for yi, (label, value, color) in zip(y, steps):
        ax_fun.barh(yi, value, height=0.62, color=color, edgecolor=MUTED if color == BAND else color,
                    linewidth=0.6, zorder=3)
        hours = value * EPOCH_SECONDS / 3600
        dur = f"{hours:.1f} h".replace(".", ",") if hours >= 1 else f"{hours * 60:.0f} min"
        pct = f"{100 * value / total:.1f} %".replace(".", ",")
        ax_fun.text(value + total * 0.012, yi, f"{value:,} épocas · {dur} · {pct}".replace(",", " ", 1)
                    if value >= 1000 else f"{value} épocas · {dur} · {pct}",
                    va="center", fontsize=6.4, color=INK, fontweight="bold" if color == ORANGE else "normal")
    ax_fun.set_yticks(y)
    ax_fun.set_yticklabels([s[0] for s in steps], fontsize=6.6)
    ax_fun.set_xlim(0, total * 1.42)
    ax_fun.set_xlabel("épocas de 10 s (el porcentaje es sobre la señal disponible en la ventana)")
    ax_fun.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax_fun.set_axisbelow(True)
    ax_fun.set_title("b) Cuánta de esa señal llega al modelo", loc="left", color=INK, fontsize=7.5, pad=4)
    _despine(ax_fun)

    # La tira se alinea con el area de barras del embudo, no con el borde de la figura.
    fig.canvas.draw()
    pos_f, pos_s = ax_fun.get_position(), ax_strip.get_position()
    ax_strip.set_position([pos_f.x0, pos_s.y0, pos_f.width, pos_s.height])
    _save(fig, "fig16_embudo_senal")


def main() -> None:
    m = medir_embudo()
    # `preprocess_patient` no devuelve cuántas épocas sobrevivieron al paso 6, así que esa
    # cifra se toma de la verificación de la Fase 2 (verify_fase2.py, posterior al arreglo
    # del CAR) y se cruza con lo medido aquí: sus épocas de entrada deben coincidir.
    stats = pd.read_csv(pp.PROCESSED_DIR / "fase2_rejection_stats.csv", dtype={"pid": str})
    row = stats[stats["pid"] == PATIENT].iloc[0]
    assert int(row["n_input"]) == m["processed"], "la verificación de la Fase 2 no coincide con lo medido"
    m["clean"] = int(row["n_clean"])
    expected = {"n_segments": 59, "processed": 4238, "clean": 4141, "saved": 2000, "model": 200}
    got = {k: m[k] for k in expected}
    assert got == expected, f"el embudo de {PATIENT} ya no reproduce el finding 020: {got}"
    assert len(m["chosen"]) == 16
    print(f"funnel verified: {m['possible']} -> {m['processed']} -> {m['clean']} -> "
          f"{m['saved']} -> {m['model']}; chosen = {(m['chosen'] + 1).tolist()}")
    figura_16_embudo(m)


if __name__ == "__main__":
    main()
