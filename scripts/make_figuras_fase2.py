"""Figuras de la Fase 2: el recorrido de la señal y el lugar del sistema en la UCI.

fig12 — El pipeline completo, del EEG crudo al pronóstico, con los parámetros
        reales de `config.py`, `preprocessing.py`, `features.py` y la ficha del
        modelo congelado. Responde al comentario del director en `06:480`.
fig14 — Los 19 canales en el sistema 10-20 (comentario C27 del consultor A).
fig13 — El flujo clínico del paciente en coma tras un paro cardíaco y el punto en
        que entra un análisis cuantitativo automático como apoyo. Responde al
        comentario del director en `05:57`.

Antes de dibujar, fig12 lee los parámetros del código y aborta si no coinciden
con lo que la figura dice: así el diagrama no puede quedarse desactualizado en
silencio.

Uso:
    cd development && uv run python scripts/make_figuras_fase2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from make_figuras_fase0 import (  # noqa: E402
    AQUA, BAND, BLUE, INK, INK_2, MUTED, ORANGE, SURFACE, TEXT_WIDTH_IN, _arrow, _box, _save,
)

LIGHT_BLUE = "#e6f0fb"
LIGHT_AQUA = "#e3f5ee"
LIGHT_ORANGE = "#fdeee7"


def verify_parameters() -> None:
    """Comprueba que lo que dice la figura es lo que hace el código."""
    import config
    import preprocessing as pp

    meta = json.loads((ROOT / "models" / "frozen_model.json").read_text())
    assert tuple(config.ROSC_WINDOW_HOURS) == (24, 72)
    assert len(config.CANONICAL_CHANNELS) == 19
    assert pp.TARGET_SFREQ_HZ == 100
    assert pp.EPOCH_SEC == 10
    assert pp.MAX_EPOCHS_PER_PATIENT == 2000
    assert pp.ROBUST_Z_THRESH == 4.0
    assert meta["classifier"] == "hgb_depth3" and meta["reduction"] == "channel_agg"
    assert meta["epochs_per_patient"] == 200 and meta["aggregation_rule"] == "mean"
    assert meta["decision_threshold"] == 0.5


def figura_12_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x, w, h = 0.25, 0.74, 0.046
    steps = [
        # (texto, color de relleno, color de borde)
        ("EEG crudo: segmentos horarios entre las horas 24 y 72", SURFACE, INK_2),
        ("Segmentos espaciados por la ventana (los justos para 2000 épocas, ×2)", SURFACE, INK_2),
        ("1. Los 19 canales del sistema 10-20", LIGHT_BLUE, BLUE),
        ("2. Notch 50 o 60 Hz (según hospital) + pasa-banda 0.5-45 Hz", LIGHT_BLUE, BLUE),
        ("3. Remuestreo a 100 Hz, después de filtrar", LIGHT_BLUE, BLUE),
        ("4. Canales muertos fuera del promedio + referencia promedio común", LIGHT_BLUE, BLUE),
        ("5. Épocas de 10 s", LIGHT_BLUE, BLUE),
        ("6. Rechazo de épocas planas y extremas (z robusto > 4)", LIGHT_BLUE, BLUE),
        ("Tope de 2000 épocas espaciadas: matriz (2000, 19, 1000)", LIGHT_BLUE, BLUE),
        ("20 características × 19 canales = 380 valores por época", LIGHT_AQUA, AQUA),
        ("200 épocas espaciadas; promedio de los 19 canales: 20 valores", LIGHT_ORANGE, ORANGE),
        ("Potenciación del gradiente: una probabilidad por época", LIGHT_ORANGE, ORANGE),
        ("Promedio de 200 probabilidades: pronóstico (≥ 0.5) + confianza", LIGHT_ORANGE, ORANGE),
    ]
    gap = (0.98 - 0.01 - len(steps) * h) / (len(steps) - 1)
    ys = [0.98 - h - i * (h + gap) for i in range(len(steps))]
    for (text, fc, ec), y in zip(steps, ys):
        _box(ax, x, y, w, h, text, fc, ec, fontsize=6.4)
    for y_top, y_next in zip(ys[:-1], ys[1:]):
        _arrow(ax, (x + w / 2, y_top), (x + w / 2, y_next + h))

    # Corchetes laterales: qué parte del trabajo hace cada tramo.
    groups = [
        (2, 8, "Preprocesa-\nmiento", BLUE),
        (9, 9, "Caracte-\nrísticas", AQUA),
        (10, 12, "Modelo\nfinal", ORANGE),
    ]
    for first, last, label, color in groups:
        top, bottom = ys[first] + h, ys[last]
        ax.plot([0.235, 0.225, 0.225, 0.235], [top, top, bottom, bottom], color=color, linewidth=1.2)
        ax.text(0.21, (top + bottom) / 2, label, ha="right", va="center", fontsize=6.8, color=color,
                linespacing=1.3)
    _save(fig, "fig12_pipeline_completo")


def figura_13_flujo_uci() -> None:
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fs = 6.2

    # Fila superior: el recorrido del paciente.
    h = 0.17
    top = [
        (0.01, 0.21, "Paro cardíaco\ny reanimación"),
        (0.26, 0.21, "Ingreso a la UCI\nen coma"),
        (0.51, 0.22, "EEG continuo\nen la UCI\n(horas a días)"),
        (0.77, 0.22, "Primeras 72 h:\nhipotermia y sedación;\naún no se decide"),
    ]
    y_top = 0.79
    for x, w, text in top:
        _box(ax, x, y_top, w, h, text, SURFACE, INK_2, fontsize=fs)
    for (x1, w1, _), (x2, _, _) in zip(top[:-1], top[1:]):
        _arrow(ax, (x1 + w1, y_top + h / 2), (x2, y_top + h / 2))

    # Fila media: las otras pruebas y las dos formas de leer el EEG.
    y_mid, h_mid = 0.43, 0.22
    _box(ax, 0.01, y_mid, 0.27, h_mid,
         "Otras pruebas: examen\nneurológico, potenciales\nevocados, biomarcadores,\nimagen", SURFACE, INK_2,
         fontsize=fs)
    _box(ax, 0.34, y_mid, 0.27, h_mid,
         "Lectura visual del EEG\npor un neurofisiólogo\n(práctica actual)", SURFACE, INK_2, fontsize=fs)
    _box(ax, 0.67, y_mid, 0.32, h_mid,
         "Análisis cuantitativo\nautomático del EEG\n(este trabajo, horas 24-72)", LIGHT_BLUE, BLUE,
         fontsize=fs, weight="bold")
    eeg_x = 0.51 + 0.22 / 2
    _arrow(ax, (eeg_x, y_top), (0.475, y_mid + h_mid))
    _arrow(ax, (eeg_x, y_top), (0.83, y_mid + h_mid))

    # Fila inferior: la evaluación y la decisión.
    y_bot, h_bot = 0.05, 0.20
    _box(ax, 0.08, y_bot, 0.46, h_bot,
         "Evaluación pronóstica multimodal\n(a partir de las 72 h)", BAND, INK_2, fontsize=6.6, weight="bold")
    _box(ax, 0.63, y_bot, 0.36, h_bot,
         "Decisión del equipo médico:\nmantener o retirar\nel soporte vital", SURFACE, INK, fontsize=fs)
    _arrow(ax, (0.145, y_mid), (0.20, y_bot + h_bot))
    _arrow(ax, (0.475, y_mid), (0.40, y_bot + h_bot))
    _arrow(ax, (0.83, y_mid), (0.50, y_bot + h_bot), color=BLUE)
    ax.text(0.745, 0.32, "apoyo, no reemplazo", fontsize=6, color=BLUE, style="italic", ha="center")
    _arrow(ax, (0.54, y_bot + h_bot / 2), (0.63, y_bot + h_bot / 2))
    _save(fig, "fig13_flujo_uci")


def figura_14_montaje() -> None:
    """Los 19 canales usados, en sus posiciones del sistema 10-20 (comentario C27)."""
    import mne
    from config import CANONICAL_CHANNELS

    info = mne.create_info(list(CANONICAL_CHANNELS), sfreq=100, ch_types="eeg")
    # I-CARE usa la nomenclatura clásica (T3, T4, T5, T6); el montaje estándar de MNE la incluye.
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    mne.viz.plot_sensors(info, kind="topomap", show_names=True, axes=ax, show=False,
                         pointsize=40, linewidth=0.8)
    for coll in ax.collections:
        coll.set_facecolor(BLUE)
        coll.set_edgecolor(BLUE)
    for text in ax.texts:
        text.set_fontsize(7)
    ax.set_title("")
    _save(fig, "fig14_montaje_10_20")


def main() -> None:
    verify_parameters()
    figura_12_pipeline()
    figura_13_flujo_uci()
    figura_14_montaje()


if __name__ == "__main__":
    main()
