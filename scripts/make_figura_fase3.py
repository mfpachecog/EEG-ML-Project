"""Figura de la Fase 3: las 20 caracteristicas, un punto por paciente y por clase.

Que muestra (finding 034), con los datos leidos del disco:

1. Cada caracteristica se resume como la ve el modelo congelado: promedio sobre
   los 19 canales (`channel_agg`, 380 -> 20). No se dibujan los 380 valores
   porque el modelo nunca los usa por separado.
2. Cada paciente aporta UN punto por caracteristica (la mediana de sus 2000
   epocas). El N real del estudio es de pacientes, no de epocas: dibujar 34 000
   puntos daria una impresion falsa de abundancia.
3. Sobre cada panel se imprime un AUC univariado DESCRIPTIVO (in-sample, sin
   validacion). Sirve para ver si alguna caracteristica separa las clases a
   simple vista; no es un resultado del modelo ni una prueba estadistica.

Antes de dibujar verifica las cifras del finding 034 y aborta si no coinciden.

Uso:
    cd development && uv run python scripts/make_figura_fase3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from make_figuras_fase0 import BLUE, MUTED, ORANGE, TEXT_WIDTH_IN, _despine, _save  # noqa: E402

FEATURES_PATH = Path(__file__).resolve().parents[1] / "data_processed" / "features.parquet"
ID_COLUMNS = ["patient_id", "label", "outcome", "epoch_idx"]

# Nombres cortos para los paneles, en el mismo orden de la Tabla tab:caracteristicas.
SHORT_NAMES = {
    "rel_delta": "Pot. rel. δ",
    "rel_theta": "Pot. rel. θ",
    "rel_alpha": "Pot. rel. α",
    "rel_beta": "Pot. rel. β",
    "rel_gamma": "Pot. rel. γ",
    "ratio_alpha_delta": "α/δ",
    "ratio_theta_alpha": "θ/α",
    "ratio_alpha_theta": "α/θ",
    "dtabr": "(δ+θ)/(α+β)",
    "spectral_entropy": "Entropía espectral",
    "sef95": "SEF95",
    "sef50": "SEF50",
    "spectral_centroid": "Centroide espectral",
    "hjorth_mobility": "Movilidad Hjorth",
    "hjorth_complexity": "Complejidad Hjorth",
    "line_length_norm": "Long. de línea norm.",
    "zero_crossing_rate": "Cruces por cero",
    "kurtosis": "Curtosis",
    "perm_entropy": "Entropía permutación",
    "skewness": "Asimetría",
}


def load_patient_table() -> pd.DataFrame:
    """Lee el parquet, lo verifica y devuelve una fila por paciente con las 20 caracteristicas."""
    df = pd.read_parquet(FEATURES_PATH)

    # Verificacion contra el finding 034: si algo no cuadra, no se dibuja nada.
    assert df.shape == (34_000, 384), f"unexpected shape {df.shape}"
    assert df["patient_id"].nunique() == 17, "expected 17 development patients"
    assert (df.groupby("patient_id").size() == 2000).all(), "expected 2000 epochs per patient"
    assert np.isfinite(df.drop(columns=ID_COLUMNS).to_numpy()).all(), "NaN or inf found"

    feature_cols = [c for c in df.columns if c not in ID_COLUMNS]
    base_names = sorted({c.rsplit("_", 1)[0] for c in feature_cols})
    assert len(base_names) == 20, f"expected 20 base features, got {len(base_names)}"
    missing = set(base_names) ^ set(SHORT_NAMES)
    assert not missing, f"feature names changed: {missing}"

    # Promedio sobre los 19 canales, igual que la regla `channel_agg` del modelo congelado.
    per_epoch = pd.DataFrame(
        {name: df[[c for c in feature_cols if c.rsplit("_", 1)[0] == name]].mean(axis=1) for name in base_names}
    )
    per_epoch["patient_id"] = df["patient_id"].to_numpy()

    # La mediana es robusta a las epocas atipicas que sobrevivieron al rechazo.
    patients = per_epoch.groupby("patient_id").median()
    labels = df.groupby("patient_id")["label"].first()
    outcomes = df.groupby("patient_id")["outcome"].first()
    patients["label"] = labels
    patients["outcome"] = outcomes

    # label = 1 debe ser el desenlace favorable (Good), 9 contra 8.
    assert (patients.loc[patients["label"] == 1, "outcome"] == "Good").all(), "label 1 is not Good"
    assert (patients["label"] == 1).sum() == 9 and (patients["label"] == 0).sum() == 8, "expected 9 Good / 8 Poor"
    return patients


def descriptive_auc(patients: pd.DataFrame) -> pd.Series:
    """AUC univariado in-sample por caracteristica (0.5 = no separa; lejos de 0.5 en cualquier direccion = separa)."""
    return pd.Series({name: roc_auc_score(patients["label"], patients[name]) for name in SHORT_NAMES})


def figura_distribucion_caracteristicas(patients: pd.DataFrame, aucs: pd.Series) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(TEXT_WIDTH_IN, 7.4))
    rng = np.random.default_rng(42)  # solo para el desplazamiento horizontal de los puntos

    for ax, name in zip(axes.flat, SHORT_NAMES):
        for x, (label, color, marker) in enumerate([(1, BLUE, "o"), (0, ORANGE, "^")]):
            values = patients.loc[patients["label"] == label, name]
            jitter = rng.uniform(-0.12, 0.12, size=len(values))
            ax.scatter(x + jitter, values, s=11, color=color, marker=marker, alpha=0.85, linewidths=0)
            ax.hlines(values.median(), x - 0.25, x + 0.25, color=color, linewidth=1.4)
        ax.set_title(f"{SHORT_NAMES[name]}\nAUC = {aucs[name]:.2f}", fontsize=6.5, pad=2)
        ax.set_xticks([0, 1], ["Fav.", "Desf."], fontsize=6)
        ax.set_xlim(-0.55, 1.55)
        ax.tick_params(axis="y", labelsize=5.5)
        _despine(ax)

    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", linestyle="", label="Favorable (CPC 1-2), n = 9"),
        plt.Line2D([], [], color=ORANGE, marker="^", linestyle="", label="Desfavorable (CPC 3-5), n = 8"),
        plt.Line2D([], [], color=MUTED, linewidth=1.4, label="Mediana del grupo"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.9, w_pad=0.6)
    _save(fig, "fig09_distribucion_caracteristicas")


def main() -> None:
    patients = load_patient_table()
    aucs = descriptive_auc(patients)
    # Distancia a 0.5: cuanto separa, sin importar la direccion.
    ranking = (aucs - 0.5).abs().sort_values(ascending=False)
    print("Descriptive in-sample AUC per feature (patient level, n=17):")
    for name in ranking.index:
        print(f"  {name:22s} AUC={aucs[name]:.3f}  |AUC-0.5|={ranking[name]:.3f}")
    figura_distribucion_caracteristicas(patients, aucs)


if __name__ == "__main__":
    main()
