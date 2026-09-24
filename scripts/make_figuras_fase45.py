"""Figuras de las Fases 4 y 5: el resultado contra su propio azar, y paciente por paciente.

fig10 — Las dos varas del azar. Cada AUC se dibuja sobre la distribución nula de
        SU protocolo: la de LOPO (media 0.388, medida con 100 permutaciones de
        etiqueta por paciente) para la CV anidada, y la nula exacta (las 3003
        formas de repartir 8 favorables entre 14 pacientes) para el held-out.
        Comparar cualquiera de los dos contra el 0.5 "de toda la vida" sería
        usar la vara de otro protocolo.
fig11 — La probabilidad de desenlace favorable que recibió cada paciente, en
        desarrollo (predicción fuera de fold) y en el held-out (pasada única),
        separada por desenlace real, con el umbral de decisión de 0.5.

Solo lee resultados ya guardados: no reentrena nada ni vuelve a tocar el
held-out. Antes de dibujar verifica que reproduce las cifras de frozen_model.json
y held_out_results.json, y aborta si no.

Uso:
    cd development && uv run python scripts/make_figuras_fase45.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from make_figuras_fase0 import BLUE, INK, MUTED, ORANGE, TEXT_WIDTH_IN, _despine, _save  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_processed"
FROZEN_META = ROOT / "models" / "frozen_model.json"
HELD_OUT_JSON = DATA / "held_out" / "held_out_results.json"


def load_and_verify() -> dict:
    """Carga los resultados guardados y comprueba que coinciden con las fichas."""
    meta = json.loads(FROZEN_META.read_text())
    held = json.loads(HELD_OUT_JSON.read_text())
    dev = pd.read_csv(DATA / "nested_cv_patient_predictions.csv", dtype={"patient_id": str})
    ho = pd.read_csv(DATA / "held_out" / "held_out_patient_predictions.csv", dtype={"patient_id": str})
    lopo_null = np.load(DATA / "lopo_null_aucs.npy")

    # Desarrollo: 17 pacientes, AUC de la CV anidada y nula LOPO.
    dev_auc = roc_auc_score(dev["y_true"], dev["prob_good"])
    assert len(dev) == 17 and dev["y_true"].sum() == 9, "development set is not 17 (9 Good)"
    assert np.isclose(dev_auc, meta["nested_cv_patient_metrics"]["roc_auc"]), "nested AUC mismatch"
    assert len(lopo_null) == meta["lopo_null_distribution"]["n_permutations"] == 100
    assert np.isclose(lopo_null.mean(), meta["lopo_null_distribution"]["mean"], atol=1e-4)
    lopo_p = (np.sum(lopo_null >= dev_auc) + 1) / (len(lopo_null) + 1)
    assert np.isclose(lopo_p, meta["lopo_permutation_p_value"], atol=1e-4), "LOPO p mismatch"

    # Held-out: 14 pacientes (0356 no tiene EEG dentro de la ventana), 8 Good / 6 Poor.
    ho_auc = roc_auc_score(ho["y_true"], ho["prob_good"])
    assert len(ho) == 14 and ho["y_true"].sum() == 8, "held-out is not 14 (8 Good)"
    assert "0356" not in set(ho["patient_id"]), "0356 should be absent"
    assert np.isclose(ho_auc, held["patient_metrics"]["roc_auc"]), "held-out AUC mismatch"

    # Nula EXACTA del held-out: todas las formas de elegir qué 8 de los 14 son "Good".
    # El modelo está congelado, así que sus probabilidades no cambian al barajar etiquetas.
    probs = ho["prob_good"].to_numpy()
    exact_null = []
    for good_idx in combinations(range(14), 8):
        labels = np.zeros(14, dtype=int)
        labels[list(good_idx)] = 1
        exact_null.append(roc_auc_score(labels, probs))
    exact_null = np.asarray(exact_null)
    ref = held["chance_reference"]["exact_null"]
    assert len(exact_null) == ref["n_permutations"] == 3003
    # Los empates con el observado cuentan como "igual de extremos" (lo conservador).
    # sklearn calcula en coma flotante y un empate exacto puede salir 1e-16 por debajo,
    # así que se compara con tolerancia; evaluation.py lo hace con rangos exactos.
    exact_p = np.mean(exact_null >= ho_auc - 1e-9)
    assert np.isclose(exact_p, ref["p_value_one_sided"]), "exact p mismatch"

    return {
        "dev": dev, "ho": ho, "dev_auc": dev_auc, "ho_auc": ho_auc,
        "lopo_null": lopo_null, "lopo_p": lopo_p,
        "exact_null": exact_null, "exact_p": exact_p, "exact_p95": ref["p95"],
    }


def figura_10_dos_varas(r: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.1), sharey=False)
    bins = np.linspace(0, 1, 21)
    panels = [
        (axes[0], r["lopo_null"], r["dev_auc"], r["lopo_p"],
         "a) Desarrollo: CV anidada (LOPO)", "Nula de LOPO (100 permutaciones)"),
        (axes[1], r["exact_null"], r["ho_auc"], r["exact_p"],
         "b) Prueba externa: modelo congelado", "Nula exacta (3003 permutaciones)"),
    ]
    for ax, null, observed, p, title, null_label in panels:
        ax.hist(null, bins=bins, color=MUTED, alpha=0.45, density=True, label=null_label)
        ax.axvline(null.mean(), color=MUTED, linewidth=1.2, linestyle="--",
                   label=f"Centro del azar = {null.mean():.3f}")
        ax.axvline(0.5, color=INK, linewidth=0.8, linestyle=":", label="0.5 teórico")
        ax.axvline(observed, color=BLUE, linewidth=2.0, label=f"Observado = {observed:.3f} (p = {p:.3f})")
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("AUC a nivel de paciente", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22))
        _despine(ax, keep=("bottom",))
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig10_dos_varas_del_azar")


def figura_11_pacientes(r: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7), sharey=True)
    rng = np.random.default_rng(42)  # solo para separar puntos superpuestos
    panels = [(axes[0], r["dev"], "a) Desarrollo (17, fuera de fold)"),
              (axes[1], r["ho"], "b) Prueba externa (14, pasada única)")]
    for ax, df, title in panels:
        for x, (label, color, marker, name) in enumerate(
            [(1, BLUE, "o", "Favorable"), (0, ORANGE, "^", "Desfavorable")]
        ):
            sub = df[df["y_true"] == label]
            jitter = rng.uniform(-0.1, 0.1, size=len(sub))
            ax.scatter(x + jitter, sub["prob_good"], s=18, color=color, marker=marker, linewidths=0)
        ax.axhline(0.5, color=INK, linewidth=0.8, linestyle=":")
        ax.text(1.45, 0.51, "umbral 0.5", fontsize=5.8, color=INK, ha="right", va="bottom")
        ax.set_xticks([0, 1], ["Favorable\n(real)", "Desfavorable\n(real)"], fontsize=6.5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(axis="y", labelsize=6.5)
        _despine(ax)
    axes[0].set_ylabel("Probabilidad de desenlace favorable", fontsize=7)

    # El error más grave del held-out: favorable real con probabilidad casi nula.
    worst = r["ho"].loc[r["ho"]["patient_id"] == "0328"].iloc[0]
    axes[1].annotate("0328", xy=(0, worst["prob_good"]), xytext=(0.3, 0.12),
                     fontsize=6.5, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig11_probabilidad_por_paciente")


def main() -> None:
    r = load_and_verify()
    print(f"dev   AUC={r['dev_auc']:.4f}  LOPO null mean={r['lopo_null'].mean():.4f}  p={r['lopo_p']:.4f}")
    print(f"held  AUC={r['ho_auc']:.4f}  exact null mean={r['exact_null'].mean():.4f}  "
          f"p={r['exact_p']:.4f}  p95={r['exact_p95']:.4f}")
    figura_10_dos_varas(r)
    figura_11_pacientes(r)


if __name__ == "__main__":
    main()
