"""Figura del marco teórico: la frontera de decisión de cada uno de los cuatro algoritmos.

fig15 — Los cuatro algoritmos evaluados (regresión logística, máquinas de vectores
        de soporte, bosque aleatorio y potenciación del gradiente) entrenados sobre
        el MISMO conjunto sintético de dos variables, cada uno con la frontera que
        traza para separar los dos grupos. Ilustra la frase del marco teórico que
        justifica la elección: la logística traza una recta, la SVM una curva y los
        dos métodos de árboles una frontera en escalones.

Los datos son sintéticos (make_moons) y NO provienen del estudio: la figura
explica qué hace cada algoritmo, no mide nada del proyecto. La potenciación del
gradiente usa la misma clase y profundidad que el modelo congelado
(HistGradientBoostingClassifier, max_depth=3) para que el dibujo corresponda a lo
que realmente se usó.

Uso:
    cd development && uv run python scripts/make_figura_algoritmos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from make_figuras_fase0 import BLUE, INK, MUTED, ORANGE, TEXT_WIDTH_IN, _despine, _save  # noqa: E402

SEED = 0


def make_panels() -> None:
    # Mismos datos para los cuatro: la única diferencia entre paneles es el algoritmo.
    X, y = make_moons(n_samples=140, noise=0.28, random_state=SEED)

    models = [
        ("a) Regresión logística", LogisticRegression(),
         "frontera recta"),
        ("b) Máquina de vectores de soporte", SVC(kernel="rbf", C=1.0, gamma=1.0),
         "frontera curva con margen"),
        ("c) Bosque aleatorio", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED),
         "voto de 200 árboles en paralelo"),
        ("d) Potenciación del gradiente", HistGradientBoostingClassifier(max_depth=3, random_state=SEED),
         "árboles en secuencia, cada uno corrige al anterior"),
    ]

    # Rejilla sobre la que se pinta la decisión de cada modelo.
    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    # Fondo tenue: naranja donde el modelo diría grupo 0, azul donde diría grupo 1.
    cmap = LinearSegmentedColormap.from_list("two_groups", [ORANGE, "#ffffff", BLUE])

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, TEXT_WIDTH_IN * 0.82))
    for ax, (title, model, subtitle) in zip(axes.ravel(), models):
        model.fit(X, y)
        if isinstance(model, SVC):
            # La SVM no da probabilidad por defecto: se pinta su función de decisión
            # (distancia con signo a la frontera), recortada para el color de fondo.
            z = model.decision_function(grid).reshape(xx.shape)
            ax.contourf(xx, yy, np.clip(z / 2 + 0.5, 0, 1), levels=20, cmap=cmap, alpha=0.35)
            ax.contour(xx, yy, z, levels=[-1, 1], colors=MUTED, linewidths=0.7, linestyles="--")
            ax.contour(xx, yy, z, levels=[0], colors=INK, linewidths=1.1)
            sv = model.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=34, facecolors="none", edgecolors=MUTED, linewidths=0.6)
        else:
            z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
            ax.contourf(xx, yy, z, levels=20, cmap=cmap, alpha=0.35)
            ax.contour(xx, yy, z, levels=[0.5], colors=INK, linewidths=1.1)

        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=9, color=ORANGE, edgecolors="white", linewidths=0.3)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=9, color=BLUE, edgecolors="white", linewidths=0.3)
        ax.set_title(title, fontsize=8, color=INK, loc="left", pad=11)
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("característica 1", fontsize=6.5, color=MUTED)
        ax.set_ylabel("característica 2", fontsize=6.5, color=MUTED)
        _despine(ax)

    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    _save(fig, "fig15_fronteras_algoritmos")


if __name__ == "__main__":
    make_panels()
