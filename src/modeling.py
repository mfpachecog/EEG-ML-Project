"""
=============================================================================
FASE 4 — Modelado y validación cruzada agrupada por paciente
=============================================================================

Toma la matriz de features de la Fase 3 (`features.parquet`, 34 000 épocas ×
380 features) y produce un **modelo congelado** que, dado un paciente nuevo,
emite un pronóstico binario MÁS una confianza.

Este módulo aporta SOLO las piezas de modelado. Toda la maquinaria de
validación honesta (splitters agrupados, agregación época->paciente, métricas
a nivel de paciente, bootstrap, controles negativos) vive en `validation.py` y
aquí se CONSUME, no se reimplementa.

-----------------------------------------------------------------------------
LAS TRES RESTRICCIONES QUE DEFINEN EL DISEÑO (medidas, no supuestas)
-----------------------------------------------------------------------------
1. EL N EFECTIVO ES 17 PACIENTES, NO 34 000 ÉPOCAS. Las ~2000 épocas de un
   paciente son casi redundantes entre sí; el contenido informativo real es
   "17 sujetos". En simulación, con la misma señal inyectada, 2000
   épocas/paciente dieron AUC 0.54 frente a 0.83 con 400: el modelo aprendía
   la HUELLA de cada paciente en vez de la señal clínica.
   ⚠️ ESE EFECTO **NO SE REPRODUJO** EN LOS DATOS REALES (medido el
   2026-07-23): AUC medio 0.465 (50 ép.) -> 0.455 (200) -> 0.500 (500) ->
   0.516 (2000), es decir tendencia plana o levemente creciente, y toda ella
   dentro del ruido. Se mantiene el nº de épocas como HIPERPARÁMETRO y se usan
   200 por presupuesto de cómputo, pero NO se afirma que menos sea mejor:
   en datos reales, simplemente da igual.

2. LA IDENTIDAD DEL PACIENTE ES CASI PERFECTAMENTE PREDECIBLE desde las
   features (accuracy 0.983 frente a un azar de 0.059). Existe una huella
   individual fortísima que compite con la señal clínica. -> regularización
   FUERTE y reducción de dimensionalidad agresiva.

3. HAY REDUNDANCIA MASIVA: 20 pares de features con |rho| > 0.9, y la
   correlación mediana entre canales de una misma familia es ~0.82. Los 19
   canales no son 19 informaciones independientes. -> promediar cada familia
   sobre los 19 canales (380 -> 20) es barato, muy defendible, y previsiblemente
   mejor que usar las 380 con 17 sujetos.

-----------------------------------------------------------------------------
⚠️ EL "AZAR" DE LOPO NO ESTÁ EN 0.5, ESTÁ EN ~0.39 (medido, 2026-07-23)
-----------------------------------------------------------------------------
En LOPO, `DummyClassifier(strategy="prior")` no da AUC 0.5: da **0.000 exacto**.
La causa es aritmética. Al dejar fuera a un paciente GOOD, el train queda 8/8 y
la prior de "good" vale 0.5000; al dejar fuera a uno POOR, queda 9/7 y sube a
0.5625. Resultado: todos los POOR reciben mayor probabilidad de "good" que todos
los GOOD -> orden perfectamente invertido -> AUC 0.

Es decir: el propio protocolo LOPO **deprime** el AUC, porque sacar a un paciente
desbalancea el train en la dirección contraria a su clase. Por eso 55 de las 76
configuraciones no-dummy del barrido cayeron por debajo de 0.5 (mediana 0.451).
Es un sesgo CONSERVADOR (a la baja): no infla resultados, los hunde.

CONSECUENCIA PRÁCTICA: comparar el AUC observado contra 0.5 "a ojo" es INCORRECTO
con este protocolo. La referencia honesta es la distribución nula EMPÍRICA del
mismo protocolo (barajar etiquetas por paciente y repetir LOPO entero), medida en
`old_scripts/06_lopo_null_distribution.py`: **media 0.388, sd 0.171, p95 0.683**.
El AUC observado de 0.556 queda por ENCIMA de ese centro pero MUY dentro del
ruido (p empírico 0.178). En las figuras, la línea de referencia debe ser esa
nula, nunca el 0.5 teórico ni el número del dummy.

-----------------------------------------------------------------------------
LA TRAMPA QUE ESTE MÓDULO NO PUEDE RESOLVER SOLO (hay que declararla)
-----------------------------------------------------------------------------
Con 17 pacientes, probar N configuraciones y quedarse con la de mejor AUC en
LOPO hace que esa AUC deje de ser una estimación honesta: se ha seleccionado
sobre el mismo dato con el que se mide (sesgo de selección de modelo, o
"optimismo del ganador"). Por eso:
  - el barrido comparativo se reporta como EXPLORACIÓN, no como resultado;
  - `nested_cv_estimate()` da la estimación (casi) insesgada, eligiendo el
    hiperparámetro DENTRO de cada fold externo;
  - y la estimación definitiva la dará el conjunto HELD-OUT en la Fase 5, que
    este módulo nunca toca.
=============================================================================
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CANONICAL_CHANNELS, PROJECT_ROOT
import validation as val

RANDOM_SEED = 42

# Directorio del modelo congelado (se crea al guardar).
MODELS_DIR = PROJECT_ROOT / "development" / "models"

# Presupuesto de cómputo: nº de épocas por paciente usado en el barrido.
# 200 x 17 = 3400 filas -> todo el barrido corre en segundos, y además la
# restricción 1 dice que menos épocas tiende a funcionar MEJOR, no peor.
DEFAULT_EPOCHS_PER_PATIENT = 200


# =============================================================================
# SUBMUESTREO DE ÉPOCAS (restricción 1)
# =============================================================================

def subsample_epochs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_per_patient: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a `n_per_patient` épocas por paciente, ESPACIADAS uniformemente.

    POR QUÉ ESPACIADAS Y NO ALEATORIAS: las épocas de un paciente cubren la
    ventana 24-72h post-ROSC y el estado neurológico CAMBIA a lo largo de ella.
    Un submuestreo por stride uniforme conserva esa cobertura temporal; coger
    las primeras N daría solo una foto temprana. Es el mismo criterio que usa
    `_strided_indices` en la Fase 2, y es DETERMINISTA (sin semilla): la misma
    entrada da siempre el mismo subconjunto, que es lo que hace reproducible
    todo el barrido.

    `n_per_patient=None` devuelve los datos intactos.
    """
    if n_per_patient is None:
        return X, y, groups

    keep = []
    for pid in np.unique(groups):
        idx = np.flatnonzero(groups == pid)
        if len(idx) <= n_per_patient:
            keep.append(idx)
        else:
            sel = np.unique(np.linspace(0, len(idx) - 1, n_per_patient).round().astype(int))
            keep.append(idx[sel])
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], groups[keep]


# =============================================================================
# AGREGACIÓN DE CANALES (restricción 3)
# =============================================================================

class ChannelAggregator(BaseEstimator, TransformerMixin):
    """Promedia cada familia de features sobre los 19 canales: 380 -> 20.

    Las columnas se llaman `<familia>_<canal>` (p. ej. `rel_delta_Fp1`). Este
    transformador agrupa por familia y devuelve una columna por familia.

    POR QUÉ ES BUENA IDEA AQUÍ: la correlación mediana entre canales de una
    misma familia es ~0.82, así que los 19 canales aportan mucha menos
    información independiente de lo que su número sugiere. Con solo 17 sujetos
    efectivos, pasar de 380 a 20 dimensiones reduce drásticamente el espacio en
    el que el modelo puede sobreajustar la huella individual del paciente.

    QUÉ SE PIERDE (hay que decirlo): toda la información TOPOGRÁFICA — dónde
    ocurre cada fenómeno en el cuero cabelludo. Para un pronóstico global
    post-anóxico, donde el daño suele ser difuso, es una pérdida asumible; para
    detectar focos localizados no lo sería.

    NO APRENDE NADA de los datos (`fit` es un no-op), así que no puede filtrar
    información entre folds. Aun así se coloca dentro del `Pipeline` por
    higiene: todo lo que transforma datos va en el pipeline.
    """

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names

    def _families(self) -> tuple[list[str], list[np.ndarray]]:
        """Agrupa los índices de columna por familia (nombre sin el canal final)."""
        channels = set(CANONICAL_CHANNELS)
        fam_to_idx: dict[str, list[int]] = {}
        for i, name in enumerate(self.feature_names):
            family, _, channel = name.rpartition("_")
            # Si el sufijo no es un canal conocido, la columna se trata como
            # familia propia (no la perdemos por un nombre inesperado).
            key = family if channel in channels else name
            fam_to_idx.setdefault(key, []).append(i)
        names = sorted(fam_to_idx)
        return names, [np.asarray(fam_to_idx[n]) for n in names]

    def fit(self, X, y=None):
        if self.feature_names is None:
            raise ValueError("ChannelAggregator requires feature_names")
        self.family_names_, self.family_index_ = self._families()
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.column_stack([X[:, idx].mean(axis=1) for idx in self.family_index_])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.family_names_, dtype=object)


# =============================================================================
# CATÁLOGO DE MODELOS Y REDUCTORES
# =============================================================================

def candidate_classifiers() -> dict:
    """Clasificadores candidatos, todos con `predict_proba`.

    `predict_proba` es OBLIGATORIO: el entregable es pronóstico + CONFIANZA, y
    la confianza sale de promediar las probabilidades por época del paciente.
    Un clasificador que solo emite clase no sirve para este producto.

    Incluye un `DummyClassifier` a propósito: es el SUELO. Con 9 good / 8 poor,
    predecir siempre la mayoritaria ya acierta el 53% — sin un suelo explícito
    es imposible saber si un modelo aporta algo o solo reproduce el balance.

    La regularización va deliberadamente sesgada a valores FUERTES (C pequeño,
    árboles poco profundos): con 17 sujetos y una huella individual predecible
    al 98%, el riesgo dominante es el sobreajuste, no el subajuste.
    """
    return {
        "dummy_majority": DummyClassifier(strategy="prior"),
        "logreg_l2_C0.001": LogisticRegression(C=0.001, penalty="l2", max_iter=2000),
        "logreg_l2_C0.01": LogisticRegression(C=0.01, penalty="l2", max_iter=2000),
        "logreg_l2_C0.1": LogisticRegression(C=0.1, penalty="l2", max_iter=2000),
        "logreg_l2_C1": LogisticRegression(C=1.0, penalty="l2", max_iter=2000),
        "logreg_l1_C0.01": LogisticRegression(
            C=0.01, penalty="l1", solver="liblinear", max_iter=2000
        ),
        "logreg_l1_C0.1": LogisticRegression(
            C=0.1, penalty="l1", solver="liblinear", max_iter=2000
        ),
        "svm_linear_C0.01": SVC(C=0.01, kernel="linear", probability=True, random_state=RANDOM_SEED),
        "svm_rbf_C1": SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_SEED),
        "rf_depth3": RandomForestClassifier(
            n_estimators=300, max_depth=3, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "rf_depth6": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "hgb_depth3": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, random_state=RANDOM_SEED
        ),
    }


def build_pipeline(
    classifier,
    feature_names: list[str],
    reduction: str = "channel_agg",
    n_components: int = 10,
    k_best: int = 20,
) -> Pipeline:
    """Arma el `Pipeline` completo: [reducción] -> escalado -> clasificador.

    ⚠️ TODO va DENTRO del pipeline por una razón metodológica, no estética: el
    `Pipeline` se ajusta SOLO con el fold de entrenamiento dentro de la CV. Si
    el `StandardScaler` o el `SelectKBest` se ajustaran fuera, verían los datos
    del paciente de test y meterían una fuga sutil — la más común del ML clásico
    y la más difícil de detectar a posteriori.

    `reduction`:
      - "none"        : las 380 features tal cual.
      - "channel_agg" : promedio por familia sobre canales (380 -> 20).
      - "pca"         : PCA a `n_components`.
      - "select_k"    : las `k_best` mejores por ANOVA F, ELEGIDAS DENTRO DEL FOLD.
      - "channel_agg+pca": agregación de canales y luego PCA.
    """
    steps: list[tuple[str, object]] = []

    if reduction in ("channel_agg", "channel_agg+pca"):
        steps.append(("channel_agg", ChannelAggregator(feature_names=feature_names)))

    steps.append(("scaler", StandardScaler()))

    if reduction == "pca" or reduction == "channel_agg+pca":
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_SEED)))
    elif reduction == "select_k":
        steps.append(("select", SelectKBest(score_func=f_classif, k=k_best)))

    steps.append(("clf", classifier))
    return Pipeline(steps)


# =============================================================================
# BARRIDO COMPARATIVO (exploración, NO resultado final)
# =============================================================================

def sweep(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    reductions: tuple[str, ...] = ("channel_agg", "none", "pca", "select_k"),
    epochs_per_patient: tuple[int | None, ...] = (DEFAULT_EPOCHS_PER_PATIENT,),
    classifiers: dict | None = None,
    out_csv: Path | None = None,
    time_budget_s: float = 180.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compara (clasificador × reducción × nº de épocas) con LOPO-CV.

    ⚠️ ESTO ES EXPLORACIÓN, NO UN RESULTADO. El máximo de esta tabla está
    SESGADO AL ALZA por selección de modelo (se elige el ganador mirando la
    misma métrica que se reporta). La estimación honesta la da
    `nested_cv_estimate()`, y la definitiva el held-out de la Fase 5.

    Escribe el CSV de forma INCREMENTAL tras cada configuración: si el proceso
    muere a mitad, lo ya calculado sobrevive.

    `time_budget_s` aborta una configuración que se pase de presupuesto y la
    anota como "too_expensive" en vez de colgar el barrido entero.
    """
    classifiers = classifiers or candidate_classifiers()
    rows: list[dict] = []

    for n_ep in epochs_per_patient:
        Xs, ys, gs = subsample_epochs(X, y, groups, n_ep)
        if verbose:
            print(f"\n=== epochs/patient = {n_ep} -> {Xs.shape[0]} rows ===", flush=True)

        for reduction in reductions:
            for clf_name, clf in classifiers.items():
                # SVM-RBF es O(n^2)-O(n^3): con muchas filas y sin reducir se
                # dispara. Regla dura de presupuesto (ver docstring del módulo).
                if clf_name.startswith("svm") and Xs.shape[0] > 5000:
                    continue

                t0 = time.perf_counter()
                try:
                    pipe = build_pipeline(clone(clf), feature_names, reduction=reduction)
                    res = val.run_grouped_cv(
                        Xs, ys, gs, pipe,
                        splitter=val.make_lopo_splitter(),
                        bootstrap=False,
                        verbose=False,
                    )
                    pm = res["patient_metrics"]
                    em = res["epoch_metrics"]
                    row = {
                        "n_epochs_per_patient": n_ep,
                        "reduction": reduction,
                        "classifier": clf_name,
                        "patient_auc": pm.get("roc_auc"),
                        "patient_balacc": pm.get("balanced_accuracy"),
                        "patient_sens": pm.get("sensitivity"),
                        "patient_spec": pm.get("specificity"),
                        # OJO: la clave que devuelve `patient_level_metrics` es
                        # `brier_score`, no `brier`. Con `.get("brier")` la columna
                        # salía 100% vacía sin avisar de nada (fallo silencioso).
                        "patient_brier": pm.get("brier_score"),
                        "epoch_auc": em.get("roc_auc"),
                        "epoch_balacc": em.get("balanced_accuracy"),
                        "seconds": round(time.perf_counter() - t0, 2),
                        "status": "ok",
                    }
                except Exception as exc:  # una config no debe tumbar el barrido
                    row = {
                        "n_epochs_per_patient": n_ep, "reduction": reduction,
                        "classifier": clf_name, "status": f"FAILED: {type(exc).__name__}: {exc}",
                        "seconds": round(time.perf_counter() - t0, 2),
                    }

                if row["seconds"] > time_budget_s and row["status"] == "ok":
                    row["status"] = "ok_over_budget"

                rows.append(row)
                if verbose:
                    print(
                        f"  {reduction:16s} {clf_name:18s} "
                        f"AUC_pac={row.get('patient_auc', float('nan')):.3f} "
                        f"balacc={row.get('patient_balacc', float('nan')):.3f} "
                        f"({row['seconds']}s) {row['status']}",
                        flush=True,
                    )
                if out_csv is not None:  # guardado incremental
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return pd.DataFrame(rows)


# =============================================================================
# ESTIMACIÓN (CASI) INSESGADA: CV ANIDADA
# =============================================================================

def nested_cv_estimate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    candidates: dict,
    reduction: str = "channel_agg",
    verbose: bool = True,
) -> dict:
    """LOPO externo para estimar; selección de hiperparámetro DENTRO de cada fold.

    POR QUÉ ES NECESARIA: si elijo el mejor modelo mirando la AUC de LOPO y
    luego reporto esa misma AUC, estoy reportando el máximo de una lista de
    números ruidosos — sesgado al alza. La CV anidada evita eso: en cada fold
    externo, el paciente de test NO participa en la elección del modelo.

    El bucle interno usa `StratifiedGroupKFold` sobre los 16 pacientes de
    entrenamiento y elige por AUC a nivel de PACIENTE (la métrica que importa).

    DEVUELVE la estimación out-of-fold agregada sobre los 17 pacientes y qué
    modelo ganó en cada fold (si el ganador baila mucho entre folds, es señal
    de que ninguno domina de verdad — información valiosa, no ruido a ocultar).
    """
    lopo = val.make_lopo_splitter()
    oof_prob = np.full(len(y), np.nan)
    chosen: list[str] = []

    for fold, (tr, te) in enumerate(lopo.split(X, y, groups)):
        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]

        best_name, best_score = None, -np.inf
        for name, clf in candidates.items():
            try:
                inner = val.make_stratified_group_splitter(n_splits=4, seed=RANDOM_SEED)
                res = val.run_grouped_cv(
                    X_tr, y_tr, g_tr,
                    build_pipeline(clone(clf), feature_names, reduction=reduction),
                    splitter=inner, bootstrap=False, verbose=False,
                )
                score = res["patient_metrics"].get("roc_auc", np.nan)
            except Exception:
                score = np.nan
            if np.isfinite(score) and score > best_score:
                best_name, best_score = name, score

        if best_name is None:  # ningún candidato utilizable en este fold
            best_name = next(iter(candidates))
        chosen.append(best_name)

        pipe = build_pipeline(clone(candidates[best_name]), feature_names, reduction=reduction)
        pipe.fit(X_tr, y_tr)
        oof_prob[te] = pipe.predict_proba(X[te])[:, list(pipe.classes_).index(1)]
        if verbose:
            print(f"  fold {fold:2d}: chose {best_name} (inner AUC {best_score:.3f})", flush=True)

    patient_df = val.aggregate_epoch_to_patient(oof_prob, groups, y_epoch=y, rule="mean")
    metrics = val.patient_level_metrics(patient_df)
    ci = val.bootstrap_ci_patient_level(patient_df, n_boot=2000, seed=RANDOM_SEED)
    return {
        "patient_df": patient_df,
        "patient_metrics": metrics,
        "patient_ci": ci,
        "chosen_per_fold": chosen,
        "oof_prob_epoch": oof_prob,
    }


# =============================================================================
# CONGELAR EL MODELO PARA LA FASE 5
# =============================================================================

def freeze_model(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    metadata: dict,
    out_dir: Path = MODELS_DIR,
    name: str = "frozen_model",
) -> tuple[Path, Path]:
    """Ajusta el pipeline con TODO el desarrollo y lo guarda junto a su ficha.

    Se ajusta con los 17 pacientes completos porque el modelo que verá el
    held-out debe haber aprovechado todos los datos de desarrollo disponibles;
    la ESTIMACIÓN de su rendimiento no sale de aquí (saldría sesgada), sale de
    la CV anidada y, definitivamente, del held-out.

    La ficha JSON acompañante existe para que la Fase 5 sea INAMBIGUA: qué
    features, en qué orden, con qué regla de agregación y con qué umbral. Sin
    eso, "aplicar el modelo congelado una sola vez" se vuelve interpretable, y
    la garantía anti-fuga depende justo de que no lo sea.
    """
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.joblib"
    meta_path = out_dir / f"{name}.json"

    pipeline = clone(pipeline)
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_features_in": int(X.shape[1]),
        "n_rows_fitted": int(X.shape[0]),
        "feature_names": list(feature_names),
        "random_seed": RANDOM_SEED,
        **metadata,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return model_path, meta_path
