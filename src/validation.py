"""
=============================================================================
FASE 4 — Marco de VALIDACIÓN (CV agrupada, agregación y métricas honestas)
=============================================================================

Este módulo NO entrena modelos "buenos": construye la BALANZA con la que se
pesan. Su única obsesión es que el número final sea creíble ante un jurado.

TRES PRINCIPIOS QUE AQUÍ NO SE RELAJAN (finding 021 §3.4-§3.8):

  1. SEPARACIÓN A NIVEL DE PACIENTE, SIEMPRE.
     Dos épocas del mismo paciente son casi la misma observación (mismo cerebro,
     mismos electrodos, minutos de diferencia). Si una cae en train y otra en
     test, el modelo no generaliza: RECUERDA. Todo splitter de este módulo es
     group-aware con `groups=patient_id`, y `assert_no_patient_leakage` lo
     COMPRUEBA fold a fold con un assert que se ejecuta de verdad.

  2. EL NIVEL DE ANÁLISIS PRIMARIO ES EL PACIENTE (N=17), NO LA ÉPOCA (N=34 000).
     La decisión clínica es por paciente: "¿despertará?" + una confianza. Las
     34 000 épocas son 17 observaciones repetidas ~2000 veces, no 34 000
     muestras independientes (finding 021 §3.7). Reportamos ambos niveles, pero
     el PRIMARIO es el de paciente, y los intervalos de confianza se calculan
     remuestreando PACIENTES.

  3. TODO PREPROCESAMIENTO SE AJUSTA DENTRO DEL FOLD.
     Escalado, selección de features, PCA, imputación: dentro de un
     `sklearn.Pipeline` que solo ve el fold de entrenamiento. Ajustar el
     `StandardScaler` sobre el dataset completo antes de la CV es la fuga más
     común (y más silenciosa) del ML clásico (finding 021 §3.5).

CONTRATO DE ENTRADA (`data_processed/features.parquet`, Fase 3):
    Una fila por época. Columnas de metadata: `patient_id` (str, p. ej. "0284"),
    `label` (int 1=good / 0=poor), `outcome` (str), `epoch_idx` (int). El resto
    son features float con patrón `<feature>_<canal>` (p. ej. `rel_delta_Fp1`),
    invariantes a escala (finding 021 §3.2), sin NaN ni inf.

    -> `patient_id` es el vector `groups` de la CV agrupada.
    -> `label`      es la `y`.

AUTO-COMPROBACIÓN: ejecutar este archivo (`uv run python src/validation.py`)
lanza una batería de verificaciones sobre datos SINTÉTICOS que respetan el
contrato, incluyendo un caso de FUGA DELIBERADA que el marco debe detectar.
=============================================================================
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CANONICAL_CHANNELS,
    EXCLUDED_PATIENTS,
    HELD_OUT_PATIENTS,
    PROJECT_ROOT,
)

# --- Constantes del marco de validación --------------------------------------

# Semilla única para TODO lo aleatorio del módulo (barajado de folds, bootstrap,
# permutaciones). Reproducibilidad = poder repetir el número exacto de la tesis.
RANDOM_SEED = 42

# Ruta esperada de la matriz de features (la produce la Fase 3).
PROCESSED_DIR = PROJECT_ROOT / "development" / "data_processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# Columnas de metadata del contrato: NO son features, nunca entran a la X.
META_COLUMNS = ("patient_id", "label", "outcome", "epoch_idx")
REQUIRED_COLUMNS = ("patient_id", "label")

# Umbral de decisión por defecto. 0.5 es la convención; con clases casi
# balanceadas (9/8) no hace falta moverlo, pero queda parametrizado porque en
# clínica el coste de un falso "despertará" no es simétrico.
DEFAULT_THRESHOLD = 0.5

# Convención de clase POSITIVA en todo el módulo: 1 = good = despertará.
# Por tanto: sensibilidad = detectar a los que despiertan;
#            especificidad = detectar a los que NO despiertan.
POSITIVE_LABEL = 1


# =============================================================================
# 1. CARGA Y VERIFICACIÓN DEL CONTRATO DE DATOS
# =============================================================================

def check_feature_contract(df: pd.DataFrame, strict_held_out: bool = True) -> dict:
    """Verifica que un DataFrame cumple el contrato de la matriz de features.

    FALLA RUIDOSAMENTE a propósito. Un NaN silencioso, una columna que se cuela
    o un paciente del held-out mezclado en desarrollo no producen un error de
    Python: producen un número BONITO Y FALSO en la tesis. Preferimos reventar
    aquí, con un mensaje claro, que descubrirlo defendiendo ante el jurado.

    QUÉ COMPRUEBA:
      - Que existen las columnas obligatorias `patient_id` y `label`.
      - Que hay al menos una columna de feature (numérica, no-metadata).
      - Que NO hay NaN ni inf en las features (romperían el escalado y el fit).
      - Que `label` solo toma valores 0/1 y es CONSTANTE dentro de cada paciente
        (el outcome es una propiedad del paciente, no de la época; si variase,
        la Fase 3 habría mezclado etiquetas).
      - Que NINGÚN paciente del held-out (config.HELD_OUT_PATIENTS) ni de los
        excluidos (ADR-004) está presente. El held-out solo se ve UNA vez, al
        final, con el modelo congelado: si aparece aquí, la columna vertebral
        anti-leakage del proyecto ya está rota.

    DEVUELVE: dict-resumen del contrato (n filas, n pacientes, balance, etc.).
    """
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CONTRACT VIOLATION: missing required columns {missing}")

    feature_names = [c for c in df.columns if c not in META_COLUMNS]
    if not feature_names:
        raise ValueError("CONTRACT VIOLATION: no feature columns found (only metadata)")

    non_numeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric feature columns: {non_numeric[:10]}")

    # NaN / inf: se comprueban sobre el bloque numérico completo de una vez.
    values = df[feature_names].to_numpy(dtype=float, copy=False)
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    if n_nan:
        bad_cols = [c for c in feature_names if df[c].isna().any()]
        problems.append(f"{n_nan} NaN values (columns: {bad_cols[:10]})")
    if n_inf:
        problems.append(f"{n_inf} inf values")

    labels = set(pd.unique(df["label"]))
    if not labels <= {0, 1}:
        problems.append(f"label must be binary 0/1, found {sorted(labels)}")

    # Una etiqueta por paciente: el outcome es del PACIENTE, no de la época.
    per_patient_labels = df.groupby("patient_id", observed=True)["label"].nunique()
    inconsistent = per_patient_labels[per_patient_labels > 1].index.tolist()
    if inconsistent:
        problems.append(f"patients with more than one label: {inconsistent}")

    present = set(df["patient_id"].astype(str).unique())
    leaked_holdout = sorted(present & set(HELD_OUT_PATIENTS))
    leaked_excluded = sorted(present & set(EXCLUDED_PATIENTS))
    if leaked_holdout and strict_held_out:
        problems.append(
            f"HELD-OUT PATIENTS PRESENT IN DEVELOPMENT DATA: {leaked_holdout} "
            "-- the external test set must never be seen during development"
        )
    if leaked_excluded:
        problems.append(f"excluded patients (ADR-004) present: {leaked_excluded}")

    if problems:
        raise ValueError(
            "CONTRACT VIOLATION(S):\n  - " + "\n  - ".join(problems)
        )

    label_by_patient = df.groupby("patient_id", observed=True)["label"].first()
    summary = {
        "n_rows": int(len(df)),
        "n_patients": int(label_by_patient.size),
        "n_features": len(feature_names),
        "n_good_patients": int((label_by_patient == 1).sum()),
        "n_poor_patients": int((label_by_patient == 0).sum()),
        "epochs_per_patient_min": int(df["patient_id"].value_counts().min()),
        "epochs_per_patient_max": int(df["patient_id"].value_counts().max()),
    }
    return summary


def load_feature_matrix(
    path: str | Path = FEATURES_PATH,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Carga la matriz de features y devuelve `(X, y, groups, feature_names, meta_df)`.

    Es la ÚNICA puerta de entrada de datos al marco de validación: pasa por
    `check_feature_contract`, así que nada entra a la CV sin haber sido auditado.

    DEVUELVE:
      - X            : (n_épocas, n_features) float64 — solo columnas de feature.
      - y            : (n_épocas,) int — etiqueta de la época (= la de su paciente).
      - groups       : (n_épocas,) str — `patient_id`; ES el vector de agrupación
                       que hace group-aware a TODOS los splitters de este módulo.
      - feature_names: nombres de las columnas de X, en el mismo orden.
      - meta_df      : las columnas de metadata (para trazar de vuelta una época).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {path}\n"
            "It is produced by Phase 3 (features.py). Until it exists, validate "
            "this module with make_synthetic_feature_frame()."
        )
    df = pd.read_parquet(path)
    # `patient_id` debe ser str: si el parquet lo trae como int, "0284" se
    # convertiría en 284 y dejaría de casar con config.HELD_OUT_PATIENTS.
    df["patient_id"] = df["patient_id"].astype(str)

    summary = check_feature_contract(df)

    feature_names = [c for c in df.columns if c not in META_COLUMNS]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["patient_id"].to_numpy(dtype=object)
    meta_df = df[[c for c in META_COLUMNS if c in df.columns]].copy()

    if verbose:
        print(
            f"Loaded {summary['n_rows']} epochs x {summary['n_features']} features "
            f"from {summary['n_patients']} patients "
            f"({summary['n_good_patients']} good / {summary['n_poor_patients']} poor)"
        )
    return X, y, groups, feature_names, meta_df


# =============================================================================
# 2. SPLITTERS AGRUPADOS  (la separación por paciente, hecha maquinaria)
# =============================================================================

def make_lopo_splitter() -> LeaveOneGroupOut:
    """LOPO-CV: Leave-One-Patient-Out. Con 17 pacientes, la opción natural.

    17 folds; en cada uno se deja fuera UN paciente entero (sus ~2000 épocas) y
    se entrena con los 16 restantes. Ventajas con N pequeño: usa el máximo de
    datos para entrenar y no depende de ningún sorteo de folds (es
    determinista, no necesita semilla).

    ⚠️ LIMITACIÓN QUE HAY QUE SABER DEFENDER: el fold de test contiene UN SOLO
    paciente, es decir UNA sola clase. El ROC-AUC a nivel de paciente NO se
    puede calcular dentro de un fold (necesita positivos y negativos). Se
    calcula UNA sola vez, al final, sobre el vector de predicciones
    OUT-OF-FOLD de los 17 pacientes (`run_grouped_cv` lo hace así). Promediar
    "AUCs por fold" en LOPO es un error frecuente: o da NaN, o se calcula sobre
    épocas de una sola clase y no significa nada.
    """
    return LeaveOneGroupOut()


def make_stratified_group_splitter(
    n_splits: int = 5, seed: int = RANDOM_SEED
) -> StratifiedGroupKFold:
    """`StratifiedGroupKFold`: k folds agrupados por paciente y ESTRATIFICADOS.

    Alternativa a LOPO cuando hace falta que cada fold de test tenga las DOS
    clases (p. ej. para reportar varianza de AUC entre folds, o para el test de
    permutación, donde 17 folds x N permutaciones sería carísimo).

    "Estratificado a nivel de grupo" significa que reparte los PACIENTES
    intentando conservar la proporción 9 good / 8 poor en cada fold, sin partir
    nunca un paciente entre train y test. Con 17 pacientes y 5 folds salen 3-4
    pacientes por fold: los folds son frágiles y su varianza es alta — eso se
    reporta, no se esconde (finding 021 §3.6).

    `shuffle=True` + `random_state` fijo = reproducible.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def assert_no_patient_leakage(
    splitter,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    verbose: bool = True,
) -> pd.DataFrame:
    """Comprueba, fold a fold, que NINGÚN paciente aparece en train y test a la vez.

    Esto NO es un comentario tranquilizador: es un `assert` que se ejecuta. Es
    la prueba material de que la separación por sujeto se cumple, y está
    pensada para poder enseñarla en la defensa ("aquí está el check, aquí su
    salida"). Comprueba además que ninguna FILA se comparte entre train y test
    (imposible si no hay solape de pacientes, pero es barato y cierra el caso).

    DEVUELVE: un DataFrame con una fila por fold — pacientes/épocas de train y
    test, composición de clases del test y el paciente dejado fuera (en LOPO).
    Sirve directo como tabla de metodología en la tesis.
    """
    rows = []
    seen_test_patients: set[str] = set()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        train_patients = set(np.asarray(groups)[train_idx])
        test_patients = set(np.asarray(groups)[test_idx])

        overlap = train_patients & test_patients
        assert not overlap, (
            f"DATA LEAKAGE in fold {fold}: patients present in BOTH train and "
            f"test: {sorted(overlap)}"
        )
        assert not (set(train_idx) & set(test_idx)), (
            f"DATA LEAKAGE in fold {fold}: overlapping row indices"
        )
        assert test_patients, f"fold {fold} has an empty test set"

        seen_test_patients |= test_patients
        y_test = np.asarray(y)[test_idx]
        rows.append(
            {
                "fold": fold,
                "n_train_patients": len(train_patients),
                "n_test_patients": len(test_patients),
                "n_train_epochs": len(train_idx),
                "n_test_epochs": len(test_idx),
                "test_patients": ",".join(sorted(test_patients)),
                "test_n_good_epochs": int((y_test == 1).sum()),
                "test_n_poor_epochs": int((y_test == 0).sum()),
            }
        )

    # Cobertura: en una CV completa, cada paciente debe haber sido test una vez.
    all_patients = set(np.asarray(groups))
    uncovered = all_patients - seen_test_patients
    assert not uncovered, f"patients never used as test: {sorted(uncovered)}"

    report = pd.DataFrame(rows)
    if verbose:
        print(
            f"No-leakage check PASSED: {len(report)} folds, "
            f"{len(all_patients)} patients, zero train/test patient overlap"
        )
    return report


# =============================================================================
# 3. AGREGACIÓN ÉPOCA -> PACIENTE  (aquí nace la "confianza" del entregable)
# =============================================================================

AGGREGATION_RULES = ("mean", "median", "good_fraction", "trimmed_mean")


def aggregate_epoch_to_patient(
    y_prob_epoch: np.ndarray,
    groups: np.ndarray,
    y_epoch: np.ndarray | None = None,
    rule: str = "mean",
    threshold: float = DEFAULT_THRESHOLD,
    trim: float = 0.1,
) -> pd.DataFrame:
    """Colapsa las ~2000 probabilidades por época de cada paciente en UNA sola.

    ⚠️ ESTA PROBABILIDAD AGREGADA **ES** LA "CONFIANZA" DEL ENTREGABLE CLÍNICO.
    El producto de la tesis es, por paciente, un pronóstico binario MÁS una
    confianza; esa confianza es exactamente el número que sale de aquí (por eso
    tiene que estar CALIBRADA — ver `calibration_report`). No es un detalle
    técnico intermedio: es la salida que vería un intensivista.

    REGLAS DISPONIBLES:
      - "mean" (por defecto): promedio de las probabilidades. Usa toda la
        información, es suave y es la que mejor se comporta como probabilidad
        calibrable. Es la recomendada.
      - "median": promedio robusto; ignora épocas con probabilidad extrema
        (artefactos residuales). Útil como análisis de sensibilidad.
      - "good_fraction": fracción de épocas clasificadas como "good"
        (prob >= threshold). Es un VOTO por mayoría; muy interpretable
        ("el 78% de su EEG parece de alguien que despierta") pero tira
        información al binarizar antes de agregar.
      - "trimmed_mean": media recortando el `trim` por cada cola. Punto medio
        entre media y mediana.

    DEVUELVE: DataFrame con UNA FILA POR PACIENTE y columnas
      `patient_id`, `n_epochs`, `prob_good` (la confianza), `pred` (0/1),
      y `y_true` si se pasó `y_epoch`. Ordenado por `patient_id`.
    """
    if rule not in AGGREGATION_RULES:
        raise ValueError(f"unknown rule '{rule}'; expected one of {AGGREGATION_RULES}")

    y_prob_epoch = np.asarray(y_prob_epoch, dtype=float)
    groups = np.asarray(groups)
    if y_prob_epoch.shape[0] != groups.shape[0]:
        raise ValueError("y_prob_epoch and groups must have the same length")

    frame = pd.DataFrame({"patient_id": groups, "prob": y_prob_epoch})
    if y_epoch is not None:
        frame["y_true"] = np.asarray(y_epoch)

    def _aggregate(probs: np.ndarray) -> float:
        if rule == "mean":
            return float(probs.mean())
        if rule == "median":
            return float(np.median(probs))
        if rule == "good_fraction":
            return float((probs >= threshold).mean())
        # trimmed_mean
        lo, hi = np.quantile(probs, [trim, 1.0 - trim])
        kept = probs[(probs >= lo) & (probs <= hi)]
        return float(kept.mean() if kept.size else probs.mean())

    rows = []
    for pid, sub in frame.groupby("patient_id", sort=True, observed=True):
        prob_good = _aggregate(sub["prob"].to_numpy(dtype=float))
        row = {
            "patient_id": pid,
            "n_epochs": int(len(sub)),
            "prob_good": prob_good,
            "pred": int(prob_good >= threshold),
        }
        if y_epoch is not None:
            # La etiqueta es del paciente: constante dentro del grupo (lo
            # garantiza check_feature_contract), así que basta con la primera.
            row["y_true"] = int(sub["y_true"].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 4. MÉTRICAS  (dos niveles; el primario es el de PACIENTE)
# =============================================================================

def _binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None
) -> dict:
    """Núcleo compartido: AUC, balanced accuracy, sensibilidad, especificidad, matriz.

    Sensibilidad y especificidad se definen con 1 = good = despertará:
      sensibilidad = TP/(TP+FN) = de los que despertaron, cuántos acertamos;
      especificidad = TN/(TN+FP) = de los que NO despertaron, cuántos acertamos.
    En un uso clínico real la métrica crítica es la SEGUNDA cara: predecir
    "no despertará" a alguien que sí habría despertado es el error grave.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # AUC exige ambas clases presentes. En LOPO (un paciente por fold) NO lo
    # están: devolvemos NaN en vez de un número inventado. Ver make_lopo_splitter.
    if y_prob is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, np.asarray(y_prob, dtype=float)))
    else:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    bal_acc = (
        float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return {
        "n": int(len(y_true)),
        "roc_auc": auc,
        "balanced_accuracy": bal_acc,
        "accuracy": float((y_true == y_pred).mean()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def epoch_level_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """Métricas a nivel de ÉPOCA (N ~ 34 000). SECUNDARIAS, no primarias.

    Se reportan por transparencia y porque son útiles para diagnosticar el
    modelo (¿acierta en casi todas las épocas o solo en la media?), pero su N
    es ENGAÑOSO: 2000 épocas del mismo paciente son casi la misma observación
    repetida, así que el N efectivo sigue siendo 17, no 34 000 (finding 021
    §3.4/§3.7). El pecado no es tanto el valor puntual (que puede salir más
    alto o más bajo que el de paciente) como la PRECISIÓN APARENTE: un
    intervalo de confianza calculado sobre 34 000 épocas sería ridículamente
    estrecho y falso. NUNCA presentar estas cifras como el resultado del
    trabajo, y nunca calcular CI sobre ellas.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    metrics = _binary_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)
    metrics["level"] = "epoch"
    return metrics


def patient_level_metrics(
    patient_df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> dict:
    """Métricas a nivel de PACIENTE (N=17). ESTAS SON LAS PRIMARIAS.

    Toma el DataFrame de `aggregate_epoch_to_patient` (una fila por paciente,
    con `prob_good` y `y_true`) y calcula sobre él. El ROC-AUC se calcula UNA
    sola vez sobre el vector completo de probabilidades OUT-OF-FOLD de los 17
    pacientes: es la única forma correcta de tener AUC en LOPO.

    Añade el Brier score (error cuadrático medio de la probabilidad frente a la
    etiqueta) porque el entregable no es solo la clase, es la CONFIANZA.
    """
    if "y_true" not in patient_df.columns:
        raise ValueError("patient_df must contain 'y_true' (pass y_epoch to aggregate)")
    y_true = patient_df["y_true"].to_numpy(dtype=int)
    y_prob = patient_df["prob_good"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = _binary_metrics(y_true, y_pred, y_prob)
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    metrics["level"] = "patient"
    return metrics


def bootstrap_ci_patient_level(
    patient_df: pd.DataFrame,
    n_boot: int = 2000,
    ci: float = 0.95,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = RANDOM_SEED,
    metrics: tuple[str, ...] = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity"),
) -> pd.DataFrame:
    """Intervalos de confianza por bootstrap REMUESTREANDO PACIENTES, no épocas.

    ⚠️ ES EL PUNTO MÁS FÁCIL DE HACER TRAMPA SIN DARSE CUENTA. Si se remuestrean
    las 34 000 ÉPOCAS, el bootstrap "cree" que hay 34 000 observaciones
    independientes y devuelve intervalos absurdamente estrechos (del tipo
    AUC 0.91 [0.90-0.92]) que serían deshonestos: la incertidumbre real del
    estudio viene de haber visto solo 17 CEREBROS. Aquí se remuestrean las 17
    filas de paciente con reemplazo.

    CON N=17 LOS INTERVALOS VAN A SALIR ANCHOS. Eso no es un fallo del método:
    es el resultado honesto de un PoC con 17 sujetos, y hay que reportarlo tal
    cual (y usarlo en "Limitaciones").

    Detalle técnico: un remuestreo puede quedarse con una sola clase; en ese
    caso AUC y balanced accuracy no existen y esa réplica se descarta para esas
    métricas (se reporta `n_valid` para que se vea cuántas quedaron).

    DEVUELVE: DataFrame con `metric`, `point` (estimación sobre la muestra
    real), `ci_low`, `ci_high`, `n_valid`.
    """
    rng = np.random.default_rng(seed)
    y_true = patient_df["y_true"].to_numpy(dtype=int)
    y_prob = patient_df["prob_good"].to_numpy(dtype=float)
    n_patients = len(y_true)

    point = patient_level_metrics(patient_df, threshold=threshold)
    samples: dict[str, list[float]] = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = rng.integers(0, n_patients, size=n_patients)
        boot = _binary_metrics(
            y_true[idx], (y_prob[idx] >= threshold).astype(int), y_prob[idx]
        )
        for m in metrics:
            value = boot.get(m, float("nan"))
            if np.isfinite(value):
                samples[m].append(value)

    alpha = (1.0 - ci) / 2.0
    rows = []
    for m in metrics:
        values = np.asarray(samples[m], dtype=float)
        if values.size:
            low, high = np.quantile(values, [alpha, 1.0 - alpha])
        else:
            low = high = float("nan")
        rows.append(
            {
                "metric": m,
                "point": point.get(m, float("nan")),
                "ci_low": float(low),
                "ci_high": float(high),
                "n_valid": int(values.size),
                "n_boot": int(n_boot),
                "n_patients": int(n_patients),
            }
        )
    return pd.DataFrame(rows)


def fold_variance_report(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume la VARIANZA ENTRE FOLDS. Con N=17, la varianza es parte del resultado.

    Un AUC medio de 0.85 con folds entre 0.55 y 1.00 NO es el mismo resultado
    que 0.85 con folds entre 0.82 y 0.88: el primero dice que el rendimiento
    depende de qué pacientes tocaron, es decir, que con 17 sujetos el modelo
    todavía no está determinado. Ocultar esa dispersión (reportar solo la media)
    es la forma educada de exagerar (finding 021 §3.6).

    DEVUELVE: DataFrame con `metric`, `mean`, `std`, `min`, `max`, `n_folds`
    (folds con valor definido; en LOPO el AUC por fold es NaN por diseño).
    """
    numeric = fold_metrics.select_dtypes(include="number")
    rows = []
    for column in numeric.columns:
        if column in ("fold", "n_test_patients"):  # identificadores, no métricas
            continue
        values = numeric[column].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "metric": column,
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "min": float(values.min()) if values.size else float("nan"),
                "max": float(values.max()) if values.size else float("nan"),
                "n_folds": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def calibration_report(
    patient_df: pd.DataFrame, n_bins: int = 4
) -> tuple[dict, pd.DataFrame]:
    """Calibración de la "confianza": Brier score + curva de calibración por paciente.

    POR QUÉ IMPORTA: el entregable no es "despertará", es "despertará con
    confianza 0.8". Si de todos los pacientes a los que el modelo les da 0.8
    solo despierta la mitad, ese 0.8 es un número decorativo y usarlo para
    decidir sobre soporte vital sería peligroso. Calibrado = "0.8" significa
    "acierta ~8 de cada 10".

    - Brier score: error cuadrático medio entre la probabilidad y la realidad
      (0 = perfecto). Mezcla discriminación y calibración en un solo número.
    - Curva de calibración: se agrupan los pacientes en `n_bins` cajas por
      probabilidad predicha y se compara la probabilidad media de la caja con
      la fracción real de "good" en ella. Ideal = diagonal.
    - ECE (expected calibration error): la desviación media respecto a esa
      diagonal, ponderada por cuántos pacientes hay en cada caja.

    ⚠️ CON 17 PACIENTES LA CURVA ES ANECDÓTICA: 4 cajas de ~4 pacientes. Sirve
    como control de sanidad ("¿las probabilidades altas van con los good?") y
    para declarar honestamente que la calibración no se puede establecer con
    este N. No la presentes como evidencia fuerte.

    DEVUELVE: (resumen dict, DataFrame de la curva con una fila por caja).
    """
    y_true = patient_df["y_true"].to_numpy(dtype=int)
    y_prob = patient_df["prob_good"].to_numpy(dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    rows, ece = [], 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_in_bin = int(mask.sum())
        if not n_in_bin:
            rows.append({"bin": b, "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                         "n_patients": 0, "mean_predicted": float("nan"),
                         "fraction_good": float("nan")})
            continue
        mean_pred = float(y_prob[mask].mean())
        frac_pos = float(y_true[mask].mean())
        ece += (n_in_bin / len(y_true)) * abs(mean_pred - frac_pos)
        rows.append({"bin": b, "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                     "n_patients": n_in_bin, "mean_predicted": mean_pred,
                     "fraction_good": frac_pos})

    summary = {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": float(ece),
        "n_patients": int(len(y_true)),
        "n_bins": int(n_bins),
        "mean_predicted_prob": float(y_prob.mean()),
        "observed_good_rate": float(y_true.mean()),
    }
    return summary, pd.DataFrame(rows)


# =============================================================================
# 5. AYUDA PARA CONSTRUIR EL PIPELINE + CV AGRUPADA COMPLETA
# =============================================================================

def make_pipeline(model, scaler: bool = True) -> Pipeline:
    """Envuelve un modelo en un `Pipeline` con `StandardScaler` delante.

    ESTA FUNCIÓN EXISTE PARA CERRAR LA FUGA DE §3.5. La media y la desviación
    del escalado son PARÁMETROS APRENDIDOS de los datos. Si se calculan sobre
    el dataset completo antes de la CV, el fold de train ya "sabe" algo del de
    test (su media, su escala) — fuga sutil, invisible, que infla el resultado.
    Metiendo el escalador DENTRO del Pipeline, `fit` solo lo ve con los datos
    de entrenamiento del fold y `transform` lo aplica al test: el test es un
    paciente nuevo, como en la vida real.

    La misma regla vale para cualquier paso que APRENDA de los datos: selección
    de features, PCA, imputación. Todo eso va aquí, como pasos previos al
    modelo, nunca fuera.

    NOTA: el modelo debe exponer `predict_proba` (LogisticRegression, RF,
    SVC(probability=True)). El entregable es un pronóstico MÁS UNA CONFIANZA:
    un clasificador que solo da la clase no sirve.
    """
    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _positive_class_proba(fitted_pipeline, X: np.ndarray) -> np.ndarray:
    """Probabilidad de la clase POSITIVA (1=good), robusta al orden de `classes_`."""
    estimator = fitted_pipeline[-1] if isinstance(fitted_pipeline, Pipeline) else fitted_pipeline
    if not hasattr(estimator, "predict_proba"):
        raise TypeError(
            f"{type(estimator).__name__} has no predict_proba. The clinical "
            "deliverable requires a calibrated confidence; use e.g. "
            "SVC(probability=True) or CalibratedClassifierCV."
        )
    proba = fitted_pipeline.predict_proba(X)
    classes = list(estimator.classes_)
    return proba[:, classes.index(POSITIVE_LABEL)]


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model,
    splitter=None,
    aggregation_rule: str = "mean",
    threshold: float = DEFAULT_THRESHOLD,
    bootstrap: bool = True,
    n_boot: int = 2000,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """Ejecuta el flujo completo de validación cruzada AGRUPADA por paciente.

    FLUJO (y por qué en este orden):
      1. Comprueba con asserts que el splitter no mezcla pacientes.
      2. Por fold: clona el pipeline (modelo LIMPIO, sin memoria del fold
         anterior), lo ajusta SOLO con el train y predice probabilidades sobre
         el test -> predicciones OUT-OF-FOLD.
      3. Junta las predicciones out-of-fold de TODAS las épocas: cada época se
         predijo exactamente una vez, por un modelo que nunca vio a su paciente.
      4. Agrega época -> paciente (la "confianza" del entregable).
      5. Calcula métricas a nivel de paciente (PRIMARIAS) y de época
         (secundarias), varianza entre folds y CI por bootstrap de PACIENTES.

    Por qué el AUC se calcula al final y no por fold: ver `make_lopo_splitter`.

    DEVUELVE un dict con:
      `oof_prob_epoch`, `patient_df`, `patient_metrics`, `epoch_metrics`,
      `fold_metrics`, `fold_variance`, `leakage_report`, `patient_ci`,
      `calibration`, `calibration_curve`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if splitter is None:
        splitter = make_lopo_splitter()

    pipeline = model if isinstance(model, Pipeline) else make_pipeline(model)

    # 1. La comprobación anti-leakage corre SIEMPRE, antes de entrenar nada.
    leakage_report = assert_no_patient_leakage(splitter, X, y, groups, verbose=verbose)

    # 2-3. Predicciones out-of-fold.
    oof_prob = np.full(len(y), np.nan, dtype=float)
    predicted_once = np.zeros(len(y), dtype=int)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        fitted = clone(pipeline).fit(X[train_idx], y[train_idx])
        prob = _positive_class_proba(fitted, X[test_idx])
        oof_prob[test_idx] = prob
        predicted_once[test_idx] += 1

        fold_metrics = epoch_level_metrics(y[test_idx], prob, threshold=threshold)
        fold_rows.append(
            {
                "fold": fold,
                "test_patients": ",".join(sorted(set(groups[test_idx]))),
                "n_test_patients": len(set(groups[test_idx])),
                "epoch_roc_auc": fold_metrics["roc_auc"],
                "epoch_accuracy": fold_metrics["accuracy"],
                "epoch_balanced_accuracy": fold_metrics["balanced_accuracy"],
                "mean_prob_good": float(prob.mean()),
            }
        )

    # Cada época debe haber sido predicha EXACTAMENTE una vez: si alguna se
    # predijo dos veces, un paciente estuvo en dos folds de test (o el splitter
    # solapa) y las métricas estarían contando datos repetidos.
    assert (predicted_once == 1).all(), (
        "each epoch must be predicted exactly once out-of-fold; got counts "
        f"{np.unique(predicted_once)}"
    )
    assert np.isfinite(oof_prob).all(), "some epochs have no out-of-fold prediction"

    # 4. Agregación época -> paciente.
    patient_df = aggregate_epoch_to_patient(
        oof_prob, groups, y_epoch=y, rule=aggregation_rule, threshold=threshold
    )

    # 5. Métricas.
    patient_metrics = patient_level_metrics(patient_df, threshold=threshold)
    epoch_metrics = epoch_level_metrics(y, oof_prob, threshold=threshold)
    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_variance = fold_variance_report(fold_metrics_df)
    calib_summary, calib_curve = calibration_report(patient_df)
    patient_ci = (
        bootstrap_ci_patient_level(patient_df, n_boot=n_boot, threshold=threshold, seed=seed)
        if bootstrap
        else None
    )

    if verbose:
        print(
            f"PATIENT level (PRIMARY, N={patient_metrics['n']}): "
            f"AUC={patient_metrics['roc_auc']:.3f}  "
            f"balacc={patient_metrics['balanced_accuracy']:.3f}  "
            f"sens={patient_metrics['sensitivity']:.3f}  "
            f"spec={patient_metrics['specificity']:.3f}  "
            f"brier={patient_metrics['brier_score']:.3f}"
        )
        print(
            f"EPOCH level (secondary, N={epoch_metrics['n']}): "
            f"AUC={epoch_metrics['roc_auc']:.3f}  "
            f"balacc={epoch_metrics['balanced_accuracy']:.3f}"
        )

    return {
        "oof_prob_epoch": oof_prob,
        "patient_df": patient_df,
        "patient_metrics": patient_metrics,
        "epoch_metrics": epoch_metrics,
        "fold_metrics": fold_metrics_df,
        "fold_variance": fold_variance,
        "leakage_report": leakage_report,
        "patient_ci": patient_ci,
        "calibration": calib_summary,
        "calibration_curve": calib_curve,
    }


# =============================================================================
# 6. CONTROLES NEGATIVOS  (finding 021 §3.8) — la sección que blinda la defensa
# =============================================================================

def _subsample_epochs_per_patient(
    groups: np.ndarray, max_per_patient: int | None, seed: int = RANDOM_SEED
) -> np.ndarray:
    """Índices de como mucho `max_per_patient` épocas por paciente (uniformes).

    Mismo criterio que el cap de la Fase 2: se toman épocas ESPACIADAS (no las
    primeras) para conservar la cobertura temporal. Solo se usa para abaratar
    los controles negativos, que repiten la CV decenas de veces.
    """
    if max_per_patient is None:
        return np.arange(len(groups))
    keep = []
    for pid in pd.unique(groups):
        idx = np.flatnonzero(groups == pid)
        if len(idx) > max_per_patient:
            take = np.unique(np.linspace(0, len(idx) - 1, max_per_patient).round().astype(int))
            idx = idx[take]
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def _patient_level_auc_from_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: Pipeline,
    splitter,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Corre una CV agrupada y devuelve SOLO el AUC a nivel de paciente.

    Versión mínima y silenciosa de `run_grouped_cv`, para usarla cientos de
    veces dentro del test de permutación sin imprimir ni calcular de más.
    """
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in splitter.split(X, y, groups):
        fitted = clone(pipeline).fit(X[train_idx], y[train_idx])
        oof[test_idx] = _positive_class_proba(fitted, X[test_idx])
    patient_df = aggregate_epoch_to_patient(oof, groups, y_epoch=y, threshold=threshold)
    if patient_df["y_true"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(patient_df["y_true"], patient_df["prob_good"]))


def permutation_test_patient_labels(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model=None,
    n_permutations: int = 100,
    splitter=None,
    max_epochs_per_patient: int | None = 400,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """Test de permutación: con etiquetas barajadas, ¿el rendimiento cae a azar?

    IDEA: si destruyo la relación entre features y etiqueta y el modelo SIGUE
    acertando, entonces no estaba usando la señal — estaba usando una fuga
    (identificadores, artefactos, estructura del split). Es el detector de
    humo del marco: barato, y responde a la pregunta que hará el jurado
    ("¿cómo sé que ese 0.9 no es un artefacto?").

    ⚠️ SE BARAJA A NIVEL DE PACIENTE, NO DE ÉPOCA. Barajar las 34 000 etiquetas
    de época repartiría ambas clases DENTRO de cada paciente; entonces el
    "azar" ya no sería 0.5 para la agregación por paciente y, sobre todo, se
    estaría rompiendo la estructura de grupo que precisamente queremos
    conservar bajo la hipótesis nula. La hipótesis nula correcta aquí es "el
    outcome del PACIENTE no tiene relación con su EEG", así que se permuta el
    vector de 17 etiquetas de paciente (lo que además conserva el balance 9/8)
    y se propaga a sus épocas.

    P-VALOR EMPÍRICO: (1 + #{AUC_permutado >= AUC_observado}) / (1 + n_permutaciones).
    El "+1" es la corrección estándar: sin ella se podría reportar p=0, que es
    imposible con un test empírico. Con 17 pacientes solo hay C(17,9)=24 310
    permutaciones distintas, así que el p mínimo alcanzable está acotado.

    DEVUELVE: dict con `observed_auc`, `null_aucs`, `null_mean`, `null_std`,
    `p_value`, `n_permutations`.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    # Submuestreo (solo por coste): el test repite la CV n_permutations+1 veces.
    keep = _subsample_epochs_per_patient(groups, max_epochs_per_patient, seed=seed)
    Xs, ys, gs = X[keep], y[keep], groups[keep]

    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=seed)
    pipeline = model if isinstance(model, Pipeline) else make_pipeline(model)
    if splitter is None:
        # Por defecto NO LOPO: 17 folds x 100 permutaciones sería carísimo.
        splitter = make_stratified_group_splitter(n_splits=5, seed=seed)

    observed = _patient_level_auc_from_cv(Xs, ys, gs, pipeline, splitter)

    # Tabla paciente -> etiqueta: lo que se permuta son ESTAS 17 filas.
    patients = pd.unique(gs)
    patient_labels = np.array([ys[gs == pid][0] for pid in patients], dtype=int)

    null_aucs = []
    for i in range(n_permutations):
        permuted = rng.permutation(patient_labels)
        mapping = dict(zip(patients, permuted))
        y_perm = np.array([mapping[g] for g in gs], dtype=int)
        null_aucs.append(_patient_level_auc_from_cv(Xs, y_perm, gs, pipeline, splitter))
        if verbose and (i + 1) % 25 == 0:
            print(f"  permutation {i + 1}/{n_permutations} ...")

    null_aucs = np.asarray(null_aucs, dtype=float)
    valid = null_aucs[np.isfinite(null_aucs)]
    p_value = float((1 + int((valid >= observed).sum())) / (1 + valid.size))

    result = {
        "observed_auc": float(observed),
        "null_aucs": null_aucs,
        "null_mean": float(valid.mean()) if valid.size else float("nan"),
        "null_std": float(valid.std(ddof=1)) if valid.size > 1 else float("nan"),
        "p_value": p_value,
        "n_permutations": int(valid.size),
        "n_epochs_used": int(len(ys)),
    }
    if verbose:
        print(
            f"Permutation test: observed patient AUC={result['observed_auc']:.3f}  "
            f"null={result['null_mean']:.3f}+-{result['null_std']:.3f}  "
            f"p={result['p_value']:.4f}  (n={result['n_permutations']})"
        )
    return result


def patient_identity_probe(
    X: np.ndarray,
    groups: np.ndarray,
    model=None,
    max_epochs_per_patient: int | None = 200,
    test_size: float = 0.3,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """Sonda de identidad: ¿las features permiten adivinar QUÉ PACIENTE es?

    Se entrena un clasificador MULTICLASE cuyo objetivo es el `patient_id`
    (17 clases) y se mide su accuracy frente al azar (1/17 = 0.059).

    ⚠️ AQUÍ EL SPLIT ES DELIBERADAMENTE NO-AGRUPADO. Es la única función del
    módulo donde épocas del mismo paciente están en train y test — y tiene que
    ser así: si el paciente es a la vez el grupo y el objetivo, un split
    agrupado haría la tarea imposible por construcción. No estamos midiendo
    generalización, estamos midiendo IDENTIFICABILIDAD. No copiar este patrón
    para nada más.

    CÓMO INTERPRETARLO: una accuracy muy alta (p. ej. >0.9) significa que cada
    paciente tiene una HUELLA propia en las features — anatomía, colocación de
    electrodos, canales muertos que el CAR reparte (finding 021 §3.3). Eso NO
    invalida el trabajo: la CV agrupada impide que el modelo memorice pacientes
    del test. Pero sí es un riesgo que hay que declarar: si esa huella
    correlaciona por azar con el outcome (fácil con 17 sujetos), el modelo
    puede estar aprendiendo "quién es" en vez de "cómo está", y con N pequeño
    eso es indistinguible desde dentro. Es material de la sección de
    Limitaciones, y la razón por la que el test held-out es imprescindible.

    DEVUELVE: dict con `accuracy`, `chance_level`, `ratio_over_chance`,
    `n_patients`, `n_train`, `n_test`.
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    keep = _subsample_epochs_per_patient(groups, max_epochs_per_patient, seed=seed)
    Xs, gs = X[keep], groups[keep]

    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=seed)
    pipeline = model if isinstance(model, Pipeline) else make_pipeline(model)

    X_train, X_test, g_train, g_test = train_test_split(
        Xs, gs, test_size=test_size, random_state=seed, stratify=gs
    )
    fitted = clone(pipeline).fit(X_train, g_train)
    accuracy = float((fitted.predict(X_test) == g_test).mean())
    n_patients = int(len(np.unique(gs)))
    chance = 1.0 / n_patients

    result = {
        "accuracy": accuracy,
        "chance_level": chance,
        "ratio_over_chance": accuracy / chance,
        "n_patients": n_patients,
        "n_train": int(len(g_train)),
        "n_test": int(len(g_test)),
    }
    if verbose:
        print(
            f"Patient-identity probe: accuracy={accuracy:.3f} "
            f"(chance={chance:.3f}, x{result['ratio_over_chance']:.1f})"
        )
    return result


# =============================================================================
# 7. DATOS SINTÉTICOS + AUTO-COMPROBACIÓN DEL MARCO
# =============================================================================

def make_synthetic_feature_frame(
    n_patients: int = 17,
    n_good: int = 9,
    n_epochs: int = 2000,
    n_features: int = 114,
    effect_size: float = 0.45,
    epoch_noise: float = 1.0,
    patient_noise: float = 1.0,
    inject_leak: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Genera una matriz de features SINTÉTICA que respeta el contrato de Fase 3.

    POR QUÉ HACE FALTA: este módulo se construyó en paralelo a la Fase 3, sin
    `features.parquet` en disco. Validar el marco con datos sintéticos no es un
    parche: es MEJOR, porque aquí SABEMOS la verdad (cuánta señal hay, dónde
    está la fuga) y podemos comprobar que el marco la detecta. Con datos reales
    solo se puede comprobar que no revienta.

    ESTRUCTURA DE LA SEÑAL (imita la realidad del problema):
      - Cada PACIENTE tiene un desplazamiento propio (efecto aleatorio de
        sujeto) = la "huella" de cada cerebro.
      - Sobre las primeras features, ese desplazamiento tiene una componente
        ligada a la etiqueta (`effect_size`) = la señal pronóstica REAL, que
        vive a nivel de paciente, no de época.
      - Cada época añade ruido independiente. Resultado: épocas del mismo
        paciente muy correlacionadas entre sí, exactamente el problema que la
        CV agrupada existe para manejar.

    `inject_leak=True` añade una columna `leak_marker_Fp1` que codifica la
    etiqueta del paciente casi sin ruido: es una fuga DELIBERADA, para
    comprobar que el marco la ve (AUC ~1.0).

    ⚠️ HALLAZGO AL CALIBRAR ESTOS DATOS (útil para la Fase 4 real): el
    rendimiento cae en picado al SUBIR el número de épocas por paciente con el
    mismo `effect_size`. Con 400 épocas/paciente y efecto 0.30 el AUC de
    paciente sale ~0.83; con 2000 épocas y el MISMO efecto, ~0.54. No es un
    bug: con 16 pacientes de entrenamiento y miles de épocas cada uno, el
    modelo tiene datos de sobra para aprender la HUELLA de cada paciente de
    entrenamiento (su efecto aleatorio de sujeto) y esa dirección no
    generaliza al paciente 17. Más épocas NO es más información — el N
    efectivo sigue siendo 17 (finding 021 §3.7). Consecuencia práctica: en la
    Fase 4 hay que regularizar fuerte y/o submuestrear épocas, y no
    interpretar "tengo 34 000 muestras" como que se pueden ajustar modelos
    complejos.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([1] * n_good + [0] * (n_patients - n_good), dtype=int)
    rng.shuffle(labels)
    patient_ids = [f"S{i:04d}" for i in range(n_patients)]

    # Nombres con el patrón <feature>_<canal> del contrato. Si se piden más
    # features que combinaciones base x canal, se añaden bandas extra numeradas
    # para que TODOS los nombres sean únicos (un duplicado rompería el parquet).
    base_names = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta", "ratio_alpha_delta",
                  "spectral_entropy", "hjorth_mobility", "hjorth_complexity",
                  "line_length", "rel_gamma"]
    feature_names: list[str] = []
    round_idx = 0
    while len(feature_names) < n_features:
        suffix = "" if round_idx == 0 else f"{round_idx + 1}"
        for base in base_names:
            for ch in CANONICAL_CHANNELS:
                feature_names.append(f"{base}{suffix}_{ch}")
                if len(feature_names) == n_features:
                    break
            if len(feature_names) == n_features:
                break
        round_idx += 1

    # Solo una parte de las features lleva señal (como en la vida real).
    n_informative = max(1, n_features // 6)

    blocks = []
    for pid, label in zip(patient_ids, labels):
        # Efecto de sujeto: constante dentro del paciente = la huella.
        patient_offset = rng.normal(0.0, patient_noise, size=n_features)
        patient_offset[:n_informative] += effect_size * (1 if label == 1 else -1)
        epochs = rng.normal(0.0, epoch_noise, size=(n_epochs, n_features)) + patient_offset

        block = pd.DataFrame(epochs, columns=feature_names)
        block.insert(0, "patient_id", pid)
        block.insert(1, "label", int(label))
        block.insert(2, "outcome", "Good" if label == 1 else "Poor")
        block.insert(3, "epoch_idx", np.arange(n_epochs))
        if inject_leak:
            # Fuga deliberada: la etiqueta, disfrazada de feature.
            block["leak_marker_Fp1"] = label + rng.normal(0.0, 0.01, size=n_epochs)
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def _self_check() -> None:
    """Batería de comprobaciones del marco sobre datos sintéticos.

    Cada bloque comprueba UNA propiedad que, si falla, invalidaría los
    resultados de la tesis. Se ejecuta con `uv run python src/validation.py`.
    """
    import time

    def banner(text: str) -> None:
        print("\n" + "=" * 78)
        print(text)
        print("=" * 78)

    t0 = time.time()

    # ---------------------------------------------------------------- CONTRATO
    banner("[1] Synthetic data + contract check")
    df = make_synthetic_feature_frame()
    summary = check_feature_contract(df)
    print(f"  shape = {df.shape}")
    print(f"  contract summary = {summary}")
    assert summary["n_rows"] == 34000, "expected 17 x 2000 = 34000 rows"
    assert summary["n_patients"] == 17
    assert (summary["n_good_patients"], summary["n_poor_patients"]) == (9, 8)
    print("  OK: contract satisfied (34000 epochs, 17 patients, 9 good / 8 poor)")

    feature_names = [c for c in df.columns if c not in META_COLUMNS]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["patient_id"].to_numpy(dtype=object)

    # ------------------------------------------------- CONTRATO: FALLOS DUROS
    banner("[2] Contract must FAIL loudly on bad input")
    bad = df.copy()
    bad.loc[0, feature_names[0]] = np.nan
    try:
        check_feature_contract(bad)
        raise AssertionError("contract check did NOT detect the injected NaN")
    except ValueError as exc:
        print(f"  OK: NaN detected -> {str(exc).splitlines()[1].strip()}")

    held = df.copy()
    held.loc[held.index[:10], "patient_id"] = sorted(HELD_OUT_PATIENTS)[0]
    try:
        check_feature_contract(held)
        raise AssertionError("contract check did NOT detect the held-out patient")
    except ValueError as exc:
        print(f"  OK: held-out leak detected -> {str(exc).splitlines()[1].strip()[:90]}")

    # ------------------------------------------------------------- SPLITTERS
    banner("[3] No patient appears in train and test (LOPO + StratifiedGroupKFold)")
    lopo_report = assert_no_patient_leakage(make_lopo_splitter(), X, y, groups)
    print(lopo_report.head(3).to_string(index=False))
    assert len(lopo_report) == 17, "LOPO must produce one fold per patient"
    assert (lopo_report["n_test_patients"] == 1).all()
    sgk_report = assert_no_patient_leakage(
        make_stratified_group_splitter(n_splits=5), X, y, groups
    )
    print(sgk_report[["fold", "n_train_patients", "n_test_patients", "test_patients"]]
          .to_string(index=False))
    assert len(sgk_report) == 5

    # -------------------------------------------------------- CV CON SEÑAL
    banner("[4] Grouped CV (LOPO) on data WITH injected patient-level signal")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    results = run_grouped_cv(X, y, groups, model, splitter=make_lopo_splitter(), n_boot=2000)
    patient_df = results["patient_df"]
    print(f"  aggregation returned {len(patient_df)} rows (one per patient)")
    assert len(patient_df) == 17, "epoch->patient aggregation must return 17 rows"
    assert set(patient_df.columns) >= {"patient_id", "prob_good", "pred", "y_true", "n_epochs"}
    assert (patient_df["n_epochs"] == 2000).all()
    print(patient_df.to_string(index=False))
    print("\n  Bootstrap CI (resampling PATIENTS, N=17):")
    print(results["patient_ci"].to_string(index=False))
    print("\n  Fold variance (LOPO; epoch AUC is NaN by design -> n_folds=0):")
    print(results["fold_variance"].to_string(index=False))
    print("\n  Calibration:", results["calibration"])
    print(results["calibration_curve"].to_string(index=False))
    assert np.isnan(results["fold_metrics"]["epoch_roc_auc"]).all(), (
        "in LOPO the within-fold AUC must be undefined (single class in test)"
    )

    # ------------------------------------------------- AGREGACIÓN: OTRAS REGLAS
    banner("[5] Aggregation rules")
    for rule in AGGREGATION_RULES:
        agg = aggregate_epoch_to_patient(
            results["oof_prob_epoch"], groups, y_epoch=y, rule=rule
        )
        metrics = patient_level_metrics(agg)
        assert len(agg) == 17
        print(f"  rule={rule:>13}: n_rows={len(agg)}  patient AUC={metrics['roc_auc']:.3f}  "
              f"balacc={metrics['balanced_accuracy']:.3f}")

    # ---------------------------------------------------- CONTROL: PERMUTACIÓN
    banner("[6] Negative control: permutation test (labels shuffled BY PATIENT)")
    perm = permutation_test_patient_labels(
        X, y, groups, n_permutations=50, max_epochs_per_patient=200
    )
    print(f"  null AUC mean={perm['null_mean']:.3f} (expected ~0.5), "
          f"std={perm['null_std']:.3f}, p={perm['p_value']:.4f}")
    assert 0.35 <= perm["null_mean"] <= 0.65, (
        f"permuted-label AUC should collapse to chance, got {perm['null_mean']:.3f}"
    )
    assert perm["observed_auc"] > perm["null_mean"], "signal must beat the null"
    print("  OK: with patient-level shuffled labels performance collapses to chance")

    # ------------------------------------------------------ CONTROL: IDENTIDAD
    banner("[7] Negative control: patient-identity probe")
    probe = patient_identity_probe(X, groups)
    print(f"  identity accuracy={probe['accuracy']:.3f} vs chance={probe['chance_level']:.3f}")

    # ------------------------------------------------------- FUGA DELIBERADA
    banner("[8] DELIBERATE LEAK: a feature that encodes the patient label")
    df_leak = make_synthetic_feature_frame(inject_leak=True)
    check_feature_contract(df_leak)
    leak_features = [c for c in df_leak.columns if c not in META_COLUMNS]
    X_leak = df_leak[leak_features].to_numpy(dtype=float)
    y_leak = df_leak["label"].to_numpy(dtype=int)
    g_leak = df_leak["patient_id"].to_numpy(dtype=object)
    print(f"  leaking column present: 'leak_marker_Fp1' in features = "
          f"{'leak_marker_Fp1' in leak_features}")
    leak_results = run_grouped_cv(
        X_leak, y_leak, g_leak, LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        splitter=make_lopo_splitter(), bootstrap=False,
    )
    leak_auc = leak_results["patient_metrics"]["roc_auc"]
    clean_auc = results["patient_metrics"]["roc_auc"]
    print(f"  patient AUC with leak = {leak_auc:.3f}  (clean run = {clean_auc:.3f})")
    assert clean_auc < 0.99, (
        "the clean synthetic run is saturated at AUC~1.0; the leak contrast is "
        "meaningless -- lower effect_size in make_synthetic_feature_frame"
    )
    assert leak_auc > 0.99, (
        "the framework FAILED to surface a deliberate leak: patient AUC should be ~1.0"
    )
    print("  OK: the deliberate leak is detected (AUC saturates at ~1.0)")
    print("  NOTE: grouped CV does NOT protect against a feature that encodes the")
    print("        label -- only the permutation test + feature inspection do.")

    banner(f"ALL CHECKS PASSED in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    # Si la matriz real ya existe, se usa como validación extra (solo lectura).
    _self_check()
    if FEATURES_PATH.exists():
        print("\n" + "=" * 78)
        print("[extra] Real features.parquet found -- running contract check on it")
        print("=" * 78)
        X, y, groups, names, meta = load_feature_matrix(FEATURES_PATH)
        assert_no_patient_leakage(make_lopo_splitter(), X, y, groups)
    else:
        print(f"\n(no real feature matrix at {FEATURES_PATH} -- synthetic check only)")
