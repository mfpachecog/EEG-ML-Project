"""
FASE 5 — Evaluación del modelo CONGELADO sobre el conjunto held-out.

QUÉ ES ESTO
-----------
El último paso del proyecto y el más delicado en cuanto a honestidad metodológica.
El modelo quedó congelado al final de la Fase 4 (`models/frozen_model.joblib`) con
los 17 pacientes de desarrollo. Aquí se le enseñan, POR PRIMERA Y ÚNICA VEZ, 15
pacientes que nunca vio: el conjunto held-out.

POR QUÉ EL HELD-OUT ES LA COLUMNA VERTEBRAL ANTI-LEAKAGE
--------------------------------------------------------
En la Fase 4 el anti-leakage descansaba en la disciplina del código: agrupar por
paciente, ajustar el escalador dentro del fold, etc. Todo eso es correcto pero es
*una promesa que hace el código*. El held-out no es una promesa: son SUJETOS
DISTINTOS, descargados aparte, que el modelo no pudo ver porque físicamente no
estaban en `data_processed/` cuando se entrenó. El leakage entre desarrollo y
prueba no es "improbable" aquí: es IMPOSIBLE por construcción. Por eso este número
es el que se lleva a la defensa.

LA REGLA QUE NO SE ROMPE: UNA SOLA PASADA
-----------------------------------------
El held-out se mira UNA vez y se reporta lo que salga. En el momento en que se
mira el resultado y se vuelve atrás a tocar el modelo, las features o el umbral,
el held-out deja de ser held-out y se convierte en un conjunto de validación más
— y el número deja de significar nada. Este módulo, por eso, SOLO llama a
`predict_proba`. No hay ni una sola llamada a `fit` en todo el archivo, y eso es
deliberado.

LAS TRES TRAMPAS QUE ESTE MÓDULO BLOQUEA (auditoría del 2026-07-23)
-------------------------------------------------------------------
1. REORDENAMIENTO SILENCIOSO DE COLUMNAS. `ChannelAggregator` (modeling.py) indexa
   las features POR POSICIÓN, no por nombre. Si el held-out llega con las mismas
   380 columnas en otro orden, el modelo promediaría familias equivocadas y
   devolvería números plausibles PERO FALSOS, sin lanzar ningún error. Por eso
   `align_features_to_frozen()` reindexa explícitamente al orden guardado en la
   ficha y falla ruidosamente si algo no cuadra.
2. COMPARAR CONTRA 0.5. El azar de este protocolo está medido en 0.388, no en 0.5
   (finding 026). Comparar el held-out contra 0.5 produce la lectura falsa de que
   "el modelo va peor que el azar". El informe incluye la referencia correcta.
3. BOOTSTRAP SOBRE ÉPOCAS. Remuestrear las ~30 000 épocas en vez de los 15
   pacientes daría un intervalo absurdamente estrecho. Se reutiliza
   `validation.bootstrap_ci_patient_level`, que remuestrea PACIENTES.

USO
---
    uv run python src/evaluation.py --check       # ¿qué pacientes están listos?
    uv run python src/evaluation.py --preprocess  # Fase 2 sobre el held-out
    uv run python src/evaluation.py --features    # Fase 3 sobre el held-out
    uv run python src/evaluation.py --evaluate    # Fase 5: el número final
    uv run python src/evaluation.py --all         # todo lo anterior en orden
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DATA_DIR, HELD_OUT_PATIENTS, PROJECT_ROOT
import features as feat
import preprocessing as prep
import validation as val


# =============================================================================
# RUTAS — el held-out vive en su PROPIO subdirectorio
# =============================================================================
# POR QUÉ SEPARADO: `features.development_fif_paths()` hace glob sobre
# `data_processed/patient_*-epo.fif`. Si los `-epo.fif` del held-out cayeran ahí,
# cualquier re-ejecución de la Fase 3 los absorbería. La guardia
# `assert_no_held_out` lo atraparía (por eso existe), pero es mejor que la
# situación no se pueda ni dar: separar los directorios hace el error imposible
# en vez de meramente detectable.
HELD_OUT_DIR = prep.PROCESSED_DIR / "held_out"
HELD_OUT_FEATURES_PATH = HELD_OUT_DIR / "features_held_out.parquet"

MODELS_DIR = PROJECT_ROOT / "development" / "models"
FROZEN_MODEL_PATH = MODELS_DIR / "frozen_model.joblib"
FROZEN_CARD_PATH = MODELS_DIR / "frozen_model.json"

RESULTS_DIR = HELD_OUT_DIR
PATIENT_PREDICTIONS_PATH = RESULTS_DIR / "held_out_patient_predictions.csv"
RESULTS_JSON_PATH = RESULTS_DIR / "held_out_results.json"

# Mínimo de épocas para que un paciente sea evaluable. El modelo consume 200
# épocas/paciente; por debajo de eso la probabilidad agregada se vuelve ruidosa.
MIN_EPOCHS_PER_PATIENT = 200


# =============================================================================
# 0. ESTADO DE LA DESCARGA
# =============================================================================

def held_out_download_status() -> pd.DataFrame:
    """Qué pacientes held-out están descargados y con cuántos segmentos en ventana.

    La descarga es lenta (~3.5 h/paciente) y corre en segundo plano, así que la
    Fase 5 debe poder arrancar sabiendo exactamente con qué cuenta. Devuelve una
    fila por paciente held-out con el estado real en disco.
    """
    rows = []
    for pid in sorted(HELD_OUT_PATIENTS):
        patient_dir = DATA_DIR / pid
        txt_ok = (patient_dir / f"{pid}.txt").exists()

        # Segmentos EEG en la ventana 24-72h con .hea Y .mat presentes: un .hea
        # sin su .mat es un segmento a medio descargar y wfdb fallaría al leerlo.
        complete = []
        if patient_dir.exists():
            for name in prep.in_window_eeg_records(patient_dir):
                if (patient_dir / f"{name}.mat").exists():
                    complete.append(name)

        label, outcome = (None, None)
        if txt_ok:
            try:
                label, outcome = prep.get_patient_label(pid)
            except Exception:  # noqa: BLE001 - .txt a medio escribir
                pass

        rows.append(
            {
                "patient_id": pid,
                "txt": txt_ok,
                "n_segments_complete": len(complete),
                "outcome": outcome,
                "label": label,
                # ~125 épocas por segmento de 10s; con 2 segmentos ya se superan
                # las 200 épocas mínimas, pero pedimos 4 por margen de rechazo.
                "ready": txt_ok and len(complete) >= 4,
            }
        )
    return pd.DataFrame(rows)


def print_download_status() -> pd.DataFrame:
    """Imprime el estado de la descarga del held-out y devuelve la tabla."""
    status = held_out_download_status()
    n_ready = int(status["ready"].sum())
    print(f"\n=== HELD-OUT DOWNLOAD STATUS ({n_ready}/{len(status)} ready) ===")
    print(status.to_string(index=False))
    if n_ready:
        ready = status[status["ready"]]
        n_good = int((ready["label"] == 1).sum())
        n_poor = int((ready["label"] == 0).sum())
        print(f"\nready balance: {n_good} good / {n_poor} poor")
    return status


# =============================================================================
# 1. FASE 2 SOBRE EL HELD-OUT
# =============================================================================

def preprocess_held_out(
    out_dir: Path = HELD_OUT_DIR,
    only_ready: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aplica el pipeline de Fase 2 a los pacientes held-out, uno por archivo.

    ⚠️ EL PIPELINE ES EXACTAMENTE EL MISMO que el de desarrollo: se llama a
    `preprocessing.preprocess_patient`, no a una copia adaptada. Cualquier
    diferencia de preprocesamiento entre desarrollo y held-out invalidaría la
    comparación — el modelo vería una distribución distinta por razones técnicas
    y no clínicas. Reutilizar la función literal es lo que garantiza que no pasa.

    No hay nada que "ajustar" aquí: el preprocesamiento es determinista y por
    paciente (filtros de coeficientes fijos, CAR intra-segmento, z-score robusto
    intra-paciente). No aprende del conjunto, así que aplicarlo al held-out no es
    leakage de ningún tipo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    status = held_out_download_status()
    targets = status[status["ready"]] if only_ready else status

    rows = []
    for _, row in targets.iterrows():
        pid = row["patient_id"]
        out_path = out_dir / f"patient_{pid}-epo.fif"
        if out_path.exists():
            if verbose:
                print(f"  {pid}: already processed, skipping")
            rows.append({"patient_id": pid, "status": "cached", "path": str(out_path)})
            continue
        try:
            epochs = prep.preprocess_patient(pid, verbose=verbose)
            epochs.save(out_path, overwrite=True, verbose="ERROR")
            rows.append(
                {
                    "patient_id": pid,
                    "status": "OK",
                    "n_epochs": len(epochs),
                    "path": str(out_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - un paciente roto no debe tumbar el batch
            print(f"  {pid}: ERROR -> {exc}")
            rows.append({"patient_id": pid, "status": "ERROR", "error": str(exc)})

    return pd.DataFrame(rows)


# =============================================================================
# 2. FASE 3 SOBRE EL HELD-OUT
# =============================================================================

def build_held_out_features(
    fif_dir: Path = HELD_OUT_DIR,
    out_path: Path = HELD_OUT_FEATURES_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extrae las 380 features del held-out con el MISMO extractor de la Fase 3.

    `allow_held_out=True` es el único sitio de todo el proyecto donde se desactiva
    la guardia anti-leakage, y es correcto hacerlo aquí: el modelo ya está
    congelado en disco, así que ver estos pacientes no puede influir en él.
    """
    paths = sorted(fif_dir.glob("patient_*-epo.fif"))
    if not paths:
        raise FileNotFoundError(
            f"No held-out epoch files in {fif_dir}. Run --preprocess first."
        )

    df = feat.build_feature_matrix(
        fif_paths=paths,
        out_path=out_path,
        allow_held_out=True,   # <- justificado arriba; NUNCA poner True en Fase 3/4
        verbose=verbose,
    )

    # Comprobación simétrica a `assert_no_held_out`: aquí el error sería el
    # CONTRARIO -- que se hubiera colado un paciente de DESARROLLO en el held-out,
    # lo que inflaría el resultado (el modelo ya vio a ese sujeto).
    intruders = sorted(set(df["patient_id"].astype(str)) - set(HELD_OUT_PATIENTS))
    if intruders:
        raise RuntimeError(
            f"DEVELOPMENT LEAKAGE GUARD: non-held-out patients in the held-out "
            f"feature matrix: {intruders}. The model has already seen them."
        )
    return df


# =============================================================================
# 3. EL MODELO CONGELADO
# =============================================================================

def load_frozen_model() -> tuple[object, dict]:
    """Carga el Pipeline congelado y su ficha JSON.

    NOTA TÉCNICA: el `.joblib` contiene un `ChannelAggregator`, que es una clase
    definida en `modeling.py`. Al deserializar, pickle necesita poder importar ese
    módulo, así que `src/` tiene que estar en `sys.path` o el `joblib.load` muere
    con `ModuleNotFoundError: No module named 'modeling'`. Se garantiza aquí para
    que el módulo funcione también si lo importa un notebook desde otra carpeta.
    """
    src_dir = str(Path(__file__).resolve().parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if not FROZEN_MODEL_PATH.exists():
        raise FileNotFoundError(f"Frozen model not found: {FROZEN_MODEL_PATH}")

    model = joblib.load(FROZEN_MODEL_PATH)
    card = json.loads(FROZEN_CARD_PATH.read_text(encoding="utf-8"))
    return model, card


def align_features_to_frozen(df: pd.DataFrame, card: dict) -> np.ndarray:
    """Reordena las columnas del held-out al ORDEN EXACTO con el que se entrenó.

    ⚠️ ESTA FUNCIÓN ES LA MÁS IMPORTANTE DEL MÓDULO, y protege contra un fallo que
    NO daría ningún error si no existiera.

    `ChannelAggregator` promedia las 380 columnas en 20 familias usando índices de
    POSICIÓN calculados durante el `fit`. Es decir: da por hecho que la columna 0
    es `rel_delta_Fp1`, la 1 `rel_delta_Fp2`, etc. Si el held-out llega con las
    mismas 380 columnas pero en otro orden — algo que un `groupby`, un `merge` o
    una versión distinta de pandas puede provocar sin avisar — el modelo
    promediaría, por ejemplo, curtosis con potencia relativa, y devolvería
    probabilidades perfectamente plausibles y completamente falsas.

    Un bug silencioso que produce números creíbles es el peor tipo de bug que
    puede tener una tesis: no se detecta al ejecutar, se detecta en la defensa.
    Por eso aquí se reindexa explícitamente por NOMBRE y se falla ruidosamente
    ante cualquier discrepancia.
    """
    expected = list(card["feature_names"])
    present = set(df.columns)

    missing = [c for c in expected if c not in present]
    if missing:
        raise ValueError(
            f"FEATURE CONTRACT VIOLATION: {len(missing)} features expected by the "
            f"frozen model are missing from the held-out matrix: {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    extra = sorted(present - set(expected) - set(val.META_COLUMNS))
    if extra:
        raise ValueError(
            f"FEATURE CONTRACT VIOLATION: {len(extra)} unexpected feature columns "
            f"in the held-out matrix: {extra[:10]}{' ...' if len(extra) > 10 else ''}"
        )

    # Reindexado explícito por nombre: esta línea es la que hace el trabajo.
    X = df[expected].to_numpy(dtype=float)

    if X.shape[1] != int(card["n_features_in"]):
        raise ValueError(
            f"FEATURE CONTRACT VIOLATION: aligned matrix has {X.shape[1]} columns, "
            f"frozen model expects {card['n_features_in']}"
        )
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(f"Held-out matrix contains {n_bad} NaN/inf values")

    return X


# =============================================================================
# 4. LA EVALUACIÓN — UNA SOLA PASADA
# =============================================================================

def evaluate_held_out(
    features_path: Path = HELD_OUT_FEATURES_PATH,
    n_boot: int = 2000,
    verbose: bool = True,
) -> dict:
    """Aplica el modelo congelado al held-out y devuelve el resultado final.

    PROTOCOLO (fijado en `frozen_model.json` ANTES de mirar el held-out):
      1. cargar features del held-out;
      2. submuestrear a `epochs_per_patient` épocas ESPACIADAS por paciente, igual
         que en la CV, para que el régimen de entrada sea el mismo;
      3. reindexar columnas al orden del modelo congelado;
      4. `predict_proba` — jamás `fit`;
      5. agregar época -> paciente con la regla de la ficha ('mean');
      6. métricas de paciente + IC bootstrap REMUESTREANDO PACIENTES;
      7. comparar contra la nula LOPO (0.388), NO contra 0.5.
    """
    model, card = load_frozen_model()
    df = pd.read_parquet(features_path)

    # --- Paso 2: mismo régimen de épocas que en la CV -------------------------
    groups_all = df["patient_id"].astype(str).to_numpy()
    keep = val._subsample_epochs_per_patient(
        groups_all, int(card.get("epochs_per_patient", 200))
    )
    df = df.iloc[keep].reset_index(drop=True)

    # --- Paso 3: el reindexado que evita el bug silencioso --------------------
    X = align_features_to_frozen(df, card)
    groups = df["patient_id"].astype(str).to_numpy()
    y_epoch = df["label"].to_numpy(dtype=int)

    # --- Paso 4: SOLO predicción. Ni un `fit` en todo el módulo. --------------
    y_prob_epoch = val._positive_class_proba(model, X)

    # --- Paso 5: época -> paciente -------------------------------------------
    rule = card.get("aggregation_rule", "mean")
    threshold = float(card.get("decision_threshold", 0.5))
    patient_df = val.aggregate_epoch_to_patient(
        y_prob_epoch, groups, y_epoch=y_epoch, rule=rule, threshold=threshold
    )

    # --- Paso 6: métricas + IC de PACIENTE -----------------------------------
    metrics = val.patient_level_metrics(patient_df, threshold=threshold)
    ci = val.bootstrap_ci_patient_level(
        patient_df, n_boot=n_boot, threshold=threshold
    )
    calibration = val.calibration_report(
        patient_df["y_true"].to_numpy(dtype=int),
        patient_df["prob_good"].to_numpy(dtype=float),
    )

    # --- Paso 7: la referencia de azar CORRECTA ------------------------------
    null = card.get("lopo_null_distribution", {})
    null_mean = float(null.get("mean", float("nan")))
    null_sd = float(null.get("sd", float("nan")))
    observed_auc = float(metrics["roc_auc"])
    z_vs_null = (
        (observed_auc - null_mean) / null_sd if np.isfinite(null_sd) and null_sd else float("nan")
    )

    results = {
        "n_patients": int(len(patient_df)),
        "n_epochs_used": int(len(df)),
        "epochs_per_patient": int(card.get("epochs_per_patient", 200)),
        "aggregation_rule": rule,
        "decision_threshold": threshold,
        "patient_metrics": metrics,
        "bootstrap_ci": ci.to_dict(orient="records"),
        "calibration": calibration if isinstance(calibration, dict) else None,
        "chance_reference": {
            "lopo_null_mean": null_mean,
            "lopo_null_sd": null_sd,
            "z_vs_lopo_null": z_vs_null,
            "warning": card.get("chance_reference_warning", card.get("chance_level_warning", "")),
        },
        "development_comparison": {
            "nested_cv_patient_auc": card.get("nested_cv_patient_metrics", {}).get("roc_auc"),
            "held_out_patient_auc": observed_auc,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(PATIENT_PREDICTIONS_PATH, index=False)
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    if verbose:
        _print_report(patient_df, results, ci)
    return results


def _print_report(patient_df: pd.DataFrame, results: dict, ci: pd.DataFrame) -> None:
    """Informe legible del resultado de la Fase 5, listo para copiar al finding."""
    m = results["patient_metrics"]
    ref = results["chance_reference"]
    print("\n" + "=" * 70)
    print("PHASE 5 -- FROZEN MODEL ON THE HELD-OUT SET (single pass)")
    print("=" * 70)
    print(f"patients: {results['n_patients']}   epochs used: {results['n_epochs_used']}")
    print(f"aggregation: {results['aggregation_rule']}   threshold: {results['decision_threshold']}")

    print("\n--- per-patient predictions ---")
    print(patient_df.to_string(index=False))

    print("\n--- patient-level metrics ---")
    for key in ("roc_auc", "balanced_accuracy", "accuracy", "sensitivity",
                "specificity", "tn", "fp", "fn", "tp", "brier_score"):
        if key in m:
            print(f"  {key:20s} {m[key]}")

    print("\n--- bootstrap CI (patients resampled, NOT epochs) ---")
    print(ci.to_string(index=False))

    print("\n--- chance reference ---")
    print(f"  LOPO null mean : {ref['lopo_null_mean']}  (sd {ref['lopo_null_sd']})")
    print(f"  observed AUC   : {m['roc_auc']}")
    print(f"  z vs LOPO null : {ref['z_vs_lopo_null']}")
    print("  !! chance for this protocol is ~0.388, NOT 0.5 -- do not compare against 0.5")

    dev = results["development_comparison"]
    print("\n--- development vs held-out ---")
    print(f"  nested CV (development, n=17) : {dev['nested_cv_patient_auc']}")
    print(f"  held-out (n={results['n_patients']})              : {dev['held_out_patient_auc']}")
    print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def _self_check() -> None:
    """Verifica la cadena completa de la Fase 5 SIN tocar el conjunto held-out.

    POR QUÉ EXISTE: el held-out se mira UNA sola vez. Eso significa que el código
    de la Fase 5 se ejecuta por primera vez sobre los datos que producen el
    resultado final de la tesis — el peor momento posible para descubrir un bug.
    Esta función ejercita todo el camino (cargar modelo → reindexar → predecir →
    agregar) con datos SINTÉTICOS, de modo que cuando lleguen los pacientes reales
    la primera ejecución ya no sea la primera.

    Mismo criterio que `validation._self_check()`: un test que puede fallar.
    """
    print("=" * 70)
    print("PHASE 5 SELF-CHECK -- synthetic data, held-out never touched")
    print("=" * 70)

    model, card = load_frozen_model()
    names = list(card["feature_names"])
    print(f"\n[1] frozen model loaded: {type(model).__name__} ({len(names)} features)")

    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    df.insert(0, "patient_id", np.repeat(["S1", "S2", "S3"], n // 3))
    df.insert(1, "label", np.repeat([1, 0, 1], n // 3))
    df.insert(2, "outcome", np.repeat(["Good", "Poor", "Good"], n // 3))
    df.insert(3, "epoch_idx", list(range(n // 3)) * 3)

    print("\n[2] happy path")
    X = align_features_to_frozen(df, card)
    proba = val._positive_class_proba(model, X)
    assert X.shape == (n, len(names))
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    print(f"    X={X.shape}, proba in [{proba.min():.3f}, {proba.max():.3f}] -- no refit")

    print("\n[3] THE TRAP: shuffled column order must be repaired, not propagated")
    shuffled = names.copy()
    rng.shuffle(shuffled)
    X_shuf = align_features_to_frozen(
        df[["patient_id", "label", "outcome", "epoch_idx"] + shuffled], card
    )
    assert np.allclose(X, X_shuf), "column realignment FAILED -- silent-bug guard is broken"
    print("    shuffled columns -> realigned, identical to original: True")

    print("\n[4] loud failures (each MUST raise)")
    cases = {
        "missing column": df.drop(columns=[names[5]]),
        "extra column": df.assign(intruder_column=1.0),
        "NaN present": df.assign(**{names[0]: np.nan}),
    }
    for label, bad in cases.items():
        try:
            align_features_to_frozen(bad, card)
            raise AssertionError(f"{label}: did NOT raise -- the guard is broken")
        except ValueError as exc:
            print(f"    {label:16s} -> blocked: {str(exc)[:55]}")

    print("\n[5] epoch -> patient aggregation")
    patient_df = val.aggregate_epoch_to_patient(
        proba, df["patient_id"].to_numpy(), y_epoch=df["label"].to_numpy(), rule="mean"
    )
    assert len(patient_df) == 3 and "prob_good" in patient_df.columns
    print(patient_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("ALL PHASE 5 CHECKS PASSED -- chain ready for real data")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    do_all = "--all" in argv

    if "--self-check" in argv:
        _self_check()
        return 0

    if not argv or "--check" in argv or do_all:
        print_download_status()

    if "--preprocess" in argv or do_all:
        print("\n=== PHASE 2 ON HELD-OUT ===")
        print(preprocess_held_out().to_string(index=False))

    if "--features" in argv or do_all:
        print("\n=== PHASE 3 ON HELD-OUT ===")
        df = build_held_out_features()
        print(f"held-out feature matrix: {df.shape}")

    if "--evaluate" in argv or do_all:
        evaluate_held_out()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
