"""Fase 7: medición del costo computacional del sistema construido.

Cumple la promesa del objetivo específico 4 y de la hipótesis: tiempo de
entrenamiento, tiempo de predicción y requisitos de hardware, medidos en la
máquina donde se desarrolló el trabajo. Mide tres etapas:

A. Un paciente nuevo, de EEG crudo a características: preprocesamiento completo
   (pasos 1-6 de la Fase 2) + extracción de las 380 características. Todo en
   memoria: NO escribe ningún .fif ni parquet, así que no toca los datos reales.
B. Entrenamiento: el ajuste del modelo congelado (misma configuración: 200 épocas
   por paciente, promedio de canales, HGB de profundidad 3) sobre los 17
   pacientes, repetido; y la validación cruzada anidada completa, una vez.
C. Predicción: el modelo congelado (frozen_model.joblib) sobre las 200 épocas de
   un paciente, repetido.

Cada etapa corre en un PROCESO APARTE para que su memoria máxima (ru_maxrss) no
se mezcle con la de las otras. La primera corrida de B y C es de calentamiento y
no se cuenta: incluye cargas en frío que no representan el costo real.

No usa el conjunto de prueba externo: la predicción se cronometra sobre un
paciente de desarrollo (el tiempo no depende de quién sea el paciente).

Uso:
    cd development && uv run python scripts/measure_cost.py
Salida:
    data_processed/cost_measurements.json
"""

from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT_PATH = ROOT / "data_processed" / "cost_measurements.json"

PATIENT_FOR_TIMING = "0284"
N_TRAIN_REPEATS = 10
N_PATIENT_REPEATS = 3
N_PREDICT_REPEATS = 100


def _peak_rss_mb() -> float:
    """Memoria máxima del proceso actual, en MB (en Linux ru_maxrss viene en kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def machine_specs() -> dict:
    """Especificaciones de la máquina y versiones: sin esto ningún tiempo significa nada."""
    import mne
    import numpy
    import sklearn

    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
         if line.startswith("model name")),
        platform.processor(),
    )
    mem_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith("MemTotal"))
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    physical = None
    cores_per_socket = sockets = None
    for line in lscpu.splitlines():
        if line.startswith("Core(s) per socket:"):
            cores_per_socket = int(line.split(":")[1])
        if line.startswith("Socket(s):"):
            sockets = int(line.split(":")[1])
    if cores_per_socket and sockets:
        physical = cores_per_socket * sockets

    gpu = "none detected"
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        gpu = r.stdout.strip() or "nvidia-smi present, no device listed"
    elif shutil.which("lspci"):
        vga = [l for l in subprocess.run(["lspci"], capture_output=True, text=True).stdout.splitlines()
               if "VGA" in l or "3D controller" in l]
        gpu = "; ".join(vga) or gpu

    return {
        "cpu_model": cpu_model,
        "physical_cores": physical,
        "logical_cores": os.cpu_count(),
        "ram_gb": round(mem_kb / 1024 / 1024, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "mne": mne.__version__,
        "graphics_hardware": gpu,
    }


# =============================================================================
# ETAPAS (cada una se ejecuta en su propio proceso)
# =============================================================================

def stage_patient_pipeline() -> dict:
    """A: EEG crudo -> 2000 épocas limpias -> 380 características, un paciente, en memoria."""
    import numpy as np
    from features import compute_epoch_features
    from preprocessing import preprocess_patient

    t0 = time.perf_counter()
    epochs = preprocess_patient(PATIENT_FOR_TIMING, verbose=False)
    t_pre = time.perf_counter() - t0

    t0 = time.perf_counter()
    feats = compute_epoch_features(epochs.get_data(), sfreq=epochs.info["sfreq"])
    t_feat = time.perf_counter() - t0

    n_epochs = len(epochs)
    n_features = int(sum(np.asarray(v).shape[1] for v in feats.values())) if isinstance(feats, dict) else None
    return {
        "patient": PATIENT_FOR_TIMING,
        "n_epochs": n_epochs,
        "n_features": n_features,
        "preprocessing_s": round(t_pre, 2),
        "feature_extraction_s": round(t_feat, 2),
        "total_s": round(t_pre + t_feat, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }


def stage_training() -> dict:
    """B: ajuste del modelo congelado (repetido) y CV anidada completa (una vez)."""
    import warnings

    import numpy as np
    from sklearn.base import clone

    warnings.filterwarnings("ignore")
    import modeling as mdl
    import validation as val

    X, y, groups, feature_names, _ = val.load_feature_matrix(verbose=False)
    Xs, ys, gs = mdl.subsample_epochs(X, y, groups, mdl.DEFAULT_EPOCHS_PER_PATIENT)
    rss_data_loaded = _peak_rss_mb()
    pipe = mdl.build_pipeline(mdl.candidate_classifiers()["hgb_depth3"], feature_names, reduction="channel_agg")

    clone(pipe).fit(Xs, ys)  # calentamiento, no se cuenta
    times = []
    for _ in range(N_TRAIN_REPEATS):
        t0 = time.perf_counter()
        clone(pipe).fit(Xs, ys)
        times.append(time.perf_counter() - t0)

    candidates = {k: v for k, v in mdl.candidate_classifiers().items()
                  if k in ("logreg_l2_C0.01", "logreg_l2_C0.1", "logreg_l1_C0.1",
                           "rf_depth3", "hgb_depth3", "dummy_majority")}
    t0 = time.perf_counter()
    nested = mdl.nested_cv_estimate(Xs, ys, gs, feature_names, candidates, reduction="channel_agg", verbose=False)
    t_nested = time.perf_counter() - t0

    return {
        "rows": int(Xs.shape[0]),
        "columns_in": int(Xs.shape[1]),
        "fit_repeats": N_TRAIN_REPEATS,
        "fit_median_s": round(float(np.median(times)), 3),
        "fit_min_s": round(float(np.min(times)), 3),
        "fit_max_s": round(float(np.max(times)), 3),
        "nested_cv_s": round(t_nested, 1),
        "nested_cv_auc_check": round(float(nested["patient_metrics"]["roc_auc"]), 4),
        "rss_after_loading_data_mb": round(rss_data_loaded, 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }


def stage_prediction() -> dict:
    """C: modelo congelado sobre las 200 épocas de un paciente -> probabilidad del paciente."""
    import joblib
    import numpy as np

    import modeling as mdl
    import validation as val

    X, y, groups, _, _ = val.load_feature_matrix(verbose=False)
    Xs, _, gs = mdl.subsample_epochs(X, y, groups, mdl.DEFAULT_EPOCHS_PER_PATIENT)
    X_patient = Xs[gs == PATIENT_FOR_TIMING]
    model = joblib.load(ROOT / "models" / "frozen_model.joblib")

    model.predict_proba(X_patient)  # calentamiento
    times = []
    for _ in range(N_PREDICT_REPEATS):
        t0 = time.perf_counter()
        prob = model.predict_proba(X_patient)[:, 1].mean()
        times.append(time.perf_counter() - t0)

    model_size_kb = (ROOT / "models" / "frozen_model.joblib").stat().st_size / 1024
    return {
        "patient": PATIENT_FOR_TIMING,
        "epochs": int(X_patient.shape[0]),
        "predict_repeats": N_PREDICT_REPEATS,
        "predict_median_ms": round(float(np.median(times)) * 1000, 2),
        "predict_max_ms": round(float(np.max(times)) * 1000, 2),
        "patient_prob_good": round(float(prob), 4),
        "model_file_kb": round(model_size_kb, 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }


STAGES = {"A": stage_patient_pipeline, "B": stage_training, "C": stage_prediction}


def main() -> None:
    # Modo hijo: ejecuta una sola etapa e imprime su JSON en la última línea.
    if len(sys.argv) == 2 and sys.argv[1] in STAGES:
        result = STAGES[sys.argv[1]]()
        print("RESULT_JSON " + json.dumps(result))
        return

    # Modo padre: especificaciones + una etapa por proceso.
    report = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine_specs()}
    print(json.dumps(report["machine"], indent=1, ensure_ascii=False), flush=True)
    for key in ("A", "B", "C"):
        # La etapa A se repite: la primera corrida lee el EEG en frío desde el disco.
        runs = []
        for _ in range(N_PATIENT_REPEATS if key == "A" else 1):
            print(f"stage {key} ...", flush=True)
            r = subprocess.run([sys.executable, __file__, key], capture_output=True, text=True, cwd=ROOT)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
            if line is None:
                print(r.stdout[-2000:], r.stderr[-4000:])
                raise SystemExit(f"stage {key} failed")
            runs.append(json.loads(line.removeprefix("RESULT_JSON ")))
            print(json.dumps(runs[-1], indent=1), flush=True)
        report[key] = runs if key == "A" else runs[0]

    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
