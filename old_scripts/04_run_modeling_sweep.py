"""Barrido de modelos de la Fase 4 (LOPO-CV) con guardado incremental.

Se ejecuta desde `development/`:
    uv run python old_scripts/04_run_modeling_sweep.py > old_scripts/sweep_log.txt 2>&1

Guarda `data_processed/modeling_sweep.csv` tras CADA configuración, de modo que
un cuelgue o un corte no destruya lo ya calculado (lección de la Fase 2, donde
un apagón se llevó la salida de consola de un batch completo).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import modeling as mdl
import validation as val

OUT_CSV = Path("data_processed/modeling_sweep.csv")

print("=" * 78)
print("FASE 4 — barrido de modelos con LOPO-CV")
print("=" * 78, flush=True)

X, y, groups, feature_names, meta = val.load_feature_matrix(verbose=True)
print(f"\nmatriz: {X.shape} · {len(np.unique(groups))} pacientes", flush=True)

t0 = time.perf_counter()
df = mdl.sweep(
    X, y, groups, feature_names,
    reductions=("channel_agg", "pca", "select_k", "none"),
    epochs_per_patient=(200,),
    out_csv=OUT_CSV,
    verbose=True,
)
print(f"\nbarrido base terminado en {time.perf_counter() - t0:.1f}s", flush=True)

# --- Efecto del nº de épocas por paciente (restricción 1) --------------------
# Se prueba solo con las configuraciones baratas: el objetivo es medir la
# TENDENCIA con el nº de épocas, no volver a comparar todos los modelos.
cheap = {
    k: v for k, v in mdl.candidate_classifiers().items()
    if k.startswith(("logreg", "dummy", "hgb", "rf_depth3"))
}
print("\n=== efecto del nº de épocas por paciente ===", flush=True)
df_ep = mdl.sweep(
    X, y, groups, feature_names,
    reductions=("channel_agg",),
    epochs_per_patient=(50, 200, 500, 2000),
    classifiers=cheap,
    out_csv=Path("data_processed/modeling_sweep_epochs.csv"),
    verbose=True,
)

full = pd.concat([df, df_ep], ignore_index=True)
full.to_csv(OUT_CSV, index=False)

ok = full[full["status"].astype(str).str.startswith("ok")]
print("\n" + "=" * 78)
print("TOP 12 por AUC a nivel de PACIENTE (⚠️ exploración, sesgada por selección)")
print("=" * 78)
cols = ["n_epochs_per_patient", "reduction", "classifier",
        "patient_auc", "patient_balacc", "patient_sens", "patient_spec", "seconds"]
print(ok.sort_values("patient_auc", ascending=False)[cols].head(12).to_string(index=False))

print("\nSUELO (dummy):")
print(ok[ok.classifier == "dummy_majority"][cols].to_string(index=False))

print(f"\nconfiguraciones: {len(full)}  ·  fallidas: {(~full.status.astype(str).str.startswith('ok')).sum()}")
print(f"CSV -> {OUT_CSV}")
