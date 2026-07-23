"""Fase 4 — estimación insesgada (CV anidada), controles negativos y congelado.

Se ejecuta desde `development/`:
    uv run python old_scripts/05_nested_cv_and_freeze.py > old_scripts/nested_log.txt 2>&1

El barrido de `04_...` es EXPLORACIÓN: su máximo está sesgado al alza porque el
ganador se elige mirando la misma métrica que se reporta. Este script produce la
estimación honesta (selección de modelo DENTRO de cada fold externo), pasa los
controles negativos y congela el modelo para la Fase 5.
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")

# sklearn 1.8 deprecó `penalty` en LogisticRegression a favor de `l1_ratio`.
# El comportamiento es el mismo en esta versión; el aviso solo ensucia la salida.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import modeling as mdl
import validation as val

SEED = mdl.RANDOM_SEED
N_EPOCHS = 200          # ver restricción 1 del módulo: menos épocas != peor
REDUCTION = "channel_agg"

print("=" * 78)
print("FASE 4 — CV anidada, controles negativos y congelado del modelo")
print("=" * 78, flush=True)

X, y, groups, feature_names, meta = val.load_feature_matrix(verbose=False)
Xs, ys, gs = mdl.subsample_epochs(X, y, groups, N_EPOCHS)
print(f"matriz completa {X.shape} -> submuestreada {Xs.shape} "
      f"({N_EPOCHS} épocas/paciente, {len(np.unique(gs))} pacientes)\n", flush=True)

# --- 0. La comprobación anti-fuga, explícita ---------------------------------
rep = val.assert_no_patient_leakage(val.make_lopo_splitter(), Xs, ys, gs)
print(f"[0] anti-fuga LOPO: {len(rep)} folds, cero solapamiento de pacientes\n", flush=True)

# --- 1. Estimación (casi) insesgada por CV anidada ---------------------------
# Candidatos: familias distintas, todas fuertemente regularizadas.
candidates = {
    k: v for k, v in mdl.candidate_classifiers().items()
    if k in ("logreg_l2_C0.01", "logreg_l2_C0.1", "logreg_l1_C0.1",
             "rf_depth3", "hgb_depth3", "dummy_majority")
}
print(f"[1] CV anidada — {len(candidates)} candidatos, LOPO externo:", flush=True)
t0 = time.perf_counter()
nested = mdl.nested_cv_estimate(Xs, ys, gs, feature_names, candidates,
                                reduction=REDUCTION, verbose=True)
print(f"    ({time.perf_counter() - t0:.1f}s)")

nm = nested["patient_metrics"]
print("\n    ESTIMACIÓN INSESGADA (nivel paciente, N=17):")
for k in ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "brier"):
    if k in nm and nm[k] is not None:
        print(f"      {k:20s} {nm[k]:.3f}")
print("\n    IC bootstrap (remuestreando PACIENTES):")
print(nested["patient_ci"].to_string(index=False))

chosen = pd.Series(nested["chosen_per_fold"]).value_counts()
print("\n    modelo elegido en cada fold externo:")
print(chosen.to_string())
print("    (si el ganador baila entre folds, ninguno domina de verdad)")

# --- 2. Comparación de reglas de agregación época -> paciente ----------------
print("\n[2] reglas de agregación época->paciente:", flush=True)
for rule in ("mean", "median", "good_fraction", "trimmed_mean"):
    pdf = val.aggregate_epoch_to_patient(nested["oof_prob_epoch"], gs, y_epoch=ys, rule=rule)
    m = val.patient_level_metrics(pdf)
    print(f"    {rule:14s} AUC={m.get('roc_auc', float('nan')):.3f}  "
          f"balacc={m.get('balanced_accuracy', float('nan')):.3f}")

# --- 3. Controles negativos --------------------------------------------------
best_name = chosen.index[0]
best_clf = mdl.candidate_classifiers()[best_name]
best_pipe = mdl.build_pipeline(best_clf, feature_names, reduction=REDUCTION)
print(f"\n[3] controles negativos sobre '{best_name}':", flush=True)

t0 = time.perf_counter()
perm = val.permutation_test_patient_labels(
    Xs, ys, gs, model=best_pipe, n_permutations=200,
    max_epochs_per_patient=N_EPOCHS, seed=SEED, verbose=False,
)
print(f"    permutación: observado={perm['observed_auc']:.3f}  "
      f"nulo={perm['null_mean']:.3f}±{perm['null_std']:.3f}  "
      f"p={perm['p_value']:.4f}  (n={perm['n_permutations']}, "
      f"{time.perf_counter() - t0:.1f}s)")

probe = val.patient_identity_probe(Xs, gs, verbose=False)
print(f"    sonda de identidad: accuracy={probe['accuracy']:.3f} "
      f"(x{probe['ratio_over_chance']:.1f} sobre azar)")

# --- 4. Calibración ----------------------------------------------------------
cal, curve = val.calibration_report(nested["patient_df"], n_bins=4)
print(f"\n[4] calibración: Brier={cal['brier_score']:.3f}  "
      f"ECE={cal['expected_calibration_error']:.3f}  "
      f"(prob. media predicha={cal['mean_predicted_prob']:.3f} vs "
      f"tasa real de good={cal['observed_good_rate']:.3f})")

# --- 5. Congelar el modelo para la Fase 5 ------------------------------------
print("\n[5] congelando el modelo elegido…", flush=True)
model_path, meta_path = mdl.freeze_model(
    best_pipe, Xs, ys, feature_names,
    metadata={
        "selection_criterion": (
            "Modelo mas votado por la CV anidada (seleccion DENTRO de cada fold "
            "externo LOPO), sobre features agregadas por canal (380->20) y 200 "
            "epocas/paciente. Criterio fijado ANTES de mirar el held-out."
        ),
        "classifier": best_name,
        "reduction": REDUCTION,
        "epochs_per_patient": N_EPOCHS,
        "aggregation_rule": "mean",
        "decision_threshold": 0.5,
        "nested_cv_patient_metrics": {
            k: float(v) for k, v in nm.items()
            if isinstance(v, (int, float, np.floating, np.integer))
            and not isinstance(v, bool)
        },
        "permutation_p_value": float(perm.get("p_value")),
        "patient_identity_probe_accuracy": float(probe.get("accuracy")),
        "n_development_patients": int(len(np.unique(gs))),
        "honest_reading": (
            "La CV anidada NO muestra discriminacion por encima del azar. El modelo "
            "se congela para poder aplicar el protocolo de Fase 5 sobre el held-out, "
            "no porque haya evidencia de senal."
        ),
    },
)
print(f"    modelo -> {model_path}")
print(f"    ficha  -> {meta_path}")

# --- 6. Guardar las predicciones por paciente --------------------------------
out = Path("data_processed/nested_cv_patient_predictions.csv")
nested["patient_df"].to_csv(out, index=False)
print(f"    predicciones por paciente -> {out}")
print("\n" + "=" * 78)
print("HECHO")
print("=" * 78)
