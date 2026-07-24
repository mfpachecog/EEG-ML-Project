"""¿Cuál es el AZAR de verdad bajo LOPO? — distribución nula del protocolo primario.

Se ejecuta desde `development/`:
    uv run python old_scripts/06_lopo_null_distribution.py > old_scripts/lopo_null_log.txt 2>&1

MOTIVO (hallazgo del 2026-07-23): en LOPO el `DummyClassifier(strategy="prior")`
no da AUC 0.5 sino **0.000 exacto**. La causa es aritmética: al dejar fuera a un
paciente GOOD el train queda 8/8 y la prior de good es 0.5000; al dejar fuera a
uno POOR queda 9/7 y sube a 0.5625. Es decir, TODOS los poor reciben una
probabilidad de "good" mayor que todos los good -> orden perfectamente invertido
-> AUC 0.

Consecuencia: el propio protocolo LOPO **deprime** el AUC, porque el simple hecho
de sacar a un paciente desbalancea el train en la dirección contraria a su clase.
Eso explica que 55 de las 76 configuraciones no-dummy cayeran por debajo de 0.5
(mediana 0.451). Es un sesgo CONSERVADOR (a la baja), no uno que infle resultados.

Pero invalida comparar el AUC observado contra 0.5 "a ojo". La referencia honesta
es la distribución nula EMPÍRICA del mismo protocolo: barajar las etiquetas a
nivel de PACIENTE y repetir LOPO completo. Eso es lo que mide este script.
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np

import modeling as mdl
import validation as val

N_PERM = 100
N_EPOCHS = 200
SEED = mdl.RANDOM_SEED

print("=" * 78)
print("Distribución nula del AUC bajo LOPO (el 'azar' real de este protocolo)")
print("=" * 78, flush=True)

X, y, groups, feature_names, meta = val.load_feature_matrix(verbose=False)
Xs, ys, gs = mdl.subsample_epochs(X, y, groups, N_EPOCHS)

clf = mdl.candidate_classifiers()["hgb_depth3"]
pipe = mdl.build_pipeline(clf, feature_names, reduction="channel_agg")

t0 = time.perf_counter()
perm = val.permutation_test_patient_labels(
    Xs, ys, gs, model=pipe,
    n_permutations=N_PERM,
    splitter=val.make_lopo_splitter(),      # <- el protocolo PRIMARIO, no el de 5 folds
    max_epochs_per_patient=N_EPOCHS,
    seed=SEED, verbose=False,
)
elapsed = time.perf_counter() - t0

null = np.asarray(perm["null_aucs"], dtype=float)
null = null[np.isfinite(null)]
obs = float(perm["observed_auc"])

print(f"\nobservado (LOPO)      : {obs:.3f}")
print(f"nulo LOPO             : media={null.mean():.3f}  sd={null.std(ddof=1):.3f}  "
      f"mediana={np.median(null):.3f}  n={null.size}")
print(f"percentiles nulo      : p5={np.percentile(null, 5):.3f}  "
      f"p50={np.percentile(null, 50):.3f}  p95={np.percentile(null, 95):.3f}")
print(f"p-valor empírico      : {perm['p_value']:.4f}")
print(f"({elapsed:.0f}s)")

np.save("data_processed/lopo_null_aucs.npy", null)
print("\ndistribución nula LOPO -> data_processed/lopo_null_aucs.npy")

print("\nLECTURA:")
print("  El centro de la nula LOPO es la referencia honesta de 'azar' para este")
print("  protocolo — NO el 0.5 teórico. El observado se juzga contra ESA nula.")
if obs <= np.percentile(null, 95):
    print("  -> El observado NO supera el percentil 95 de la nula: sin evidencia de señal.")
else:
    print("  -> El observado SUPERA el percentil 95 de la nula: hay indicio, revisar.")
