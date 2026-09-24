"""Fase 2: re-ejecuta el preprocesamiento de los 17 pacientes EN MEMORIA y lo verifica.

Dos cosas a la vez:
1. Recoge, por paciente, las cifras que necesita el capítulo de Resultados:
   segmentos usados / disponibles, épocas candidatas, planas, extremas, limpias
   y guardadas. `preprocess_patient` solo las imprime, así que se capturan de
   su salida.
2. Compara las épocas recién calculadas con las del `-epo.fif` guardado. Si son
   idénticas, el pipeline es determinista y los archivos de disco son
   lo que el código produce hoy (salvo el redondeo de guardarlos en float32).

No escribe ningún .fif: solo un CSV con las cifras.

Uso:
    cd development && uv run python scripts/verify_fase2.py
Salida:
    data_processed/fase2_rejection_stats.csv
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from preprocessing import preprocess_patient  # noqa: E402

PROCESSED = ROOT / "data_processed"
OUT_CSV = PROCESSED / "fase2_rejection_stats.csv"

# Formato de la línea que imprime preprocess_patient(verbose=True).
LINE = re.compile(
    r"(?P<pid>\d{4}) \[\s*(?P<outcome>\w+)\] : (?P<seg_used>\d+)/(?P<seg_total>\d+) segments -> "
    r"(?P<n_input>\d+) epochs -> clean (?P<n_clean>\d+) \(flat (?P<n_flat>\d+), "
    r"extreme (?P<n_extreme>\d+)\) -> saved (?P<n_saved>\d+)"
)


def main() -> None:
    fifs = sorted(PROCESSED.glob("patient_*-epo.fif"))
    assert len(fifs) == 17, f"expected 17 development files, found {len(fifs)}"
    rows = []
    for fif in fifs:
        pid = fif.name.split("_")[1].split("-")[0]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            epochs = preprocess_patient(pid, verbose=True)
        match = LINE.search(buf.getvalue())
        assert match, f"could not parse stats for {pid}: {buf.getvalue()!r}"
        row = {k: (v if k in ("pid", "outcome") else int(v)) for k, v in match.groupdict().items()}

        saved = mne.read_epochs(fif, preload=True, verbose="ERROR").get_data()
        fresh = epochs.get_data()
        # MNE guarda las épocas en precisión simple (float32): la comparación exacta con lo
        # recién calculado en doble precisión falla por redondeo. El criterio correcto es que la
        # diferencia no supere el error de redondeo de float32 a la escala de la señal.
        same_shape = saved.shape == fresh.shape
        row["max_abs_diff"] = float(np.max(np.abs(saved - fresh))) if same_shape else float("nan")
        row["float32_rounding_bound"] = float(np.finfo(np.float32).eps * np.max(np.abs(saved)))
        row["matches_saved"] = bool(same_shape and row["max_abs_diff"] <= row["float32_rounding_bound"])
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(f"matches saved (within float32 rounding): {df['matches_saved'].sum()}/17  "
          f"max diff: {df['max_abs_diff'].max():.3g}")


if __name__ == "__main__":
    main()
