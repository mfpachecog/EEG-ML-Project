"""
=============================================================================
FASE 2 — Pipeline de preprocesamiento de la señal EEG (motor: MNE)
=============================================================================

Convierte los segmentos EEG CRUDOS de un paciente I-CARE en un conjunto de
ÉPOCAS limpias y homogéneas, listas para extraer features (Fase 3).

El pipeline sigue el orden EXACTO decidido en el finding 016 (y ADR-005):

    1. Seleccionar los 19 canales estándar 10-20 en orden canónico (ADR-003).
    2. Filtrar: notch a la frecuencia de red del paciente + pasa-banda 0.5-45 Hz.
    3. Re-muestrear a 100 Hz — DESPUÉS de filtrar (el corte en 45 Hz es el
       anti-aliasing: Nyquist a 100 Hz = 50 Hz > 45 Hz).
    4. Re-referenciar a CAR (promedio común de todos los canales).
    5. Segmentar en épocas fijas de 10 s (estacionariedad + muchas muestras).
    6. Rechazar épocas malas por z-score ROBUSTO (no umbral fijo en µV, porque
       las amplitudes I-CARE están en 'nu' sin calibrar): descartar tramos
       planos (varianza casi-cero = desconexión) y extremos relativos.

DECISIONES DE DATOS (verificadas en la auditoría, finding 019):
    - La HORA post-ROSC de cada segmento está en su nombre: `PID_seg_HORA_EEG`.
      Solo procesamos segmentos con HORA en la ventana 24-72h (ADR-002).
    - La FRECUENCIA DE RED es por paciente y viene en la cabecera `.hea`
      (`#Utility frequency: 50` o `60`). El notch la lee de ahí, no la hardcodea.
    - La ETIQUETA binaria sale del `.txt` del paciente: `Outcome: Good/Poor`
      -> 1/0 (equivalente a CPC 1-2 vs 3-5).

NOTA SOBRE UNIDADES ('nu'):
    MNE asume Voltios internamente, pero las amplitudes I-CARE están en 'nu'
    (unidades normalizadas, sin calibrar). No pasa nada: TODAS las operaciones
    del pipeline son lineales o RELATIVAS (filtrado, CAR, z-score robusto), así
    que el resultado no depende de la escala absoluta. Solo evitamos interpretar
    los valores como microvoltios reales.
=============================================================================
"""
from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import wfdb

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    CANONICAL_CHANNELS,
    ROSC_WINDOW_HOURS,
    TARGET_SFREQ_HZ,
    development_patient_ids,
)

# MNE es muy verboso por defecto; lo bajamos a WARNING para no ahogar la salida.
mne.set_log_level("WARNING")

# --- Constantes del pipeline (un único lugar donde ajustar la Fase 2) --------
BANDPASS_HZ = (0.5, 45.0)     # pasa-banda (paso 2)
EPOCH_SEC = 10.0              # duración de cada época (paso 5)

# Umbrales del rechazo de épocas (paso 6). Son RELATIVOS, no absolutos en µV.
ROBUST_Z_THRESH = 4.0         # |z robusto| mayor a esto = época extrema -> se descarta
FLAT_STD_FRACTION = 1e-3      # canal con std < fracción de la mediana global = plano

# Cap de épocas por paciente (decisión: balancear el peso entre sujetos + acotar disco).
# 48h de EEG dan ~16k épocas/paciente; con 17 sujetos eso es desbalance + decenas de GB.
# Nos quedamos con un techo por paciente, distribuido uniformemente en el tiempo (stride),
# suficiente de sobra para un PoC. None = conservar todas.
MAX_EPOCHS_PER_PATIENT = 2000
# Para NO procesar los 59 segmentos y botar casi todo, acotamos ANTES cuántos segmentos
# tocamos: procesamos solo los suficientes (espaciados) para reunir el cap con margen.
# Ahorra cómputo además de disco. Estimación deliberadamente conservadora:
_EPOCHS_PER_SEGMENT_EST = 250  # épocas ~esperadas por segmento tras re-muestreo (~52 min / 10 s)
_SEGMENT_SAFETY = 2            # margen x2 por segmentos cortos o épocas rechazadas

# Mapa de nomenclatura moderna -> antigua (misma convención que el script 03).
_MODERN_TO_OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

# Directorio de salida de las épocas procesadas (data derivada; va en .gitignore).
PROCESSED_DIR = PROJECT_ROOT / "development" / "data_processed"


# =============================================================================
# NORMALIZACIÓN Y SELECCIÓN DE CANALES  (Paso 1)
# =============================================================================

def normalize_channel_name(ch_name: str) -> str:
    """Normaliza la nomenclatura de un canal: T7->T3, FP1->Fp1, etc.

    Distintos equipos etiquetan el mismo electrodo de formas distintas. Sin
    esto, 'T7' y 'T3' parecerían canales diferentes y romperían la alineación.
    """
    ch = ch_name.strip()
    if ch in _MODERN_TO_OLD:
        return _MODERN_TO_OLD[ch]
    if ch.upper().startswith("FP"):
        return "Fp" + ch.upper()[2:]
    return ch


def _select_canonical_columns(signal_data: np.ndarray, raw_names: list[str]) -> np.ndarray:
    """Devuelve `signal_data` reducido y reordenado a los 19 canales canónicos.

    ENTRADA:  signal_data (n_muestras, n_canales_originales), raw_names de wfdb.
    SALIDA:   (n_muestras, 19) con las columnas en el orden de CANONICAL_CHANNELS.

    POR QUÉ EL ORDEN IMPORTA: garantiza que la columna i signifique SIEMPRE el
    mismo electrodo en todos los pacientes; así los vectores de features quedan
    alineados entre sujetos y el modelo no aprende basura.
    """
    norm_to_idx: dict[str, int] = {}
    for idx, raw in enumerate(raw_names):
        norm = normalize_channel_name(raw)
        norm_to_idx.setdefault(norm, idx)  # nos quedamos con la primera aparición

    cols, missing = [], []
    for ch in CANONICAL_CHANNELS:
        (cols.append(norm_to_idx[ch]) if ch in norm_to_idx else missing.append(ch))
    if missing:
        raise ValueError(f"Missing standard channels: {missing}")

    return signal_data[:, cols]


# =============================================================================
# LECTURA DE METADATA DE LA CABECERA  (frecuencia de red, hora)
# =============================================================================

def parse_utility_freq(hea_path: Path, default: int = 50) -> int:
    """Lee la frecuencia de red (`#Utility frequency`) de una cabecera `.hea`.

    I-CARE la declara por registro (50 o 60 Hz según la región del hospital).
    Leerla en vez de asumirla evita dejar un pico de red sin filtrar.
    """
    for line in hea_path.read_text().splitlines():
        if "utility frequency" in line.lower():
            match = re.search(r"(\d+)", line)
            if match:
                return int(match.group(1))
    return default


def segment_hour(record_name: str) -> int:
    """Extrae la HORA post-ROSC del nombre del segmento (`PID_seg_HORA_EEG`)."""
    # p. ej. '0284_001_004_EEG' -> 4
    return int(record_name.split("_")[2])


def in_window_eeg_records(patient_dir: Path) -> list[str]:
    """Nombres de los segmentos EEG del paciente DENTRO de la ventana 24-72h.

    Solo estos entran al pipeline: reflejan el estado neurológico real, sin el
    sesgo de sedación/hipotermia de las primeras horas (ADR-002).
    """
    low, high = ROSC_WINDOW_HOURS
    records = []
    for hea in sorted(patient_dir.glob("*_EEG.hea")):
        name = hea.stem  # sin extensión
        if low <= segment_hour(name) <= high:
            records.append(name)
    return records


# =============================================================================
# CONSTRUCCIÓN DEL OBJETO Raw DE MNE  (Paso 1)
# =============================================================================

def load_segment_as_raw(record_path: Path) -> tuple[mne.io.RawArray, int]:
    """Carga UN segmento crudo y lo devuelve como `mne.RawArray` de 19 canales.

    Hace de puente entre el I/O de PhysioNet (wfdb) y el ecosistema MNE:
    lee la señal, selecciona/reordena los 19 canales canónicos y arma el Raw.

    DEVUELVE: (raw, utility_freq) — la frecuencia de red para el notch posterior.
    """
    record = wfdb.rdrecord(str(record_path))
    utility_freq = parse_utility_freq(record_path.with_suffix(".hea"))

    # Seleccionar y reordenar a los 19 canónicos; wfdb da (n_muestras, n_canales).
    sig_19 = _select_canonical_columns(record.p_signal, list(record.sig_name))

    # MNE espera (n_canales, n_muestras); además interpola NaN por seguridad
    # (filtfilt/filtros propagan NaN a toda la señal si no se tratan).
    data = _interpolate_nans(sig_19).T

    info = mne.create_info(
        ch_names=list(CANONICAL_CHANNELS), sfreq=record.fs, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)
    return raw, utility_freq


def _interpolate_nans(data: np.ndarray) -> np.ndarray:
    """Interpola linealmente los NaN de cada canal (matriz n_muestras x n_canales).

    Manejo mínimo para que los filtros no propaguen NaN. El rechazo serio de
    artefactos es el paso 6; esto solo asegura una señal continua para filtrar.
    """
    data = data.copy()
    n_samples, n_channels = data.shape
    x = np.arange(n_samples)
    for c in range(n_channels):
        col = data[:, c]
        nan_mask = np.isnan(col)
        if nan_mask.all():
            data[:, c] = 0.0                      # canal muerto -> queda en cero
        elif nan_mask.any():
            col[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], col[~nan_mask])
            data[:, c] = col
    return data


# =============================================================================
# PASOS 2-4: FILTRADO, RE-MUESTREO, RE-REFERENCIA
# =============================================================================

def filter_raw(raw: mne.io.BaseRaw, utility_freq: int) -> mne.io.BaseRaw:
    """Paso 2 — notch a la frecuencia de red + pasa-banda 0.5-45 Hz (in place).

    ORDEN: primero el notch (quita el pico agudo de la red eléctrica), luego el
    pasa-banda (recorta deriva lenta <0.5 Hz y alta frecuencia/EMG >45 Hz). El
    notch es cinturón-y-tirantes: aunque el pasa-banda ya atenúa por encima de
    45 Hz, si la red (50/60) es fuerte conviene eliminarla explícitamente.
    """
    # Notch en la fundamental y armónicos que aún caigan bajo Nyquist.
    nyq = raw.info["sfreq"] / 2.0
    notch_freqs = np.arange(utility_freq, nyq, utility_freq)
    if len(notch_freqs):
        raw.notch_filter(freqs=notch_freqs)
    raw.filter(l_freq=BANDPASS_HZ[0], h_freq=BANDPASS_HZ[1])
    return raw


def resample_raw(raw: mne.io.BaseRaw, sfreq: int = TARGET_SFREQ_HZ) -> mne.io.BaseRaw:
    """Paso 3 — re-muestrea a `sfreq` Hz (por defecto 100).

    Va DESPUÉS del filtrado: el pasa-banda a 45 Hz actúa como anti-aliasing,
    porque la nueva Nyquist (sfreq/2 = 50 Hz) queda por encima de 45 Hz.
    Homogeneiza las tasas heterogéneas del dataset (250-2048 Hz) a una común.
    """
    raw.resample(sfreq)
    return raw


def rereference_car(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Paso 4 — referencia promedio común (CAR).

    Resta, en cada instante, el promedio de los 19 canales. Elimina el ruido
    COMÚN a todo el cuero cabelludo (que no aporta información de localización)
    y hace la señal independiente del electrodo de referencia original.
    """
    raw.set_eeg_reference(ref_channels="average", projection=False)
    return raw


# =============================================================================
# PASO 5: SEGMENTACIÓN EN ÉPOCAS
# =============================================================================

def epoch_raw(raw: mne.io.BaseRaw, epoch_sec: float = EPOCH_SEC) -> mne.Epochs:
    """Paso 5 — parte la señal continua en épocas fijas y contiguas de `epoch_sec`.

    Épocas de 10 s: cortas para asumir estacionariedad (la señal no cambia de
    régimen dentro de la ventana) pero largas para estimar bien las bandas de
    frecuencia. Cada segmento genera muchas épocas -> muchas muestras/paciente.
    """
    return mne.make_fixed_length_epochs(raw, duration=epoch_sec, preload=True)


# =============================================================================
# PASO 6: RECHAZO DE ÉPOCAS MALAS  (z-score robusto)
# =============================================================================

def reject_bad_epochs(
    epochs: mne.Epochs,
    z_thresh: float = ROBUST_Z_THRESH,
    flat_fraction: float = FLAT_STD_FRACTION,
) -> tuple[mne.Epochs, dict]:
    """Paso 6 — descarta épocas planas o de amplitud extrema (criterio RELATIVO).

    POR QUÉ ROBUSTO Y RELATIVO: las amplitudes están en 'nu' sin calibrar, así
    que un umbral fijo en µV no tiene sentido. Usamos la MEDIANA y la MAD
    (desviación absoluta mediana), que NO se dejan arrastrar por los propios
    outliers que queremos detectar — a diferencia de media y desviación estándar.

    DOS CRITERIOS DE RECHAZO (finding 016):
      - PLANA: algún canal con std casi-cero (< `flat_fraction` de la mediana
        global de stds) = electrodo desconectado o tramo muerto.
      - EXTREMA: la amplitud de la época (su std mediana entre canales) es un
        outlier robusto, |z| > `z_thresh`, respecto al resto de épocas.

    DEVUELVE: (épocas_limpias, stats) con el conteo y las razones del descarte.
    """
    data = epochs.get_data()                       # (n_ep, n_ch, n_times)
    n_ep = data.shape[0]

    # std por (época, canal) -> base de ambos criterios.
    ch_std = data.std(axis=2)                       # (n_ep, n_ch)

    # --- Criterio PLANA: canal con varianza casi nula ---
    global_median_std = np.median(ch_std)
    flat_thresh = flat_fraction * global_median_std
    is_flat = (ch_std < flat_thresh).any(axis=1)    # (n_ep,) True si algún canal plano

    # --- Criterio EXTREMA: z-score robusto sobre la amplitud de la época ---
    epoch_amp = np.median(ch_std, axis=1)           # una amplitud representativa por época
    med = np.median(epoch_amp)
    mad = np.median(np.abs(epoch_amp - med))
    # 1.4826 hace la MAD comparable a la desviación estándar en una normal.
    robust_z = (epoch_amp - med) / (1.4826 * mad) if mad > 0 else np.zeros(n_ep)
    is_extreme = np.abs(robust_z) > z_thresh

    keep_mask = ~(is_flat | is_extreme)
    stats = {
        "n_input": int(n_ep),
        "n_flat": int(is_flat.sum()),
        "n_extreme": int(is_extreme.sum()),
        "n_rejected": int((~keep_mask).sum()),
        "n_kept": int(keep_mask.sum()),
    }
    return epochs[keep_mask], stats


# =============================================================================
# ETIQUETA DEL PACIENTE
# =============================================================================

def get_patient_label(pid: str) -> tuple[int, str]:
    """Lee la etiqueta binaria del `.txt` del paciente. DEVUELVE (label, outcome).

    `Outcome: Good` -> 1 (despertará), `Outcome: Poor` -> 0. Equivale a la
    dicotomía CPC 1-2 (good) vs 3-5 (poor) del pronóstico neurológico.
    """
    txt = (DATA_DIR / pid / f"{pid}.txt").read_text()
    match = re.search(r"Outcome:\s*(\w+)", txt)
    if not match:
        raise ValueError(f"No Outcome found for patient {pid}")
    outcome = match.group(1)
    return (1 if outcome.lower() == "good" else 0), outcome


# =============================================================================
# SUBMUESTREO UNIFORME (cap por paciente)
# =============================================================================

def _strided_indices(n_items: int, n_keep: int) -> np.ndarray:
    """Índices de `n_keep` elementos ESPACIADOS uniformemente entre 0 y n_items-1.

    Preserva la cobertura temporal: en vez de tomar las primeras N épocas (un
    trozo de las 48h), toma épocas repartidas por toda la ventana. Si `n_keep`
    es >= n_items, devuelve todos los índices.
    """
    if n_keep >= n_items:
        return np.arange(n_items)
    return np.unique(np.linspace(0, n_items - 1, n_keep).round().astype(int))


# =============================================================================
# ORQUESTACIÓN POR PACIENTE
# =============================================================================

def preprocess_patient(
    pid: str,
    max_epochs: int | None = MAX_EPOCHS_PER_PATIENT,
    verbose: bool = True,
) -> mne.Epochs:
    """Aplica el pipeline completo (pasos 1-6) a un paciente y devuelve sus épocas.

    Para no procesar 48h de EEG y luego botar casi todo, primero acota cuántos
    SEGMENTOS toca (espaciados, cubriendo la ventana) hasta reunir `max_epochs`
    con margen; procesa esos (pasos 1-5), concatena, rechaza malas (paso 6), y
    recorta al cap con submuestreo uniforme. Adjunta la etiqueta como metadata.

    `max_epochs=None` procesa TODOS los segmentos y conserva todas las épocas.
    """
    patient_dir = DATA_DIR / pid
    records = in_window_eeg_records(patient_dir)
    if not records:
        raise ValueError(f"Patient {pid} has no EEG segments in the 24-72h window")
    n_total_segments = len(records)

    # Acotar segmentos a procesar (ahorra cómputo): los justos para el cap + margen.
    if max_epochs is not None:
        n_seg = min(
            n_total_segments,
            int(np.ceil(_SEGMENT_SAFETY * max_epochs / _EPOCHS_PER_SEGMENT_EST)),
        )
        records = [records[i] for i in _strided_indices(n_total_segments, n_seg)]

    label, outcome = get_patient_label(pid)

    per_segment_epochs = []
    for name in records:
        raw, utility_freq = load_segment_as_raw(patient_dir / name)
        filter_raw(raw, utility_freq)      # paso 2
        resample_raw(raw)                  # paso 3
        rereference_car(raw)               # paso 4
        per_segment_epochs.append(epoch_raw(raw))  # paso 5

    # Unir las épocas de todos los segmentos procesados en un solo objeto.
    all_epochs = mne.concatenate_epochs(per_segment_epochs)
    clean_epochs, stats = reject_bad_epochs(all_epochs)  # paso 6

    # Recortar al cap con submuestreo uniforme en el tiempo.
    n_before_cap = len(clean_epochs)
    if max_epochs is not None and n_before_cap > max_epochs:
        clean_epochs = clean_epochs[_strided_indices(n_before_cap, max_epochs)]

    # Adjuntar etiqueta y trazabilidad como metadata (viaja con el -epo.fif).
    clean_epochs.metadata = pd.DataFrame(
        {"patient_id": pid, "label": label, "outcome": outcome},
        index=range(len(clean_epochs)),
    )

    if verbose:
        print(
            f"  {pid} [{outcome:>4}] : {len(records)}/{n_total_segments} segments -> "
            f"{stats['n_input']} epochs -> clean {stats['n_kept']} "
            f"(flat {stats['n_flat']}, extreme {stats['n_extreme']}) -> "
            f"saved {len(clean_epochs)}"
        )
    return clean_epochs


# =============================================================================
# BATCH SOBRE EL CONJUNTO DE DESARROLLO
# =============================================================================

def run_batch(out_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Procesa los 17 pacientes de desarrollo y guarda un `-epo.fif` por paciente.

    UN ARCHIVO POR PACIENTE, con el ID en el nombre: hace que la separación por
    sujeto sea FÍSICA. Es imposible que épocas del mismo paciente caigan a la vez
    en train y test -> misma filosofía anti-leakage que el conjunto held-out.

    DEVUELVE un DataFrame-resumen (una fila por paciente) para inspección rápida.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in development_patient_ids():
        try:
            epochs = preprocess_patient(pid)
            out_path = out_dir / f"patient_{pid}-epo.fif"
            epochs.save(out_path, overwrite=True)
            rows.append(
                {"patient_id": pid, "label": int(epochs.metadata["label"].iloc[0]),
                 "outcome": epochs.metadata["outcome"].iloc[0],
                 "n_epochs": len(epochs), "file": out_path.name}
            )
        except Exception as exc:  # no dejamos que un paciente tumbe el batch
            print(f"  !! {pid}: FAILED ({exc})")
            rows.append({"patient_id": pid, "label": None, "outcome": "ERROR",
                         "n_epochs": 0, "file": str(exc)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Ejecución directa = corre el batch completo sobre los 17 de desarrollo.
    print("=" * 70)
    print("FASE 2 — batch de preprocesamiento (17 pacientes de desarrollo)")
    print("=" * 70)
    summary = run_batch()
    print("\n" + "=" * 70)
    print(summary.to_string(index=False))
    total = int(summary["n_epochs"].sum())
    print(f"\nTotal epochs saved: {total}  ->  {PROCESSED_DIR}")
