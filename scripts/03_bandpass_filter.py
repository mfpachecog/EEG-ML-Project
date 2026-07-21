"""
=============================================================================
FASE 2 — PASO 1: Filtrado pasa-banda de la señal EEG
=============================================================================

OBJETIVO:
    Tomar UN segmento crudo, seleccionar los 19 canales estándar en orden
    canónico, aplicar un filtro pasa-banda (0.5-45 Hz), y visualizar el
    resultado ANTES vs DESPUÉS, tanto en el dominio del tiempo como en el
    de la frecuencia.

POR QUÉ EMPEZAMOS CON UN SOLO SEGMENTO:
    Antes de procesar 20 pacientes × decenas de segmentos cada uno,
    necesitamos VER que el filtro hace lo que esperamos en un caso.
    "Bloques pequeños y verificables."

CONCEPTOS QUE VAS A VER EN ACCIÓN:
    - Selección de canales en orden canónico (alineación de features)
    - Filtro Butterworth (banda de paso plana, sin distorsión de amplitud)
    - filtfilt: filtrado de fase cero (sin desfase temporal)
    - PSD por método de Welch (cómo se distribuye la energía por frecuencia)
=============================================================================
"""

import os
import sys
import wfdb
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

# Ruta robusta a los datos: la centraliza src/config.py (evita rutas absolutas hardcodeadas)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from src.config import DATA_DIR
PATIENT_ID = "0284"

# Canal que vamos a visualizar en detalle (cámbialo para explorar otros)
VIZ_CHANNEL = "Fp1"

# ----- Parámetros del filtro -----
LOWCUT = 0.5      # Hz — límite inferior (quita deriva lenta)
HIGHCUT = 45.0    # Hz — límite superior (quita EMG y red eléctrica)
FILTER_ORDER = 4  # Orden del Butterworth (más alto = caída más pronunciada)

# ----- Orden canónico de los 19 canales del sistema 10-20 -----
# El ORDEN importa: garantiza que la columna 0 sea siempre Fp1 para TODOS
# los pacientes, la columna 1 siempre Fp2, etc. Así los vectores de features
# quedan alineados entre pacientes.
CANONICAL_ORDER = [
    'Fp1', 'Fp2',
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

MODERN_TO_OLD = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}


# ===========================================================================
# SELECCIÓN Y ORDENAMIENTO DE CANALES
# ===========================================================================

def normalize_channel_name(ch_name: str) -> str:
    """Normaliza nomenclatura: T7->T3, FP1->Fp1, etc."""
    ch = ch_name.strip()
    if ch in MODERN_TO_OLD:
        return MODERN_TO_OLD[ch]
    if ch.upper().startswith('FP'):
        return 'Fp' + ch.upper()[2:]
    return ch


def select_standard_channels(signal_data: np.ndarray, raw_names: list):
    """
    Selecciona los 19 canales estándar y los reordena al orden canónico.

    ENTRADA:
        signal_data: matriz (n_muestras, n_canales_originales)
        raw_names: lista de nombres de canales tal como vienen del archivo

    SALIDA:
        matriz (n_muestras, 19) con los canales en CANONICAL_ORDER

    POR QUÉ EL REORDENAMIENTO ES CRÍTICO:
        El paciente A puede tener sus canales en orden [C3, F3, Cz, ...] y
        el paciente B en orden [Fp1, Fp2, F7, ...]. Si no los reordenamos,
        la "columna 5" significaría cosas distintas en cada paciente y el
        modelo aprendería basura. El orden canónico garantiza consistencia.
    """
    # Construir un mapa: nombre_normalizado -> índice de columna original
    norm_to_idx = {}
    for idx, raw in enumerate(raw_names):
        norm = normalize_channel_name(raw)
        if norm not in norm_to_idx:  # quedarse con la primera aparición
            norm_to_idx[norm] = idx

    # Seleccionar columnas en el orden canónico
    cols = []
    missing = []
    for ch in CANONICAL_ORDER:
        if ch in norm_to_idx:
            cols.append(norm_to_idx[ch])
        else:
            missing.append(ch)

    if missing:
        raise ValueError(f"Faltan canales estándar: {missing}")

    selected = signal_data[:, cols]
    return selected


# ===========================================================================
# MANEJO BÁSICO DE NaN (placeholder hasta el paso de artefactos)
# ===========================================================================

def interpolate_nans_per_channel(data: np.ndarray) -> np.ndarray:
    """
    Interpola linealmente los NaN de cada canal.

    POR QUÉ: filtfilt NO funciona si hay NaN (los propaga a toda la señal).
    NOTA: esto es un PLACEHOLDER simple. El manejo serio de artefactos
          (rechazo de épocas malas) viene en el Paso 5 de la Fase 2.
    """
    data = data.copy()
    n_samples, n_channels = data.shape

    for c in range(n_channels):
        col = data[:, c]
        nan_mask = np.isnan(col)

        if nan_mask.all():
            # Canal completamente muerto: lo dejamos en cero
            data[:, c] = 0.0
        elif nan_mask.any():
            # Interpolar los NaN usando los valores válidos vecinos
            x = np.arange(n_samples)
            col[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], col[~nan_mask])
            data[:, c] = col

    return data


# ===========================================================================
# DISEÑO Y APLICACIÓN DEL FILTRO
# ===========================================================================

def design_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """
    Diseña un filtro Butterworth pasa-banda.

    DEVUELVE los coeficientes (b, a) que definen el filtro.

    CONCEPTO — FRECUENCIA NORMALIZADA:
        scipy espera las frecuencias de corte normalizadas respecto a la
        frecuencia de Nyquist (fs/2). Por eso dividimos lowcut y highcut
        entre nyq. Una frecuencia normalizada de 1.0 = Nyquist = fs/2.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def apply_zerophase_filter(data: np.ndarray, b, a) -> np.ndarray:
    """
    Aplica el filtro con fase cero (filtfilt) a lo largo del eje temporal.

    axis=0 porque nuestra matriz es (n_muestras, n_canales): el tiempo
    corre por las filas. filtfilt procesa cada canal (columna) por separado.

    CONCEPTO — FILTFILT (fase cero):
        filtfilt aplica el filtro hacia adelante, luego invierte la señal
        y lo aplica de nuevo. El desfase que introduce la primera pasada
        se cancela con la segunda. Resultado: la señal filtrada está
        perfectamente alineada en el tiempo con la original (sin retraso).
        Costo: el orden efectivo del filtro se duplica.
    """
    return signal.filtfilt(b, a, data, axis=0)


# ===========================================================================
# ANÁLISIS ESPECTRAL (PSD por método de Welch)
# ===========================================================================

def compute_psd(signal_1ch: np.ndarray, fs: float, window_sec: float = 2.0):
    """
    Calcula la Densidad Espectral de Potencia (PSD) de un canal.

    CONCEPTO — PSD (Power Spectral Density):
        Nos dice CUÁNTA energía tiene la señal en CADA frecuencia.
        Es como pasar la señal por un prisma: descompone la onda compleja
        en sus frecuencias componentes y mide la intensidad de cada una.

    CONCEPTO — MÉTODO DE WELCH:
        En vez de hacer una sola FFT sobre toda la señal (ruidosa), Welch
        parte la señal en ventanas, calcula el espectro de cada una, y los
        promedia. Resultado: un espectro más suave y confiable.

    NOTA: este método es EXACTAMENTE el que usaremos en la Fase 3 para
          extraer las features de potencia por banda. Aquí lo usamos solo
          para visualizar el efecto del filtro.
    """
    nperseg = int(window_sec * fs)  # muestras por ventana
    freqs, psd = signal.welch(signal_1ch, fs=fs, nperseg=nperseg)
    return freqs, psd


# ===========================================================================
# VISUALIZACIÓN ANTES / DESPUÉS
# ===========================================================================

def plot_before_after(raw_19, filt_19, fs, viz_idx, viz_name, out_path,
                      window_sec=10):
    """
    Genera una figura con 3 paneles:
    - Arriba izquierda: señal CRUDA en el tiempo
    - Arriba derecha: señal FILTRADA en el tiempo
    - Abajo (ancho completo): PSD cruda vs filtrada (dominio de frecuencia)
    """
    fig, axd = plt.subplot_mosaic(
        [['raw', 'filt'], ['psd', 'psd']],
        figsize=(15, 9)
    )

    # --- Ventana temporal para los plots de tiempo ---
    n = min(int(window_sec * fs), raw_19.shape[0])
    t = np.arange(n) / fs
    raw_ch = raw_19[:n, viz_idx]
    filt_ch = filt_19[:n, viz_idx]

    # --- Panel 1: señal cruda (tiempo) ---
    axd['raw'].plot(t, raw_ch, linewidth=0.6, color='#999999')
    axd['raw'].set_title(f'CRUDA — {viz_name} (dominio del tiempo)',
                         fontweight='bold')
    axd['raw'].set_xlabel('Tiempo (s)')
    axd['raw'].set_ylabel('Amplitud')
    axd['raw'].grid(alpha=0.3)

    # --- Panel 2: señal filtrada (tiempo) ---
    axd['filt'].plot(t, filt_ch, linewidth=0.6, color='#1a73e8')
    axd['filt'].set_title(f'FILTRADA 0.5-45 Hz — {viz_name} (dominio del tiempo)',
                          fontweight='bold')
    axd['filt'].set_xlabel('Tiempo (s)')
    axd['filt'].set_ylabel('Amplitud')
    axd['filt'].grid(alpha=0.3)

    # --- Panel 3: PSD cruda vs filtrada (frecuencia) ---
    # Usamos la señal COMPLETA (no solo la ventana) para la PSD
    f_raw, p_raw = compute_psd(raw_19[:, viz_idx], fs)
    f_filt, p_filt = compute_psd(filt_19[:, viz_idx], fs)

    axd['psd'].semilogy(f_raw, p_raw, color='#999999',
                        label='Cruda', linewidth=1)
    axd['psd'].semilogy(f_filt, p_filt, color='#1a73e8',
                        label='Filtrada', linewidth=1.2)

    # Marcar las bandas de frecuencia EEG
    band_edges = [0.5, 4, 8, 13, 30, 45]
    for edge in band_edges:
        axd['psd'].axvline(edge, color='green', alpha=0.25,
                          linestyle='--', linewidth=0.8)

    # Marcar las frecuencias de red eléctrica (50 y 60 Hz)
    for plf in [50, 60]:
        axd['psd'].axvline(plf, color='red', alpha=0.4,
                          linestyle=':', linewidth=1)
        axd['psd'].text(plf, axd['psd'].get_ylim()[1] * 0.3,
                       f'{plf}Hz', color='red', fontsize=8, ha='center')

    axd['psd'].set_xlim(0, 70)
    axd['psd'].set_title('PSD: dominio de la frecuencia '
                        '(verde=bandas EEG, rojo=red eléctrica)',
                        fontweight='bold')
    axd['psd'].set_xlabel('Frecuencia (Hz)')
    axd['psd'].set_ylabel('Densidad de potencia (escala log)')
    axd['psd'].legend()
    axd['psd'].grid(alpha=0.3)

    plt.suptitle(f'Efecto del filtrado pasa-banda — Canal {viz_name}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Figura guardada en: {out_path}")
    plt.show()


# ===========================================================================
# EJECUCIÓN PRINCIPAL
# ===========================================================================

def main():
    # --- Cargar el primer segmento EEG del paciente ---
    patient_dir = os.path.join(DATA_DIR, PATIENT_ID)
    all_files = os.listdir(patient_dir)
    eeg_hea = sorted([f for f in all_files
                      if f.endswith('.hea') and 'EEG' in f.upper()])

    if not eeg_hea:
        print(f"❌ No hay segmentos EEG para {PATIENT_ID}")
        sys.exit(1)

    record_name = eeg_hea[0].replace('.hea', '')
    record_path = os.path.join(patient_dir, record_name)

    print("=" * 70)
    print(f"FILTRADO — Paciente {PATIENT_ID}, segmento {record_name}")
    print("=" * 70)

    record = wfdb.rdrecord(record_path)
    fs = record.fs
    raw_signal = record.p_signal  # (n_muestras, n_canales)

    print(f"  fs original: {fs} Hz")
    print(f"  Forma cruda: {raw_signal.shape}")

    # --- Paso 1: seleccionar y ordenar los 19 canales ---
    sig_19 = select_standard_channels(raw_signal, record.sig_name)
    print(f"  Tras selección de 19 canales: {sig_19.shape}")

    # --- Paso 2: interpolar NaN (placeholder) ---
    sig_19 = interpolate_nans_per_channel(sig_19)

    # --- Paso 3: diseñar y aplicar el filtro pasa-banda ---
    b, a = design_bandpass(LOWCUT, HIGHCUT, fs, order=FILTER_ORDER)
    sig_filtered = apply_zerophase_filter(sig_19, b, a)
    print(f"  Filtro pasa-banda {LOWCUT}-{HIGHCUT} Hz aplicado")

    # --- Paso 4: estadísticas antes/después ---
    print(f"\n  Estadísticas del canal {VIZ_CHANNEL}:")
    viz_idx = CANONICAL_ORDER.index(VIZ_CHANNEL)
    print(f"    Cruda    — std: {np.std(sig_19[:, viz_idx]):8.2f}, "
          f"rango: [{np.min(sig_19[:, viz_idx]):.1f}, "
          f"{np.max(sig_19[:, viz_idx]):.1f}]")
    print(f"    Filtrada — std: {np.std(sig_filtered[:, viz_idx]):8.2f}, "
          f"rango: [{np.min(sig_filtered[:, viz_idx]):.1f}, "
          f"{np.max(sig_filtered[:, viz_idx]):.1f}]")

    # --- Paso 5: visualizar ---
    out_path = f"filtrado_{record_name}_{VIZ_CHANNEL}.png"
    plot_before_after(sig_19, sig_filtered, fs, viz_idx, VIZ_CHANNEL, out_path)

    print(f"\n{'='*70}")
    print("QUÉ OBSERVAR EN LA FIGURA")
    print(f"{'='*70}")
    print("""
    DOMINIO DEL TIEMPO (paneles superiores):
    - ¿La señal filtrada ya NO 'flota' ni se desplaza lentamente?
      → El filtro quitó la deriva (drift) de baja frecuencia.
    - ¿Se ve más 'centrada' en cero y más limpia?

    DOMINIO DE LA FRECUENCIA (panel inferior, el más importante):
    - La curva CRUDA (gris): mira si sube mucho hacia la izquierda
      (frecuencias <0.5 Hz) → esa es la deriva lenta.
    - ¿Hay un pico en 50 o 60 Hz (líneas rojas)? → red eléctrica.
    - La curva FILTRADA (azul): debe CAER en picada después de 45 Hz
      y también por debajo de 0.5 Hz.
    - PREGUNTA CLAVE: ¿el pico de 50/60 Hz desapareció en la curva azul?
      Si sí → el pasa-banda YA elimina la red eléctrica y NO necesitamos
      un filtro notch separado.
    """)


if __name__ == "__main__":
    main()
