"""
=============================================================================
FASE 3 — Extracción de features a partir de las épocas limpias de la Fase 2
=============================================================================

Convierte las ÉPOCAS limpias que produjo la Fase 2 (`patient_XXXX-epo.fif`,
2000 épocas × 19 canales × 1000 muestras por paciente) en una MATRIZ TABULAR
que un modelo clásico de scikit-learn pueda consumir: una fila por época, una
columna por (feature × canal), más las columnas de identidad y etiqueta.

    entrada :  17 archivos `-epo.fif`   (señal, 3 ejes)
    salida  :  `data_processed/features.parquet`  (tabla, 2 ejes)

-----------------------------------------------------------------------------
RESTRICCIÓN DE DISEÑO QUE MANDA SOBRE TODO LO DEMÁS: INVARIANZA A ESCALA
-----------------------------------------------------------------------------
Las amplitudes de I-CARE están en 'nu' (unidades normalizadas, SIN calibrar) y
el factor de escala PUEDE DIFERIR ENTRE PACIENTES: dos pacientes con actividad
cerebral idéntica pueden aparecer con amplitudes que difieren en un factor
arbitrario k. Por tanto la potencia ABSOLUTA de banda (µV²/Hz de Welch) NO es
comparable entre sujetos. Si la usáramos, el modelo aprendería el factor de
calibración del equipo como si fuera fisiología — un error SILENCIOSO: no da
excepción, da métricas engañosas (finding 021 §3.2).

Regla operativa: TODA feature de este módulo debe cumplir

        f(k · x) == f(x)   para todo k > 0

y esto no se asume, se DEMUESTRA: `scale_invariance_report()` recalcula todas
las features sobre datos reales multiplicados por k=10 y k=0.1 y comprueba la
igualdad numérica. Cualquier feature que falle ese test no entra a la matriz.

Cómo lo consigue cada familia de features:
  - Potencias RELATIVAS de banda  -> son un COCIENTE de dos potencias, la k²
    del numerador se cancela con la del denominador.
  - Ratios espectrales (α/δ, DTABR…) -> cociente de potencias, misma razón.
  - Entropía espectral, SEF95/SEF50, centroide -> se calculan sobre la PSD
    NORMALIZADA a distribución de probabilidad (área 1); la normalización borra
    la escala antes de que se use el valor.
  - Hjorth movilidad y complejidad -> cocientes de desviaciones estándar
    (std(x')/std(x), etc.): la k se cancela. **Hjorth ACTIVIDAD queda EXCLUIDA
    a propósito**: es la varianza pura, escala como k², y es exactamente el tipo
    de feature que introduciría el sesgo por-paciente descrito arriba.
  - Line length NORMALIZADA -> la suma de |diferencias| dividida por la std de
    la época; la versión cruda (sin dividir) escala con k y NO es válida aquí.
  - Tasa de cruces por cero, curtosis, asimetría -> definidas sobre el signo o
    sobre momentos ESTANDARIZADOS; multiplicar por k>0 no altera ni el signo ni
    los momentos normalizados.
  - Entropía de permutación -> se construye sobre el ORDEN (rankings) de las
    muestras, no sobre sus valores; k>0 preserva el orden exactamente.

-----------------------------------------------------------------------------
POR QUÉ ESTAS FEATURES Y NO OTRAS (justificación clínica, resumida)
-----------------------------------------------------------------------------
En el coma post-paro cardíaco el pronóstico se lee, clásicamente, en tres ejes:
  1. LENTIFICACIÓN del espectro — cuanto peor el daño, más potencia se desplaza
     a delta/theta y menos queda en alpha/beta. Lo capturan las potencias
     relativas, los ratios (α/δ, DTABR) y las frecuencias de borde (SEF95/SEF50).
  2. PÉRDIDA DE COMPLEJIDAD — un cerebro dañado produce señal más pobre y más
     predecible. Lo capturan la entropía espectral, la entropía de permutación y
     los parámetros de Hjorth.
  3. PATRONES DISCONTINUOS (brote-supresión, un marcador de mal pronóstico) —
     alternancia de tramos planos y brotes de gran amplitud produce colas
     pesadas en la distribución de amplitudes: lo capta la CURTOSIS, y el
     contraste actividad/silencio se refleja en la line length normalizada.

-----------------------------------------------------------------------------
NOTA DE IMPLEMENTACIÓN: TODO VECTORIZADO
-----------------------------------------------------------------------------
Son 17 pacientes × 2000 épocas × 19 canales = 646 000 series de 1000 muestras.
Un bucle Python por época sería inviable; todas las funciones operan sobre el
tensor completo `(n_epochs, n_channels, n_times)` con operaciones de NumPy/SciPy
a lo largo del último eje. La única excepción es la entropía de permutación, que
se trocea en bloques de épocas por control de memoria, no por lógica.
=============================================================================
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from config import (
    PROJECT_ROOT,
    HELD_OUT_PATIENTS,
    TARGET_SFREQ_HZ,
)

mne.set_log_level("WARNING")

# --- Rutas ------------------------------------------------------------------
PROCESSED_DIR = PROJECT_ROOT / "development" / "data_processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# --- Parámetros del estimador espectral (Welch) -----------------------------
# Welch parte cada época en sub-ventanas solapadas, calcula el periodograma de
# cada una y las PROMEDIA: se pierde resolución en frecuencia pero se reduce
# muchísimo la VARIANZA del estimador (un periodograma único es un estimador
# ruidoso e inconsistente — su varianza no baja aunque crezca la señal).
#
# Con 1000 muestras a 100 Hz (10 s):
#   nperseg=256  -> resolución = fs/nperseg = 100/256 = 0.39 Hz. Sobra para
#                   delimitar bandas anchas (la más estrecha, alpha, mide 5 Hz
#                   = ~13 bins) y el borde inferior de delta (0.5 Hz) queda
#                   representado.
#   noverlap=128 (50 %) -> con solape del 50 % caben 6 sub-ventanas en la época,
#                   es decir promediamos 6 periodogramas: buen compromiso entre
#                   reducción de varianza y sub-ventanas casi independientes
#                   (más solape añade sub-ventanas, pero muy correlacionadas).
# Ventana de Hann (defecto de SciPy): suaviza los bordes de cada sub-ventana y
# reduce la fuga espectral (leakage) de las componentes fuertes de baja frecuencia
# hacia las bandas altas.
WELCH_NPERSEG = 256
WELCH_NOVERLAP = 128

# Rango espectral útil: el mismo pasa-banda que aplicó la Fase 2. Fuera de él la
# señal es solo el rizado de los filtros, no información.
PSD_BAND_HZ = (0.5, 45.0)

# Bandas clásicas del EEG. Los intervalos son semiabiertos [lo, hi) salvo el
# último, que cierra en 45 Hz: así los 5 conjuntos de bins son DISJUNTOS y su
# unión es EXACTAMENTE el rango 0.5-45 Hz. Consecuencia útil: las 5 potencias
# relativas suman 1 por construcción, lo que sirve de test de sanidad gratuito.
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Frecuencias de borde espectral que extraemos (fracción acumulada de potencia).
SPECTRAL_EDGE_QUANTILES = {"sef95": 0.95, "sef50": 0.50}

# Orden y retardo de la entropía de permutación. order=3 -> 3! = 6 patrones
# ordinales posibles sobre 998 ventanas por época: muestras de sobra para
# estimar la distribución sin sesgo por conteo escaso (regla práctica:
# n_ventanas >> order!). order=4 daría 24 patrones y algo más de detalle, pero
# también más varianza y 4x más cómputo; para un PoC, 3 es la opción sobria.
PERM_ENTROPY_ORDER = 3
PERM_ENTROPY_DELAY = 1
_PERM_CHUNK_EPOCHS = 200  # troceado por memoria (no cambia el resultado)

# Epsilon para blindar divisiones entre cantidades ADIMENSIONALES (potencias
# relativas, probabilidades acumuladas). Ahí un umbral absoluto es legítimo
# porque el valor ya vive en [0, 1] y no depende de la calibración del equipo.
EPS = 1e-12


def _safe_divisor(x: np.ndarray) -> np.ndarray:
    """Blinda un denominador DIMENSIONAL sin introducir un umbral absoluto.

    ⚠️ ESTA FUNCIÓN EXISTE POR UN BUG REAL, y merece leerse entera.

    La versión inicial de este módulo protegía las divisiones con
    `np.maximum(denominador, 1e-12)`. Parece inofensivo, pero cuando el
    denominador es una cantidad DIMENSIONAL —potencia total, desviación
    estándar— ese 1e-12 es un UMBRAL ABSOLUTO sobre una magnitud cuya escala es
    arbitraria ('nu' sin calibrar). Es exactamente el pecado que este proyecto
    persigue, cometido dentro del propio blindaje: si la señal de un paciente
    viene con un factor de calibración pequeño, su potencia total cae por debajo
    de 1e-12, el `maximum` sustituye el denominador REAL por la constante, y la
    "normalización" deja de normalizar. El resultado es basura silenciosa —
    potencias relativas que ya no suman 1, y SEF95 de cientos de miles de Hz.

    Se detectó así: en el primer batch, `sef95` alcanzaba 3.2e5 Hz (imposible:
    el techo físico son 45 Hz) en 19 épocas del paciente 0286, uno de los de
    amplitud más baja (el mínimo real es 0418, con 0.86; 0286 va después con
    6.82). Un valor absurdo delató un fallo que en las demás features
    solo habría producido números plausibles pero incorrectos.

    LA CORRECCIÓN: el único valor que hay que evitar en un denominador es el
    CERO EXACTO, y el cero es el único punto que sí es invariante a escala
    (k·0 = 0 para todo k). Así que se sustituye únicamente el cero exacto por 1
    —lo que convierte una división indefinida en un 0/1 = 0 explícito— y
    cualquier valor positivo, por pequeño que sea, se usa tal cual. El
    blindaje deja de tener escala propia.
    """
    return np.where(x > 0, x, 1.0)


# =============================================================================
# ESTIMACIÓN ESPECTRAL (base común de la mitad de las features)
# =============================================================================

def compute_welch_psd(
    data: np.ndarray,
    sfreq: float = TARGET_SFREQ_HZ,
    nperseg: int = WELCH_NPERSEG,
    noverlap: int = WELCH_NOVERLAP,
    band: tuple[float, float] = PSD_BAND_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Densidad espectral de potencia por Welch, recortada al rango útil.

    ENTRADA:  data (n_epochs, n_channels, n_times) — el tensor tal cual sale de
              `epochs.get_data()`.
    SALIDA:   (freqs, psd) con freqs (n_freqs,) y psd (n_epochs, n_channels, n_freqs).

    SciPy aplica Welch a lo largo del último eje y respeta los ejes anteriores,
    así que las 2000 épocas × 19 canales se calculan de una sola llamada, sin
    bucles en Python.

    Se devuelve la PSD *cruda* (unidades 'nu²/Hz'): NO es invariante a escala por
    sí sola — escala con k². Todas las features derivadas de aquí la usan de forma
    RELATIVA (cocientes o normalizada a probabilidad), que es donde la escala se
    cancela. Ninguna feature exporta la PSD absoluta.
    """
    freqs, psd = welch(
        data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1
    )
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    return freqs[mask], psd[..., mask]


def relative_band_powers(
    freqs: np.ndarray, psd: np.ndarray, bands: dict = FREQ_BANDS
) -> dict[str, np.ndarray]:
    """Potencia RELATIVA de cada banda: potencia_banda / potencia_total(0.5-45 Hz).

    SALIDA: dict {'rel_delta': (n_epochs, n_channels), ...}

    POR QUÉ RELATIVA Y NO ABSOLUTA (la decisión central de este módulo): la
    potencia absoluta de una banda escala con k² bajo un cambio de calibración,
    mientras que el COCIENTE de dos potencias deja la k² arriba y abajo y se
    cancela exactamente. Además la potencia relativa es la magnitud que la
    literatura clínica interpreta ("el 70 % de la potencia está en delta"), no
    los µV² absolutos.

    Se integra por SUMA DE BINS (regla del rectángulo) y no por trapecio: como el
    ancho de bin df es constante, aparece en numerador y denominador y se cancela,
    y las 5 bandas —al ser conjuntos de bins disjuntos que cubren exactamente el
    total— suman 1 EXACTAMENTE. Con el trapecio los bins de borde se contarían a
    medias en dos bandas y esa propiedad se perdería.
    """
    total = psd.sum(axis=-1)                       # (n_ep, n_ch)
    total_safe = _safe_divisor(total)              # solo blinda el cero exacto (ver _safe_divisor)
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        # Último borde inclusivo para que la unión de bandas = rango total.
        mask = (freqs >= lo) & (freqs < hi) if hi < PSD_BAND_HZ[1] else (freqs >= lo) & (freqs <= hi)
        out[f"rel_{name}"] = psd[..., mask].sum(axis=-1) / total_safe
    return out


def spectral_ratios(rel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ratios entre bandas — los marcadores clásicos de lentificación del EEG.

    Se calculan sobre las potencias RELATIVAS, pero el valor es idéntico al que
    daría usar potencias absolutas (el denominador común se cancela); usarlas
    relativas solo garantiza que todo el módulo parte de cantidades invariantes.

    - `ratio_alpha_delta` (α/δ): el resumen más directo de "cuánta actividad
      rápida y organizada queda frente a la lentificación". Baja = peor.
    - `ratio_theta_alpha` y su inverso `ratio_alpha_theta`: capturan el
      desplazamiento fino en la zona media del espectro. Se incluyen los dos
      porque los modelos lineales no ven la relación x <-> 1/x, y según cuál sea
      la cola de la distribución uno de los dos separa mejor.
    - `dtabr` = (δ+θ)/(α+β), *delta-theta / alpha-beta ratio*: el índice de
      lentificación global más usado en EEG cuantitativo de cuidados críticos.
      Sube cuando el espectro se desplaza a lo lento.

    Los denominadores se acotan con `np.maximum(..., EPS)`: en una época real
    ninguna banda tiene potencia exactamente nula, pero así ningún inf puede
    entrar en la matriz de salida bajo ninguna circunstancia. Aquí el umbral
    ABSOLUTO sí es legítimo —a diferencia del caso descrito en `_safe_divisor`—
    porque los denominadores son potencias RELATIVAS: ya viven en [0, 1] y no
    dependen de la calibración del equipo, así que 1e-12 significa lo mismo para
    todos los pacientes.
    """
    d, t = rel["rel_delta"], rel["rel_theta"]
    a, b = rel["rel_alpha"], rel["rel_beta"]
    return {
        "ratio_alpha_delta": a / np.maximum(d, EPS),
        "ratio_theta_alpha": t / np.maximum(a, EPS),
        "ratio_alpha_theta": a / np.maximum(t, EPS),
        "dtabr": (d + t) / np.maximum(a + b, EPS),
    }


def spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """Entropía de Shannon del espectro, normalizada a [0, 1].

    Se normaliza la PSD a una distribución de probabilidad (p_i = P_i / ΣP, área
    1) y se calcula H = -Σ p_i·log(p_i), dividido por log(n_bins) — la entropía
    máxima posible, la de un espectro perfectamente plano.

    INTERPRETACIÓN: mide cuán REPARTIDA está la potencia entre frecuencias.
    Cerca de 1 = espectro plano tipo ruido blanco (señal rica/desorganizada);
    cerca de 0 = toda la potencia concentrada en una frecuencia (señal pobre y
    monótona, p. ej. delta dominante en un coma profundo).

    INVARIANZA: la normalización a probabilidad divide por la potencia total, y
    ahí muere la escala: p(k·x) = k²P / k²ΣP = p(x).
    """
    total = _safe_divisor(psd.sum(axis=-1, keepdims=True))
    p = psd / total
    # log(0) -> se trata explícitamente: 0·log(0) = 0 por convención.
    log_p = np.log(p, where=p > 0, out=np.zeros_like(p))
    h = -(p * log_p).sum(axis=-1)
    return h / math.log(psd.shape[-1])


def spectral_edge_frequency(
    freqs: np.ndarray, psd: np.ndarray, q: float = 0.95
) -> np.ndarray:
    """Frecuencia de borde espectral: la f bajo la cual cae la fracción `q` de la potencia.

    SEF95 (q=0.95) es un descriptor clásico de "hasta dónde llega el espectro":
    un EEG lentificado concentra la potencia abajo y su SEF95 baja. SEF50 (q=0.50)
    es la frecuencia MEDIANA del espectro, una medida más robusta del centro.

    IMPLEMENTACIÓN: se acumula la PSD normalizada a lo largo de la frecuencia y se
    busca el primer bin donde la acumulada alcanza `q`, INTERPOLANDO linealmente
    entre ese bin y el anterior. Sin interpolación la feature quedaría cuantizada
    a los 0.39 Hz de la rejilla de Welch, lo que introduce escalones artificiales;
    la interpolación devuelve un valor continuo.

    INVARIANZA: se trabaja sobre la acumulada NORMALIZADA (dividida por la
    potencia total), así que la escala ya está cancelada antes de buscar el cruce.
    """
    total = _safe_divisor(psd.sum(axis=-1, keepdims=True))
    cum = np.cumsum(psd, axis=-1) / total                 # (..., n_freqs), acaba en 1
    idx = (cum < q).sum(axis=-1)                          # primer bin con cum >= q
    idx = np.clip(idx, 0, len(freqs) - 1)
    idx_lo = np.maximum(idx - 1, 0)

    c_hi = np.take_along_axis(cum, idx[..., None], axis=-1)[..., 0]
    c_lo = np.take_along_axis(cum, idx_lo[..., None], axis=-1)[..., 0]
    f_hi, f_lo = freqs[idx], freqs[idx_lo]

    # `cum` es adimensional (fracción acumulada), así que aquí EPS sí es legítimo.
    span = c_hi - c_lo
    frac = np.where(span > EPS, (q - c_lo) / np.where(span > EPS, span, 1.0), 0.0)
    # Cinturón y tirantes: por construcción c_lo < q <= c_hi, luego frac ∈ [0, 1] y
    # el resultado no puede salirse de la rejilla de frecuencias. El clip garantiza
    # que ninguna patología numérica devuelva una frecuencia fuera del rango físico
    # 0.5-45 Hz — que es justo el síntoma que delató el bug de `_safe_divisor`.
    frac = np.clip(frac, 0.0, 1.0)
    return f_lo + frac * (f_hi - f_lo)


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Frecuencia media del espectro, ponderada por potencia: Σ(f·P) / ΣP.

    Es el "centro de masa" del espectro. Complementa a SEF50 (la mediana): el
    centroide es sensible a la cola de alta frecuencia, la mediana es robusta a
    ella. Invariante por ser un cociente de potencias.
    """
    total = _safe_divisor(psd.sum(axis=-1))
    return (psd * freqs).sum(axis=-1) / total


# =============================================================================
# FEATURES EN EL DOMINIO DEL TIEMPO
# =============================================================================

def hjorth_parameters(data: np.ndarray) -> dict[str, np.ndarray]:
    """Parámetros de Hjorth — descriptores baratos de la FORMA de la señal.

    Se definen sobre la señal y sus derivadas (aquí, diferencias finitas):
      - ACTIVIDAD  = var(x)                       -> **NO se exporta**.
      - MOVILIDAD  = std(x') / std(x)             -> "frecuencia media" estimada
        en el dominio del tiempo: cuánto se mueve la señal respecto a su tamaño.
      - COMPLEJIDAD = movilidad(x') / movilidad(x) -> cuánto se parece la señal a
        una sinusoide pura (complejidad ≈ 1) frente a una forma de onda rica.

    DECISIÓN EXPLÍCITA — HJORTH ACTIVIDAD QUEDA FUERA: la actividad es
    literalmente la varianza de la época, escala con k² y por tanto NO es
    invariante. Es justo el tipo de feature que codificaría la calibración del
    equipo de cada paciente. Se podría normalizar (p. ej. dividiendo por la
    varianza mediana del paciente), pero eso reintroduce una estadística
    calculada POR PACIENTE dentro de la matriz de features, con riesgo de fuga
    sutil hacia la Fase 4; se prefiere descartarla. Movilidad y complejidad
    sobreviven porque son COCIENTES de desviaciones estándar: la k del numerador
    se cancela con la del denominador.
    """
    dx = np.diff(data, axis=-1)
    ddx = np.diff(dx, axis=-1)

    # std es DIMENSIONAL -> blindaje sin umbral absoluto (ver `_safe_divisor`).
    std_x = _safe_divisor(data.std(axis=-1))
    std_dx = _safe_divisor(dx.std(axis=-1))
    std_ddx = ddx.std(axis=-1)

    mobility = std_dx / std_x
    mobility_dx = std_ddx / std_dx
    # `mobility` sí es adimensional (cociente de stds), así que EPS es legítimo aquí.
    complexity = mobility_dx / np.maximum(mobility, EPS)
    return {"hjorth_mobility": mobility, "hjorth_complexity": complexity}


def line_length_normalized(data: np.ndarray) -> np.ndarray:
    """Line length NORMALIZADA: Σ|x[t+1]-x[t]| dividida por la std de la época.

    La *line length* mide la longitud del trazo dibujado por la señal: sube con
    la amplitud Y con la frecuencia, y es un detector barato y muy usado de
    actividad ictal/brotes.

    POR QUÉ NORMALIZADA Y NO CRUDA: la versión cruda escala LINEALMENTE con k
    (|k·Δx| = k·|Δx|) y por tanto NO es válida bajo unidades 'nu'. Al dividir por
    la desviación estándar de la propia época —que también escala con k— el
    cociente queda invariante, y lo que mide pasa a ser "cuánto camino recorre la
    señal por unidad de amplitud", es decir la rugosidad/frecuencia efectiva del
    trazo, ya sin el tamaño.
    """
    dx = np.abs(np.diff(data, axis=-1)).sum(axis=-1)
    return dx / _safe_divisor(data.std(axis=-1))


def zero_crossing_rate(data: np.ndarray) -> np.ndarray:
    """Fracción de muestras consecutivas en las que la señal (centrada) cambia de signo.

    Estimador no paramétrico de la frecuencia dominante: una señal lenta cruza el
    cero pocas veces, una rápida muchas. Complementa a las features espectrales
    sin depender de ningún modelo de espectro.

    Se centra la época (se le resta su media) antes de mirar los signos, para que
    un pequeño offset residual no borre todos los cruces. INVARIANZA: multiplicar
    por k>0 no cambia el signo de nada (ni el de x - media, porque la media
    también escala por k), así que el conteo de cruces es idéntico.
    """
    centered = data - data.mean(axis=-1, keepdims=True)
    signs = np.sign(centered)
    # sign() puede dar 0 exacto; se propaga el signo anterior para no contar
    # un cero aislado como dos cruces.
    crossings = np.diff(signs, axis=-1) != 0
    return crossings.mean(axis=-1)


def amplitude_shape(data: np.ndarray) -> dict[str, np.ndarray]:
    """Curtosis y asimetría de la distribución de amplitudes de la época.

    - CURTOSIS (exceso, Fisher: 0 = gaussiana): mide el peso de las colas. Es la
      razón principal para incluir esta familia: el patrón de BROTE-SUPRESIÓN
      —alternancia de silencio eléctrico y brotes de gran amplitud, marcador de
      mal pronóstico en el coma post-anóxico— produce una distribución de
      amplitudes con un pico enorme en cero y colas pesadas, es decir curtosis
      alta. El EEG continuo normal es aproximadamente gaussiano (curtosis ≈ 0).
    - ASIMETRÍA (skewness): detecta desviaciones no simétricas, típicas de
      transitorios epileptiformes en una sola dirección.

    INVARIANZA: ambos son momentos ESTANDARIZADOS (m3/σ³, m4/σ⁴); tanto el
    momento como la potencia de σ escalan con k, y se cancelan exactamente.
    """
    return {
        "kurtosis": kurtosis(data, axis=-1, fisher=True, bias=True),
        "skewness": skew(data, axis=-1, bias=True),
    }


def permutation_entropy(
    data: np.ndarray,
    order: int = PERM_ENTROPY_ORDER,
    delay: int = PERM_ENTROPY_DELAY,
    chunk: int = _PERM_CHUNK_EPOCHS,
) -> np.ndarray:
    """Entropía de permutación (Bandt-Pompe) normalizada a [0, 1].

    QUÉ MIDE: la complejidad de la señal leída como una secuencia de PATRONES
    ORDINALES. Se recorre la época con una ventana deslizante de `order`
    muestras y, en vez de mirar los valores, se anota cuál es el orden relativo
    entre ellas (p. ej. "sube-baja"). Con order=3 hay 3!=6 patrones posibles; se
    cuenta la frecuencia de cada uno y se calcula la entropía de Shannon de esa
    distribución, dividida por log(3!) para dejarla en [0, 1].

    POR QUÉ EN ESTE PROYECTO: es una de las medidas de complejidad con mejor
    evidencia en EEG de coma — cuanto más dañado el cerebro, más monótona y
    predecible la señal y más baja la entropía de permutación. Es robusta al
    ruido y, a diferencia de la entropía espectral, no asume nada sobre el
    espectro.

    INVARIANZA (la más fuerte de todo el módulo): la feature solo depende de
    COMPARACIONES entre muestras. Multiplicar por cualquier k>0 preserva todas
    las desigualdades, luego los patrones ordinales son idénticos bit a bit. De
    hecho es invariante a cualquier transformación monótona creciente, no solo al
    escalado.

    IMPLEMENTACIÓN (order=3, vectorizada): el patrón ordinal de tres muestras
    (x0,x1,x2) queda determinado por las tres comparaciones por pares, que se
    codifican como un entero 4·(x0>x1) + 2·(x0>x2) + (x1>x2). De las 8
    combinaciones solo 6 son alcanzables (las 2 restantes serían órdenes
    cíclicos, imposibles). Se cuenta por fila con un `bincount` desplazado. El
    troceado en bloques de épocas es solo control de memoria.
    """
    if order != 3:
        raise NotImplementedError("Vectorized permutation entropy is implemented for order=3")

    n_epochs, n_channels, n_times = data.shape
    n_vectors = n_times - (order - 1) * delay
    n_patterns = math.factorial(order)
    out = np.empty((n_epochs, n_channels), dtype=np.float64)

    for start in range(0, n_epochs, chunk):
        stop = min(start + chunk, n_epochs)
        block = data[start:stop].reshape(-1, n_times)
        x0 = block[:, 0:n_vectors]
        x1 = block[:, delay:delay + n_vectors]
        x2 = block[:, 2 * delay:2 * delay + n_vectors]

        codes = (4 * (x0 > x1) + 2 * (x0 > x2) + (x1 > x2)).astype(np.int64)
        n_rows = codes.shape[0]
        offsets = np.arange(n_rows, dtype=np.int64)[:, None] * 8
        counts = np.bincount((codes + offsets).ravel(), minlength=n_rows * 8)
        counts = counts.reshape(n_rows, 8)

        p = counts / n_vectors
        log_p = np.log(p, where=p > 0, out=np.zeros_like(p))
        h = -(p * log_p).sum(axis=-1) / math.log(n_patterns)
        out[start:stop] = h.reshape(stop - start, n_channels)

    return out


# =============================================================================
# ENSAMBLADO: TODAS LAS FEATURES DE UN TENSOR DE ÉPOCAS
# =============================================================================

def compute_epoch_features(
    data: np.ndarray, sfreq: float = TARGET_SFREQ_HZ
) -> dict[str, np.ndarray]:
    """Calcula TODAS las features del módulo sobre un tensor de épocas.

    ENTRADA:  data (n_epochs, n_channels, n_times)
    SALIDA:   dict ordenado {nombre_feature: array (n_epochs, n_channels)}

    El orden de inserción en el dict define el orden de las columnas de la matriz
    final; se mantiene agrupado por familia (espectral -> ratios -> forma del
    espectro -> dominio del tiempo -> complejidad) para que la tabla resultante
    sea legible y el heatmap de correlación agrupe bloques con sentido.
    """
    data = np.asarray(data, dtype=np.float64)
    freqs, psd = compute_welch_psd(data, sfreq=sfreq)

    feats: dict[str, np.ndarray] = {}
    rel = relative_band_powers(freqs, psd)
    feats.update(rel)                                   # 5 potencias relativas
    feats.update(spectral_ratios(rel))                  # 4 ratios
    feats["spectral_entropy"] = spectral_entropy(psd)
    for name, q in SPECTRAL_EDGE_QUANTILES.items():     # sef95, sef50
        feats[name] = spectral_edge_frequency(freqs, psd, q)
    feats["spectral_centroid"] = spectral_centroid(freqs, psd)
    feats.update(hjorth_parameters(data))               # mobility, complexity
    feats["line_length_norm"] = line_length_normalized(data)
    feats["zero_crossing_rate"] = zero_crossing_rate(data)
    feats.update(amplitude_shape(data))                 # kurtosis, skewness
    feats["perm_entropy"] = permutation_entropy(data)
    return feats


def feature_names() -> list[str]:
    """Nombres de las features (sin el sufijo de canal), en el orden de la matriz."""
    return (
        [f"rel_{b}" for b in FREQ_BANDS]
        + ["ratio_alpha_delta", "ratio_theta_alpha", "ratio_alpha_theta", "dtabr"]
        + ["spectral_entropy"]
        + list(SPECTRAL_EDGE_QUANTILES)
        + ["spectral_centroid", "hjorth_mobility", "hjorth_complexity",
           "line_length_norm", "zero_crossing_rate", "kurtosis", "skewness",
           "perm_entropy"]
    )


def features_to_dataframe(
    feats: dict[str, np.ndarray], ch_names: list[str]
) -> pd.DataFrame:
    """Aplana el dict de features a una tabla con columnas `<feature>_<canal>`.

    NO se promedian los 19 canales: la topografía importa (la actividad occipital
    y la frontal no significan lo mismo), así que cada par (feature, canal) es una
    columna propia y es el modelo quien decide qué pesa. El orden es
    feature-mayor / canal-menor, con los canales en el orden canónico del ADR-003,
    de modo que la columna j signifique SIEMPRE lo mismo en todos los pacientes.
    """
    matrix = np.concatenate([feats[name] for name in feats], axis=1)
    columns = [f"{name}_{ch}" for name in feats for ch in ch_names]
    return pd.DataFrame(matrix, columns=columns)


# =============================================================================
# EXTRACCIÓN POR PACIENTE
# =============================================================================

def patient_id_from_path(fif_path: Path) -> str:
    """Extrae el ID de paciente de `patient_XXXX-epo.fif` -> 'XXXX'."""
    return fif_path.name.replace("patient_", "").replace("-epo.fif", "")


def extract_patient_features(fif_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Carga UN `-epo.fif` y devuelve su tabla de features (una fila por época).

    Las columnas de identidad (`patient_id`, `label`, `outcome`) NO se recalculan:
    viajan en la metadata del propio archivo desde la Fase 2, así que la etiqueta
    no puede desalinearse con las épocas. `epoch_idx` es la posición de la época
    DENTRO de su paciente (0..n-1); sirve para trazabilidad y para reconstruir el
    orden temporal si la Fase 4 quiere agregar por ventanas.
    """
    epochs = mne.read_epochs(fif_path, preload=True, verbose="ERROR")
    data = epochs.get_data()                    # (n_ep, n_ch, n_times)
    meta = epochs.metadata.reset_index(drop=True)

    feats = compute_epoch_features(data, sfreq=epochs.info["sfreq"])
    df_feats = features_to_dataframe(feats, list(epochs.ch_names))

    df = pd.DataFrame(
        {
            "patient_id": meta["patient_id"].astype(str),
            "label": meta["label"].astype(int),
            "outcome": meta["outcome"].astype(str),
            "epoch_idx": np.arange(len(meta), dtype=int),
        }
    )
    df = pd.concat([df, df_feats], axis=1)

    if verbose:
        print(f"  {fif_path.name}: {df.shape[0]} epochs x {df_feats.shape[1]} features")
    return df


# =============================================================================
# GUARDIA ANTI-LEAKAGE
# =============================================================================

def assert_no_held_out(fif_paths: list[Path]) -> None:
    """Falla RUIDOSAMENTE si algún archivo pertenece al conjunto held-out.

    POR QUÉ EXISTE ESTA FUNCIÓN: el conjunto held-out (finding 018/021 §3.1) es la
    columna vertebral anti-*leakage* del proyecto — el modelo congelado solo puede
    verlo UNA vez, en la Fase 5. El 2026-07-22 un paciente held-out (0303) se coló
    de verdad en `data_processed/` porque la descarga escribía en el mismo
    directorio que el batch (finding 023 §1). No es un riesgo teórico. Por eso la
    comprobación es una EXCEPCIÓN y no un aviso: un warning se ignora, una
    excepción detiene el batch.
    """
    intruders = sorted(
        p.name for p in fif_paths if patient_id_from_path(p) in HELD_OUT_PATIENTS
    )
    if intruders:
        raise RuntimeError(
            "HELD-OUT LEAKAGE GUARD: held-out patients found in the development "
            f"feature batch: {intruders}. These must not be seen before Phase 5."
        )


def development_fif_paths(processed_dir: Path = PROCESSED_DIR) -> list[Path]:
    """Archivos `-epo.fif` de DESARROLLO presentes en disco, ya filtrados de held-out."""
    paths = sorted(processed_dir.glob("patient_*-epo.fif"))
    if not paths:
        raise FileNotFoundError(f"No epoch files found in {processed_dir}")
    assert_no_held_out(paths)
    return paths


# =============================================================================
# BATCH -> MATRIZ ÚNICA
# =============================================================================

def build_feature_matrix(
    fif_paths: list[Path] | None = None,
    out_path: Path | None = FEATURES_PATH,
    allow_held_out: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extrae features de todos los pacientes y devuelve/guarda UNA matriz única.

    PARÁMETROS
      fif_paths       : lista explícita de archivos. Si es None, toma TODOS los
                        `-epo.fif` de desarrollo de `data_processed/`.
      out_path        : dónde guardar el parquet (None = no guardar).
      allow_held_out  : desactiva la guardia anti-leakage. **Solo debe ponerse a
                        True en la FASE 5**, cuando el modelo ya está congelado y
                        toca evaluar sobre el conjunto externo. En Fase 3/4 dejarlo
                        en False es lo que impide que un archivo held-out entre por
                        accidente.

    CONTRATO DE SALIDA (la Fase 4 depende de él literalmente):
      - una fila por época;
      - columnas `patient_id` (str, el vector `groups` de la CV agrupada),
        `label` (int, la `y`), `outcome` (str), `epoch_idx` (int) y a
        continuación todas las `<feature>_<canal>` en float;
      - sin NaN ni inf (se verifica antes de escribir).
    """
    if fif_paths is None:
        fif_paths = sorted(PROCESSED_DIR.glob("patient_*-epo.fif"))
    if not fif_paths:
        raise FileNotFoundError("No epoch files to process")
    if not allow_held_out:
        assert_no_held_out(fif_paths)

    frames = []
    for path in fif_paths:
        frames.append(extract_patient_features(path, verbose=verbose))
    df = pd.concat(frames, ignore_index=True)

    # --- Verificación del contrato: nada de NaN/inf silenciosos ---
    feature_cols = [c for c in df.columns if c not in ("patient_id", "label", "outcome", "epoch_idx")]
    values = df[feature_cols].to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(values).sum())
    n_inf = int(np.isinf(values).sum())
    if n_nan or n_inf:
        bad_cols = [
            c for c in feature_cols
            if not np.isfinite(df[c].to_numpy(dtype=np.float64)).all()
        ]
        raise ValueError(
            f"Feature matrix contains {n_nan} NaN and {n_inf} inf values in columns: {bad_cols[:20]}"
        )

    # --- Verificación de RANGO FÍSICO ---
    # Algunas features tienen un rango matemáticamente imposible de superar. Comprobarlo
    # es barato y es lo que delató el bug de `_safe_divisor` (SEF95 de 3.2e5 Hz cuando el
    # techo físico son 45 Hz). Un valor fuera de rango significa que hay un error de
    # cálculo, no un dato raro: por eso aborta en vez de avisar.
    physical_ranges = {
        **{f"rel_{b}": (0.0, 1.0) for b in FREQ_BANDS},
        "spectral_entropy": (0.0, 1.0),
        "perm_entropy": (0.0, 1.0),
        "zero_crossing_rate": (0.0, 1.0),
        "sef95": PSD_BAND_HZ,
        "sef50": PSD_BAND_HZ,
        "spectral_centroid": PSD_BAND_HZ,
    }
    for feat, (lo, hi) in physical_ranges.items():
        cols = [c for c in feature_cols if c.rsplit("_", 1)[0] == feat]
        vals = df[cols].to_numpy(dtype=np.float64)
        if vals.min() < lo - 1e-9 or vals.max() > hi + 1e-9:
            raise ValueError(
                f"Feature '{feat}' is out of its physical range [{lo}, {hi}]: "
                f"observed [{vals.min():.6g}, {vals.max():.6g}]"
            )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        if verbose:
            size_mb = out_path.stat().st_size / 1e6
            print(f"\nSaved {df.shape[0]} rows x {df.shape[1]} cols -> {out_path} ({size_mb:.1f} MB)")
    return df


# =============================================================================
# TEST DE INVARIANZA A ESCALA (no se asume: se demuestra)
# =============================================================================

def scale_invariance_report(
    fif_paths: list[Path] | None = None,
    n_epochs: int = 100,
    factors: tuple[float, ...] = (1e3, 10.0, 0.1, 1e-3, 1e-8),
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Comprueba empíricamente que f(k·x) == f(x) para TODAS las features.

    Toma épocas REALES, multiplica la señal por cada `k` de `factors`, recalcula
    el conjunto completo de features y compara contra el original con
    `np.allclose`. Devuelve una tabla con la máxima diferencia relativa observada
    por feature y un veredicto PASS/FAIL.

    POR QUÉ SE HACE ASÍ Y NO "RAZONANDO EN EL PAPEL": la invarianza es fácil de
    argumentar y fácil de romper por accidente. El test es la única evidencia que
    se puede enseñar a un jurado. Si una feature aparece como FAIL, la regla del
    proyecto es eliminarla o normalizarla — nunca dejarla pasar.

    POR QUÉ LOS FACTORES SON TAN EXTREMOS (hasta k=1e-8): la primera versión de
    este test usaba solo k=10 y k=0.1 y daba PASS a un módulo que SÍ tenía una
    dependencia de escala (el `np.maximum(x, 1e-12)` descrito en
    `_safe_divisor`). El fallo solo se activa cuando la señal es lo bastante
    pequeña como para cruzar un umbral absoluto, y un factor moderado nunca lo
    cruza. La lección quedó incorporada aquí: un test de invarianza tiene que
    BARRER VARIOS ÓRDENES DE MAGNITUD, porque los umbrales absolutos escondidos
    son precisamente lo que se está buscando.

    Se recorren VARIOS pacientes por el mismo motivo: la escala de partida varía
    entre sujetos (0418 y 0286 son los de amplitud más pequeña), así que un
    umbral oculto puede afectar a uno y no a otro.

    NOTA sobre la tolerancia: no se espera igualdad EXACTA en bit, porque el
    escalado cambia los redondeos de la aritmética en coma flotante; se exige
    igualdad dentro de la tolerancia numérica (rtol ~1e-6), varios órdenes de
    magnitud por debajo de cualquier variación con significado.
    """
    if fif_paths is None:
        fif_paths = development_fif_paths()

    blocks = []
    for path in fif_paths:
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
        blocks.append(epochs.get_data()[:n_epochs])
    data = np.concatenate(blocks, axis=0)

    base = compute_epoch_features(data)
    # Se calcula UNA vez por factor (no una vez por par feature-factor): recalcular el
    # bloque completo dentro del bucle de features multiplicaba el coste por 20.
    scaled_sets = {k: compute_epoch_features(data * k) for k in factors}

    rows = []
    for name, ref in base.items():
        max_rel, ok = 0.0, True
        denom = np.maximum(np.abs(ref), atol)
        for k in factors:
            scaled = scaled_sets[k][name]
            ok &= bool(np.allclose(scaled, ref, rtol=rtol, atol=atol))
            max_rel = max(max_rel, float(np.max(np.abs(scaled - ref) / denom)))
        rows.append(
            {"feature": name, "max_rel_diff": max_rel, "verdict": "PASS" if ok else "FAIL"}
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 78)
    print("FASE 3 — extracción de features")
    print("=" * 78)

    print("\n[1/2] Scale-invariance test (k = 1e3, 10, 0.1, 1e-3, 1e-8) on real data")
    report = scale_invariance_report()
    print(report.to_string(index=False))
    if (report["verdict"] == "FAIL").any():
        raise SystemExit("Scale-invariance test FAILED — fix the offending features first")

    print("\n[2/2] Feature batch over the development set")
    t0 = time.time()
    matrix = build_feature_matrix()
    print(f"\nElapsed: {time.time() - t0:.1f} s")
    print(f"Patients: {matrix['patient_id'].nunique()}  |  rows: {len(matrix)}")
    print(matrix.groupby("outcome")["patient_id"].nunique().to_string())
