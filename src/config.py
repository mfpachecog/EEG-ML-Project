"""
Configuración central del proyecto: rutas robustas y constantes canónicas.

Todas las rutas se derivan de la ubicación de ESTE archivo, así el código corre
sin importar desde dónde se invoque ni la ruta absoluta de la máquina. Esto elimina
la clase de bug de las rutas absolutas hardcodeadas (ver finding 019).
"""
from pathlib import Path

# Raíz del proyecto: este archivo está en src/ -> development/ -> eeg-ml-project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Datos crudos I-CARE (descargados aparte; ~65 GB, no versionados)
DATA_DIR = PROJECT_ROOT / "patients_data_raw" / "physionet.org" / "files" / "i-care" / "2.1" / "training"

# Ventana de entrenamiento post-ROSC en horas (ADR-002): el EEG refleja el estado real del cerebro
ROSC_WINDOW_HOURS = (24, 72)

# Frecuencia de re-muestreo común para todos los pacientes (ADR-001)
TARGET_SFREQ_HZ = 100

# 19 canales estándar 10-20 en ORDEN CANÓNICO (ADR-003).
# El orden importa: garantiza que la columna i signifique el MISMO canal en todos los pacientes,
# de modo que los vectores de features queden alineados entre sujetos.
CANONICAL_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Pacientes excluidos del desarrollo por no tener NINGÚN segmento en la ventana 24-72h (ADR-004).
# Los datos crudos se conservan en disco; el pipeline simplemente los salta.
EXCLUDED_PATIENTS = {"0296", "0341", "0342"}


def development_patient_ids() -> list[str]:
    """IDs de pacientes de desarrollo: carpetas numéricas en DATA_DIR menos los excluidos (ADR-004)."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")
    ids = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    return [pid for pid in ids if pid not in EXCLUDED_PATIENTS]
