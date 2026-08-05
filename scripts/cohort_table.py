"""Extrae la tabla de cohorte de los 35 pacientes descargados, leyendo el disco.

Por qué existe este módulo: las figuras de la Fase 0 (Etapa 2) necesitan la
caracterizacion clinica y la cobertura temporal de cada paciente. Esos datos ya
estan registrados en el finding 031 §10.2, pero copiarlos a mano rompe la
trazabilidad: si manana cambia un archivo, la figura mentiria en silencio. Por eso
se vuelven a leer del disco en cada ejecucion, y el propio script compara el
resultado contra las cifras publicadas del finding.

No toca el pipeline de la v1: solo lee los .txt de metadatos y los NOMBRES de los
archivos de cabecera .hea. No abre ninguna senal ni recalcula ningun resultado.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Raiz de los datos crudos de PhysioNet. Se resuelve relativa al repo de codigo,
# que vive dentro del proyecto (development/ es hermano de patients_data_raw/).
DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "patients_data_raw"
    / "physionet.org"
    / "files"
    / "i-care"
    / "2.1"
    / "training"
)

# Los tres grupos de la cohorte. Se replican aqui desde src/config.py en lugar de
# importarlos porque este script tambien debe poder correrse suelto; el chequeo de
# consistencia contra config.py se hace en `verify_against_config()`.
EXCLUDED_PATIENTS = {"0296", "0341", "0342"}
HELD_OUT_PATIENTS = {
    "0303", "0306", "0312", "0313", "0316", "0320", "0326", "0328",
    "0349", "0353", "0356", "0358", "0359", "0360", "0364",
}

# La ventana de interes clinico, en horas desde el paro cardiaco (ADR-004).
WINDOW_START_H = 24
WINDOW_END_H = 72

# Campos del .txt de cada paciente que interesan a las figuras.
_TXT_FIELDS = ("Hospital", "Age", "Sex", "OHCA", "Shockable Rhythm", "TTM", "Outcome", "CPC")

# Patron de nombre de segmento: PID_NNN_HHH_EEG. El tercer campo es la HORA
# ABSOLUTA desde el paro, no el orden de grabacion (verificado en el finding 031 §8.1
# contra el campo "Start time" de la cabecera).
_SEGMENT_RE = re.compile(r"^(\d{4})_(\d+)_(\d+)_EEG\.hea$")


def _parse_patient_txt(txt_path: Path) -> dict[str, str]:
    """Lee el .txt de metadatos de un paciente a un diccionario campo -> valor."""
    fields: dict[str, str] = {}
    for line in txt_path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def _segment_hours(patient_dir: Path) -> list[int]:
    """Devuelve las horas absolutas de todos los segmentos EEG de un paciente.

    Solo se miran los archivos `_EEG` porque son los unicos que el pipeline lee
    (preprocessing.py:161). Los `_ECG` y `_OTHER` quedan fuera a proposito.
    """
    hours: list[int] = []
    for hea in patient_dir.glob("*_EEG.hea"):
        match = _SEGMENT_RE.match(hea.name)
        if match:
            hours.append(int(match.group(3)))
    return sorted(hours)


def _group_of(patient_id: str) -> str:
    """Asigna cada paciente a su conjunto: excluido, prueba externa o desarrollo."""
    if patient_id in EXCLUDED_PATIENTS:
        return "excluido"
    if patient_id in HELD_OUT_PATIENTS:
        return "held-out"
    return "desarrollo"


def build_cohort() -> pd.DataFrame:
    """Construye la tabla completa de la cohorte descargada, un renglon por paciente."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"raw data directory not found: {DATA_DIR}")

    rows = []
    for patient_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit()):
        patient_id = patient_dir.name
        txt_path = patient_dir / f"{patient_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"metadata file missing: {txt_path}")

        meta = _parse_patient_txt(txt_path)
        hours = _segment_hours(patient_dir)
        in_window = [h for h in hours if WINDOW_START_H <= h <= WINDOW_END_H]

        rows.append(
            {
                "patient": patient_id,
                "group": _group_of(patient_id),
                "hospital": meta.get("Hospital", ""),
                # La edad viene como texto y puede ser 'nan' en algun paciente.
                "age": pd.to_numeric(meta.get("Age"), errors="coerce"),
                "sex": meta.get("Sex", ""),
                "ohca": meta.get("OHCA", ""),
                "shockable": meta.get("Shockable Rhythm", ""),
                "ttm": meta.get("TTM", ""),
                "outcome": meta.get("Outcome", ""),
                "cpc": pd.to_numeric(meta.get("CPC"), errors="coerce"),
                "n_segments": len(hours),
                "h_min": min(hours) if hours else pd.NA,
                "h_max": max(hours) if hours else pd.NA,
                "n_segments_in_window": len(in_window),
            }
        )

    return pd.DataFrame(rows)


# Cifras publicadas en el finding 031 §10.2, usadas como control de regresion: si el
# disco deja de reproducirlas, el script debe gritar en vez de dibujar una figura falsa.
_FINDING_031_SPOT_CHECKS = {
    # paciente: (grupo, cpc, n_segmentos, h_min, h_max)
    "0284": ("desarrollo", 1, 85, 4, 74),
    "0311": ("desarrollo", 2, 18, 46, 63),
    "0348": ("desarrollo", 5, 170, 5, 169),
    "0356": ("held-out", 5, 0, None, None),
    "0296": ("excluido", 1, 6, 2, 6),
    "0525": ("desarrollo", 3, 88, 3, 73),
}


def verify_against_finding_031(cohort: pd.DataFrame) -> list[str]:
    """Compara la tabla recien leida contra las cifras publicadas del finding 031.

    Devuelve la lista de discrepancias encontradas (vacia si todo cuadra).
    """
    problems: list[str] = []

    # 1. Los tres conjuntos deben tener el tamano registrado.
    sizes = cohort["group"].value_counts().to_dict()
    for group, expected in (("desarrollo", 17), ("held-out", 15), ("excluido", 3)):
        if sizes.get(group, 0) != expected:
            problems.append(f"group '{group}': expected {expected} patients, found {sizes.get(group, 0)}")

    # 2. Good <-> CPC 1-2 y Poor <-> CPC 3-5, sin una sola excepcion (finding 031 §4.1).
    good_bad = cohort[(cohort["outcome"] == "Good") & (cohort["cpc"] > 2)]
    poor_bad = cohort[(cohort["outcome"] == "Poor") & (cohort["cpc"] <= 2)]
    if len(good_bad) or len(poor_bad):
        problems.append(f"Outcome/CPC mismatch in {len(good_bad) + len(poor_bad)} patients")

    # 3. Muestreo puntual de renglones concretos.
    indexed = cohort.set_index("patient")
    for patient_id, (group, cpc, n_seg, h_min, h_max) in _FINDING_031_SPOT_CHECKS.items():
        row = indexed.loc[patient_id]
        observed = (
            row["group"],
            int(row["cpc"]),
            int(row["n_segments"]),
            None if pd.isna(row["h_min"]) else int(row["h_min"]),
            None if pd.isna(row["h_max"]) else int(row["h_max"]),
        )
        if observed != (group, cpc, n_seg, h_min, h_max):
            problems.append(
                f"patient {patient_id}: finding 031 says {(group, cpc, n_seg, h_min, h_max)}, disk says {observed}"
            )

    return problems


if __name__ == "__main__":
    cohort = build_cohort()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 40)
    print(cohort.to_string(index=False))

    print("\n--- cross-check against finding 031 ---")
    problems = verify_against_finding_031(cohort)
    if problems:
        for problem in problems:
            print(f"  MISMATCH: {problem}")
        raise SystemExit(1)
    print("  OK: disk reproduces every published figure checked.")
