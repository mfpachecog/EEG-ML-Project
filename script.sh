#!/usr/bin/env bash
#
# setup_project_structure.sh
# Reorganiza development/ en una estructura modular:
#   development/
#   ├── scripts/     <- scripts originales (exploración, inventario, filtrado)
#   ├── src/         <- módulos reutilizables (preprocessing.py, etc.)
#   └── notebooks/   <- notebooks de Jupyter
#
# Uso:  bash setup_project_structure.sh   (ejecutar desde development/)

# ----- Modo seguro de bash -----
# -e         : aborta si cualquier comando falla (no seguir con estado a medias)
# -u         : error si se usa una variable no definida (atrapa typos)
# -o pipefail: si un comando dentro de un pipe falla, todo el pipe falla
set -euo pipefail

echo "==> Directorio actual: $(pwd)"
echo "    (Debe ser la carpeta development/)"
echo

# ----- 1. Crear las carpetas nuevas -----
# mkdir -p : crea la carpeta y NO falla si ya existe
mkdir -p scripts src notebooks
echo "==> Carpetas creadas/aseguradas: scripts/  src/  notebooks/"

# ----- 2. Hacer src/ importable como paquete de Python -----
# Un __init__.py vacío le dice a Python "esto es un paquete",
# lo que habilita:  from src.preprocessing import preprocess_segment
touch src/__init__.py
echo "==> src/__init__.py creado (permite importar desde src/)"

# ----- 3. Mover los scripts originales a scripts/ -----
# Según tu terminal, tus scripts viven en first_pipeline/.
# Si no es así, cambia el valor de SRC_DIR.
SRC_DIR="first_pipeline"

if [ -d "$SRC_DIR" ]; then
  echo "==> Moviendo archivos .py de $SRC_DIR/ hacia scripts/"
  # find ... -print0  +  read -d ''  manejan nombres CON ESPACIOS
  # (tu '03 bandpass filter.py' tiene espacios; esto lo respeta)
  find "$SRC_DIR" -maxdepth 1 -type f -name "*.py" -print0 |
    while IFS= read -r -d '' file; do
      echo "     moviendo: $(basename "$file")"
      mv "$file" scripts/
    done
  # Si first_pipeline/ quedó vacía, se elimina; si no, se conserva
  rmdir "$SRC_DIR" 2>/dev/null \
    && echo "==> $SRC_DIR/ estaba vacía y fue eliminada" \
    || echo "==> $SRC_DIR/ conserva otros archivos; no se elimina"
else
  echo "==> No encontré $SRC_DIR/. Si tus scripts están en otra carpeta,"
  echo "    edita la variable SRC_DIR arriba y vuelve a correr el script."
fi

echo
echo "==> Estructura final:"
if command -v tree >/dev/null 2>&1; then
  tree -L 2 -I '.venv|__pycache__|patients_data_raw'
else
  ls -la
fi

echo
echo "==> Si usas git, registra el cambio:"
echo "    git add -A && git commit -m 'chore: reorganizar estructura (scripts/, src/, notebooks/)'"
