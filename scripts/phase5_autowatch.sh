#!/usr/bin/env bash
# =============================================================================
# VIGILANTE AUTONOMO DE LA FASE 5
# =============================================================================
# POR QUE EXISTE: el usuario pidio "si y solo si la descarga del held-out
# finaliza, ejecuta la Fase 5". La descarga tarda ~40 h a ~55 KB/s, mucho mas
# que cualquier sesion. Este script desacopla el disparador del ciclo de vida de
# la sesion: se lanza con `setsid`, SOBREVIVE al cierre de la terminal, y ejecuta
# la Fase 5 EL solo cuando la descarga esta REALMENTE completa.
#
# POR QUE ES SEGURO DISPARAR LA FASE 5 SIN HUMANO DELANTE: la "unica mirada" al
# held-out es una DISCIPLINA metodologica (no iterar sobre el modelo tras ver el
# resultado), no una irreversibilidad tecnica. `evaluation.py` es determinista y
# re-ejecutable: no modifica el modelo congelado ni los datos, solo predice y
# escribe. Ejecutarlo N veces da el mismo numero. Lo que este script protege es
# que NO se dispare sobre datos incompletos o corruptos.
#
# LAS TRES GUARDIAS (si alguna falla, NO ejecuta y deja diagnostico):
#   1. el proceso de descarga ya no corre (pgrep vacio)
#   2. los 15 pacientes held-out estan listos (txt + >= 4 segmentos completos)
#   3. hay pacientes de AMBAS clases (si no, el AUC no esta ni definido)
#
# USO (desde development/):
#   setsid ./scripts/phase5_autowatch.sh >> scripts/phase5_autowatch.log 2>&1 < /dev/null &
#
# REANUDABLE / IDEMPOTENTE: si la Fase 5 ya se ejecuto (existe el JSON de
# resultados), sale sin hacer nada. Se puede relanzar sin riesgo.
# =============================================================================
set -u

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1   # -> development/

LOG_PREFIX="[phase5-autowatch]"
RESULTS_JSON="data_processed/held_out/held_out_results.json"
POLL_SECONDS=1800          # revisa cada 30 min (la descarga es lentisima)
MAX_HOURS=72               # se rinde tras 3 dias para no quedar colgado eterno

log() { echo "$LOG_PREFIX $(date -Is) :: $*"; }

log "watcher started. polling every ${POLL_SECONDS}s, giving up after ${MAX_HOURS}h."

# Si la Fase 5 ya corrio, no repetir.
if [ -f "$RESULTS_JSON" ]; then
    log "results already exist ($RESULTS_JSON). nothing to do. exiting."
    exit 0
fi

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))

while :; do
    # --- Guardia 1: ¿sigue corriendo la descarga? ---------------------------
    if pgrep -f script_holdout_light > /dev/null; then
        log "download still running; waiting."
        sleep "$POLL_SECONDS"
        [ "$(date +%s)" -ge "$deadline" ] && { log "GAVE UP: deadline reached with download still running."; exit 2; }
        continue
    fi

    log "download process is no longer running. checking readiness..."

    # --- Guardias 2 y 3: readiness y balance de clases ----------------------
    # Se delega en el propio evaluation.py (unica fuente de verdad del criterio
    # de 'ready'), que imprime la tabla de estado. Contamos ready y clases.
    status_out="$(uv run python src/evaluation.py --check 2>/dev/null)"
    echo "$status_out"

    n_ready="$(echo "$status_out" | sed -n 's/.*STATUS (\([0-9]*\)\/15 ready).*/\1/p')"
    n_ready="${n_ready:-0}"

    if [ "$n_ready" -lt 15 ]; then
        log "download stopped but only ${n_ready}/15 patients ready -> INCOMPLETE."
        log "NOT running Phase 5. Plan B applies (report nested CV). Relaunch the"
        log "download to resume:  cd patients_data_raw && setsid ./script_holdout_light.sh >> holdout_light.log 2>&1 < /dev/null &"
        exit 3
    fi

    # Balance de clases entre los ready (la fila 'ready balance: X good / Y poor')
    n_good="$(echo "$status_out" | sed -n 's/.*ready balance: \([0-9]*\) good.*/\1/p')"
    n_poor="$(echo "$status_out" | sed -n 's/.*good \/ \([0-9]*\) poor.*/\1/p')"
    n_good="${n_good:-0}"; n_poor="${n_poor:-0}"

    if [ "$n_good" -lt 1 ] || [ "$n_poor" -lt 1 ]; then
        log "15 ready but single-class (${n_good} good / ${n_poor} poor) -> AUC undefined. NOT running Phase 5."
        exit 4
    fi

    log "ALL GUARDS PASSED: 15/15 ready, both classes (${n_good} good / ${n_poor} poor)."
    break
done

# =============================================================================
# DISPARO DE LA FASE 5
# =============================================================================
log "===== RUNNING PHASE 5 (single deterministic pass) ====="

# Auto-verificacion primero: la cadena con datos sinteticos antes de tocar nada real.
log "step 0/4: self-check on synthetic data"
uv run python src/evaluation.py --self-check || { log "SELF-CHECK FAILED -> aborting before touching held-out."; exit 5; }

log "step 1/4: preprocess held-out (Phase 2 pipeline)"
uv run python src/evaluation.py --preprocess || { log "PREPROCESS FAILED"; exit 6; }

log "step 2/4: extract held-out features (Phase 3)"
uv run python src/evaluation.py --features || { log "FEATURES FAILED"; exit 7; }

log "step 3/4: evaluate frozen model on held-out (Phase 5)"
uv run python src/evaluation.py --evaluate || { log "EVALUATE FAILED"; exit 8; }

log "step 4/4: done. results in $RESULTS_JSON and data_processed/held_out/"
log "===== PHASE 5 COMPLETE ====="
log "REMINDER: the held-out has now been seen. Do NOT tweak the model, features"
log "or threshold in response to this number -- that would break the single-look"
log "discipline. Report it as-is, compared against the LOPO null (0.388), not 0.5."
exit 0
