#!/usr/bin/env bash
set -uo pipefail

# Resumable RunPod supervisor for the generation-only Hunyuan3D batches.
# Both canonical_batch.py and production_batches.py persist their state before
# returning, so restarting them never regenerates assets already marked success.

control_root="${CONTROL_ROOT:-/workspace/control}"
work_root="${WORK_ROOT:-/workspace/production-generation-only}"
outgoing_root="${OUTGOING_ROOT:-/workspace/outgoing-generation-only}"
comfy_root="${COMFY_ROOT:-/opt/ComfyUI}"
python_bin="${PYTHON_BIN:-/opt/comfy-venv/bin/python}"
workflow="${WORKFLOW:-/workspace/workflows/hy3d_example_01.user-supplied.api.json}"
manifest="${MANIFEST:-${control_root}/reference-manifest.json}"
extension_site_packages="${EXTENSION_SITE_PACKAGES:-/workspace/.asset4sim-build-cache/py3.12-torch2.8.0+cu128-cuda12.8-arch8.6-2609efa38f6a/site-packages}"
comfy_url="${COMFY_URL:-http://127.0.0.1:8188}"
start_index="${START_INDEX:-102}"
batch_size="${BATCH_SIZE:-20}"
max_no_progress_restarts="${MAX_NO_PROGRESS_RESTARTS:-4}"

supervisor_log="${control_root}/production.supervisor.log"
comfy_log="${control_root}/comfy.supervised.log"
producer_log="${control_root}/production.stdout.log"
supervisor_pid="${control_root}/production.supervisor.pid"
comfy_pid="${control_root}/comfy.supervised.pid"
producer_pid="${control_root}/production.pid"
blocked_marker="${control_root}/PRODUCTION_BLOCKED"
complete_marker="${outgoing_root}/PRODUCTION_COMPLETE.json"

mkdir -p "${control_root}" "${work_root}" "${outgoing_root}"
printf '%s\n' "$$" > "${supervisor_pid}"
rm -f "${blocked_marker}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "${supervisor_log}"
}

comfy_ready() {
  curl -fsS --max-time 5 "${comfy_url}/system_stats" >/dev/null 2>&1
}

start_comfy() {
  if comfy_ready; then
    return 0
  fi

  if pgrep -f '^/opt/comfy-venv/bin/python main.py .*--port 8188' >/dev/null 2>&1; then
    log "ComfyUI process exists but API is not ready; waiting"
  else
    log "starting ComfyUI"
    (
      cd "${comfy_root}" || exit 1
      exec env PYTHONPATH="${extension_site_packages}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" main.py \
        --listen 0.0.0.0 --port 8188 --enable-cors-header \
        --input-directory /workspace/input \
        --output-directory /workspace/output \
        --user-directory /workspace/user \
        --database-url sqlite:////workspace/user/comfyui.db
    ) >> "${comfy_log}" 2>&1 &
    printf '%s\n' "$!" > "${comfy_pid}"
  fi

  for _ in $(seq 1 90); do
    if comfy_ready; then
      log "ComfyUI API ready"
      return 0
    fi
    sleep 2
  done
  log "ComfyUI API did not become ready within 180 seconds"
  return 1
}

success_count() {
  "${python_bin}" -c 'import glob,json; print(sum(sum(1 for value in json.load(open(path)).get("assets", {}).values() if value.get("status") == "success") for path in glob.glob("/workspace/production-generation-only/batches/*/initial-state.json")))' 2>/dev/null || printf '0\n'
}

producer_running() {
  pgrep -f '^/opt/comfy-venv/bin/python /workspace/control/production_batches.py produce' >/dev/null 2>&1
}

wait_for_existing_producer() {
  while producer_running && [[ ! -f "${complete_marker}" ]]; do
    sleep 10
  done
}

run_producer() {
  log "starting resumable producer at manifest index ${start_index}"
  "${python_bin}" "${control_root}/production_batches.py" produce \
    --manifest "${manifest}" \
    --workflow "${workflow}" \
    --comfy-output /workspace/output \
    --comfy-url "${comfy_url}" \
    --work-root "${work_root}" \
    --outgoing-root "${outgoing_root}" \
    --start "${start_index}" \
    --batch-size "${batch_size}" \
    --timeout 7200 \
    --transfer-poll 10 \
    --generation-only >> "${producer_log}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${producer_pid}"
  wait "${pid}"
}

no_progress_restarts=0
previous_successes="$(success_count)"
log "supervisor started with ${previous_successes} successful assets"

while [[ ! -f "${complete_marker}" ]]; do
  if ! start_comfy; then
    no_progress_restarts=$((no_progress_restarts + 1))
  elif producer_running; then
    log "an existing producer is active; monitoring it"
    wait_for_existing_producer
  else
    run_producer
    producer_status=$?
    log "producer exited with status ${producer_status}"
  fi

  [[ -f "${complete_marker}" ]] && break

  current_successes="$(success_count)"
  if (( current_successes > previous_successes )); then
    log "progress advanced from ${previous_successes} to ${current_successes}; restart budget reset"
    previous_successes="${current_successes}"
    no_progress_restarts=0
  else
    no_progress_restarts=$((no_progress_restarts + 1))
    log "no new success before restart (${no_progress_restarts}/${max_no_progress_restarts})"
  fi

  if (( no_progress_restarts >= max_no_progress_restarts )); then
    log "production blocked after ${no_progress_restarts} consecutive no-progress restarts"
    printf '%s\n' "$(date -Is) successes=${current_successes}" > "${blocked_marker}"
    exit 1
  fi

  sleep 5
done

log "production complete"
exit 0
