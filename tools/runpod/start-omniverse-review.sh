#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
WORKSPACE_MOUNT="${FW_OMNI_WORKSPACE_MOUNT:-/workspace}"
VOLUME_ROOT="${FW_OMNI_VOLUME_ROOT:-${WORKSPACE_MOUNT}/fireviewer-omniverse}"
KIT_ROOT="${VOLUME_ROOT}/runtime/kit-app-template"
RELEASE_ROOT="${KIT_ROOT}/_build/linux-x86_64/release"
EDITOR_LAUNCHER="${RELEASE_ROOT}/fireviewer_usd_composer.kit.sh"
BASE_ZONES_CSV="${FW_OMNI_BASE_ZONES:-}"
REVIEW_SCENE="${FW_OMNI_REVIEW_SCENE:-SIM-01}"
VARIANT_SCENES_ROOT="${FW_OMNI_VARIANT_SCENES_ROOT:-${VOLUME_ROOT}/variant-scenes}"
SCENE_ROOT="${VARIANT_SCENES_ROOT}/${REVIEW_SCENE}"
ROOT_USD="${SCENE_ROOT}/build/root.usdc"
BUILD_RECEIPT="${SCENE_ROOT}/build/build-receipt.json"
PENDING_RECEIPT="${SCENE_ROOT}/editor-review-pending.json"
OPENED_RECEIPT="${SCENE_ROOT}/review-opened.json"
REVIEW_SCRIPT="${REPO_ROOT}/tools/open-zone-scene-in-composer.py"
STATE_ROOT="${VOLUME_ROOT}/state/review-services"
LOG_ROOT="${VOLUME_ROOT}/logs/review"
NGINX_TEMPLATE="${SCRIPT_DIR}/nginx-novnc.conf.template"
REVIEW_USER="${FW_OMNI_REVIEW_USER:-fireviewer}"
REVIEW_PASSWORD="${FW_OMNI_REVIEW_PASSWORD:-}"
DISPLAY_NUMBER="${FW_OMNI_DISPLAY_NUMBER:-99}"
export DISPLAY=":${DISPLAY_NUMBER}"

fail() {
    printf 'FireViewer review launch blocked: %s\n' "$*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "missing file: $1"
}

pid_is_live() {
    local pid_file="$1"
    [[ -f "${pid_file}" ]] || return 1
    local pid
    pid="$(<"${pid_file}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

start_service() {
    local name="$1"
    shift
    local pid_file="${STATE_ROOT}/${name}.pid"
    if pid_is_live "${pid_file}"; then
        return
    fi
    "$@" >>"${LOG_ROOT}/${name}.log" 2>&1 &
    local pid=$!
    printf '%s\n' "${pid}" >"${pid_file}"
    sleep 1
    kill -0 "${pid}" 2>/dev/null || fail "${name} exited during startup"
}

[[ "$(uname -s)" == "Linux" ]] || fail "the RunPod review launcher is Linux-only"
[[ -n "${BASE_ZONES_CSV}" ]] \
    || fail "FW_OMNI_BASE_ZONES must name the four accepted base scenes"
IFS=',' read -r -a BASE_ZONES <<<"${BASE_ZONES_CSV}"
[[ "${#BASE_ZONES[@]}" -eq 4 ]] \
    || fail "FW_OMNI_BASE_ZONES must contain exactly four scene identifiers"
declare -A SEEN_BASE_ZONES=()
for BASE_ZONE in "${BASE_ZONES[@]}"; do
    [[ "${BASE_ZONE}" =~ ^Z[0-9][0-9]$ ]] \
        || fail "invalid base scene identifier: ${BASE_ZONE}"
    [[ -z "${SEEN_BASE_ZONES[${BASE_ZONE}]:-}" ]] \
        || fail "FW_OMNI_BASE_ZONES contains a duplicate scene"
    SEEN_BASE_ZONES["${BASE_ZONE}"]=1
done
[[ "${REVIEW_SCENE}" == "SIM-01" ]] \
    || fail "the pre-simulation manual review target must be SIM-01"
[[ "${REVIEW_USER}" =~ ^[A-Za-z0-9._-]{3,32}$ ]] || fail "invalid review user"
(( ${#REVIEW_PASSWORD} >= 16 )) || fail "FW_OMNI_REVIEW_PASSWORD must contain at least 16 characters"
require_file "${EDITOR_LAUNCHER}"
require_file "${ROOT_USD}"
require_file "${BUILD_RECEIPT}"
require_file "${PENDING_RECEIPT}"
require_file "${REVIEW_SCRIPT}"
require_file "${NGINX_TEMPLATE}"

mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"
chmod 700 "${STATE_ROOT}"

HTPASSWD_FILE="${STATE_ROOT}/review.htpasswd"
printf '%s\n' "${REVIEW_PASSWORD}" | htpasswd -i -c "${HTPASSWD_FILE}" "${REVIEW_USER}" >/dev/null
chmod 600 "${HTPASSWD_FILE}"

NGINX_CONFIG="${STATE_ROOT}/nginx.conf"
sed \
    -e "s|__NGINX_PID__|${STATE_ROOT}/nginx.pid|g" \
    -e "s|__NGINX_ERROR_LOG__|${LOG_ROOT}/nginx-error.log|g" \
    -e "s|__NGINX_ACCESS_LOG__|${LOG_ROOT}/nginx-access.log|g" \
    -e "s|__HTPASSWD_FILE__|${HTPASSWD_FILE}|g" \
    "${NGINX_TEMPLATE}" >"${NGINX_CONFIG}"
chmod 600 "${NGINX_CONFIG}"

start_service xvfb Xvfb "${DISPLAY}" -screen 0 1920x1080x24 -nolisten tcp -noreset
start_service fluxbox env DISPLAY="${DISPLAY}" fluxbox
start_service x11vnc x11vnc \
    -display "${DISPLAY}" \
    -localhost \
    -forever \
    -shared \
    -rfbport 5900 \
    -nopw \
    -noxdamage
start_service websockify websockify \
    --web=/usr/share/novnc \
    127.0.0.1:6081 \
    127.0.0.1:5900

if pid_is_live "${STATE_ROOT}/nginx.pid"; then
    nginx -p "${STATE_ROOT}" -c "${NGINX_CONFIG}" -s reload
else
    nginx -p "${STATE_ROOT}" -c "${NGINX_CONFIG}"
fi
sleep 1
ss -ltn | grep -q '127.0.0.1:5900' || fail "VNC is not restricted to loopback"
ss -ltn | grep -q '127.0.0.1:6081' || fail "noVNC websocket is not restricted to loopback"
ss -ltn | grep -q '0.0.0.0:6080' || fail "authenticated HTTP review port is not listening"

export PYTHONPATH="${REPO_ROOT}/src"
export FW_SDG_REVIEW_USD="${ROOT_USD}"
export FW_SDG_REVIEW_OPENED_RECEIPT="${OPENED_RECEIPT}"
export FW_SDG_REVIEW_ZONE="${REVIEW_SCENE}"
export FW_SDG_REVIEW_PENDING_RECEIPT="${PENDING_RECEIPT}"
export FW_SDG_REVIEW_BUILD_RECEIPT="${BUILD_RECEIPT}"
export FW_OMNI_EDITOR_TARGET_FPS="${FW_OMNI_EDITOR_TARGET_FPS:-60}"
export XDG_CACHE_HOME="${VOLUME_ROOT}/cache/xdg"
export XDG_DATA_HOME="${VOLUME_ROOT}/data/xdg"
export OMNI_CONFIG_PATH="${VOLUME_ROOT}/config/omniverse"

ROOT_SHA256="$(sha256sum "${ROOT_USD}" | cut -d' ' -f1)"
EDITOR_BINDING="${STATE_ROOT}/editor-root.sha256"
if pid_is_live "${STATE_ROOT}/editor.pid"; then
    require_file "${EDITOR_BINDING}"
    [[ "$(<"${EDITOR_BINDING}")" == "${ROOT_SHA256}" ]] \
        || fail "a live Editor is bound to another scene root"
else
    printf '%s\n' "${ROOT_SHA256}" >"${EDITOR_BINDING}"
    chmod 600 "${EDITOR_BINDING}"
    start_service editor env \
        DISPLAY="${DISPLAY}" \
        PYTHONPATH="${PYTHONPATH}" \
        FW_SDG_REVIEW_USD="${FW_SDG_REVIEW_USD}" \
        FW_SDG_REVIEW_OPENED_RECEIPT="${FW_SDG_REVIEW_OPENED_RECEIPT}" \
        FW_SDG_REVIEW_ZONE="${FW_SDG_REVIEW_ZONE}" \
        FW_SDG_REVIEW_PENDING_RECEIPT="${FW_SDG_REVIEW_PENDING_RECEIPT}" \
        FW_SDG_REVIEW_BUILD_RECEIPT="${FW_SDG_REVIEW_BUILD_RECEIPT}" \
        FW_OMNI_EDITOR_TARGET_FPS="${FW_OMNI_EDITOR_TARGET_FPS}" \
        XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        XDG_DATA_HOME="${XDG_DATA_HOME}" \
        OMNI_CONFIG_PATH="${OMNI_CONFIG_PATH}" \
        "${EDITOR_LAUNCHER}" \
        --no-ros-env \
        --exec "${REVIEW_SCRIPT}"
fi

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    printf 'Real Omniverse Editor review: https://%s-6080.proxy.runpod.net/vnc.html?autoconnect=1&resize=scale\n' "${RUNPOD_POD_ID}"
else
    printf 'Real Omniverse Editor review is listening on authenticated HTTP port 6080.\n'
fi
printf 'Human review remains pending; no fire simulation was started or accepted.\n'
