#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
PLAYBACK="${SCRIPT_DIR}/fireviewer-usd-composer.playback.toml"
GEOSPATIAL_SPEC="${REPO_ROOT}/config/runpod-geospatial-env.yml"
WORKSPACE_MOUNT="${FW_OMNI_WORKSPACE_MOUNT:-/workspace}"
VOLUME_ROOT="${FW_OMNI_VOLUME_ROOT:-${WORKSPACE_MOUNT}/fireviewer-omniverse}"
STORAGE_MODE="${FW_OMNI_STORAGE_MODE:-ephemeral-nvme}"
RUNTIME_ROOT="${VOLUME_ROOT}/runtime"
MICROMAMBA_RELEASE="2.6.2-1"
MICROMAMBA_VERSION="2.6.2"
MICROMAMBA_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_RELEASE}/micromamba-linux-64"
MICROMAMBA_SHA256="e9683b483df06dbd3fdd8a37f1b6826d7e5caf4e85bf15a0af4fbad3d4ad1a58"
BLACKWELL_MIN_DRIVER_VERSION="570.158.01"
MICROMAMBA_ROOT="${RUNTIME_ROOT}/micromamba-${MICROMAMBA_RELEASE}"
MICROMAMBA_BIN="${MICROMAMBA_ROOT}/bin/micromamba"
GEOSPATIAL_ENV="${RUNTIME_ROOT}/geospatial-pdal-2.10.2"
KIT_ROOT="${RUNTIME_ROOT}/kit-app-template"
RELEASE_ROOT="${KIT_ROOT}/_build/linux-x86_64/release"
EDITOR_LAUNCHER="${RELEASE_ROOT}/fireviewer_usd_composer.kit.sh"
APP_KIT="${RELEASE_ROOT}/apps/fireviewer_usd_composer.kit"
SOURCE_APP_KIT="${KIT_ROOT}/source/apps/fireviewer_usd_composer.kit"
ISAAC_ROOT="${RUNTIME_ROOT}/isaacsim-6.0.1.0"
ISAAC_PYTHON="${ISAAC_ROOT}/bin/python"
OBJAVERSE_VERSION="0.1.7"
OBJAVERSE_WHEEL_SHA256="7396d119efde5794d0e87d3ca03047d0b0585b2a83ea381a8cc2ddc219d6f1a3"
OBJAVERSE_WHEEL_URL="https://files.pythonhosted.org/packages/ff/bd/e639a506581ee27b77cf036530d596ab80674d8aaec316226b4594680506/objaverse-0.1.7-py3-none-any.whl"
OBJAVERSE_TQDM_VERSION="4.67.1"
OBJAVERSE_TQDM_WHEEL_SHA256="26445eca388f82e72884e0d580d5464cd801a3ea01e63e5601bdff9ba6a48de2"
OBJAVERSE_TQDM_WHEEL_URL="https://files.pythonhosted.org/packages/d0/30/dc54f88dd4a2b5dc8a0279bdd7270e735851848b762aeb1c1184ed1f6b14/tqdm-4.67.1-py3-none-any.whl"
OBJAVERSE_CLIENT_ROOT="${RUNTIME_ROOT}/objaverse-client-${OBJAVERSE_VERSION}"
OBJAVERSE_CLIENT_PYTHON="${OBJAVERSE_CLIENT_ROOT}/bin/python"
OBJAVERSE_DOWNLOADER="${SCRIPT_DIR}/download-community-building-assets.py"
OBJAVERSE_CACHE_ROOT="${VOLUME_ROOT}/cache/objaverse-${OBJAVERSE_VERSION}"
STATE_ROOT="${VOLUME_ROOT}/state"
CONTRACT_ROOT="${VOLUME_ROOT}/contracts"
GEOSPATIAL_LOCK="${CONTRACT_ROOT}/runpod-geospatial-linux-64.cep23.txt"
GEOSPATIAL_RECEIPT="${CONTRACT_ROOT}/runpod-geospatial-runtime.json"
PDAL_DRIVER_INVENTORY="${CONTRACT_ROOT}/pdal-drivers.txt"
GDAL_DRIVER_INVENTORY="${CONTRACT_ROOT}/gdal-raster-drivers.txt"
OGR_DRIVER_INVENTORY="${CONTRACT_ROOT}/ogr-vector-drivers.txt"
GPU_DRIVER_INVENTORY="${CONTRACT_ROOT}/gpu-driver.txt"
GEOSPATIAL_SOLVE_MARKER="${STATE_ROOT}/geospatial-solve-in-progress.txt"
ASSET_ROOT="${VOLUME_ROOT}/input"
OFFICIAL_ASSET_MANIFEST="${ASSET_ROOT}/simready-assets-hd-v3.json"
ASSET_MANIFEST="${OFFICIAL_ASSET_MANIFEST}"
COMMUNITY_BUILDING_SOURCE_ROOT="${ASSET_ROOT}/objaverse-buildings"
COMMUNITY_BUILDING_METADATA="${COMMUNITY_BUILDING_SOURCE_ROOT}/metadata.json"
GROUND_BUNDLE_ROOT="${ASSET_ROOT}/ground-pbr-4k"
GROUND_MATERIAL_MANIFEST="${GROUND_BUNDLE_ROOT}/manifest-v3.json"
CAMPAIGN_ASSET_ROOT="${ASSET_ROOT}/campaign-assets"
CAMPAIGN_ASSET_MANIFEST="${CAMPAIGN_ASSET_ROOT}/manifest-v3.json"
CAMPAIGN_ASSET_RECEIPT="${CONTRACT_ROOT}/campaign-assets.json"
ASSET_RECEIPT="${CONTRACT_ROOT}/assets-materialized.json"
ASSET_BUNDLE_RECEIPT="${CONTRACT_ROOT}/asset-bundle-install.json"
ASSET_BUNDLE_NATIVE_LOD_RECEIPT="${CONTRACT_ROOT}/asset-bundle-native-lods.json"
ASSET_BUNDLE_NATIVE_PBR_RECEIPT="${CONTRACT_ROOT}/asset-bundle-native-pbr.json"
ASSET_BUNDLE_ENABLED=0
ASSET_BUNDLE_SHA256=""
ASSET_BUNDLE_ROOT=""
ASSET_BUNDLE_ARCHIVE=""
CURATED_ASSET_MANIFEST=""
MERGED_SOURCE_MANIFEST=""
SOURCE_MERGE_RECEIPT=""
CAMPAIGN_INDEX="${CONTRACT_ROOT}/campaign-index.json"
RUNTIME_PREFLIGHT="${CONTRACT_ROOT}/setup-preflight.json"
KIT_BUILD_STAMP="${STATE_ROOT}/kit-editor-build.json"
ZONE_WORKSPACE="${VOLUME_ROOT}/production"
BASE_ZONES_CSV="${FW_OMNI_BASE_ZONES:-}"
PILOT_ZONE="${FW_OMNI_PILOT_ZONE:-${BASE_ZONES_CSV%%,*}}"
PILOT_ROOT="${ZONE_WORKSPACE}/zone-scenes/${PILOT_ZONE}"
LIDAR_EVIDENCE="${PILOT_ROOT}/lidar-evidence.json"
ROOT_USD="${PILOT_ROOT}/build/${PILOT_ZONE}_root.usdc"
BUILD_RECEIPT="${PILOT_ROOT}/build/build-receipt.json"
SCENE_GATE_RECEIPT="${PILOT_ROOT}/scene-auto-validation.json"
PENDING_RECEIPT="${PILOT_ROOT}/editor-review-pending.json"
COMPOSITION_ROOT="${VOLUME_ROOT}/composition-sources"
VARIANT_PLAN_ROOT="${VOLUME_ROOT}/variant-plan"
VARIANT_PLAN="${VARIANT_PLAN_ROOT}/campaign-plan.json"
VARIANT_SCENES_ROOT="${FW_OMNI_VARIANT_SCENES_ROOT:-${VOLUME_ROOT}/variant-scenes}"
REVIEW_SCENE="${FW_OMNI_REVIEW_SCENE:-SIM-01}"
REVIEW_SCENE_ROOT="${VARIANT_SCENES_ROOT}/${REVIEW_SCENE}"
REVIEW_ROOT_USD="${REVIEW_SCENE_ROOT}/build/root.usdc"
REVIEW_BUILD_RECEIPT="${REVIEW_SCENE_ROOT}/build/build-receipt.json"
REVIEW_SCENE_GATE_RECEIPT="${REVIEW_SCENE_ROOT}/scene-auto-validation.json"
REVIEW_PENDING_RECEIPT="${REVIEW_SCENE_ROOT}/editor-review-pending.json"
KIT_TEMPLATE_COMMIT="483e364a4176f102f2d3c3aaf9f301a103d61d69"
KIT_TEMPLATE_ORIGIN="https://github.com/NVIDIA-Omniverse/kit-app-template.git"
PHASE="${1:-prepare}"

export PYTHONPATH="${REPO_ROOT}/src"
export FW_OMNI_VOLUME_ROOT="${VOLUME_ROOT}"
export FW_SDG_VOLUME_ROOT="${VOLUME_ROOT}/sdg"
export FW_SDG_RUNTIME_ROOT="${ISAAC_ROOT}"
export FW_SDG_SIMREADY_ASSET_MANIFEST="${ASSET_MANIFEST}"
export FW_OMNI_GROUND_BUNDLE_ROOT="${GROUND_BUNDLE_ROOT}"
export FW_OMNI_GROUND_MATERIAL_MANIFEST="${GROUND_MATERIAL_MANIFEST}"
export FW_SDG_PHOTOREAL_ASSETS_REQUIRED=1
export FW_SDG_ZONE_WORKSPACE_ROOT="${ZONE_WORKSPACE}"
export FW_SDG_VARIANT_BASE_ZONES="${BASE_ZONES_CSV}"
export FW_OMNI_VARIANT_SCENES_ROOT="${VARIANT_SCENES_ROOT}"
export FW_OMNI_REVIEW_SCENE="${REVIEW_SCENE}"
export FW_SDG_FOREST_INSTANCE_BUDGET="${FW_SDG_FOREST_INSTANCE_BUDGET:-2500000}"
export FW_SDG_PHOTOREAL_BUILDING_INSTANCE_LIMIT="${FW_SDG_PHOTOREAL_BUILDING_INSTANCE_LIMIT:-20000}"
export MAMBA_ROOT_PREFIX="${VOLUME_ROOT}/cache/micromamba"
export FW_SDG_GEOSPATIAL_ENV="${GEOSPATIAL_ENV}"
export FW_SDG_PDAL_BIN="${GEOSPATIAL_ENV}/bin/pdal"
export FW_SDG_GDALINFO_BIN="${GEOSPATIAL_ENV}/bin/gdalinfo"
export FW_SDG_OGRINFO_BIN="${GEOSPATIAL_ENV}/bin/ogrinfo"
export FW_SDG_LIDAR_EVIDENCE_RECEIPT="${LIDAR_EVIDENCE}"
export XDG_CACHE_HOME="${VOLUME_ROOT}/cache/xdg"
export XDG_DATA_HOME="${VOLUME_ROOT}/data/xdg"
export OMNI_CONFIG_PATH="${VOLUME_ROOT}/config/omniverse"
export PM_PACKAGES_ROOT="${VOLUME_ROOT}/cache/packman"
export UV_CACHE_DIR="${VOLUME_ROOT}/cache/uv"
export PIP_CACHE_DIR="${VOLUME_ROOT}/cache/pip"

fail() {
    printf 'FireViewer RunPod setup blocked: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command is absent: $1"
}

require_file() {
    [[ -f "$1" ]] || fail "required file is absent: $1"
}

kit_overlay_sha256() {
    (
        cd "${KIT_ROOT}"
        {
            printf '%s\0' \
                premake5.lua \
                repo.toml \
                source/rendered_template_metadata.json \
                source/apps/fireviewer_usd_composer.kit
            find source/extensions/fireviewer_usd_composer_setup \
                -type f \
                ! -path '*/__pycache__/*' \
                ! -name '*.pyc' \
                -print0
        } \
            | sort -z \
            | xargs -0 sha256sum \
            | sha256sum \
            | cut -d' ' -f1
    )
}

kit_stamp_matches_overlay() {
    local playback_sha="$1"
    local overlay_sha="$2"
    local source_app_sha="$3"
    python3.12 -c \
        'import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected={"template_commit":sys.argv[2],"playback_sha256":sys.argv[3],"overlay_sha256":sys.argv[4],"source_application_sha256":sys.argv[5]}
raise SystemExit(0 if all(p.get(k)==v for k,v in expected.items()) else 1)' \
        "${KIT_BUILD_STAMP}" \
        "${KIT_TEMPLATE_COMMIT}" \
        "${playback_sha}" \
        "${overlay_sha}" \
        "${source_app_sha}"
}

kit_stamp_state() {
    python3.12 -c \
        'import json,sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("state",""))' \
        "${KIT_BUILD_STAMP}"
}

write_kit_overlay_stamp() {
    local playback_sha="$1"
    local overlay_sha="$2"
    local source_app_sha="$3"
    local temporary="${KIT_BUILD_STAMP}.tmp"
    python3.12 -c \
        'import json,sys
from pathlib import Path
payload={
  "schema_version":1,
  "state":"KIT_OVERLAY_GENERATED",
  "template_origin":sys.argv[2],
  "template_commit":sys.argv[3],
  "playback_sha256":sys.argv[4],
  "overlay_sha256":sys.argv[5],
  "source_application_sha256":sys.argv[6],
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")' \
        "${temporary}" \
        "${KIT_TEMPLATE_ORIGIN}" \
        "${KIT_TEMPLATE_COMMIT}" \
        "${playback_sha}" \
        "${overlay_sha}" \
        "${source_app_sha}"
    mv "${temporary}" "${KIT_BUILD_STAMP}"
}

write_kit_build_stamp() {
    local playback_sha="$1"
    local overlay_sha="$2"
    local source_app_sha="$3"
    local temporary="${KIT_BUILD_STAMP}.tmp"
    python3.12 -c \
        'import json,sys
from pathlib import Path
payload={
  "schema_version":1,
  "state":"KIT_EDITOR_BUILT",
  "template_origin":sys.argv[2],
  "template_commit":sys.argv[3],
  "playback_sha256":sys.argv[4],
  "overlay_sha256":sys.argv[5],
  "source_application_sha256":sys.argv[6],
  "built_application_sha256":sys.argv[7],
  "launcher_sha256":sys.argv[8],
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")' \
        "${temporary}" \
        "${KIT_TEMPLATE_ORIGIN}" \
        "${KIT_TEMPLATE_COMMIT}" \
        "${playback_sha}" \
        "${overlay_sha}" \
        "${source_app_sha}" \
        "$(sha256sum "${APP_KIT}" | cut -d' ' -f1)" \
        "$(sha256sum "${EDITOR_LAUNCHER}" | cut -d' ' -f1)"
    mv "${temporary}" "${KIT_BUILD_STAMP}"
}

ensure_layout() {
    [[ "$(uname -s)" == "Linux" ]] || fail "this setup is Linux-only"
    [[ "${STORAGE_MODE}" == "persistent-volume" || "${STORAGE_MODE}" == "ephemeral-nvme" ]] \
        || fail "FW_OMNI_STORAGE_MODE must be persistent-volume or ephemeral-nvme"
    [[ "${VOLUME_ROOT}" == "${WORKSPACE_MOUNT}/"* ]] \
        || fail "workspace data must stay below ${WORKSPACE_MOUNT}"
    [[ ! -L "${WORKSPACE_MOUNT}" && ! -L "${VOLUME_ROOT}" ]] \
        || fail "workspace roots may not be symlinks"
    mkdir -p \
        "${RUNTIME_ROOT}" \
        "${STATE_ROOT}" \
        "${CONTRACT_ROOT}" \
        "${ASSET_ROOT}" \
        "${ZONE_WORKSPACE}" \
        "${COMPOSITION_ROOT}" \
        "${VOLUME_ROOT}/cache" \
        "${VOLUME_ROOT}/config/omniverse" \
        "${VOLUME_ROOT}/logs"
    exec 9>"${STATE_ROOT}/setup.lock"
    flock -n 9 || fail "another setup process already owns ${STATE_ROOT}/setup.lock"
}

install_system_dependencies() {
    [[ "$(id -u)" -eq 0 ]] || fail "system dependency installation requires the pod root user"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install --yes --no-install-recommends \
        apache2-utils \
        build-essential \
        ca-certificates \
        curl \
        dbus-x11 \
        fluxbox \
        git \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libglu1-mesa \
        libgomp1 \
        libnss3 \
        libsm6 \
        libvulkan1 \
        libx11-6 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxinerama1 \
        libxkbcommon-x11-0 \
        libxrandr2 \
        libxrender1 \
        libxt6 \
        libxxf86vm1 \
        iproute2 \
        nginx \
        novnc \
        openssl \
        python3.12 \
        python3.12-venv \
        unzip \
        vulkan-tools \
        websockify \
        x11vnc \
        xvfb
}

hardware_preflight() {
    local workspace_fstype
    if [[ "${STORAGE_MODE}" == "persistent-volume" ]]; then
        mountpoint -q "${WORKSPACE_MOUNT}" \
            || fail "${WORKSPACE_MOUNT} is not a persistent mount"
    else
        workspace_fstype="$(findmnt --noheadings --target "${WORKSPACE_MOUNT}" --output FSTYPE)"
        case "${workspace_fstype,,}" in
            9p|ceph|cifs|fuse.rclone|fuse.s3fs|nfs|nfs4|smb3)
                fail "ephemeral-nvme mode refuses network-backed workspace storage: ${workspace_fstype}"
                ;;
        esac
    fi
    [[ -w "${VOLUME_ROOT}" ]] || fail "${VOLUME_ROOT} is not writable"
    require_command nvidia-smi
    require_command vulkaninfo
    local gpu_inventory required_gpu minimum_vram minimum_system_ram minimum_storage_gb
    local system_ram_bytes system_ram_mib cgroup_limit_file cgroup_limit cgroup_source
    local storage_total_bytes minimum_storage_bytes
    gpu_inventory="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits)"
    required_gpu="${FW_OMNI_REQUIRED_GPU_NAME:-RTX PRO 6000 Blackwell Server Edition}"
    minimum_vram="${FW_OMNI_MIN_VRAM_MIB:-90000}"
    minimum_system_ram="${FW_OMNI_MIN_SYSTEM_RAM_MIB:-138000}"
    minimum_storage_gb="${FW_OMNI_MIN_STORAGE_GB:-1500}"
    system_ram_bytes=""
    cgroup_source=""
    # /proc/meminfo describes the RunPod host and is never accepted as pod RAM
    # evidence. The smallest finite cgroup v2/v1 limit is authoritative.
    for cgroup_limit_file in \
        /sys/fs/cgroup/memory.max \
        /sys/fs/cgroup/memory/memory.limit_in_bytes; do
        [[ -r "${cgroup_limit_file}" ]] || continue
        cgroup_limit="$(<"${cgroup_limit_file}")"
        if [[ "${cgroup_limit}" =~ ^[0-9]+$ ]] \
            && (( cgroup_limit > 0 )) \
            && { [[ -z "${system_ram_bytes}" ]] \
                || (( cgroup_limit < system_ram_bytes )); }; then
            system_ram_bytes="${cgroup_limit}"
            cgroup_source="${cgroup_limit_file}"
        fi
    done
    [[ "${system_ram_bytes}" =~ ^[0-9]+$ ]] \
        || fail "no finite container cgroup memory limit is available"
    system_ram_mib="$(( system_ram_bytes / 1024 / 1024 ))"
    [[ "${system_ram_mib}" =~ ^[0-9]+$ ]] \
        || fail "could not resolve effective container RAM"
    (( system_ram_mib >= minimum_system_ram )) \
        || fail "system RAM ${system_ram_mib} MiB is below required ${minimum_system_ram} MiB"
    storage_total_bytes="$(
        df -B1 --output=size "${WORKSPACE_MOUNT}" | tail -n 1 | tr -d ' '
    )"
    [[ "${storage_total_bytes}" =~ ^[0-9]+$ ]] \
        || fail "could not measure workspace storage capacity"
    minimum_storage_bytes="$(( minimum_storage_gb * 1000000000 ))"
    if [[ "${STORAGE_MODE}" == "ephemeral-nvme" ]]; then
        (( storage_total_bytes >= minimum_storage_bytes )) \
            || fail "ephemeral NVMe ${storage_total_bytes} bytes is below required ${minimum_storage_bytes} bytes"
    fi
    python3.12 -c \
        'import re,sys
def normalized_name(value):
    value=" ".join(value.casefold().split())
    return value.removeprefix("nvidia ")
required=normalized_name(sys.argv[1])
minimum=int(sys.argv[2])
minimum_driver=tuple(int(part) for part in sys.argv[3].split("."))
rows=[]
for raw in sys.argv[4].splitlines():
    parts=[part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"invalid nvidia-smi inventory row: {raw!r}")
    rows.append((parts[0],parts[1],int(parts[2])))
def version_tuple(value):
    return tuple(int(part) for part in re.findall(r"\d+", value))
matches=[
    row for row in rows
    if normalized_name(row[0]) == required
    and row[2] >= minimum
    and version_tuple(row[1]) >= minimum_driver
]
if not matches:
    summary="; ".join(f"{name} driver={driver} vram={memory} MiB" for name,driver,memory in rows)
    raise SystemExit(
        f"required GPU {sys.argv[1]!r} with >= {minimum} MiB and "
        f"driver >= {sys.argv[3]} is absent: {summary}"
    )' \
        "${required_gpu}" \
        "${minimum_vram}" \
        "${BLACKWELL_MIN_DRIVER_VERSION}" \
        "${gpu_inventory}" \
        || fail "RTX PRO 6000 96 GB hardware contract failed"
    printf '%s\n' "${gpu_inventory}"
    printf 'system_ram_mib=%s\n' "${system_ram_mib}"
    printf 'system_ram_cgroup_source=%s\n' "${cgroup_source}"
    printf 'storage_capacity_bytes=%s\n' "${storage_total_bytes}"
    vulkaninfo --summary >/dev/null
}

validate_cep23_lock() {
    local lock_path="$1"
    local expected_spec_sha="$2"
    python3.12 -c \
        'import re,sys
from pathlib import Path
from urllib.parse import urlparse
path=Path(sys.argv[1])
expected=sys.argv[2]
lines=path.read_text(encoding="utf-8").splitlines()
binding=f"# fireviewer-source-spec-sha256: {expected}"
if binding not in lines:
    raise SystemExit(f"CEP-23 lock is not bound to the current source spec: {path}")
if "# platform: linux-64" not in lines:
    raise SystemExit(f"CEP-23 lock has no linux-64 platform declaration: {path}")
content=[line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
if not content or content[0] != "@EXPLICIT":
    raise SystemExit(f"CEP-23 lock does not begin with @EXPLICIT: {path}")
entries=content[1:]
if not entries:
    raise SystemExit(f"CEP-23 lock contains no package entries: {path}")
for entry in entries:
    parsed=urlparse(entry)
    if parsed.scheme != "https" or parsed.hostname != "conda.anaconda.org":
        raise SystemExit(f"CEP-23 package URL is outside conda-forge: {entry}")
    if parsed.username or parsed.password or parsed.query:
        raise SystemExit(f"CEP-23 package URL contains credentials or query parameters: {entry}")
    if not re.fullmatch(r"/conda-forge/(?:linux-64|noarch)/[^/]+", parsed.path):
        raise SystemExit(f"CEP-23 package URL has an unexpected platform path: {entry}")
    if not re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", parsed.fragment):
        raise SystemExit(f"CEP-23 package URL has no MD5/SHA-256 checksum: {entry}")' \
        "${lock_path}" \
        "${expected_spec_sha}" \
        || fail "invalid CEP-23 geospatial lock"
}

write_cep23_lock() {
    local spec_sha="$1"
    local raw temporary
    raw="$(mktemp "${STATE_ROOT}/geospatial-explicit.XXXXXX")"
    temporary="${GEOSPATIAL_LOCK}.tmp"
    "${MICROMAMBA_BIN}" --no-rc list \
        --prefix "${GEOSPATIAL_ENV}" \
        --explicit \
        --sha256 >"${raw}"
    {
        printf '# FireViewer immutable geospatial environment lock (CEP-23)\n'
        printf '# platform: linux-64\n'
        printf '# fireviewer-source-spec-sha256: %s\n' "${spec_sha}"
        printf '# fireviewer-micromamba-release: %s\n' "${MICROMAMBA_RELEASE}"
        printf '@EXPLICIT\n'
        sed -n '\|^https://|p' "${raw}"
    } >"${temporary}"
    validate_cep23_lock "${temporary}" "${spec_sha}"
    mv "${temporary}" "${GEOSPATIAL_LOCK}"
    rm -f "${raw}"
}

compare_environment_to_lock() {
    local spec_sha="$1"
    local current
    current="$(mktemp "${STATE_ROOT}/geospatial-current.XXXXXX")"
    "${MICROMAMBA_BIN}" --no-rc list \
        --prefix "${GEOSPATIAL_ENV}" \
        --explicit \
        --sha256 >"${current}"
    python3.12 -c \
        'import sys
from pathlib import Path
def entries(path):
    return sorted(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("https://")
    )
expected=entries(sys.argv[1])
current=entries(sys.argv[2])
if expected != current:
    missing=sorted(set(expected)-set(current))
    unexpected=sorted(set(current)-set(expected))
    raise SystemExit(
        "geospatial environment differs from CEP-23 lock; "
        f"missing={missing[:3]} unexpected={unexpected[:3]}"
    )' \
        "${GEOSPATIAL_LOCK}" \
        "${current}" \
        || fail "geospatial environment drifted from its CEP-23 lock"
    rm -f "${current}"
    validate_cep23_lock "${GEOSPATIAL_LOCK}" "${spec_sha}"
}

install_micromamba() {
    local partial actual_version
    mkdir -p "${MICROMAMBA_ROOT}/bin"
    if [[ -f "${MICROMAMBA_BIN}" ]]; then
        [[ "$(sha256sum "${MICROMAMBA_BIN}" | cut -d' ' -f1)" == "${MICROMAMBA_SHA256}" ]] \
            || fail "persisted micromamba binary SHA-256 mismatch"
    else
        partial="${MICROMAMBA_BIN}.partial"
        curl \
            --proto '=https' \
            --tlsv1.2 \
            --fail \
            --location \
            --retry 5 \
            --retry-all-errors \
            --output "${partial}" \
            "${MICROMAMBA_URL}"
        [[ "$(sha256sum "${partial}" | cut -d' ' -f1)" == "${MICROMAMBA_SHA256}" ]] \
            || fail "micromamba ${MICROMAMBA_RELEASE} SHA-256 mismatch"
        chmod 0750 "${partial}"
        mv "${partial}" "${MICROMAMBA_BIN}"
    fi
    actual_version="$("${MICROMAMBA_BIN}" --version)"
    [[ "${actual_version}" == "${MICROMAMBA_VERSION}" ]] \
        || fail "unexpected micromamba version: ${actual_version}"
}

write_geospatial_solve_marker() {
    local spec_sha="$1"
    local temporary="${GEOSPATIAL_SOLVE_MARKER}.tmp"
    {
        printf 'source_spec_sha256=%s\n' "${spec_sha}"
        printf 'micromamba_release=%s\n' "${MICROMAMBA_RELEASE}"
        printf 'micromamba_sha256=%s\n' "${MICROMAMBA_SHA256}"
    } >"${temporary}"
    mv "${temporary}" "${GEOSPATIAL_SOLVE_MARKER}"
}

validate_geospatial_solve_marker() {
    local spec_sha="$1"
    require_file "${GEOSPATIAL_SOLVE_MARKER}"
    grep -Fxq "source_spec_sha256=${spec_sha}" "${GEOSPATIAL_SOLVE_MARKER}" \
        || fail "interrupted geospatial solve marker has a different source spec"
    grep -Fxq "micromamba_release=${MICROMAMBA_RELEASE}" "${GEOSPATIAL_SOLVE_MARKER}" \
        || fail "interrupted geospatial solve marker has a different micromamba release"
    grep -Fxq "micromamba_sha256=${MICROMAMBA_SHA256}" "${GEOSPATIAL_SOLVE_MARKER}" \
        || fail "interrupted geospatial solve marker has a different micromamba binary"
}

write_geospatial_receipt() {
    local spec_sha="$1"
    local temporary="${GEOSPATIAL_RECEIPT}.tmp"
    local versions="${CONTRACT_ROOT}/runpod-geospatial-versions.txt"
    local versions_tmp="${versions}.tmp"
    local pdal_tmp="${PDAL_DRIVER_INVENTORY}.tmp"
    local gdal_tmp="${GDAL_DRIVER_INVENTORY}.tmp"
    local ogr_tmp="${OGR_DRIVER_INVENTORY}.tmp"
    local gpu_tmp="${GPU_DRIVER_INVENTORY}.tmp"

    {
        printf 'micromamba=%s\n' "$("${MICROMAMBA_BIN}" --version)"
        printf 'pdal=%s\n' "$("${FW_SDG_PDAL_BIN}" --version)"
        printf 'gdal=%s\n' "$("${FW_SDG_GDALINFO_BIN}" --version)"
        printf 'proj=%s\n' "$("${GEOSPATIAL_ENV}/bin/cct" --version)"
    } >"${versions_tmp}"
    "${FW_SDG_PDAL_BIN}" --drivers >"${pdal_tmp}" 2>&1
    "${FW_SDG_GDALINFO_BIN}" --formats >"${gdal_tmp}" 2>&1
    "${FW_SDG_OGRINFO_BIN}" --formats >"${ogr_tmp}" 2>&1
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader,nounits >"${gpu_tmp}"

    grep -Fq '2.10.2' "${versions_tmp}" \
        || fail "resolved PDAL version is not 2.10.2"
    grep -Fq 'readers.copc' "${pdal_tmp}" \
        || fail "PDAL readers.copc driver is absent"
    grep -Fq 'readers.las' "${pdal_tmp}" \
        || fail "PDAL readers.las driver is absent"
    grep -Fq 'filters.stats' "${pdal_tmp}" \
        || fail "PDAL filters.stats driver is absent"
    grep -Fq 'filters.hag_dem' "${pdal_tmp}" \
        || fail "PDAL filters.hag_dem driver is absent"
    grep -Fq 'filters.expression' "${pdal_tmp}" \
        || fail "PDAL filters.expression driver is absent"
    grep -Fq 'filters.reprojection' "${pdal_tmp}" \
        || fail "PDAL filters.reprojection driver is absent"
    grep -Fq 'writers.gdal' "${pdal_tmp}" \
        || fail "PDAL writers.gdal driver is absent"
    grep -Fq 'writers.las' "${pdal_tmp}" \
        || fail "PDAL writers.las driver is absent"
    grep -Fq 'GTiff' "${gdal_tmp}" \
        || fail "GDAL GTiff raster driver is absent"
    grep -Fq 'COG' "${gdal_tmp}" \
        || fail "GDAL COG raster driver is absent"
    grep -Fq 'GeoJSON' "${ogr_tmp}" \
        || fail "OGR GeoJSON vector driver is absent"
    grep -Fq 'GPKG' "${ogr_tmp}" \
        || fail "OGR GPKG vector driver is absent"

    mv "${versions_tmp}" "${versions}"
    mv "${pdal_tmp}" "${PDAL_DRIVER_INVENTORY}"
    mv "${gdal_tmp}" "${GDAL_DRIVER_INVENTORY}"
    mv "${ogr_tmp}" "${OGR_DRIVER_INVENTORY}"
    mv "${gpu_tmp}" "${GPU_DRIVER_INVENTORY}"

    python3.12 -c \
        'import hashlib,json,sys
from pathlib import Path
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
payload={
  "schema_version":1,
  "state":"GEOSPATIAL_RUNTIME_LOCKED",
  "platform":"linux-64",
  "micromamba":{
    "release":sys.argv[2],
    "binary_sha256":sys.argv[3],
  },
  "environment":{
    "prefix":sys.argv[4],
    "source_spec_sha256":sys.argv[5],
    "cep23_lock_sha256":digest(sys.argv[6]),
  },
  "evidence":{
    "versions":{"path":sys.argv[7],"sha256":digest(sys.argv[7])},
    "pdal_drivers":{"path":sys.argv[8],"sha256":digest(sys.argv[8])},
    "gdal_raster_drivers":{"path":sys.argv[9],"sha256":digest(sys.argv[9])},
    "ogr_vector_drivers":{"path":sys.argv[10],"sha256":digest(sys.argv[10])},
    "gpu_driver":{"path":sys.argv[11],"sha256":digest(sys.argv[11])},
  },
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")' \
        "${temporary}" \
        "${MICROMAMBA_RELEASE}" \
        "${MICROMAMBA_SHA256}" \
        "${GEOSPATIAL_ENV}" \
        "${spec_sha}" \
        "${GEOSPATIAL_LOCK}" \
        "${versions}" \
        "${PDAL_DRIVER_INVENTORY}" \
        "${GDAL_DRIVER_INVENTORY}" \
        "${OGR_DRIVER_INVENTORY}" \
        "${GPU_DRIVER_INVENTORY}"
    mv "${temporary}" "${GEOSPATIAL_RECEIPT}"
}

ensure_geospatial_runtime() {
    require_file "${GEOSPATIAL_SPEC}"
    install_micromamba
    local spec_sha
    spec_sha="$(sha256sum "${GEOSPATIAL_SPEC}" | cut -d' ' -f1)"
    if [[ -f "${GEOSPATIAL_LOCK}" ]]; then
        validate_cep23_lock "${GEOSPATIAL_LOCK}" "${spec_sha}"
        if [[ ! -d "${GEOSPATIAL_ENV}/conda-meta" ]]; then
            [[ ! -e "${GEOSPATIAL_ENV}" ]] \
                || fail "incomplete geospatial environment target exists: ${GEOSPATIAL_ENV}"
            "${MICROMAMBA_BIN}" --no-rc create \
                --yes \
                --prefix "${GEOSPATIAL_ENV}" \
                --file "${GEOSPATIAL_LOCK}"
        fi
    else
        if [[ -e "${GEOSPATIAL_ENV}" ]]; then
            validate_geospatial_solve_marker "${spec_sha}"
            [[ -d "${GEOSPATIAL_ENV}/conda-meta" ]] \
                || fail "interrupted geospatial environment has no conda metadata"
            "${MICROMAMBA_BIN}" --no-rc install \
                --yes \
                --prefix "${GEOSPATIAL_ENV}" \
                --file "${GEOSPATIAL_SPEC}" \
                --override-channels \
                --channel conda-forge \
                --channel-priority strict
        else
            write_geospatial_solve_marker "${spec_sha}"
            "${MICROMAMBA_BIN}" --no-rc create \
                --yes \
                --prefix "${GEOSPATIAL_ENV}" \
                --file "${GEOSPATIAL_SPEC}" \
                --override-channels \
                --channel conda-forge \
                --channel-priority strict
        fi
        write_cep23_lock "${spec_sha}"
    fi
    require_file "${FW_SDG_PDAL_BIN}"
    require_file "${FW_SDG_GDALINFO_BIN}"
    require_file "${FW_SDG_OGRINFO_BIN}"
    require_file "${GEOSPATIAL_ENV}/bin/projinfo"
    require_file "${GEOSPATIAL_ENV}/bin/cct"
    compare_environment_to_lock "${spec_sha}"
    write_geospatial_receipt "${spec_sha}"
    rm -f "${GEOSPATIAL_SOLVE_MARKER}"
}

accept_nvidia_terms() {
    [[ "${FW_ACCEPT_NVIDIA_EULA:-NO}" == "YES" ]] || fail "set FW_ACCEPT_NVIDIA_EULA=YES after reviewing NVIDIA terms"
    export OMNI_KIT_ACCEPT_EULA=YES
    export ACCEPT_EULA=Y
    export PRIVACY_CONSENT=Y
}

build_editor() {
    accept_nvidia_terms
    require_file "${PLAYBACK}"
    local playback_sha
    playback_sha="$(sha256sum "${PLAYBACK}" | cut -d' ' -f1)"
    if [[ ! -d "${KIT_ROOT}/.git" ]]; then
        [[ ! -e "${KIT_ROOT}" ]] || fail "incomplete non-Git Kit directory exists: ${KIT_ROOT}"
        git clone --filter=blob:none --no-checkout "${KIT_TEMPLATE_ORIGIN}" "${KIT_ROOT}"
        git -C "${KIT_ROOT}" checkout --detach "${KIT_TEMPLATE_COMMIT}"
    fi
    local origin commit
    origin="$(git -C "${KIT_ROOT}" remote get-url origin)"
    commit="$(git -C "${KIT_ROOT}" rev-parse HEAD)"
    [[ "${origin%/}" == "${KIT_TEMPLATE_ORIGIN%/}" ]] || fail "unexpected Kit origin: ${origin}"
    [[ "${commit}" == "${KIT_TEMPLATE_COMMIT}" ]] || fail "unexpected Kit commit: ${commit}"
    # The pinned template frontend does not consume OMNI_KIT_ACCEPT_EULA in
    # non-interactive playback mode. The checked env gate above is therefore
    # materialized as the exact breadcrumb read by TemplateTool._eula_check.
    touch "${KIT_ROOT}/.omniverse_eula_accepted.txt"
    chmod 0640 "${KIT_ROOT}/.omniverse_eula_accepted.txt"
    chmod +x "${KIT_ROOT}/repo.sh"
    local replayed=0
    if [[ ! -f "${SOURCE_APP_KIT}" ]]; then
        [[ -z "$(git -C "${KIT_ROOT}" status --porcelain)" ]] \
            || fail "Kit checkout is dirty before deterministic template replay"
        (
            cd "${KIT_ROOT}"
            ./repo.sh template replay "${PLAYBACK}"
        )
        replayed=1
    elif [[ ! -f "${KIT_BUILD_STAMP}" ]]; then
        fail "generated Kit overlay exists without its reproducibility stamp"
    fi
    require_file "${SOURCE_APP_KIT}"
    require_file "${KIT_ROOT}/source/rendered_template_metadata.json"
    require_file "${KIT_ROOT}/source/extensions/fireviewer_usd_composer_setup/config/extension.toml"
    local overlay_sha source_app_sha
    overlay_sha="$(kit_overlay_sha256)"
    source_app_sha="$(sha256sum "${SOURCE_APP_KIT}" | cut -d' ' -f1)"
    if [[ "${replayed}" -eq 0 ]] && ! kit_stamp_matches_overlay \
        "${playback_sha}" "${overlay_sha}" "${source_app_sha}"; then
        if [[ "$(kit_stamp_state)" != "KIT_OVERLAY_GENERATED" ]] \
            || [[ ! -f "${EDITOR_LAUNCHER}" ]] \
            || [[ ! -f "${APP_KIT}" ]] \
            || ! cmp -s "${SOURCE_APP_KIT}" "${APP_KIT}"; then
            fail "Kit playback or generated overlay drifted from its build stamp"
        fi
    fi
    if [[ "${replayed}" -eq 1 ]]; then
        write_kit_overlay_stamp "${playback_sha}" "${overlay_sha}" "${source_app_sha}"
    fi
    if [[ "${replayed}" -eq 1 || ! -f "${EDITOR_LAUNCHER}" || ! -f "${APP_KIT}" ]]; then
        (
            cd "${KIT_ROOT}"
            ./repo.sh build
        )
    fi
    require_file "${EDITOR_LAUNCHER}"
    require_file "${APP_KIT}"
    cmp -s "${SOURCE_APP_KIT}" "${APP_KIT}" \
        || fail "built application differs from the generated source application"
    # Version-lock generation is an expected deterministic build output in the
    # source application. Bind the final stamp to the post-build overlay.
    overlay_sha="$(kit_overlay_sha256)"
    source_app_sha="$(sha256sum "${SOURCE_APP_KIT}" | cut -d' ' -f1)"
    write_kit_build_stamp "${playback_sha}" "${overlay_sha}" "${source_app_sha}"
}

runtime_preflight() {
    python3.12 -m fireviewer_sdg.omniverse_pod runtime-preflight \
        --workspace-mount "${WORKSPACE_MOUNT}" \
        --volume-root "${VOLUME_ROOT}" \
        --storage-mode "${STORAGE_MODE}" \
        --editor-launcher "${EDITOR_LAUNCHER}" \
        --app-kit "${APP_KIT}" \
        --kit-checkout "${KIT_ROOT}" \
        --template-playback "${PLAYBACK}" \
        --editor-build-stamp "${KIT_BUILD_STAMP}" \
        --output "${RUNTIME_PREFLIGHT}" \
        --minimum-free-gib "${FW_OMNI_MIN_FREE_GIB:-300}" \
        --minimum-storage-gb "${FW_OMNI_MIN_STORAGE_GB:-1500}" \
        --minimum-vram-mib "${FW_OMNI_MIN_VRAM_MIB:-90000}" \
        --minimum-system-ram-mib "${FW_OMNI_MIN_SYSTEM_RAM_MIB:-138000}" \
        --required-gpu-name "${FW_OMNI_REQUIRED_GPU_NAME:-RTX PRO 6000 Blackwell Server Edition}"
}

configure_asset_bundle_contract() {
    local url="${FW_OMNI_ASSET_BUNDLE_URL:-}"
    local expected="${FW_OMNI_ASSET_BUNDLE_SHA256:-}"
    local allowed="${FW_OMNI_ASSET_BUNDLE_ALLOWED_HOSTS:-}"
    local manifest_relative="${FW_OMNI_ASSET_BUNDLE_MANIFEST_RELATIVE:-manifest-v3.json}"
    ASSET_BUNDLE_ENABLED=0
    ASSET_BUNDLE_SHA256=""
    ASSET_BUNDLE_ROOT=""
    ASSET_BUNDLE_ARCHIVE=""
    CURATED_ASSET_MANIFEST=""
    MERGED_SOURCE_MANIFEST=""
    SOURCE_MERGE_RECEIPT=""
    ASSET_MANIFEST="${OFFICIAL_ASSET_MANIFEST}"
    export FW_SDG_SIMREADY_ASSET_MANIFEST="${ASSET_MANIFEST}"
    if [[ -z "${url}" && -z "${expected}" && -z "${allowed}" ]]; then
        return
    fi
    [[ -n "${url}" ]] || fail "FW_OMNI_ASSET_BUNDLE_URL is required when a curated asset bundle is configured"
    [[ "${expected}" =~ ^[0-9a-fA-F]{64}$ ]] \
        || fail "FW_OMNI_ASSET_BUNDLE_SHA256 must be exactly 64 hexadecimal characters"
    [[ -n "${allowed}" ]] \
        || fail "FW_OMNI_ASSET_BUNDLE_ALLOWED_HOSTS is required when a curated asset bundle is configured"
    python3.12 -c \
        'import sys
from pathlib import PurePosixPath
value=sys.argv[1]
path=PurePosixPath(value)
if not value or "\\" in value or path.is_absolute() or ".." in path.parts or path.parts[0] in {"", "."}:
    raise SystemExit("asset bundle manifest must be a safe relative POSIX path")' \
        "${manifest_relative}" \
        || fail "FW_OMNI_ASSET_BUNDLE_MANIFEST_RELATIVE is invalid"
    ASSET_BUNDLE_ENABLED=1
    ASSET_BUNDLE_SHA256="${expected,,}"
    ASSET_BUNDLE_ROOT="${ASSET_ROOT}/asset-bundles/${ASSET_BUNDLE_SHA256}"
    ASSET_BUNDLE_ARCHIVE="${ASSET_ROOT}/downloads/asset-bundle-${ASSET_BUNDLE_SHA256}.archive"
    CURATED_ASSET_MANIFEST="${ASSET_BUNDLE_ROOT}/${manifest_relative}"
    MERGED_SOURCE_MANIFEST="$(dirname -- "${CURATED_ASSET_MANIFEST}")/merged-source-v3.json"
    SOURCE_MERGE_RECEIPT="${CONTRACT_ROOT}/source-assets-merged-${ASSET_BUNDLE_SHA256}.json"
    ASSET_MANIFEST="${CURATED_ASSET_MANIFEST}"
    export FW_SDG_SIMREADY_ASSET_MANIFEST="${ASSET_MANIFEST}"
}

validate_asset_bundle_source() {
    local url="${FW_OMNI_ASSET_BUNDLE_URL}"
    local allowed="${FW_OMNI_ASSET_BUNDLE_ALLOWED_HOSTS}"
    python3.12 -c \
        'import sys
from urllib.parse import urlparse
url=urlparse(sys.argv[1])
allowed={host.strip().casefold() for host in sys.argv[2].split(",") if host.strip()}
host=(url.hostname or "").casefold()
if (
    url.scheme != "https"
    or not host
    or url.username
    or url.password
    or url.fragment
    or url.port not in {None, 443}
):
    raise SystemExit("asset bundle URL must be credential-free HTTPS on port 443")
if host not in allowed:
    raise SystemExit(f"asset bundle host is outside the exact allowlist: {host}")' \
        "${url}" \
        "${allowed}" \
        || fail "curated asset bundle HTTPS/host contract failed"
}

install_curated_asset_bundle() {
    [[ "${ASSET_BUNDLE_ENABLED}" -eq 1 ]] \
        || fail "internal error: curated asset bundle is not configured"
    validate_asset_bundle_source
    mkdir -p "$(dirname -- "${ASSET_BUNDLE_ARCHIVE}")"
    local max_archive_bytes
    max_archive_bytes="$(python3.12 -c \
        'import math,sys
value=float(sys.argv[1])
if not math.isfinite(value) or value <= 0:
    raise SystemExit("asset bundle archive GiB limit must be positive")
print(int(value * 1024**3))' \
        "${FW_OMNI_ASSET_BUNDLE_MAX_ARCHIVE_GIB:-200}")" \
        || fail "FW_OMNI_ASSET_BUNDLE_MAX_ARCHIVE_GIB is invalid"
    if [[ ! -f "${ASSET_BUNDLE_ARCHIVE}" ]] \
        || [[ "$(sha256sum "${ASSET_BUNDLE_ARCHIVE}" | cut -d' ' -f1)" != "${ASSET_BUNDLE_SHA256}" ]]; then
        local partial="${ASSET_BUNDLE_ARCHIVE}.partial"
        python3.12 -c \
            'import math,shutil,sys
from pathlib import Path
partial=Path(sys.argv[1])
maximum=int(sys.argv[2])
reserve_gib=float(sys.argv[3])
if not math.isfinite(reserve_gib) or reserve_gib < 0:
    raise SystemExit("asset bundle free-space reserve must be non-negative")
partial_size=partial.stat().st_size if partial.is_file() else 0
if partial_size > maximum:
    raise SystemExit("partial asset bundle already exceeds its archive limit")
remaining=maximum-partial_size
reserve=int(reserve_gib*1024**3)
free=shutil.disk_usage(partial.parent).free
if free < remaining+reserve:
    raise SystemExit(
        "insufficient workspace space before asset bundle download: "
        f"free={free} required={remaining+reserve}"
    )' \
            "${partial}" \
            "${max_archive_bytes}" \
            "${FW_OMNI_ASSET_BUNDLE_MIN_FREE_AFTER_INSTALL_GIB:-100}" \
            || fail "asset bundle download would consume the workspace reserve"
        curl \
            --proto '=https' \
            --proto-redir '=https' \
            --tlsv1.2 \
            --fail \
            --connect-timeout "${FW_OMNI_ASSET_BUNDLE_CONNECT_TIMEOUT_SECONDS:-30}" \
            --max-time "${FW_OMNI_ASSET_BUNDLE_DOWNLOAD_TIMEOUT_SECONDS:-21600}" \
            --max-filesize "${max_archive_bytes}" \
            --retry 5 \
            --retry-all-errors \
            --continue-at - \
            --output "${partial}" \
            "${FW_OMNI_ASSET_BUNDLE_URL}"
        [[ "$(sha256sum "${partial}" | cut -d' ' -f1)" == "${ASSET_BUNDLE_SHA256}" ]] \
            || fail "curated asset bundle SHA-256 mismatch"
        mv "${partial}" "${ASSET_BUNDLE_ARCHIVE}"
    fi
    "${ISAAC_PYTHON}" -m fireviewer_sdg.asset_bundle \
        --archive "${ASSET_BUNDLE_ARCHIVE}" \
        --sha256 "${ASSET_BUNDLE_SHA256}" \
        --volume-root "${VOLUME_ROOT}" \
        --destination-root "${ASSET_BUNDLE_ROOT}" \
        --manifest-relative "${FW_OMNI_ASSET_BUNDLE_MANIFEST_RELATIVE:-manifest-v3.json}" \
        --receipt "${ASSET_BUNDLE_RECEIPT}" \
        --max-files "${FW_OMNI_ASSET_BUNDLE_MAX_FILES:-100000}" \
        --max-unpacked-gib "${FW_OMNI_ASSET_BUNDLE_MAX_UNPACKED_GIB:-500}" \
        --minimum-free-after-install-gib "${FW_OMNI_ASSET_BUNDLE_MIN_FREE_AFTER_INSTALL_GIB:-100}"
    require_file "${CURATED_ASSET_MANIFEST}"
}

verify_campaign_asset_bundle() {
    require_file "${CAMPAIGN_ASSET_RECEIPT}"
    require_file "${CAMPAIGN_ASSET_ROOT}/.fireviewer-asset-bundle.json"
    cmp -s \
        "${CAMPAIGN_ASSET_RECEIPT}" \
        "${CAMPAIGN_ASSET_ROOT}/.fireviewer-asset-bundle.json" \
        || fail "campaign asset assembly receipt is absent or stale"
    python3.12 -c \
        'import json,sys
from pathlib import Path
from fireviewer_sdg.asset_bundle import verify_native_quality_receipts
print(json.dumps(verify_native_quality_receipts(
    manifest_path=Path(sys.argv[1]),
    volume_root=Path(sys.argv[2]),
    bundle_root=Path(sys.argv[3]),
    native_lod_receipt=Path(sys.argv[4]),
    native_pbr_receipt=Path(sys.argv[5]),
),sort_keys=True))' \
        "${CAMPAIGN_ASSET_MANIFEST}" \
        "${VOLUME_ROOT}" \
        "${CAMPAIGN_ASSET_ROOT}" \
        "${ASSET_BUNDLE_NATIVE_LOD_RECEIPT}" \
        "${ASSET_BUNDLE_NATIVE_PBR_RECEIPT}" \
        || fail "campaign asset bundle native-quality receipts are absent or stale"
}

select_merged_source_manifest() {
    require_file "${MERGED_SOURCE_MANIFEST}"
    ASSET_MANIFEST="${MERGED_SOURCE_MANIFEST}"
    export FW_SDG_SIMREADY_ASSET_MANIFEST="${ASSET_MANIFEST}"
}

source_manifest_merge_resume_ready() {
    python3.12 -m fireviewer_sdg.source_manifest_merge \
        --volume-root "${VOLUME_ROOT}" \
        --curated-manifest "${CURATED_ASSET_MANIFEST}" \
        --official-manifest "${OFFICIAL_ASSET_MANIFEST}" \
        --output-manifest "${MERGED_SOURCE_MANIFEST}" \
        --receipt "${SOURCE_MERGE_RECEIPT}" \
        --curated-bundle-root "${ASSET_BUNDLE_ROOT}" \
        --curated-bundle-sha256 "${ASSET_BUNDLE_SHA256}" \
        --verify-only
}

community_source_merge_resume_ready() {
    python3.12 -m fireviewer_sdg.source_manifest_merge \
        --volume-root "${VOLUME_ROOT}" \
        --curated-manifest "${CURATED_ASSET_MANIFEST}" \
        --official-manifest "${OFFICIAL_ASSET_MANIFEST}" \
        --output-manifest "${MERGED_SOURCE_MANIFEST}" \
        --receipt "${SOURCE_MERGE_RECEIPT}" \
        --curated-bundle-root "${ASSET_BUNDLE_ROOT}" \
        --curated-bundle-sha256 "${ASSET_BUNDLE_SHA256}" \
        --verify-only \
        --require-community
}

merge_curated_and_official_sources() {
    require_file "${CURATED_ASSET_MANIFEST}"
    require_file "${OFFICIAL_ASSET_MANIFEST}"
    python3.12 -m fireviewer_sdg.source_manifest_merge \
        --volume-root "${VOLUME_ROOT}" \
        --curated-manifest "${CURATED_ASSET_MANIFEST}" \
        --official-manifest "${OFFICIAL_ASSET_MANIFEST}" \
        --output-manifest "${MERGED_SOURCE_MANIFEST}" \
        --receipt "${SOURCE_MERGE_RECEIPT}" \
        --curated-bundle-root "${ASSET_BUNDLE_ROOT}" \
        --curated-bundle-sha256 "${ASSET_BUNDLE_SHA256}" \
        || fail "curated and corrected official source-manifest merge failed"
    select_merged_source_manifest
}

provision_official_nvidia_source_manifest() {
    "${ISAAC_PYTHON}" -c \
        'import sys
from pathlib import Path
from fireviewer_sdg.simready_assets import provision_official_nvidia_manifest
volume=Path(sys.argv[1])
manifest=Path(sys.argv[2])
result=provision_official_nvidia_manifest(
    volume_root=volume,
    manifest_path=manifest,
)
community={
    "buildings.agricultural",
    "buildings.industrial",
    "buildings.annex",
}
missing=set(result["missing_environment"])
if missing != community:
    raise SystemExit(
        "corrected official NVIDIA inventory must leave exactly the three "
        f"reviewed community families missing; observed={sorted(missing)}"
    )' \
        "${VOLUME_ROOT}" \
        "${OFFICIAL_ASSET_MANIFEST}" \
        || fail "official NVIDIA source manifest could not satisfy its exact pre-community contract"
    require_file "${OFFICIAL_ASSET_MANIFEST}"
}

require_exact_source_actor_classes() {
    python3.12 -c \
        'import json,sys
from pathlib import Path
from fireviewer_sdg.asset_bundle import REQUIRED_ACTOR_CLASSES
manifest=Path(sys.argv[1])
payload=json.loads(manifest.read_text(encoding="utf-8"))
actors=payload.get("actors") if isinstance(payload,dict) else None
expected=set(REQUIRED_ACTOR_CLASSES)
observed=set(actors) if isinstance(actors,dict) else set()
if observed != expected:
    raise SystemExit(
        "source manifest requires the exact seven reviewed actor classes: "
        f"missing={sorted(expected-observed)}, "
        f"unexpected={sorted(observed-expected)}"
    )
for class_id in REQUIRED_ACTOR_CLASSES:
    entry=actors[class_id]
    if (
        not isinstance(entry,dict)
        or not isinstance(entry.get("family"),str)
        or entry["family"] != f"actors.{class_id}"
    ):
        raise SystemExit(
            f"source actor {class_id} is malformed or belongs to another class"
        )' \
        "${ASSET_MANIFEST}" \
        || fail "exact reviewed actors are absent; configure the locked FW_OMNI_ASSET_BUNDLE_* source"
}

validate_catalog_host() {
    local url="$1"
    local allowed="${FW_OMNI_CATALOG_ALLOWED_HOSTS:-}"
    [[ -n "${allowed}" ]] || fail "FW_OMNI_CATALOG_ALLOWED_HOSTS is required for catalog download"
    local host
    host="$(python3.12 -c 'import sys, urllib.parse; u=urllib.parse.urlparse(sys.argv[1]); assert u.scheme == "https" and u.hostname; print(u.hostname.lower())' "${url}")" \
        || fail "catalog URL must be valid HTTPS"
    case ",${allowed,,}," in
        *",${host},"*) ;;
        *) fail "catalog host is outside FW_OMNI_CATALOG_ALLOWED_HOSTS: ${host}" ;;
    esac
}

ensure_catalog() {
    local configured="${FW_SDG_ZONE_CATALOG_ROOT:-}"
    if [[ -n "${configured}" ]]; then
        CATALOG_ROOT="$(cd -- "${configured}" && pwd -P)"
    else
        local url="${FW_OMNI_CATALOG_ARCHIVE_URL:-}"
        local expected="${FW_OMNI_CATALOG_ARCHIVE_SHA256:-}"
        [[ -n "${url}" ]] || fail "FW_OMNI_CATALOG_ARCHIVE_URL or FW_SDG_ZONE_CATALOG_ROOT is required"
        [[ "${expected}" =~ ^[0-9a-fA-F]{64}$ ]] || fail "FW_OMNI_CATALOG_ARCHIVE_SHA256 must be exact"
        expected="${expected,,}"
        validate_catalog_host "${url}"
        local archive="${VOLUME_ROOT}/catalog/zone-catalog-${expected}.zip"
        local partial="${archive}.partial"
        mkdir -p "${VOLUME_ROOT}/catalog"
        if [[ ! -f "${archive}" ]] || [[ "$(sha256sum "${archive}" | cut -d' ' -f1)" != "${expected}" ]]; then
            curl \
                --proto '=https' \
                --tlsv1.2 \
                --fail \
                --location \
                --retry 5 \
                --retry-all-errors \
                --continue-at - \
                --output "${partial}" \
                "${url}"
            [[ "$(sha256sum "${partial}" | cut -d' ' -f1)" == "${expected}" ]] \
                || fail "catalog archive SHA-256 mismatch"
            mv "${partial}" "${archive}"
        fi
        CATALOG_ROOT="${VOLUME_ROOT}/catalog/20-zones-${expected}"
        if [[ ! -f "${CATALOG_ROOT}/package_manifest.json" ]]; then
            [[ ! -e "${CATALOG_ROOT}" ]] || fail "incomplete catalog target exists: ${CATALOG_ROOT}"
            local staging
            staging="$(mktemp -d "${VOLUME_ROOT}/catalog/catalog-stage.XXXXXX")"
            unzip -q "${archive}" -d "${staging}"
            mapfile -t manifests < <(find "${staging}" -type f -name package_manifest.json -print)
            [[ "${#manifests[@]}" -eq 1 ]] || fail "catalog archive must expose exactly one package_manifest.json"
            mv "$(dirname -- "${manifests[0]}")" "${CATALOG_ROOT}"
        fi
    fi
    [[ "${CATALOG_ROOT}" == "${WORKSPACE_MOUNT}/"* ]] || fail "catalog must stay on the configured workspace"
    export FW_SDG_ZONE_CATALOG_ROOT="${CATALOG_ROOT}"
    python3.12 -c \
        'import sys; from pathlib import Path; from fireviewer_sdg.zone_scenes import validate_catalog; receipt=validate_catalog(Path(sys.argv[1])); print(f"catalog zones={len(receipt[\"zones\"])} rows={receipt[\"inventory\"][\"rows\"]}")' \
        "${CATALOG_ROOT}"
}

ensure_isaac_runtime() {
    accept_nvidia_terms
    python3.12 -c \
        'from fireviewer_sdg.runtime_bootstrap import ensure_runtime; ensure_runtime()'
    require_file "${ISAAC_PYTHON}"
    mapfile -t flow_presets < <(
        find "${ISAAC_ROOT}" \
            -type f \
            -path '*/omni.flowusd-*/data/presets/Fire/Fire.usda' \
            -print
    )
    [[ "${#flow_presets[@]}" -ge 1 ]] || fail "pinned Isaac runtime has no Flow Fire preset"
    export FW_SDG_FLOW_PRESET="${flow_presets[0]}"
}

ensure_objaverse_client() {
    require_file "${OBJAVERSE_DOWNLOADER}"
    [[ ! -L "${OBJAVERSE_CLIENT_ROOT}" ]] \
        || fail "Objaverse client root may not be a symlink"
    [[ ! -e "${OBJAVERSE_CLIENT_ROOT}" || -d "${OBJAVERSE_CLIENT_ROOT}" ]] \
        || fail "Objaverse client root is not a directory"
    if [[ ! -x "${OBJAVERSE_CLIENT_PYTHON}" ]]; then
        python3.12 -m venv "${OBJAVERSE_CLIENT_ROOT}"
    fi
    local installed
    installed="$(
        "${OBJAVERSE_CLIENT_PYTHON}" -c \
            'import importlib.metadata; print(importlib.metadata.version("objaverse"))' \
            2>/dev/null || true
    )"
    if [[ -n "${installed}" && "${installed}" != "${OBJAVERSE_VERSION}" ]]; then
        fail "Objaverse client version drifted: expected ${OBJAVERSE_VERSION}, found ${installed}"
    fi
    if ! "${OBJAVERSE_CLIENT_PYTHON}" -c \
        'import importlib.metadata, objaverse, sys, tqdm
expected=sys.argv[1]
installed=importlib.metadata.version("objaverse")
installed_tqdm=importlib.metadata.version("tqdm")
raise SystemExit(
    0 if installed == expected and installed_tqdm == sys.argv[2] else 1
)' \
        "${OBJAVERSE_VERSION}" \
        "${OBJAVERSE_TQDM_VERSION}" >/dev/null 2>&1; then
        "${OBJAVERSE_CLIENT_PYTHON}" -m pip install \
            --disable-pip-version-check \
            --force-reinstall \
            --no-input \
            --no-deps \
            "${OBJAVERSE_TQDM_WHEEL_URL}#sha256=${OBJAVERSE_TQDM_WHEEL_SHA256}" \
            "${OBJAVERSE_WHEEL_URL}#sha256=${OBJAVERSE_WHEEL_SHA256}"
    fi
    "${OBJAVERSE_CLIENT_PYTHON}" -c \
        'import importlib.metadata, objaverse, sys, tqdm
expected=sys.argv[1]
installed=importlib.metadata.version("objaverse")
installed_tqdm=importlib.metadata.version("tqdm")
assert installed == expected, f"expected objaverse=={expected}, found {installed}"
assert installed_tqdm == sys.argv[2], (
    f"expected tqdm=={sys.argv[2]}, found {installed_tqdm}"
)
assert callable(objaverse.load_annotations)
assert callable(objaverse.load_objects)' \
        "${OBJAVERSE_VERSION}" \
        "${OBJAVERSE_TQDM_VERSION}" \
        || fail "pinned Objaverse client is incomplete"
}

install_community_building_sources() {
    ensure_objaverse_client
    local download_timeout="${FW_OMNI_OBJAVERSE_DOWNLOAD_TIMEOUT_SECONDS:-21600}"
    [[ "${download_timeout}" =~ ^[1-9][0-9]*$ ]] \
        || fail "FW_OMNI_OBJAVERSE_DOWNLOAD_TIMEOUT_SECONDS must be a positive integer"
    command -v timeout >/dev/null 2>&1 \
        || fail "GNU timeout is required for the bounded Objaverse download"
    timeout \
        --signal=TERM \
        --kill-after=60s \
        "${download_timeout}s" \
        "${OBJAVERSE_CLIENT_PYTHON}" "${OBJAVERSE_DOWNLOADER}" \
        --volume-root "${VOLUME_ROOT}" \
        --destination-root "${COMMUNITY_BUILDING_SOURCE_ROOT}" \
        --cache-root "${OBJAVERSE_CACHE_ROOT}" \
        --workers 4 \
        || fail "reviewed Objaverse building download failed"
    require_file "${COMMUNITY_BUILDING_METADATA}"
    "${ISAAC_PYTHON}" -m fireviewer_sdg.community_building_assets \
        --volume-root "${VOLUME_ROOT}" \
        --manifest "${ASSET_MANIFEST}" \
        --source-root "${COMMUNITY_BUILDING_SOURCE_ROOT}" \
        --metadata "${COMMUNITY_BUILDING_METADATA}" \
        || fail "Kit failed to install the reviewed community building assets"
}

assemble_campaign_asset_bundle() {
    require_file "${ASSET_MANIFEST}"
    require_file "${GROUND_MATERIAL_MANIFEST}"
    "${ISAAC_PYTHON}" -m fireviewer_sdg.campaign_asset_bundle \
        --volume-root "${VOLUME_ROOT}" \
        --official-manifest "${ASSET_MANIFEST}" \
        --ground-manifest "${GROUND_MATERIAL_MANIFEST}" \
        --destination-root "${CAMPAIGN_ASSET_ROOT}" \
        --receipt "${CAMPAIGN_ASSET_RECEIPT}" \
        --manifest-name "$(basename -- "${CAMPAIGN_ASSET_MANIFEST}")" \
        || fail "Kit failed to assemble the final HERO/MID/FAR campaign bundle"
    require_file "${CAMPAIGN_ASSET_MANIFEST}"
    require_file "${CAMPAIGN_ASSET_RECEIPT}"
    "${ISAAC_PYTHON}" -c \
        'import json,sys
from pathlib import Path
from fireviewer_sdg.asset_bundle import (
    validate_native_lod_quality,
    validate_native_pbr_quality,
)
manifest=Path(sys.argv[1])
volume=Path(sys.argv[2])
bundle=Path(sys.argv[3])
lod=validate_native_lod_quality(
    manifest_path=manifest,
    volume_root=volume,
    bundle_root=bundle,
    receipt_path=Path(sys.argv[4]),
)
pbr=validate_native_pbr_quality(
    manifest_path=manifest,
    volume_root=volume,
    bundle_root=bundle,
    receipt_path=Path(sys.argv[5]),
)
print(json.dumps(dict(lod=lod["state"],pbr=pbr["state"]),sort_keys=True))' \
        "${CAMPAIGN_ASSET_MANIFEST}" \
        "${VOLUME_ROOT}" \
        "${CAMPAIGN_ASSET_ROOT}" \
        "${ASSET_BUNDLE_NATIVE_LOD_RECEIPT}" \
        "${ASSET_BUNDLE_NATIVE_PBR_RECEIPT}" \
        || fail "native HERO/MID/FAR or PBR campaign validation failed"
    verify_campaign_asset_bundle
}

select_campaign_asset_bundle() {
    require_file "${CAMPAIGN_ASSET_MANIFEST}"
    require_file "${CAMPAIGN_ASSET_RECEIPT}"
    require_file "${ASSET_BUNDLE_NATIVE_LOD_RECEIPT}"
    require_file "${ASSET_BUNDLE_NATIVE_PBR_RECEIPT}"
    verify_campaign_asset_bundle
    ASSET_MANIFEST="${CAMPAIGN_ASSET_MANIFEST}"
    ASSET_BUNDLE_ROOT="${CAMPAIGN_ASSET_ROOT}"
    export FW_SDG_SIMREADY_ASSET_MANIFEST="${ASSET_MANIFEST}"
}

verify_materialized_asset_receipt() {
    require_file "${ASSET_RECEIPT}"
    "${ISAAC_PYTHON}" -c \
        'import json,sys
from pathlib import Path
from fireviewer_sdg.omniverse_pod import validate_materialized_assets
receipt_path=Path(sys.argv[1])
manifest=Path(sys.argv[2])
volume=Path(sys.argv[3])
if receipt_path.is_symlink():
    raise SystemExit("materialized asset receipt may not be a symlink")
stored=json.loads(receipt_path.read_text(encoding="utf-8"))
if not isinstance(stored,dict):
    raise SystemExit("materialized asset receipt must be an object")
current=validate_materialized_assets(
    manifest_path=manifest,
    volume_root=volume,
)
observed=dict(stored)
quality=observed.pop("usd_quality",None)
observed.pop("validated_at",None)
expected=dict(current)
expected.pop("validated_at",None)
if observed != expected:
    raise SystemExit("materialized asset receipt is stale for the campaign manifest")
if (
    not isinstance(quality,dict)
    or quality.get("validator")
        != "fireviewer_native_usd_photoreal_quality_v2"
    or quality.get("validated_assets") != current["asset_count"]
    or quality.get("family_counts") != current["family_counts"]
):
    raise SystemExit("native USD quality receipt is absent or stale")
quality_assets=quality.get("assets")
if (
    not isinstance(quality_assets,list)
    or len(quality_assets) != current["asset_count"]
    or any(not isinstance(item,dict) for item in quality_assets)
):
    raise SystemExit("native USD quality receipt has no asset identities")
expected_identities=[
    (item["role"],item["asset_id"],item["family"])
    for item in current["assets"]
]
observed_identities=[
    (item.get("role"),item.get("asset_id"),item.get("family"))
    for item in quality_assets
]
if observed_identities != expected_identities:
    raise SystemExit("native USD quality receipt is bound to other assets")
print(json.dumps({
    "state":"MATERIALIZED_ASSET_RECEIPT_CURRENT",
    "manifest_sha256":current["manifest_sha256"],
    "asset_content_sha256":current["asset_content_sha256"],
    "asset_count":current["asset_count"],
},sort_keys=True))' \
        "${ASSET_RECEIPT}" \
        "${ASSET_MANIFEST}" \
        "${VOLUME_ROOT}" \
        || fail "final materialized/native asset receipt is absent or stale"
}

bind_campaign_asset_bundle() {
    select_campaign_asset_bundle
    verify_materialized_asset_receipt
}

run_native_terrain_python() {
    local geospatial_site_packages="${GEOSPATIAL_ENV}/lib/python3.12/site-packages"
    [[ -d "${geospatial_site_packages}" ]] \
        || fail "geospatial Python packages are absent: ${geospatial_site_packages}"
    env \
        "PYTHONPATH=${REPO_ROOT}/src:${geospatial_site_packages}" \
        "LD_LIBRARY_PATH=${GEOSPATIAL_ENV}/lib:${LD_LIBRARY_PATH:-}" \
        "${ISAAC_PYTHON}" \
        "$@"
}

verify_native_terrain_runtime() {
    run_native_terrain_python -c \
        'import json
import numpy
from osgeo import gdal, osr
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
print(json.dumps({
    "state":"NATIVE_TERRAIN_RUNTIME_READY",
    "numpy":numpy.__version__,
    "gdal":gdal.VersionInfo("RELEASE_NAME"),
    "pxr_usd":Usd.GetVersion(),
    "spatial_reference":bool(osr.SpatialReference()),
},sort_keys=True))'
}

materialize_assets() {
    configure_asset_bundle_contract
    ensure_isaac_runtime
    if [[ -f "${CAMPAIGN_ASSET_MANIFEST}" ]] \
        && [[ -f "${CAMPAIGN_ASSET_RECEIPT}" ]] \
        && [[ -f "${ASSET_BUNDLE_NATIVE_LOD_RECEIPT}" ]] \
        && [[ -f "${ASSET_BUNDLE_NATIVE_PBR_RECEIPT}" ]] \
        && [[ -f "${ASSET_RECEIPT}" ]] \
        && (bind_campaign_asset_bundle) >/dev/null 2>&1; then
        bind_campaign_asset_bundle
        printf 'Final campaign asset bundle and receipts are current; reuse accepted.\n'
        return 0
    fi
    local community_source_ready=0
    if [[ "${ASSET_BUNDLE_ENABLED}" -eq 1 ]]; then
        if community_source_merge_resume_ready; then
            select_merged_source_manifest
            community_source_ready=1
            printf 'Locked source merge and exact community supplement found; source downloads skipped.\n'
        elif source_manifest_merge_resume_ready; then
            select_merged_source_manifest
            printf 'Locked pre-community source merge found; curated archive reuse skipped.\n'
        else
            install_curated_asset_bundle
            provision_official_nvidia_source_manifest
            merge_curated_and_official_sources
        fi
    else
        provision_official_nvidia_source_manifest
        ASSET_MANIFEST="${OFFICIAL_ASSET_MANIFEST}"
        export FW_SDG_SIMREADY_ASSET_MANIFEST="${ASSET_MANIFEST}"
        require_exact_source_actor_classes
    fi
    if [[ "${community_source_ready}" -eq 0 ]]; then
        install_community_building_sources
        if [[ "${ASSET_BUNDLE_ENABLED}" -eq 1 ]]; then
            community_source_merge_resume_ready \
                || fail "post-install community source merge verification failed"
        fi
    fi
    python3.12 -m fireviewer_sdg.ground_material_bundle \
        --output-root "${GROUND_BUNDLE_ROOT}" \
        --workers "${FW_OMNI_MATERIAL_DOWNLOAD_WORKERS:-8}"
    require_file "${GROUND_MATERIAL_MANIFEST}"
    require_file "${GROUND_BUNDLE_ROOT}/.fireviewer-asset-bundle.json"
    assemble_campaign_asset_bundle
    select_campaign_asset_bundle
    "${ISAAC_PYTHON}" -m fireviewer_sdg.omniverse_pod validate-assets \
        --manifest "${ASSET_MANIFEST}" \
        --volume-root "${VOLUME_ROOT}" \
        --receipt "${ASSET_RECEIPT}" \
        --native-usd-quality \
        || fail "final materialized/native campaign asset validation failed"
    bind_campaign_asset_bundle
}

write_campaign_contract() {
    [[ -n "${BASE_ZONES_CSV}" ]] \
        || fail "FW_OMNI_BASE_ZONES must name exactly four accepted base zones"
    local -a base_zones base_args
    IFS=',' read -r -a base_zones <<<"${BASE_ZONES_CSV}"
    (( ${#base_zones[@]} == 4 )) \
        || fail "FW_OMNI_BASE_ZONES must contain exactly four comma-separated zones"
    [[ -n "${PILOT_ZONE}" ]] \
        || fail "FW_OMNI_PILOT_ZONE or the first base zone is required"
    local zone
    for zone in "${base_zones[@]}"; do
        [[ "${zone}" =~ ^Z[0-9][0-9]$ ]] \
            || fail "invalid base zone identifier: ${zone}"
        base_args+=(--base-zone "${zone}")
    done
    [[ ",${BASE_ZONES_CSV}," == *",${PILOT_ZONE},"* ]] \
        || fail "FW_OMNI_PILOT_ZONE must belong to FW_OMNI_BASE_ZONES"
    bind_campaign_asset_bundle
    python3.12 -m fireviewer_sdg.omniverse_pod campaign-index \
        --catalog-root "${CATALOG_ROOT}" \
        --asset-manifest "${ASSET_MANIFEST}" \
        --volume-root "${VOLUME_ROOT}" \
        --output "${CAMPAIGN_INDEX}" \
        --pilot-zone "${PILOT_ZONE}" \
        "${base_args[@]}"
}

zone_phase() {
    local zone="$1"
    python3.12 -c \
        'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); z=sys.argv[2]; print(json.loads(p.read_text())["zones"][z]["phase"] if p.is_file() else "not_started")' \
        "${ZONE_WORKSPACE}/zone-scenes/production-state.json" \
        "${zone}"
}

run_zone_phase() {
    local python="$1"
    local zone="$2"
    local phase="$3"
    shift 3
    "${python}" -m fireviewer_sdg.zone_scenes \
        --catalog-root "${CATALOG_ROOT}" \
        --workspace-root "${ZONE_WORKSPACE}" \
        --zone "${zone}" \
        --phase "${phase}" \
        "$@"
}

ensure_zone_lidar_evidence() {
    local zone="$1"
    local zone_root="${ZONE_WORKSPACE}/zone-scenes/${zone}"
    local source_root="${zone_root}/raw/lidar"
    local evidence="${zone_root}/lidar-evidence.json"
    [[ -d "${source_root}" ]] \
        || fail "${zone} LiDAR source directory is absent: ${source_root}"
    if [[ -f "${evidence}" ]]; then
        python3.12 -m fireviewer_sdg.lidar_evidence verify \
            --source-root "${source_root}" \
            --receipt "${evidence}"
    else
        python3.12 -m fireviewer_sdg.lidar_evidence create \
            --source-root "${source_root}" \
            --output "${evidence}" \
            --require-class 2 \
            --require-class 5 \
            --require-class 6
    fi
}

build_base_scene() {
    local zone="$1"
    local zone_root="${ZONE_WORKSPACE}/zone-scenes/${zone}"
    local lidar_evidence="${zone_root}/lidar-evidence.json"
    local root_usd="${zone_root}/build/${zone}_root.usdc"
    local build_receipt="${zone_root}/build/build-receipt.json"
    local scene_gate_receipt="${zone_root}/scene-auto-validation.json"
    run_zone_phase python3.12 "${zone}" preflight
    local current
    current="$(zone_phase "${zone}")"
    if [[ "${current}" == "catalog_validated" ]]; then
        run_zone_phase python3.12 "${zone}" resolve --timeout 60 --retries 5
        current="$(zone_phase "${zone}")"
    fi
    if [[ "${current}" == "sources_resolved" ]]; then
        local lidar_scope="${FW_OMNI_LIDAR_SCOPE:-full-zone}"
        local -a lidar_args=()
        if [[ "${lidar_scope}" == "full-zone" ]]; then
            mapfile -t tile_refs < <(
                python3.12 -c \
                    'import sys; from pathlib import Path; from fireviewer_sdg.zone_scenes import _zone_rows; print("\n".join(row["tile_ref"] for row in _zone_rows(Path(sys.argv[1]), sys.argv[2])))' \
                    "${CATALOG_ROOT}" \
                    "${zone}"
            )
            local tile
            for tile in "${tile_refs[@]}"; do
                lidar_args+=(--lod0-tile "${tile}")
            done
        elif [[ "${lidar_scope}" != "review-cameras" ]]; then
            fail "FW_OMNI_LIDAR_SCOPE must be full-zone or review-cameras"
        fi
        run_zone_phase python3.12 "${zone}" acquire \
            --source-profile full \
            --download-workers "${FW_OMNI_DOWNLOAD_WORKERS:-3}" \
            --minimum-free-gib "${FW_OMNI_MIN_FREE_GIB:-300}" \
            --timeout 180 \
            "${lidar_args[@]}"
        current="$(zone_phase "${zone}")"
    fi
    if [[ "${current}" == "sources_acquired" ]]; then
        ensure_zone_lidar_evidence "${zone}"
        export FW_SDG_LIDAR_EVIDENCE_RECEIPT="${lidar_evidence}"
        run_zone_phase "${ISAAC_PYTHON}" "${zone}" build \
            --build-timeout "${FW_OMNI_BUILD_TIMEOUT:-14400}"
        current="$(zone_phase "${zone}")"
    elif [[ "${current}" =~ ^(scene_built|review_launch_requested|renders_registered|qa_accepted)$ ]]; then
        require_file "${lidar_evidence}"
    fi
    case "${current}" in
        scene_built|review_launch_requested|renders_registered|qa_accepted) ;;
        *) fail "${zone} did not reach a reviewable build state; current phase=${current}" ;;
    esac
    require_file "${root_usd}"
    require_file "${build_receipt}"
    "${ISAAC_PYTHON}" -m fireviewer_sdg.omniverse_scene_gate \
        --root-usd "${root_usd}" \
        --build-receipt "${build_receipt}" \
        --asset-manifest "${ASSET_MANIFEST}" \
        --output "${scene_gate_receipt}" \
        --minimum-tree-instances "${FW_OMNI_MIN_TREE_INSTANCES:-25000}" \
        --minimum-building-instances "${FW_OMNI_MIN_BUILDING_INSTANCES:-1}" \
        --minimum-forest-span-metres "${FW_OMNI_MIN_FOREST_SPAN_METRES:-2000}"
    printf 'BASE_SCENE_READY zone=%s root=%s\n' "${zone}" "${root_usd}"
}

build_pilot_scene() {
    ensure_isaac_runtime
    build_base_scene "${PILOT_ZONE}"
    python3.12 -m fireviewer_sdg.omniverse_pod review-pending \
        --zone "${PILOT_ZONE}" \
        --root-usd "${ROOT_USD}" \
        --runtime-preflight "${RUNTIME_PREFLIGHT}" \
        --campaign-index "${CAMPAIGN_INDEX}" \
        --asset-manifest "${ASSET_MANIFEST}" \
        --volume-root "${VOLUME_ROOT}" \
        --build-receipt "${BUILD_RECEIPT}" \
        --scene-auto-validation "${SCENE_GATE_RECEIPT}" \
        --output "${PENDING_RECEIPT}"
    printf 'AWAITING_EDITOR_REVIEW zone=%s root=%s\n' "${PILOT_ZONE}" "${ROOT_USD}"
    printf 'No fire simulation has been started or authorized.\n'
}

build_all_base_scenes() {
    ensure_isaac_runtime
    local -a requested_zones ordered_zones
    IFS=',' read -r -a requested_zones <<<"${BASE_ZONES_CSV}"
    (( ${#requested_zones[@]} == 4 )) \
        || fail "the final base build requires exactly four scenes"
    mapfile -t ordered_zones < <(printf '%s\n' "${requested_zones[@]}" | sort)
    local zone
    for zone in "${ordered_zones[@]}"; do
        build_base_scene "${zone}"
    done
    printf 'ALL_BASE_SCENES_READY count=4\n'
}

show_status() {
    configure_asset_bundle_contract
    local path
    local -a status_paths=(
        "${RUNTIME_PREFLIGHT}" \
        "${OFFICIAL_ASSET_MANIFEST}" \
        "${COMMUNITY_BUILDING_METADATA}" \
        "${ASSET_BUNDLE_RECEIPT}" \
        "${CAMPAIGN_ASSET_MANIFEST}" \
        "${CAMPAIGN_ASSET_RECEIPT}" \
        "${ASSET_BUNDLE_NATIVE_LOD_RECEIPT}" \
        "${ASSET_BUNDLE_NATIVE_PBR_RECEIPT}" \
        "${ASSET_RECEIPT}" \
        "${GROUND_MATERIAL_MANIFEST}" \
        "${GROUND_BUNDLE_ROOT}/.fireviewer-asset-bundle.json" \
        "${GEOSPATIAL_RECEIPT}" \
        "${GEOSPATIAL_LOCK}" \
        "${LIDAR_EVIDENCE}" \
        "${CAMPAIGN_INDEX}" \
        "${BUILD_RECEIPT}" \
        "${SCENE_GATE_RECEIPT}" \
        "${PENDING_RECEIPT}"
    )
    if [[ "${ASSET_BUNDLE_ENABLED}" -eq 1 ]]; then
        status_paths+=(
            "${CURATED_ASSET_MANIFEST}"
            "${MERGED_SOURCE_MANIFEST}"
            "${SOURCE_MERGE_RECEIPT}"
        )
    fi
    for path in "${status_paths[@]}"; do
        if [[ -f "${path}" ]]; then
            printf 'present %s sha256=%s\n' "${path}" "$(sha256sum "${path}" | cut -d' ' -f1)"
        else
            printf 'absent  %s\n' "${path}"
        fi
    done
}

ensure_layout
case "${PHASE}" in
    system-deps)
        install_system_dependencies
        ;;
    geospatial)
        hardware_preflight
        ensure_geospatial_runtime
        ;;
    editor)
        hardware_preflight
        build_editor
        runtime_preflight
        ;;
    catalog)
        ensure_catalog
        ;;
    assets)
        materialize_assets
        ;;
    pilot)
        hardware_preflight
        ensure_geospatial_runtime
        ensure_catalog
        write_campaign_contract
        build_pilot_scene
        ;;
    bases)
        hardware_preflight
        ensure_geospatial_runtime
        ensure_catalog
        write_campaign_contract
        build_all_base_scenes
        ;;
    review)
        accept_nvidia_terms
        ensure_catalog
        exec bash "${SCRIPT_DIR}/start-omniverse-review.sh"
        ;;
    status)
        show_status
        ;;
    prepare)
        install_system_dependencies
        hardware_preflight
        ensure_geospatial_runtime
        build_editor
        runtime_preflight
        ensure_catalog
        materialize_assets
        write_campaign_contract
        build_pilot_scene
        ;;
    *)
        fail "usage: $0 {prepare|system-deps|geospatial|editor|catalog|assets|pilot|bases|review|status}"
        ;;
esac
