#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

# Build the complete pre-fire FireViewer portfolio on the RunPod NVMe.
# This script deliberately stops at the pending human Editor review.  It never
# accepts that review, creates a simulation-allowed receipt, or starts fire.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
SETUP_SCRIPT="${SCRIPT_DIR}/setup-omniverse-pod.sh"

WORKSPACE_MOUNT="${FW_OMNI_WORKSPACE_MOUNT:-/workspace}"
VOLUME_ROOT="${FW_OMNI_VOLUME_ROOT:-${WORKSPACE_MOUNT}/fireviewer-omniverse}"
RUNTIME_ROOT="${VOLUME_ROOT}/runtime"
ISAAC_ROOT="${RUNTIME_ROOT}/isaacsim-6.0.1.0"
ISAAC_PYTHON="${ISAAC_ROOT}/bin/python"
GEOSPATIAL_ENV="${RUNTIME_ROOT}/geospatial-pdal-2.10.2"
GEOSPATIAL_SITE_PACKAGES="${GEOSPATIAL_ENV}/lib/python3.12/site-packages"
CONTROL_PYTHON="${FW_OMNI_CONTROL_PYTHON:-python3.12}"

CONTRACT_ROOT="${VOLUME_ROOT}/contracts"
RUNTIME_PREFLIGHT="${CONTRACT_ROOT}/setup-preflight.json"
CAMPAIGN_INDEX="${CONTRACT_ROOT}/campaign-index.json"
ASSET_LOD_VALIDATION="${FW_OMNI_ASSET_LOD_VALIDATION:-${CONTRACT_ROOT}/asset-bundle-native-lods.json}"
ASSET_PBR_VALIDATION="${FW_OMNI_ASSET_PBR_VALIDATION:-${CONTRACT_ROOT}/asset-bundle-native-pbr.json}"

ASSET_BUNDLE_ROOT="${FW_OMNI_ASSET_BUNDLE_ROOT:-}"
if [[ -z "${ASSET_BUNDLE_ROOT}" && -n "${FW_OMNI_ASSET_BUNDLE_SHA256:-}" ]]; then
    ASSET_BUNDLE_ROOT="${VOLUME_ROOT}/input/asset-bundles/${FW_OMNI_ASSET_BUNDLE_SHA256}"
fi
DEFAULT_ASSET_MANIFEST="${VOLUME_ROOT}/input/simready-assets-hd-v3.json"
if [[ -n "${ASSET_BUNDLE_ROOT}" ]]; then
    DEFAULT_ASSET_MANIFEST="${ASSET_BUNDLE_ROOT}/${FW_OMNI_ASSET_BUNDLE_MANIFEST_RELATIVE:-manifest-v3.json}"
fi
ASSET_MANIFEST="${FW_SDG_SIMREADY_ASSET_MANIFEST:-${DEFAULT_ASSET_MANIFEST}}"
GROUND_BUNDLE_ROOT="${FW_OMNI_GROUND_BUNDLE_ROOT:-${VOLUME_ROOT}/input/ground-pbr-4k}"
GROUND_MATERIAL_MANIFEST="${FW_OMNI_GROUND_MATERIAL_MANIFEST:-${GROUND_BUNDLE_ROOT}/manifest-v3.json}"
GROUND_BUNDLE_MARKER="${GROUND_BUNDLE_ROOT}/.fireviewer-asset-bundle.json"

ZONE_WORKSPACE="${VOLUME_ROOT}/production"
TERRAIN_ROOT="${VOLUME_ROOT}/terrain-pbr"
COMPOSITION_PREPARED_ROOT="${VOLUME_ROOT}/composition-prepared"
COMPOSITION_ROOT="${VOLUME_ROOT}/composition-sources"
VARIANT_PLAN_ROOT="${VOLUME_ROOT}/variant-plan"
VARIANT_PLAN="${VARIANT_PLAN_ROOT}/campaign-plan.json"
VARIANT_SCENES_ROOT="${FW_OMNI_VARIANT_SCENES_ROOT:-${VOLUME_ROOT}/variant-scenes}"
AUTHORING_RECEIPT="${VARIANT_SCENES_ROOT}/authoring-receipt.json"
CAMPAIGN_VERIFICATION="${CONTRACT_ROOT}/variant-campaign-verification.json"

REVIEW_SCENE="${FW_OMNI_REVIEW_SCENE:-SIM-01}"
REVIEW_SCENE_ROOT="${VARIANT_SCENES_ROOT}/${REVIEW_SCENE}"
REVIEW_ROOT_USD="${REVIEW_SCENE_ROOT}/build/root.usdc"
REVIEW_BUILD_RECEIPT="${REVIEW_SCENE_ROOT}/build/build-receipt.json"
REVIEW_SCENE_GATE_RECEIPT="${REVIEW_SCENE_ROOT}/scene-auto-validation.json"
REVIEW_PENDING_RECEIPT="${REVIEW_SCENE_ROOT}/editor-review-pending.json"

QA_ROOT="${FW_OMNI_SIM01_QA_ROOT:-${REVIEW_SCENE_ROOT}/internal-qa}"
QA_REVIEW_CAMERA_PLAN="${FW_OMNI_SIM01_REVIEW_CAMERA_PLAN:-${QA_ROOT}/review-camera-plan.json}"
QA_PROOF_PACK="${FW_OMNI_SIM01_QA_PROOF_PACK:-${QA_ROOT}/proof-pack.json}"
QA_QUALITY_REPORT="${FW_OMNI_SIM01_QA_QUALITY_REPORT:-${QA_ROOT}/quality-report.json}"
QA_STABILITY_REPORT="${FW_OMNI_SIM01_QA_STABILITY_REPORT:-${QA_ROOT}/stability-report.json}"
QA_RECEIPT="${FW_OMNI_SIM01_QA_RECEIPT:-${QA_ROOT}/sim01-internal-qa.json}"

BASE_ZONES_CSV="${FW_OMNI_BASE_ZONES:-}"
MASTER_SEED="${FW_OMNI_MASTER_SEED:-20260729}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

fail() {
    printf 'FireViewer photoreal campaign blocked: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 \
        || fail "required command is absent: $1"
}

require_file() {
    [[ -f "$1" && ! -L "$1" ]] || fail "required regular file is absent: $1"
}

require_directory() {
    [[ -d "$1" && ! -L "$1" ]] || fail "required directory is absent: $1"
}

json_equals() {
    local path="$1"
    local dotted_field="$2"
    local expected="$3"
    "${CONTROL_PYTHON}" -c \
        'import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for part in sys.argv[2].split("."):
    if not isinstance(value,dict) or part not in value:
        raise SystemExit(2)
    value=value[part]
raise SystemExit(0 if str(value).lower()==sys.argv[3].lower() else 3)' \
        "${path}" \
        "${dotted_field}" \
        "${expected}"
}

require_json_equals() {
    local path="$1"
    local dotted_field="$2"
    local expected="$3"
    require_file "${path}"
    json_equals "${path}" "${dotted_field}" "${expected}" \
        || fail "${path} does not contain ${dotted_field}=${expected}"
}

validate_campaign_verification_receipt() {
    local path="$1"
    require_file "${path}"
    "${CONTROL_PYTHON}" -c \
        'import hashlib,json,sys
from pathlib import Path
receipt_path,plan_path,authoring_path=map(Path,sys.argv[1:4])
payload=json.loads(receipt_path.read_text(encoding="utf-8"))
sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
required={
    "state":"VARIANT_CAMPAIGN_VERIFIED",
    "plan_sha256":sha256(plan_path),
    "authoring_receipt_sha256":sha256(authoring_path),
    "layout_count":4,
    "simulation_count":20,
    "root_usd_rehashed":20,
    "build_receipts_rehashed":20,
    "identity_contracts_verified":20,
    "manual_editor_review":"required",
    "fire_simulation_status":"blocked_pending_editor_review",
}
if any(payload.get(key)!=value for key,value in required.items()):
    raise SystemExit(2)
for key in (
    "terrain_payload_references_verified",
    "terrain_payload_unique_files_rehashed",
    "object_lod_payloads_rehashed",
    "ground_material_references_verified",
    "ground_material_unique_files_rehashed",
    "hash_operations",
    "bytes_hashed",
):
    value=payload.get(key)
    if not isinstance(value,int) or isinstance(value,bool) or value <= 0:
        raise SystemExit(3)' \
        "${path}" \
        "${VARIANT_PLAN}" \
        "${AUTHORING_RECEIPT}" \
        || fail "campaign verification receipt is stale or incomplete: ${path}"
}

run_native_python() {
    env \
        "PYTHONPATH=${REPO_ROOT}/src:${GEOSPATIAL_SITE_PACKAGES}" \
        "LD_LIBRARY_PATH=${GEOSPATIAL_ENV}/lib:${LD_LIBRARY_PATH:-}" \
        "${ISAAC_PYTHON}" \
        "$@"
}

verify_native_runtime() {
    require_file "${ISAAC_PYTHON}"
    require_directory "${GEOSPATIAL_SITE_PACKAGES}"
    run_native_python -c \
        'import json
import numpy
from osgeo import gdal, osr
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
if not osr.SpatialReference():
    raise SystemExit("OSR spatial reference construction failed")
print(json.dumps({
    "state":"NATIVE_TERRAIN_RUNTIME_READY",
    "gdal":gdal.VersionInfo("RELEASE_NAME"),
    "numpy":numpy.__version__,
    "usd":Usd.GetVersion(),
},sort_keys=True))'
}

validate_four_bases() {
    [[ -n "${BASE_ZONES_CSV}" ]] \
        || fail "FW_OMNI_BASE_ZONES must name exactly four zones"
    IFS=',' read -r -a BASE_ZONES <<<"${BASE_ZONES_CSV}"
    (( ${#BASE_ZONES[@]} == 4 )) \
        || fail "FW_OMNI_BASE_ZONES must contain exactly four zones"
    declare -A seen=()
    local zone
    for zone in "${BASE_ZONES[@]}"; do
        [[ "${zone}" =~ ^Z[0-9][0-9]$ ]] \
            || fail "invalid base zone identifier: ${zone}"
        [[ -z "${seen[${zone}]:-}" ]] \
            || fail "FW_OMNI_BASE_ZONES contains a duplicate zone"
        seen["${zone}"]=1
    done
    [[ "${REVIEW_SCENE}" == "SIM-01" ]] \
        || fail "the only pre-fire review target is SIM-01"
    [[ "${MASTER_SEED}" =~ ^[0-9]+$ ]] \
        || fail "FW_OMNI_MASTER_SEED must be a non-negative integer"
    [[ "${VARIANT_SCENES_ROOT}" == "${VOLUME_ROOT}/variant-scenes" ]] \
        || fail "variant scenes must use the campaign-index path ${VOLUME_ROOT}/variant-scenes"
}

bases_are_complete() {
    [[ -f "${CAMPAIGN_INDEX}" ]] || return 1
    local zone zone_root
    for zone in "${BASE_ZONES[@]}"; do
        zone_root="${ZONE_WORKSPACE}/zone-scenes/${zone}"
        [[ -f "${zone_root}/build/${zone}_root.usdc" ]] || return 1
        [[ -f "${zone_root}/build/build-receipt.json" ]] || return 1
        [[ -f "${zone_root}/scene-auto-validation.json" ]] || return 1
        json_equals \
            "${zone_root}/scene-auto-validation.json" \
            state \
            AUTO_VALIDATED \
            || return 1
    done
}

ensure_four_bases() {
    if bases_are_complete; then
        printf 'BASE_SCENES_REUSED count=4\n'
    else
        FW_OMNI_LIDAR_SCOPE=full-zone \
            bash "${SETUP_SCRIPT}" bases
    fi
    require_json_equals \
        "${CAMPAIGN_INDEX}" \
        fire_simulation_status \
        blocked_pending_editor_review
    local zone zone_root
    for zone in "${BASE_ZONES[@]}"; do
        zone_root="${ZONE_WORKSPACE}/zone-scenes/${zone}"
        require_file "${zone_root}/build/${zone}_root.usdc"
        require_file "${zone_root}/build/build-receipt.json"
        require_json_equals \
            "${zone_root}/scene-auto-validation.json" \
            state \
            AUTO_VALIDATED
    done
}

ensure_terrain_and_composition() {
    local zone zone_root ground_root request receipt prepared output contract
    local contract_sha
    LAYOUTS=()
    for zone in "${BASE_ZONES[@]}"; do
        zone_root="${ZONE_WORKSPACE}/zone-scenes/${zone}"
        ground_root="${TERRAIN_ROOT}/${zone}"
        request="${ground_root}/terrain-authoring-request.json"
        receipt="${ground_root}/authored/ground-authoring-receipt.json"
        prepared="${COMPOSITION_PREPARED_ROOT}/${zone}"
        output="${COMPOSITION_ROOT}/${zone}"
        contract="${prepared}/composition-export-input.json"

        run_native_python -m fireviewer_sdg.terrain_pbr prepare-native \
            --volume-root "${VOLUME_ROOT}" \
            --zone-root "${zone_root}" \
            --scene-auto-validation "${zone_root}/scene-auto-validation.json" \
            --bundle-root "${GROUND_BUNDLE_ROOT}" \
            --material-manifest "${GROUND_MATERIAL_MANIFEST}" \
            --artifact-root "${ground_root}" \
            --request-output "${request}"
        require_file "${request}"
        run_native_python -m fireviewer_sdg.terrain_pbr author-native \
            --request "${request}"
        require_json_equals \
            "${receipt}" \
            state \
            COMPOSITE_GROUND_MATERIAL_NATIVE_VALIDATED

        if [[ ! -f "${contract}" ]]; then
            run_native_python -m fireviewer_sdg.composition_source build \
                --volume-root "${VOLUME_ROOT}" \
                --zone-root "${zone_root}" \
                --scene-auto-validation "${zone_root}/scene-auto-validation.json" \
                --asset-manifest "${ASSET_MANIFEST}" \
                --asset-lod-validation "${ASSET_LOD_VALIDATION}" \
                --asset-pbr-validation "${ASSET_PBR_VALIDATION}" \
                --ground-artifact-root "${ground_root}" \
                --ground-authoring-receipt "${receipt}" \
                --prepared-output "${prepared}" \
                --output "${output}"
        else
            contract_sha="$(sha256sum "${contract}" | cut -d' ' -f1)"
            run_native_python -m fireviewer_sdg.composition_source verify \
                --volume-root "${VOLUME_ROOT}" \
                --contract "${contract}" \
                --contract-sha256 "${contract_sha}"
            run_native_python -m fireviewer_sdg.composition_source export \
                --volume-root "${VOLUME_ROOT}" \
                --contract "${contract}" \
                --contract-sha256 "${contract_sha}" \
                --output "${output}"
        fi
        require_json_equals \
            "${output}/composition-source.json" \
            state \
            COMPOSITION_SOURCE_READY
        LAYOUTS+=("${output}/composition-source.json")
        printf 'BASE_COMPOSITION_READY zone=%s tiles=400\n' "${zone}"
    done
}

authoring_is_complete() {
    [[ -f "${AUTHORING_RECEIPT}" ]] || return 1
    "${CONTROL_PYTHON}" -c \
        'import json,sys
from pathlib import Path
receipt=Path(sys.argv[1])
root=receipt.parent
payload=json.loads(receipt.read_text(encoding="utf-8"))
expected=[f"SIM-{index:02d}" for index in range(1,21)]
variants=payload.get("variants")
ok=(
    payload.get("state")=="VARIANT_USD_AUTHORED"
    and payload.get("simulation_count")==20
    and payload.get("fire_simulation_status")=="blocked_pending_editor_review"
    and isinstance(variants,list)
    and [item.get("simulation_id") for item in variants]==expected
    and all((root/scene/"build"/"root.usdc").is_file() for scene in expected)
    and all((root/scene/"build"/"build-receipt.json").is_file() for scene in expected)
)
raise SystemExit(0 if ok else 1)' \
        "${AUTHORING_RECEIPT}"
}

ensure_variant_campaign() {
    local -a layout_args=()
    local layout
    for layout in "${LAYOUTS[@]}"; do
        layout_args+=(--layout "${layout}")
    done
    if [[ ! -f "${VARIANT_PLAN}" ]]; then
        [[ ! -e "${VARIANT_PLAN_ROOT}" ]] \
            || fail "variant-plan directory exists without a complete campaign-plan.json"
        run_native_python -m fireviewer_sdg.native_variant_campaign plan \
            --volume-root "${VOLUME_ROOT}" \
            "${layout_args[@]}" \
            --master-seed "${MASTER_SEED}" \
            --output "${VARIANT_PLAN_ROOT}"
    fi
    require_json_equals "${VARIANT_PLAN}" state VARIANT_PLAN_READY

    if authoring_is_complete; then
        printf 'VARIANT_AUTHORING_REUSED simulations=20\n'
    else
        [[ ! -e "${VARIANT_SCENES_ROOT}" ]] \
            || fail "variant-scenes exists without a complete authoring receipt"
        run_native_python -m fireviewer_sdg.native_variant_campaign author \
            --volume-root "${VOLUME_ROOT}" \
            --plan "${VARIANT_PLAN}" \
            --output "${VARIANT_SCENES_ROOT}"
    fi
    authoring_is_complete \
        || fail "native authoring did not produce the exact blocked 20-scene portfolio"

    if [[ -e "${CAMPAIGN_VERIFICATION}" ]] && (
        [[ ! -f "${CAMPAIGN_VERIFICATION}" ]] \
            || [[ -L "${CAMPAIGN_VERIFICATION}" ]]
    ); then
        fail "campaign verification receipt must be a regular non-symlink file"
    fi
    mkdir -p "${CONTRACT_ROOT}"
    local temporary
    temporary="$(mktemp --tmpdir="${CONTRACT_ROOT}" .variant-campaign-verification.XXXXXX)"
    if ! run_native_python -m fireviewer_sdg.native_variant_campaign verify \
        --volume-root "${VOLUME_ROOT}" \
        --plan "${VARIANT_PLAN}" \
        "${layout_args[@]}" \
        --authoring-receipt "${AUTHORING_RECEIPT}" \
        >"${temporary}"; then
        fail "20-scene native campaign verification failed; evidence retained at ${temporary}"
    fi
    validate_campaign_verification_receipt "${temporary}"
    mv -- "${temporary}" "${CAMPAIGN_VERIFICATION}"
    validate_campaign_verification_receipt "${CAMPAIGN_VERIFICATION}"
}

ensure_sim01_scene_gate() {
    require_file "${REVIEW_ROOT_USD}"
    require_file "${REVIEW_BUILD_RECEIPT}"
    if [[ -f "${REVIEW_SCENE_GATE_RECEIPT}" ]]; then
        require_json_equals \
            "${REVIEW_SCENE_GATE_RECEIPT}" \
            state \
            AUTO_VALIDATED
        return
    fi
    run_native_python -m fireviewer_sdg.omniverse_scene_gate \
        --root-usd "${REVIEW_ROOT_USD}" \
        --build-receipt "${REVIEW_BUILD_RECEIPT}" \
        --asset-manifest "${ASSET_MANIFEST}" \
        --volume-root "${VOLUME_ROOT}" \
        --output "${REVIEW_SCENE_GATE_RECEIPT}" \
        --minimum-tree-instances "${FW_OMNI_MIN_TREE_INSTANCES:-25000}" \
        --minimum-building-instances "${FW_OMNI_MIN_BUILDING_INSTANCES:-1}" \
        --minimum-forest-span-metres "${FW_OMNI_MIN_FOREST_SPAN_METRES:-2000}"
    require_json_equals \
        "${REVIEW_SCENE_GATE_RECEIPT}" \
        state \
        AUTO_VALIDATED
}

ensure_sim01_internal_qa() {
    run_native_python -m fireviewer_sdg.sim01_qa_renderer produce \
        --volume-root "${VOLUME_ROOT}" \
        --runtime-preflight "${RUNTIME_PREFLIGHT}" \
        --root-usd "${REVIEW_ROOT_USD}" \
        --build-receipt "${REVIEW_BUILD_RECEIPT}" \
        --scene-auto-validation "${REVIEW_SCENE_GATE_RECEIPT}" \
        --output-root "${QA_ROOT}"
    require_file "${QA_REVIEW_CAMERA_PLAN}"
    require_file "${QA_PROOF_PACK}"
    require_file "${QA_QUALITY_REPORT}"
    require_file "${QA_STABILITY_REPORT}"
    "${CONTROL_PYTHON}" -m fireviewer_sdg.sim01_quality_gate \
        --volume-root "${VOLUME_ROOT}" \
        --runtime-preflight "${RUNTIME_PREFLIGHT}" \
        --authoring-receipt "${AUTHORING_RECEIPT}" \
        --campaign-verification "${CAMPAIGN_VERIFICATION}" \
        --scene-auto-validation "${REVIEW_SCENE_GATE_RECEIPT}" \
        --review-camera-plan "${QA_REVIEW_CAMERA_PLAN}" \
        --proof-pack "${QA_PROOF_PACK}" \
        --quality-report "${QA_QUALITY_REPORT}" \
        --stability-report "${QA_STABILITY_REPORT}" \
        --output "${QA_RECEIPT}"
    require_json_equals \
        "${QA_RECEIPT}" \
        state \
        SIM01_INTERNAL_QA_PASSED
    require_json_equals \
        "${QA_RECEIPT}" \
        review_handoff_ready \
        true
}

ensure_review_pending() {
    require_json_equals \
        "${QA_RECEIPT}" \
        state \
        SIM01_INTERNAL_QA_PASSED
    "${CONTROL_PYTHON}" -m fireviewer_sdg.omniverse_pod review-pending \
        --scene "${REVIEW_SCENE}" \
        --root-usd "${REVIEW_ROOT_USD}" \
        --runtime-preflight "${RUNTIME_PREFLIGHT}" \
        --campaign-index "${CAMPAIGN_INDEX}" \
        --asset-manifest "${ASSET_MANIFEST}" \
        --volume-root "${VOLUME_ROOT}" \
        --build-receipt "${REVIEW_BUILD_RECEIPT}" \
        --scene-auto-validation "${REVIEW_SCENE_GATE_RECEIPT}" \
        --internal-qa-receipt "${QA_RECEIPT}" \
        --output "${REVIEW_PENDING_RECEIPT}"
    require_json_equals \
        "${REVIEW_PENDING_RECEIPT}" \
        status \
        AWAITING_EDITOR_REVIEW
}

main() {
    [[ "$(uname -s)" == "Linux" ]] || fail "this production script is Linux-only"
    require_command bash
    require_command "${CONTROL_PYTHON}"
    require_command sha256sum
    require_file "${SETUP_SCRIPT}"
    validate_four_bases
    require_json_equals \
        "${RUNTIME_PREFLIGHT}" \
        state \
        SETUP_PREFLIGHT_PASSED
    require_file "${ASSET_MANIFEST}"
    require_file "${ASSET_LOD_VALIDATION}"
    require_file "${ASSET_PBR_VALIDATION}"
    [[ -n "${ASSET_BUNDLE_ROOT}" ]] \
        || fail "FW_OMNI_ASSET_BUNDLE_ROOT or FW_OMNI_ASSET_BUNDLE_SHA256 is required"
    require_directory "${ASSET_BUNDLE_ROOT}"
    require_file "${ASSET_BUNDLE_ROOT}/.fireviewer-asset-bundle.json"
    require_directory "${GROUND_BUNDLE_ROOT}"
    require_file "${GROUND_MATERIAL_MANIFEST}"
    require_json_equals \
        "${GROUND_BUNDLE_MARKER}" \
        state \
        ASSET_BUNDLE_INSTALLED

    verify_native_runtime
    ensure_four_bases
    ensure_terrain_and_composition
    ensure_variant_campaign
    ensure_sim01_scene_gate
    ensure_sim01_internal_qa
    ensure_review_pending

    printf 'AWAITING_EDITOR_REVIEW scene=SIM-01 root=%s\n' "${REVIEW_ROOT_USD}"
    printf 'No fire simulation, capture campaign, or review acceptance was started.\n'
}

main "$@"
