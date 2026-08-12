"""Fail-closed internal QA gate for the final native ``SIM-01`` scene.

The gate consumes evidence produced by other steps.  It never opens Kit,
renders an image, advances a simulation, or manufactures a human decision.
Only after every current artifact is hash-bound and every required machine or
internal-QA result is positive does it write ``SIM01_INTERNAL_QA_PASSED``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
INTERNAL_QA_STATE = "SIM01_INTERNAL_QA_PASSED"
PROOF_PACK_STATE = "SIM01_NATIVE_PROOF_PACK_CAPTURED"
QUALITY_REPORT_STATE = "SIM01_SCENE_QUALITY_PASSED"
STABILITY_REPORT_STATE = "SIM01_HEADLESS_NATIVE_QA_STABILITY_PASSED"
REVIEW_CAMERA_PLAN_STATE = "SIM01_REVIEW_CAMERA_PLAN_LOCKED"
CAMPAIGN_ID = "fireviewer-omniverse-20-photoreal-simulations-v1"
CAMERA_IDS = tuple(f"VIEW-{index:02d}" for index in range(1, 41))
REQUIRED_PROOF_VIEW_ROLES = frozenset({"vertical", "low", "inclined"})
REQUIRED_QUALITY_SECTIONS = (
    "structure",
    "density",
    "lod",
    "pbr",
    "topology",
    "native_visual",
)
PRODUCTION_GPU_NAME = "RTX PRO 6000 Blackwell Server Edition"
MINIMUM_VRAM_MIB = 90_000
MINIMUM_SYSTEM_RAM_MIB = 138_000
MINIMUM_STORAGE_BYTES = 1_500_000_000_000
PROOF_WIDTH_PX = 3_840
PROOF_HEIGHT_PX = 2_160
MINIMUM_PROOF_FILE_BYTES = 1_024
MINIMUM_RGB_STDDEV = 4.0
MINIMUM_LUMINANCE_STDDEV = 3.0
MAXIMUM_CLIPPED_FRACTION = 0.985
MINIMUM_EDGE_ENERGY = 0.15
MINIMUM_ACCEPTED_FPS = 30.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_EVIDENCE_VALUES = frozenset(
    {
        "--",
        "n/a",
        "na",
        "none",
        "not reviewed",
        "not_reviewed",
        "not run",
        "not_run",
        "not started",
        "not_started",
        "pending",
        "todo",
        "tbd",
        "unknown",
        "unverified",
    }
)
_IMAGE_SUFFIXES = frozenset({".png"})
_VISUAL_METRIC_FIELDS = (
    "rgb_stddev",
    "luminance_stddev",
    "dark_fraction",
    "bright_fraction",
    "edge_energy",
)


class Sim01QualityGateError(RuntimeError):
    """Raised when internal QA evidence is absent, stale, or negative."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_file(
    *,
    volume_root: Path,
    value: Path | str,
    label: str,
    anchor: Path | None = None,
) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else (anchor or volume_root) / raw
    if candidate.is_symlink():
        raise Sim01QualityGateError(
            f"{label} must not be a symlink"
        )
    path = candidate.resolve()
    if not _inside(volume_root, path) or not path.is_file():
        raise Sim01QualityGateError(
            f"{label} must be a regular non-symlink file below the volume"
        )
    return path


def _read_json_file(
    *,
    volume_root: Path,
    path: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = _resolved_file(
        volume_root=volume_root,
        value=path,
        label=label,
    )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Sim01QualityGateError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise Sim01QualityGateError(f"{label} must contain a JSON object")
    _reject_sentinels(payload, label=label)
    return resolved, payload


def _reject_sentinels(value: object, *, label: str) -> None:
    stack: list[tuple[str, object]] = [(label, value)]
    while stack:
        current_label, current = stack.pop()
        if current is None:
            raise Sim01QualityGateError(f"{current_label} is absent")
        if isinstance(current, str):
            if current.strip().casefold() in _FORBIDDEN_EVIDENCE_VALUES:
                raise Sim01QualityGateError(
                    f"{current_label} contains a forbidden evidence sentinel"
                )
            continue
        if isinstance(current, Mapping):
            stack.extend(
                (f"{current_label}.{key}", item)
                for key, item in current.items()
            )
            continue
        if isinstance(current, list):
            stack.extend(
                (f"{current_label}[{index}]", item)
                for index, item in enumerate(current)
            )


def _reject_forbidden_keys(
    value: object,
    *,
    forbidden: frozenset[str],
    label: str,
) -> None:
    stack: list[tuple[str, object]] = [(label, value)]
    while stack:
        current_label, current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                key_text = str(key)
                if key_text in forbidden:
                    raise Sim01QualityGateError(
                        f"{current_label}.{key_text} is a forbidden "
                        "post-review dependency"
                    )
                stack.append((f"{current_label}.{key_text}", item))
        elif isinstance(current, list):
            stack.extend(
                (f"{current_label}[{index}]", item)
                for index, item in enumerate(current)
            )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Sim01QualityGateError(f"{label} must be an object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise Sim01QualityGateError(f"{label} must be a list")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Sim01QualityGateError(f"{label} must be a non-empty string")
    text = value.strip()
    if text.casefold() in _FORBIDDEN_EVIDENCE_VALUES:
        raise Sim01QualityGateError(
            f"{label} contains a forbidden evidence sentinel"
        )
    return text


def _sha256(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    if not _SHA256.fullmatch(text):
        raise Sim01QualityGateError(f"{label} must be a lowercase SHA-256")
    return text


def _integer(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Sim01QualityGateError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise Sim01QualityGateError(f"{label} must be at least {minimum}")
    return value


def _number(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise Sim01QualityGateError(f"{label} must be finite")
    result = float(value)
    if strictly_positive and result <= 0.0:
        raise Sim01QualityGateError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise Sim01QualityGateError(f"{label} must be at least {minimum}")
    return result


def _passed(value: object, *, label: str) -> None:
    if value != "passed":
        raise Sim01QualityGateError(f"{label} is not passed")


def _zero(value: object, *, label: str) -> None:
    if _integer(value, label=label, minimum=0) != 0:
        raise Sim01QualityGateError(f"{label} must be zero")


def _artifact(
    record: object,
    *,
    volume_root: Path,
    anchor: Path,
    label: str,
    require_size: bool = False,
    suffixes: frozenset[str] | None = None,
) -> tuple[Path, dict[str, object]]:
    payload = _mapping(record, label=label)
    raw_path = _text(payload.get("path"), label=f"{label}.path")
    expected_sha = _sha256(payload.get("sha256"), label=f"{label}.sha256")
    path = _resolved_file(
        volume_root=volume_root,
        value=raw_path,
        anchor=anchor,
        label=label,
    )
    if suffixes is not None and path.suffix.casefold() not in suffixes:
        raise Sim01QualityGateError(f"{label} has an unsupported suffix")
    size = path.stat().st_size
    if size <= 0:
        raise Sim01QualityGateError(f"{label} is empty")
    if _sha256_file(path) != expected_sha:
        raise Sim01QualityGateError(f"{label} hash mismatch")
    if require_size:
        declared_size = _integer(
            payload.get("size_bytes"),
            label=f"{label}.size_bytes",
            minimum=1,
        )
        if declared_size != size:
            raise Sim01QualityGateError(f"{label} size mismatch")
    return path, {
        "path": path.relative_to(volume_root).as_posix(),
        "sha256": expected_sha,
        "size_bytes": size,
    }


def _input_record(path: Path, *, volume_root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(volume_root).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _normalized_gpu_name(value: object) -> str:
    normalized = " ".join(str(value).strip().casefold().split())
    if normalized.startswith("nvidia "):
        normalized = normalized.removeprefix("nvidia ")
    return normalized


def _driver_tuple(value: object) -> tuple[int, ...]:
    text = _text(value, label="runtime.gpu.driver_version")
    try:
        return tuple(int(item) for item in text.split("."))
    except ValueError as exc:
        raise Sim01QualityGateError(
            "runtime.gpu.driver_version is malformed"
        ) from exc


def _validate_runtime(payload: Mapping[str, Any]) -> dict[str, object]:
    if payload.get("state") != "SETUP_PREFLIGHT_PASSED":
        raise Sim01QualityGateError("runtime preflight is not passed")
    gpu = _mapping(payload.get("gpu"), label="runtime.gpu")
    if _normalized_gpu_name(gpu.get("name")) != _normalized_gpu_name(
        PRODUCTION_GPU_NAME
    ):
        raise Sim01QualityGateError(
            "runtime GPU is not RTX PRO 6000 Blackwell Server Edition"
        )
    memory_mib = _integer(
        gpu.get("memory_mib"),
        label="runtime.gpu.memory_mib",
        minimum=MINIMUM_VRAM_MIB,
    )
    _integer(
        gpu.get("minimum_memory_mib"),
        label="runtime.gpu.minimum_memory_mib",
        minimum=MINIMUM_VRAM_MIB,
    )
    if _normalized_gpu_name(gpu.get("required_name_exact")) != (
        _normalized_gpu_name(PRODUCTION_GPU_NAME)
    ):
        raise Sim01QualityGateError("runtime exact GPU requirement is absent")
    if _driver_tuple(gpu.get("driver_version")) < (570, 158, 1):
        raise Sim01QualityGateError("runtime NVIDIA driver is below 570.158.01")
    _sha256(
        gpu.get("vulkan_summary_sha256"),
        label="runtime.gpu.vulkan_summary_sha256",
    )

    system = _mapping(
        payload.get("system_memory"),
        label="runtime.system_memory",
    )
    effective_mib = _integer(
        system.get("effective_mib"),
        label="runtime.system_memory.effective_mib",
        minimum=MINIMUM_SYSTEM_RAM_MIB,
    )
    _integer(
        system.get("minimum_effective_mib"),
        label="runtime.system_memory.minimum_effective_mib",
        minimum=MINIMUM_SYSTEM_RAM_MIB,
    )
    if (
        system.get("measurement") != "finite_container_cgroup_limit"
        or system.get("host_proc_meminfo_used") is not False
        or not _text(
            system.get("source"),
            label="runtime.system_memory.source",
        ).startswith("/sys/fs/cgroup/")
    ):
        raise Sim01QualityGateError(
            "runtime RAM is not a finite container cgroup measurement"
        )
    limit_bytes = _integer(
        system.get("limit_bytes"),
        label="runtime.system_memory.limit_bytes",
        minimum=MINIMUM_SYSTEM_RAM_MIB * 1024 * 1024,
    )
    if effective_mib > limit_bytes // (1024 * 1024):
        raise Sim01QualityGateError(
            "runtime effective RAM exceeds its cgroup limit"
        )

    storage = _mapping(payload.get("storage"), label="runtime.storage")
    if (
        storage.get("mode") != "ephemeral-nvme"
        or storage.get("automatic_stop_allowed") is not False
    ):
        raise Sim01QualityGateError(
            "runtime storage is not the retained ephemeral NVMe profile"
        )
    capacity_bytes = _integer(
        storage.get("capacity_bytes"),
        label="runtime.storage.capacity_bytes",
        minimum=MINIMUM_STORAGE_BYTES,
    )
    _number(
        storage.get("minimum_capacity_gb_decimal"),
        label="runtime.storage.minimum_capacity_gb_decimal",
        minimum=1500.0,
    )
    _integer(
        storage.get("free_bytes"),
        label="runtime.storage.free_bytes",
        minimum=1,
    )
    return {
        "gpu_name": _text(gpu.get("name"), label="runtime.gpu.name"),
        "vram_mib": memory_mib,
        "effective_ram_mib": effective_mib,
        "cgroup_limit_bytes": limit_bytes,
        "storage_capacity_bytes": capacity_bytes,
    }


def _validate_authoring(
    *,
    payload: Mapping[str, Any],
    authoring_path: Path,
    volume_root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, Any], str]:
    if (
        payload.get("schema_version") != 1
        or payload.get("state") != "VARIANT_USD_AUTHORED"
        or payload.get("simulation_count") != 20
        or payload.get("manual_editor_review") != "required"
        or payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise Sim01QualityGateError(
            "authoring receipt is not the blocked verified 20-scene result"
        )
    plan_path, plan_record = _artifact(
        payload.get("plan"),
        volume_root=volume_root,
        anchor=volume_root,
        label="authoring.plan",
    )
    del plan_path
    variants = _list(payload.get("variants"), label="authoring.variants")
    if len(variants) != 20:
        raise Sim01QualityGateError(
            "authoring receipt must contain exactly 20 variants"
        )
    scene_artifacts: dict[str, dict[str, object]] = {}
    scene_builds: dict[str, Any] = {}
    for index, raw_variant in enumerate(variants, start=1):
        simulation_id = f"SIM-{index:02d}"
        variant = _mapping(
            raw_variant,
            label=f"authoring.variants[{index - 1}]",
        )
        artifacts = _mapping(
            variant.get("artifacts"),
            label=f"{simulation_id}.artifacts",
        )
        if (
            variant.get("simulation_id") != simulation_id
            or variant.get("scene_kind") != "fictive_variant"
            or variant.get("fire_simulation_status")
            != "blocked_pending_editor_review"
            or artifacts.get("scene_kind") != "fictive_variant"
            or artifacts.get("streaming_tile_count") != 400
            or artifacts.get("object_lod_payload_count") != 1200
            or artifacts.get("monolithic_object_payloads") is not False
        ):
            raise Sim01QualityGateError(
                f"{simulation_id} authoring contract is incomplete"
            )
        coverage = _list(
            artifacts.get("tile_coverage"),
            label=f"{simulation_id}.tile_coverage",
        )
        if len(coverage) != 400:
            raise Sim01QualityGateError(
                f"{simulation_id} does not expose exactly 400 tiles"
            )
        variant_root = authoring_path.parent / simulation_id
        root_path, root_record = _artifact(
            artifacts.get("root_usd"),
            volume_root=volume_root,
            anchor=variant_root,
            label=f"{simulation_id}.root_usd",
            suffixes=frozenset({".usd", ".usda", ".usdc"}),
        )
        build_path, build_record = _artifact(
            artifacts.get("composer_build_receipt"),
            volume_root=volume_root,
            anchor=variant_root,
            label=f"{simulation_id}.build_receipt",
            suffixes=frozenset({".json"}),
        )
        _, build_payload = _read_json_file(
            volume_root=volume_root,
            path=build_path,
            label=f"{simulation_id} build receipt",
        )
        if (
            build_payload.get("schema_version") != 2
            or build_payload.get("zone_id") != simulation_id
            or build_payload.get("scene_kind") != "fictive_variant"
            or build_payload.get("source_profile") != "full"
            or build_payload.get("fire_simulation_status")
            != "blocked_pending_editor_review"
            or _mapping(
                build_payload.get("root_usd"),
                label=f"{simulation_id}.build.root_usd",
            ).get("sha256")
            != root_record["sha256"]
        ):
            raise Sim01QualityGateError(
                f"{simulation_id} build receipt is stale"
            )
        scene_artifacts[simulation_id] = {
            "root_path": root_path,
            "root": root_record,
            "build_path": build_path,
            "build": build_record,
        }
        scene_builds[simulation_id] = build_payload
    review_target = _mapping(
        payload.get("review_target"),
        label="authoring.review_target",
    )
    if (
        review_target.get("simulation_id") != "SIM-01"
        or review_target.get("must_be_reviewed_before_fire") is not True
    ):
        raise Sim01QualityGateError("SIM-01 is not the authoring review target")
    return scene_artifacts, scene_builds["SIM-01"], str(plan_record["sha256"])


def _validate_campaign_verification(
    *,
    payload: Mapping[str, Any],
    authoring_path: Path,
    authoring_plan_sha256: str,
) -> dict[str, int]:
    expected_exact = {
        "layout_count": 4,
        "simulation_count": 20,
        "root_usd_rehashed": 20,
        "build_receipts_rehashed": 20,
        "terrain_payload_references_verified": 8_000,
        "object_lod_payloads_rehashed": 24_000,
        "ground_material_references_verified": 8_000,
        "identity_contracts_verified": 20,
    }
    if (
        payload.get("state") != "VARIANT_CAMPAIGN_VERIFIED"
        or payload.get("plan_sha256") != authoring_plan_sha256
        or payload.get("authoring_receipt_sha256")
        != _sha256_file(authoring_path)
        or payload.get("manual_editor_review") != "required"
        or payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise Sim01QualityGateError(
            "campaign verification is stale or not the blocked 20-scene result"
        )
    for field, expected in expected_exact.items():
        if payload.get(field) != expected:
            raise Sim01QualityGateError(
                f"campaign verification {field} must equal {expected}"
            )
    for field in (
        "terrain_payload_unique_files_rehashed",
        "ground_material_unique_files_rehashed",
        "water_payload_references_verified",
        "water_payload_unique_files_rehashed",
        "hash_operations",
        "bytes_hashed",
    ):
        _integer(
            payload.get(field),
            label=f"campaign_verification.{field}",
            minimum=1,
        )
    return expected_exact


def _validate_scene_auto_validation(
    *,
    payload: Mapping[str, Any],
    validation_path: Path,
    volume_root: Path,
    sim01: Mapping[str, object],
) -> dict[str, int]:
    root_record = _mapping(sim01.get("root"), label="SIM-01 root record")
    build_record = _mapping(sim01.get("build"), label="SIM-01 build record")
    if (
        payload.get("schema_version") != 2
        or payload.get("state") != "AUTO_VALIDATED"
        or payload.get("scene_kind") != "fictive_variant"
        or payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or payload.get("root_usd_sha256") != root_record.get("sha256")
        or payload.get("build_receipt_sha256") != build_record.get("sha256")
    ):
        raise Sim01QualityGateError(
            "SIM-01 auto-validation is stale or incomplete"
        )
    _sha256(
        payload.get("asset_manifest_sha256"),
        label="scene_auto_validation.asset_manifest_sha256",
    )
    streaming = _mapping(
        payload.get("streaming"),
        label="scene_auto_validation.streaming",
    )
    expected_streaming = {
        "terrain_payloads_inspected_incrementally": 400,
        "detail_payloads_inspected_incrementally": 400,
        "simultaneously_retained_detail_stages": 1,
        "root_terrain_payload_arcs": 400,
        "root_detail_payload_arcs": 400,
        "root_ground_material_payload_arcs": 400,
    }
    if streaming.get("root_initial_load_set") != "LoadNone":
        raise Sim01QualityGateError(
            "SIM-01 auto-validation did not use bounded payload loading"
        )
    for field, expected in expected_streaming.items():
        if streaming.get(field) != expected:
            raise Sim01QualityGateError(
                f"SIM-01 auto-validation {field} must equal {expected}"
            )
    terrain = _mapping(
        payload.get("terrain"),
        label="scene_auto_validation.terrain",
    )
    details = _mapping(
        payload.get("details"),
        label="scene_auto_validation.details",
    )
    terrain_tiles = _list(terrain.get("tiles"), label="auto.terrain.tiles")
    detail_tiles = _list(details.get("tiles"), label="auto.details.tiles")
    lod0_count = _integer(
        terrain.get("lod0_tile_count"),
        label="auto.terrain.lod0_tile_count",
        minimum=1,
    )
    if (
        terrain.get("payload_count") != 400
        or len(terrain_tiles) != 400
        or details.get("payload_count") != 400
        or len(detail_tiles) != 400
    ):
        raise Sim01QualityGateError(
            "SIM-01 auto-validation does not cover all 400 tiles"
        )
    vegetation = _integer(
        payload.get("vegetation_instances"),
        label="auto.vegetation_instances",
        minimum=1,
    )
    buildings = _integer(
        payload.get("building_instances"),
        label="auto.building_instances",
        minimum=1,
    )
    expected_totals = _mapping(
        details.get("expected_totals"),
        label="auto.details.expected_totals",
    )
    if (
        expected_totals.get("vegetation") != vegetation
        or expected_totals.get("buildings") != buildings
    ):
        raise Sim01QualityGateError(
            "SIM-01 auto-validation totals are internally inconsistent"
        )
    families = _mapping(
        payload.get("vegetation_family_instances"),
        label="auto.vegetation_family_instances",
    )
    if _integer(families.get("trees"), label="auto.trees", minimum=1) > vegetation:
        raise Sim01QualityGateError("SIM-01 tree count exceeds vegetation total")
    unique_xy = _integer(
        payload.get("forest_unique_xy"),
        label="auto.forest_unique_xy",
        minimum=1,
    )
    if unique_xy < math.ceil(vegetation * 0.98):
        raise Sim01QualityGateError(
            "SIM-01 forest contains excessive duplicate XY positions"
        )
    near_origin = _integer(
        payload.get("forest_near_origin_instances"),
        label="auto.forest_near_origin_instances",
        minimum=0,
    )
    if near_origin > max(8, int(vegetation * 0.002)):
        raise Sim01QualityGateError(
            "SIM-01 forest is concentrated at the scene origin"
        )
    bounds = _mapping(
        payload.get("forest_world_bounds"),
        label="auto.forest_world_bounds",
    )
    span = _list(bounds.get("span_metres"), label="auto.forest_span")
    if len(span) != 3 or any(
        _number(item, label="auto.forest_span", strictly_positive=True) <= 0
        for item in span[:2]
    ):
        raise Sim01QualityGateError("SIM-01 forest span is invalid")
    used_layers = _list(payload.get("used_layers"), label="auto.used_layers")
    if not used_layers:
        raise Sim01QualityGateError("SIM-01 auto-validation has no layer inventory")
    used_root = False
    for index, record in enumerate(used_layers):
        path, verified = _artifact(
            record,
            volume_root=volume_root,
            anchor=validation_path.parent,
            label=f"auto.used_layers[{index}]",
        )
        if (
            path == sim01["root_path"]
            and verified["sha256"] == root_record["sha256"]
        ):
            used_root = True
    if not used_root:
        raise Sim01QualityGateError(
            "SIM-01 auto-validation layer inventory omits the root USD"
        )
    return {
        "terrain_tiles": 400,
        "detail_tiles": 400,
        "lod0_tiles": lod0_count,
        "vegetation_instances": vegetation,
        "building_instances": buildings,
    }


def _finite_vector(
    value: object,
    *,
    length: int,
    label: str,
) -> list[float]:
    rows = _list(value, label=label)
    if len(rows) != length:
        raise Sim01QualityGateError(f"{label} must contain {length} numbers")
    return [_number(item, label=label) for item in rows]


def _validate_camera(camera: Mapping[str, Any], *, camera_id: str) -> str:
    if set(camera) != {
        "camera_id",
        "pose_local",
        "intrinsics",
        "camera_contract_sha256",
    } or camera.get("camera_id") != camera_id:
        raise Sim01QualityGateError(f"{camera_id} camera contract is malformed")
    pose = _mapping(camera.get("pose_local"), label=f"{camera_id}.pose_local")
    if set(pose) != {"position_m", "orientation_xyzw"}:
        raise Sim01QualityGateError(f"{camera_id} pose is incomplete")
    _finite_vector(
        pose.get("position_m"),
        length=3,
        label=f"{camera_id}.position_m",
    )
    orientation = _finite_vector(
        pose.get("orientation_xyzw"),
        length=4,
        label=f"{camera_id}.orientation_xyzw",
    )
    if not math.isclose(
        math.sqrt(math.fsum(item * item for item in orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-5,
    ):
        raise Sim01QualityGateError(
            f"{camera_id} orientation quaternion is not normalized"
        )
    intrinsics = _mapping(
        camera.get("intrinsics"),
        label=f"{camera_id}.intrinsics",
    )
    required_intrinsics = {
        "model",
        "width_px",
        "height_px",
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "near_clip_m",
        "far_clip_m",
    }
    if set(intrinsics) != required_intrinsics or intrinsics.get("model") != "pinhole":
        raise Sim01QualityGateError(f"{camera_id} intrinsics are incomplete")
    width = _integer(
        intrinsics.get("width_px"),
        label=f"{camera_id}.width_px",
        minimum=1,
    )
    height = _integer(
        intrinsics.get("height_px"),
        label=f"{camera_id}.height_px",
        minimum=1,
    )
    fx = _number(
        intrinsics.get("fx_px"),
        label=f"{camera_id}.fx_px",
        strictly_positive=True,
    )
    fy = _number(
        intrinsics.get("fy_px"),
        label=f"{camera_id}.fy_px",
        strictly_positive=True,
    )
    cx = _number(intrinsics.get("cx_px"), label=f"{camera_id}.cx_px")
    cy = _number(intrinsics.get("cy_px"), label=f"{camera_id}.cy_px")
    near = _number(
        intrinsics.get("near_clip_m"),
        label=f"{camera_id}.near_clip_m",
        strictly_positive=True,
    )
    far = _number(
        intrinsics.get("far_clip_m"),
        label=f"{camera_id}.far_clip_m",
        strictly_positive=True,
    )
    if (
        fx <= 0
        or fy <= 0
        or not 0 <= cx < width
        or not 0 <= cy < height
        or far <= near
    ):
        raise Sim01QualityGateError(
            f"{camera_id} intrinsics are physically invalid"
        )
    core = {
        "camera_id": camera_id,
        "pose_local": dict(pose),
        "intrinsics": dict(intrinsics),
    }
    expected = _canonical_sha256(core)
    actual = _sha256(
        camera.get("camera_contract_sha256"),
        label=f"{camera_id}.camera_contract_sha256",
    )
    if actual != expected:
        raise Sim01QualityGateError(f"{camera_id} camera hash mismatch")
    return actual


def _validate_review_camera_plan(
    *,
    payload: Mapping[str, Any],
    sim01: Mapping[str, object],
    scene_auto_validation_sha256: str,
) -> tuple[str, dict[str, str]]:
    if (
        payload.get("schema_version") != 1
        or payload.get("state") != REVIEW_CAMERA_PLAN_STATE
        or payload.get("simulation_id") != "SIM-01"
        or payload.get("root_usd_sha256") != sim01["root"]["sha256"]
        or payload.get("build_receipt_sha256") != sim01["build"]["sha256"]
        or payload.get("scene_auto_validation_sha256")
        != scene_auto_validation_sha256
        or payload.get("camera_count") != 40
        or payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or payload.get("simulation_execution_performed") is not False
        or payload.get("render_execution_performed") is not False
    ):
        raise Sim01QualityGateError(
            "review camera plan is not the blocked autonomous SIM-01 plan"
        )
    _reject_forbidden_keys(
        payload,
        forbidden=frozenset(
            {
                "simulation_allowed_receipt",
                "simulation_gate_inputs",
                "editor_acceptance",
                "acceptance_sha256",
            }
        ),
        label="review_camera_plan",
    )
    coverage = _mapping(
        payload.get("coverage_gate"),
        label="review_camera_plan.coverage_gate",
    )
    _passed(
        coverage.get("status"),
        label="review_camera_plan.coverage_gate.status",
    )
    if coverage.get("covered_tile_count") != 400:
        raise Sim01QualityGateError(
            "review cameras do not cover all 400 SIM-01 tiles"
        )
    for field in (
        "occluded_view_count",
        "below_terrain_view_count",
        "out_of_bounds_view_count",
        "duplicate_view_count",
        "non_finite_projection_count",
    ):
        _zero(
            coverage.get(field),
            label=f"review_camera_plan.coverage_gate.{field}",
        )
    cameras = _list(
        payload.get("cameras"),
        label="review_camera_plan.cameras",
    )
    if len(cameras) != 40:
        raise Sim01QualityGateError(
            "SIM-01 requires exactly 40 pre-review cameras"
        )
    camera_hashes: dict[str, str] = {}
    view_signatures: set[str] = set()
    for camera_index, raw_camera in enumerate(cameras):
        camera_id = CAMERA_IDS[camera_index]
        camera = _mapping(
            raw_camera,
            label=f"review_camera_plan.{camera_id}",
        )
        camera_hashes[camera_id] = _validate_camera(
            camera,
            camera_id=camera_id,
        )
        view_signature = _canonical_sha256(
            {
                "pose_local": camera["pose_local"],
                "intrinsics": camera["intrinsics"],
            }
        )
        if view_signature in view_signatures:
            raise Sim01QualityGateError(
                "review camera plan contains duplicate camera views"
            )
        view_signatures.add(view_signature)
    camera_checks = _list(
        payload.get("camera_checks"),
        label="review_camera_plan.camera_checks",
    )
    if len(camera_checks) != 40:
        raise Sim01QualityGateError(
            "review camera plan must contain 40 per-camera checks"
        )
    for index, raw_check in enumerate(camera_checks):
        camera_id = CAMERA_IDS[index]
        check = _mapping(
            raw_check,
            label=f"review_camera_plan.camera_checks[{index}]",
        )
        if (
            check.get("camera_id") != camera_id
            or check.get("camera_contract_sha256")
            != camera_hashes[camera_id]
            or check.get("inside_extent") is not True
            or check.get("projection_finite") is not True
        ):
            raise Sim01QualityGateError(
                f"{camera_id} per-camera QA binding is stale"
            )
        _passed(
            check.get("status"),
            label=f"review_camera_plan.{camera_id}.status",
        )
        _integer(
            check.get("covered_tile_count"),
            label=f"review_camera_plan.{camera_id}.covered_tile_count",
            minimum=1,
        )
        clearance = _number(
            check.get("minimum_terrain_clearance_m"),
            label=(
                f"review_camera_plan.{camera_id}."
                "minimum_terrain_clearance_m"
            ),
            strictly_positive=True,
        )
        occlusion = _number(
            check.get("permanent_occlusion_fraction"),
            label=(
                f"review_camera_plan.{camera_id}."
                "permanent_occlusion_fraction"
            ),
            minimum=0.0,
        )
        if clearance <= 0.0 or occlusion >= 1.0:
            raise Sim01QualityGateError(
                f"{camera_id} is below terrain or permanently occluded"
            )
    plan_without_hash = dict(payload)
    declared_plan_hash = _sha256(
        plan_without_hash.pop("plan_sha256", None),
        label="review_camera_plan.plan_sha256",
    )
    if declared_plan_hash != _canonical_sha256(plan_without_hash):
        raise Sim01QualityGateError("review camera plan self-hash mismatch")
    return declared_plan_hash, camera_hashes


def _decode_proof_image(path: Path, *, label: str) -> dict[str, float | int]:
    try:
        import numpy as np
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise Sim01QualityGateError(
            "NumPy and Pillow are required to validate native proof images"
        ) from exc

    try:
        with Image.open(path) as probe:
            image_format = probe.format
            dimensions = probe.size
            probe.verify()
        with Image.open(path) as decoded:
            decoded.load()
            array = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise Sim01QualityGateError(
            f"{label} is not a decodable native PNG render"
        ) from exc
    if (
        image_format != "PNG"
        or dimensions != (PROOF_WIDTH_PX, PROOF_HEIGHT_PX)
        or array.shape != (PROOF_HEIGHT_PX, PROOF_WIDTH_PX, 3)
    ):
        raise Sim01QualityGateError(
            f"{label} must decode as a {PROOF_WIDTH_PX}x"
            f"{PROOF_HEIGHT_PX} RGB PNG"
        )

    float_rgb = array.astype(np.float32)
    luminance = (
        float_rgb[:, :, 0] * 0.2126
        + float_rgb[:, :, 1] * 0.7152
        + float_rgb[:, :, 2] * 0.0722
    )
    metrics: dict[str, float | int] = {
        "width_px": PROOF_WIDTH_PX,
        "height_px": PROOF_HEIGHT_PX,
        "rgb_stddev": float(float_rgb.std()),
        "luminance_stddev": float(luminance.std()),
        "dark_fraction": float((luminance <= 2.0).mean()),
        "bright_fraction": float((luminance >= 253.0).mean()),
        "edge_energy": (
            float(np.abs(np.diff(luminance, axis=1)).mean())
            + float(np.abs(np.diff(luminance, axis=0)).mean())
        )
        * 0.5,
    }
    if (
        metrics["rgb_stddev"] < MINIMUM_RGB_STDDEV
        or metrics["luminance_stddev"] < MINIMUM_LUMINANCE_STDDEV
        or metrics["dark_fraction"] >= MAXIMUM_CLIPPED_FRACTION
        or metrics["bright_fraction"] >= MAXIMUM_CLIPPED_FRACTION
        or metrics["edge_energy"] < MINIMUM_EDGE_ENERGY
    ):
        raise Sim01QualityGateError(
            f"{label} is blank, clipped, uniform, or visually unresolved"
        )
    return metrics


def _validate_visual_metrics(
    value: object,
    *,
    actual: Mapping[str, float | int],
    label: str,
) -> None:
    metrics = _mapping(value, label=label)
    expected_keys = {
        "width_px",
        "height_px",
        *_VISUAL_METRIC_FIELDS,
    }
    if set(metrics) != expected_keys:
        raise Sim01QualityGateError(f"{label} is incomplete")
    if (
        metrics.get("width_px") != PROOF_WIDTH_PX
        or metrics.get("height_px") != PROOF_HEIGHT_PX
    ):
        raise Sim01QualityGateError(f"{label} dimensions are stale")
    for field in _VISUAL_METRIC_FIELDS:
        declared = _number(metrics.get(field), label=f"{label}.{field}")
        if not math.isclose(
            declared,
            float(actual[field]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise Sim01QualityGateError(
                f"{label}.{field} differs from the decoded image"
            )


def _validate_proof_pack(
    *,
    payload: Mapping[str, Any],
    proof_path: Path,
    volume_root: Path,
    sim01: Mapping[str, object],
    review_camera_plan_sha256: str,
    camera_hashes: Mapping[str, str],
) -> tuple[list[dict[str, object]], dict[str, float | int | str]]:
    if (
        payload.get("schema_version") != 1
        or payload.get("state") != PROOF_PACK_STATE
        or payload.get("simulation_id") != "SIM-01"
        or payload.get("root_usd_sha256") != sim01["root"]["sha256"]
        or payload.get("build_receipt_sha256") != sim01["build"]["sha256"]
        or payload.get("review_camera_plan_sha256")
        != review_camera_plan_sha256
        or payload.get("render_count") != 8
        or payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or payload.get("simulation_execution_performed") is not False
    ):
        raise Sim01QualityGateError("SIM-01 proof pack is stale or incomplete")
    renderer = _mapping(payload.get("renderer"), label="proof_pack.renderer")
    if (
        renderer.get("backend") != "kit_rtx_native"
        or renderer.get("native_render") is not True
        or renderer.get("screen_capture") is not False
        or renderer.get("execution_mode") != "headless_native_qa"
        or renderer.get("render_mode") != "RayTracedLighting"
        or renderer.get("resolution_px")
        != [PROOF_WIDTH_PX, PROOF_HEIGHT_PX]
    ):
        raise Sim01QualityGateError(
            "proof pack was not produced by the native Kit RTX renderer"
        )
    _integer(
        renderer.get("rt_subframes"),
        label="proof_pack.renderer.rt_subframes",
        minimum=1,
    )
    human_editor = _mapping(
        payload.get("human_editor_validation"),
        label="proof_pack.human_editor_validation",
    )
    if (
        human_editor.get("state") != "required_before_fire_simulation"
        or human_editor.get("performed") is not False
        or human_editor.get("required") is not True
    ):
        raise Sim01QualityGateError(
            "proof pack fabricated a human Editor validation"
        )
    _passed(
        payload.get("inspection_decision"),
        label="proof_pack.inspection_decision",
    )
    if payload.get("inspection_scope") != "internal_visual_qa":
        raise Sim01QualityGateError(
            "proof pack internal inspection scope is absent"
        )
    renders = _list(payload.get("renders"), label="proof_pack.renders")
    if len(renders) != 8:
        raise Sim01QualityGateError(
            "proof pack must contain exactly eight renders"
        )
    declared_visual = _list(
        payload.get("visual_metrics"),
        label="proof_pack.visual_metrics",
    )
    if len(declared_visual) != 8:
        raise Sim01QualityGateError(
            "proof pack must contain eight visual metric records"
        )
    declared_visual_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_metrics in enumerate(declared_visual, start=1):
        render_id = f"PROOF-{index:02d}"
        metric_record = _mapping(
            raw_metrics,
            label=f"proof_pack.visual_metrics[{index - 1}]",
        )
        if (
            metric_record.get("render_id") != render_id
            or set(metric_record)
            != {"render_id", "width_px", "height_px", *_VISUAL_METRIC_FIELDS}
        ):
            raise Sim01QualityGateError(
                f"proof pack visual metrics for {render_id} are malformed"
            )
        declared_visual_by_id[render_id] = {
            key: value
            for key, value in metric_record.items()
            if key != "render_id"
        }
    camera_ids: set[str] = set()
    image_paths: set[Path] = set()
    image_hashes: set[str] = set()
    metadata_paths: set[Path] = set()
    roles: set[str] = set()
    decoded_metrics: list[dict[str, float | int]] = []
    verified_renders: list[dict[str, object]] = []
    for index, raw_render in enumerate(renders, start=1):
        label = f"proof_pack.renders[{index - 1}]"
        render = _mapping(raw_render, label=label)
        render_id = f"PROOF-{index:02d}"
        camera_id = _text(render.get("camera_id"), label=f"{label}.camera_id")
        role = _text(render.get("view_role"), label=f"{label}.view_role")
        camera_hash = _sha256(
            render.get("camera_contract_sha256"),
            label=f"{label}.camera_contract_sha256",
        )
        if (
            render.get("render_id") != render_id
            or camera_id not in camera_hashes
            or camera_hash != camera_hashes[camera_id]
            or role not in {"vertical", "low", "inclined", "oblique"}
        ):
            raise Sim01QualityGateError(f"{label} camera binding is invalid")
        image_record = _mapping(render.get("image"), label=f"{label}.image")
        image_path, image = _artifact(
            image_record,
            volume_root=volume_root,
            anchor=proof_path.parent,
            label=f"{label}.image",
            require_size=True,
            suffixes=_IMAGE_SUFFIXES,
        )
        width = _integer(
            image_record.get("width_px"),
            label=f"{label}.image.width_px",
            minimum=PROOF_WIDTH_PX,
        )
        height = _integer(
            image_record.get("height_px"),
            label=f"{label}.image.height_px",
            minimum=PROOF_HEIGHT_PX,
        )
        if (
            width != PROOF_WIDTH_PX
            or height != PROOF_HEIGHT_PX
            or int(image["size_bytes"]) <= MINIMUM_PROOF_FILE_BYTES
        ):
            raise Sim01QualityGateError(
                f"{label} is not a non-empty 3840x2160 proof render"
            )
        actual_visual = _decode_proof_image(image_path, label=f"{label}.image")
        _validate_visual_metrics(
            declared_visual_by_id[render_id],
            actual=actual_visual,
            label=f"proof_pack.visual_metrics.{render_id}",
        )
        metadata_path, metadata_record = _artifact(
            render.get("metadata"),
            volume_root=volume_root,
            anchor=proof_path.parent,
            label=f"{label}.metadata",
            require_size=True,
            suffixes=frozenset({".json"}),
        )
        _, metadata = _read_json_file(
            volume_root=volume_root,
            path=metadata_path,
            label=f"{label} metadata",
        )
        if (
            metadata.get("state") != "NATIVE_PROOF_RENDER_CAPTURED"
            or metadata.get("simulation_id") != "SIM-01"
            or metadata.get("render_id") != render_id
            or metadata.get("camera_id") != camera_id
            or metadata.get("view_role") != role
            or metadata.get("camera_contract_sha256") != camera_hash
            or metadata.get("root_usd_sha256") != sim01["root"]["sha256"]
            or metadata.get("build_receipt_sha256")
            != sim01["build"]["sha256"]
            or metadata.get("review_camera_plan_sha256")
            != review_camera_plan_sha256
            or metadata.get("image_sha256") != image["sha256"]
            or metadata.get("width_px") != width
            or metadata.get("height_px") != height
            or metadata.get("renderer_backend") != "kit_rtx_native"
            or metadata.get("fire_simulation_status")
            != "blocked_pending_editor_review"
            or metadata.get("timeline_advanced") is not False
        ):
            raise Sim01QualityGateError(f"{label} metadata is stale")
        _validate_visual_metrics(
            metadata.get("visual_metrics"),
            actual=actual_visual,
            label=f"{label}.metadata.visual_metrics",
        )
        if (
            camera_id in camera_ids
            or image_path in image_paths
            or image["sha256"] in image_hashes
            or metadata_path in metadata_paths
        ):
            raise Sim01QualityGateError(
                "proof pack renders or cameras are not distinct"
            )
        camera_ids.add(camera_id)
        image_paths.add(image_path)
        image_hashes.add(str(image["sha256"]))
        metadata_paths.add(metadata_path)
        roles.add(role)
        decoded_metrics.append(actual_visual)
        verified_renders.append(
            {
                "render_id": render_id,
                "camera_id": camera_id,
                "view_role": role,
                "image": image,
                "metadata": metadata_record,
                "visual_metrics": actual_visual,
            }
        )
    if not REQUIRED_PROOF_VIEW_ROLES.issubset(roles):
        raise Sim01QualityGateError(
            "proof pack must include vertical, low and inclined views"
        )
    return verified_renders, {
        "render_count": 8,
        "distinct_image_count": len(image_hashes),
        "minimum_rgb_stddev": min(
            float(item["rgb_stddev"]) for item in decoded_metrics
        ),
        "minimum_edge_energy": min(
            float(item["edge_energy"]) for item in decoded_metrics
        ),
        "renderer_backend": "kit_rtx_native",
    }


def _quality_section(
    checks: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    section = _mapping(checks.get(name), label=f"quality_report.{name}")
    _passed(section.get("status"), label=f"quality_report.{name}.status")
    return section


def _validate_quality_report(
    *,
    payload: Mapping[str, Any],
    scene_validation_path: Path,
    sim01: Mapping[str, object],
    build: Mapping[str, Any],
    scene_counts: Mapping[str, int],
    proof_visual: Mapping[str, float | int | str],
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("state") != QUALITY_REPORT_STATE
        or payload.get("simulation_id") != "SIM-01"
        or payload.get("root_usd_sha256") != sim01["root"]["sha256"]
        or payload.get("build_receipt_sha256") != sim01["build"]["sha256"]
        or payload.get("scene_auto_validation_sha256")
        != _sha256_file(scene_validation_path)
    ):
        raise Sim01QualityGateError("SIM-01 quality report is stale")
    _passed(payload.get("status"), label="quality_report.status")
    _zero(payload.get("defect_count"), label="quality_report.defect_count")
    checks = _mapping(payload.get("checks"), label="quality_report.checks")
    for name in REQUIRED_QUALITY_SECTIONS:
        if name not in checks:
            raise Sim01QualityGateError(
                f"quality report is missing the {name} section"
            )

    structure = _quality_section(checks, "structure")
    exact_structure = {
        "terrain_tile_count": 400,
        "hero_payload_count": 400,
        "mid_payload_count": 400,
        "far_payload_count": 400,
        "collision_tile_count": 400,
        "forbidden_primitive_count": 0,
        "placeholder_count": 0,
        "empty_zone_count": 0,
        "invalid_reference_count": 0,
    }
    for field, expected in exact_structure.items():
        if structure.get(field) != expected:
            raise Sim01QualityGateError(
                f"quality structure {field} must equal {expected}"
            )

    density = _quality_section(checks, "density")
    if (
        density.get("occupied_terrain_tile_count") != 400
        or density.get("vegetation_instance_count")
        != scene_counts["vegetation_instances"]
        or density.get("building_instance_count")
        != scene_counts["building_instances"]
    ):
        raise Sim01QualityGateError("quality density counts are stale")
    for field in (
        "empty_tile_count",
        "failed_habitat_placement_count",
        "origin_pile_count",
    ):
        _zero(density.get(field), label=f"quality density {field}")

    lod = _quality_section(checks, "lod")
    exact_lod = {
        "terrain_payload_count": 400,
        "hero_payload_count": 400,
        "mid_payload_count": 400,
        "far_payload_count": 400,
        "missing_transition_count": 0,
        "missing_collision_count": 0,
    }
    for field, expected in exact_lod.items():
        if lod.get(field) != expected:
            raise Sim01QualityGateError(
                f"quality LOD {field} must equal {expected}"
            )
    if (
        lod.get("terrain_lod0_tile_count") != scene_counts["lod0_tiles"]
        or lod.get("collision_levels") != ["NEAR", "FAR"]
    ):
        raise Sim01QualityGateError("quality LOD bindings are stale")

    layers = _mapping(build.get("layers"), label="SIM-01 build.layers")
    vegetation_layer = _mapping(
        layers.get("vegetation"),
        label="SIM-01 build.layers.vegetation",
    )
    building_layer = _mapping(
        layers.get("buildings"),
        label="SIM-01 build.layers.buildings",
    )
    if (
        vegetation_layer.get("prim_count")
        != scene_counts["vegetation_instances"]
        or building_layer.get("prim_count")
        != scene_counts["building_instances"]
    ):
        raise Sim01QualityGateError(
            "quality density differs from the native build counts"
        )
    terrain_layer = _mapping(
        layers.get("terrain"),
        label="SIM-01 build.layers.terrain",
    )
    pbr = _quality_section(checks, "pbr")
    expected_ground = _integer(
        terrain_layer.get("ground_material_payload_count"),
        label="SIM-01 ground material payload count",
        minimum=1,
    )
    if (
        expected_ground != 400
        or pbr.get("ground_material_payload_count") != expected_ground
        or pbr.get("object_free_ground_tile_count") != 400
    ):
        raise Sim01QualityGateError("quality PBR tile counts are stale")
    for field in (
        "unbound_material_count",
        "global_aggregate_material_count",
        "phantom_imagery_count",
        "material_discontinuity_count",
    ):
        _zero(pbr.get(field), label=f"quality PBR {field}")

    topology = _quality_section(checks, "topology")
    roads = _mapping(layers.get("roads"), label="SIM-01 build.layers.roads")
    hydrology = _mapping(
        layers.get("hydrology"),
        label="SIM-01 build.layers.hydrology",
    )
    expected_topology = {
        "route_source_count": _integer(
            roads.get("source_feature_count"),
            label="SIM-01 route source count",
            minimum=1,
        ),
        "route_fragment_count": _integer(
            roads.get("prim_count"),
            label="SIM-01 route fragment count",
            minimum=1,
        ),
        "hydrology_source_count": _integer(
            hydrology.get("source_feature_count"),
            label="SIM-01 hydrology source count",
            minimum=1,
        ),
        "hydrology_fragment_count": _integer(
            hydrology.get("prim_count"),
            label="SIM-01 hydrology fragment count",
            minimum=1,
        ),
    }
    for field, expected in expected_topology.items():
        if topology.get(field) != expected:
            raise Sim01QualityGateError(
                f"quality topology {field} is stale"
            )
    for field in (
        "disconnected_route_count",
        "invalid_bridge_count",
        "water_overlap_count",
        "topology_violation_count",
    ):
        _zero(topology.get(field), label=f"quality topology {field}")

    native_visual = _quality_section(checks, "native_visual")
    if (
        native_visual.get("render_count") != proof_visual["render_count"]
        or native_visual.get("distinct_image_count")
        != proof_visual["distinct_image_count"]
        or native_visual.get("renderer_backend")
        != proof_visual["renderer_backend"]
    ):
        raise Sim01QualityGateError(
            "quality native visual counts or renderer are stale"
        )
    for field in ("minimum_rgb_stddev", "minimum_edge_energy"):
        declared = _number(
            native_visual.get(field),
            label=f"quality native_visual.{field}",
        )
        if not math.isclose(
            declared,
            float(proof_visual[field]),
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise Sim01QualityGateError(
                f"quality native_visual.{field} differs from proof images"
            )


def _validate_stability_report(
    *,
    payload: Mapping[str, Any],
    runtime: Mapping[str, object],
    runtime_preflight_sha256: str,
    sim01: Mapping[str, object],
) -> dict[str, object]:
    if (
        payload.get("schema_version") != 1
        or payload.get("state") != STABILITY_REPORT_STATE
        or payload.get("simulation_id") != "SIM-01"
        or payload.get("root_usd_sha256") != sim01["root"]["sha256"]
        or payload.get("build_receipt_sha256") != sim01["build"]["sha256"]
        or payload.get("runtime_preflight_sha256")
        != runtime_preflight_sha256
        or payload.get("workflow") != "headless_native_qa"
        or payload.get("execution_mode") != "headless_native_qa"
    ):
        raise Sim01QualityGateError("SIM-01 stability report is stale")
    _passed(payload.get("status"), label="stability_report.status")
    human_editor = _mapping(
        payload.get("human_editor_validation"),
        label="stability_report.human_editor_validation",
    )
    if (
        human_editor.get("state") != "required_before_fire_simulation"
        or human_editor.get("performed") is not False
        or human_editor.get("required") is not True
    ):
        raise Sim01QualityGateError(
            "headless stability evidence fabricated a human Editor decision"
        )
    attempts = _integer(
        payload.get("stage_open_attempt_count"),
        label="stability_report.stage_open_attempt_count",
        minimum=1,
    )
    if payload.get("successful_stage_open_count") != attempts:
        raise Sim01QualityGateError(
            "SIM-01 headless stage open attempts did not all succeed"
        )
    _zero(
        payload.get("failed_stage_open_count"),
        label="stability_report.failed_stage_open_count",
    )
    _number(
        payload.get("duration_seconds"),
        label="stability_report.duration_seconds",
        strictly_positive=True,
    )
    _number(
        payload.get("stage_open_seconds"),
        label="stability_report.stage_open_seconds",
        strictly_positive=True,
    )
    _number(
        payload.get("payload_settle_seconds"),
        label="stability_report.payload_settle_seconds",
        strictly_positive=True,
    )

    fps = _mapping(payload.get("fps"), label="stability_report.fps")
    _passed(fps.get("status"), label="stability_report.fps.status")
    minimum_accepted = _number(
        fps.get("acceptance_threshold_fps"),
        label="stability_report.fps.acceptance_threshold_fps",
        strictly_positive=True,
    )
    if (
        fps.get("measurement_scope")
        != "headless_kit_with_live_3840x2160_render_product"
        or fps.get("camera_id") not in CAMERA_IDS
        or fps.get("render_product_resolution_px")
        != [PROOF_WIDTH_PX, PROOF_HEIGHT_PX]
        or minimum_accepted < MINIMUM_ACCEPTED_FPS
        or minimum_accepted > 60.0
    ):
        raise Sim01QualityGateError(
            "stability FPS must measure a live 4K product with a 30-60 FPS bound"
        )
    observed_minimum = _number(
        fps.get("observed_minimum_fps"),
        label="stability_report.fps.observed_minimum_fps",
        strictly_positive=True,
    )
    observed_mean = _number(
        fps.get("observed_mean_fps"),
        label="stability_report.fps.observed_mean_fps",
        strictly_positive=True,
    )
    if observed_minimum < minimum_accepted or observed_mean < minimum_accepted:
        raise Sim01QualityGateError("SIM-01 measured FPS is below its accepted bound")
    fps_samples = _integer(
        fps.get("sample_count"),
        label="stability_report.fps.sample_count",
        minimum=1,
    )

    vram = _mapping(payload.get("vram"), label="stability_report.vram")
    _passed(vram.get("status"), label="stability_report.vram.status")
    if vram.get("measurement_scope") != "live_4k_render_product":
        raise Sim01QualityGateError(
            "SIM-01 VRAM was not measured with a live 4K render product"
        )
    vram_total = _integer(
        vram.get("total_mib"),
        label="stability_report.vram.total_mib",
        minimum=MINIMUM_VRAM_MIB,
    )
    vram_peak = _number(
        vram.get("peak_mib"),
        label="stability_report.vram.peak_mib",
        strictly_positive=True,
    )
    if vram_total != runtime["vram_mib"] or vram_peak >= vram_total:
        raise Sim01QualityGateError(
            "SIM-01 VRAM measurement exceeds or differs from the runtime"
        )
    _integer(
        vram.get("sample_count"),
        label="stability_report.vram.sample_count",
        minimum=1,
    )
    _zero(vram.get("oom_count"), label="stability_report.vram.oom_count")

    ram = _mapping(payload.get("ram"), label="stability_report.ram")
    _passed(ram.get("status"), label="stability_report.ram.status")
    if ram.get("measurement_scope") != "live_4k_render_product":
        raise Sim01QualityGateError(
            "SIM-01 RAM was not measured with a live 4K render product"
        )
    ram_limit = _integer(
        ram.get("cgroup_limit_mib"),
        label="stability_report.ram.cgroup_limit_mib",
        minimum=MINIMUM_SYSTEM_RAM_MIB,
    )
    ram_peak = _number(
        ram.get("peak_mib"),
        label="stability_report.ram.peak_mib",
        strictly_positive=True,
    )
    if ram_limit != runtime["effective_ram_mib"] or ram_peak >= ram_limit:
        raise Sim01QualityGateError(
            "SIM-01 RAM measurement exceeds or differs from the cgroup runtime"
        )
    _integer(
        ram.get("sample_count"),
        label="stability_report.ram.sample_count",
        minimum=1,
    )
    _zero(ram.get("oom_count"), label="stability_report.ram.oom_count")
    for field in (
        "crash_count",
        "hang_count",
        "device_lost_count",
        "fatal_error_count",
    ):
        _zero(payload.get(field), label=f"stability_report.{field}")
    return {
        "measurement_mode": "headless_native_qa",
        "human_editor_validation": "required_before_fire_simulation",
        "render_product_resolution_px": [PROOF_WIDTH_PX, PROOF_HEIGHT_PX],
        "acceptance_threshold_fps": minimum_accepted,
        "observed_minimum_fps": observed_minimum,
        "observed_mean_fps": observed_mean,
        "fps_sample_count": fps_samples,
        "peak_vram_mib": vram_peak,
        "peak_ram_mib": ram_peak,
    }


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Sim01QualityGateError(
            "internal QA output already exists; refusing to overwrite evidence"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Sim01QualityGateError(
                "internal QA output already exists; refusing to overwrite evidence"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def evaluate_sim01_internal_qa(
    *,
    volume_root: Path,
    runtime_preflight_path: Path,
    authoring_receipt_path: Path,
    campaign_verification_path: Path,
    scene_auto_validation_path: Path,
    review_camera_plan_path: Path,
    proof_pack_path: Path,
    quality_report_path: Path,
    stability_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate every pre-review artifact and atomically write the QA receipt."""

    raw_volume = volume_root.expanduser()
    if raw_volume.is_symlink():
        raise Sim01QualityGateError("volume_root must not be a symlink")
    volume = raw_volume.resolve()
    if not volume.is_dir():
        raise Sim01QualityGateError(
            "volume_root must be an existing non-symlink directory"
        )
    output = output_path.expanduser()
    output = output if output.is_absolute() else volume / output
    output = output.resolve()
    if not _inside(volume, output):
        raise Sim01QualityGateError("internal QA output must stay below the volume")
    runtime_path, runtime_payload = _read_json_file(
        volume_root=volume,
        path=runtime_preflight_path,
        label="runtime preflight",
    )
    authoring_path, authoring_payload = _read_json_file(
        volume_root=volume,
        path=authoring_receipt_path,
        label="authoring receipt",
    )
    verification_path, verification_payload = _read_json_file(
        volume_root=volume,
        path=campaign_verification_path,
        label="campaign verification",
    )
    scene_validation_path, scene_validation_payload = _read_json_file(
        volume_root=volume,
        path=scene_auto_validation_path,
        label="SIM-01 scene auto-validation",
    )
    review_camera_plan_path, review_camera_plan_payload = _read_json_file(
        volume_root=volume,
        path=review_camera_plan_path,
        label="SIM-01 pre-review camera plan",
    )
    proof_path, proof_payload = _read_json_file(
        volume_root=volume,
        path=proof_pack_path,
        label="SIM-01 proof pack",
    )
    quality_path, quality_payload = _read_json_file(
        volume_root=volume,
        path=quality_report_path,
        label="SIM-01 quality report",
    )
    stability_path, stability_payload = _read_json_file(
        volume_root=volume,
        path=stability_report_path,
        label="SIM-01 stability report",
    )

    runtime = _validate_runtime(runtime_payload)
    scene_artifacts, sim01_build, authoring_plan_sha = _validate_authoring(
        payload=authoring_payload,
        authoring_path=authoring_path,
        volume_root=volume,
    )
    verification = _validate_campaign_verification(
        payload=verification_payload,
        authoring_path=authoring_path,
        authoring_plan_sha256=authoring_plan_sha,
    )
    sim01 = scene_artifacts["SIM-01"]
    scene_counts = _validate_scene_auto_validation(
        payload=scene_validation_payload,
        validation_path=scene_validation_path,
        volume_root=volume,
        sim01=sim01,
    )
    review_camera_plan_sha, camera_hashes = _validate_review_camera_plan(
        payload=review_camera_plan_payload,
        sim01=sim01,
        scene_auto_validation_sha256=_sha256_file(scene_validation_path),
    )
    verified_renders, proof_visual = _validate_proof_pack(
        payload=proof_payload,
        proof_path=proof_path,
        volume_root=volume,
        sim01=sim01,
        review_camera_plan_sha256=review_camera_plan_sha,
        camera_hashes=camera_hashes,
    )
    _validate_quality_report(
        payload=quality_payload,
        scene_validation_path=scene_validation_path,
        sim01=sim01,
        build=sim01_build,
        scene_counts=scene_counts,
        proof_visual=proof_visual,
    )
    stability = _validate_stability_report(
        payload=stability_payload,
        runtime=runtime,
        runtime_preflight_sha256=_sha256_file(runtime_path),
        sim01=sim01,
    )

    inputs = {
        "runtime_preflight": _input_record(runtime_path, volume_root=volume),
        "authoring_receipt": _input_record(authoring_path, volume_root=volume),
        "campaign_verification": _input_record(
            verification_path,
            volume_root=volume,
        ),
        "scene_auto_validation": _input_record(
            scene_validation_path,
            volume_root=volume,
        ),
        "review_camera_plan": _input_record(
            review_camera_plan_path,
            volume_root=volume,
        ),
        "proof_pack": _input_record(proof_path, volume_root=volume),
        "quality_report": _input_record(quality_path, volume_root=volume),
        "stability_report": _input_record(stability_path, volume_root=volume),
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": INTERNAL_QA_STATE,
        "simulation_id": "SIM-01",
        "checked_at": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "review_handoff_ready": True,
        "next_gate": "human_editor_review_required",
        "fire_simulation_status": "blocked_pending_editor_review",
        "rendering_performed_by_gate": False,
        "simulation_performed_by_gate": False,
        "bindings": {
            "root_usd_sha256": sim01["root"]["sha256"],
            "build_receipt_sha256": sim01["build"]["sha256"],
            "asset_manifest_sha256": scene_validation_payload[
                "asset_manifest_sha256"
            ],
            "authoring_plan_sha256": authoring_plan_sha,
            "review_camera_plan_sha256": review_camera_plan_sha,
            "inputs": inputs,
        },
        "gates": {
            "runtime_cgroup_rtx_nvme": "passed",
            "twenty_scene_authoring": "passed",
            "twenty_scene_verification": "passed",
            "sim01_auto_validation": "passed",
            "forty_capture_cameras": "passed",
            "eight_native_proof_renders": "passed",
            "structure_density_lod_pbr_topology": "passed",
            "editor_stability": "passed",
            "no_primitives_placeholders_or_empty_zones": "passed",
        },
        "counts": {
            "authored_simulations": 20,
            "verified_root_usd": verification["root_usd_rehashed"],
            "capture_cameras": 40,
            "proof_renders": len(verified_renders),
            "terrain_tiles": scene_counts["terrain_tiles"],
            "object_lod_payloads": 1200,
            "vegetation_instances": scene_counts["vegetation_instances"],
            "building_instances": scene_counts["building_instances"],
        },
        "proof_renders": verified_renders,
        "stability": stability,
        "proof_boundary": (
            "internal evidence gate only; no human Editor acceptance and no "
            "fire simulation are claimed"
        ),
    }
    if output.exists():
        _, existing = _read_json_file(
            volume_root=volume,
            path=output,
            label="existing SIM-01 internal QA receipt",
        )
        expected_without_time = dict(receipt)
        existing_without_time = dict(existing)
        expected_without_time.pop("checked_at", None)
        existing_without_time.pop("checked_at", None)
        if existing_without_time != expected_without_time:
            raise Sim01QualityGateError(
                "existing SIM-01 internal QA receipt is stale for current evidence"
            )
        return existing
    _atomic_write_new_json(output, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the complete SIM-01 internal QA evidence without "
            "opening Kit, rendering, or simulating"
        )
    )
    parser.add_argument("--volume-root", required=True, type=Path)
    parser.add_argument("--runtime-preflight", required=True, type=Path)
    parser.add_argument("--authoring-receipt", required=True, type=Path)
    parser.add_argument("--campaign-verification", required=True, type=Path)
    parser.add_argument("--scene-auto-validation", required=True, type=Path)
    parser.add_argument("--review-camera-plan", required=True, type=Path)
    parser.add_argument("--proof-pack", required=True, type=Path)
    parser.add_argument("--quality-report", required=True, type=Path)
    parser.add_argument("--stability-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = evaluate_sim01_internal_qa(
        volume_root=args.volume_root,
        runtime_preflight_path=args.runtime_preflight,
        authoring_receipt_path=args.authoring_receipt,
        campaign_verification_path=args.campaign_verification,
        scene_auto_validation_path=args.scene_auto_validation,
        review_camera_plan_path=args.review_camera_plan,
        proof_pack_path=args.proof_pack,
        quality_report_path=args.quality_report,
        stability_report_path=args.stability_report,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "state": receipt["state"],
                "simulation_id": receipt["simulation_id"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTERNAL_QA_STATE",
    "PROOF_PACK_STATE",
    "QUALITY_REPORT_STATE",
    "REVIEW_CAMERA_PLAN_STATE",
    "STABILITY_REPORT_STATE",
    "Sim01QualityGateError",
    "build_parser",
    "evaluate_sim01_internal_qa",
    "main",
]
