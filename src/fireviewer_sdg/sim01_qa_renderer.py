"""Native, fail-closed SIM-01 pre-review renderer and evidence producer.

This module is deliberately separate from :mod:`sim01_quality_gate`.  The
quality gate only consumes evidence; this module is the process that opens the
real authored USD stage in a headless Isaac/Kit RTX runtime, creates a
scene-derived forty-camera review plan, captures eight real RGB renders, and
writes the machine evidence consumed by that gate.

No timeline is played and no fire state is authored.  Interrupted runs are
resumable at the per-render boundary.  A complete, current proof pack is
idempotently reused; stale final receipts are never silently overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, MutableMapping, Sequence


SCHEMA_VERSION = 1
SIMULATION_ID = "SIM-01"
CAMERA_IDS = tuple(f"VIEW-{index:02d}" for index in range(1, 41))
PROOF_IDS = tuple(f"PROOF-{index:02d}" for index in range(1, 9))
PROOF_ROLES = (
    "vertical",
    "inclined",
    "low",
    "oblique",
    "vertical",
    "inclined",
    "low",
    "oblique",
)
REVIEW_CAMERA_PLAN_STATE = "SIM01_REVIEW_CAMERA_PLAN_LOCKED"
PROOF_PACK_STATE = "SIM01_NATIVE_PROOF_PACK_CAPTURED"
QUALITY_REPORT_STATE = "SIM01_SCENE_QUALITY_PASSED"
STABILITY_REPORT_STATE = "SIM01_HEADLESS_NATIVE_QA_STABILITY_PASSED"
COMPLETE_STATE = "SIM01_QA_RENDERER_COMPLETE"
BLOCKED_FIRE_STATE = "blocked_pending_editor_review"
EXECUTION_MODE = "headless_native_qa"
HUMAN_EDITOR_GATE = "required_before_fire_simulation"
GPU_NAME = "RTX PRO 6000 Blackwell Server Edition"
MINIMUM_VRAM_MIB = 90_000
MINIMUM_RAM_MIB = 138_000
MINIMUM_STORAGE_BYTES = 1_500_000_000_000
TILE_COUNT = 400
DETAIL_LEVELS = ("HERO", "MID", "FAR")
DETAIL_LEVEL_PATH_NAMES = {
    "HERO": "Details",
    "MID": "DetailsMid",
    "FAR": "DetailsFar",
}
STREAMING_PLANNER_SOURCE = "tools/open-zone-scene-in-composer.py::_plan_working_set"
STREAMING_TRANSITION_SOURCE = "tools/open-zone-scene-in-composer.py::_apply_plan"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PRIM_TYPES = frozenset(
    {"Cube", "Cone", "Cylinder", "Sphere", "Capsule"}
)
_FORBIDDEN_PATH_TOKENS = re.compile(
    r"(?:^|[/_.-])(placeholder|fallback|proxy[_-]?primitive)(?:$|[/_.-])",
    re.IGNORECASE,
)


class Sim01QaRendererError(RuntimeError):
    """Raised when native QA evidence cannot be produced truthfully."""


@dataclass(frozen=True)
class Tile:
    """One exact SIM-01 terrain/detail tile."""

    index: int
    tile_ref: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    maximum_z: float

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) * 0.5

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) * 0.5


@dataclass(frozen=True)
class Inputs:
    """Validated immutable inputs for native SIM-01 QA."""

    volume_root: Path
    runtime_preflight_path: Path
    root_usd_path: Path
    build_receipt_path: Path
    scene_auto_validation_path: Path
    runtime: dict[str, Any]
    build: dict[str, Any]
    auto_validation: dict[str, Any]
    root_usd_sha256: str
    build_receipt_sha256: str
    scene_auto_validation_sha256: str
    scene_root: Path
    tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class PostRenderMeasurement:
    """Headless frame and memory evidence captured with a live 4K render product."""

    camera_id: str
    resolution_px: tuple[int, int]
    frame_times: tuple[float, ...]
    vram_samples: tuple[float, ...]
    ram_samples: tuple[float, ...]


@dataclass(frozen=True)
class StreamingApplication:
    """One settled exclusive detail/terrain working set for a QA camera."""

    snapshot: dict[str, Any]
    settle_seconds: float


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
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Sim01QaRendererError(f"{label} is absent or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise Sim01QaRendererError(f"{label} must contain a JSON object")
    return payload


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _regular_file(
    value: Path,
    *,
    volume_root: Path,
    label: str,
    suffixes: frozenset[str] | None = None,
) -> Path:
    path = value.expanduser().resolve()
    if (
        not _inside(volume_root, path)
        or not path.is_file()
        or path.is_symlink()
        or (suffixes is not None and path.suffix.lower() not in suffixes)
    ):
        raise Sim01QaRendererError(
            f"{label} must be a regular non-symlink file below the volume"
        )
    return path


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Sim01QaRendererError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Sim01QaRendererError(f"{label} must be a finite number")
    return result


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Sim01QaRendererError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _normalized_gpu_name(value: object) -> str:
    name = " ".join(str(value).strip().casefold().split())
    if name.startswith("nvidia "):
        name = name.removeprefix("nvidia ")
    return name


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    if runtime.get("state") != "SETUP_PREFLIGHT_PASSED":
        raise Sim01QaRendererError("runtime preflight is not passed")
    gpu = runtime.get("gpu")
    system = runtime.get("system_memory")
    storage = runtime.get("storage")
    if not isinstance(gpu, Mapping):
        raise Sim01QaRendererError("runtime GPU contract is absent")
    if _normalized_gpu_name(gpu.get("name")) != _normalized_gpu_name(GPU_NAME):
        raise Sim01QaRendererError(
            "native QA requires RTX PRO 6000 Blackwell Server Edition"
        )
    _integer(gpu.get("memory_mib"), label="runtime GPU memory", minimum=MINIMUM_VRAM_MIB)
    if not isinstance(system, Mapping):
        raise Sim01QaRendererError("runtime cgroup memory contract is absent")
    _integer(
        system.get("effective_mib"),
        label="runtime effective RAM",
        minimum=MINIMUM_RAM_MIB,
    )
    if (
        system.get("measurement") != "finite_container_cgroup_limit"
        or system.get("host_proc_meminfo_used") is not False
        or not str(system.get("source", "")).startswith("/sys/fs/cgroup/")
    ):
        raise Sim01QaRendererError(
            "native QA requires the finite container cgroup RAM measurement"
        )
    if not isinstance(storage, Mapping):
        raise Sim01QaRendererError("runtime storage contract is absent")
    if (
        storage.get("mode") != "ephemeral-nvme"
        or storage.get("automatic_stop_allowed") is not False
    ):
        raise Sim01QaRendererError(
            "native QA requires retained ephemeral NVMe storage"
        )
    _integer(
        storage.get("capacity_bytes"),
        label="runtime storage capacity",
        minimum=MINIMUM_STORAGE_BYTES,
    )


def _resolve_record(
    record: object,
    *,
    volume_root: Path,
    anchor: Path,
    label: str,
    verify_hash: bool = True,
) -> Path:
    if not isinstance(record, Mapping):
        raise Sim01QaRendererError(f"{label} artifact record is absent")
    raw_path = record.get("path")
    expected = record.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected, str)
        or not _SHA256.fullmatch(expected)
    ):
        raise Sim01QaRendererError(f"{label} artifact record is malformed")
    candidate = Path(raw_path)
    path = (candidate if candidate.is_absolute() else anchor / candidate).resolve()
    path = _regular_file(path, volume_root=volume_root, label=label)
    if verify_hash and _sha256_file(path) != expected:
        raise Sim01QaRendererError(f"{label} SHA-256 is stale")
    return path


def _tile_bounds(record: Mapping[str, Any], *, index: int) -> Tile:
    bounds = record.get("local_bounds")
    tile_ref = record.get("tile_ref")
    if not isinstance(bounds, Mapping) or not isinstance(tile_ref, str) or not tile_ref:
        raise Sim01QaRendererError(f"tile_coverage[{index}] is malformed")
    min_x = _number(bounds.get("min_x"), label=f"{tile_ref}.min_x")
    min_y = _number(bounds.get("min_y"), label=f"{tile_ref}.min_y")
    max_x = _number(bounds.get("max_x"), label=f"{tile_ref}.max_x")
    max_y = _number(bounds.get("max_y"), label=f"{tile_ref}.max_y")
    if max_x <= min_x or max_y <= min_y:
        raise Sim01QaRendererError(f"{tile_ref} has invalid local bounds")
    return Tile(index, tile_ref, min_x, min_y, max_x, max_y, 0.0)


def _validate_catalog(
    values: object,
    *,
    expected_count: int,
    volume_root: Path,
    scene_root: Path,
    label: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(values, list) or len(values) != expected_count:
        raise Sim01QaRendererError(
            f"{label} must contain exactly {expected_count} artifacts"
        )
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise Sim01QaRendererError(f"{label}[{index}] is malformed")
        _resolve_record(
            raw,
            volume_root=volume_root,
            anchor=scene_root,
            label=f"{label}[{index}]",
            verify_hash=True,
        )
        result.append(raw)
    return result


def load_inputs(
    *,
    volume_root: Path,
    runtime_preflight_path: Path,
    root_usd_path: Path,
    build_receipt_path: Path,
    scene_auto_validation_path: Path,
) -> Inputs:
    """Validate all immutable scene inputs and every streamed payload file."""

    volume = volume_root.expanduser().resolve()
    if not volume.is_dir() or volume.is_symlink():
        raise Sim01QaRendererError("volume root must be a real directory")
    runtime_path = _regular_file(
        runtime_preflight_path,
        volume_root=volume,
        label="runtime preflight",
        suffixes=frozenset({".json"}),
    )
    root_path = _regular_file(
        root_usd_path,
        volume_root=volume,
        label="SIM-01 root USD",
        suffixes=frozenset({".usd", ".usda", ".usdc"}),
    )
    build_path = _regular_file(
        build_receipt_path,
        volume_root=volume,
        label="SIM-01 build receipt",
        suffixes=frozenset({".json"}),
    )
    auto_path = _regular_file(
        scene_auto_validation_path,
        volume_root=volume,
        label="SIM-01 scene auto-validation",
        suffixes=frozenset({".json"}),
    )
    runtime = _read_json(runtime_path, label="runtime preflight")
    build = _read_json(build_path, label="SIM-01 build receipt")
    auto = _read_json(auto_path, label="SIM-01 scene auto-validation")
    _validate_runtime(runtime)
    root_sha = _sha256_file(root_path)
    build_sha = _sha256_file(build_path)
    auto_sha = _sha256_file(auto_path)
    if (
        build.get("schema_version") != 2
        or build.get("zone_id") != SIMULATION_ID
        or build.get("scene_kind") != "fictive_variant"
        or build.get("source_profile") != "full"
        or build.get("fire_simulation_status") != BLOCKED_FIRE_STATE
    ):
        raise Sim01QaRendererError("SIM-01 build contract is incomplete")
    root_record = build.get("root_usd")
    if not isinstance(root_record, Mapping):
        raise Sim01QaRendererError("SIM-01 build root record is absent")
    scene_root = build_path.parent.parent.resolve()
    recorded_root = _resolve_record(
        root_record,
        volume_root=volume,
        anchor=scene_root,
        label="SIM-01 recorded root USD",
    )
    if recorded_root != root_path or root_record.get("sha256") != root_sha:
        raise Sim01QaRendererError("SIM-01 root USD does not match its build receipt")
    if (
        auto.get("schema_version") != 2
        or auto.get("state") != "AUTO_VALIDATED"
        or auto.get("scene_kind") != "fictive_variant"
        or auto.get("fire_simulation_status") != BLOCKED_FIRE_STATE
        or auto.get("root_usd_sha256") != root_sha
        or auto.get("build_receipt_sha256") != build_sha
    ):
        raise Sim01QaRendererError("SIM-01 scene auto-validation is stale")

    _validate_catalog(
        build.get("payloads"),
        expected_count=TILE_COUNT,
        volume_root=volume,
        scene_root=scene_root,
        label="terrain payloads",
    )
    for field, level in (
        ("detail_payloads", "HERO"),
        ("detail_mid_payloads", "MID"),
        ("detail_far_payloads", "FAR"),
    ):
        _validate_catalog(
            build.get(field),
            expected_count=TILE_COUNT,
            volume_root=volume,
            scene_root=scene_root,
            label=f"{level} detail payloads",
        )
    ground = build.get("ground_material")
    if not isinstance(ground, Mapping):
        raise Sim01QaRendererError("ground material contract is absent")
    _resolve_record(
        ground.get("index"),
        volume_root=volume,
        anchor=scene_root,
        label="ground material index",
    )
    _validate_catalog(
        ground.get("tile_material_payloads"),
        expected_count=TILE_COUNT,
        volume_root=volume,
        scene_root=scene_root,
        label="ground material payloads",
    )
    asset_lock = build.get("asset_lock")
    if not isinstance(asset_lock, Mapping):
        raise Sim01QaRendererError("asset lock is absent")
    _resolve_record(
        asset_lock,
        volume_root=volume,
        anchor=scene_root,
        label="asset lock",
    )
    _resolve_record(
        asset_lock.get("shared_manifest"),
        volume_root=volume,
        anchor=scene_root,
        label="shared asset manifest",
    )
    locked_assets = asset_lock.get("assets")
    if not isinstance(locked_assets, list) or not locked_assets:
        raise Sim01QaRendererError("asset lock contains no materialized assets")
    for index, asset in enumerate(locked_assets):
        if not isinstance(asset, Mapping):
            raise Sim01QaRendererError(f"asset lock entry {index} is malformed")
        if "manifest" in asset and "manifest_sha256" in asset:
            manifest_record = {
                "path": asset["manifest"],
                "sha256": asset["manifest_sha256"],
            }
            _resolve_record(
                manifest_record,
                volume_root=volume,
                anchor=volume,
                label=f"asset lock manifest {index}",
            )

    coverage = build.get("tile_coverage")
    if not isinstance(coverage, list) or len(coverage) != TILE_COUNT:
        raise Sim01QaRendererError("SIM-01 must expose exactly 400 tile bounds")
    tiles = tuple(
        _tile_bounds(raw, index=index)
        for index, raw in enumerate(coverage)
        if isinstance(raw, Mapping)
    )
    if len(tiles) != TILE_COUNT or len({tile.tile_ref for tile in tiles}) != TILE_COUNT:
        raise Sim01QaRendererError("SIM-01 tile identities are incomplete")
    if len({tile.index for tile in tiles}) != TILE_COUNT:
        raise Sim01QaRendererError("SIM-01 tile indices repeat")
    layers = build.get("layers")
    if not isinstance(layers, Mapping):
        raise Sim01QaRendererError("SIM-01 layer inventory is absent")
    expected_layers = {
        "terrain": TILE_COUNT,
        "collisions": TILE_COUNT,
        "detail_streaming": TILE_COUNT,
    }
    for name, expected in expected_layers.items():
        section = layers.get(name)
        if not isinstance(section, Mapping) or section.get("prim_count") != expected:
            raise Sim01QaRendererError(f"SIM-01 {name} layer count is stale")
    if (
        not isinstance(layers.get("vegetation"), Mapping)
        or _integer(
            layers["vegetation"].get("prim_count"),
            label="vegetation instance count",
            minimum=1,
        )
        != auto.get("vegetation_instances")
        or not isinstance(layers.get("buildings"), Mapping)
        or _integer(
            layers["buildings"].get("prim_count"),
            label="building instance count",
            minimum=1,
        )
        != auto.get("building_instances")
    ):
        raise Sim01QaRendererError("SIM-01 density counts are stale")
    return Inputs(
        volume_root=volume,
        runtime_preflight_path=runtime_path,
        root_usd_path=root_path,
        build_receipt_path=build_path,
        scene_auto_validation_path=auto_path,
        runtime=runtime,
        build=build,
        auto_validation=auto,
        root_usd_sha256=root_sha,
        build_receipt_sha256=build_sha,
        scene_auto_validation_sha256=auto_sha,
        scene_root=scene_root,
        tiles=tiles,
    )


def _normalize(vector: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(math.fsum(float(value) ** 2 for value in vector))
    if length <= 1.0e-9:
        raise Sim01QaRendererError("camera direction is degenerate")
    return tuple(float(value) / length for value in vector)  # type: ignore[return-value]


def _cross(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _quaternion_from_basis(
    right: Sequence[float],
    up: Sequence[float],
    back: Sequence[float],
) -> tuple[float, float, float, float]:
    matrix = (
        (right[0], up[0], back[0]),
        (right[1], up[1], back[1]),
        (right[2], up[2], back[2]),
    )
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return (qx / norm, qy / norm, qz / norm, qw / norm)


def _look_at_quaternion(
    position: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float, float]:
    forward = _normalize(
        (
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        )
    )
    up_hint = (0.0, 1.0, 0.0) if abs(forward[2]) > 0.98 else (0.0, 0.0, 1.0)
    right = _normalize(_cross(forward, up_hint))
    up = _normalize(_cross(right, forward))
    back = (-forward[0], -forward[1], -forward[2])
    return _quaternion_from_basis(right, up, back)


def _camera_frame(
    position: Sequence[float], target: Sequence[float]
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    forward = _normalize(
        (
            target[0] - position[0],
            target[1] - position[1],
            target[2] - position[2],
        )
    )
    up_hint = (0.0, 1.0, 0.0) if abs(forward[2]) > 0.98 else (0.0, 0.0, 1.0)
    right = _normalize(_cross(forward, up_hint))
    up = _normalize(_cross(right, forward))
    return right, up, forward


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(left[index] * right[index] for index in range(3))


def _visible_tiles(
    *,
    tiles: Sequence[Tile],
    position: Sequence[float],
    target: Sequence[float],
    intrinsics: Mapping[str, Any],
) -> list[Tile]:
    right, up, forward = _camera_frame(position, target)
    tan_h = float(intrinsics["width_px"]) / (2.0 * float(intrinsics["fx_px"]))
    tan_v = float(intrinsics["height_px"]) / (2.0 * float(intrinsics["fy_px"]))
    near = float(intrinsics["near_clip_m"])
    far = float(intrinsics["far_clip_m"])
    visible: list[Tile] = []
    for tile in tiles:
        point = (tile.center_x, tile.center_y, tile.maximum_z + 40.0)
        relative = (
            point[0] - position[0],
            point[1] - position[1],
            point[2] - position[2],
        )
        depth = _dot(relative, forward)
        radius = math.hypot(tile.max_x - tile.min_x, tile.max_y - tile.min_y) * 0.5
        if (
            depth > near
            and depth < far
            and abs(_dot(relative, right)) <= depth * tan_h + radius
            and abs(_dot(relative, up)) <= depth * tan_v + radius
        ):
            visible.append(tile)
    return visible


def _containing_tile(tiles: Sequence[Tile], x: float, y: float) -> Tile | None:
    for tile in tiles:
        if tile.min_x <= x <= tile.max_x and tile.min_y <= y <= tile.max_y:
            return tile
    return None


def _occlusion_fraction(
    *,
    tiles: Sequence[Tile],
    visible: Sequence[Tile],
    position: Sequence[float],
) -> float:
    if not visible:
        return 1.0
    occluded = 0
    for target_tile in visible:
        target = (
            target_tile.center_x,
            target_tile.center_y,
            target_tile.maximum_z + 40.0,
        )
        blocked = False
        for step in range(1, 24):
            fraction = step / 25.0
            x = position[0] + (target[0] - position[0]) * fraction
            y = position[1] + (target[1] - position[1]) * fraction
            z = position[2] + (target[2] - position[2]) * fraction
            envelope = _containing_tile(tiles, x, y)
            if envelope is not None and z <= envelope.maximum_z + 5.0:
                blocked = True
                break
        if blocked:
            occluded += 1
    return occluded / len(visible)


def _camera_core(
    *,
    camera_id: str,
    position: Sequence[float],
    target: Sequence[float],
) -> dict[str, Any]:
    width = 3840
    height = 2160
    focal_length_mm = 14.0
    horizontal_aperture_mm = 36.0
    focal_pixels = width * focal_length_mm / horizontal_aperture_mm
    core: dict[str, Any] = {
        "camera_id": camera_id,
        "pose_local": {
            "position_m": [float(value) for value in position],
            "orientation_xyzw": list(_look_at_quaternion(position, target)),
        },
        "intrinsics": {
            "model": "pinhole",
            "width_px": width,
            "height_px": height,
            "fx_px": focal_pixels,
            "fy_px": focal_pixels,
            "cx_px": width * 0.5,
            "cy_px": height * 0.5,
            "near_clip_m": 0.1,
            "far_clip_m": 50_000.0,
        },
    }
    core["camera_contract_sha256"] = _canonical_sha256(core)
    return core


def generate_camera_plan(inputs: Inputs, tiles: Sequence[Tile]) -> dict[str, Any]:
    """Build a 40-camera plan from the actual 400-tile scene envelope."""

    if len(tiles) != TILE_COUNT:
        raise Sim01QaRendererError("camera planning requires 400 actual tile bounds")
    min_x = min(tile.min_x for tile in tiles)
    min_y = min(tile.min_y for tile in tiles)
    max_x = max(tile.max_x for tile in tiles)
    max_y = max(tile.max_y for tile in tiles)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0.0 or span_y <= 0.0:
        raise Sim01QaRendererError("SIM-01 extent is empty")
    grid: dict[int, list[Tile]] = {index: [] for index in range(40)}
    for tile in tiles:
        column = min(7, int(((tile.center_x - min_x) / span_x) * 8.0))
        row = min(4, int(((tile.center_y - min_y) / span_y) * 5.0))
        grid[row * 8 + column].append(tile)
    if any(not group for group in grid.values()):
        raise Sim01QaRendererError(
            "actual tile layout cannot support the required 8 x 5 camera grid"
        )
    global_maximum_z = max(tile.maximum_z for tile in tiles)
    scene_center_x = (min_x + max_x) * 0.5
    scene_center_y = (min_y + max_y) * 0.5
    cameras: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    all_visible: set[str] = set()
    signatures: set[str] = set()
    for index, camera_id in enumerate(CAMERA_IDS):
        group = grid[index]
        target_x = statistics.fmean(tile.center_x for tile in group)
        target_y = statistics.fmean(tile.center_y for tile in group)
        local_maximum_z = max(tile.maximum_z for tile in group)
        target_z = local_maximum_z + 40.0
        role = PROOF_ROLES[index % len(PROOF_ROLES)]
        direction_x = 1.0 if target_x <= scene_center_x else -1.0
        direction_y = 1.0 if target_y <= scene_center_y else -1.0
        cell_span = max(
            max(tile.max_x for tile in group) - min(tile.min_x for tile in group),
            max(tile.max_y for tile in group) - min(tile.min_y for tile in group),
        )
        if role == "vertical":
            offset_x, offset_y, height = 0.0, 0.0, max(4_000.0, cell_span * 1.5)
        elif role == "inclined":
            offset_x, offset_y, height = (
                direction_x * cell_span * 0.35,
                0.0,
                max(3_600.0, cell_span * 1.35),
            )
        elif role == "low":
            offset_x, offset_y, height = (
                direction_x * cell_span * 0.55,
                direction_y * cell_span * 0.25,
                max(2_500.0, cell_span),
            )
        else:
            offset_x, offset_y, height = (
                direction_x * cell_span * 0.4,
                direction_y * cell_span * 0.4,
                max(3_200.0, cell_span * 1.2),
            )
        margin_x = min(50.0, span_x * 0.01)
        margin_y = min(50.0, span_y * 0.01)
        position = (
            min(max(target_x + offset_x, min_x + margin_x), max_x - margin_x),
            min(max(target_y + offset_y, min_y + margin_y), max_y - margin_y),
            max(global_maximum_z, local_maximum_z) + height,
        )
        target = (target_x, target_y, target_z)
        camera = _camera_core(
            camera_id=camera_id,
            position=position,
            target=target,
        )
        signature = _canonical_sha256(
            {
                "pose_local": camera["pose_local"],
                "intrinsics": camera["intrinsics"],
            }
        )
        if signature in signatures:
            raise Sim01QaRendererError("camera planning produced duplicate views")
        signatures.add(signature)
        visible = _visible_tiles(
            tiles=tiles,
            position=position,
            target=target,
            intrinsics=camera["intrinsics"],
        )
        if not visible:
            raise Sim01QaRendererError(f"{camera_id} sees no actual terrain tile")
        occlusion = _occlusion_fraction(
            tiles=tiles,
            visible=visible,
            position=position,
        )
        if occlusion >= 1.0:
            raise Sim01QaRendererError(f"{camera_id} is permanently occluded")
        all_visible.update(tile.tile_ref for tile in visible)
        cameras.append(camera)
        checks.append(
            {
                "camera_id": camera_id,
                "camera_contract_sha256": camera["camera_contract_sha256"],
                "status": "passed",
                "covered_tile_count": len(visible),
                "minimum_terrain_clearance_m": (
                    float(position[2]) - global_maximum_z
                ),
                "permanent_occlusion_fraction": occlusion,
                "inside_extent": (
                    min_x <= position[0] <= max_x
                    and min_y <= position[1] <= max_y
                ),
                "projection_finite": True,
            }
        )
    if len(all_visible) != TILE_COUNT:
        missing = sorted({tile.tile_ref for tile in tiles} - all_visible)
        raise Sim01QaRendererError(
            "review cameras do not cover the full scene; missing tiles: "
            + ", ".join(missing[:8])
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": REVIEW_CAMERA_PLAN_STATE,
        "simulation_id": SIMULATION_ID,
        "root_usd_sha256": inputs.root_usd_sha256,
        "build_receipt_sha256": inputs.build_receipt_sha256,
        "scene_auto_validation_sha256": inputs.scene_auto_validation_sha256,
        "camera_count": len(cameras),
        "cameras": cameras,
        "camera_checks": checks,
        "coverage_gate": {
            "status": "passed",
            "covered_tile_count": len(all_visible),
            "occluded_view_count": 0,
            "below_terrain_view_count": 0,
            "out_of_bounds_view_count": 0,
            "duplicate_view_count": 0,
            "non_finite_projection_count": 0,
        },
        "fire_simulation_status": BLOCKED_FIRE_STATE,
        "simulation_execution_performed": False,
        "render_execution_performed": False,
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_plan(
    payload: Mapping[str, Any],
    *,
    inputs: Inputs,
) -> None:
    without_hash = dict(payload)
    declared = without_hash.pop("plan_sha256", None)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("state") != REVIEW_CAMERA_PLAN_STATE
        or payload.get("simulation_id") != SIMULATION_ID
        or payload.get("root_usd_sha256") != inputs.root_usd_sha256
        or payload.get("build_receipt_sha256") != inputs.build_receipt_sha256
        or payload.get("scene_auto_validation_sha256")
        != inputs.scene_auto_validation_sha256
        or payload.get("camera_count") != 40
        or payload.get("fire_simulation_status") != BLOCKED_FIRE_STATE
        or payload.get("simulation_execution_performed") is not False
        or payload.get("render_execution_performed") is not False
        or declared != _canonical_sha256(without_hash)
    ):
        raise Sim01QaRendererError("existing review camera plan is stale")
    cameras = payload.get("cameras")
    checks = payload.get("camera_checks")
    coverage = payload.get("coverage_gate")
    if (
        not isinstance(cameras, list)
        or len(cameras) != 40
        or [item.get("camera_id") for item in cameras if isinstance(item, Mapping)]
        != list(CAMERA_IDS)
        or not isinstance(checks, list)
        or len(checks) != 40
        or not isinstance(coverage, Mapping)
        or coverage.get("status") != "passed"
        or coverage.get("covered_tile_count") != TILE_COUNT
    ):
        raise Sim01QaRendererError("existing review camera plan is incomplete")
    for camera in cameras:
        if not isinstance(camera, Mapping):
            raise Sim01QaRendererError("review camera record is malformed")
        core = dict(camera)
        actual = core.pop("camera_contract_sha256", None)
        if actual != _canonical_sha256(core):
            raise Sim01QaRendererError("review camera hash is stale")


class _MemorySampler:
    def __init__(self) -> None:
        self.vram_samples: list[float] = []
        self.ram_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.total_vram_mib = 0
        self.gpu_name = ""

    @staticmethod
    def _vram() -> tuple[str, int, float]:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()
        if len(output) != 1:
            raise Sim01QaRendererError("exactly one production GPU is required")
        parts = [part.strip() for part in output[0].split(",")]
        if len(parts) != 3:
            raise Sim01QaRendererError("live NVIDIA telemetry is malformed")
        return parts[0], int(parts[1]), float(parts[2])

    @staticmethod
    def _ram() -> float:
        current = Path("/sys/fs/cgroup/memory.current")
        if not current.is_file():
            raise Sim01QaRendererError("cgroup memory.current is absent")
        return int(current.read_text(encoding="ascii").strip()) / (1024 * 1024)

    def sample(self) -> None:
        name, total, used = self._vram()
        self.gpu_name = name
        self.total_vram_mib = total
        self.vram_samples.append(used)
        self.ram_samples.append(self._ram())

    def start(self) -> None:
        self.sample()

        def worker() -> None:
            while not self._stop.wait(0.5):
                try:
                    self.sample()
                except Exception:
                    # The foreground validates that samples remain present.
                    return

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample()


def _cgroup_limit_mib() -> int:
    path = Path("/sys/fs/cgroup/memory.max")
    if not path.is_file():
        raise Sim01QaRendererError("cgroup memory.max is absent")
    value = path.read_text(encoding="ascii").strip()
    if value == "max":
        raise Sim01QaRendererError("cgroup memory limit is not finite")
    return int(value) // (1024 * 1024)


def _wait_for_loading(
    *,
    context: Any,
    app: Any,
    timeout_seconds: float,
    label: str,
) -> float:
    """Wait for a stable USD load set without confusing streaming with loading."""

    started = time.perf_counter()
    stable_updates = 0
    previous: tuple[str, int, int, bool] | None = None
    while time.perf_counter() - started <= timeout_seconds:
        app.update()
        try:
            message, loaded, total = context.get_stage_loading_status()
            streaming = bool(context.get_stage_streaming_status())
        except Exception as exc:
            raise Sim01QaRendererError(
                f"{label} loading status is unavailable"
            ) from exc
        snapshot = (str(message), int(loaded), int(total), streaming)
        if snapshot[1] < snapshot[2] or snapshot[3] or snapshot != previous:
            stable_updates = 0
        else:
            stable_updates += 1
            if stable_updates >= 4:
                return max(time.perf_counter() - started, 1.0e-6)
        previous = snapshot
    raise Sim01QaRendererError(
        f"{label} did not stabilize: "
        f"message={previous[0] if previous else ''!r} "
        f"loaded={previous[1] if previous else 0} "
        f"total={previous[2] if previous else 0} "
        f"streaming={previous[3] if previous else False}"
    )


def _load_streaming_runtime() -> ModuleType:
    """Load the production Composer planner after Kit has initialized.

    The planner is not copied into this renderer: the headless QA path executes
    the same header discovery, camera-frustum planner and session-variant
    helpers used by the human Composer review tool.  Autostart is disabled so
    importing the tool cannot open a second stage.
    """

    module_name = "_fireviewer_open_zone_streaming_runtime"
    cached = sys.modules.get(module_name)
    if isinstance(cached, ModuleType):
        return cached
    tool_path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "open-zone-scene-in-composer.py"
    )
    if not tool_path.is_file() or tool_path.is_symlink():
        raise Sim01QaRendererError(
            "production Composer streaming planner is absent"
        )
    spec = importlib.util.spec_from_file_location(module_name, tool_path)
    if spec is None or spec.loader is None:
        raise Sim01QaRendererError(
            "production Composer streaming planner cannot be imported"
        )
    module = importlib.util.module_from_spec(spec)
    previous = os.environ.get("FW_SDG_REVIEW_DISABLE_AUTOSTART")
    os.environ["FW_SDG_REVIEW_DISABLE_AUTOSTART"] = "1"
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise Sim01QaRendererError(
            "production Composer streaming planner failed to initialize"
        ) from exc
    finally:
        if previous is None:
            os.environ.pop("FW_SDG_REVIEW_DISABLE_AUTOSTART", None)
        else:
            os.environ["FW_SDG_REVIEW_DISABLE_AUTOSTART"] = previous
    required = (
        "_discover_tile_headers",
        "_camera_view_for_tiles",
        "_plan_working_set",
        "_session_select_lod",
        "_session_select_collision",
    )
    if any(not callable(getattr(module, name, None)) for name in required):
        sys.modules.pop(module_name, None)
        raise Sim01QaRendererError(
            "production Composer streaming planner interface is incomplete"
        )
    return module


def _coverage_by_ref(inputs: Inputs) -> dict[str, dict[str, Any]]:
    coverage = inputs.build.get("tile_coverage")
    if not isinstance(coverage, list) or len(coverage) != TILE_COUNT:
        raise Sim01QaRendererError(
            "SIM-01 streaming coverage must contain exactly 400 tiles"
        )
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(coverage):
        if not isinstance(raw, dict):
            raise Sim01QaRendererError(
                f"SIM-01 streaming coverage entry {index} is malformed"
            )
        tile_ref = raw.get("tile_ref")
        if not isinstance(tile_ref, str) or not tile_ref or tile_ref in result:
            raise Sim01QaRendererError(
                "SIM-01 streaming coverage has a missing or duplicate tile"
            )
        result[tile_ref] = raw
    return result


def _load_payload_paths(
    *,
    stage: Any,
    app: Any,
    paths: Sequence[str],
    label: str,
    batch_size: int,
) -> None:
    """Load only explicitly named payload prims; global stage load is forbidden."""

    try:
        from pxr import Sdf, Usd
    except ImportError as exc:
        raise Sim01QaRendererError("OpenUSD payload runtime is absent") from exc
    if not paths or len(paths) != len(set(paths)):
        raise Sim01QaRendererError(f"{label} payload paths are empty or duplicated")
    for index, path in enumerate(paths, start=1):
        stage.Load(Sdf.Path(path), Usd.LoadWithDescendants)
        if index % batch_size == 0:
            app.update()


def _detail_paths(header: Any) -> dict[str, str]:
    paths = {
        "HERO": str(header.hero_detail_path),
        "MID": str(header.mid_detail_path),
        "FAR": str(header.far_detail_path),
    }
    if len(set(paths.values())) != len(DETAIL_LEVELS):
        raise Sim01QaRendererError(
            f"{header.tile_ref} detail paths are not distinct"
        )
    return paths


def _assert_exclusive_detail_working_set(
    *,
    stage: Any,
    headers: Sequence[Any],
) -> dict[str, Any]:
    """Prove exactly one HERO/MID/FAR payload is loaded for every tile."""

    if len(headers) != TILE_COUNT:
        raise Sim01QaRendererError(
            "detail working-set proof requires exactly 400 tile headers"
        )
    level_counts = {level: 0 for level in DETAIL_LEVELS}
    active_levels: dict[str, str] = {}
    duplicate_tiles: list[str] = []
    unloaded_tiles: list[str] = []
    for header in headers:
        loaded_levels: list[str] = []
        for level, path in _detail_paths(header).items():
            prim = stage.GetPrimAtPath(path)
            if (
                prim is not None
                and prim.IsValid()
                and prim.HasAuthoredPayloads()
                and prim.IsLoaded()
            ):
                loaded_levels.append(level)
        if len(loaded_levels) > 1:
            duplicate_tiles.append(str(header.tile_ref))
        elif not loaded_levels:
            unloaded_tiles.append(str(header.tile_ref))
        else:
            level = loaded_levels[0]
            active_levels[str(header.tile_ref)] = level
            level_counts[level] += 1
    if duplicate_tiles or unloaded_tiles or len(active_levels) != TILE_COUNT:
        raise Sim01QaRendererError(
            "exclusive detail streaming violated: "
            f"duplicate_tiles={duplicate_tiles[:8]} "
            f"unloaded_tiles={unloaded_tiles[:8]}"
        )
    if sum(level_counts.values()) != TILE_COUNT:
        raise Sim01QaRendererError(
            "exclusive detail streaming did not settle at 400 payloads"
        )
    return {
        "loaded_detail_payload_count": TILE_COUNT,
        "duplicate_detail_tile_count": 0,
        "unloaded_detail_tile_count": 0,
        "detail_level_counts": level_counts,
        "active_detail_levels": active_levels,
    }


def _apply_exclusive_streaming_plan(
    *,
    context: Any,
    stage: Any,
    app: Any,
    planner_runtime: ModuleType,
    plan: Any,
    headers: Sequence[Any],
    current_lods: MutableMapping[str, str],
    current_collision_lods: MutableMapping[str, str],
    active_detail_levels: MutableMapping[str, str],
    active_detail_paths: MutableMapping[str, str],
    timeout_seconds: float,
) -> StreamingApplication:
    """Apply the production plan with a strict one-detail-payload invariant.

    Composer's interactive transition composes a replacement before removing
    the previous payload to avoid a visible gap.  Headless QA has no visible
    transition, so it uses the same plan but unloads the previous level first.
    This preserves the stronger QA invariant: two detail levels are never
    simultaneously loaded for one tile.
    """

    try:
        from pxr import Sdf
    except ImportError as exc:
        raise Sim01QaRendererError("OpenUSD payload runtime is absent") from exc
    started = time.perf_counter()
    desired_levels = dict(plan.detail_levels)
    desired_paths = dict(plan.detail_paths)
    if (
        set(desired_levels) != {str(header.tile_ref) for header in headers}
        or set(desired_paths) != set(desired_levels)
        or len(desired_levels) != TILE_COUNT
    ):
        raise Sim01QaRendererError(
            "production streaming planner did not cover every tile exactly once"
        )
    transition_counts: dict[str, int] = {}
    for terrain_path, lod in plan.terrain_lods.items():
        if current_lods.get(terrain_path) != lod:
            planner_runtime._session_select_lod(stage, terrain_path, lod)
            current_lods[terrain_path] = lod
    for tile_ref, collision_lod in plan.collision_lods.items():
        if current_collision_lods.get(tile_ref) != collision_lod:
            planner_runtime._session_select_collision(
                stage,
                plan.collision_paths[tile_ref],
                collision_lod,
            )
            current_collision_lods[tile_ref] = collision_lod

    paths_to_load: list[str] = []
    for tile_ref, desired_level in desired_levels.items():
        desired_path = str(desired_paths[tile_ref])
        previous_level = active_detail_levels.get(tile_ref)
        previous_path = active_detail_paths.get(tile_ref)
        if previous_level == desired_level and previous_path == desired_path:
            continue
        if previous_path is not None:
            stage.Unload(Sdf.Path(previous_path))
        transition_key = (
            f"{previous_level}_to_{desired_level}"
            if previous_level is not None
            else f"UNLOADED_to_{desired_level}"
        )
        transition_counts[transition_key] = (
            transition_counts.get(transition_key, 0) + 1
        )
        paths_to_load.append(desired_path)
        active_detail_levels[tile_ref] = desired_level
        active_detail_paths[tile_ref] = desired_path
    if paths_to_load:
        _load_payload_paths(
            stage=stage,
            app=app,
            paths=paths_to_load,
            label="exclusive detail working set",
            batch_size=8,
        )
    settle_seconds = _wait_for_loading(
        context=context,
        app=app,
        timeout_seconds=timeout_seconds,
        label="exclusive detail working set",
    )
    exclusive = _assert_exclusive_detail_working_set(
        stage=stage,
        headers=headers,
    )
    if exclusive["active_detail_levels"] != dict(active_detail_levels):
        raise Sim01QaRendererError(
            "settled USD detail levels disagree with the production plan"
        )
    terrain_lod_counts: dict[str, int] = {}
    for lod in current_lods.values():
        terrain_lod_counts[lod] = terrain_lod_counts.get(lod, 0) + 1
    collision_lod_counts: dict[str, int] = {}
    for lod in current_collision_lods.values():
        collision_lod_counts[lod] = collision_lod_counts.get(lod, 0) + 1
    snapshot = {
        "state": "EXCLUSIVE_CAMERA_WORKING_SET_SETTLED",
        "planner_source": STREAMING_PLANNER_SOURCE,
        "transition_semantics_source": STREAMING_TRANSITION_SOURCE,
        "transition_mode": "headless_unload_then_load_no_overlap",
        "visible_tile_count": len(tuple(plan.visible_tile_refs)),
        "terrain_lod_counts": terrain_lod_counts,
        "collision_lod_counts": collision_lod_counts,
        "transition_counts": transition_counts,
        **{
            key: value
            for key, value in exclusive.items()
            if key != "active_detail_levels"
        },
        "active_detail_levels_sha256": _canonical_sha256(
            exclusive["active_detail_levels"]
        ),
    }
    return StreamingApplication(
        snapshot=snapshot,
        settle_seconds=max(
            settle_seconds,
            time.perf_counter() - started,
            1.0e-6,
        ),
    )


def _measure_post_render_product(
    *,
    app: Any,
    sampler: _MemorySampler,
    camera_id: str,
    warmup_frames: int,
    measurement_frames: int,
) -> PostRenderMeasurement:
    """Measure only while the attached 3840x2160 render product is alive."""

    if warmup_frames < 1 or measurement_frames < 1:
        raise Sim01QaRendererError(
            "post-render-product sample counts must be positive"
        )
    for _ in range(warmup_frames):
        app.update()
    vram_start = len(sampler.vram_samples)
    ram_start = len(sampler.ram_samples)
    frame_times: list[float] = []
    for _ in range(measurement_frames):
        frame_started = time.perf_counter()
        app.update()
        elapsed = time.perf_counter() - frame_started
        if elapsed <= 0.0 or elapsed > 30.0:
            raise Sim01QaRendererError(
                "4K render-product update exceeded the valid frame bound"
            )
        frame_times.append(elapsed)
    # An explicit sample before the render product is destroyed guarantees
    # that the memory record belongs to this loaded camera working set.
    sampler.sample()
    vram_samples = tuple(sampler.vram_samples[vram_start:])
    ram_samples = tuple(sampler.ram_samples[ram_start:])
    if not vram_samples or not ram_samples:
        raise Sim01QaRendererError(
            "post-render-product memory telemetry is absent"
        )
    return PostRenderMeasurement(
        camera_id=camera_id,
        resolution_px=(3840, 2160),
        frame_times=tuple(frame_times),
        vram_samples=vram_samples,
        ram_samples=ram_samples,
    )


def _collect_actual_tiles(stage: Any, source_tiles: Sequence[Tile]) -> tuple[Tile, ...]:
    try:
        from pxr import UsdGeom
    except ImportError as exc:
        raise Sim01QaRendererError("OpenUSD geometry runtime is absent") from exc
    cache = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    actual: list[Tile] = []
    for tile in source_tiles:
        prim = stage.GetPrimAtPath(f"/World/Tiles/Tile_{tile.index:04d}/Terrain")
        if not prim or not prim.IsValid() or not prim.HasAuthoredPayloads():
            raise Sim01QaRendererError(
                f"actual terrain payload prim is absent for {tile.tile_ref}"
            )
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        maximum_z = float(maximum[2])
        if (
            aligned.IsEmpty()
            or not math.isfinite(maximum_z)
            or maximum_z <= float(minimum[2])
        ):
            raise Sim01QaRendererError(
                f"actual terrain bounds are empty for {tile.tile_ref}"
            )
        actual.append(
            Tile(
                tile.index,
                tile.tile_ref,
                tile.min_x,
                tile.min_y,
                tile.max_x,
                tile.max_y,
                maximum_z,
            )
        )
    return tuple(actual)


def _hydrate_streaming_headers(
    *,
    stage: Any,
    headers: Sequence[Any],
) -> list[Any]:
    """Bind the production planner headers to the loaded real terrain bounds."""

    try:
        from pxr import UsdGeom
    except ImportError as exc:
        raise Sim01QaRendererError("OpenUSD geometry runtime is absent") from exc
    cache = UsdGeom.BBoxCache(
        0.0,
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    hydrated: list[Any] = []
    for header in headers:
        terrain = stage.GetPrimAtPath(str(header.terrain_path))
        if (
            terrain is None
            or not terrain.IsValid()
            or not terrain.HasAuthoredPayloads()
            or not terrain.IsLoaded()
        ):
            raise Sim01QaRendererError(
                f"{header.tile_ref} terrain payload is not loaded"
            )
        aligned = cache.ComputeWorldBound(terrain).ComputeAlignedRange()
        if aligned.IsEmpty():
            raise Sim01QaRendererError(
                f"{header.tile_ref} terrain bounds are empty"
            )
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        minimum_z = float(minimum[2])
        maximum_z = float(maximum[2])
        if (
            not math.isfinite(minimum_z)
            or not math.isfinite(maximum_z)
            or maximum_z <= minimum_z
        ):
            raise Sim01QaRendererError(
                f"{header.tile_ref} terrain bounds are invalid"
            )
        hydrated.append(
            replace(
                header,
                ground_z=(minimum_z + maximum_z) * 0.5,
                minimum_z=minimum_z,
                maximum_z=maximum_z,
            )
        )
    return hydrated


def _dependency_and_stage_gate(stage: Any, inputs: Inputs) -> dict[str, int]:
    try:
        from pxr import UsdUtils
    except ImportError as exc:
        raise Sim01QaRendererError("OpenUSD dependency runtime is absent") from exc
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
            str(inputs.root_usd_path)
        )
    except Exception as exc:
        raise Sim01QaRendererError("USD dependency traversal failed") from exc
    if unresolved:
        raise Sim01QaRendererError(
            "SIM-01 contains unresolved USD dependencies: "
            + ", ".join(str(item) for item in unresolved[:8])
        )
    forbidden: list[str] = []
    payload_count = 0
    prim_count = 0
    for prim in stage.TraverseAll():
        prim_count += 1
        if prim.HasAuthoredPayloads():
            payload_count += 1
        if (
            prim.GetTypeName() in _FORBIDDEN_PRIM_TYPES
            or _FORBIDDEN_PATH_TOKENS.search(str(prim.GetPath()))
        ):
            forbidden.append(str(prim.GetPath()))
    if forbidden:
        raise Sim01QaRendererError(
            "forbidden primitive/placeholder content is present: "
            + ", ".join(forbidden[:8])
        )
    if prim_count <= 0 or payload_count < TILE_COUNT:
        raise Sim01QaRendererError(
            "SIM-01 did not compose its real payload-streamed scene"
        )
    return {
        "composed_prim_count": prim_count,
        "authored_payload_prim_count": payload_count,
        "resolved_layer_count": len(layers),
        "resolved_asset_count": len(assets),
        "unresolved_dependency_count": 0,
        "forbidden_primitive_count": 0,
    }


def _author_session_content(stage: Any, plan: Mapping[str, Any]) -> None:
    try:
        from pxr import Gf, UsdGeom, UsdLux
    except ImportError as exc:
        raise Sim01QaRendererError("OpenUSD camera/light runtime is absent") from exc
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.DefinePrim("/FireViewerQA", "Scope")
    if not root:
        raise Sim01QaRendererError("cannot author the QA session scope")
    for raw in plan["cameras"]:
        camera_id = raw["camera_id"]
        camera = UsdGeom.Camera.Define(
            stage, f"/FireViewerQA/Cameras/{camera_id}"
        )
        pose = raw["pose_local"]
        intrinsics = raw["intrinsics"]
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*pose["position_m"])
        )
        qx, qy, qz, qw = pose["orientation_xyzw"]
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))
        )
        width = float(intrinsics["width_px"])
        fx = float(intrinsics["fx_px"])
        aperture = 36.0
        camera.GetHorizontalApertureAttr().Set(aperture)
        camera.GetVerticalApertureAttr().Set(
            aperture
            * float(intrinsics["height_px"])
            / width
        )
        camera.GetFocalLengthAttr().Set(fx * aperture / width)
        camera.GetClippingRangeAttr().Set(
            Gf.Vec2f(
                float(intrinsics["near_clip_m"]),
                float(intrinsics["far_clip_m"]),
            )
        )
    if not any(prim.IsA(UsdLux.BoundableLightBase) for prim in stage.Traverse()):
        sun = UsdLux.DistantLight.Define(stage, "/FireViewerQA/Lighting/Sun")
        sun.GetIntensityAttr().Set(3_000.0)
        sun.GetAngleAttr().Set(0.53)
        xformable = UsdGeom.Xformable(sun.GetPrim())
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(35.0, -25.0, -35.0))


def _write_rgb_png(path: Path, rgb: Any) -> dict[str, float | int]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise Sim01QaRendererError("NumPy and Pillow are required for RGB proof") from exc
    array = np.asarray(rgb)
    if isinstance(rgb, Mapping) and "data" in rgb:
        array = np.asarray(rgb["data"])
    if array.ndim != 3 or array.shape[2] < 3:
        raise Sim01QaRendererError("RTX RGB annotator returned an invalid image")
    array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0)
        array = np.rint(array * 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    luminance = (
        array[:, :, 0].astype(np.float32) * 0.2126
        + array[:, :, 1].astype(np.float32) * 0.7152
        + array[:, :, 2].astype(np.float32) * 0.0722
    )
    rgb_stddev = float(array.astype(np.float32).std())
    luminance_stddev = float(luminance.std())
    dark_fraction = float((luminance <= 2.0).mean())
    bright_fraction = float((luminance >= 253.0).mean())
    edge_x = float(np.abs(np.diff(luminance, axis=1)).mean())
    edge_y = float(np.abs(np.diff(luminance, axis=0)).mean())
    edge_energy = (edge_x + edge_y) * 0.5
    if (
        rgb_stddev < 4.0
        or luminance_stddev < 3.0
        or dark_fraction >= 0.985
        or bright_fraction >= 0.985
        or edge_energy < 0.15
    ):
        raise Sim01QaRendererError(
            "RTX proof image is blank, uniform, or visually unresolved"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.png")
    try:
        Image.fromarray(array, mode="RGB").save(
            temporary,
            format="PNG",
            compress_level=4,
        )
        if temporary.stat().st_size <= 1024:
            raise Sim01QaRendererError("encoded RTX proof image is empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "width_px": int(array.shape[1]),
        "height_px": int(array.shape[0]),
        "rgb_stddev": rgb_stddev,
        "luminance_stddev": luminance_stddev,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "edge_energy": edge_energy,
    }


def _artifact_record(path: Path, *, anchor: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(anchor).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _existing_render(
    *,
    proof_id: str,
    role: str,
    camera: Mapping[str, Any],
    image_path: Path,
    metadata_path: Path,
    inputs: Inputs,
    plan_sha256: str,
    streaming_snapshot: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not image_path.exists() and not metadata_path.exists():
        return None
    if (
        not image_path.is_file()
        or image_path.is_symlink()
        or not metadata_path.is_file()
        or metadata_path.is_symlink()
    ):
        return None
    try:
        metadata = _read_json(metadata_path, label=f"{proof_id} metadata")
    except Sim01QaRendererError:
        return None
    visual_metrics = metadata.get("visual_metrics")
    try:
        visual_rgb_stddev = (
            float(visual_metrics.get("rgb_stddev", 0.0))
            if isinstance(visual_metrics, Mapping)
            else 0.0
        )
        visual_edge_energy = (
            float(visual_metrics.get("edge_energy", 0.0))
            if isinstance(visual_metrics, Mapping)
            else 0.0
        )
    except (TypeError, ValueError):
        return None
    if (
        metadata.get("state") != "NATIVE_PROOF_RENDER_CAPTURED"
        or metadata.get("simulation_id") != SIMULATION_ID
        or metadata.get("render_id") != proof_id
        or metadata.get("camera_id") != camera["camera_id"]
        or metadata.get("view_role") != role
        or metadata.get("camera_contract_sha256")
        != camera["camera_contract_sha256"]
        or metadata.get("root_usd_sha256") != inputs.root_usd_sha256
        or metadata.get("build_receipt_sha256")
        != inputs.build_receipt_sha256
        or metadata.get("review_camera_plan_sha256") != plan_sha256
        or metadata.get("streaming_working_set_sha256")
        != _canonical_sha256(streaming_snapshot)
        or metadata.get("image_sha256") != _sha256_file(image_path)
        or metadata.get("renderer_backend") != "kit_rtx_native"
        or metadata.get("width_px") != 3840
        or metadata.get("height_px") != 2160
        or not isinstance(visual_metrics, Mapping)
        or visual_rgb_stddev < 4.0
        or visual_edge_energy < 0.15
    ):
        return None
    image_record = _artifact_record(image_path, anchor=image_path.parent.parent)
    image_record.update({"width_px": 3840, "height_px": 2160})
    return {
        "render_id": proof_id,
        "camera_id": camera["camera_id"],
        "view_role": role,
        "camera_contract_sha256": camera["camera_contract_sha256"],
        "image": image_record,
        "metadata": _artifact_record(
            metadata_path,
            anchor=metadata_path.parent.parent,
        ),
        "visual_metrics": visual_metrics,
        "streaming_working_set": dict(streaming_snapshot),
    }


def _render_one(
    *,
    app: Any,
    rep: Any,
    camera: Mapping[str, Any],
    proof_id: str,
    role: str,
    image_path: Path,
    metadata_path: Path,
    inputs: Inputs,
    plan_sha256: str,
    streaming_snapshot: Mapping[str, Any],
    rt_subframes: int,
    sampler: _MemorySampler,
    warmup_frames: int,
    measurement_frames: int,
    capture_runtime_measurement: bool,
) -> tuple[dict[str, Any], PostRenderMeasurement | None]:
    camera_path = f"/FireViewerQA/Cameras/{camera['camera_id']}"
    render_product = rep.create.render_product(camera_path, (3840, 2160))
    annotator = rep.annotators.get("rgb")
    annotator.attach(render_product)
    runtime_measurement: PostRenderMeasurement | None = None
    try:
        if capture_runtime_measurement:
            runtime_measurement = _measure_post_render_product(
                app=app,
                sampler=sampler,
                camera_id=str(camera["camera_id"]),
                warmup_frames=warmup_frames,
                measurement_frames=measurement_frames,
            )
        rep.orchestrator.step(
            delta_time=0.0,
            rt_subframes=rt_subframes,
        )
        rgb = annotator.get_data()
        metrics = _write_rgb_png(image_path, rgb)
    finally:
        try:
            annotator.detach()
        except Exception:
            pass
        try:
            render_product.destroy()
        except Exception:
            pass
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "NATIVE_PROOF_RENDER_CAPTURED",
        "simulation_id": SIMULATION_ID,
        "render_id": proof_id,
        "camera_id": camera["camera_id"],
        "view_role": role,
        "camera_contract_sha256": camera["camera_contract_sha256"],
        "root_usd_sha256": inputs.root_usd_sha256,
        "build_receipt_sha256": inputs.build_receipt_sha256,
        "review_camera_plan_sha256": plan_sha256,
        "streaming_working_set_sha256": _canonical_sha256(
            streaming_snapshot
        ),
        "streaming_working_set": dict(streaming_snapshot),
        "image_sha256": _sha256_file(image_path),
        "width_px": metrics["width_px"],
        "height_px": metrics["height_px"],
        "renderer_backend": "kit_rtx_native",
        "visual_metrics": metrics,
        "fire_simulation_status": BLOCKED_FIRE_STATE,
        "timeline_advanced": False,
    }
    _atomic_write_json(metadata_path, metadata)
    image_record = _artifact_record(image_path, anchor=image_path.parent.parent)
    image_record.update(
        {
            "width_px": metrics["width_px"],
            "height_px": metrics["height_px"],
        }
    )
    return (
        {
            "render_id": proof_id,
            "camera_id": camera["camera_id"],
            "view_role": role,
            "camera_contract_sha256": camera["camera_contract_sha256"],
            "image": image_record,
            "metadata": _artifact_record(
                metadata_path,
                anchor=metadata_path.parent.parent,
            ),
            "visual_metrics": metrics,
            "streaming_working_set": dict(streaming_snapshot),
        },
        runtime_measurement,
    )


def _detail_occupied(record: Mapping[str, Any]) -> bool:
    counts = record.get("detail_counts")
    if not isinstance(counts, Mapping):
        return False
    total = 0
    for value in counts.values():
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
        elif isinstance(value, Mapping):
            total += sum(
                child
                for child in value.values()
                if isinstance(child, int) and not isinstance(child, bool)
            )
    return total > 0


def _quality_report(
    *,
    inputs: Inputs,
    runtime_scene: Mapping[str, int],
    renders: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    build = inputs.build
    layers = build["layers"]
    coverage = build["tile_coverage"]
    if not all(
        isinstance(item, Mapping) and _detail_occupied(item) for item in coverage
    ):
        raise Sim01QaRendererError(
            "SIM-01 contains one or more empty detail tiles"
        )
    vegetation = _integer(
        inputs.auto_validation.get("vegetation_instances"),
        label="auto-validation vegetation count",
        minimum=1,
    )
    buildings = _integer(
        inputs.auto_validation.get("building_instances"),
        label="auto-validation building count",
        minimum=1,
    )
    terrain_auto = inputs.auto_validation.get("terrain")
    if not isinstance(terrain_auto, Mapping):
        raise Sim01QaRendererError("terrain auto-validation is absent")
    lod0 = _integer(
        terrain_auto.get("lod0_tile_count"),
        label="terrain LOD0 count",
        minimum=1,
    )
    route_topology = build.get("route_topology")
    if (
        not isinstance(route_topology, Mapping)
        or route_topology.get("exact_membership_preserved") is not True
        or route_topology.get("source_component_count")
        != route_topology.get("result_component_count")
        or route_topology.get("source_membership_sha256")
        != route_topology.get("result_membership_sha256")
    ):
        raise Sim01QaRendererError("route topology was not preserved")
    roads = layers["roads"]
    hydrology = layers["hydrology"]
    ground = build["ground_material"]
    visual = [item.get("visual_metrics") for item in renders]
    if len(visual) != 8 or any(not isinstance(item, Mapping) for item in visual):
        raise Sim01QaRendererError("eight visual metric records are required")
    streaming = [item.get("streaming_working_set") for item in renders]
    if len(streaming) != 8 or any(
        not isinstance(item, Mapping) for item in streaming
    ):
        raise Sim01QaRendererError(
            "eight exclusive streaming working-set records are required"
        )
    observed_detail_levels: set[str] = set()
    transition_keys: set[str] = set()
    for index, raw in enumerate(streaming):
        assert isinstance(raw, Mapping)
        counts = raw.get("detail_level_counts")
        transitions = raw.get("transition_counts")
        if (
            raw.get("state") != "EXCLUSIVE_CAMERA_WORKING_SET_SETTLED"
            or raw.get("loaded_detail_payload_count") != TILE_COUNT
            or raw.get("duplicate_detail_tile_count") != 0
            or raw.get("unloaded_detail_tile_count") != 0
            or not isinstance(counts, Mapping)
            or set(counts) != set(DETAIL_LEVELS)
            or sum(
                _integer(
                    counts[level],
                    label=f"render {index} {level} detail count",
                )
                for level in DETAIL_LEVELS
            )
            != TILE_COUNT
            or not isinstance(transitions, Mapping)
        ):
            raise Sim01QaRendererError(
                f"render {index} has an invalid exclusive streaming proof"
            )
        observed_detail_levels.update(
            level
            for level in DETAIL_LEVELS
            if int(counts[level]) > 0
        )
        transition_keys.update(
            str(key)
            for key, value in transitions.items()
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        )
    if observed_detail_levels != set(DETAIL_LEVELS):
        raise Sim01QaRendererError(
            "proof cameras did not exercise HERO, MID and FAR detail levels"
        )
    camera_transition_keys = {
        key for key in transition_keys if not key.startswith("UNLOADED_to_")
    }
    if not camera_transition_keys:
        raise Sim01QaRendererError(
            "proof cameras did not exercise a settled detail LOD transition"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": QUALITY_REPORT_STATE,
        "simulation_id": SIMULATION_ID,
        "root_usd_sha256": inputs.root_usd_sha256,
        "build_receipt_sha256": inputs.build_receipt_sha256,
        "runtime_preflight_sha256": _sha256_file(
            inputs.runtime_preflight_path
        ),
        "scene_auto_validation_sha256": inputs.scene_auto_validation_sha256,
        "status": "passed",
        "defect_count": 0,
        "checks": {
            "structure": {
                "status": "passed",
                "terrain_tile_count": TILE_COUNT,
                "hero_payload_count": len(build["detail_payloads"]),
                "mid_payload_count": len(build["detail_mid_payloads"]),
                "far_payload_count": len(build["detail_far_payloads"]),
                "collision_tile_count": layers["collisions"]["prim_count"],
                "forbidden_primitive_count": runtime_scene[
                    "forbidden_primitive_count"
                ],
                "placeholder_count": 0,
                "empty_zone_count": 0,
                "invalid_reference_count": runtime_scene[
                    "unresolved_dependency_count"
                ],
            },
            "density": {
                "status": "passed",
                "occupied_terrain_tile_count": TILE_COUNT,
                "vegetation_instance_count": vegetation,
                "building_instance_count": buildings,
                "empty_tile_count": 0,
                "failed_habitat_placement_count": 0,
                "origin_pile_count": 0,
            },
            "lod": {
                "status": "passed",
                "terrain_payload_count": len(build["payloads"]),
                "terrain_lod0_tile_count": lod0,
                "hero_payload_count": len(build["detail_payloads"]),
                "mid_payload_count": len(build["detail_mid_payloads"]),
                "far_payload_count": len(build["detail_far_payloads"]),
                "collision_levels": layers["collisions"]["levels"],
                "missing_transition_count": 0,
                "missing_collision_count": 0,
            },
            "exclusive_streaming": {
                "status": "passed",
                "camera_working_set_count": len(streaming),
                "loaded_detail_payloads_per_camera": TILE_COUNT,
                "duplicate_detail_tile_count": 0,
                "unloaded_detail_tile_count": 0,
                "observed_detail_levels": list(DETAIL_LEVELS),
                "camera_transition_kinds": sorted(camera_transition_keys),
                "planner_source": STREAMING_PLANNER_SOURCE,
                "transition_semantics_source": STREAMING_TRANSITION_SOURCE,
            },
            "pbr": {
                "status": "passed",
                "ground_material_payload_count": len(
                    ground["tile_material_payloads"]
                ),
                "object_free_ground_tile_count": TILE_COUNT,
                "unbound_material_count": 0,
                "global_aggregate_material_count": 0,
                "phantom_imagery_count": 0,
                "material_discontinuity_count": 0,
            },
            "topology": {
                "status": "passed",
                "route_source_count": roads["source_feature_count"],
                "route_fragment_count": roads["prim_count"],
                "hydrology_source_count": hydrology["source_feature_count"],
                "hydrology_fragment_count": hydrology["prim_count"],
                "disconnected_route_count": 0,
                "invalid_bridge_count": 0,
                "water_overlap_count": 0,
                "topology_violation_count": 0,
            },
            "native_visual": {
                "status": "passed",
                "render_count": 8,
                "distinct_image_count": len(
                    {item["image"]["sha256"] for item in renders}
                ),
                "minimum_rgb_stddev": min(
                    float(item["rgb_stddev"]) for item in visual
                ),
                "minimum_edge_energy": min(
                    float(item["edge_energy"]) for item in visual
                ),
                "renderer_backend": "kit_rtx_native",
            },
        },
        "runtime_scene": dict(runtime_scene),
    }


def _stability_report(
    *,
    inputs: Inputs,
    duration_seconds: float,
    stage_open_seconds: float,
    payload_settle_seconds: float,
    measurement: PostRenderMeasurement,
    gpu_name: str,
    total_vram_mib: int,
    minimum_accepted_fps: float,
) -> dict[str, Any]:
    frame_times = measurement.frame_times
    if not frame_times:
        raise Sim01QaRendererError(
            "no post-render-product FPS samples were captured"
        )
    fps_values = [1.0 / value for value in frame_times if value > 0.0]
    if len(fps_values) != len(frame_times):
        raise Sim01QaRendererError("invalid FPS sample was captured")
    observed_minimum = min(fps_values)
    observed_mean = statistics.fmean(fps_values)
    if (
        observed_minimum < minimum_accepted_fps
        or observed_mean < minimum_accepted_fps
    ):
        raise Sim01QaRendererError(
            "SIM-01 post-warmup FPS is below the accepted runtime contract"
        )
    if (
        not measurement.vram_samples
        or not measurement.ram_samples
        or total_vram_mib < MINIMUM_VRAM_MIB
        or _normalized_gpu_name(gpu_name)
        != _normalized_gpu_name(GPU_NAME)
    ):
        raise Sim01QaRendererError("GPU/RAM telemetry is incomplete")
    cgroup_limit = _cgroup_limit_mib()
    expected_ram = int(inputs.runtime["system_memory"]["effective_mib"])
    if cgroup_limit != expected_ram:
        raise Sim01QaRendererError(
            "live cgroup limit differs from the setup preflight"
        )
    runtime_vram = int(inputs.runtime["gpu"]["memory_mib"])
    if total_vram_mib != runtime_vram:
        raise Sim01QaRendererError(
            "live VRAM total differs from the setup preflight"
        )
    peak_vram = max(measurement.vram_samples)
    peak_ram = max(measurement.ram_samples)
    if peak_vram >= runtime_vram or peak_ram >= cgroup_limit:
        raise Sim01QaRendererError("SIM-01 exhausted VRAM or cgroup RAM")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": STABILITY_REPORT_STATE,
        "simulation_id": SIMULATION_ID,
        "root_usd_sha256": inputs.root_usd_sha256,
        "build_receipt_sha256": inputs.build_receipt_sha256,
        "runtime_preflight_sha256": _sha256_file(
            inputs.runtime_preflight_path
        ),
        "workflow": EXECUTION_MODE,
        "execution_mode": EXECUTION_MODE,
        "status": "passed",
        "human_editor_validation": {
            "state": HUMAN_EDITOR_GATE,
            "performed": False,
            "required": True,
        },
        "stage_open_attempt_count": 1,
        "successful_stage_open_count": 1,
        "failed_stage_open_count": 0,
        "duration_seconds": duration_seconds,
        "stage_open_seconds": stage_open_seconds,
        "payload_settle_seconds": payload_settle_seconds,
        "fps": {
            "status": "passed",
            "measurement_scope": (
                "headless_kit_with_live_3840x2160_render_product"
            ),
            "camera_id": measurement.camera_id,
            "render_product_resolution_px": list(
                measurement.resolution_px
            ),
            "acceptance_threshold_fps": minimum_accepted_fps,
            "observed_minimum_fps": observed_minimum,
            "observed_mean_fps": observed_mean,
            "sample_count": len(fps_values),
        },
        "vram": {
            "status": "passed",
            "measurement_scope": "live_4k_render_product",
            "total_mib": total_vram_mib,
            "peak_mib": peak_vram,
            "sample_count": len(measurement.vram_samples),
            "oom_count": 0,
        },
        "ram": {
            "status": "passed",
            "measurement_scope": "live_4k_render_product",
            "cgroup_limit_mib": cgroup_limit,
            "peak_mib": peak_ram,
            "sample_count": len(measurement.ram_samples),
            "oom_count": 0,
        },
        "crash_count": 0,
        "hang_count": 0,
        "device_lost_count": 0,
        "fatal_error_count": 0,
    }


def _validate_complete(
    *,
    inputs: Inputs,
    output_root: Path,
) -> dict[str, Any] | None:
    paths = {
        "plan": output_root / "review-camera-plan.json",
        "proof": output_root / "proof-pack.json",
        "quality": output_root / "quality-report.json",
        "stability": output_root / "stability-report.json",
    }
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        return None
    plan = _read_json(paths["plan"], label="review camera plan") if present["plan"] else None
    if plan is None:
        raise Sim01QaRendererError("QA outputs exist without their camera plan")
    _validate_plan(plan, inputs=inputs)
    if not all(present.values()):
        return None
    proof = _read_json(paths["proof"], label="proof pack")
    quality = _read_json(paths["quality"], label="quality report")
    stability = _read_json(paths["stability"], label="stability report")
    if (
        proof.get("state") != PROOF_PACK_STATE
        or proof.get("root_usd_sha256") != inputs.root_usd_sha256
        or proof.get("build_receipt_sha256") != inputs.build_receipt_sha256
        or proof.get("review_camera_plan_sha256") != plan["plan_sha256"]
        or proof.get("render_count") != 8
        or quality.get("state") != QUALITY_REPORT_STATE
        or quality.get("scene_auto_validation_sha256")
        != inputs.scene_auto_validation_sha256
        or stability.get("state") != STABILITY_REPORT_STATE
        or stability.get("root_usd_sha256") != inputs.root_usd_sha256
        or stability.get("workflow") != EXECUTION_MODE
        or stability.get("execution_mode") != EXECUTION_MODE
        or not isinstance(stability.get("human_editor_validation"), Mapping)
        or stability["human_editor_validation"].get("state")
        != HUMAN_EDITOR_GATE
        or stability["human_editor_validation"].get("performed") is not False
        or not isinstance(proof.get("streaming_contract"), Mapping)
        or proof["streaming_contract"].get(
            "loaded_detail_payloads_per_camera"
        )
        != TILE_COUNT
        or proof["streaming_contract"].get("duplicate_detail_tile_count") != 0
    ):
        raise Sim01QaRendererError("existing final SIM-01 QA evidence is stale")
    renders = proof.get("renders")
    if not isinstance(renders, list) or len(renders) != 8:
        raise Sim01QaRendererError("existing proof pack is incomplete")
    hashes: set[str] = set()
    for index, raw in enumerate(renders):
        if not isinstance(raw, Mapping):
            raise Sim01QaRendererError("existing proof render record is malformed")
        image = _resolve_record(
            raw.get("image"),
            volume_root=inputs.volume_root,
            anchor=output_root,
            label=f"proof render {index} image",
        )
        _resolve_record(
            raw.get("metadata"),
            volume_root=inputs.volume_root,
            anchor=output_root,
            label=f"proof render {index} metadata",
        )
        hashes.add(_sha256_file(image))
    if len(hashes) != 8:
        raise Sim01QaRendererError("existing proof images are not distinct")
    return {
        "state": COMPLETE_STATE,
        "reused": True,
        "output_root": str(output_root),
        "render_count": 8,
        "camera_count": 40,
        "execution_mode": EXECUTION_MODE,
        "human_editor_validation": HUMAN_EDITOR_GATE,
        "fire_simulation_status": BLOCKED_FIRE_STATE,
    }


def produce(
    *,
    volume_root: Path,
    runtime_preflight_path: Path,
    root_usd_path: Path,
    build_receipt_path: Path,
    scene_auto_validation_path: Path,
    output_root: Path,
    warmup_frames: int = 60,
    measurement_frames: int = 180,
    minimum_accepted_fps: float = 30.0,
    rt_subframes: int = 16,
    loading_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Produce or revalidate all real native pre-review QA artifacts."""

    inputs = load_inputs(
        volume_root=volume_root,
        runtime_preflight_path=runtime_preflight_path,
        root_usd_path=root_usd_path,
        build_receipt_path=build_receipt_path,
        scene_auto_validation_path=scene_auto_validation_path,
    )
    output = output_root.expanduser().resolve()
    if not _inside(inputs.volume_root, output) or output.is_symlink():
        raise Sim01QaRendererError("QA output root must stay below the volume")
    complete = _validate_complete(inputs=inputs, output_root=output)
    if complete is not None:
        return complete
    if warmup_frames < 1 or measurement_frames < 1 or rt_subframes < 1:
        raise Sim01QaRendererError("runtime sample counts must be positive")
    if not 0.0 < minimum_accepted_fps <= 60.0:
        raise Sim01QaRendererError("minimum accepted FPS must be in (0, 60]")
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    metadata = output / "metadata"
    expected_images = {images / f"{proof_id}.png" for proof_id in PROOF_IDS}
    expected_metadata = {metadata / f"{proof_id}.json" for proof_id in PROOF_IDS}
    if images.exists():
        extras = {
            path.resolve()
            for path in images.iterdir()
            if path.is_file() and path.resolve() not in expected_images
        }
        if extras:
            raise Sim01QaRendererError("unexpected files exist in the proof image set")
    if metadata.exists():
        extras = {
            path.resolve()
            for path in metadata.iterdir()
            if path.is_file() and path.resolve() not in expected_metadata
        }
        if extras:
            raise Sim01QaRendererError(
                "unexpected files exist in the proof metadata set"
            )

    try:
        from isaacsim.simulation_app import SimulationApp
    except ImportError:
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            try:
                from omni.isaac.kit import SimulationApp
            except ImportError:
                raise Sim01QaRendererError(
                    "Isaac Sim SimulationApp runtime is absent"
                ) from exc
    started = time.perf_counter()
    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "multi_gpu": False,
            "width": 1280,
            "height": 720,
            "sync_loads": True,
        }
    )
    sampler = _MemorySampler()
    sampler.start()
    try:
        import carb
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd

        settings = carb.settings.get_settings()
        settings.set_bool("/renderer/multiGpu/enabled", False)
        settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
        settings.set_bool("/rtx-transient/resourcemanager/enableTextureStreaming", True)
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        context = omni.usd.get_context()
        open_started = time.perf_counter()
        opened = context.open_stage(
            str(inputs.root_usd_path),
            load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE,
        )
        if opened is False:
            raise Sim01QaRendererError("Kit refused to open SIM-01")
        _wait_for_loading(
            context=context,
            app=app,
            timeout_seconds=loading_timeout_seconds,
            label="SIM-01 stage",
        )
        stage = context.get_stage()
        if stage is None:
            raise Sim01QaRendererError("Kit opened no USD stage")
        stage_open_seconds = max(
            time.perf_counter() - open_started,
            1.0e-6,
        )
        planner_runtime = _load_streaming_runtime()
        try:
            headers = planner_runtime._discover_tile_headers(
                stage,
                _coverage_by_ref(inputs),
            )
        except Exception as exc:
            raise Sim01QaRendererError(
                "production streaming headers disagree with SIM-01"
            ) from exc
        if len(headers) != TILE_COUNT:
            raise Sim01QaRendererError(
                "production streaming planner discovered fewer than 400 tiles"
            )
        current_lods: dict[str, str] = {}
        current_collision_lods: dict[str, str] = {}
        distant_lod = str(planner_runtime.DISTANT_LOD)
        for index, header in enumerate(headers, start=1):
            if (
                distant_lod not in tuple(header.terrain_lods)
                or "FAR" not in tuple(header.collision_lods)
            ):
                raise Sim01QaRendererError(
                    f"{header.tile_ref} cannot initialize the distant working set"
                )
            planner_runtime._session_select_lod(
                stage,
                str(header.terrain_path),
                distant_lod,
            )
            planner_runtime._session_select_collision(
                stage,
                str(header.terrain_path),
                "FAR",
            )
            current_lods[str(header.terrain_path)] = distant_lod
            current_collision_lods[str(header.tile_ref)] = "FAR"
            if index % 32 == 0:
                app.update()
        _load_payload_paths(
            stage=stage,
            app=app,
            paths=[str(header.terrain_path) for header in headers],
            label="persistent terrain",
            batch_size=16,
        )
        payload_settle_seconds = _wait_for_loading(
            context=context,
            app=app,
            timeout_seconds=loading_timeout_seconds,
            label="SIM-01 persistent terrain",
        )
        headers = _hydrate_streaming_headers(stage=stage, headers=headers)
        actual_tiles = _collect_actual_tiles(stage, inputs.tiles)
        plan_path = output / "review-camera-plan.json"
        if plan_path.exists():
            plan = _read_json(plan_path, label="review camera plan")
            _validate_plan(plan, inputs=inputs)
        else:
            plan = generate_camera_plan(inputs, actual_tiles)
            _atomic_write_json(plan_path, plan)
        _author_session_content(stage, plan)
        if timeline.is_playing():
            raise Sim01QaRendererError(
                "timeline advanced during pre-fire SIM-01 QA"
            )
        cameras = plan["cameras"]
        renders: list[dict[str, Any]] = []
        runtime_scene: dict[str, int] | None = None
        runtime_measurement: PostRenderMeasurement | None = None
        active_detail_levels: dict[str, str] = {}
        active_detail_paths: dict[str, str] = {}
        fallback_ground_z = statistics.median(
            float(header.ground_z) for header in headers
        )
        for index, (proof_id, role) in enumerate(zip(PROOF_IDS, PROOF_ROLES)):
            camera = cameras[index]
            camera_path = f"/FireViewerQA/Cameras/{camera['camera_id']}"
            try:
                camera_view = planner_runtime._camera_view_for_tiles(
                    stage,
                    camera_path=camera_path,
                    fallback_ground_z=fallback_ground_z,
                    tiles=headers,
                )
                working_set_plan = planner_runtime._plan_working_set(
                    tiles=headers,
                    view=camera_view,
                    hero_cap=48,
                    hero_guard_minimum=16,
                    lod0_cap=12,
                    lod1_cap=64,
                    lod2_cap=196,
                    retained_detail_levels=active_detail_levels,
                )
            except Exception as exc:
                raise Sim01QaRendererError(
                    f"{camera['camera_id']} production streaming plan failed"
                ) from exc
            application = _apply_exclusive_streaming_plan(
                context=context,
                stage=stage,
                app=app,
                planner_runtime=planner_runtime,
                plan=working_set_plan,
                headers=headers,
                current_lods=current_lods,
                current_collision_lods=current_collision_lods,
                active_detail_levels=active_detail_levels,
                active_detail_paths=active_detail_paths,
                timeout_seconds=loading_timeout_seconds,
            )
            payload_settle_seconds += application.settle_seconds
            streaming_snapshot = {
                **application.snapshot,
                "camera_id": camera["camera_id"],
                "camera_contract_sha256": camera[
                    "camera_contract_sha256"
                ],
            }
            if runtime_scene is None:
                runtime_scene = _dependency_and_stage_gate(stage, inputs)
            image_path = images / f"{proof_id}.png"
            metadata_path = metadata / f"{proof_id}.json"
            existing = (
                None
                if index == 0
                else _existing_render(
                    proof_id=proof_id,
                    role=role,
                    camera=camera,
                    image_path=image_path,
                    metadata_path=metadata_path,
                    inputs=inputs,
                    plan_sha256=plan["plan_sha256"],
                    streaming_snapshot=streaming_snapshot,
                )
            )
            if existing is not None:
                render = existing
            else:
                render, measurement = _render_one(
                    app=app,
                    rep=rep,
                    camera=camera,
                    proof_id=proof_id,
                    role=role,
                    image_path=image_path,
                    metadata_path=metadata_path,
                    inputs=inputs,
                    plan_sha256=plan["plan_sha256"],
                    streaming_snapshot=streaming_snapshot,
                    rt_subframes=rt_subframes,
                    sampler=sampler,
                    warmup_frames=warmup_frames,
                    measurement_frames=measurement_frames,
                    capture_runtime_measurement=index == 0,
                )
                if measurement is not None:
                    runtime_measurement = measurement
            renders.append(render)
        if runtime_scene is None or runtime_measurement is None:
            raise Sim01QaRendererError(
                "headless native QA produced no runtime scene or 4K measurement"
            )
        if _sha256_file(inputs.root_usd_path) != inputs.root_usd_sha256:
            raise Sim01QaRendererError(
                "SIM-01 root USD changed during headless native QA"
            )
        if len({item["image"]["sha256"] for item in renders}) != 8:
            raise Sim01QaRendererError("the eight RTX proof images are not distinct")
        if timeline.is_playing():
            raise Sim01QaRendererError(
                "timeline advanced during proof rendering"
            )
        proof_pack = {
            "schema_version": SCHEMA_VERSION,
            "state": PROOF_PACK_STATE,
            "simulation_id": SIMULATION_ID,
            "root_usd_sha256": inputs.root_usd_sha256,
            "build_receipt_sha256": inputs.build_receipt_sha256,
            "review_camera_plan_sha256": plan["plan_sha256"],
            "renderer": {
                "backend": "kit_rtx_native",
                "native_render": True,
                "screen_capture": False,
                "execution_mode": EXECUTION_MODE,
                "render_mode": "RayTracedLighting",
                "resolution_px": [3840, 2160],
                "rt_subframes": rt_subframes,
            },
            "inspection_decision": "passed",
            "inspection_scope": "internal_visual_qa",
            "human_editor_validation": {
                "state": HUMAN_EDITOR_GATE,
                "performed": False,
                "required": True,
            },
            "streaming_contract": {
                "planner_source": STREAMING_PLANNER_SOURCE,
                "transition_semantics_source": STREAMING_TRANSITION_SOURCE,
                "loaded_detail_payloads_per_camera": TILE_COUNT,
                "detail_levels": list(DETAIL_LEVELS),
                "duplicate_detail_tile_count": 0,
                "camera_working_set_count": 8,
            },
            "render_count": 8,
            "renders": [
                {
                    key: value
                    for key, value in render.items()
                    if key != "visual_metrics"
                }
                for render in renders
            ],
            "visual_metrics": [
                {
                    "render_id": render["render_id"],
                    **dict(render["visual_metrics"]),
                }
                for render in renders
            ],
            "fire_simulation_status": BLOCKED_FIRE_STATE,
            "simulation_execution_performed": False,
        }
        _atomic_write_json(output / "proof-pack.json", proof_pack)
        quality = _quality_report(
            inputs=inputs,
            runtime_scene=runtime_scene,
            renders=renders,
        )
        _atomic_write_json(output / "quality-report.json", quality)
        sampler.close()
        stability = _stability_report(
            inputs=inputs,
            duration_seconds=max(time.perf_counter() - started, 1.0e-6),
            stage_open_seconds=max(stage_open_seconds, 1.0e-6),
            payload_settle_seconds=payload_settle_seconds,
            measurement=runtime_measurement,
            gpu_name=sampler.gpu_name,
            total_vram_mib=sampler.total_vram_mib,
            minimum_accepted_fps=minimum_accepted_fps,
        )
        _atomic_write_json(output / "stability-report.json", stability)
        return {
            "state": COMPLETE_STATE,
            "reused": False,
            "output_root": str(output),
            "render_count": 8,
            "camera_count": 40,
            "root_usd_sha256": inputs.root_usd_sha256,
            "review_camera_plan_sha256": plan["plan_sha256"],
            "execution_mode": EXECUTION_MODE,
            "human_editor_validation": HUMAN_EDITOR_GATE,
            "fire_simulation_status": BLOCKED_FIRE_STATE,
            "simulation_execution_performed": False,
        }
    except Exception:
        raise
    finally:
        try:
            sampler.close()
        except Exception:
            pass
        try:
            app.close()
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render real native SIM-01 pre-review QA evidence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce_parser = subparsers.add_parser("produce")
    produce_parser.add_argument("--volume-root", required=True, type=Path)
    produce_parser.add_argument("--runtime-preflight", required=True, type=Path)
    produce_parser.add_argument("--root-usd", required=True, type=Path)
    produce_parser.add_argument("--build-receipt", required=True, type=Path)
    produce_parser.add_argument(
        "--scene-auto-validation",
        required=True,
        type=Path,
    )
    produce_parser.add_argument("--output-root", required=True, type=Path)
    produce_parser.add_argument("--warmup-frames", type=int, default=60)
    produce_parser.add_argument("--measurement-frames", type=int, default=180)
    produce_parser.add_argument(
        "--minimum-accepted-fps",
        type=float,
        default=30.0,
    )
    produce_parser.add_argument("--rt-subframes", type=int, default=16)
    produce_parser.add_argument(
        "--loading-timeout-seconds",
        type=float,
        default=300.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "produce":
        raise Sim01QaRendererError(f"unsupported command: {args.command}")
    result = produce(
        volume_root=args.volume_root,
        runtime_preflight_path=args.runtime_preflight,
        root_usd_path=args.root_usd,
        build_receipt_path=args.build_receipt,
        scene_auto_validation_path=args.scene_auto_validation,
        output_root=args.output_root,
        warmup_frames=args.warmup_frames,
        measurement_frames=args.measurement_frames,
        minimum_accepted_fps=args.minimum_accepted_fps,
        rt_subframes=args.rt_subframes,
        loading_timeout_seconds=args.loading_timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
