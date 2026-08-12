"""FireViewer capture writer with explicit visual and truth products.

The generic Replicator writer is deliberately not used here.  It cannot encode
the FireViewer distinction between rendered evidence (``FireVisual``) and the
simulation truth that produced it (``FireTruth``), nor can it prove that a
front was genuinely visible from the recorded camera.  Runtime dependencies
are intentionally injected so this module remains testable without Kit.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FIRE_TRUTH_ROOT = "/World/FireTruth"
FIRE_VISUAL_ROOT = "/World/FireVisual"
FIRE_TRUTH_TARGETS_KEY = "fireviewer:active_front_targets_local_m"
FIRE_TRUTH_STATE_KEY = "fireviewer:fire_state_id"
REQUIRED_ANNOTATORS = (
    "rgb",
    "semantic_segmentation",
    "instance_segmentation",
    "distance_to_camera",
)


class FireViewerWriterError(RuntimeError):
    """Raised when a FireViewer capture cannot be proven complete."""


RaycastClosest = Callable[
    [Sequence[float], Sequence[float], float], Mapping[str, Any] | None
]


def _finite_vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise FireViewerWriterError(f"{label} must contain exactly three coordinates")
    result = [float(component) for component in value]
    if not all(math.isfinite(component) for component in result):
        raise FireViewerWriterError(f"{label} contains a non-finite coordinate")
    return result


def _vector_subtract(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(left[index]) - float(right[index]) for index in range(3)]


def _length(value: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(component) ** 2 for component in value))


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_save_array(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(data), allow_pickle=False)
    os.replace(temporary, path)


def _prim_is_valid(prim: object) -> bool:
    valid = getattr(prim, "IsValid", None)
    return bool(callable(valid) and valid())


def _front_targets_from_truth_prim(prim: object) -> list[list[float]]:
    getter = getattr(prim, "GetCustomDataByKey", None)
    if not callable(getter):
        raise FireViewerWriterError("FireTruth prim cannot expose custom data")
    raw = getter(FIRE_TRUTH_TARGETS_KEY)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FireViewerWriterError(
                "FireTruth active-front target data is not valid JSON"
            ) from exc
    if not isinstance(raw, (list, tuple)) or not raw:
        raise FireViewerWriterError("FireTruth has no active-front raycast targets")
    return [
        _finite_vector(target, label="FireTruth active-front target")
        for target in raw
    ]


def validate_fire_layers(stage: object) -> dict[str, object]:
    """Require independently addressable truth and visual layers in USD."""

    get_prim = getattr(stage, "GetPrimAtPath", None)
    if not callable(get_prim):
        raise FireViewerWriterError("capture stage cannot resolve FireViewer prims")
    truth = get_prim(FIRE_TRUTH_ROOT)
    visual = get_prim(FIRE_VISUAL_ROOT)
    if not _prim_is_valid(truth):
        raise FireViewerWriterError(f"required FireTruth layer is absent: {FIRE_TRUTH_ROOT}")
    if not _prim_is_valid(visual):
        raise FireViewerWriterError(f"required FireVisual layer is absent: {FIRE_VISUAL_ROOT}")
    state_getter = getattr(truth, "GetCustomDataByKey", None)
    state = state_getter(FIRE_TRUTH_STATE_KEY) if callable(state_getter) else None
    if not isinstance(state, str) or not state.strip():
        raise FireViewerWriterError("FireTruth has no stable fire_state_id")
    targets = _front_targets_from_truth_prim(truth)
    return {
        "truth_root": FIRE_TRUTH_ROOT,
        "visual_root": FIRE_VISUAL_ROOT,
        "fire_state_id": state.strip(),
        "active_front_targets_local_m": targets,
    }


def raycast_visibility(
    *,
    camera_position_m: Sequence[float],
    targets_local_m: Sequence[Sequence[float]],
    raycast_closest: RaycastClosest,
    hit_tolerance_m: float = 0.5,
) -> dict[str, object]:
    """Measure physical line-of-sight from actual closest-hit raycasts.

    A target is visible only when the first collision lies at the target
    distance (within tolerance).  A missing hit, malformed hit or an earlier
    collision is recorded as non-visible; it is never promoted from a 2D
    projection estimate.
    """

    origin = _finite_vector(camera_position_m, label="camera position")
    if hit_tolerance_m <= 0.0 or not math.isfinite(hit_tolerance_m):
        raise FireViewerWriterError("raycast hit tolerance must be finite and positive")
    records: list[dict[str, object]] = []
    for index, raw_target in enumerate(targets_local_m):
        target = _finite_vector(raw_target, label="active-front target")
        direction = _vector_subtract(target, origin)
        target_distance = _length(direction)
        if target_distance <= 1.0e-6:
            raise FireViewerWriterError("camera position coincides with an active-front target")
        unit_direction = [component / target_distance for component in direction]
        hit = raycast_closest(origin, unit_direction, target_distance)
        hit_distance: float | None = None
        if isinstance(hit, Mapping) and hit.get("hit") is not False:
            candidate = hit.get("distance")
            if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
                hit_distance = float(candidate)
        visible = (
            hit_distance is not None
            and abs(hit_distance - target_distance) <= hit_tolerance_m
        )
        records.append(
            {
                "target_index": index,
                "target_local_m": target,
                "target_distance_m": target_distance,
                "first_hit_distance_m": hit_distance,
                "visible": visible,
            }
        )
    visible_count = sum(1 for record in records if record["visible"])
    return {
        "method": "physx_closest_hit_distance_v1",
        "ray_count": len(records),
        "visible_ray_count": visible_count,
        "visibility_fraction": visible_count / len(records),
        "rays": records,
    }


class FireViewerReplicatorWriter:
    """Small, explicit runtime writer for FireViewer visual/truth products."""

    def __init__(
        self,
        *,
        rep: Any,
        stage: object,
        output_root: Path,
        raycast_closest: RaycastClosest,
    ) -> None:
        self._rep = rep
        self._stage = stage
        self._output_root = output_root.resolve()
        self._raycast_closest = raycast_closest
        self._annotators: dict[str, Any] = {}
        self._layers = validate_fire_layers(stage)
        self._attached = False

    def attach(self, render_product: object) -> None:
        if self._attached:
            raise FireViewerWriterError("FireViewer writer is already attached")
        registry = getattr(self._rep, "AnnotatorRegistry", None)
        get_annotator = getattr(registry, "get_annotator", None)
        if not callable(get_annotator):
            raise FireViewerWriterError("Replicator AnnotatorRegistry is unavailable")
        try:
            for name in REQUIRED_ANNOTATORS:
                annotator = get_annotator(name)
                annotator.attach([render_product])
                self._annotators[name] = annotator
        except Exception:
            self.detach()
            raise
        self._attached = True

    def capture_frame(
        self,
        *,
        frame_index: int,
        camera_position_m: Sequence[float],
        camera_pose: Mapping[str, object],
        intrinsics: Mapping[str, object],
    ) -> dict[str, object]:
        if not self._attached:
            raise FireViewerWriterError("FireViewer writer must be attached before capture")
        if frame_index < 0:
            raise FireViewerWriterError("frame index must be non-negative")
        current_layers = validate_fire_layers(self._stage)
        if current_layers["fire_state_id"] != self._layers["fire_state_id"]:
            # A state transition is valid only when the caller recreates the
            # writer for the new simulation instant; silently crossing this
            # boundary would mix truth from one state with pixels from another.
            raise FireViewerWriterError("FireTruth state changed while writer was attached")
        visibility = raycast_visibility(
            camera_position_m=camera_position_m,
            targets_local_m=current_layers["active_front_targets_local_m"],
            raycast_closest=self._raycast_closest,
        )
        if int(visibility["visible_ray_count"]) < 1:
            raise FireViewerWriterError("active front has no true raycast-visible sample")
        captured = {
            name: np.asarray(annotator.get_data()).copy()
            for name, annotator in self._annotators.items()
        }
        if captured["rgb"].ndim < 2 or captured["distance_to_camera"].ndim < 2:
            raise FireViewerWriterError("Replicator returned invalid RGB or depth data")
        frame_id = f"frame-{frame_index:06d}"
        visual_root = self._output_root / "FireVisual"
        truth_root = self._output_root / "FireTruth"
        _atomic_save_array(visual_root / "rgb" / f"{frame_id}.npy", captured["rgb"])
        _atomic_save_array(
            truth_root / "depth_distance_to_camera" / f"{frame_id}.npy",
            captured["distance_to_camera"],
        )
        _atomic_save_array(
            truth_root / "semantic_segmentation" / f"{frame_id}.npy",
            captured["semantic_segmentation"],
        )
        _atomic_save_array(
            truth_root / "instance_segmentation" / f"{frame_id}.npy",
            captured["instance_segmentation"],
        )
        receipt = {
            "schema_version": 1,
            "frame_index": frame_index,
            "frame_id": frame_id,
            "fire_truth": current_layers,
            "fire_visual": {
                "rgb_path": f"FireVisual/rgb/{frame_id}.npy",
                "camera_pose": dict(camera_pose),
                "intrinsics": dict(intrinsics),
            },
            "depth_path": f"FireTruth/depth_distance_to_camera/{frame_id}.npy",
            "semantic_path": f"FireTruth/semantic_segmentation/{frame_id}.npy",
            "instance_path": f"FireTruth/instance_segmentation/{frame_id}.npy",
            "visibility": visibility,
        }
        _atomic_write_json(truth_root / "receipts" / f"{frame_id}.json", receipt)
        return receipt

    def detach(self) -> None:
        for annotator in self._annotators.values():
            detach = getattr(annotator, "detach", None)
            if callable(detach):
                detach()
        self._annotators.clear()
        self._attached = False


def kit_physx_raycast_closest() -> RaycastClosest:
    """Create the actual closest-hit adapter; no screen-space fallback exists."""

    try:
        from omni.physx import get_physx_scene_query_interface
    except ImportError as exc:  # pragma: no cover - requires native Kit
        raise FireViewerWriterError("PhysX raycast interface is unavailable") from exc
    query = get_physx_scene_query_interface()
    if query is None:
        raise FireViewerWriterError("PhysX scene query interface is unavailable")

    def raycast(
        origin: Sequence[float], direction: Sequence[float], distance: float
    ) -> Mapping[str, Any] | None:
        result = query.raycast_closest(tuple(origin), tuple(direction), distance)
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise FireViewerWriterError("PhysX closest-hit response is malformed")
        return result

    return raycast


__all__ = [
    "FIRE_TRUTH_ROOT",
    "FIRE_VISUAL_ROOT",
    "FIRE_TRUTH_TARGETS_KEY",
    "FIRE_TRUTH_STATE_KEY",
    "REQUIRED_ANNOTATORS",
    "FireViewerWriterError",
    "FireViewerReplicatorWriter",
    "kit_physx_raycast_closest",
    "raycast_visibility",
    "validate_fire_layers",
]
