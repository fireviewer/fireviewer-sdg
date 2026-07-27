"""Generate reviewable case packages for one bounded production batch."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fireviewer_sdg.artifacts import (
    artifact,
    case_record_path,
    finalize_case_record,
    write_json,
)
from fireviewer_sdg.geometry import (
    assert_visible,
    camera_contract,
    project_aabb,
    project_point,
)
from fireviewer_sdg.incident_days import generate_incident_day
from fireviewer_sdg.event_catalog import case_assignment, load_event_catalog
from fireviewer_sdg.real_world import (
    apply_case_variation,
    calibrate_flow_for_wildfire,
    compose_omniverse_stage,
    select_actor_variation,
    validate_output_render_profile,
)


GENERATORS = {
    "terrestrial_fire_points": "isaac_terrestrial_fire_points",
    "france_cross_view": "isaac_france_cross_view",
    "response_engagement": "isaac_response_engagement",
    "france_incident_days": "fictional_incident_day",
}
PREFIXES = {
    "terrestrial_fire_points": "tfp",
    "france_cross_view": "fcv",
    "response_engagement": "reg",
    "france_incident_days": "fid",
}
FOCAL_LENGTH_MM = 24.0
HORIZONTAL_APERTURE_MM = 20.955
MIN_RESPONSE_BOX_EDGE_PX = {
    "near": 48.0,
    "medium": 32.0,
    "far": 24.0,
    "very_far": 16.0,
}
NUREC_RECONSTRUCTION_FORMATS = frozenset({"particle_field", "nurec_usdz"})
MAX_STAGE_LOADING_UPDATES = 1200
STAGE_STABLE_UPDATES = 8
FRAME_SETTLE_UPDATES = 12
BASE_FILM_ISO = {
    "day": 100.0,
    "dawn": 160.0,
    "dusk": 160.0,
    "night": 400.0,
}


class _GpuMemorySampler:
    """Observe device-memory usage during each visual case through nvidia-smi."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._baseline_bytes = 0
        self._peak_bytes = 0
        self._samples = 0
        self._last_error = ""

    @staticmethod
    def _read_device_memory_bytes() -> int:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        first_line = completed.stdout.splitlines()[0].strip()
        return int(float(first_line)) * 1024 * 1024

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="fireviewer-vram-sampler",
            daemon=True,
        )
        self._thread.start()

    def begin_case(self) -> None:
        try:
            baseline = self._read_device_memory_bytes()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            baseline = 0
            self._last_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._baseline_bytes = baseline
            self._peak_bytes = baseline
            self._samples = int(baseline > 0)
            self._active = True

    def end_case(self) -> dict[str, Any]:
        try:
            observed = self._read_device_memory_bytes()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            observed = 0
            self._last_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            if observed > 0:
                self._peak_bytes = max(self._peak_bytes, observed)
                self._samples += 1
            self._active = False
            baseline = self._baseline_bytes
            peak = self._peak_bytes
            samples = self._samples
        if samples <= 0 or peak <= 0:
            raise RuntimeError(
                "VRAM measurement failed during visual pilot: "
                + (self._last_error or "nvidia-smi returned no sample")
            )
        return {
            "vram_measurement": "nvidia_smi_device_total_memory",
            "vram_baseline_bytes": baseline,
            "vram_peak_bytes": peak,
            "vram_delta_peak_bytes": max(0, peak - baseline),
            "vram_sample_count": samples,
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            with self._lock:
                active = self._active
            if not active:
                continue
            try:
                observed = self._read_device_memory_bytes()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                continue
            with self._lock:
                if self._active:
                    self._peak_bytes = max(self._peak_bytes, observed)
                    self._samples += 1


def _setup_reconstruction_renderer(stage: Any, reconstruction_format: str) -> None:
    """Configure NuRec only for contracts that actually contain a NuRec asset."""
    if reconstruction_format not in NUREC_RECONSTRUCTION_FORMATS:
        return
    try:
        nurec_utils = importlib.import_module("isaacsim.replicator.nurec_utils")
    except ImportError as exc:
        raise RuntimeError(
            "this NuRec contract requires an Isaac Sim build that ships "
            "isaacsim.replicator.nurec_utils"
        ) from exc
    nurec_utils.setup_for_rendering(stage)


def _wait_for_stage_loading(
    context: Any,
    application: Any,
    *,
    max_updates: int = MAX_STAGE_LOADING_UPDATES,
    stable_updates: int = STAGE_STABLE_UPDATES,
) -> tuple[str, int, int]:
    """Wait for USD loading and streaming to settle while the timeline is stopped."""
    stable = 0
    last_status = ("", 0, 0)
    for _ in range(max_updates):
        application.update()
        message, loaded, total = context.get_stage_loading_status()
        last_status = (str(message), int(loaded), int(total))
        loading = loaded < total
        streaming = bool(context.get_stage_streaming_status())
        if not loading and not streaming:
            stable += 1
            if stable >= stable_updates:
                return last_status
        else:
            stable = 0
    raise RuntimeError(
        "USD stage did not finish loading before the render lifecycle: "
        f"message={last_status[0]!r} loaded={last_status[1]} total={last_status[2]}"
    )


def _current_stage_ready(context: Any, application: Any) -> Any:
    """Use the writable stage already attached to a fresh SimulationApp process."""
    _wait_for_stage_loading(context, application)
    if not context.is_writable():
        raise RuntimeError("SimulationApp current USD stage is not writable")
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("SimulationApp started without an attached USD stage")
    return stage


def _background_source(contract: dict[str, Any]) -> str:
    if contract["capture"]["source"] == "new_real_world_capture":
        return "new_real_world_capture_nurec_reconstruction"
    return "new_omniverse_synthetic_french_reference_scene"


def load_batch_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported batch schema_version")
    if payload.get("data_origin") != "new_synthetic_generation":
        raise ValueError("batch cannot reuse an existing corpus")
    category = str(payload.get("category", ""))
    if category not in GENERATORS:
        raise ValueError("unsupported batch category")
    if payload.get("generator") != GENERATORS[category]:
        raise ValueError("batch generator does not match its category")
    stage = str(payload.get("production_stage", ""))
    if stage not in {"pilot", "bulk", "replacement"}:
        raise ValueError("unsupported production stage")
    start = int(payload.get("case_start", -1))
    count = int(payload.get("case_count", 0))
    if not 0 <= start <= 1_000_000 or not 1 <= count <= 256:
        raise ValueError("batch case range is invalid")
    seed_base = int(payload.get("seed_base", -1))
    if not 0 <= seed_base + start < seed_base + start + count <= 2**32:
        raise ValueError("batch seed range is invalid")
    volume_root = Path(str(payload.get("volume_root", ""))).resolve()
    batch_root = Path(str(payload.get("batch_root", ""))).resolve()
    if volume_root not in batch_root.parents:
        raise ValueError("batch_root escapes volume_root")
    render = validate_output_render_profile(
        payload.get("render_profile"), payload.get("resolution")
    )
    catalog_value = str(payload.get("real_world_catalog", "")).strip()
    if not catalog_value:
        raise ValueError("real_world_catalog is required")
    catalog_path = Path(catalog_value).resolve()
    if catalog_path != volume_root and volume_root not in catalog_path.parents:
        raise ValueError("real_world_catalog escapes volume_root")
    target_per_category = int(payload.get("target_per_category", 0))
    if target_per_category not in {4096, 8192}:
        raise ValueError("target_per_category must be 4096 or 8192")
    rt_subframes = int(payload.get("rt_subframes", 0))
    warmup_steps = int(payload.get("warmup_steps", 0))
    if not 8 <= rt_subframes <= 64:
        raise ValueError("rt_subframes must be between 8 and 64")
    if not 16 <= warmup_steps <= 512:
        raise ValueError("warmup_steps must be between 16 and 512")
    payload.update(
        category=category,
        production_stage=stage,
        case_start=start,
        case_count=count,
        seed_base=seed_base,
        resolution=render["resolution"],
        render_profile=render["profile"],
        render_revision=render["revision"],
        volume_root=volume_root,
        batch_root=batch_root,
        real_world_catalog=catalog_path,
        target_per_category=target_per_category,
        rt_subframes=rt_subframes,
        warmup_steps=warmup_steps,
    )
    return payload


def _case_id(category: str, index: int) -> str:
    return f"{PREFIXES[category]}-{index:06d}"


def _case_root(spec: dict[str, Any], case_id: str) -> Path:
    return (
        spec["volume_root"]
        / "production"
        / "generated"
        / spec["category"]
        / case_id
    )


def _existing_record(spec: dict[str, Any], case_id: str) -> bool:
    path = case_record_path(spec["batch_root"], case_id)
    if not path.is_file():
        return False
    if spec["category"] == "france_incident_days":
        return True
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record.get("render", {}).get("revision") == spec["render_revision"]


def _write_batch_progress(
    spec: dict[str, Any],
    *,
    state: str,
    current_case_id: str | None = None,
    assignment: dict[str, Any] | None = None,
    last_completed_record: dict[str, Any] | None = None,
) -> None:
    spec["batch_root"].mkdir(parents=True, exist_ok=True)
    expected_ids = [
        _case_id(spec["category"], case_index)
        for case_index in range(
            spec["case_start"],
            spec["case_start"] + spec["case_count"],
        )
    ]
    produced = sum(
        1 for case_id in expected_ids if _existing_record(spec, case_id)
    )
    current: dict[str, Any] | None = None
    if current_case_id is not None:
        current = {"case_id": current_case_id}
        if assignment is not None:
            event = assignment.get("event", {})
            variation = assignment.get("variation", {})
            flow = variation.get("flow", {})
            lighting = variation.get("lighting", {})
            viewpoint = variation.get("camera_pose", {}).get("viewpoint", {})
            current.update(
                event_id=str(event.get("event_id", "")),
                progression=str(flow.get("progression", "")),
                time_of_day=str(lighting.get("time_of_day", "")),
                distance_band=str(viewpoint.get("distance_band", "")),
                occlusion=str(viewpoint.get("occlusion", "")),
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "state": state,
        "category": spec["category"],
        "case_start": spec["case_start"],
        "case_count": spec["case_count"],
        "produced": produced,
        "remaining": spec["case_count"] - produced,
        "percent": round(100.0 * produced / spec["case_count"], 3),
        "current": current,
    }
    if last_completed_record is not None:
        payload["last_completed"] = {
            "case_id": last_completed_record["case_id"],
            "preview_relpath": last_completed_record["preview_relpath"],
            "overlays": last_completed_record.get("overlays", []),
        }
    write_json(spec["batch_root"] / "batch-progress.json", payload)


def _save_rgb(path: Path, data: object) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - pod runtime contract
        raise RuntimeError("NumPy and Pillow are required for RGB capture") from exc
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise RuntimeError(f"unexpected RGB annotator shape: {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array[:, :, :3], mode="RGB").save(
        path, format="JPEG", quality=97, subsampling=0, optimize=True
    )


def _frame_quality_metrics(data: object) -> dict[str, float]:
    """Measure obvious exposure failures before a render enters review inventory."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - pod runtime contract
        raise RuntimeError("NumPy is required for RGB quality validation") from exc
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise RuntimeError(f"unexpected RGB annotator shape: {array.shape}")
    rgb = array[:, :, :3].astype(np.float32, copy=False)
    luminance = (
        rgb[:, :, 0] * 0.2126
        + rgb[:, :, 1] * 0.7152
        + rgb[:, :, 2] * 0.0722
    )
    warm_emissive = (
        (rgb[:, :, 0] >= 210.0)
        & (rgb[:, :, 1] >= 150.0)
        & ((rgb[:, :, 0] - rgb[:, :, 2]) >= 25.0)
        & ((rgb[:, :, 1] - rgb[:, :, 2]) >= 10.0)
    )
    rows, columns = np.nonzero(warm_emissive)
    if rows.size:
        warm_bbox_fraction = float(
            (rows.max() - rows.min() + 1)
            * (columns.max() - columns.min() + 1)
            / warm_emissive.size
        )
    else:
        warm_bbox_fraction = 0.0
    return {
        "mean_luminance": round(float(luminance.mean()), 4),
        "std_luminance": round(float(luminance.std()), 4),
        "p05_luminance": round(float(np.percentile(luminance, 5)), 4),
        "p50_luminance": round(float(np.percentile(luminance, 50)), 4),
        "p95_luminance": round(float(np.percentile(luminance, 95)), 4),
        "bright_fraction_240": round(float((luminance >= 240.0).mean()), 6),
        "dark_fraction_10": round(float((luminance <= 10.0).mean()), 6),
        "warm_emissive_fraction": round(float(warm_emissive.mean()), 6),
        "warm_emissive_bbox_fraction": round(warm_bbox_fraction, 6),
    }


def _frame_quality_failure(metrics: dict[str, float]) -> str | None:
    if (
        metrics["mean_luminance"] > 200.0
        or metrics["p05_luminance"] > 125.0
        or metrics["bright_fraction_240"] > 0.12
    ):
        return "overexposed"
    if metrics["warm_emissive_fraction"] > 0.12:
        return "oversized_emissive_region"
    if metrics["mean_luminance"] < 18.0 or metrics["p95_luminance"] < 55.0:
        return "underexposed"
    if metrics["std_luminance"] < 12.0:
        return "insufficient_dynamic_range"
    return None


def _coarse_mask_components(mask: object) -> tuple[object, list[list[tuple[int, int]]]]:
    """Downsample a pixel mask and return its 8-connected components."""

    import numpy as np

    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    grid_height = min(45, height)
    grid_width = min(80, width)
    y_edges = np.linspace(0, height, grid_height + 1, dtype=int)
    x_edges = np.linspace(0, width, grid_width + 1, dtype=int)
    coarse = np.zeros((grid_height, grid_width), dtype=bool)
    for row in range(grid_height):
        for column in range(grid_width):
            block = source[
                y_edges[row] : y_edges[row + 1],
                x_edges[column] : x_edges[column + 1],
            ]
            coarse[row, column] = bool(block.size and block.mean() >= 0.06)
    visited = np.zeros_like(coarse, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_row, start_column in zip(*np.nonzero(coarse), strict=True):
        if visited[start_row, start_column]:
            continue
        component: list[tuple[int, int]] = []
        pending = [(int(start_row), int(start_column))]
        visited[start_row, start_column] = True
        while pending:
            row, column = pending.pop()
            component.append((row, column))
            for delta_row in (-1, 0, 1):
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbor_row = row + delta_row
                    neighbor_column = column + delta_column
                    if (
                        0 <= neighbor_row < grid_height
                        and 0 <= neighbor_column < grid_width
                        and coarse[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        pending.append((neighbor_row, neighbor_column))
        components.append(component)
    return coarse, components


def _composite_quality_metrics(
    background: object,
    composite: object,
    *,
    expected_anchors: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Measure how much Flow hides or replaces the validated clean scene."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - pod runtime contract
        raise RuntimeError("NumPy is required for composite quality validation") from exc
    background_rgb = np.asarray(background)
    composite_rgb = np.asarray(composite)
    if (
        background_rgb.ndim != 3
        or composite_rgb.ndim != 3
        or background_rgb.shape != composite_rgb.shape
        or background_rgb.shape[2] not in {3, 4}
    ):
        raise RuntimeError(
            "background and composite RGB frames must have the same RGB(A) shape"
        )
    clean = background_rgb[:, :, :3].astype(np.float32, copy=False)
    rendered = composite_rgb[:, :, :3].astype(np.float32, copy=False)
    absolute_delta = np.abs(rendered - clean).mean(axis=2)
    affected = absolute_delta >= 18.0
    strongly_affected = absolute_delta >= 45.0

    clean_luminance = (
        clean[:, :, 0] * 0.2126
        + clean[:, :, 1] * 0.7152
        + clean[:, :, 2] * 0.0722
    )
    rendered_luminance = (
        rendered[:, :, 0] * 0.2126
        + rendered[:, :, 1] * 0.7152
        + rendered[:, :, 2] * 0.0722
    )

    def edge_energy(values: object) -> float:
        array = np.asarray(values)
        horizontal = np.abs(np.diff(array, axis=1)).mean()
        vertical = np.abs(np.diff(array, axis=0)).mean()
        return float(horizontal + vertical)

    clean_edges = edge_energy(clean_luminance)
    rendered_edges = edge_energy(rendered_luminance)
    clean_warm = (
        (clean[:, :, 0] >= 210.0)
        & (clean[:, :, 1] >= 145.0)
        & ((clean[:, :, 0] - clean[:, :, 2]) >= 25.0)
    )
    rendered_warm = (
        (rendered[:, :, 0] >= 210.0)
        & (rendered[:, :, 1] >= 145.0)
        & ((rendered[:, :, 0] - rendered[:, :, 2]) >= 25.0)
    )
    added_emissive = rendered_warm & ~clean_warm & affected
    rows, columns = np.nonzero(added_emissive)
    if rows.size:
        centroid_x = float(columns.mean() / max(1, rendered.shape[1] - 1))
        centroid_y = float(rows.mean() / max(1, rendered.shape[0] - 1))
        bbox_fraction = float(
            (rows.max() - rows.min() + 1)
            * (columns.max() - columns.min() + 1)
            / added_emissive.size
        )
    else:
        centroid_x = -1.0
        centroid_y = -1.0
        bbox_fraction = 0.0
    affected_rows, affected_columns = np.nonzero(affected)
    if affected_rows.size:
        flow_bbox_fraction = float(
            (affected_rows.max() - affected_rows.min() + 1)
            * (affected_columns.max() - affected_columns.min() + 1)
            / affected.size
        )
    else:
        flow_bbox_fraction = 0.0
    metrics = {
        "flow_affected_fraction": round(float(affected.mean()), 6),
        "flow_strongly_affected_fraction": round(float(strongly_affected.mean()), 6),
        "flow_bbox_fraction": round(flow_bbox_fraction, 6),
        "scene_edge_retention": round(
            rendered_edges / max(clean_edges, 1e-6), 6
        ),
        "added_emissive_fraction": round(float(added_emissive.mean()), 6),
        "added_emissive_bbox_fraction": round(bbox_fraction, 6),
        "added_emissive_centroid_x": round(centroid_x, 6),
        "added_emissive_centroid_y": round(centroid_y, 6),
        "mean_absolute_flow_delta": round(float(absolute_delta.mean()), 4),
    }
    if expected_anchors:
        smoke_base = expected_anchors.get("smoke_column_base")
        fire_points = [
            expected_anchors[label]
            for label in ("active_fire_point", "visible_fire_front_point")
            if label in expected_anchors
        ]
        if smoke_base is not None:
            smoke_x = float(smoke_base["x_normalized"])
            smoke_y = float(smoke_base["y_normalized"])
            normalized_x = np.linspace(0.0, 1.0, affected.shape[1])[None, :]
            normalized_y = np.linspace(0.0, 1.0, affected.shape[0])[:, None]
            base_region = (
                (normalized_x - smoke_x) ** 2 + (normalized_y - smoke_y) ** 2
            ) <= 0.065**2
            metrics["smoke_base_effect_fraction"] = round(
                float(affected[base_region].mean()) if base_region.any() else 0.0,
                6,
            )
            coarse_affected, affected_components = _coarse_mask_components(affected)
            grid_height, grid_width = coarse_affected.shape
            base_row = int(round(smoke_y * max(1, grid_height - 1)))
            base_column = int(round(smoke_x * max(1, grid_width - 1)))
            connected_component: list[tuple[int, int]] = []
            for component in affected_components:
                if any(
                    math.hypot(
                        (row - base_row) / max(1, grid_height - 1),
                        (column - base_column) / max(1, grid_width - 1),
                    )
                    <= 0.075
                    for row, column in component
                ):
                    connected_component = component
                    break
            metrics["smoke_column_connected_to_base"] = float(
                bool(connected_component)
            )
            metrics["smoke_column_upward_span_fraction"] = round(
                (
                    max(
                        0.0,
                        (
                            base_row
                            - min(row for row, _column in connected_component)
                        )
                        / max(1, grid_height - 1),
                    )
                    if connected_component
                    else 0.0
                ),
                6,
            )

        _coarse_emissive, emissive_components = _coarse_mask_components(
            added_emissive
        )
        detached_cells = 0
        emissive_grid_height, emissive_grid_width = _coarse_emissive.shape
        for component in emissive_components:
            if len(component) < 2:
                continue
            component_x = sum(column for _row, column in component) / (
                len(component) * max(1, emissive_grid_width - 1)
            )
            component_y = sum(row for row, _column in component) / (
                len(component) * max(1, emissive_grid_height - 1)
            )
            if fire_points and min(
                math.dist(
                    (component_x, component_y),
                    (
                        float(point["x_normalized"]),
                        float(point["y_normalized"]),
                    ),
                )
                for point in fire_points
            ) > 0.18:
                detached_cells += len(component)
        metrics["detached_emissive_component_fraction"] = round(
            detached_cells
            / max(1, emissive_grid_height * emissive_grid_width),
            6,
        )
    return metrics


def _composite_quality_failure(
    metrics: dict[str, float],
    *,
    expected_fire_points: list[dict[str, float]] | None = None,
    expected_anchors: dict[str, dict[str, float]] | None = None,
) -> str | None:
    """Reject absent, screen-filling, opaque, or spatially detached fire effects."""
    if metrics["flow_affected_fraction"] < 0.0002:
        return "missing_fire_smoke_effect"
    if (
        metrics["flow_affected_fraction"] > 0.32
        or metrics["flow_strongly_affected_fraction"] > 0.18
        or metrics["flow_bbox_fraction"] > 0.72
    ):
        return "excessive_smoke_coverage"
    if metrics["scene_edge_retention"] < 0.62:
        return "scene_obscured_by_smoke"
    if metrics.get("detached_emissive_component_fraction", 0.0) > 0.0:
        return "detached_sun_like_emissive_component"
    if (
        metrics["added_emissive_fraction"] > 0.06
        or metrics["added_emissive_bbox_fraction"] > 0.24
    ):
        return "oversized_emissive_effect"
    if expected_anchors and "smoke_column_base" in expected_anchors:
        if metrics.get("smoke_base_effect_fraction", 0.0) < 0.02:
            return "smoke_column_detached_from_base"
        if metrics.get("smoke_column_connected_to_base", 0.0) < 0.5:
            return "smoke_column_detached_from_base"
        if metrics.get("smoke_column_upward_span_fraction", 0.0) < 0.045:
            return "smoke_column_has_no_localized_upward_extent"
        expected_fire_points = [
            expected_anchors[label]
            for label in ("active_fire_point", "visible_fire_front_point")
            if label in expected_anchors
        ]
    if expected_fire_points and metrics["added_emissive_fraction"] >= 0.0001:
        centroid = (
            metrics["added_emissive_centroid_x"],
            metrics["added_emissive_centroid_y"],
        )
        minimum_distance = min(
            math.dist(
                centroid,
                (
                    float(point["x_normalized"]),
                    float(point["y_normalized"]),
                ),
            )
            for point in expected_fire_points
        )
        if minimum_distance > 0.20:
            return "emissive_effect_detached_from_fire"
    return None


def _capture_validated_rgb(
    *,
    rep: Any,
    annotator: Any,
    application: Any,
    stage: Any,
    spec: dict[str, Any],
    time_of_day: str,
    expected_anchors: dict[str, dict[str, float]] | None = None,
) -> tuple[object, dict[str, Any]]:
    """Compare clean/background renders and reject unusable Flow composites."""
    import carb.settings
    import numpy as np
    from pxr import UsdGeom

    settings = carb.settings.get_settings()
    film_iso = BASE_FILM_ISO[time_of_day]
    attempts: list[dict[str, Any]] = []
    flow_prim = stage.GetPrimAtPath("/World/FireAndSmoke")
    if not flow_prim or not flow_prim.IsValid():
        raise RuntimeError("Flow root is absent during RGB validation")
    flow_imageable = UsdGeom.Imageable(flow_prim)
    try:
        for attempt in range(1, 4):
            settings.set("/rtx/post/tonemap/filmIso", film_iso)
            flow_imageable.MakeInvisible()
            for _ in range(FRAME_SETTLE_UPDATES):
                application.update()
            rep.orchestrator.step(
                delta_time=0.0, rt_subframes=spec["rt_subframes"]
            )
            background = np.asarray(annotator.get_data()).copy()
            background_metrics = _frame_quality_metrics(background)
            background_failure = _frame_quality_failure(background_metrics)

            flow_imageable.MakeVisible()
            for _ in range(FRAME_SETTLE_UPDATES):
                application.update()
            rep.orchestrator.step(
                delta_time=0.0, rt_subframes=spec["rt_subframes"]
            )
            data = np.asarray(annotator.get_data()).copy()
            frame_metrics = _frame_quality_metrics(data)
            frame_failure = _frame_quality_failure(frame_metrics)
            composite_metrics = _composite_quality_metrics(
                background,
                data,
                expected_anchors=expected_anchors,
            )
            composite_failure = _composite_quality_failure(
                composite_metrics,
                expected_anchors=expected_anchors,
            )
            failure = background_failure or frame_failure or composite_failure
            attempts.append(
                {
                    "attempt": attempt,
                    "film_iso": film_iso,
                    "result": failure or "passed",
                    "background_failure": background_failure,
                    "frame_failure": frame_failure,
                    "composite_failure": composite_failure,
                    "background": background_metrics,
                    "composite_frame": frame_metrics,
                    "flow_composite": composite_metrics,
                }
            )
            if failure is None:
                return data, {
                    "validation": "clean_scene_and_bounded_flow_composite_gate_passed",
                    "film_iso": film_iso,
                    "attempts": attempts,
                    "background": background_metrics,
                    "composite_frame": frame_metrics,
                    "flow_composite": composite_metrics,
                }
            if failure == "overexposed":
                film_iso = max(25.0, film_iso / 2.0)
            elif failure == "underexposed":
                film_iso = min(800.0, film_iso * 2.0)
            else:
                break
    finally:
        flow_imageable.MakeVisible()
    raise RuntimeError(
        "render failed clean-scene/composite quality gate: "
        + json.dumps(attempts, sort_keys=True)
    )


def _contact_sheet(
    path: Path, *, ground_path: Path, ortho_path: Path, mnt_preview_path: Path, title: str
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - pod runtime contract
        raise RuntimeError("Pillow is required for cross-view contact sheets") from exc
    canvas = Image.new("RGB", (1024, 768), "#101416")
    ground = Image.open(ground_path).convert("RGB")
    ortho = Image.open(ortho_path).convert("RGB")
    mnt = Image.open(mnt_preview_path).convert("RGB")
    ground.thumbnail((640, 590))
    ortho.thumbnail((330, 280))
    mnt.thumbnail((330, 280))
    canvas.paste(ground, (20, 88))
    canvas.paste(ortho, (674, 88))
    canvas.paste(mnt, (674, 400))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((20, 24), title, fill="#f0f2f3", font=font)
    draw.text((20, 64), "PHOTO TERRESTRE", fill="#9ca7ac", font=font)
    draw.text((674, 64), "ORTHOPHOTO", fill="#9ca7ac", font=font)
    draw.text((674, 376), "MNT VERIFIE DU SITE", fill="#9ca7ac", font=font)
    draw.text((696, 744), "FEU INSERE — REVUE HUMAINE", fill="#e56a2f", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=97, subsampling=0, optimize=True)


def _configure_usd_camera(
    *, stage: Any, path: str, width: int, height: int, orthographic: bool = False
) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"camera prim is absent: {path}")
    camera = UsdGeom.Camera(prim)
    camera.GetFocalLengthAttr().Set(FOCAL_LENGTH_MM)
    if orthographic:
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        camera.GetHorizontalApertureAttr().Set(80.0)
        camera.GetVerticalApertureAttr().Set(60.0)
    else:
        camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        camera.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE_MM)
        camera.GetVerticalApertureAttr().Set(HORIZONTAL_APERTURE_MM * height / width)


def _create_ground_camera(rep: Any, stage: Any, width: int, height: int) -> Any:
    camera = rep.functional.create.camera(
        position=(0.0, -24.0, 7.0),
        look_at=(0.0, 0.0, 2.0),
        focal_length=FOCAL_LENGTH_MM,
        focus_distance=400.0,
        clipping_range=(0.1, 100000.0),
        parent="/World/Cameras",
        name="GroundCamera",
    )
    _configure_usd_camera(
        stage=stage,
        path="/World/Cameras/GroundCamera",
        width=width,
        height=height,
    )
    return camera


def _attach_rgb(rep: Any, render_product: Any) -> Any:
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    return annotator


def _generate_terrestrial(
    spec: dict[str, Any],
    rep: Any,
    stage: Any,
    application: Any,
    contract: dict[str, Any],
    assignments: list[dict[str, Any]],
) -> None:
    width, height = spec["resolution"]
    camera = _create_ground_camera(rep, stage, width, height)
    render_product = rep.create.render_product(
        camera, (width, height), name="GroundRenderProduct"
    )
    rgb = _attach_rgb(rep, render_product)
    gpu_sampler: _GpuMemorySampler = spec["_gpu_sampler"]
    for assignment in assignments:
        case_index = assignment["case_index"]
        case_id = _case_id(spec["category"], case_index)
        if _existing_record(spec, case_id):
            continue
        case_started = time.perf_counter()
        gpu_sampler.begin_case()
        seed = spec["seed_base"] + case_index
        variation = assignment["variation"]
        _write_batch_progress(
            spec,
            state="rendering",
            current_case_id=case_id,
            assignment=assignment,
        )
        apply_case_variation(stage, application, variation)
        pose = variation["camera_pose"]
        anchors = variation["flow"]["anchors_world_m"]
        position = pose["position"]
        look_at = pose["look_at"]
        camera_data = camera_contract(
            position=position,
            look_at=look_at,
            width=width,
            height=height,
            focal_length_mm=FOCAL_LENGTH_MM,
            horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
        )
        projections = {
            label: project_point(world, camera_data) for label, world in anchors.items()
        }
        assert_visible(projections.values(), margin=0.03)
        rep.functional.modify.pose(
            camera,
            position_value=tuple(position),
            look_at_value=tuple(look_at),
            look_at_up_axis=(0.0, 0.0, 1.0),
            write_to_usd=True,
        )
        image_data, pixel_quality = _capture_validated_rgb(
            rep=rep,
            annotator=rgb,
            application=application,
            stage=stage,
            spec=spec,
            time_of_day=variation["lighting"]["time_of_day"],
            expected_anchors=projections,
        )
        root = _case_root(spec, case_id)
        photo_path = root / "ground-photo.jpg"
        annotations_path = root / "point-annotations.json"
        _save_rgb(photo_path, image_data)
        annotations = {
            "schema_version": 1,
            "coordinate_space": "render_pixels_and_normalized",
            "projection": "exact_from_usd_camera_and_authored_flow_anchors_pending_review",
            "points": [
                {"label": label, "world_m": list(anchors[label]), **projection}
                for label, projection in projections.items()
            ],
        }
        write_json(annotations_path, annotations)
        overlays = [
            {
                "kind": "point",
                "label": label,
                "x_normalized": projection["x_normalized"],
                "y_normalized": projection["y_normalized"],
            }
            for label, projection in projections.items()
        ]
        record = {
            "schema_version": 1,
            "category": spec["category"],
            "case_id": case_id,
            "data_origin": "new_synthetic_generation",
            "production_stage": spec["production_stage"],
            "seed": seed,
            "preview_relpath": photo_path.relative_to(spec["volume_root"]).as_posix(),
            "overlays": overlays,
            "artifacts": [
                artifact(spec["volume_root"], photo_path, kind="ground_photo"),
                artifact(spec["volume_root"], annotations_path, kind="point_annotations"),
            ],
            "truth": {
                "synthetic": True,
                "real_world_claim": False,
                "background_source": _background_source(contract),
                "capture_manifest_sha256": contract["capture"]["capture_manifest_sha256"],
                "scene_asset_sha256": contract["reconstruction"]["asset_sha256"],
                "annotation_source": "authored_flow_geometry_projected_with_recorded_camera_pending_review",
                "anchors_world_m": {key: list(value) for key, value in anchors.items()},
                "human_review_required": True,
                "usable_for_training": False,
                "event_id": contract["event_id"],
                "fire_duration_days": contract["duration_days"],
                "landscape_profile": contract["geospatial"]["landscape_profile"],
                "progression": variation["flow"]["progression"],
            },
            "camera": camera_data,
            "render": {
                "profile": spec["render_profile"],
                "revision": spec["render_revision"],
                "camera_pose_id": pose["id"],
                "rt_subframes": spec["rt_subframes"],
                "warmup_steps": spec["warmup_steps"],
                "variation_id": variation["id"],
                "lighting_variant_id": variation["lighting"]["id"],
                "time_of_day": variation["lighting"]["time_of_day"],
                "flow_state_id": variation["flow"]["id"],
                "diversity_signature": variation["diversity_signature"],
                "viewpoint": pose["viewpoint"],
                "pixel_quality": pixel_quality,
            },
        }
        gpu_memory = gpu_sampler.end_case()
        finalize_case_record(
            batch_root=spec["batch_root"],
            record=record,
            started_monotonic=case_started,
            gpu_memory=gpu_memory,
        )
        _write_batch_progress(
            spec,
            state="running",
            last_completed_record=record,
        )
        print(f"fireviewer cases: produced {spec['category']}/{case_id}", flush=True)
    rgb.detach([render_product])
    render_product.destroy()


def _generate_cross_view(
    spec: dict[str, Any],
    rep: Any,
    stage: Any,
    application: Any,
    contract: dict[str, Any],
    assignments: list[dict[str, Any]],
) -> None:
    width, height = spec["resolution"]
    ground_camera = _create_ground_camera(rep, stage, width, height)
    ground_product = rep.create.render_product(
        ground_camera, (width, height), name="GroundRenderProduct"
    )
    ground_rgb = _attach_rgb(rep, ground_product)
    gpu_sampler: _GpuMemorySampler = spec["_gpu_sampler"]
    geospatial = contract["geospatial"]
    ortho_path = geospatial["orthophoto"]
    mnt_path = geospatial["mnt"]
    mnt_preview_path = geospatial["mnt_preview"]
    world_origin = geospatial["world_origin_lambert93_m"]
    site_code = contract["site_id"]
    for assignment in assignments:
        case_index = assignment["case_index"]
        case_id = _case_id(spec["category"], case_index)
        if _existing_record(spec, case_id):
            continue
        case_started = time.perf_counter()
        gpu_sampler.begin_case()
        seed = spec["seed_base"] + case_index
        variation = assignment["variation"]
        _write_batch_progress(
            spec,
            state="rendering",
            current_case_id=case_id,
            assignment=assignment,
        )
        apply_case_variation(stage, application, variation)
        pose = variation["camera_pose"]
        anchors = variation["flow"]["anchors_world_m"]
        fire_local = anchors["active_fire_point"]
        position = pose["position"]
        look_at = pose["look_at"]
        camera_data = camera_contract(
            position=position,
            look_at=look_at,
            width=width,
            height=height,
            focal_length_mm=FOCAL_LENGTH_MM,
            horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
        )
        projections = {
            label: project_point(world, camera_data)
            for label, world in anchors.items()
        }
        fire_projection = projections["active_fire_point"]
        assert_visible(projections.values(), margin=0.03)
        rep.functional.modify.pose(
            ground_camera,
            position_value=tuple(position),
            look_at_value=tuple(look_at),
            look_at_up_axis=(0.0, 0.0, 1.0),
            write_to_usd=True,
        )
        image_data, pixel_quality = _capture_validated_rgb(
            rep=rep,
            annotator=ground_rgb,
            application=application,
            stage=stage,
            spec=spec,
            time_of_day=variation["lighting"]["time_of_day"],
            expected_anchors=projections,
        )
        root = _case_root(spec, case_id)
        ground_path = root / "ground-photo.jpg"
        preview_path = root / "cross-view-preview.jpg"
        manifest_path = root / "site-manifest.json"
        _save_rgb(ground_path, image_data)
        camera_data["position_lambert93_virtual"] = [
            round(world_origin[index] + position[index], 6) for index in range(3)
        ]
        fire_lambert93 = [
            round(world_origin[index] + fire_local[index], 6) for index in range(3)
        ]
        manifest = {
            "schema_version": 1,
            "site_code": site_code,
            "jurisdiction_profile": "france_metropolitaine_nurec_background",
            "coordinate_warning": "real captured site background; inserted fire remains synthetic",
            "photo": ground_path.name,
            "camera": camera_data,
            "orthophoto": {
                "path": ortho_path.relative_to(spec["volume_root"]).as_posix(),
                "sha256": geospatial["orthophoto_sha256"],
                "crs": "EPSG:2154",
            },
            "mnt": {
                "path": mnt_path.relative_to(spec["volume_root"]).as_posix(),
                "sha256": geospatial["mnt_sha256"],
                "model": (
                    "verified_real_world_terrain_model"
                    if contract["geospatial"]["landscape_origin"] == "real_french_capture"
                    else "coherent_synthetic_french_reference_terrain_model"
                ),
            },
            "fire_position": {
                "local_world_m": fire_local,
                "lambert93_m": fire_lambert93,
                "verified_from_flow_anchor_and_scene_georeferencing": True,
            },
            "capture_manifest_sha256": contract["capture"]["capture_manifest_sha256"],
            "scene_asset_sha256": contract["reconstruction"]["asset_sha256"],
        }
        write_json(manifest_path, manifest)
        _contact_sheet(
            preview_path,
            ground_path=ground_path,
            ortho_path=ortho_path,
            mnt_preview_path=mnt_preview_path,
            title=f"{case_id} — {site_code}",
        )
        record = {
            "schema_version": 1,
            "category": spec["category"],
            "case_id": case_id,
            "data_origin": "new_synthetic_generation",
            "production_stage": spec["production_stage"],
            "seed": seed,
            "preview_relpath": preview_path.relative_to(spec["volume_root"]).as_posix(),
            "overlays": [],
            "artifacts": [
                artifact(spec["volume_root"], ground_path, kind="ground_photo"),
                artifact(spec["volume_root"], ortho_path, kind="orthophoto"),
                artifact(spec["volume_root"], mnt_path, kind="mnt"),
                artifact(spec["volume_root"], manifest_path, kind="site_manifest"),
                artifact(spec["volume_root"], preview_path, kind="preview"),
            ],
            "truth": {
                "synthetic": True,
                "real_world_claim": False,
                "background_source": _background_source(contract),
                "site_code": site_code,
                "fire_position_local_m": fire_local,
                "fire_position_lambert93_virtual": fire_lambert93,
                "fire_position_verified_from_generator": True,
                "capture_manifest_sha256": contract["capture"]["capture_manifest_sha256"],
                "scene_asset_sha256": contract["reconstruction"]["asset_sha256"],
                "human_review_required": True,
                "usable_for_training": False,
                "event_id": contract["event_id"],
                "fire_duration_days": contract["duration_days"],
                "landscape_profile": contract["geospatial"]["landscape_profile"],
                "progression": variation["flow"]["progression"],
            },
            "camera": camera_data,
            "render": {
                "profile": spec["render_profile"],
                "revision": spec["render_revision"],
                "camera_pose_id": pose["id"],
                "rt_subframes": spec["rt_subframes"],
                "warmup_steps": spec["warmup_steps"],
                "variation_id": variation["id"],
                "lighting_variant_id": variation["lighting"]["id"],
                "time_of_day": variation["lighting"]["time_of_day"],
                "flow_state_id": variation["flow"]["id"],
                "diversity_signature": variation["diversity_signature"],
                "viewpoint": pose["viewpoint"],
                "pixel_quality": pixel_quality,
            },
        }
        gpu_memory = gpu_sampler.end_case()
        finalize_case_record(
            batch_root=spec["batch_root"],
            record=record,
            started_monotonic=case_started,
            gpu_memory=gpu_memory,
        )
        _write_batch_progress(
            spec,
            state="running",
            last_completed_record=record,
        )
        print(f"fireviewer cases: produced {spec['category']}/{case_id}", flush=True)
    ground_rgb.detach([ground_product])
    ground_product.destroy()


def _response_objects(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Expose review-gated actor candidates from the scene contract."""
    return {
        actor["class_id"]: {
            "prim_path": f"/World/Actors/Actor{index:02d}",
            "center": actor["center_world_m"],
            "minimum": actor["aabb_min_world_m"],
            "maximum": actor["aabb_max_world_m"],
            "positive": actor["positive"],
            "asset_sha256": actor["asset_sha256"],
        }
        for index, actor in enumerate(contract["composition"]["actors"])
    }


def _show_only_response_actor(
    stage: Any, objects: dict[str, dict[str, Any]], target_class: str
) -> None:
    """Prevent unlabelled response actors from contaminating a detector image."""
    from pxr import UsdGeom

    if target_class not in objects:
        raise RuntimeError(f"response target actor is absent: {target_class}")
    for class_id, definition in objects.items():
        prim = stage.GetPrimAtPath(definition["prim_path"])
        if not prim or not prim.IsValid():
            raise RuntimeError(f"response actor prim is absent: {class_id}")
        imageable = UsdGeom.Imageable(prim)
        if class_id == target_class:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()


def _validate_response_box(
    box: dict[str, float], *, width: int, height: int, distance_band: str
) -> dict[str, float]:
    """Reject tiny or clipped boxes before they enter the review inventory."""
    if distance_band not in MIN_RESPONSE_BOX_EDGE_PX:
        raise RuntimeError(f"unsupported response distance band: {distance_band}")
    minimum_edge_px = MIN_RESPONSE_BOX_EDGE_PX[distance_band]
    width_px = (box["x_max"] - box["x_min"]) * width
    height_px = (box["y_max"] - box["y_min"]) * height
    if width_px < minimum_edge_px or height_px < minimum_edge_px:
        raise RuntimeError(
            "response actor is too small for a usable HD training box: "
            f"distance_band={distance_band} minimum_edge_px={minimum_edge_px:.0f} "
            f"width_px={width_px:.2f} height_px={height_px:.2f}"
        )
    margin_px = min(
        box["x_min"] * width,
        (1.0 - box["x_max"]) * width,
        box["y_min"] * height,
        (1.0 - box["y_max"]) * height,
    )
    if margin_px < 2.0:
        raise RuntimeError("response actor box touches the image boundary")
    return {
        "width_px": round(width_px, 3),
        "height_px": round(height_px, 3),
        "required_minimum_edge_px": minimum_edge_px,
        "minimum_border_margin_px": round(margin_px, 3),
    }


def _generate_response(
    spec: dict[str, Any],
    rep: Any,
    stage: Any,
    application: Any,
    contract: dict[str, Any],
    objects: dict[str, dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> None:
    width, height = spec["resolution"]
    camera = _create_ground_camera(rep, stage, width, height)
    product = rep.create.render_product(camera, (width, height), name="GroundRenderProduct")
    rgb = _attach_rgb(rep, product)
    gpu_sampler: _GpuMemorySampler = spec["_gpu_sampler"]
    classes = list(objects)
    for assignment in assignments:
        case_index = assignment["case_index"]
        case_id = _case_id(spec["category"], case_index)
        if _existing_record(spec, case_id):
            continue
        case_started = time.perf_counter()
        gpu_sampler.begin_case()
        seed = spec["seed_base"] + case_index
        label = classes[case_index % len(classes)]
        definition = objects[label]
        variation_capacity = len(contract["composition"]["camera_poses"]) * len(
            contract["composition"]["flow_states"]
        )
        class_ordinal = case_index % variation_capacity
        variation = select_actor_variation(contract, label, class_ordinal)
        _write_batch_progress(
            spec,
            state="rendering",
            current_case_id=case_id,
            assignment={**assignment, "variation": variation},
        )
        apply_case_variation(stage, application, variation)
        _show_only_response_actor(stage, objects, label)
        pose = variation["camera_pose"]
        position = pose["position"]
        look_at = definition["center"]
        camera_data = camera_contract(
            position=position,
            look_at=look_at,
            width=width,
            height=height,
            focal_length_mm=FOCAL_LENGTH_MM,
            horizontal_aperture_mm=HORIZONTAL_APERTURE_MM,
        )
        box = project_aabb(definition["minimum"], definition["maximum"], camera_data)
        box_quality = _validate_response_box(
            box,
            width=width,
            height=height,
            distance_band=pose["viewpoint"]["distance_band"],
        )
        rep.functional.modify.pose(camera, position_value=tuple(position), look_at_value=tuple(look_at), look_at_up_axis=(0.0, 0.0, 1.0), write_to_usd=True)
        image_data, pixel_quality = _capture_validated_rgb(
            rep=rep,
            annotator=rgb,
            application=application,
            stage=stage,
            spec=spec,
            time_of_day=variation["lighting"]["time_of_day"],
        )
        root = _case_root(spec, case_id)
        photo_path = root / "ground-photo.jpg"
        annotations_path = root / "box-annotations.json"
        _save_rgb(photo_path, image_data)
        annotations = {
            "schema_version": 1,
            "box_source": "exact_projection_of_review_gated_actor_aabb",
            "object_class": label,
            "actor_asset_sha256": definition["asset_sha256"],
            "box_normalized": box,
            "box_quality": box_quality,
            "aabb_world_m": {
                "minimum": definition["minimum"],
                "maximum": definition["maximum"],
            },
            "human_validation_required": True,
        }
        write_json(annotations_path, annotations)
        engagement = "simulated_engagement" if definition["positive"] else "not_engaged_hard_negative"
        record = {
            "schema_version": 1,
            "category": spec["category"],
            "case_id": case_id,
            "data_origin": "new_synthetic_generation",
            "production_stage": spec["production_stage"],
            "seed": seed,
            "preview_relpath": photo_path.relative_to(spec["volume_root"]).as_posix(),
            "overlays": [{"kind": "box", "label": label, **box}],
            "artifacts": [
                artifact(spec["volume_root"], photo_path, kind="ground_photo"),
                artifact(spec["volume_root"], annotations_path, kind="box_annotations"),
            ],
            "truth": {
                "synthetic": True,
                "real_world_claim": False,
                "object_class": label,
                "engagement_label": engagement,
                "operational_truth": "synthetic_only",
                "negative_is_visually_close": not definition["positive"],
                "target_actor_isolated": True,
                "background_source": _background_source(contract),
                "capture_manifest_sha256": contract["capture"]["capture_manifest_sha256"],
                "scene_asset_sha256": contract["reconstruction"]["asset_sha256"],
                "actor_asset_sha256": definition["asset_sha256"],
                "human_review_required": True,
                "usable_for_training": False,
                "event_id": contract["event_id"],
                "fire_duration_days": contract["duration_days"],
                "landscape_profile": contract["geospatial"]["landscape_profile"],
                "progression": variation["flow"]["progression"],
            },
            "camera": camera_data,
            "render": {
                "profile": spec["render_profile"],
                "revision": spec["render_revision"],
                "camera_pose_id": pose["id"],
                "rt_subframes": spec["rt_subframes"],
                "warmup_steps": spec["warmup_steps"],
                "variation_id": variation["id"],
                "lighting_variant_id": variation["lighting"]["id"],
                "time_of_day": variation["lighting"]["time_of_day"],
                "flow_state_id": variation["flow"]["id"],
                "diversity_signature": variation["diversity_signature"],
                "viewpoint": pose["viewpoint"],
                "target_actor_distance_m": round(
                    math.dist(position, definition["center"]), 6
                ),
                "camera_focus": "target_actor_center",
                "pixel_quality": pixel_quality,
            },
        }
        gpu_memory = gpu_sampler.end_case()
        finalize_case_record(
            batch_root=spec["batch_root"],
            record=record,
            started_monotonic=case_started,
            gpu_memory=gpu_memory,
        )
        _write_batch_progress(
            spec,
            state="running",
            last_completed_record=record,
        )
        print(f"fireviewer cases: produced {spec['category']}/{case_id}", flush=True)
    rgb.detach([product])
    product.destroy()


def generate_batch(spec_path: Path) -> None:
    spec = load_batch_spec(spec_path)
    spec["batch_root"].mkdir(parents=True, exist_ok=True)
    if spec["category"] == "france_incident_days":
        _write_batch_progress(spec, state="starting")
        for case_index in range(spec["case_start"], spec["case_start"] + spec["case_count"]):
            case_id = _case_id(spec["category"], case_index)
            if _existing_record(spec, case_id):
                continue
            _write_batch_progress(
                spec,
                state="generating_document",
                current_case_id=case_id,
            )
            path = generate_incident_day(
                volume_root=spec["volume_root"],
                batch_root=spec["batch_root"],
                case_index=case_index,
                seed=spec["seed_base"] + case_index,
                production_stage=spec["production_stage"],
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            _write_batch_progress(
                spec,
                state="running",
                last_completed_record=record,
            )
            print(f"fireviewer cases: produced {spec['category']}/{path.stem}", flush=True)
        _write_batch_progress(spec, state="completed")
        return

    catalog = load_event_catalog(
        spec["real_world_catalog"],
        volume_root=spec["volume_root"],
        target_per_category=spec["target_per_category"],
    )
    selected_assignments = [
        case_assignment(catalog, case_index)
        for case_index in range(
            spec["case_start"], spec["case_start"] + spec["case_count"]
        )
    ]
    event_ids = {
        str(assignment["event"]["event_id"])
        for assignment in selected_assignments
    }
    if len(event_ids) != 1:
        raise RuntimeError(
            "visual batch must contain exactly one fire event to isolate the "
            "USD, Flow, and Replicator lifecycle"
        )
    _write_batch_progress(spec, state="starting")
    event_id = next(iter(event_ids))
    assignments = selected_assignments

    import isaacsim  # noqa: F401 - initializes the pip namespace package
    from isaacsim.simulation_app import SimulationApp

    application = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "multi_gpu": False}
    )
    gpu_sampler = _GpuMemorySampler()
    gpu_sampler.start()
    spec["_gpu_sampler"] = gpu_sampler
    try:
        import carb.settings
        import omni.kit.app
        import omni.timeline
        from pxr import UsdGeom

        settings = carb.settings.get_settings()
        settings.set("/renderer/multiGpu/enabled", False)
        settings.set("/rtx/rendermode", "RayTracedLighting")
        settings.set("/rtx/rtpt/gaussian/skipTonemapping/enabled", False)
        settings.set("/rtx/post/tonemap/enabled", True)
        settings.set("/rtx/post/tonemap/op", 4)
        settings.set("/rtx/post/tonemap/filmIso", BASE_FILM_ISO["day"])
        settings.set("/rtx/post/tonemap/whitepoint", 6500.0)
        settings.set("/rtx/post/histogram/enabled", False)
        settings.set("/rtx/post/lensFlares/enabled", False)
        settings.set("/rtx/post/tvNoise/enabled", False)
        settings.set("/rtx/post/aa/op", 3)
        settings.set("/rtx/post/dlss/execMode", 2)
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.flowusd", True)
        if not extension_manager.is_extension_enabled("omni.flowusd"):
            raise RuntimeError("required omni.flowusd extension could not be enabled")
        # NVIDIA SimulationApp already owns a writable current USD stage. Each
        # worker process handles one fire, so reopening the context is needless
        # and unsafe while Kit's graph extensions are active.
        usd_context = application.context
        stage = _current_stage_ready(usd_context, application)
        print("fireviewer stage: current-stage-ready", flush=True)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        contract = assignments[0]["event"]["contract"]
        provenance = compose_omniverse_stage(stage, contract)
        _setup_reconstruction_renderer(
            stage, contract["reconstruction"]["format"]
        )

        # Keep the timeline stopped while references, Flow, Hydra, and Fabric
        # settle. No UsdContext reopen occurs in this one-event process.
        loading_status = _wait_for_stage_loading(usd_context, application)
        print(
            "fireviewer stage: composition-settled "
            + json.dumps(
                {
                    "loading_message": loading_status[0],
                    "loaded": loading_status[1],
                    "total": loading_status[2],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        flow_calibration = calibrate_flow_for_wildfire(stage, contract)
        print(
            "fireviewer stage: flow-metric-calibration "
            + json.dumps(flow_calibration, sort_keys=True),
            flush=True,
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(spec["warmup_steps"]):
            application.update()
        timeline.pause()
        print("fireviewer stage: flow-warmup-complete", flush=True)

        print("fireviewer stage: replicator-initializing", flush=True)
        import omni.replicator.core as rep

        rep.orchestrator.set_capture_on_play(False)
        rep.set_global_seed(spec["seed_base"] + spec["case_start"])
        if spec["category"] == "terrestrial_fire_points":
            _generate_terrestrial(
                spec, rep, stage, application, contract, assignments
            )
        elif spec["category"] == "france_cross_view":
            _generate_cross_view(
                spec, rep, stage, application, contract, assignments
            )
        else:
            objects = _response_objects(contract)
            _generate_response(
                spec,
                rep,
                stage,
                application,
                contract,
                objects,
                assignments,
            )
        print(
            "fireviewer event: "
            + json.dumps(
                {
                    **provenance,
                    "event_id": event_id,
                    "case_count": len(assignments),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        rep.orchestrator.wait_until_complete()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        gpu_sampler.close()
        application.close(
            wait_for_replicator=False,
            skip_cleanup=True,
            exit_code=0,
        )
    _write_batch_progress(spec, state="completed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-spec", type=Path, required=True)
    arguments = parser.parse_args()
    generate_batch(arguments.batch_spec.resolve())


if __name__ == "__main__":
    main()
