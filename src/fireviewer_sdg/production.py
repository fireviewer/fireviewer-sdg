"""Persisted pilot-gated production of four disjoint synthetic deliverables."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fireviewer_sdg.artifacts import write_json
from fireviewer_sdg.event_catalog import load_event_catalog
from fireviewer_sdg.real_world import (
    RENDER_REVISION,
    validate_output_render_profile,
)
from fireviewer_sdg.review_store import (
    CATEGORY_IDS,
    PILOT_PER_CATEGORY,
    MAX_TARGET_PER_CATEGORY,
    TARGET_PER_CATEGORY,
    CaseStore,
)
from fireviewer_sdg.storage import (
    assert_remaining_capacity,
    assert_storage_architecture,
    validate_storage_plan,
)


GENERATORS = {
    "terrestrial_fire_points": "isaac_terrestrial_fire_points",
    "france_cross_view": "isaac_france_cross_view",
    "response_engagement": "isaac_response_engagement",
    "france_incident_days": "fictional_incident_day",
}
CASE_PREFIXES = {
    "terrestrial_fire_points": "tfp",
    "france_cross_view": "fcv",
    "response_engagement": "reg",
    "france_incident_days": "fid",
}
INPUT_READINESS_NAME = "input-readiness-hd-v2.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_production_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("unsupported production schema_version")
    if payload.get("production_id") != "fireviewer-new-synthetic-cases-v1":
        raise ValueError("unsupported production_id")
    if payload.get("data_origin") != "new_synthetic_generation":
        raise ValueError("production plan cannot reuse an existing corpus")
    target_per_category = int(payload.get("target_per_category", 0))
    if target_per_category not in {TARGET_PER_CATEGORY, MAX_TARGET_PER_CATEGORY}:
        raise ValueError("production target must be 4096 or 8192 new cases per category")
    if int(payload.get("maximum_target_per_category", 0)) != MAX_TARGET_PER_CATEGORY:
        raise ValueError("production plan must reserve capacity for 8192 cases per category")
    if int(payload.get("pilot_per_category", 0)) != PILOT_PER_CATEGORY:
        raise ValueError(
            f"production pilot must contain {PILOT_PER_CATEGORY} cases per category"
        )
    batch_size = int(payload.get("batch_size", 0))
    if not 1 <= batch_size <= 256:
        raise ValueError("production batch_size must be between 1 and 256")
    render = validate_output_render_profile(
        payload.get("render_profile"), payload.get("resolution")
    )
    real_world_catalog = os.path.expandvars(
        str(payload.get("real_world_catalog", "")).strip()
    )
    if not real_world_catalog or not (
        Path(real_world_catalog).is_absolute() or real_world_catalog.startswith("/")
    ):
        raise ValueError("production real_world_catalog must be an absolute runtime path")
    rt_subframes = int(payload.get("rt_subframes", 0))
    warmup_steps = int(payload.get("warmup_steps", 0))
    if not 8 <= rt_subframes <= 64:
        raise ValueError("production rt_subframes must be between 8 and 64")
    if not 16 <= warmup_steps <= 512:
        raise ValueError("production warmup_steps must be between 16 and 512")
    storage = validate_storage_plan(payload.get("storage"))
    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("production plan must define at least one supported category")
    by_category: dict[str, dict[str, Any]] = {}
    seed_ranges: list[tuple[int, int, str]] = []
    for item in categories:
        if not isinstance(item, dict):
            raise ValueError("production categories must be objects")
        category = str(item.get("category", ""))
        if category not in CATEGORY_IDS or category in by_category:
            raise ValueError("production categories must be unique and supported")
        if item.get("generator") != GENERATORS[category]:
            raise ValueError(f"unexpected generator for {category}")
        seed_base = int(item.get("seed_base", -1))
        seed_end = seed_base + target_per_category
        if not 0 <= seed_base < seed_end <= 2**32:
            raise ValueError(f"seed range is invalid for {category}")
        seed_ranges.append((seed_base, seed_end, category))
        by_category[category] = dict(item)
    scope = payload.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("production scope must be an object")
    excluded_categories = scope.get("excluded_categories", [])
    if not isinstance(excluded_categories, list):
        raise ValueError("scope.excluded_categories must be a list")
    excluded = tuple(str(category) for category in excluded_categories)
    missing = set(CATEGORY_IDS) - set(by_category)
    if set(excluded) != missing:
        raise ValueError(
            "scope.excluded_categories must exactly identify inactive categories"
        )
    if "response_engagement" in missing:
        excluded_content = scope.get("excluded_content", [])
        if not isinstance(excluded_content, list) or not {
            "vehicles",
            "humans",
        }.issubset({str(item) for item in excluded_content}):
            raise ValueError(
                "a V1 without response engagement must explicitly exclude vehicles and humans"
            )
    for index, (start, end, category) in enumerate(seed_ranges):
        for other_start, other_end, other_category in seed_ranges[index + 1 :]:
            if max(start, other_start) < min(end, other_end):
                raise ValueError(
                    f"seed ranges overlap: {category} and {other_category}"
                )
    payload["batch_size"] = batch_size
    payload["target_per_category"] = target_per_category
    payload["resolution"] = render["resolution"]
    payload["render_profile"] = render["profile"]
    payload["render_revision"] = render["revision"]
    payload["real_world_catalog"] = real_world_catalog
    payload["rt_subframes"] = rt_subframes
    payload["warmup_steps"] = warmup_steps
    payload["storage"] = storage
    payload["by_category"] = by_category
    payload["active_categories"] = tuple(
        category for category in CATEGORY_IDS if category in by_category
    )
    payload["excluded_categories"] = excluded
    return payload


def _stage_range(stage: str, target_per_category: int) -> tuple[int, int]:
    if stage == "pilot":
        return 0, PILOT_PER_CATEGORY
    if stage == "bulk":
        return PILOT_PER_CATEGORY, target_per_category
    raise ValueError("unsupported production stage")


def _batches(start: int, stop: int, batch_size: int) -> list[tuple[int, int]]:
    return [
        (offset, min(batch_size, stop - offset))
        for offset in range(start, stop, batch_size)
    ]


def _event_aligned_batches(
    assignments: list[dict[str, Any]],
    start: int,
    stop: int,
    batch_size: int,
) -> list[tuple[int, int]]:
    """Split a range without ever putting two fire events in one Isaac process."""
    if not 0 <= start <= stop <= len(assignments):
        raise ValueError("event-aligned batch range is outside the catalog")
    batches: list[tuple[int, int]] = []
    offset = start
    while offset < stop:
        event_id = str(assignments[offset]["event"]["event_id"])
        event_stop = offset + 1
        while (
            event_stop < stop
            and str(assignments[event_stop]["event"]["event_id"]) == event_id
        ):
            event_stop += 1
        batches.extend(_batches(offset, event_stop, batch_size))
        offset = event_stop
    return batches


def _batch_root(
    volume_root: Path, *, category: str, stage: str, start: int, count: int
) -> Path:
    return (
        volume_root
        / "production"
        / "batches"
        / category
        / f"{stage}-{start:06d}-{count:04d}"
    )


def _expected_ids(category: str, start: int, count: int) -> set[str]:
    prefix = CASE_PREFIXES[category]
    return {f"{prefix}-{index:06d}" for index in range(start, start + count)}


def _load_batch_records(
    *,
    batch_root: Path,
    category: str,
    start: int,
    count: int,
    render_revision: str = RENDER_REVISION,
) -> list[dict[str, Any]]:
    records_root = batch_root / "records"
    if not records_root.is_dir():
        return []
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(records_root.glob("*.json"))
    ]
    if category != "france_incident_days" and any(
        record.get("render", {}).get("revision") != render_revision
        for record in records
    ):
        # A quality-revision bump makes interrupted/unreviewed visual batches
        # resumable by rerendering them instead of silently reusing bad pixels.
        return []
    case_ids = {str(record.get("case_id", "")) for record in records}
    expected = _expected_ids(category, start, count)
    if records and not case_ids.issubset(expected):
        unexpected = sorted(case_ids - expected)[:5]
        raise RuntimeError(
            "batch record inventory mismatch: "
            f"unexpected={unexpected}"
        )
    return records


def _run_batch_generator(spec_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fireviewer_sdg.case_generation",
            "--batch-spec",
            str(spec_path),
        ],
        check=True,
    )


def run_production_stage(
    *,
    plan_path: Path,
    volume_root: Path,
    case_store: CaseStore,
    stage: str,
) -> dict[str, Any]:
    plan = load_production_plan(plan_path)
    assert_storage_architecture(volume_root, plan["storage"])
    event_catalog = load_event_catalog(
        Path(plan["real_world_catalog"]),
        volume_root=volume_root,
        target_per_category=plan["target_per_category"],
    )
    start, stop = _stage_range(stage, plan["target_per_category"])
    state_path = volume_root / "production" / "production-state.json"
    started_at = _utc_now()
    completed_batches: list[dict[str, Any]] = []
    batches_by_category = {
        category: (
            _batches(start, stop, plan["batch_size"])
            if category == "france_incident_days"
            else _event_aligned_batches(
                event_catalog["assignments"],
                start,
                stop,
                plan["batch_size"],
            )
        )
        for category in plan["active_categories"]
    }
    total_batches = sum(len(items) for items in batches_by_category.values())
    total_cases = (stop - start) * len(plan["active_categories"])
    state: dict[str, Any] = {
        "schema_version": 2,
        "production_id": plan["production_id"],
        "data_origin": "new_synthetic_generation",
        "state": "running",
        "stage": stage,
        "started_at": started_at,
        "range": {"start": start, "stop": stop},
        "totals": {
            "cases": total_cases,
            "batches": total_batches,
            "categories": len(plan["active_categories"]),
        },
        "active_categories": list(plan["active_categories"]),
        "excluded_categories": list(plan["excluded_categories"]),
        "progress": {
            "completed_cases": 0,
            "completed_batches": 0,
            "percent": 0.0,
        },
        "current_batch": None,
        "completed_batches": completed_batches,
    }
    write_json(state_path, state)
    case_store.log_event(
        "production_stage_started", stage=stage, start=start, stop=stop
    )
    try:
        batch_ordinal = 0
        for category in plan["active_categories"]:
            item = plan["by_category"][category]
            batches = batches_by_category[category]
            for batch_start, batch_count in batches:
                batch_ordinal += 1
                deliverables = case_store.status()
                remaining_cases = sum(
                    max(0, item["target"] - item["produced"])
                    for item in deliverables["categories"]
                )
                assert_remaining_capacity(
                    volume_root,
                    plan["storage"],
                    remaining_cases=remaining_cases,
                )
                root = _batch_root(
                    volume_root,
                    category=category,
                    stage=stage,
                    start=batch_start,
                    count=batch_count,
                )
                spec_path = root / "batch-spec.json"
                event_id = (
                    None
                    if category == "france_incident_days"
                    else str(
                        event_catalog["assignments"][batch_start]["event"][
                            "event_id"
                        ]
                    )
                )
                state["current_batch"] = {
                    "category": category,
                    "stage": stage,
                    "case_start": batch_start,
                    "case_count": batch_count,
                    "batch_index": batch_ordinal,
                    "batch_total": total_batches,
                    "event_id": event_id,
                    "progress_relpath": (
                        root / "batch-progress.json"
                    ).relative_to(volume_root).as_posix(),
                }
                write_json(state_path, state)
                spec = {
                    "schema_version": 1,
                    "production_id": plan["production_id"],
                    "data_origin": "new_synthetic_generation",
                    "production_stage": stage,
                    "category": category,
                    "generator": item["generator"],
                    "case_start": batch_start,
                    "case_count": batch_count,
                    "seed_base": int(item["seed_base"]),
                    "resolution": plan["resolution"],
                    "render_profile": plan["render_profile"],
                    "real_world_catalog": plan["real_world_catalog"],
                    "target_per_category": plan["target_per_category"],
                    "rt_subframes": plan["rt_subframes"],
                    "warmup_steps": plan["warmup_steps"],
                    "volume_root": str(volume_root),
                    "batch_root": str(root),
                }
                write_json(spec_path, spec)
                records = _load_batch_records(
                    batch_root=root,
                    category=category,
                    start=batch_start,
                    count=batch_count,
                    render_revision=plan["render_revision"],
                )
                resumed = len(records) == batch_count
                if not resumed:
                    case_store.log_event(
                        "batch_started",
                        category=category,
                        stage=stage,
                        case_start=batch_start,
                        case_count=batch_count,
                    )
                    _run_batch_generator(spec_path)
                    records = _load_batch_records(
                        batch_root=root,
                        category=category,
                        start=batch_start,
                        count=batch_count,
                        render_revision=plan["render_revision"],
                    )
                if len(records) != batch_count:
                    raise RuntimeError(
                        f"batch did not produce every case: {category}/{batch_start}"
                    )
                for record in records:
                    case_store.register(record)
                completed = {
                    "category": category,
                    "stage": stage,
                    "case_start": batch_start,
                    "case_count": batch_count,
                    "resumed": resumed,
                    "completed_at": _utc_now(),
                }
                completed_batches.append(completed)
                state["completed_batches"] = completed_batches
                completed_cases = sum(
                    int(batch["case_count"]) for batch in completed_batches
                )
                state["last_completed_batch"] = completed
                state["current_batch"] = None
                state["progress"] = {
                    "completed_cases": completed_cases,
                    "completed_batches": len(completed_batches),
                    "percent": round(
                        100.0 * completed_cases / total_cases,
                        3,
                    ),
                }
                write_json(state_path, state)
                case_store.log_event("batch_completed", **completed)
        final_state = {
            **state,
            "state": (
                "awaiting_pilot_review" if stage == "pilot" else "awaiting_full_review"
            ),
            "completed_at": _utc_now(),
            "deliverables": case_store.status(),
        }
        write_json(state_path, final_state)
        case_store.log_event(
            "production_stage_completed", stage=stage, state=final_state["state"]
        )
        return final_state
    except BaseException as exc:
        failed = {
            **state,
            "state": "failed",
            "failed_at": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(state_path, failed)
        case_store.log_event(
            "production_stage_failed", stage=stage, error=failed["error"]
        )
        raise


class ProductionManager:
    """Run one production stage at a time and enforce the visual pilot gate."""

    def __init__(self, volume_root: Path, case_store: CaseStore) -> None:
        self._volume_root = volume_root.resolve()
        self._case_store = case_store
        self._state_path = self._volume_root / "production" / "production-state.json"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._snapshot: dict[str, Any] = {"state": "idle"}
        if self._state_path.is_file():
            try:
                current = json.loads(self._state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                current = {"state": "state_unreadable"}
            if current.get("state") in {"queued", "running"}:
                current = {
                    **current,
                    "state": "interrupted_recoverable",
                    "interrupted_at": _utc_now(),
                }
                write_json(self._state_path, current)
            self._snapshot = current

    def _launch(self, plan_path: Path, *, stage: str) -> None:
        plan = load_production_plan(plan_path)
        with self._lock:
            if self._case_store.target_per_category != plan["target_per_category"]:
                raise RuntimeError("case store target does not match the production plan")
            if self._case_store.categories != plan["active_categories"]:
                raise RuntimeError(
                    "case store categories do not match the active production scope"
                )
            if self._case_store.render_revision != plan["render_revision"]:
                raise RuntimeError(
                    "case store render revision does not match the production plan"
                )
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a production stage is already running")
            deliverables = self._case_store.status()
            if stage == "pilot" and all(
                item["pilot_produced"] >= item["pilot_target"]
                for item in deliverables["categories"]
            ):
                raise RuntimeError("the pilot inventory is already complete")
            if stage == "bulk" and not deliverables["pilot_ready"]:
                raise RuntimeError(
                    f"bulk production is locked until {PILOT_PER_CATEGORY} pilots are accepted per category"
                )
            if stage == "bulk" and all(
                item["produced"] >= item["target"]
                for item in deliverables["categories"]
            ):
                raise RuntimeError("bulk production is already complete")
            if stage == "bulk":
                readiness_path = (
                    self._volume_root / "input" / INPUT_READINESS_NAME
                )
                if not readiness_path.is_file():
                    raise RuntimeError(
                        "bulk production is locked until the geographic site "
                        "readiness report exists"
                    )
                readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
                if readiness.get("bulk_allowed") is not True:
                    raise RuntimeError(
                        "bulk production is locked: the three-site setup is "
                        "pilot-only and requires geographic expansion"
                    )
            try:
                assert_storage_architecture(self._volume_root, plan["storage"])
                load_event_catalog(
                    Path(plan["real_world_catalog"]),
                    volume_root=self._volume_root,
                    target_per_category=plan["target_per_category"],
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"real-world input gate is blocked: {exc}") from exc
            self._snapshot = {
                "schema_version": 2,
                "state": "queued",
                "stage": stage,
                "queued_at": _utc_now(),
            }
            write_json(self._state_path, self._snapshot)
            self._thread = threading.Thread(
                target=self._run,
                args=(plan_path, stage),
                name=f"fireviewer-sdg-{stage}",
                daemon=False,
            )
            self._thread.start()

    def start_pilot(self, plan_path: Path) -> None:
        self._launch(plan_path, stage="pilot")

    def continue_bulk(self, plan_path: Path) -> None:
        self._launch(plan_path, stage="bulk")

    def _run(self, plan_path: Path, stage: str) -> None:
        try:
            result = run_production_stage(
                plan_path=plan_path,
                volume_root=self._volume_root,
                case_store=self._case_store,
                stage=stage,
            )
        except BaseException as exc:
            with self._lock:
                self._snapshot = {
                    **self._snapshot,
                    "state": "failed",
                    "failed_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return
        with self._lock:
            self._snapshot = result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = dict(self._snapshot)
        if self._state_path.is_file():
            try:
                snapshot = json.loads(
                    self._state_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                pass
            else:
                current = snapshot.get("current_batch")
                if isinstance(current, dict):
                    relative = str(
                        current.get("progress_relpath", "")
                    ).strip()
                    progress_path = (
                        self._volume_root / relative
                    ).resolve()
                    if (
                        relative
                        and self._volume_root in progress_path.parents
                        and progress_path.is_file()
                    ):
                        try:
                            progress = json.loads(
                                progress_path.read_text(encoding="utf-8")
                            )
                        except (OSError, ValueError):
                            progress = None
                        if isinstance(progress, dict):
                            snapshot["current_batch"] = {
                                **current,
                                "progress": progress,
                            }
                return snapshot
        return snapshot

    def wait(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()
