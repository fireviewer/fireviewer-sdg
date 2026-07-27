"""Validated scheduling of many operational fire events without oversampling one fire."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from fireviewer_sdg.real_world import (
    REQUIRED_CAMPAIGN_PROGRESSION_PHASES,
    load_real_world_contract,
    select_case_variation,
)


MIN_FIRE_EVENTS = 512
MAX_FIRE_DURATION_DAYS = 15
MIN_CASES_PER_FIRE_PER_CATEGORY = 4
MAX_CASES_PER_FIRE_PER_CATEGORY = 24
MIN_DISTINCT_IMAGE_COUNTS = 4
MIN_PROGRESSION_PROFILES = 32
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_volume(value: object, *, root: Path, field: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field} is required")
    path = Path(raw)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{field} must remain inside the production volume")
    if not resolved.is_file():
        raise ValueError(f"{field} is absent: {resolved}")
    return resolved


def _assert_coverage(
    *,
    assignments: list[dict[str, Any]],
    progression_profiles: Counter[tuple[str, ...]],
    landscape_families: Counter[str],
) -> None:
    azimuth_sectors = {
        int(item["variation"]["camera_pose"]["viewpoint"]["azimuth_deg"] // 30)
        for item in assignments
    }
    if azimuth_sectors != set(range(12)):
        raise ValueError("catalog must cover all twelve operational azimuth sectors")

    if len(progression_profiles) < MIN_PROGRESSION_PROFILES:
        raise ValueError(
            f"catalog requires at least {MIN_PROGRESSION_PROFILES} fire progression profiles"
        )
    if max(progression_profiles.values()) > math.ceil(sum(progression_profiles.values()) * 0.10):
        raise ValueError("one fire progression profile is over-represented")
    for family in ("rural", "mountain", "agricultural"):
        if landscape_families[family] < math.ceil(MIN_FIRE_EVENTS * 0.10):
            raise ValueError(f"catalog under-represents French {family} terrain")


def _assert_fire_coverage(
    *, contract: dict[str, Any], variations: list[dict[str, Any]]
) -> None:
    phases = {item["flow"]["progression"]["phase"] for item in variations}
    missing_phases = REQUIRED_CAMPAIGN_PROGRESSION_PHASES - phases
    if missing_phases:
        raise ValueError(
            f"each fire must cover progression phases: {sorted(missing_phases)}"
        )
    times = {item["lighting"]["time_of_day"] for item in variations}
    if not {"day", "night"}.issubset(times):
        raise ValueError("each fire must include both day and night observations")
    distances = {
        item["camera_pose"]["viewpoint"]["distance_band"] for item in variations
    }
    if not {"near", "very_far"}.issubset(distances):
        raise ValueError("each fire must include near and very-far observations")
    occlusions = {
        item["camera_pose"]["viewpoint"]["occlusion"] for item in variations
    }
    required_occlusions = {"partial_building", "partial_mountain"}
    if not required_occlusions.issubset(occlusions):
        raise ValueError("each fire must include building and mountain partial occlusions")
    selected_poses = {item["camera_pose"]["id"] for item in variations}
    operational_poses = {
        item["id"] for item in contract["composition"]["camera_poses"]
    }
    if selected_poses != operational_poses:
        raise ValueError("each fire must use every validated operational viewpoint")


def load_event_catalog(
    path: Path,
    *,
    volume_root: Path,
    target_per_category: int,
    contract_loader: Callable[..., dict[str, Any]] = load_real_world_contract,
) -> dict[str, Any]:
    """Load 512+ distinct fires and materialize their bounded case assignments."""
    volume = volume_root.resolve()
    catalog_path = path.resolve()
    if catalog_path != volume and volume not in catalog_path.parents:
        raise ValueError("event catalog must remain inside the production volume")
    if not catalog_path.is_file():
        raise ValueError(f"event catalog is absent: {catalog_path}")
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported event catalog schema_version")
    if payload.get("data_origin") != "new_synthetic_generation":
        raise ValueError("event catalog must schedule new generated cases")
    if int(payload.get("minimum_fire_events", 0)) != MIN_FIRE_EVENTS:
        raise ValueError("event catalog must lock at least 512 distinct fires")
    if int(payload.get("maximum_fire_duration_days", 0)) != MAX_FIRE_DURATION_DAYS:
        raise ValueError("event catalog must cap each fire at fifteen days")
    if (
        int(payload.get("max_cases_per_fire_per_category", 0))
        != MAX_CASES_PER_FIRE_PER_CATEGORY
    ):
        raise ValueError("event catalog must cap each fire at twenty-four cases per category")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or len(raw_events) < MIN_FIRE_EVENTS:
        raise ValueError("event catalog requires at least 512 fire events")

    events: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    contract_hashes: set[str] = set()
    duration_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    progression_profiles: Counter[tuple[str, ...]] = Counter()
    landscape_families: Counter[str] = Counter()
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError("event catalog entries must be objects")
        event_id = str(raw.get("event_id", "")).strip()
        if not event_id or event_id in event_ids:
            raise ValueError("event ids must be non-empty and unique")
        event_ids.add(event_id)
        slots = int(raw.get("case_slots_per_category", 0))
        if not MIN_CASES_PER_FIRE_PER_CATEGORY <= slots <= MAX_CASES_PER_FIRE_PER_CATEGORY:
            raise ValueError("each fire must schedule between four and twenty-four cases")
        image_counts[slots] += 1

        contract_path = _inside_volume(
            raw.get("real_world_contract"),
            root=volume,
            field=f"events[{index}].real_world_contract",
        )
        expected_digest = str(raw.get("real_world_contract_sha256", ""))
        if not SHA256.fullmatch(expected_digest):
            raise ValueError("event contract requires a lowercase SHA-256")
        actual_digest = _sha256(contract_path)
        if not hmac.compare_digest(expected_digest, actual_digest):
            raise ValueError("event contract SHA-256 mismatch")
        if actual_digest in contract_hashes:
            raise ValueError("distinct fires require distinct event contracts")
        contract_hashes.add(actual_digest)
        contract = contract_loader(contract_path, volume_root=volume)
        if contract.get("event_id") != event_id:
            raise ValueError("catalog event_id does not match its real-world contract")
        duration_days = int(contract.get("duration_days", 0))
        if not 1 <= duration_days <= MAX_FIRE_DURATION_DAYS:
            raise ValueError("each fire duration must be between one and fifteen days")
        duration_counts[duration_days] += 1
        landscape_profile = str(
            contract.get("geospatial", {}).get("landscape_profile", "")
        )
        for family in landscape_profile.split("_"):
            landscape_families[family] += 1
        if int(contract["composition"]["diversity"]["capacity_per_category"]) < slots:
            raise ValueError("event contract cannot satisfy its scheduled case slots")
        phases = tuple(
            state["progression"]["phase"]
            for state in contract["composition"]["flow_states"]
        )
        progression_profiles[phases] += 1
        event = {
            **raw,
            "event_id": event_id,
            "duration_days": duration_days,
            "case_slots_per_category": slots,
            "real_world_contract": contract_path,
            "real_world_contract_sha256": actual_digest,
            "contract": contract,
        }
        events.append(event)
        event_variations: list[dict[str, Any]] = []
        for event_slot in range(slots):
            variation = select_case_variation(contract, event_slot)
            event_variations.append(variation)
            assignments.append(
                {
                    "case_index": len(assignments),
                    "event_slot": event_slot,
                    "event": event,
                    "variation": variation,
                }
            )
        _assert_fire_coverage(contract=contract, variations=event_variations)

    if len(assignments) != target_per_category:
        raise ValueError(
            "event catalog case-slot total must equal the production target exactly"
        )
    if set(duration_counts) != set(range(1, MAX_FIRE_DURATION_DAYS + 1)):
        raise ValueError("fire durations must cover every value from one to fifteen days")
    if len(image_counts) < MIN_DISTINCT_IMAGE_COUNTS:
        raise ValueError("the number of images per fire must vary across the catalog")
    _assert_coverage(
        assignments=assignments,
        progression_profiles=progression_profiles,
        landscape_families=landscape_families,
    )
    return {
        **payload,
        "catalog_path": catalog_path,
        "events": events,
        "assignments": assignments,
        "coverage": {
            "fire_events": len(events),
            "fire_duration_days": sorted(duration_counts),
            "case_slots_per_category": len(assignments),
            "distinct_image_counts_per_fire": sorted(image_counts),
            "progression_profiles": len(progression_profiles),
            "landscape_families": dict(landscape_families),
        },
    }


def case_assignment(catalog: dict[str, Any], case_index: int) -> dict[str, Any]:
    assignments = catalog["assignments"]
    if not 0 <= case_index < len(assignments):
        raise ValueError("case_index is outside the validated event catalog")
    return dict(assignments[case_index])
