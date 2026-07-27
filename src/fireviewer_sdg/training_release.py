"""Fail-closed, immutable training release manifests for reviewed case packages."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fireviewer_sdg.artifacts import sha256, write_json
from fireviewer_sdg.review_store import (
    REQUIRED_QUALITY_CHECKS,
    RESPONSE_CLASSES,
    TARGET_PER_CATEGORY,
    CaseStore,
    validate_case_record,
)


MIN_FIRE_EVENTS = 512
MAX_CASES_PER_FIRE = 24
REQUIRED_PROGRESSION_PHASES = {"advancing_flame_zone", "front_split", "reignition"}


TRAINING_SCHEMA = "fireviewer_training_release_v1"
PRIMARY_ARTIFACT_KIND = {
    "terrestrial_fire_points": "ground_photo",
    "france_cross_view": "ground_photo",
    "response_engagement": "ground_photo",
    "france_incident_days": "source_packet",
}


class TrainingReleaseLocked(RuntimeError):
    """Raised when accepted data do not satisfy the training contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _accepted_records(store: CaseStore, category: str) -> list[dict[str, Any]]:
    offset = 0
    records: list[dict[str, Any]] = []
    while True:
        page = store.list(
            category=category,
            offset=offset,
            limit=100,
            review_state="accepted",
        )
        records.extend(page["items"])
        offset += len(page["items"])
        if offset >= int(page["total"]):
            return records


def _integrity_entry(store: CaseStore, item: dict[str, Any]) -> dict[str, Any]:
    review = item.get("review")
    if not isinstance(review, dict) or review.get("decision") != "accepted":
        raise TrainingReleaseLocked("training release contains a case without accepted review")
    category = str(item.get("category", ""))
    quality_checks = review.get("quality_checks")
    required_checks = REQUIRED_QUALITY_CHECKS.get(category, frozenset())
    if (
        not isinstance(quality_checks, dict)
        or any(quality_checks.get(key) is not True for key in required_checks)
    ):
        raise TrainingReleaseLocked(
            f"{category}/{item.get('case_id')}: accepted review lacks mandatory quality evidence"
        )
    source_record = {key: value for key, value in item.items() if key != "review"}
    try:
        record = validate_case_record(source_record, store.volume_root)
    except (OSError, ValueError) as exc:
        raise TrainingReleaseLocked(
            f"case integrity validation failed: {item.get('category')}/{item.get('case_id')}: {exc}"
        ) from exc
    return {
        "schema": TRAINING_SCHEMA,
        "category": record["category"],
        "case_id": record["case_id"],
        "seed": record["seed"],
        "production_stage": record["production_stage"],
        "record_relpath": (
            Path("production")
            / "cases"
            / record["category"]
            / record["case_id"]
            / "case.json"
        ).as_posix(),
        "artifacts": record["artifacts"],
        "truth": record["truth"],
        "camera": record.get("camera"),
        "render": record.get("render"),
        "review": {
            "decision": "accepted",
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
            "quality_checks": {
                key: True for key in sorted(required_checks)
            },
        },
        "usable_for_training": True,
    }


def _primary_digest(entry: dict[str, Any]) -> str:
    kind = PRIMARY_ARTIFACT_KIND[entry["category"]]
    matches = [artifact["sha256"] for artifact in entry["artifacts"] if artifact["kind"] == kind]
    if len(matches) != 1:
        raise TrainingReleaseLocked(
            f"{entry['category']}/{entry['case_id']} must contain exactly one {kind} artifact"
        )
    return str(matches[0])


def _validate_selected(category: str, entries: list[dict[str, Any]], expected: int) -> None:
    identifiers = [entry["case_id"] for entry in entries]
    seeds = [entry["seed"] for entry in entries]
    primary_digests = [_primary_digest(entry) for entry in entries]
    diversity_signatures = [
        str(entry.get("render", {}).get("diversity_signature", ""))
        for entry in entries
        if entry.get("render") is not None
    ]
    if len(set(identifiers)) != expected:
        raise TrainingReleaseLocked(f"{category} contains duplicate case identifiers")
    if len(set(seeds)) != expected:
        raise TrainingReleaseLocked(f"{category} contains duplicate generation seeds")
    if len(set(primary_digests)) != expected:
        raise TrainingReleaseLocked(f"{category} contains duplicate primary training payloads")
    if category != "france_incident_days" and len(set(diversity_signatures)) != expected:
        raise TrainingReleaseLocked(f"{category} contains duplicate planned visual variations")

    if category != "france_incident_days" and expected >= TARGET_PER_CATEGORY:
        by_event: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            event_id = str(entry["truth"].get("event_id", ""))
            by_event.setdefault(event_id, []).append(entry)
        if len(by_event) < MIN_FIRE_EVENTS:
            raise TrainingReleaseLocked("visual release requires at least 512 distinct fires")
        event_counts = [len(items) for items in by_event.values()]
        if any(not 4 <= count <= MAX_CASES_PER_FIRE for count in event_counts):
            raise TrainingReleaseLocked("each fire must contribute between 4 and 24 views")
        if len(set(event_counts)) < 4:
            raise TrainingReleaseLocked("the number of accepted views per fire must vary")
        durations = {
            int(entry["truth"].get("fire_duration_days", 0)) for entry in entries
        }
        if durations != set(range(1, 16)):
            raise TrainingReleaseLocked("accepted fires must cover durations from 1 to 15 days")
        for event_id, event_entries in by_event.items():
            phases = {
                str(entry["truth"].get("progression", {}).get("phase", ""))
                for entry in event_entries
            }
            if not REQUIRED_PROGRESSION_PHASES.issubset(phases):
                raise TrainingReleaseLocked(f"fire {event_id} lacks progression coverage")
            times = {str(entry["render"].get("time_of_day", "")) for entry in event_entries}
            if not {"day", "night"}.issubset(times):
                raise TrainingReleaseLocked(f"fire {event_id} lacks day/night coverage")
            distances = {
                str(entry["render"].get("viewpoint", {}).get("distance_band", ""))
                for entry in event_entries
            }
            if not {"near", "very_far"}.issubset(distances):
                raise TrainingReleaseLocked(f"fire {event_id} lacks near/very-far coverage")
            occlusions = {
                str(entry["render"].get("viewpoint", {}).get("occlusion", ""))
                for entry in event_entries
            }
            if not {"partial_building", "partial_mountain"}.issubset(occlusions):
                raise TrainingReleaseLocked(f"fire {event_id} lacks occlusion coverage")

    if category == "response_engagement":
        counts = {class_id: 0 for class_id in RESPONSE_CLASSES}
        for entry in entries:
            counts[str(entry["truth"].get("object_class", ""))] += 1
        lower = expected // len(RESPONSE_CLASSES)
        upper = math.ceil(expected / len(RESPONSE_CLASSES))
        if any(count not in {lower, upper} for count in counts.values()):
            raise TrainingReleaseLocked(
                "response training release must balance all seven positive and hard-negative classes"
            )


def build_training_release(
    store: CaseStore,
    *,
    expected_per_category: int | None = None,
) -> dict[str, Any]:
    """Audit accepted cases and atomically create local manifests; never copies payloads."""
    if expected_per_category is None:
        expected_per_category = store.target_per_category
    if expected_per_category < 1:
        raise ValueError("expected_per_category must be positive")
    selected: dict[str, list[dict[str, Any]]] = {}
    for category in store.categories:
        accepted = sorted(_accepted_records(store, category), key=lambda item: item["case_id"])
        if len(accepted) < expected_per_category:
            raise TrainingReleaseLocked(
                f"{category}: accepted={len(accepted)} required={expected_per_category}"
            )
        entries = [
            _integrity_entry(store, item)
            for item in accepted[:expected_per_category]
        ]
        _validate_selected(category, entries, expected_per_category)
        selected[category] = entries

    inventory = {
        category: [
            {
                "case_id": entry["case_id"],
                "seed": entry["seed"],
                "artifacts": [artifact["sha256"] for artifact in entry["artifacts"]],
                "reviewed_at": entry["review"]["reviewed_at"],
            }
            for entry in entries
        ]
        for category, entries in selected.items()
    }
    inventory_sha256 = hashlib.sha256(_canonical(inventory).encode()).hexdigest()
    release_id = f"training-{inventory_sha256[:20]}"
    releases_root = store.volume_root / "training" / "releases"
    release_root = releases_root / release_id
    manifest_path = release_root / "release.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if release_root.exists():
        raise TrainingReleaseLocked(f"incomplete release directory already exists: {release_id}")

    releases_root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{release_id}-", dir=releases_root))
    category_manifests: dict[str, dict[str, Any]] = {}
    for category, entries in selected.items():
        path = temporary_root / f"{category}.jsonl"
        path.write_text("".join(f"{_canonical(entry)}\n" for entry in entries), encoding="utf-8")
        category_manifests[category] = {
            "path": path.name,
            "case_count": len(entries),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "schema": TRAINING_SCHEMA,
        "release_id": release_id,
        "created_at": _utc_now(),
        "data_origin": "new_synthetic_generation",
        "transfer_performed": False,
        "expected_per_category": expected_per_category,
        "total_cases": expected_per_category * len(store.categories),
        "inventory_sha256": inventory_sha256,
        "categories": category_manifests,
        "guard": "all_cases_integrity_checked_and_human_accepted",
    }
    write_json(temporary_root / "release.json", manifest)
    temporary_root.replace(release_root)
    store.log_event(
        "training_release_created",
        release_id=release_id,
        total_cases=manifest["total_cases"],
        transfer_performed=False,
    )
    return manifest
