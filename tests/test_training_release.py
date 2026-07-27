from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.incident_days import generate_incident_day  # noqa: E402
from fireviewer_sdg.review_store import (  # noqa: E402
    REQUIRED_QUALITY_CHECKS,
    RESPONSE_CLASSES,
    CaseStore,
)
from fireviewer_sdg.training_release import (  # noqa: E402
    TrainingReleaseLocked,
    build_training_release,
)
from fireviewer_sdg.real_world import RENDER_PROFILE, RENDER_REVISION  # noqa: E402


def _artifact(volume: Path, path: Path, kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "relpath": path.relative_to(volume).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _camera() -> dict[str, object]:
    return {
        "position": [0.0, -20.0, 6.0],
        "axis": {"forward": [0.0, 1.0, -0.2]},
        "intrinsics": {"fx": 1800.0, "fy": 1800.0, "cx": 1024.0, "cy": 768.0},
    }


def _visual_truth(**extra: object) -> dict[str, object]:
    return {
        "synthetic": True,
        "real_world_claim": False,
        "background_source": "new_real_world_capture_nurec_reconstruction",
        "capture_manifest_sha256": "1" * 64,
        "scene_asset_sha256": "2" * 64,
        "human_review_required": True,
        "usable_for_training": False,
        "event_id": "fire-test-0001",
        "fire_duration_days": 3,
        "landscape_profile": "rural_agricultural",
        "progression": {"phase": "advancing_flame_zone"},
        **extra,
    }


def _render(index: int) -> dict[str, object]:
    return {
        "profile": RENDER_PROFILE,
        "revision": RENDER_REVISION,
        "camera_pose_id": f"pose-{index:03d}",
        "rt_subframes": 16,
        "warmup_steps": 32,
        "variation_id": f"variation-{index:06d}",
        "lighting_variant_id": f"light-{index % 8}",
        "flow_state_id": f"flow-{index % 8}",
        "time_of_day": "day" if index % 2 == 0 else "night",
        "diversity_signature": hashlib.sha256(f"variation-{index}".encode()).hexdigest(),
        "viewpoint": {
            "distance_band": "near" if index % 2 == 0 else "very_far",
            "occlusion": "partial_building" if index % 2 == 0 else "partial_mountain",
            "reference_validation": "pending_console_review",
        },
    }


def _record(volume: Path, category: str, index: int) -> dict[str, object]:
    prefixes = {
        "terrestrial_fire_points": "tfp",
        "france_cross_view": "fcv",
        "response_engagement": "reg",
    }
    case_id = f"{prefixes[category]}-{index:06d}"
    root = volume / "payloads" / category / case_id
    root.mkdir(parents=True, exist_ok=True)
    photo = root / "ground.jpg"
    photo.write_bytes(f"unique-photo-{category}-{index}".encode())
    common: dict[str, object] = {
        "schema_version": 1,
        "category": category,
        "case_id": case_id,
        "data_origin": "new_synthetic_generation",
        "production_stage": "bulk",
        "seed": 100_000 * (list(prefixes).index(category) + 1) + index,
        "preview_relpath": photo.relative_to(volume).as_posix(),
        "camera": _camera(),
        "render": _render(index),
    }
    if category == "terrestrial_fire_points":
        annotations = root / "points.json"
        annotations.write_bytes(f"points-{index}".encode())
        common.update(
            overlays=[
                {"kind": "point", "label": "active_fire_point", "x_normalized": 0.4, "y_normalized": 0.6},
                {"kind": "point", "label": "visible_fire_front_point", "x_normalized": 0.5, "y_normalized": 0.61},
                {"kind": "point", "label": "smoke_column_base", "x_normalized": 0.45, "y_normalized": 0.5},
            ],
            artifacts=[_artifact(volume, photo, "ground_photo"), _artifact(volume, annotations, "point_annotations")],
            truth=_visual_truth(),
        )
    elif category == "france_cross_view":
        ortho = root / "ortho.tif"
        mnt = root / "mnt.tif"
        site = root / "site.json"
        ortho.write_bytes(f"ortho-{index}".encode())
        mnt.write_bytes(f"mnt-{index}".encode())
        site.write_bytes(f"site-{index}".encode())
        common.update(
            overlays=[],
            artifacts=[
                _artifact(volume, photo, "ground_photo"),
                _artifact(volume, ortho, "orthophoto"),
                _artifact(volume, mnt, "mnt"),
                _artifact(volume, site, "site_manifest"),
            ],
            truth=_visual_truth(
                site_code="fr-site-one",
                fire_position_verified_from_generator=True,
            ),
        )
    else:
        classes = sorted(RESPONSE_CLASSES)
        class_id = classes[index % len(classes)]
        annotations = root / "boxes.json"
        annotations.write_bytes(f"boxes-{index}".encode())
        common.update(
            overlays=[
                {
                    "kind": "box",
                    "label": class_id,
                    "x_min": 0.2,
                    "y_min": 0.2,
                    "x_max": 0.7,
                    "y_max": 0.7,
                }
            ],
            artifacts=[_artifact(volume, photo, "ground_photo"), _artifact(volume, annotations, "box_annotations")],
            truth=_visual_truth(
                object_class=class_id,
                engagement_label=(
                    "not_engaged_hard_negative"
                    if class_id.startswith("hard_negative")
                    else "simulated_engagement"
                ),
                operational_truth="synthetic_only",
                actor_asset_sha256="3" * 64,
                target_actor_isolated=True,
            ),
        )
    artifact_bytes = sum(
        (volume / str(artifact["relpath"])).stat().st_size
        for artifact in common["artifacts"]
    )
    record_bytes = 1_000
    common["performance"] = {
        "measurement": "observed_pilot_case_v1",
        "elapsed_seconds": 1.5,
        "artifact_bytes": artifact_bytes,
        "record_bytes": record_bytes,
        "case_output_bytes": artifact_bytes + record_bytes,
        "vram_measurement": "nvidia_smi_device_total_memory",
        "vram_baseline_bytes": 1024 * 1024 * 1024,
        "vram_peak_bytes": 1280 * 1024 * 1024,
        "vram_delta_peak_bytes": 256 * 1024 * 1024,
        "vram_sample_count": 4,
    }
    return common


class TrainingReleaseTests(unittest.TestCase):
    def test_release_is_locked_before_all_categories_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = CaseStore(Path(directory))
            with self.assertRaisesRegex(TrainingReleaseLocked, "accepted=0 required=4096"):
                build_training_release(store)

    def test_release_audits_and_writes_only_immutable_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            for category in (
                "terrestrial_fire_points",
                "france_cross_view",
                "response_engagement",
            ):
                for index in range(7):
                    record = store.register(_record(volume, category, index))
                    store.review(
                        category=category,
                        case_id=record["case_id"],
                        decision="accepted",
                        reviewer="operator-test",
                        notes="verified",
                        quality_checks={
                            key: True for key in REQUIRED_QUALITY_CHECKS[category]
                        },
                    )
            for index in range(7):
                batch_root = volume / "incident-batches" / str(index)
                record_path = generate_incident_day(
                    volume_root=volume,
                    batch_root=batch_root,
                    case_index=index,
                    seed=900_000 + index,
                    production_stage="bulk",
                )
                import json

                record = store.register(json.loads(record_path.read_text(encoding="utf-8")))
                store.review(
                    category="france_incident_days",
                    case_id=record["case_id"],
                    decision="accepted",
                    reviewer="operator-test",
                    notes="verified",
                    quality_checks={
                        key: True
                        for key in REQUIRED_QUALITY_CHECKS["france_incident_days"]
                    },
                )

            release = build_training_release(store, expected_per_category=7)
            self.assertEqual(release["total_cases"], 28)
            self.assertFalse(release["transfer_performed"])
            release_root = volume / "training" / "releases" / release["release_id"]
            self.assertTrue((release_root / "release.json").is_file())
            self.assertEqual(len(list(release_root.glob("*.jsonl"))), 4)
            self.assertEqual(build_training_release(store, expected_per_category=7)["release_id"], release["release_id"])


if __name__ == "__main__":
    unittest.main()
