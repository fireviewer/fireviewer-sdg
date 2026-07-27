from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.review_store import (  # noqa: E402
    REQUIRED_QUALITY_CHECKS,
    CaseStore,
)
from fireviewer_sdg.real_world import (  # noqa: E402
    LOCAL_RENDER_PROFILE,
    LOCAL_RENDER_REVISION,
    RENDER_PROFILE,
    RENDER_REVISION,
)


def _case(volume: Path, case_id: str = "case-0001") -> dict[str, object]:
    preview = volume / "production" / "payloads" / f"{case_id}.png"
    annotations = volume / "production" / "payloads" / f"{case_id}.json"
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"preview")
    annotations.write_bytes(b"annotations")
    return {
        "schema_version": 1,
        "category": "terrestrial_fire_points",
        "case_id": case_id,
        "data_origin": "new_synthetic_generation",
        "production_stage": "pilot",
        "seed": 123,
        "preview_relpath": preview.relative_to(volume).as_posix(),
        "overlays": [
            {
                "kind": "point",
                "label": "active_fire_point",
                "x_normalized": 0.4,
                "y_normalized": 0.7,
            },
            {
                "kind": "point",
                "label": "visible_fire_front_point",
                "x_normalized": 0.5,
                "y_normalized": 0.72,
            },
            {
                "kind": "point",
                "label": "smoke_column_base",
                "x_normalized": 0.45,
                "y_normalized": 0.55,
            },
        ],
        "artifacts": [
            {
                "kind": "ground_photo",
                "relpath": preview.relative_to(volume).as_posix(),
                "sha256": hashlib.sha256(b"preview").hexdigest(),
            },
            {
                "kind": "point_annotations",
                "relpath": annotations.relative_to(volume).as_posix(),
                "sha256": hashlib.sha256(b"annotations").hexdigest(),
            },
        ],
        "performance": {
            "measurement": "observed_pilot_case_v1",
            "elapsed_seconds": 1.5,
            "artifact_bytes": len(b"preview") + len(b"annotations"),
            "record_bytes": 1000,
            "case_output_bytes": len(b"preview") + len(b"annotations") + 1000,
            "vram_measurement": "nvidia_smi_device_total_memory",
            "vram_baseline_bytes": 1024 * 1024 * 1024,
            "vram_peak_bytes": 1280 * 1024 * 1024,
            "vram_delta_peak_bytes": 256 * 1024 * 1024,
            "vram_sample_count": 4,
        },
        "truth": {
            "synthetic": True,
            "real_world_claim": False,
            "background_source": "new_real_world_capture_nurec_reconstruction",
            "capture_manifest_sha256": "1" * 64,
            "scene_asset_sha256": "2" * 64,
            "human_review_required": True,
            "usable_for_training": False,
            "event_id": "fire-test-0001",
            "fire_duration_days": 3,
            "landscape_profile": "rural_mountain",
            "progression": {"phase": "advancing_flame_zone"},
        },
        "camera": {
            "model": "pinhole",
            "position": [0.0, 0.0, 1.0],
            "axis": {"forward": [0.0, 1.0, 0.0]},
            "intrinsics": {"fx": 1000.0},
        },
        "render": {
            "profile": RENDER_PROFILE,
            "revision": RENDER_REVISION,
            "camera_pose_id": "pose-001",
            "rt_subframes": 16,
            "warmup_steps": 32,
            "variation_id": "variation-000001",
            "lighting_variant_id": "light-1",
            "flow_state_id": "flow-1",
            "time_of_day": "day",
            "diversity_signature": "4" * 64,
            "viewpoint": {
                "distance_band": "near",
                "occlusion": "partial_mountain",
                "reference_validation": "pending_console_review",
            },
        },
    }


class ReviewStoreTests(unittest.TestCase):
    def test_local_campaign_accepts_720p_revision_and_hides_excluded_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            record = _case(volume)
            record["render"]["profile"] = LOCAL_RENDER_PROFILE
            record["render"]["revision"] = LOCAL_RENDER_REVISION
            store = CaseStore(
                volume,
                active_categories=(
                    "terrestrial_fire_points",
                    "france_cross_view",
                    "france_incident_days",
                ),
                render_revision=LOCAL_RENDER_REVISION,
            )
            store.register(record)
            self.assertEqual(
                [item["category"] for item in store.status()["categories"]],
                [
                    "terrestrial_fire_points",
                    "france_cross_view",
                    "france_incident_days",
                ],
            )
            with self.assertRaisesRegex(ValueError, "unsupported case category"):
                store.list(
                    category="response_engagement",
                    offset=0,
                    limit=10,
                )

    def test_register_review_and_status_remain_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            registered = store.register(_case(volume))
            self.assertEqual(registered["case_id"], "case-0001")
            listing = store.list(
                category="terrestrial_fire_points", offset=0, limit=10
            )
            self.assertEqual(listing["total"], 1)
            self.assertIsNone(listing["items"][0]["review"])

            review = store.review(
                category="terrestrial_fire_points",
                case_id="case-0001",
                decision="accepted",
                reviewer="operator-1",
                notes="overlay verified",
                quality_checks={
                    key: True
                    for key in REQUIRED_QUALITY_CHECKS["terrestrial_fire_points"]
                },
            )
            self.assertEqual(review["decision"], "accepted")
            status = store.status()
            category = status["categories"][0]
            self.assertEqual(category["produced"], 1)
            self.assertEqual(category["accepted"], 1)
            self.assertEqual(
                category["pilot_measurements"]["observed_case_count"],
                1,
            )
            self.assertEqual(
                category["pilot_measurements"]["rejection_rate_reviewed"],
                0.0,
            )
            self.assertEqual(
                category["pilot_measurements"]["vram_peak_bytes_max"],
                1280 * 1024 * 1024,
            )
            self.assertFalse(status["export_ready"])
            events = [
                json.loads(line)
                for line in (
                    volume / "production" / "reviews" / "review-events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[-1]["reviewer"], "operator-1")

    def test_acceptance_requires_every_quality_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            store.register(_case(volume))
            with self.assertRaisesRegex(ValueError, "mandatory quality checks"):
                store.review(
                    category="terrestrial_fire_points",
                    case_id="case-0001",
                    decision="accepted",
                    reviewer="operator-1",
                    notes="looks plausible",
                    quality_checks={},
                )

    def test_idempotent_register_refuses_case_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            record = _case(volume)
            store.register(record)
            store.register(record)
            changed = dict(record)
            changed["seed"] = 124
            with self.assertRaisesRegex(RuntimeError, "collision"):
                store.register(changed)

    def test_preview_path_cannot_escape_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            record = _case(volume)
            record["preview_relpath"] = "../outside.png"
            with self.assertRaisesRegex(ValueError, "invalid preview_relpath"):
                store.register(record)

    def test_existing_corpus_cannot_be_registered_as_new_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            record = _case(volume)
            record["data_origin"] = "existing_corpus"
            with self.assertRaisesRegex(ValueError, "new synthetic"):
                store.register(record)

    def test_registration_recalculates_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            record = _case(volume)
            record["artifacts"][0]["sha256"] = "0" * 64  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                store.register(record)

    def test_all_three_point_semantics_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            record = _case(volume)
            record["overlays"] = record["overlays"][:2]  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "exactly all three point"):
                store.register(record)


if __name__ == "__main__":
    unittest.main()
