from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.case_generation import load_batch_spec  # noqa: E402
from fireviewer_sdg.production import (  # noqa: E402
    ProductionManager,
    _batches,
    _event_aligned_batches,
    _load_batch_records,
    load_production_plan,
)
from fireviewer_sdg.review_store import CATEGORY_IDS, CaseStore  # noqa: E402


class ProductionContractTests(unittest.TestCase):
    def test_bundled_plan_defines_8192_new_disjoint_cases(self) -> None:
        payload = load_production_plan(
            ROOT / "campaigns" / "fireviewer-new-synthetic-cases-v1.json"
        )
        self.assertEqual(payload["data_origin"], "new_synthetic_generation")
        self.assertEqual(payload["target_per_category"], 4096)
        self.assertEqual(payload["maximum_target_per_category"], 8192)
        self.assertEqual(set(payload["by_category"]), set(CATEGORY_IDS))
        seeds = {
            int(item["seed_base"]) + index
            for item in payload["categories"]
            for index in range(4096)
        }
        self.assertEqual(len(seeds), 16384)

    def test_bundled_double_plan_defines_16384_new_disjoint_cases(self) -> None:
        payload = load_production_plan(
            ROOT / "campaigns" / "fireviewer-new-synthetic-cases-v1-double.json"
        )
        self.assertEqual(payload["target_per_category"], 8192)
        seeds = {
            int(item["seed_base"]) + index
            for item in payload["categories"]
            for index in range(8192)
        }
        self.assertEqual(len(seeds), 32768)

    def test_local_windows_plan_is_locked_to_720p(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("FW_SDG_VOLUME_ROOT")
            os.environ["FW_SDG_VOLUME_ROOT"] = directory
            try:
                payload = load_production_plan(
                    ROOT
                    / "campaigns"
                    / "fireviewer-new-synthetic-cases-local-720p-v1.json"
                )
            finally:
                if previous is None:
                    os.environ.pop("FW_SDG_VOLUME_ROOT", None)
                else:
                    os.environ["FW_SDG_VOLUME_ROOT"] = previous
        self.assertEqual(payload["resolution"], [1280, 720])
        self.assertEqual(
            payload["render_profile"],
            "omniverse_realworld_local_720p_v1",
        )
        self.assertEqual(
            payload["render_revision"],
            "realworld-local-720p-composite-gate-v2",
        )
        self.assertEqual(
            Path(payload["real_world_catalog"]),
            Path(directory) / "input" / "event-catalog-4096-hd-v2.json",
        )
        self.assertEqual(
            payload["active_categories"],
            (
                "terrestrial_fire_points",
                "france_cross_view",
                "france_incident_days",
            ),
        )
        self.assertEqual(payload["excluded_categories"], ("response_engagement",))

    def test_render_profiles_reject_crossed_resolutions(self) -> None:
        source = json.loads(
            (
                ROOT / "campaigns" / "fireviewer-new-synthetic-cases-v1.json"
            ).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            source["resolution"] = [1280, 720]
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires exactly 1920x1080"):
                load_production_plan(path)

    def test_bulk_batches_exclude_the_eight_pilot_indices(self) -> None:
        batches = _batches(8, 4096, 64)
        self.assertEqual(batches[0], (8, 64))
        self.assertEqual(batches[-1], (4040, 56))
        self.assertEqual(sum(count for _, count in batches), 4088)

    def test_visual_batches_never_cross_fire_event_boundaries(self) -> None:
        assignments = [
            {"event": {"event_id": event_id}}
            for event_id in ["fire-a"] * 4 + ["fire-b"] * 6 + ["fire-c"] * 5
        ]
        batches = _event_aligned_batches(assignments, 0, 15, 4)
        self.assertEqual(
            batches,
            [(0, 4), (4, 4), (8, 2), (10, 4), (14, 1)],
        )
        for start, count in batches:
            event_ids = {
                item["event"]["event_id"]
                for item in assignments[start : start + count]
            }
            self.assertEqual(len(event_ids), 1)

    def test_event_aligned_bulk_can_start_inside_a_fire(self) -> None:
        assignments = [
            {"event": {"event_id": event_id}}
            for event_id in ["fire-a"] * 10 + ["fire-b"] * 4
        ]
        self.assertEqual(
            _event_aligned_batches(assignments, 8, 14, 64),
            [(8, 2), (10, 4)],
        )

    def test_partial_current_visual_batch_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            records = batch / "records"
            records.mkdir()
            (records / "tfp-000000.json").write_text(
                json.dumps(
                    {
                        "case_id": "tfp-000000",
                        "render": {
                            "revision": "realworld-hd-composite-gate-v20"
                        },
                    }
                ),
                encoding="utf-8",
            )
            loaded = _load_batch_records(
                batch_root=batch,
                category="terrestrial_fire_points",
                start=0,
                count=4,
            )
            self.assertEqual([item["case_id"] for item in loaded], ["tfp-000000"])

    def test_batch_spec_is_bounded_to_new_generation_volume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            batch = volume / "production" / "batches" / "test"
            spec_path = Path(directory) / "batch.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "production_id": "fireviewer-new-synthetic-cases-v1",
                        "data_origin": "new_synthetic_generation",
                        "production_stage": "pilot",
                        "category": "france_incident_days",
                        "generator": "fictional_incident_day",
                        "case_start": 0,
                        "case_count": 8,
                        "seed_base": 640000000,
                        "resolution": [1920, 1080],
                        "render_profile": "omniverse_realworld_hd_v1",
                        "real_world_catalog": str(
                            volume / "input" / "event-catalog-4096-hd-v2.json"
                        ),
                        "target_per_category": 4096,
                        "rt_subframes": 16,
                        "warmup_steps": 32,
                        "volume_root": str(volume),
                        "batch_root": str(batch),
                    }
                ),
                encoding="utf-8",
            )
            parsed = load_batch_spec(spec_path)
            self.assertEqual(parsed["case_count"], 8)
            self.assertEqual(parsed["volume_root"], volume.resolve())

    def test_local_batch_spec_keeps_720p_revision_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            batch = volume / "production" / "batches" / "local"
            spec_path = Path(directory) / "batch-local.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "production_id": "fireviewer-new-synthetic-cases-v1",
                        "data_origin": "new_synthetic_generation",
                        "production_stage": "pilot",
                        "category": "france_incident_days",
                        "generator": "fictional_incident_day",
                        "case_start": 0,
                        "case_count": 8,
                        "seed_base": 640000000,
                        "resolution": [1280, 720],
                        "render_profile": "omniverse_realworld_local_720p_v1",
                        "real_world_catalog": str(
                            volume / "input" / "event-catalog-4096-hd-v2.json"
                        ),
                        "target_per_category": 4096,
                        "rt_subframes": 16,
                        "warmup_steps": 48,
                        "volume_root": str(volume),
                        "batch_root": str(batch),
                    }
                ),
                encoding="utf-8",
            )
            parsed = load_batch_spec(spec_path)
        self.assertEqual(parsed["resolution"], [1280, 720])
        self.assertEqual(
            parsed["render_revision"],
            "realworld-local-720p-composite-gate-v2",
        )

    def test_bulk_is_locked_before_pilot_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            store = CaseStore(volume)
            manager = ProductionManager(volume, store)
            with self.assertRaisesRegex(RuntimeError, "8 pilots"):
                manager.continue_bulk(
                    ROOT / "campaigns" / "fireviewer-new-synthetic-cases-v1.json"
                )


if __name__ == "__main__":
    unittest.main()
