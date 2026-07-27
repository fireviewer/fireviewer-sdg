from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from itertools import permutations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.event_catalog import case_assignment, load_event_catalog  # noqa: E402


PHASES = [
    "advancing_flame_zone",
    "front_split",
    "reignition",
    "initial_growth",
    "partial_suppression",
    "multi_front_spread",
    "decay",
]
PROFILES = []
for fourth in PHASES[3:]:
    PROFILES.extend(tuple(value) for value in permutations(PHASES[:3] + [fourth]))


def _contract(event_id: str, index: int) -> dict[str, object]:
    distances = ["near", "medium", "far", "very_far"]
    occlusions = ["clear", "partial_building", "partial_mountain", "clear"]
    times = ["day", "night", "dawn", "dusk"]
    profile = PROFILES[index % len(PROFILES)]
    return {
        "event_id": event_id,
        "duration_days": index % 15 + 1,
        "geospatial": {
            "landscape_profile": ["rural", "mountain", "agricultural"][index % 3]
        },
        "composition": {
            "diversity": {"capacity_per_category": 16},
            "camera_poses": [
                {
                    "id": f"pose-{pose}",
                    "viewpoint": {
                        "distance_band": distances[(pose + index) % 4],
                        "occlusion": occlusions[(pose + index) % 4],
                        "azimuth_deg": float((index * 30 + pose * 90) % 360),
                    },
                }
                for pose in range(4)
            ],
            "lighting_variants": [
                {"id": f"light-{state}", "time_of_day": times[(state + index) % 4]}
                for state in range(4)
            ],
            "flow_states": [
                {
                    "id": f"state-{state}",
                    "lighting_variant_id": f"light-{state}",
                    "progression": {"phase": profile[state]},
                }
                for state in range(4)
            ],
        },
    }


class EventCatalogTests(unittest.TestCase):
    def _catalog(self, volume: Path) -> tuple[Path, dict[Path, dict[str, object]]]:
        contracts: dict[Path, dict[str, object]] = {}
        events = []
        slot_pattern = [4, 6, 10, 12]
        for index in range(512):
            event_id = f"fire-{index:04d}"
            path = volume / "events" / f"{event_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"event_id": event_id, "nonce": index}), encoding="utf-8")
            contracts[path.resolve()] = _contract(event_id, index)
            events.append(
                {
                    "event_id": event_id,
                    "case_slots_per_category": slot_pattern[index % 4],
                    "real_world_contract": str(path),
                    "real_world_contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        catalog = {
            "schema_version": 1,
            "data_origin": "new_synthetic_generation",
            "minimum_fire_events": 512,
            "maximum_fire_duration_days": 15,
            "max_cases_per_fire_per_category": 24,
            "events": events,
        }
        catalog_path = volume / "event-catalog.json"
        catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
        return catalog_path, contracts

    def test_catalog_schedules_4096_cases_over_512_varied_fires_up_to_15_days(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path, contracts = self._catalog(volume)

            def loader(contract_path: Path, *, volume_root: Path) -> dict[str, object]:
                self.assertEqual(volume_root, volume.resolve())
                return contracts[contract_path.resolve()]

            catalog = load_event_catalog(
                path,
                volume_root=volume,
                target_per_category=4096,
                contract_loader=loader,
            )
            self.assertEqual(catalog["coverage"]["fire_events"], 512)
            self.assertEqual(catalog["coverage"]["fire_duration_days"], list(range(1, 16)))
            self.assertEqual(catalog["coverage"]["case_slots_per_category"], 4096)
            self.assertEqual(catalog["coverage"]["distinct_image_counts_per_fire"], [4, 6, 10, 12])
            self.assertEqual(case_assignment(catalog, 0)["event"]["event_id"], "fire-0000")
            self.assertEqual(case_assignment(catalog, 4095)["case_index"], 4095)

    def test_catalog_refuses_uniform_image_count_per_fire(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            path, contracts = self._catalog(volume)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for event in payload["events"]:
                event["case_slots_per_category"] = 8
            path.write_text(json.dumps(payload), encoding="utf-8")

            def loader(contract_path: Path, *, volume_root: Path) -> dict[str, object]:
                return contracts[contract_path.resolve()]

            with self.assertRaisesRegex(ValueError, "number of images per fire"):
                load_event_catalog(
                    path,
                    volume_root=volume,
                    target_per_category=4096,
                    contract_loader=loader,
                )


if __name__ == "__main__":
    unittest.main()
