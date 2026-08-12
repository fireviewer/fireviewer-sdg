from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "restore-level1-site-input.py"
SPEC = importlib.util.spec_from_file_location("restore_level1_site_input", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"cannot import {MODULE_PATH}")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _full_source_lock() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for y in range(6400, 6420):
        for x in range(878, 898):
            tile_ref = f"L93_{x:04d}_{y:04d}"
            for dataset in sorted(subject.zone_scenes.ALL_DATASETS):
                entries.append(
                    {
                        "id": f"{tile_ref}:{dataset}",
                        "dataset": dataset,
                        "tile_ref": tile_ref,
                        "url": (
                            "https://example.invalid/wms?BBOX="
                            f"{x * 1000},{(y - 1) * 1000},{(x + 1) * 1000},{y * 1000}"
                            if dataset == "ortho50"
                            else f"https://example.invalid/{tile_ref}/{dataset}"
                        ),
                        "download": {
                            "bytes": 1,
                            "sha256": "a" * 64,
                            "relpath": f"{tile_ref}_{dataset}.bin",
                        },
                    }
                )
    return {
        "zone_id": "Z16",
        "acquisition": {"source_profile": subject.FULL_SOURCE_PROFILE},
        "entries": entries,
    }


class CompactLidarRasterProfileTests(unittest.TestCase):
    def test_derives_one_sixteen_tile_site_without_raw_point_clouds(self) -> None:
        compact = subject._compact_lidar_raster_lock(
            _full_source_lock(),
            full_lock_sha256="b" * 64,
            base_id="Z16-base-01",
        )

        self.assertEqual(
            compact["acquisition"]["source_profile"],
            subject.COMPACT_LIDAR_RASTER_SOURCE_PROFILE,
        )
        self.assertEqual(
            compact["acquisition"]["raw_point_cloud_policy"], "forbidden"
        )
        self.assertEqual(
            compact["acquisition"]["terrain_height_resolution_m"], 1.0
        )
        self.assertEqual(len(compact["entries"]), 80)
        self.assertEqual(
            {entry["dataset"] for entry in compact["entries"]},
            subject.COMPACT_SITE_RASTER_DATASETS,
        )
        self.assertEqual(
            {entry["tile_ref"] for entry in compact["entries"]},
            set(compact["acquisition"]["selected_tile_site"]["tile_refs"]),
        )
        self.assertEqual(len(subject._locked_entries(compact)), 80)
        site = compact["acquisition"]["selected_tile_site"]
        self.assertEqual(site["extent_m"], [879000, 6401000, 883000, 6405000])
        self.assertEqual(
            site["tile_bounds_epsg2154"]["L93_0879_6402"],
            [879000, 6401000, 880000, 6402000],
        )

    def test_rejects_an_unknown_compact_site(self) -> None:
        with self.assertRaisesRegex(subject.Level1RestoreError, "unknown compact study site"):
            subject._compact_lidar_raster_lock(
                copy.deepcopy(_full_source_lock()),
                full_lock_sha256="b" * 64,
                base_id="Z16-base-99",
            )
