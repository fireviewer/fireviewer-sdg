from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "omniverse" / "prepare-z16-scene-library.py"
SPEC = importlib.util.spec_from_file_location("prepare_z16_scene_library", SCRIPT)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrepareZ16SceneLibraryTests(unittest.TestCase):
    def _write_receipts(self, site_root: Path) -> None:
        materialized = site_root / "assets" / "materialized"
        lod_root = site_root / "assets" / "lod"
        materialized.mkdir(parents=True)
        lod_root.mkdir(parents=True)
        (materialized / "materialization-receipt.json").write_text(
            json.dumps({"state": "Z16_RECOVERED_ASSETS_MATERIALIZED"}),
            encoding="utf-8",
        )
        assets = []
        for index, (asset_id, profile) in enumerate(sorted(subject._PROFILES.items())):
            role = (
                "vegetation"
                if profile["placement_class"] == "vegetation"
                else "building"
                if profile["placement_class"] == "building"
                else "actor"
            )
            paths = {}
            for level in ("HERO", "MID", "FAR"):
                path = lod_root / asset_id / f"{level.lower()}.usdc"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{asset_id}:{level}".encode("ascii"))
                paths[level] = {
                    "path": f"{asset_id}/{level.lower()}.usdc",
                    "sha256": _sha256(path),
                }
            assets.append(
                {
                    "asset_id": asset_id,
                    "role": role,
                    "lod_paths": paths,
                    "native_metrics": {
                        "HERO": {
                            "world_bounds": {
                                "dimensions": [100.0 + index, 200.0 + index, 300.0 + index],
                                "minimum": [-50.0, -100.0, 0.0],
                            }
                        }
                    },
                }
            )
        (lod_root / "lod-receipt.json").write_text(
            json.dumps({"state": "Z16_RECOVERED_ASSET_LODS_READY", "assets": assets}),
            encoding="utf-8",
        )

    def test_prepares_complete_source_backed_library(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site_root = Path(temporary)
            self._write_receipts(site_root)
            result = subject.prepare_scene_library(site_root=site_root)
            self.assertEqual(result["state"], subject.STATE)
            manifest = json.loads((site_root / result["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["asset_count"], 9)
            tree = next(item for item in manifest["assets"] if item["family"] == "hero_tree")
            self.assertEqual(tree["uniform_scale"], 0.01)
            self.assertTrue(tree["near_camera_allowed"])
            fill = next(item for item in manifest["assets"] if item["family"] == "forest_canopy_cluster")
            self.assertFalse(fill["near_camera_allowed"])
            self.assertTrue(fill["dense_forest_fill_allowed"])
            for item in manifest["assets"]:
                self.assertEqual(set(item["lod_paths"]), {"HERO", "MID", "FAR"})
                self.assertEqual(item["primitive_substitution"], "forbidden")

    def test_rejects_missing_real_lod_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site_root = Path(temporary)
            self._write_receipts(site_root)
            missing = next((site_root / "assets" / "lod").rglob("far.usdc"))
            missing.unlink()
            with self.assertRaises(subject.SceneLibraryError):
                subject.prepare_scene_library(site_root=site_root)


if __name__ == "__main__":
    unittest.main()
