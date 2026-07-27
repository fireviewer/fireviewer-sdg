from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import simready_assets  # noqa: E402


def _asset(name: str, *, size: int = 50_000_000) -> dict[str, object]:
    return {
        "uri": (
            f"{simready_assets.DEFAULT_NVIDIA_ASSET_ROOT}/"
            f"Isaac/SimReady/{name}.usd"
        ),
        "provider_hash": f"provider-{name}",
        "provider_version": "6.0",
        "size_bytes": size,
    }


def _complete_inventory() -> list[dict[str, object]]:
    return [
        *[_asset(f"Outdoor/Pine_Tree_{index:02d}") for index in range(6)],
        _asset("Outdoor/Rural_Barn"),
        _asset("Response/SDIS_CCF"),
        _asset("Response/Canadair_CL415"),
        _asset("Response/Dash_8_Air_Tanker"),
        _asset("Response/Securite_Civile_H145_Dragon"),
        _asset("Negatives/Construction_Dump_Truck"),
        _asset("Negatives/Crop_Duster"),
        _asset("Negatives/Utility_Helicopter"),
    ]


class SimReadyAssetTests(unittest.TestCase):
    def test_indexed_discovery_is_bounded_and_excludes_internal_variants(self) -> None:
        manifest_csv = (
            "_Path,_Link,_Supplier\n"
            "SimReady/Outdoor/Oak_Tree_01/oak_tree_01.usd,link,NVIDIA\n"
            "SimReady/Outdoor/Oak_Tree_01/oak_tree_01_base.usd,link,NVIDIA\n"
        ).encode()
        rivermark_index = "\n".join(
            (
                "/nv_content/common_assets/props_vegetation/"
                "bush_gen_06/bush_gen_06.usd",
                "/nv_content/common_assets/props_vegetation/"
                "bush_gen_06/bush_gen_06_inst.usd",
                "/nv_core/props_vegetation/tree_unmatched.usd",
            )
        ).encode()
        contents = {
            "manifest.csv": manifest_csv,
            "file_list.txt": rivermark_index,
        }
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            simready_assets.cache_official_nvidia_indexes(
                volume_root=volume,
                reader=lambda uri: contents[Path(uri).name],
            )
            inventory = (
                simready_assets.discover_official_nvidia_assets_from_indexes(
                    volume_root=volume,
                    vegetation_inventory={"objects": []},
                )
            )
        uris = [str(entry["uri"]) for entry in inventory]
        self.assertEqual(len(uris), 2)
        self.assertTrue(any(uri.endswith("oak_tree_01.usd") for uri in uris))
        self.assertTrue(any(uri.endswith("bush_gen_06.usd") for uri in uris))
        self.assertFalse(any("_base.usd" in uri for uri in uris))
        self.assertFalse(any("_inst.usd" in uri for uri in uris))
        self.assertFalse(any("unmatched" in uri for uri in uris))

    def test_selector_requires_exact_response_semantics(self) -> None:
        selection = simready_assets.select_simready_assets(
            [
                *[_asset(f"Outdoor/Oak_Tree_{index:02d}") for index in range(6)],
                _asset("Outdoor/Farm_Building"),
                _asset("Vehicles/Generic_Fire_Truck"),
                _asset("Aircraft/Generic_Helicopter"),
            ]
        )
        self.assertNotIn("sdis_vehicle", selection["actors"])
        self.assertNotIn("securite_civile_helicopter", selection["actors"])
        self.assertIn("sdis_vehicle", selection["missing_actor_classes"])
        self.assertIn(
            "securite_civile_helicopter",
            selection["missing_actor_classes"],
        )

    def test_selector_finds_unique_environment_and_exact_actor_roles(self) -> None:
        selection = simready_assets.select_simready_assets(_complete_inventory())
        self.assertEqual(len(selection["vegetation"]), 6)
        self.assertIsNotNone(selection["rural_building"])
        self.assertEqual(selection["missing_environment"], [])
        self.assertEqual(selection["missing_actor_classes"], [])
        selected = [
            *(asset["uri"] for asset in selection["vegetation"]),
            selection["rural_building"]["uri"],
            *(asset["uri"] for asset in selection["actors"].values()),
        ]
        self.assertEqual(len(selected), len(set(selected)))

    def test_provisioner_writes_hashable_remote_reference_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest = volume / "input" / "simready-assets-hd-v2.json"
            with patch.object(
                simready_assets,
                "discover_official_nvidia_assets",
                return_value=_complete_inventory(),
            ):
                result = simready_assets.provision_official_nvidia_manifest(
                    volume_root=volume,
                    manifest_path=manifest,
                )
            self.assertEqual(result["missing_environment"], [])
            self.assertEqual(result["missing_actor_classes"], [])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["profile"], simready_assets.MANIFEST_PROFILE)
            self.assertEqual(
                payload["discovery"]["semantic_policy"],
                "exact_response_identity_only_no_generic_vehicle_promotion",
            )
            entries = [
                *payload["environment"]["vegetation"],
                payload["environment"]["rural_building"],
                *payload["actors"].values(),
            ]
            for entry in entries:
                wrapper = manifest.parent / entry["path"]
                self.assertTrue(wrapper.is_file())
                source = wrapper.read_text(encoding="utf-8")
                self.assertIn("prepend references = @https://", source)
                self.assertEqual(entry["quality_validation"], "pending_console_review")
                self.assertEqual(entry["provenance"], "nvidia_simready")

    def test_automatic_discovery_rejects_unreviewed_hosts(self) -> None:
        with self.assertRaisesRegex(ValueError, "official NVIDIA"):
            simready_assets.discover_official_nvidia_assets(
                "https://unreviewed.example.test/Assets/Isaac/6.0",
                lister=lambda _uri: [],
            )


if __name__ == "__main__":
    unittest.main()
