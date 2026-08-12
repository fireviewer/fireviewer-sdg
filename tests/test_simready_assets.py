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
        *[
            _asset(f"Outdoor/Vegetation/Trees/Pine_Tree_{index:02d}")
            for index in range(8)
        ],
        *[
            _asset(f"Outdoor/Vegetation/Shrubs/Juniper_Shrub_{index:02d}")
            for index in range(4)
        ],
        *[
            _asset(f"Outdoor/Vegetation/Understory/Fern_{index:02d}")
            for index in range(4)
        ],
        *[
            _asset(f"Outdoor/Buildings/Habitat/House_{index:02d}")
            for index in range(4)
        ],
        *[
            _asset(f"Outdoor/Buildings/Agricultural/Barn_{index:02d}")
            for index in range(2)
        ],
        *[
            _asset(f"Outdoor/Buildings/Industrial/Warehouse_{index:02d}")
            for index in range(2)
        ],
        *[
            _asset(f"Outdoor/Buildings/Annex/Garage_{index:02d}")
            for index in range(2)
        ],
        _asset("Response/SDIS_CCF"),
        _asset("Response/Canadair_CL415"),
        _asset("Response/Dash_8_Air_Tanker"),
        _asset("Response/Securite_Civile_H145_Dragon"),
        _asset("Negatives/Construction_Dump_Truck"),
        _asset("Negatives/Crop_Duster"),
        _asset("Negatives/Utility_Helicopter"),
    ]


class SimReadyAssetTests(unittest.TestCase):
    def test_editable_payload_preserves_valid_indexed_main_layer(self) -> None:
        root = Path("C:/provider/valid")
        main = (root / "building.usd").resolve()
        backing = (root / "building_base.usd").resolve()

        selected = simready_assets._select_editable_nvidia_payload(
            main_path=main,
            inspections={
                main: {"state": "passed"},
                backing: {"state": "passed"},
            },
            usd_dependencies={
                main: {backing},
                backing: set(),
            },
        )

        self.assertEqual(selected, main)

    def test_editable_payload_selects_composition_root_not_backing_layer(
        self,
    ) -> None:
        root = Path("C:/provider/kasa_house_01")
        main = (root / "kasa_house_01.usd").resolve()
        inst = (root / "kasa_house_01_inst.usd").resolve()
        inst_base = (root / "kasa_house_01_inst_base.usd").resolve()
        inspections = {
            main: {
                "state": "rejected",
                "reason": "no_editable_material_bound_geometry",
            },
            inst: {
                "state": "passed",
                "editable_mesh_count": 542,
                "material_bound_mesh_count": 512,
                "face_count": 10_000,
            },
            inst_base: {
                "state": "passed",
                "editable_mesh_count": 542,
                "material_bound_mesh_count": 512,
                "face_count": 10_000,
            },
        }

        selected = simready_assets._select_editable_nvidia_payload(
            main_path=main,
            inspections=inspections,
            usd_dependencies={
                main: set(),
                inst: {inst_base},
                inst_base: set(),
            },
        )

        self.assertEqual(selected, inst)

    def test_editable_payload_refuses_ambiguous_composition_roots(self) -> None:
        root = Path("C:/provider/ambiguous")
        main = (root / "tagging.usd").resolve()
        first = (root / "first.usd").resolve()
        second = (root / "second.usd").resolve()
        inspections = {
            main: {"state": "rejected"},
            first: {"state": "passed"},
            second: {"state": "passed"},
        }

        with self.assertRaisesRegex(RuntimeError, "no unique editable"):
            simready_assets._select_editable_nvidia_payload(
                main_path=main,
                inspections=inspections,
                usd_dependencies={
                    main: set(),
                    first: set(),
                    second: set(),
                },
            )

    def test_tagging_paths_are_not_environment_geometry(self) -> None:
        self.assertTrue(
            simready_assets._is_technical_render_prim_path(
                "/Tagging/ThumbRig/primary/SimReady"
            )
        )
        self.assertFalse(
            simready_assets._is_technical_render_prim_path(
                "/RootNode/KasaHouse/Exterior"
            )
        )

    def test_only_known_kit_mdl_module_is_runtime_resolved(self) -> None:
        self.assertTrue(
            simready_assets._is_builtin_omniverse_mdl_module("OmniPBR.mdl")
        )
        self.assertFalse(
            simready_assets._is_builtin_omniverse_mdl_module("SimPBR.mdl")
        )
        self.assertFalse(
            simready_assets._is_builtin_omniverse_mdl_module(
                "../materials/OmniPBR.mdl"
            )
        )

    def test_selector_excludes_official_tree_with_ignored_material_bindings(self) -> None:
        inventory = _complete_inventory()
        inventory.insert(
            0,
            {
                "uri": (
                    f"{simready_assets.NVIDIA_BUCKET_ORIGIN}/"
                    "Assets/Vegetation/Trees/Norway_Spruce.usd"
                ),
                "provider_hash": "broken-binding-tree",
                "size_bytes": 100_000_000,
                "source_index": "Assets/Vegetation S3 inventory",
            },
        )

        selection = simready_assets.select_simready_assets(inventory)

        selected = selection["environment"]["vegetation"]["trees"]
        self.assertEqual(len(selected), 8)
        self.assertFalse(
            any(str(asset["uri"]).endswith("/Norway_Spruce.usd") for asset in selected)
        )

    def test_official_vegetation_uses_real_groundcover_before_stale_rivermark(self) -> None:
        official_root = simready_assets.NVIDIA_BUCKET_ORIGIN
        inventory = [
            *[
                {
                    "uri": f"{official_root}/Assets/Vegetation/Trees/Tree_{index}.usd",
                    "provider_hash": f"tree-{index}",
                    "size_bytes": 10_000_000,
                    "source_index": "Assets/Vegetation S3 inventory",
                }
                for index in range(8)
            ],
            *[
                {
                    "uri": (
                        f"{official_root}/Assets/Vegetation/Shrub/"
                        f"{name}.usd"
                    ),
                    "provider_hash": f"shrub-{index}",
                    "size_bytes": 10_000_000,
                    "source_index": "Assets/Vegetation S3 inventory",
                }
                for index, name in enumerate(
                    ("Juniper", "Barberry", "Boxwood", "Holly")
                )
            ],
            *[
                {
                    "uri": (
                        f"{official_root}/Assets/Vegetation/"
                        f"{folder}/{name}.usd"
                    ),
                    "provider_hash": f"groundcover-{index}",
                    "size_bytes": 10_000_000,
                    "source_index": "Assets/Vegetation S3 inventory",
                }
                for index, (folder, name) in enumerate(
                    (
                        ("Plant_Tropical", "Australian_Tree_Fern"),
                        ("Plant_Tropical", "Crane_Lily"),
                        ("Plant_Tropical", "Japanese_Painted_Fern"),
                        ("Shrub", "Fountain_Grass_Short"),
                    )
                )
            ],
            *[
                {
                    "uri": (
                        f"{official_root}/Assets/Vegetation/"
                        f"Plant_Tropical/{name}.usd"
                    ),
                    "provider_hash": f"tropical-distractor-{index}",
                    "size_bytes": 20_000_000,
                    "source_index": "Assets/Vegetation S3 inventory",
                }
                for index, name in enumerate(
                    ("Honey_Myrtle", "Japanese_Flame", "Windmill_Palm")
                )
            ],
            {
                "uri": (
                    f"{simready_assets.DEFAULT_NVIDIA_ASSET_ROOT}/"
                    "Isaac/Environments/Outdoor/Rivermark/dsready_content/"
                    "nv_content/common_assets/props_vegetation/"
                    "bush_gen_06/bush_gen_06.usd"
                ),
                "provider_hash": "",
                "size_bytes": 0,
                "source_index": (
                    "Isaac/Environments/Outdoor/Rivermark/"
                    "dsready_content/file_list.txt"
                ),
            },
        ]

        selection = simready_assets.select_simready_assets(inventory)

        self.assertEqual(
            len(selection["environment"]["vegetation"]["understory"]),
            4,
        )
        selected_vegetation = [
            asset
            for family in selection["environment"]["vegetation"].values()
            for asset in family
        ]
        self.assertTrue(
            all(
                asset["source_index"] == "Assets/Vegetation S3 inventory"
                for asset in selected_vegetation
            )
        )
        selected_uris = {
            str(asset["uri"]) for asset in selected_vegetation
        }
        self.assertFalse(
            any(
                name in uri
                for name in ("Honey_Myrtle", "Japanese_Flame", "Windmill_Palm")
                for uri in selected_uris
            )
        )

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
                *[_asset(f"Outdoor/Oak_Tree_{index:02d}") for index in range(8)],
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
        for kind, family_minimums in (
            simready_assets.PHOTOREAL_FAMILY_MINIMUMS.items()
        ):
            for family, minimum in family_minimums.items():
                self.assertEqual(
                    len(selection["environment"][kind][family]),
                    minimum,
                )
        self.assertEqual(selection["missing_environment"], [])
        self.assertEqual(selection["missing_actor_classes"], [])
        selected = [
            *(
                asset["uri"]
                for families in selection["environment"].values()
                for assets in families.values()
                for asset in assets
            ),
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
            self.assertEqual(
                payload["library_policy"],
                simready_assets.PHOTOREAL_LIBRARY_POLICY,
            )
            entries = [
                *(
                    entry
                    for families in payload["environment"].values()
                    for assets in families.values()
                    for entry in assets
                ),
                *payload["actors"].values(),
            ]
            for entry in entries:
                wrapper = manifest.parent / entry["path"]
                self.assertTrue(wrapper.is_file())
                source = wrapper.read_text(encoding="utf-8")
                self.assertIn("prepend references = @https://", source)
                self.assertEqual(
                    entry["quality_validation"],
                    "pending_native_validation",
                )
                self.assertEqual(
                    entry["provenance"]["provider"],
                    "NVIDIA Omniverse",
                )
                self.assertEqual(
                    entry["placement"]["scale_policy"],
                    "uniform_only",
                )

    def test_selector_reports_every_missing_photoreal_family(self) -> None:
        selection = simready_assets.select_simready_assets(
            [_asset(f"Outdoor/Vegetation/Trees/Pine_{index:02d}") for index in range(8)]
        )
        self.assertEqual(
            set(selection["missing_environment"]),
            {
                "vegetation.shrubs",
                "vegetation.understory",
                "buildings.habitat",
                "buildings.agricultural",
                "buildings.industrial",
                "buildings.annex",
            },
        )

    def test_warehouse_components_are_not_promoted_to_buildings(self) -> None:
        selection = simready_assets.select_simready_assets(
            [
                _asset(
                    "Industrial/Warehouse/Barriers/Corner_Rail/"
                    "sm_rail_corner_a01_01"
                ),
                _asset(
                    "Industrial/Warehouse/Equipment/Pipe/"
                    "sm_pipe_plastic_gray_b08_01"
                ),
            ]
        )
        self.assertEqual(selection["environment"]["buildings"]["industrial"], [])
        self.assertIn(
            "buildings.industrial",
            selection["missing_environment"],
        )

    def test_automatic_discovery_rejects_unreviewed_hosts(self) -> None:
        with self.assertRaisesRegex(ValueError, "official NVIDIA"):
            simready_assets.discover_official_nvidia_assets(
                "https://unreviewed.example.test/Assets/Isaac/6.0",
                lister=lambda _uri: [],
            )


if __name__ == "__main__":
    unittest.main()
