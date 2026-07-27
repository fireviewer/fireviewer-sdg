from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import ign_catalog  # noqa: E402
from fireviewer_sdg.case_generation import _validate_response_box  # noqa: E402
from fireviewer_sdg.event_catalog import load_event_catalog  # noqa: E402
from fireviewer_sdg.geometry import (  # noqa: E402
    assert_visible,
    camera_contract,
    project_aabb,
    project_point,
)
from fireviewer_sdg.real_world import select_actor_variation  # noqa: E402


def _site_reference_fixture(
    sites: list[dict[str, object]],
    *,
    volume_root: Path,
) -> list[dict[str, object]]:
    reference_root = volume_root / "input" / "test-site-references"
    reference_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for site in sites:
        reference = reference_root / f"{site['id']}.jpg"
        reference.write_bytes(b"reviewed-reference")
        results.append(
            {
                "site_id": str(site["id"]),
                "automatic_validation": "passed",
                "reference_image": reference.relative_to(volume_root).as_posix(),
                "reference_image_sha256": hashlib.sha256(
                    reference.read_bytes()
                ).hexdigest(),
                "raycast": {
                    "validation": "usd_heightfield_line_of_sight_passed",
                    "passed": True,
                },
            }
        )
    return results


class IgnCatalogTests(unittest.TestCase):
    def test_landscape_profiles_use_distinct_realistic_species_layouts(self) -> None:
        mountain = ign_catalog._vegetation_layout(
            site_id="montmaur",
            profile="rural_mountain",
        )
        agricultural = ign_catalog._vegetation_layout(
            site_id="barsac",
            profile="rural_agricultural",
        )
        mixed = ign_catalog._vegetation_layout(
            site_id="ausson",
            profile="mountain_agricultural",
        )
        mountain_species = Counter(item[0] for item in mountain)
        agricultural_species = Counter(item[0] for item in agricultural)

        self.assertEqual(len(mountain), 1_800)
        self.assertEqual(len(agricultural), 1_016)
        self.assertEqual(len(mixed), 1_300)
        self.assertGreater(
            mountain_species[ign_catalog.VEGETATION_SPECIES.index("norway_spruce")]
            + mountain_species[
                ign_catalog.VEGETATION_SPECIES.index("douglas_fir")
            ],
            1_200,
        )
        self.assertLess(
            mountain_species[ign_catalog.VEGETATION_SPECIES.index("common_apple")],
            50,
        )
        self.assertGreaterEqual(
            agricultural_species[
                ign_catalog.VEGETATION_SPECIES.index("common_apple")
            ],
            216,
        )
        self.assertEqual(
            mountain,
            ign_catalog._vegetation_layout(
                site_id="montmaur",
                profile="rural_mountain",
            ),
        )
        self.assertNotEqual(mountain[:100], mixed[:100])

    def test_preparation_auto_discovery_stays_blocked_without_rural_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            prepared = ign_catalog.prepare_ign_catalog(
                volume,
                asset_provisioner=lambda **_kwargs: {
                    "manifest": volume / "input" / ign_catalog.ASSET_MANIFEST_NAME,
                    "report": volume / "input" / "simready-discovery-report.json",
                    "candidate_count": 12,
                    "missing_environment": ["vegetation", "rural_building"],
                    "missing_actor_classes": list(ign_catalog.ACTOR_CLASSES),
                },
            )
            self.assertEqual(prepared["state"], "blocked")
            self.assertEqual(prepared["phase"], "environment_assets")
            self.assertEqual(prepared["synthetic_cases_written"], 0)
            self.assertFalse(
                (volume / "input" / ign_catalog.CATALOG_NAME).exists()
            )

    def test_environment_is_prepared_before_missing_response_assets_block_flow(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            asset_root = volume / "input" / "reviewed-assets"
            asset_root.mkdir(parents=True)

            def asset_entry(name: str) -> dict[str, str]:
                path = asset_root / f"{name}.usd"
                path.write_text(
                    (
                        '#usda 1.0\n(defaultPrim = "Asset")\n'
                        f'def Xform "Asset" {{ string fireviewer:name = "{name}" }}\n'
                    ),
                    encoding="utf-8",
                )
                return {
                    "path": path.relative_to(volume / "input").as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "quality_validation": "simready_asset_human_approved",
                    "placement_validation": (
                        "usd_z_up_meters_grounded_human_approved"
                    ),
                    "provenance": "owned_original",
                    "source_uri": f"https://assets.example.test/{name}",
                    "license_id": "LicenseRef-FireViewer-Test",
                }

            manifest = volume / "input" / ign_catalog.ASSET_MANIFEST_NAME
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile": ign_catalog.ASSET_MANIFEST_PROFILE,
                        "environment": {
                            "vegetation": [
                                asset_entry(f"tree_{index}")
                                for index in range(
                                    ign_catalog.MIN_VEGETATION_VARIANTS
                                )
                            ],
                            "rural_building": asset_entry("rural_barn"),
                        },
                        "actors": {},
                    }
                ),
                encoding="utf-8",
            )

            def site_fixture(
                root: Path,
                site: dict[str, object],
                **_kwargs: object,
            ) -> dict[str, object]:
                site_root = root / "sites" / str(site["id"])
                site_root.mkdir(parents=True, exist_ok=True)
                scene = site_root / "terrain-simready.usda"
                scene.write_text(
                    '#usda 1.0\n(defaultPrim = "Site")\ndef Xform "Site" {}\n',
                    encoding="utf-8",
                )
                return {**site, "scene": scene}

            def validate(paths: list[Path]) -> dict[str, object]:
                return {
                    "count": len(paths),
                    "assets": [str(path.resolve()) for path in paths],
                    "quality": {
                        str(path.resolve()): {
                            "primitive_geometry_count": 0,
                            "mesh_point_count": 20_000,
                            "material_count": 4,
                            "texture_asset_count": 4,
                            "aabb_min_m": [-2.0, -2.0, 0.0],
                            "aabb_max_m": [2.0, 2.0, 8.0],
                        }
                        for path in paths
                    },
                }

            with (
                patch.object(ign_catalog, "_prepare_site", side_effect=site_fixture),
                patch.object(ign_catalog, "_discover_flow_fire_preset") as flow,
            ):
                prepared = ign_catalog.prepare_ign_catalog(
                    volume,
                    asset_manifest=manifest,
                    usd_validator=validate,
                    site_reference_validator=_site_reference_fixture,
                )
            self.assertEqual(prepared["state"], "blocked")
            self.assertEqual(prepared["phase"], "response_assets")
            self.assertEqual(prepared["environment"]["site_count"], 3)
            self.assertEqual(
                set(prepared["missing_actor_classes"]),
                set(ign_catalog.ACTOR_CLASSES),
            )
            self.assertEqual(prepared["synthetic_cases_written"], 0)
            self.assertFalse(
                (volume / "input" / ign_catalog.CATALOG_NAME).exists()
            )
            flow.assert_not_called()

    def test_ign_download_retries_a_transient_http_400_without_weakening_raster_validation(
        self,
    ) -> None:
        class RasterResponse(io.BytesIO):
            def __enter__(self) -> "RasterResponse":
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        failure = urllib.error.HTTPError(
            "https://data.geopf.fr/wms-r/wms",
            400,
            "Bad Request",
            {},
            io.BytesIO(b"temporary WMS rejection"),
        )
        response = RasterResponse(b"valid-geotiff-payload")
        response.headers = Mock()  # type: ignore[attr-defined]
        response.headers.get_content_type.return_value = "image/geotiff"  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "terrain.tif"
            with (
                patch.object(
                    ign_catalog.urllib.request,
                    "urlopen",
                    side_effect=[failure, response],
                ) as urlopen,
                patch.object(ign_catalog.time, "sleep") as sleep,
            ):
                ign_catalog._download(
                    "https://data.geopf.fr/wms-r/wms?REQUEST=GetMap",
                    destination,
                )

            self.assertEqual(destination.read_bytes(), b"valid-geotiff-payload")
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(ign_catalog.IGN_RETRY_DELAYS_S[0])

    def test_generated_usda_layer_metadata_uses_a_valid_dictionary_key(self) -> None:
        source = Path(ign_catalog.__file__).read_text(encoding="utf-8")
        self.assertIn('string fireviewer_source = "IGN Geoplateforme EPSG:2154"', source)
        self.assertNotIn('string fireviewer:source =', source)

    def test_usda_formatter_separates_all_compact_declarations(self) -> None:
        compact = '''#usda 1.0
(
    customLayerData = { string fireviewer_class = "test" string fireviewer_review = "pending" }
)
def Xform "Root" { def DistantLight "Sun" { float intensity = 4200 color3f color = (1, 1, 1) float angle = 0.8 } def Cube "Body" { double3 xformOp:scale = (1, 1, 1) uniform token[] xformOpOrder = ["xformOp:scale"] rel material:binding = </Root/Looks/Body> } }
'''
        formatted = ign_catalog._format_generated_usda(compact)
        self.assertNotIn('{ string ', formatted)
        self.assertNotIn('" string ', formatted)
        self.assertNotIn('{ def ', formatted)
        self.assertNotIn('{ float ', formatted)
        self.assertNotIn(' color3f ', formatted)
        self.assertNotIn(' double3 ', formatted)
        self.assertNotIn(' uniform token[] ', formatted)
        self.assertNotIn('\nuniform\ntoken', formatted)
        self.assertIn('\nuniform token[] xformOpOrder', formatted)
        self.assertNotIn(' rel material:', formatted)
        self.assertNotIn(' } def ', formatted)

    def test_preparation_builds_512_distinct_review_gated_fires(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            preparation = volume / "input" / ign_catalog.PREPARATION_VERSION
            preset = Path(directory) / "runtime" / "official-fire-preset.usda"
            preset.parent.mkdir(parents=True)
            preset.write_text('#usda 1.0\ndef Xform "Fire" {}\n', encoding="utf-8")
            asset_root = volume / "input" / "reviewed-assets"
            asset_root.mkdir(parents=True)

            def asset_entry(name: str) -> dict[str, str]:
                path = asset_root / f"{name}.usd"
                path.write_text(
                    (
                        '#usda 1.0\n(defaultPrim = "Asset")\n'
                        f'def Xform "Asset" {{ string fireviewer:testName = "{name}" }}\n'
                    ),
                    encoding="utf-8",
                )
                return {
                    "path": path.relative_to(volume / "input").as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "quality_validation": "simready_asset_human_approved",
                    "placement_validation": (
                        "usd_z_up_meters_grounded_human_approved"
                    ),
                    "provenance": "owned_original",
                    "source_uri": f"https://assets.example.test/{name}",
                    "license_id": "LicenseRef-FireViewer-Test",
                }

            vegetation = [
                asset_entry(f"tree_variant_{index:02d}")
                for index in range(ign_catalog.MIN_VEGETATION_VARIANTS)
            ]
            building = asset_entry("rural_barn")
            actors = {
                class_id: asset_entry(f"response_{class_id}")
                for class_id in ign_catalog.ACTOR_CLASSES
            }
            asset_manifest = volume / "input" / ign_catalog.ASSET_MANIFEST_NAME
            asset_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile": ign_catalog.ASSET_MANIFEST_PROFILE,
                        "environment": {
                            "vegetation": vegetation,
                            "rural_building": building,
                        },
                        "actors": actors,
                    }
                ),
                encoding="utf-8",
            )

            def site_fixture(
                root: Path,
                site: dict[str, object],
                *,
                fetcher: object,
                vegetation_assets: object,
                building_asset: object,
            ) -> dict[str, object]:
                del fetcher, vegetation_assets, building_asset
                site_root = root / "sites" / str(site["id"])
                site_root.mkdir(parents=True, exist_ok=True)
                ortho = site_root / "orthophoto-ign.tif"
                mnt = site_root / "mnt-ign.tif"
                preview = site_root / "mnt-preview.png"
                texture = site_root / "orthophoto-render.jpg"
                scene = site_root / "terrain-simready.usda"
                ortho.write_bytes(b"official-ortho-fixture")
                mnt.write_bytes(b"official-mnt-fixture")
                preview.write_bytes(b"mnt-preview-fixture")
                texture.write_bytes(b"ortho-texture-fixture")
                scene.write_text('#usda 1.0\ndef Xform "Site" {}\n', encoding="utf-8")
                mnt_image = Image.new("F", (32, 32))
                mnt_image.putdata(
                    [
                        120.0 + (column - 15.5) * 2.0 + (row - 15.5) * 4.0
                        for row in range(32)
                        for column in range(32)
                    ]
                )
                return {
                    **site,
                    "root": site_root,
                    "ortho": ortho,
                    "ortho_render": texture,
                    "mnt": mnt,
                    "mnt_preview": preview,
                    "scene": scene,
                    "base_elevation": 120.0,
                    "mnt_image": mnt_image,
                }

            def validate_assets(paths: list[Path]) -> dict[str, object]:
                actor_bounds = {
                    "sdis_vehicle": ([-3.4, -1.5, -1.0], [3.4, 1.5, 1.8]),
                    "canadair": ([-4.4, -8.8, -2.5], [4.4, 8.8, 2.5]),
                    "dash": ([-6.0, -10.2, -3.5], [6.0, 10.2, 3.5]),
                    "securite_civile_helicopter": (
                        [-5.5, -5.5, -2.1],
                        [5.5, 5.5, 2.1],
                    ),
                    "hard_negative_construction_truck": (
                        [-3.4, -1.5, -1.0],
                        [3.4, 1.5, 1.8],
                    ),
                    "hard_negative_crop_duster": (
                        [-3.8, -6.4, -1.8],
                        [3.8, 6.4, 1.8],
                    ),
                    "hard_negative_utility_helicopter": (
                        [-5.5, -5.5, -2.1],
                        [5.5, 5.5, 2.1],
                    ),
                }
                quality = {}
                for path in paths:
                    minimum, maximum = ([-2.0, -2.0, 0.0], [2.0, 2.0, 8.0])
                    for class_id, bounds in actor_bounds.items():
                        if class_id in path.stem:
                            minimum, maximum = bounds
                            break
                    quality[str(path.resolve())] = {
                        "primitive_geometry_count": 0,
                        "mesh_point_count": 20_000,
                        "material_count": 4,
                        "texture_asset_count": 4,
                        "aabb_min_m": minimum,
                        "aabb_max_m": maximum,
                    }
                return {
                    "count": len(paths),
                    "assets": [str(path.resolve()) for path in paths],
                    "quality": quality,
                }

            with (
                patch.object(ign_catalog, "_prepare_site", side_effect=site_fixture),
                patch.object(
                    ign_catalog,
                    "_discover_flow_fire_preset",
                    return_value=preset,
                ),
            ):
                prepared = ign_catalog.prepare_ign_catalog(
                    volume,
                    asset_manifest=asset_manifest,
                    usd_validator=validate_assets,
                    site_reference_validator=_site_reference_fixture,
                )

            self.assertEqual(prepared["fire_events"], 512)
            self.assertEqual(prepared["site_count"], 3)
            self.assertEqual(prepared["usd_validation"]["count"], 19)
            catalog = load_event_catalog(
                volume / "input" / ign_catalog.CATALOG_NAME,
                volume_root=volume,
                target_per_category=4096,
            )
            self.assertEqual(catalog["coverage"]["fire_events"], 512)
            self.assertEqual(catalog["coverage"]["case_slots_per_category"], 4096)
            self.assertGreaterEqual(catalog["coverage"]["progression_profiles"], 32)
            fire_positions = {
                tuple(
                    round(value, 3)
                    for value in event["contract"]["composition"]["flow_states"][0][
                        "anchors_world_m"
                    ]["active_fire_point"][:2]
                )
                for event in catalog["events"]
            }
            self.assertEqual(len(fire_positions), 512)
            self.assertEqual(
                catalog["coverage"]["distinct_image_counts_per_fire"],
                [4, 6, 10, 12],
            )
            progression_signatures = {
                tuple(
                    (
                        state["time_seconds"],
                        state["progression"]["spread_heading_deg"],
                        state["progression"]["wind_speed_mps"],
                    )
                    for state in event["contract"]["composition"]["flow_states"]
                )
                for event in catalog["events"]
            }
            self.assertEqual(len(progression_signatures), 512)
            for event in catalog["events"]:
                contract = event["contract"]
                anchors = [
                    point
                    for state in contract["composition"]["flow_states"]
                    for point in state["anchors_world_m"].values()
                ]
                for pose in contract["composition"]["camera_poses"]:
                    camera = camera_contract(
                        position=pose["position"],
                        look_at=pose["look_at"],
                        width=ign_catalog.RENDER_WIDTH,
                        height=ign_catalog.RENDER_HEIGHT,
                    )
                    assert_visible(
                        [project_point(point, camera) for point in anchors],
                        margin=0.03,
                    )
                    viewpoint = pose["viewpoint"]
                    self.assertLessEqual(
                        viewpoint["distance_m"],
                        ign_catalog.MAX_FRAMING_DISTANCE_M[
                            viewpoint["distance_band"]
                        ],
                    )
            actor_coverage = {
                class_id: {"distance": set(), "lighting": set()}
                for class_id in ign_catalog.ACTOR_CLASSES
            }
            pilot_distance_bands = set()
            for assignment in catalog["assignments"]:
                case_index = assignment["case_index"]
                contract = assignment["event"]["contract"]
                class_id = ign_catalog.ACTOR_CLASSES[
                    case_index % len(ign_catalog.ACTOR_CLASSES)
                ]
                actor = next(
                    item
                    for item in contract["composition"]["actors"]
                    if item["class_id"] == class_id
                )
                capacity = len(contract["composition"]["camera_poses"]) * len(
                    contract["composition"]["flow_states"]
                )
                variation = select_actor_variation(
                    contract, class_id, case_index % capacity
                )
                pose = variation["camera_pose"]
                camera = camera_contract(
                    position=pose["position"],
                    look_at=actor["center_world_m"],
                    width=ign_catalog.RENDER_WIDTH,
                    height=ign_catalog.RENDER_HEIGHT,
                )
                box = project_aabb(
                    actor["aabb_min_world_m"],
                    actor["aabb_max_world_m"],
                    camera,
                )
                distance_band = pose["viewpoint"]["distance_band"]
                _validate_response_box(
                    box,
                    width=ign_catalog.RENDER_WIDTH,
                    height=ign_catalog.RENDER_HEIGHT,
                    distance_band=distance_band,
                )
                actor_coverage[class_id]["distance"].add(distance_band)
                actor_coverage[class_id]["lighting"].add(
                    variation["lighting"]["time_of_day"]
                )
                if case_index < 8:
                    pilot_distance_bands.add(distance_band)
            self.assertEqual(pilot_distance_bands, {"near", "medium", "far", "very_far"})
            for coverage in actor_coverage.values():
                self.assertEqual(
                    coverage["distance"], {"near", "medium", "far", "very_far"}
                )
                self.assertEqual(
                    coverage["lighting"], {"day", "night", "dawn", "dusk"}
                )
            first = catalog["events"][0]["contract"]
            self.assertEqual(
                first["reconstruction"]["metrics"],
                {"quality_review": "pending_console_review"},
            )
            self.assertEqual(
                first["composition"]["flow_validation"][
                    "preset_rendered_and_anchor_verified"
                ],
                "pending_console_review",
            )
            self.assertTrue(
                all(
                    actor["quality_validation"] == "simready_asset_human_approved"
                    for actor in first["composition"]["actors"]
                )
            )


if __name__ == "__main__":
    unittest.main()
