from __future__ import annotations

import hashlib
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from fireviewer_sdg.native_zone_scene import (
    _ElevationGrid,
    _building_features_for_tile,
    _clip_line_to_bounds,
    _entry_path,
    _exclusive_zone_build,
    _feature_lines,
    _feature_polygons,
    _lock_flow_preset,
    _mosaic_tile_values,
    _oriented_footprint,
    _polygon_centroid,
    _read_raster_values,
    _detail_lod_candidates,
    _author_instance_identity_primvars,
    _stable_instance_id,
    _write_terrain_pbr_maps,
    _validate_height_products,
)


class NativeZoneSceneGeometryTests(unittest.TestCase):
    def test_locked_source_accepts_every_acquisition_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory)
            source = zone_root / "raw" / "mnt" / "tile.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"locked-source")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            for state in (
                "downloaded",
                "downloaded_segmented",
                "verified_existing",
                "recovered_complete_partial",
            ):
                entry = {
                    "id": f"tile:{state}",
                    "dataset": "mnt",
                    "download": {
                        "state": state,
                        "relpath": source.name,
                        "sha256": digest,
                    },
                }
                self.assertEqual(_entry_path(zone_root, entry), source.resolve())

    def test_locked_source_rejects_non_terminal_acquisition_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory)
            source = zone_root / "raw" / "mnt" / "tile.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"pending-source")
            entry = {
                "id": "tile:pending",
                "dataset": "mnt",
                "download": {
                    "state": "pending",
                    "relpath": source.name,
                    "sha256": "0" * 64,
                },
            }
            with self.assertRaisesRegex(RuntimeError, "not been downloaded"):
                _entry_path(zone_root, entry)

    def test_variant_identity_primvars_are_vertex_scoped_and_complete(self) -> None:
        authored: dict[str, dict[str, object]] = {}

        class FakePrimvar:
            def __init__(self, name: str) -> None:
                self.name = name

            def Set(self, value: object) -> None:
                authored[self.name]["value"] = value

        class FakeApi:
            def CreatePrimvar(
                self, name: str, value_type: object, interpolation: object
            ) -> FakePrimvar:
                authored[name] = {
                    "type": value_type,
                    "interpolation": interpolation,
                }
                return FakePrimvar(name)

        class FakePrim:
            def __init__(self) -> None:
                self.custom_data: dict[str, object] = {}

            def SetCustomDataByKey(self, key: str, value: object) -> None:
                self.custom_data[key] = value

        prim = FakePrim()
        usd_geom = SimpleNamespace(
            Tokens=SimpleNamespace(vertex="vertex"),
            PrimvarsAPI=lambda _instancer: FakeApi(),
        )
        sdf = SimpleNamespace(
            ValueTypeNames=SimpleNamespace(
                StringArray="StringArray",
                FloatArray="FloatArray",
            )
        )
        instancer = SimpleNamespace(GetPrim=lambda: prim)
        _author_instance_identity_primvars(
            instancer=instancer,
            stable_ids=["tile-1:trees:10", "tile-1:trees:11"],
            footprint_radii_m=[1.5, 2.0],
            group_ids=["forest:0:0", "forest:0:1"],
            usd_geom=usd_geom,
            sdf=sdf,
        )
        self.assertEqual(set(authored), {
            "fireviewer_stable_id",
            "fireviewer_footprint_radius_m",
            "fireviewer_group_id",
        })
        self.assertTrue(
            all(value["interpolation"] == "vertex" for value in authored.values())
        )
        self.assertEqual(
            prim.custom_data["fireviewer:instance_identity_contract"],
            "ids+stable_id+footprint_radius_m+group_id",
        )

    def test_detail_lods_keep_real_candidates_without_empty_fallbacks(self) -> None:
        candidates = list(range(33))
        self.assertEqual(_detail_lod_candidates(candidates, "HERO"), candidates)
        self.assertEqual(_detail_lod_candidates(candidates, "MID"), candidates[::4])
        self.assertEqual(_detail_lod_candidates(candidates, "FAR"), candidates[::16])
        self.assertEqual(_detail_lod_candidates([9], "FAR"), [9])
        self.assertEqual(_detail_lod_candidates([], "FAR"), [])

    def test_terrain_pbr_maps_are_metric_and_non_flat_on_a_slope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = np.tile(
                np.linspace(0.0, 100.0, 64, dtype=np.float32),
                (64, 1),
            )
            normal, roughness = _write_terrain_pbr_maps(
                values=values,
                output_root=Path(directory),
                tile_ref="T001",
            )
            with Image.open(normal) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertNotEqual(
                    tuple(image.getpixel((32, 32))), (128, 128, 255)
                )
            with Image.open(roughness) as image:
                self.assertEqual(image.mode, "L")
                self.assertGreater(image.getpixel((32, 32)), 0)

    def test_instance_ids_are_globally_namespaced_and_family_scoped(self) -> None:
        tree = _stable_instance_id(
            tile_namespace=17, family="trees", local_index=42
        )
        building = _stable_instance_id(
            tile_namespace=17, family="buildings", local_index=42
        )
        next_tile = _stable_instance_id(
            tile_namespace=18, family="trees", local_index=42
        )
        self.assertEqual(len({tree, building, next_tile}), 3)
        self.assertLess(max(tree, building, next_tile), 2**63)
        self.assertEqual(tree >> 43, 17)
        with self.assertRaisesRegex(ValueError, "between"):
            _stable_instance_id(
                tile_namespace=0, family="trees", local_index=0
            )

    def test_native_builder_rejects_a_concurrent_zone_writer_and_releases_its_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            zone_root.mkdir()
            with _exclusive_zone_build(zone_root):
                self.assertTrue((zone_root / ".native-zone-build.lock").is_file())
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with _exclusive_zone_build(zone_root):
                        pass
            self.assertFalse((zone_root / ".native-zone-build.lock").exists())

    def test_native_builder_reclaims_a_stale_zone_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            zone_root.mkdir()
            lock_path = zone_root / ".native-zone-build.lock"
            lock_path.write_text('{"pid": 12345}\n', encoding="utf-8")
            with patch("fireviewer_sdg.native_zone_scene._build_lock_owner_is_live", return_value=False):
                with _exclusive_zone_build(zone_root):
                    self.assertTrue(lock_path.is_file())
            self.assertFalse(lock_path.exists())

    def test_polygon_normalization_drops_closing_vertex_and_centroid_is_local(self) -> None:
        polygons = list(
            _feature_polygons(
                {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [6, 0], [0, 6], [0, 0]]],
                }
            )
        )
        self.assertEqual(polygons, [[[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]]])
        self.assertEqual(_polygon_centroid(polygons[0]), (2.0, 2.0))

    def test_multiline_normalization_rejects_invalid_empty_lines(self) -> None:
        lines = list(
            _feature_lines(
                {
                    "type": "MultiLineString",
                    "coordinates": [[], [[1, 2], [3, 4]]],
                }
            )
        )
        self.assertEqual(lines, [[[1.0, 2.0], [3.0, 4.0]]])

    def test_elevation_grid_preserves_north_to_south_raster_orientation(self) -> None:
        grid = _ElevationGrid(samples=2)
        values = np.array([[100.0, 110.0], [200.0, 210.0]], dtype=np.float32)
        grid.add(xmin=0, ymin=0, xmax=1000, ymax=1000, values=values)
        self.assertEqual(grid.elevation(0, 1000, fallback=0.0), 100.0)
        self.assertEqual(grid.elevation(1000, 0, fallback=0.0), 210.0)
        self.assertEqual(grid.elevation(500, 500, fallback=0.0), 155.0)

    def test_light_mosaic_extracts_a_tile_with_north_up_orientation(self) -> None:
        values = np.arange(251 * 251, dtype=np.float32).reshape(251, 251)
        tile = {"xmin": "0", "ymin": "1000", "xmax": "1000", "ymax": "2000"}
        zone = {"xmin": 0, "ymin": 0, "xmax": 2000, "ymax": 2000}
        extracted = _mosaic_tile_values(values=values, tile=tile, zone=zone)
        self.assertEqual(extracted.shape, (129, 129))
        self.assertEqual(extracted[0, 0], values[0, 0])
        self.assertEqual(extracted[-1, -1], values[125, 125])

    def test_installed_flow_preset_is_packaged_with_source_and_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (
                root
                / "runtime"
                / "omni.flowusd-106.0.1"
                / "data"
                / "presets"
                / "Fire"
                / "Fire.usda"
            )
            source.parent.mkdir(parents=True)
            source.write_text(
                '#usda 1.0\n(\n    metersPerUnit = 0.01\n    upAxis = "Y"\n)\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"FW_SDG_FLOW_PRESET": str(source)}):
                lock = _lock_flow_preset(build_root=root / "build")
            packaged = root / "build" / str(lock["packaged_path"])
            self.assertTrue(packaged.is_file())
            self.assertEqual(packaged.read_bytes(), source.read_bytes())
            self.assertEqual(len(str(lock["source_sha256"])), 64)
            self.assertEqual(len(str(lock["packaged_sha256"])), 64)
            self.assertEqual(lock["source_version"], "omni.flowusd-106.0.1")

    def test_raster_reader_rejects_material_nodata_instead_of_making_a_plateau(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnh.tif"
            values = np.full((10, 10), 8.0, dtype=np.float32)
            values[0, :2] = -9999.0
            Image.fromarray(values, mode="F").save(path)
            with self.assertRaisesRegex(ValueError, "NoData"):
                _read_raster_values(path, label="MNH")

    def test_height_products_must_agree_with_surface_minus_ground(self) -> None:
        ground = np.full((8, 8), 100.0, dtype=np.float32)
        canopy = np.full((8, 8), 14.0, dtype=np.float32)
        surface = ground + canopy
        quality = _validate_height_products(
            mnt=ground,
            mns=surface,
            mnh=canopy,
            label="tile",
        )
        self.assertEqual(quality["p95_absolute_residual_metres"], 0.0)
        with self.assertRaisesRegex(ValueError, "MNH disagrees"):
            _validate_height_products(
                mnt=ground,
                mns=surface + 5.0,
                mnh=canopy,
                label="tile",
            )

    def test_building_orientation_uses_the_source_footprint(self) -> None:
        length, width, angle = _oriented_footprint(
            [[0.0, 0.0], [12.0, 0.0], [12.0, 4.0], [0.0, 4.0]]
        )
        self.assertAlmostEqual(length, 12.0)
        self.assertAlmostEqual(width, 4.0)
        self.assertAlmostEqual(abs(np.sin(angle)), 0.0)

    def test_buildings_are_assigned_once_and_roads_are_clipped_to_the_tile(self) -> None:
        features = [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[900, 100], [1100, 100], [1100, 300], [900, 300], [900, 100]]
                    ],
                }
            }
        ]
        self.assertEqual(
            len(
                _building_features_for_tile(
                    features, bounds=(0.0, 0.0, 1000.0, 1000.0)
                )
            ),
            0,
        )
        self.assertEqual(
            len(
                _building_features_for_tile(
                    features, bounds=(1000.0, 0.0, 2000.0, 1000.0)
                )
            ),
            1,
        )
        self.assertEqual(
            _clip_line_to_bounds(
                [[-10.0, 500.0], [500.0, 500.0], [1010.0, 500.0]],
                (0.0, 0.0, 1000.0, 1000.0),
            ),
            [[[0.0, 500.0], [500.0, 500.0], [1000.0, 500.0]]],
        )
