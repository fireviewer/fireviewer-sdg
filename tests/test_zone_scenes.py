from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from PIL import Image, TiffImagePlugin

from fireviewer_sdg.artifacts import write_json
from fireviewer_sdg.zone_scenes import (
    BASELINE_DATASETS,
    DEFAULT_DIRECT_DOWNLOAD_WORKERS,
    DEFAULT_RASTER_DOWNLOAD_WORKERS,
    MAX_DIRECT_DOWNLOAD_WORKERS,
    MAX_RASTER_DOWNLOAD_WORKERS,
    ZONE_ORDER,
    ZoneSceneProduction,
    _acquisition_lock,
    _assert_capacity,
    _assert_exact_range_coverage,
    _download,
    _download_direct_entries,
    _download_raster_entries,
    _download_with_retries,
    _download_vector_layer,
    _assert_turn,
    _load_state,
    _light_source_lock_from_full,
    _measure,
    _measure_direct_entries,
    _measure_direct_entries_with_retries,
    _measurement_samples,
    _partition_byte_ranges,
    _RequestStartPacer,
    _review_camera_lod0_tiles,
    _selected_entries,
    _segment_paths,
    _source_measurement_fingerprint,
    _tail_segmentation_plan,
    _validate_build_receipt,
    _vector_request_url,
    validate_catalog,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geotiff_bytes(
    *,
    width: int = 2,
    height: int = 2,
    bbox: tuple[float, float, float, float] = (100.0, 200.0, 102.0, 202.0),
    epsg: str = "EPSG:2154|",
    tiepoint_xy: tuple[float, float] | None = None,
) -> bytes:
    xmin, _ymin, xmax, ymax = bbox
    pixel_x = (xmax - xmin) / width
    pixel_y = (bbox[3] - bbox[1]) / height
    tiepoint_x, tiepoint_y = tiepoint_xy or (xmin, ymax)
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[33550] = (pixel_x, pixel_y, 0.0)
    tags[33922] = (
        0.0,
        0.0,
        0.0,
        tiepoint_x,
        tiepoint_y,
        0.0,
    )
    tags[34737] = epsg
    tags[42113] = "-9999"
    output = io.BytesIO()
    Image.new("F", (width, height), 1.0).save(
        output, format="TIFF", tiffinfo=tags
    )
    return output.getvalue()


def _write_material_usda(path: Path, prim_name: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'#usda 1.0\n\ndef Xform "{prim_name}" {{}}\n',
        encoding="utf-8",
    )
    return {"path": path.as_posix(), "sha256": _digest(path)}


def _write_full_build_receipt(
    zone_root: Path,
    *,
    terrain_count: int = 400,
    detail_count: int = 400,
    header_only_terrain: int | None = None,
) -> Path:
    build_root = zone_root / "build"
    root_record = _write_material_usda(build_root / "Z16_root.usda", "World")
    root_record["path"] = (build_root / "Z16_root.usda").relative_to(zone_root).as_posix()
    payloads: list[dict[str, str]] = []
    for index in range(terrain_count):
        path = build_root / "payloads" / f"terrain-{index:03d}.usda"
        if index == header_only_terrain:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#usda 1.0\n", encoding="utf-8")
            record = {"path": path.as_posix(), "sha256": _digest(path)}
        else:
            record = _write_material_usda(path, f"Terrain{index:03d}")
        record["path"] = path.relative_to(zone_root).as_posix()
        payloads.append(record)
    details_by_level: dict[str, list[dict[str, str]]] = {}
    for level in ("HERO", "MID", "FAR"):
        details: list[dict[str, str]] = []
        count = detail_count if level == "HERO" else 400
        for index in range(count):
            path = (
                build_root
                / "details"
                / level.lower()
                / f"detail-{index:03d}.usda"
            )
            record = _write_material_usda(
                path, f"{level.title()}Detail{index:03d}"
            )
            record["path"] = path.relative_to(zone_root).as_posix()
            details.append(record)
        details_by_level[level] = details
    lidar = build_root / "lidar-evidence.json"
    lidar.write_text('{"schema_version":1}', encoding="utf-8")
    coverage = [
        {
            "tile_ref": f"L93_{index:03d}",
            "terrain_payload": payloads[index]["path"] if index < len(payloads) else "",
            "detail_payload": (
                details_by_level["HERO"][index]["path"]
                if index < len(details_by_level["HERO"])
                else ""
            ),
            "detail_lods": {
                level: (
                    details_by_level[level][index]["path"]
                    if index < len(details_by_level[level])
                    else ""
                )
                for level in ("HERO", "MID", "FAR")
            },
            "terrain_lods": ["LOD1", "LOD2", "LOD3"],
            "collision_lods": ["NEAR", "FAR"],
            "detail_counts": {
                "buildings": 1,
                "roads": 1,
                "hydrology": 1,
                "vegetation": 1,
            },
            "detail_lod_counts": {
                level: {
                    "buildings": 1,
                    "roads": 1,
                    "hydrology": 1,
                    "vegetation": 1,
                }
                for level in ("HERO", "MID", "FAR")
            },
            "instance_namespace": index + 1,
        }
        for index in range(400)
    ]
    receipt = {
        "schema_version": 2,
        "zone_id": "Z16",
        "coordinate_convention": "usd_z_up_meters_lambert93",
        "source_profile": "full",
        "root_usd": root_record,
        "payloads": payloads,
        "detail_payloads": details_by_level["HERO"],
        "detail_mid_payloads": details_by_level["MID"],
        "detail_far_payloads": details_by_level["FAR"],
        "lidar_quality": {
            "path": lidar.relative_to(zone_root).as_posix(),
            "sha256": _digest(lidar),
            "source_count": 400,
        },
        "tile_coverage": coverage,
        "layers": {
            name: {"prim_count": 1}
            for name in (
                "terrain",
                "imagery",
                "hydrology",
                "roads",
                "buildings",
                "vegetation",
                "collisions",
                "semantics",
                "detail_streaming",
            )
        },
    }
    receipt["layers"]["detail_streaming"]["levels"] = [
        "HERO",
        "MID",
        "FAR",
    ]
    receipt["layers"]["collisions"].update(
        {
            "levels": ["NEAR", "FAR"],
            "near_spacing_m": 4.0,
            "far_spacing_m": 32.0,
        }
    )
    receipt_path = build_root / "scene-build.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path


def _write_catalog(root: Path) -> Path:
    root.mkdir()
    manifests = root / "manifests"
    manifests.mkdir()
    fields = [
        "zone_id", "tile_ref", "xmin", "ymin", "xmax", "ymax",
        "lidar_expected", "mnt_expected", "mns_expected", "mnh_expected",
        "ortho_ref", "ortho_wms_20cm", "ortho_wms_50cm",
    ]
    rows: list[dict[str, str]] = []
    zones: list[dict[str, str]] = []
    checksum_paths: list[Path] = []
    for zone_index, zone_id in enumerate(ZONE_ORDER):
        tiles: list[dict[str, object]] = []
        base_x = 100_000 + zone_index * 30_000
        base_y = 6_000_000
        for row_index in range(20):
            for column_index in range(20):
                xmin = base_x + column_index * 1000
                ymin = base_y + row_index * 1000
                ref = f"L93_{xmin // 1000:04d}_{(ymin // 1000) + 1:04d}"
                item = {
                    "tile_ref": ref,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmin + 1000,
                    "ymax": ymin + 1000,
                    "lidar_expected": f"{ref}.laz",
                    "mnt_expected": f"{ref}_mnt.tif",
                    "mns_expected": f"{ref}_mns.tif",
                    "mnh_expected": f"{ref}_mnh.tif",
                    "ortho_ref": f"{ref}_ortho",
                }
                tiles.append(item)
                rows.append(
                    {
                        "zone_id": zone_id,
                        **{key: str(value) for key, value in item.items()},
                        "ortho_wms_20cm": f"https://example.invalid/20/{ref}",
                        "ortho_wms_50cm": f"https://example.invalid/50/{ref}",
                    }
                )
        zones.append({"id": zone_id})
        manifest = manifests / f"{zone_id}.json"
        manifest.write_text(json.dumps({"zone": {"id": zone_id}, "tiles": tiles}), encoding="utf-8")
        checksum_paths.append(manifest)

    with (root / "zones_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id"])
        writer.writeheader()
        writer.writerows(zones)
    checksum_paths.append(root / "zones_summary.csv")
    with (root / "tiles_1km_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(root / "tiles_1km_inventory.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    checksum_paths.append(root / "tiles_1km_inventory.csv.gz")
    package = {
        "crs": "EPSG:2154",
        "zone_count": 20,
        "tile_count": 8000,
    }
    (root / "package_manifest.json").write_text(json.dumps(package), encoding="utf-8")
    checksum_paths.append(root / "package_manifest.json")
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{_digest(path)}  {path.relative_to(root).as_posix()}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return root


class ZoneScenesTests(unittest.TestCase):
    def test_catalog_receipt_validates_all_twenty_zones_and_uncompressed_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = _write_catalog(Path(directory) / "catalog")
            receipt = validate_catalog(catalog)
        self.assertEqual(receipt["inventory"]["rows"], 8000)
        self.assertEqual(len(receipt["manifests"]), 20)
        self.assertEqual(set(receipt["zones"]), set(ZONE_ORDER))

    def test_successor_cannot_resolve_until_predecessor_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            pilot = ZoneSceneProduction(catalog_root=catalog, workspace_root=workspace, zone_id="Z16")
            pilot.preflight()
            successor = ZoneSceneProduction(catalog_root=catalog, workspace_root=workspace, zone_id="Z10")
            successor.preflight()
            with self.assertRaisesRegex(RuntimeError, "Z16"):
                successor.resolve(timeout=1, retries=1)

    def test_four_variant_bases_are_independent_and_other_zones_are_blocked(
        self,
    ) -> None:
        state = {
            "zones": {},
            "zone_order": list(ZONE_ORDER),
        }
        configured = ",".join(ZONE_ORDER[:4])
        with patch.dict(
            os.environ,
            {"FW_SDG_VARIANT_BASE_ZONES": configured},
            clear=False,
        ):
            for zone_id in ZONE_ORDER[:4]:
                _assert_turn(state, zone_id)
            with self.assertRaisesRegex(
                RuntimeError,
                "outside the configured four-scene",
            ):
                _assert_turn(state, ZONE_ORDER[4])

    def test_preflight_revalidates_catalog_without_regressing_resumable_zone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            production = ZoneSceneProduction(
                catalog_root=catalog,
                workspace_root=workspace,
                zone_id="Z16",
            )
            production.preflight()
            state = _load_state(workspace)
            state["zones"]["Z16"]["phase"] = "sources_resolved"
            write_json(workspace / "zone-scenes" / "production-state.json", state)

            production.preflight()

            resumed = _load_state(workspace)["zones"]["Z16"]
            self.assertEqual(resumed["phase"], "sources_resolved")
            self.assertEqual(resumed["history"][-1]["phase"], "catalog_revalidated")
            self.assertEqual(
                resumed["history"][-1]["preserved_phase"], "sources_resolved"
            )

    def test_acquisition_profile_excludes_lidar_until_a_lod0_tile_is_selected(self) -> None:
        entries = [
            {"id": f"L93_0001_0001:{dataset}", "dataset": dataset, "tile_ref": "L93_0001_0001", "url": "https://example.invalid/source"}
            for dataset in ("lidar", "mnt", "mns", "mnh", "ortho20", "ortho50")
        ]
        lock = {"entries": entries}
        baseline = _selected_entries(lock, lod0_tiles=set())
        self.assertEqual({item["dataset"] for item in baseline}, set(BASELINE_DATASETS))
        close = _selected_entries(lock, lod0_tiles={"L93_0001_0001"})
        self.assertEqual({item["dataset"] for item in close}, {"lidar", "mnt", "mns", "mnh", "ortho20", "ortho50"})

    def test_acquisition_rejects_non_https_sources(self) -> None:
        lock = {
            "entries": [
                {
                    "id": "insecure",
                    "dataset": "mnt",
                    "tile_ref": "L93_0001_0001",
                    "url": "http://example.invalid/source.tif",
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            _selected_entries(lock, lod0_tiles=set())

    def test_acquisition_excludes_only_complete_unpublished_direct_quartet(self) -> None:
        entries: list[dict[str, object]] = []
        for tile_ref in ("L93_0001_0001", "L93_0002_0001"):
            for dataset in ("lidar", "mnt", "mns", "mnh", "ortho20", "ortho50"):
                missing_direct = (
                    tile_ref == "L93_0001_0001"
                    and dataset in {"lidar", "mnt", "mns", "mnh"}
                )
                entries.append(
                    {
                        "id": f"{tile_ref}:{dataset}",
                        "dataset": dataset,
                        "tile_ref": tile_ref,
                        "url": (
                            ""
                            if missing_direct
                            else f"https://example.invalid/{tile_ref}/{dataset}"
                        ),
                        "resolution_status": (
                            "unresolved" if missing_direct else "available"
                        ),
                    }
                )
        lock: dict[str, object] = {"entries": entries}

        selected = _selected_entries(
            lock,
            lod0_tiles={"L93_0001_0001", "L93_0002_0001"},
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {
                item["dataset"]
                for item in selected
                if item["tile_ref"] == "L93_0001_0001"
            },
            {"ortho20", "ortho50"},
        )
        self.assertEqual(
            lock["unpublished_direct_coverage"],
            {
                "schema_version": 1,
                "state": "UNPUBLISHED_DIRECT_ELEVATION_QUARTETS_LOCKED",
                "datasets": ["lidar", "mnh", "mns", "mnt"],
                "tile_refs": ["L93_0001_0001"],
                "tile_count": 1,
                "representation": "excluded_no_synthetic_surface",
            },
        )

    def test_acquisition_rejects_partial_unpublished_direct_quartet(self) -> None:
        entries = [
            {
                "id": f"L93_0001_0001:{dataset}",
                "dataset": dataset,
                "tile_ref": "L93_0001_0001",
                "url": (
                    ""
                    if dataset in {"lidar", "mnt", "mns"}
                    else f"https://example.invalid/{dataset}"
                ),
                "resolution_status": (
                    "unresolved"
                    if dataset in {"lidar", "mnt", "mns"}
                    else "available"
                ),
            }
            for dataset in ("lidar", "mnt", "mns", "mnh", "ortho20", "ortho50")
        ]
        with self.assertRaisesRegex(RuntimeError, "incomplete direct elevation quartet"):
            _selected_entries({"entries": entries}, lod0_tiles={"L93_0001_0001"})

    def test_existing_download_requires_the_locked_size_and_prior_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.laz"
            destination.write_bytes(b"locked-source")
            entry = {
                "url": "https://example.invalid/tile.laz",
                "content_length_bytes": destination.stat().st_size + 1,
            }
            with self.assertRaisesRegex(RuntimeError, "size differs"):
                _download(entry, destination, timeout=1)

            entry["content_length_bytes"] = destination.stat().st_size
            entry["download"] = {"sha256": "0" * 64}
            with self.assertRaisesRegex(RuntimeError, "SHA-256 differs"):
                _download(entry, destination, timeout=1)

    def test_complete_partial_is_recovered_without_another_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.laz"
            partial = destination.with_suffix(".laz.partial")
            partial.write_bytes(b"complete-source")
            entry = {
                "url": "https://example.invalid/tile.laz",
                "content_length_bytes": partial.stat().st_size,
                "relative_path": "lidar/tile.laz",
            }
            with patch("fireviewer_sdg.zone_scenes.urlopen") as urlopen:
                _download(entry, destination, timeout=1)
            self.assertTrue(destination.is_file())
            self.assertFalse(partial.exists())
        urlopen.assert_not_called()
        self.assertEqual(entry["download"]["state"], "recovered_complete_partial")
        self.assertEqual(entry["download"]["bytes"], len(b"complete-source"))

    def test_wms_download_verifies_content_length_and_tiff_signature(self) -> None:
        payload = _geotiff_bytes()
        source_url = (
            "https://example.invalid/wms?SERVICE=WMS&FORMAT=image/tiff&"
            "WIDTH=2&HEIGHT=2&BBOX=100,200,102,202"
        )

        class Response:
            status = 200
            headers = {
                "Content-Length": str(len(payload)),
                "Content-Type": "image/tiff",
            }

            def __init__(self) -> None:
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return source_url

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return payload

        entry = {
            "id": "mnt",
            "dataset": "mnt",
            "url": source_url,
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.tif"
            with patch(
                "fireviewer_sdg.zone_scenes.urlopen",
                return_value=Response(),
            ):
                _download(entry, destination, timeout=1)
            self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(entry["content_length_bytes"], len(payload))
        self.assertEqual(entry["download"]["state"], "downloaded")

    def test_wms_download_rejects_wrong_content_length_atomically(self) -> None:
        payload = _geotiff_bytes()
        source_url = (
            "https://example.invalid/wms?SERVICE=WMS&FORMAT=image/tiff&"
            "WIDTH=2&HEIGHT=2&BBOX=100,200,102,202"
        )

        class Response:
            status = 200
            headers = {
                "Content-Length": str(len(payload) + 1),
                "Content-Type": "image/tiff",
            }

            def __init__(self) -> None:
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return source_url

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return payload

        entry = {
            "id": "mnt",
            "dataset": "mnt",
            "url": source_url,
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.tif"
            with (
                patch(
                    "fireviewer_sdg.zone_scenes.urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(RuntimeError, "Content-Length"),
            ):
                _download(entry, destination, timeout=1)
            self.assertFalse(destination.exists())
            self.assertTrue(
                destination.with_suffix(".tif.partial").is_file()
            )

    def test_wms_download_rejects_wrong_format_before_promotion(self) -> None:
        payload = b"not-tiff"
        source_url = (
            "https://example.invalid/wms?SERVICE=WMS&FORMAT=image/tiff&"
            "WIDTH=2&HEIGHT=2&BBOX=100,200,102,202"
        )

        class Response:
            status = 200
            headers = {
                "Content-Length": str(len(payload)),
                "Content-Type": "image/tiff",
            }

            def __init__(self) -> None:
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return source_url

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return payload

        entry = {
            "id": "mnt",
            "dataset": "mnt",
            "url": source_url,
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.tif"
            with (
                patch(
                    "fireviewer_sdg.zone_scenes.urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(RuntimeError, "native tiff image"),
            ):
                _download(entry, destination, timeout=1)
            self.assertFalse(destination.exists())

    def test_wms_chunked_geotiff_uses_native_geospatial_validation(self) -> None:
        source_url = (
            "https://example.invalid/wms?SERVICE=WMS&FORMAT=image/tiff&"
            "WIDTH=2&HEIGHT=2&BBOX=100,200,102,202"
        )

        class Response:
            status = 200
            headers = {
                "Transfer-Encoding": "chunked",
                "Content-Type": "image/tiff",
            }

            def __init__(self, payload: bytes) -> None:
                self.payload = payload
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return source_url

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return self.payload

        entry = {
            "id": "mnt",
            "dataset": "mnt",
            "url": source_url,
        }
        payload = _geotiff_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.tif"
            with patch(
                "fireviewer_sdg.zone_scenes.urlopen",
                return_value=Response(payload),
            ):
                _download(entry, destination, timeout=1)
            self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(
            entry["size_measurement"], "download_native_validated_size"
        )
        self.assertEqual(entry["content_length_bytes"], len(payload))

    def test_wms_native_validation_rejects_dimensions_bbox_and_crs(self) -> None:
        source_url = (
            "https://example.invalid/wms?SERVICE=WMS&FORMAT=image/tiff&"
            "WIDTH=2&HEIGHT=2&BBOX=100,200,102,202"
        )

        class Response:
            status = 200
            headers = {
                "Transfer-Encoding": "chunked",
                "Content-Type": "image/tiff",
            }

            def __init__(self, payload: bytes) -> None:
                self.payload = payload
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return source_url

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return self.payload

        cases = (
            ("dimensions", _geotiff_bytes(width=3), "dimensions"),
            (
                "bbox",
                _geotiff_bytes(bbox=(101.0, 200.0, 103.0, 202.0)),
                "georeferencing",
            ),
            (
                "half-pixel-offset",
                _geotiff_bytes(tiepoint_xy=(99.5, 202.5)),
                "georeferencing",
            ),
            ("crs", _geotiff_bytes(epsg="EPSG:4326|"), "EPSG:2154"),
        )
        for name, payload, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "tile.tif"
                entry = {
                    "id": "mnt",
                    "dataset": "mnt",
                    "url": source_url,
                }
                with (
                    patch(
                        "fireviewer_sdg.zone_scenes.urlopen",
                        return_value=Response(payload),
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    _download(entry, destination, timeout=1)
                self.assertFalse(destination.exists())

    def test_light_profile_uses_full_zone_context_camera_lod0_and_locked_assets_without_lidar(self) -> None:
        rows = [
            {
                "tile_ref": f"L93_{x:04d}_{y:04d}",
                "xmin": str(x * 1000),
                "ymin": str(y * 1000),
                "xmax": str((x + 1) * 1000),
                "ymax": str((y + 1) * 1000),
                "ortho_wms_20cm": (
                    "https://example.invalid/wms?SERVICE=WMS&REQUEST=GetMap&"
                    f"WIDTH=5000&HEIGHT=5000&BBOX={x * 1000},{y * 1000},{(x + 1) * 1000},{(y + 1) * 1000}"
                ),
            }
            for y in range(2)
            for x in range(2)
        ]
        full_lock = {
            "catalog_inventory_sha256": "a" * 64,
            "entries": [
                {
                    "dataset": "mnt",
                    "url": (
                        "https://example.invalid/wms?SERVICE=WMS&REQUEST=GetMap&"
                        "LAYERS=IGNF_MNT-LIDAR-HD&WIDTH=2000&HEIGHT=2000"
                    ),
                    "wfs_layer": "IGNF_MNT-LIDAR-HD:dalle",
                }
            ],
        }
        lock = _light_source_lock_from_full(
            full_lock=full_lock, zone_id="Z16", rows=rows
        )
        self.assertEqual(lock["source_profile"], "light")
        self.assertEqual(
            [item["dataset"] for item in lock["entries"][:9]],
            ["terrain_lod3", "ortho_lod2", "ortho_lod2", "ortho_lod2", "ortho_lod2", "ortho_lod0", "ortho_lod0", "ortho_lod0", "ortho_lod0"],
        )
        self.assertTrue(all(item["dataset"] == "assets" for item in lock["entries"][9:]))
        self.assertEqual(lock["light_profile"]["imagery"]["resolution_metres"], 2)
        self.assertEqual(lock["light_profile"]["hero_imagery"]["resolution_metres"], 0.2)
        query = parse_qs(urlparse(lock["entries"][0]["url"]).query)
        self.assertEqual(query["WIDTH"], ["2500"])
        self.assertEqual(query["HEIGHT"], ["2500"])
        self.assertEqual(query["BBOX"], ["0,0,2000,2000"])
        self.assertEqual(
            _selected_entries(lock, lod0_tiles={"L93_0000_0000"}), lock["entries"]
        )

    def test_acquisition_rejects_an_unsafe_worker_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            production = ZoneSceneProduction(
                catalog_root=catalog,
                workspace_root=base / "workspace",
                zone_id="Z16",
            )
            with self.assertRaisesRegex(ValueError, "workers"):
                production.acquire(lod0_tiles=(), download_workers=129)
            with self.assertRaisesRegex(ValueError, "direct download workers"):
                production.acquire(
                    lod0_tiles=(), direct_download_workers=129
                )
        self.assertEqual(MAX_DIRECT_DOWNLOAD_WORKERS, 128)
        self.assertEqual(DEFAULT_DIRECT_DOWNLOAD_WORKERS, 128)
        self.assertEqual(MAX_RASTER_DOWNLOAD_WORKERS, 128)
        self.assertEqual(DEFAULT_RASTER_DOWNLOAD_WORKERS, 128)

    def test_review_camera_grid_derives_twelve_lod0_tiles(self) -> None:
        rows = [
            {
                "tile_ref": f"L93_{x:04d}_{y:04d}",
                "xmin": str(x * 1000),
                "ymin": str(y * 1000),
                "xmax": str((x + 1) * 1000),
                "ymax": str((y + 1) * 1000),
            }
            for y in range(0, 20)
            for x in range(0, 20)
        ]
        derived = _review_camera_lod0_tiles(rows)
        self.assertEqual(len(derived), 12)
        self.assertIn("L93_0003_0003", derived)
        self.assertIn("L93_0015_0013", derived)

    def test_capacity_is_bounded_from_locked_wms_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            entries = [
                {
                    "id": "L93_0001_0001:mnt",
                    "dataset": "mnt",
                    "url": (
                        "https://example.invalid/wms?SERVICE=WMS&WIDTH=2000&HEIGHT=2000"
                    ),
                    "content_length_bytes": None,
                },
                {
                    "id": "L93_0002_0001:ortho50",
                    "dataset": "ortho50",
                    "url": (
                        "https://example.invalid/wms?SERVICE=WMS&WIDTH=1000&HEIGHT=1000"
                    ),
                    "content_length_bytes": None,
                },
            ]
            capacity = _assert_capacity(
                workspace, entries, minimum_free_gib=0
            )
        self.assertEqual(capacity["announced_download_bytes"], 0)
        self.assertEqual(
            capacity["wms_uncompressed_bound_bytes"],
            (2000 * 2000 * 4) + (1000 * 1000 * 4),
        )

    def test_capacity_refuses_unbounded_direct_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "cannot be bounded"):
                _assert_capacity(
                    Path(directory),
                    [
                        {
                            "id": "L93_0001_0001:lidar",
                            "dataset": "lidar",
                            "url": "https://example.invalid/tile.laz",
                            "content_length_bytes": None,
                        }
                    ],
                    minimum_free_gib=0,
                )

    def test_measurement_probes_one_source_per_dataset(self) -> None:
        entries = [
            {"id": "a", "dataset": "mnt"},
            {"id": "b", "dataset": "mnt"},
            {"id": "c", "dataset": "mnh"},
        ]
        self.assertEqual(
            [entry["id"] for entry in _measurement_samples(entries)],
            ["c", "a"],
        )

    def test_measurement_keeps_each_direct_download_for_a_capacity_probe(self) -> None:
        entries = [
            {"id": "lidar-a", "dataset": "lidar", "url": "https://example.invalid/a.laz"},
            {"id": "lidar-b", "dataset": "lidar", "url": "https://example.invalid/b.laz"},
            {
                "id": "mnt-a",
                "dataset": "mnt",
                "url": "https://example.invalid/wms?SERVICE=WMS&WIDTH=1000&HEIGHT=1000",
            },
            {
                "id": "mnt-b",
                "dataset": "mnt",
                "url": "https://example.invalid/wms?SERVICE=WMS&WIDTH=1000&HEIGHT=1000",
            },
        ]
        self.assertEqual(
            [entry["id"] for entry in _measurement_samples(entries)],
            ["lidar-a", "lidar-b", "mnt-a"],
        )

    def test_direct_measurement_retries_only_unresolved_and_checkpoints_successes(
        self,
    ) -> None:
        entries = [
            {"id": "lidar-a", "dataset": "lidar"},
            {"id": "lidar-b", "dataset": "lidar"},
        ]
        calls: list[list[str]] = []
        checkpoints: list[list[int | None]] = []

        def fake_measure(batch: list[dict[str, object]], **_: object) -> None:
            calls.append([str(entry["id"]) for entry in batch])
            if len(calls) == 1:
                batch[0]["content_length_bytes"] = 10
            else:
                batch[0]["content_length_bytes"] = 20

        with patch(
            "fireviewer_sdg.zone_scenes._measure_direct_entries",
            side_effect=fake_measure,
        ):
            unresolved = _measure_direct_entries_with_retries(
                entries,
                timeout=2,
                max_workers=2,
                attempts=3,
                request_pacer=_RequestStartPacer(0.001),
                checkpoint=lambda: checkpoints.append(
                    [
                        entry.get("content_length_bytes")
                        if isinstance(entry.get("content_length_bytes"), int)
                        else None
                        for entry in entries
                    ]
                ),
            )

        self.assertEqual(unresolved, [])
        self.assertEqual(calls, [["lidar-a", "lidar-b"], ["lidar-b"]])
        self.assertEqual(checkpoints, [[10, None], [10, 20]])

    def test_direct_measurement_remains_fail_closed_after_bounded_retries(
        self,
    ) -> None:
        entries = [{"id": "lidar-a", "dataset": "lidar"}]
        checkpoints = 0

        def checkpoint() -> None:
            nonlocal checkpoints
            checkpoints += 1

        with patch("fireviewer_sdg.zone_scenes._measure_direct_entries"):
            unresolved = _measure_direct_entries_with_retries(
                entries,
                timeout=2,
                max_workers=1,
                attempts=3,
                request_pacer=_RequestStartPacer(0.001),
                checkpoint=checkpoint,
            )

        self.assertEqual(unresolved, entries)
        self.assertEqual(checkpoints, 3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "cannot be bounded"):
                _assert_capacity(
                    Path(directory),
                    entries,
                    minimum_free_gib=0,
                )

    def test_measurement_uses_one_byte_range_when_head_has_no_length(self) -> None:
        class Response:
            def __init__(self, headers: dict[str, str]) -> None:
                self.headers = headers

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self, _size: int) -> bytes:
                return b"x"

        entry = {"id": "lidar", "url": "https://example.invalid/tile.laz"}
        with patch(
            "fireviewer_sdg.zone_scenes.urlopen",
            side_effect=[
                Response({"Content-Type": "application/octet-stream"}),
                Response(
                    {
                        "Content-Type": "application/octet-stream",
                        "Content-Range": "bytes 0-0/202383520",
                    }
                ),
            ],
        ):
            _measure(entry, timeout=1.0)
        self.assertEqual(entry["content_length_bytes"], 202383520)
        self.assertEqual(entry["size_measurement"], "range_content_range_0_0")

    def test_measurement_reuses_a_locked_direct_source_size(self) -> None:
        entry = {
            "id": "lidar",
            "url": "https://example.invalid/tile.laz",
            "content_length_bytes": 202383520,
            "size_measurement": "range_content_range_0_0",
        }
        with patch("fireviewer_sdg.zone_scenes.urlopen") as urlopen:
            _measure(entry, timeout=1.0)
        urlopen.assert_not_called()

    def test_measurement_captures_a_strong_remote_etag(self) -> None:
        class Response:
            headers = {
                "Content-Range": "bytes 0-0/8",
                "Content-Type": "application/octet-stream",
                "ETag": '"version-1"',
            }

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self, _size: int) -> bytes:
                return b"a"

        entry = {"url": "https://example.invalid/tile.laz"}
        with patch(
            "fireviewer_sdg.zone_scenes.urlopen",
            return_value=Response(),
        ):
            _measure(
                entry,
                timeout=1,
                range_first=True,
                require_remote_validator=True,
            )
        self.assertEqual(
            entry["remote_version_validator"],
            {"kind": "strong_etag", "value": '"version-1"'},
        )

    def test_measurement_does_not_enable_segmentation_without_validator(
        self,
    ) -> None:
        class Response:
            headers = {
                "Content-Range": "bytes 0-0/8",
                "Content-Length": "8",
            }

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self, _size: int) -> bytes:
                return b"a"

        entry = {"url": "https://example.invalid/tile.laz"}
        with patch(
            "fireviewer_sdg.zone_scenes.urlopen",
            side_effect=[Response(), Response()],
        ):
            _measure(
                entry,
                timeout=1,
                range_first=True,
                require_remote_validator=True,
            )
        self.assertNotIn("remote_version_validator", entry)
        self.assertIn("no strong ETag or Last-Modified", entry["probe_error"])

    def test_direct_measurements_use_range_first_with_overlapping_workers(
        self,
    ) -> None:
        entries = [
            {
                "id": f"lidar-{index}",
                "dataset": "lidar",
                "url": f"https://example.invalid/{index}.laz",
            }
            for index in range(3)
        ]
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()
        all_started = threading.Barrier(len(entries))

        def fake_measure(
            _entry: dict[str, object],
            *,
            timeout: float,
            range_first: bool,
            before_request: object,
        ) -> None:
            nonlocal active, maximum_active
            self.assertEqual(timeout, 2.0)
            self.assertTrue(range_first)
            self.assertTrue(callable(before_request))
            before_request()
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                all_started.wait(timeout=2.0)
            finally:
                with active_lock:
                    active -= 1

        with (
            patch(
                "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                0.001,
            ),
            patch(
                "fireviewer_sdg.zone_scenes._measure",
                side_effect=fake_measure,
            ),
        ):
            _measure_direct_entries(entries, timeout=2.0, max_workers=3)
        self.assertEqual(maximum_active, 3)

    def test_direct_downloads_overlap_without_skipping_checkpoints(self) -> None:
        entries = [
            {
                "id": f"lidar-{index}",
                "dataset": "lidar",
                "url": f"https://example.invalid/{index}.laz",
                "relative_path": f"lidar/{index}.laz",
            }
            for index in range(3)
        ]
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()
        all_started = threading.Barrier(len(entries))
        checkpoints: list[int] = []

        def fake_download(
            _entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            nonlocal active, maximum_active
            self.assertEqual(timeout, 2.0)
            self.assertTrue(callable(before_request))
            self.assertTrue(callable(should_cancel))
            before_request()
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                all_started.wait(timeout=2.0)
            finally:
                with active_lock:
                    active -= 1

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                0.001,
            ),
            patch(
                "fireviewer_sdg.zone_scenes._download",
                side_effect=fake_download,
            ),
        ):
            _download_direct_entries(
                entries,
                raw_root=Path(directory),
                timeout=2.0,
                max_workers=3,
                retries=3,
                checkpoint=lambda: checkpoints.append(1),
            )
        self.assertEqual(maximum_active, 3)
        self.assertEqual(len(checkpoints), 3)

    def test_direct_download_pool_supports_128_overlapping_tiles(self) -> None:
        worker_count = 128
        entries = [
            {
                "id": f"lidar-{index}",
                "dataset": "lidar",
                "url": f"https://example.invalid/{index}.laz",
                "relative_path": f"{index}.laz",
            }
            for index in range(worker_count)
        ]
        all_started = threading.Barrier(worker_count)
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()

        def fake_download(
            _entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            nonlocal active, maximum_active
            self.assertEqual(timeout, 2)
            self.assertTrue(callable(should_cancel))
            before_request()
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                all_started.wait(timeout=5)
            finally:
                with active_lock:
                    active -= 1

        checkpoints: list[int] = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                0.00001,
            ),
            patch("fireviewer_sdg.zone_scenes._download", side_effect=fake_download),
        ):
            _download_direct_entries(
                entries,
                raw_root=Path(directory),
                timeout=2,
                max_workers=worker_count,
                retries=1,
                checkpoint=lambda: checkpoints.append(1),
            )
        self.assertEqual(maximum_active, worker_count)
        self.assertEqual(len(checkpoints), worker_count)

    def test_raster_download_pool_supports_32_overlapping_sources(self) -> None:
        worker_count = 32
        entries = [
            {
                "id": f"raster-{index}",
                "dataset": "mnt",
                "url": (
                    "https://example.invalid/wms?SERVICE=WMS&"
                    f"FORMAT=image/tiff&TILE={index}"
                ),
                "relative_path": f"{index}.tif",
            }
            for index in range(worker_count)
        ]
        all_started = threading.Barrier(worker_count)
        active = 0
        maximum_active = 0
        active_lock = threading.Lock()

        def fake_download(
            _entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            nonlocal active, maximum_active
            self.assertEqual(timeout, 2)
            self.assertTrue(callable(should_cancel))
            before_request()
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                all_started.wait(timeout=5)
            finally:
                with active_lock:
                    active -= 1

        checkpoints: list[int] = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes._download", side_effect=fake_download),
        ):
            _download_raster_entries(
                entries,
                raw_root=Path(directory),
                timeout=2,
                max_workers=worker_count,
                retries=1,
                checkpoint=lambda: checkpoints.append(1),
                request_pacer=_RequestStartPacer(0.00001),
            )
        self.assertEqual(maximum_active, worker_count)
        self.assertEqual(len(checkpoints), worker_count)

    def test_segment_ranges_have_exact_coverage_without_overlap(self) -> None:
        ranges = _partition_byte_ranges(10, 3)
        self.assertEqual(
            [(item["start"], item["end"]) for item in ranges],
            [(0, 3), (4, 6), (7, 9)],
        )
        broken = [dict(item) for item in ranges]
        broken[1]["start"] = 5
        with self.assertRaisesRegex(RuntimeError, "exact, disjoint"):
            _assert_exact_range_coverage(broken, total_bytes=10)

    def test_tail_segmentation_activates_only_well_below_pool_size(self) -> None:
        entries = [
            {
                "id": f"lidar-{index}",
                "dataset": "lidar",
                "url": f"https://example.invalid/{index}.laz",
                "relative_path": f"{index}.laz",
                "content_length_bytes": 16,
                "remote_version_validator": {
                    "kind": "strong_etag",
                    "value": f'"version-{index}"',
                },
            }
            for index in range(33)
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes.TAIL_SEGMENT_MIN_BYTES", 1),
        ):
            raw_root = Path(directory)
            self.assertIsNone(
                _tail_segmentation_plan(
                    entries, raw_root=raw_root, max_workers=128
                )
            )
            plan = _tail_segmentation_plan(
                entries[:32], raw_root=raw_root, max_workers=128
            )
        self.assertIsNotNone(plan)
        self.assertLessEqual(
            sum(len(item["ranges"]) for item in plan),
            128,
        )

    def test_tail_segmentation_downloads_disjoint_ranges_and_assembles(
        self,
    ) -> None:
        payload = b"abcdefgh"
        requested_ranges: list[str] = []
        request_lock = threading.Lock()

        class Response:
            status = 206

            def __init__(self, start: int, end: int) -> None:
                self.headers = {
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "ETag": '"version-1"',
                }
                self.body = payload[start : end + 1]
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return self.body

        def fake_urlopen(request: object, *, timeout: float) -> Response:
            self.assertEqual(timeout, 2)
            raw_range = request.get_header("Range")
            self.assertEqual(request.get_header("If-range"), '"version-1"')
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", raw_range)
            self.assertIsNotNone(match)
            start, end = (int(value) for value in match.groups())
            with request_lock:
                requested_ranges.append(raw_range)
            return Response(start, end)

        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "remote_version_validator": {
                "kind": "strong_etag",
                "value": '"version-1"',
            },
        }
        checkpoints: list[int] = []
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes.TAIL_SEGMENT_MIN_BYTES", 1),
            patch(
                "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                0.00001,
            ),
            patch(
                "fireviewer_sdg.zone_scenes.urlopen",
                side_effect=fake_urlopen,
            ),
        ):
            raw_root = Path(directory)
            _download_direct_entries(
                [entry],
                raw_root=raw_root,
                timeout=2,
                max_workers=4,
                retries=1,
                checkpoint=lambda: checkpoints.append(1),
            )
            destination = raw_root / "lidar" / "tile.laz"
            self.assertEqual(destination.read_bytes(), payload)
            self.assertTrue(
                destination.with_suffix(".laz.ranges.json").is_file()
            )
            self.assertFalse(
                (destination.parent / ".tile.laz.segments").exists()
            )
        self.assertEqual(
            sorted(requested_ranges),
            ["bytes=0-1", "bytes=2-3", "bytes=4-5", "bytes=6-7"],
        )
        self.assertEqual(checkpoints, [1])
        self.assertEqual(entry["download"]["state"], "downloaded_segmented")
        self.assertEqual(len(entry["download"]["range_receipts"]), 4)

    def test_tail_segmentation_resumes_verified_range_receipts(self) -> None:
        payload = b"abcdefgh"
        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "remote_version_validator": {
                "kind": "strong_etag",
                "value": '"version-1"',
            },
        }
        ranges = _partition_byte_ranges(len(payload), 4)
        requested_ranges: list[str] = []

        class Response:
            status = 206

            def __init__(self, start: int, end: int) -> None:
                self.headers = {
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "ETag": '"version-1"',
                }
                self.body = payload[start : end + 1]
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return self.body

        def fake_urlopen(request: object, *, timeout: float) -> Response:
            raw_range = request.get_header("Range")
            requested_ranges.append(raw_range)
            start, end = (
                int(value)
                for value in raw_range.removeprefix("bytes=").split("-")
            )
            return Response(start, end)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes.TAIL_SEGMENT_MIN_BYTES", 1),
            patch(
                "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                0.00001,
            ),
        ):
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "tile.laz"
            segment, receipt_path, _temporary = _segment_paths(
                destination, ranges[0]
            )
            segment.parent.mkdir(parents=True)
            segment.write_bytes(payload[0:2])
            write_json(
                receipt_path,
                {
                    "schema_version": 1,
                    "source_url": entry["url"],
                    "remote_version_validator": entry[
                        "remote_version_validator"
                    ],
                    **ranges[0],
                    "sha256": _digest(segment),
                },
            )
            with patch(
                "fireviewer_sdg.zone_scenes.urlopen",
                side_effect=fake_urlopen,
            ):
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=2,
                    max_workers=4,
                    retries=1,
                    checkpoint=lambda: None,
                )
            self.assertEqual(destination.read_bytes(), payload)
        self.assertNotIn("bytes=0-1", requested_ranges)
        self.assertEqual(len(requested_ranges), 3)

    def test_tail_segmentation_fails_closed_without_remote_validator(
        self,
    ) -> None:
        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": 8,
        }

        def fake_download(
            work_entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            work_entry["download"] = {"state": "sequential"}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes.TAIL_SEGMENT_MIN_BYTES", 1),
            patch("fireviewer_sdg.zone_scenes._download", side_effect=fake_download),
            patch("fireviewer_sdg.zone_scenes.urlopen") as urlopen,
        ):
            _download_direct_entries(
                [entry],
                raw_root=Path(directory),
                timeout=2,
                max_workers=4,
                retries=1,
                checkpoint=lambda: None,
            )
        urlopen.assert_not_called()
        self.assertEqual(entry["download"]["state"], "sequential")

    def test_tail_resume_reconciles_promoted_file_before_checkpoint(self) -> None:
        completed = {
            "id": "completed",
            "dataset": "lidar",
            "url": "https://example.invalid/completed.laz",
            "relative_path": "completed.laz",
            "content_length_bytes": 4,
        }
        checkpoints: list[int] = []
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "completed.laz"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"done")
            _download_direct_entries(
                [completed],
                raw_root=raw_root,
                timeout=2,
                max_workers=128,
                retries=1,
                checkpoint=lambda: checkpoints.append(1),
            )
        self.assertEqual(completed["download"]["state"], "verified_existing")
        self.assertEqual(
            completed["download"]["sha256"],
            hashlib.sha256(b"done").hexdigest(),
        )
        self.assertEqual(checkpoints, [1])

    def test_tail_segmentation_rejects_changed_content_range(self) -> None:
        class Response:
            status = 206
            headers = {
                "Content-Range": "bytes 1-2/8",
                "ETag": '"version-1"',
            }

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                return b"ab"

        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": 8,
            "remote_version_validator": {
                "kind": "strong_etag",
                "value": '"version-1"',
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            with (
                patch("fireviewer_sdg.zone_scenes.TAIL_SEGMENT_MIN_BYTES", 1),
                patch(
                    "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                    0.00001,
                ),
                patch(
                    "fireviewer_sdg.zone_scenes.urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(RuntimeError, "Content-Range"),
            ):
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=2,
                    max_workers=4,
                    retries=1,
                    checkpoint=lambda: None,
                )
            self.assertFalse((raw_root / "lidar" / "tile.laz").exists())

    def test_tail_segmentation_never_promotes_wrong_final_sha(self) -> None:
        payload = b"abcdefgh"

        class Response:
            status = 206

            def __init__(self, start: int, end: int) -> None:
                self.headers = {
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "ETag": '"version-1"',
                }
                self.body = payload[start : end + 1]
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return self.body

        def fake_urlopen(request: object, *, timeout: float) -> Response:
            start, end = (
                int(value)
                for value in request.get_header("Range")
                .removeprefix("bytes=")
                .split("-")
            )
            return Response(start, end)

        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "remote_version_validator": {
                "kind": "strong_etag",
                "value": '"version-1"',
            },
            "download": {"sha256": hashlib.sha256(b"bad-data").hexdigest()},
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "tile.laz"
            with (
                patch("fireviewer_sdg.zone_scenes.TAIL_SEGMENT_MIN_BYTES", 1),
                patch(
                    "fireviewer_sdg.zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS",
                    0.00001,
                ),
                patch(
                    "fireviewer_sdg.zone_scenes.urlopen",
                    side_effect=fake_urlopen,
                ),
                self.assertRaisesRegex(RuntimeError, "locked SHA-256"),
            ):
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=2,
                    max_workers=4,
                    retries=1,
                    checkpoint=lambda: None,
                )
            self.assertFalse(destination.exists())

    def test_measurements_and_downloads_can_share_one_global_pacer(self) -> None:
        measurement = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "lidar/tile.laz",
        }
        observed: list[object] = []
        pacer = _RequestStartPacer(0.001)

        def fake_measure(
            _entry: dict[str, object],
            *,
            timeout: float,
            range_first: bool,
            before_request: object,
        ) -> None:
            self.assertEqual(timeout, 1)
            self.assertTrue(range_first)
            observed.append(getattr(before_request, "__self__", None))

        def fake_download(
            _entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            self.assertEqual(timeout, 1)
            self.assertTrue(callable(should_cancel))
            observed.append(getattr(before_request, "__self__", None))

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes._measure", side_effect=fake_measure),
            patch("fireviewer_sdg.zone_scenes._download", side_effect=fake_download),
        ):
            _measure_direct_entries(
                [measurement],
                timeout=1,
                max_workers=1,
                request_pacer=pacer,
            )
            _download_direct_entries(
                [measurement],
                raw_root=Path(directory),
                timeout=1,
                max_workers=1,
                retries=1,
                checkpoint=lambda: None,
                request_pacer=pacer,
            )
        self.assertEqual(observed, [pacer, pacer])

    def test_resume_rejects_wrong_content_range_without_touching_partial(self) -> None:
        class Response:
            status = 206
            headers = {"Content-Range": "bytes 0-3/8"}

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                return b"more"

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.laz"
            partial = destination.with_suffix(".laz.partial")
            partial.write_bytes(b"keep")
            entry = {
                "url": "https://example.invalid/tile.laz",
                "content_length_bytes": 8,
            }
            with (
                patch(
                    "fireviewer_sdg.zone_scenes.urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(RuntimeError, "Content-Range"),
            ):
                _download(entry, destination, timeout=1)
            self.assertEqual(partial.read_bytes(), b"keep")
            self.assertFalse(destination.exists())

    def test_resume_keeps_old_partial_when_server_ignores_range_and_fails(self) -> None:
        class Response:
            status = 200
            headers: dict[str, str] = {}

            def __init__(self) -> None:
                self.calls = 0

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    return b"new"
                raise OSError("connection lost")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.laz"
            partial = destination.with_suffix(".laz.partial")
            partial.write_bytes(b"old")
            entry = {
                "url": "https://example.invalid/tile.laz",
                "content_length_bytes": 6,
            }
            with (
                patch(
                    "fireviewer_sdg.zone_scenes.urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(OSError, "connection lost"),
            ):
                _download(entry, destination, timeout=1)
            self.assertEqual(partial.read_bytes(), b"old")
            self.assertFalse(destination.exists())

    def test_resume_ignored_range_replaces_only_after_validation(self) -> None:
        class Response:
            status = 200
            headers: dict[str, str] = {}

            def __init__(self) -> None:
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return b"new-full"

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.laz"
            partial = destination.with_suffix(".laz.partial")
            partial.write_bytes(b"old")
            entry = {
                "url": "https://example.invalid/tile.laz",
                "content_length_bytes": len(b"new-full"),
            }
            with patch(
                "fireviewer_sdg.zone_scenes.urlopen",
                return_value=Response(),
            ):
                _download(entry, destination, timeout=1)
            self.assertEqual(destination.read_bytes(), b"new-full")
            self.assertFalse(partial.exists())
            self.assertFalse(
                destination.with_suffix(".laz.restart.partial").exists()
            )

    def test_complete_partial_with_wrong_locked_sha_is_redownloaded(self) -> None:
        class Response:
            status = 200
            headers: dict[str, str] = {}

            def __init__(self) -> None:
                self.sent = False

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def geturl(self) -> str:
                return "https://example.invalid/tile.laz"

            def read(self, _size: int) -> bytes:
                if self.sent:
                    return b""
                self.sent = True
                return b"good"

        expected = hashlib.sha256(b"good").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "tile.laz"
            partial = destination.with_suffix(".laz.partial")
            partial.write_bytes(b"evil")
            entry = {
                "url": "https://example.invalid/tile.laz",
                "content_length_bytes": 4,
                "download": {"sha256": expected},
            }
            with patch(
                "fireviewer_sdg.zone_scenes.urlopen",
                return_value=Response(),
            ) as urlopen:
                _download(entry, destination, timeout=1)
            urlopen.assert_called_once()
            self.assertEqual(destination.read_bytes(), b"good")
            self.assertEqual(entry["download"]["sha256"], expected)

    def test_direct_download_rejects_duplicate_destinations_before_workers(self) -> None:
        entries = [
            {
                "id": "a",
                "dataset": "lidar",
                "url": "https://example.invalid/a.laz",
                "relative_path": "lidar/same.laz",
            },
            {
                "id": "b",
                "dataset": "lidar",
                "url": "https://example.invalid/b.laz",
                "relative_path": "lidar/same.laz",
            },
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes._download") as download,
            self.assertRaisesRegex(RuntimeError, "same destination"),
        ):
            _download_direct_entries(
                entries,
                raw_root=Path(directory),
                timeout=1,
                max_workers=64,
                retries=1,
                checkpoint=lambda: None,
            )
        download.assert_not_called()

    def test_direct_download_checkpoint_never_observes_worker_mutation(self) -> None:
        entries = [
            {
                "id": name,
                "dataset": "lidar",
                "url": f"https://example.invalid/{name}.laz",
                "relative_path": f"{name}.laz",
            }
            for name in ("first", "second")
        ]
        second_started = threading.Event()
        release_second = threading.Event()
        checkpoint_states: list[bool] = []

        def fake_download(
            entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            self.assertEqual(timeout, 1)
            entry["worker_mutation"] = True
            if entry["id"] == "second":
                second_started.set()
                self.assertTrue(release_second.wait(timeout=2))
            else:
                self.assertTrue(second_started.wait(timeout=2))

        def checkpoint() -> None:
            checkpoint_states.append("worker_mutation" in entries[1])
            release_second.set()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes._download", side_effect=fake_download),
        ):
            _download_direct_entries(
                entries,
                raw_root=Path(directory),
                timeout=1,
                max_workers=2,
                retries=1,
                checkpoint=checkpoint,
            )
        self.assertEqual(checkpoint_states, [False, True])

    def test_direct_download_failure_cancels_peers_without_retry_sleep(self) -> None:
        entries = [
            {
                "id": name,
                "dataset": "lidar",
                "url": f"https://example.invalid/{name}.laz",
                "relative_path": f"{name}.laz",
            }
            for name in ("failure", "peer")
        ]
        peer_started = threading.Event()
        peer_observed_cancel = threading.Event()

        def fake_download(
            entry: dict[str, object],
            _destination: Path,
            *,
            timeout: float,
            before_request: object,
            should_cancel: object,
        ) -> None:
            self.assertEqual(timeout, 1)
            self.assertTrue(callable(should_cancel))
            if entry["id"] == "failure":
                self.assertTrue(peer_started.wait(timeout=2))
                raise OSError("primary failure")
            peer_started.set()
            while not should_cancel():
                threading.Event().wait(0.001)
            peer_observed_cancel.set()
            raise RuntimeError("cancelled")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("fireviewer_sdg.zone_scenes._download", side_effect=fake_download),
            patch("fireviewer_sdg.zone_scenes.time.sleep") as sleep,
            self.assertRaisesRegex(OSError, "primary failure"),
        ):
            _download_direct_entries(
                entries,
                raw_root=Path(directory),
                timeout=1,
                max_workers=2,
                retries=3,
                checkpoint=lambda: None,
            )
        self.assertTrue(peer_observed_cancel.is_set())
        self.assertEqual(
            [item.args[0] for item in sleep.call_args_list],
            [1, 2],
        )

    def test_capacity_counts_only_remaining_partial_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            raw_root = workspace / "raw"
            partial = raw_root / "lidar" / "lidar" / "tile.laz.partial"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"1234")
            capacity = _assert_capacity(
                workspace,
                [
                    {
                        "id": "lidar",
                        "dataset": "lidar",
                        "url": "https://example.invalid/tile.laz",
                        "relative_path": "lidar/tile.laz",
                        "content_length_bytes": 10,
                    }
                ],
                minimum_free_gib=0,
                raw_root=raw_root,
            )
        self.assertEqual(capacity["announced_download_bytes"], 10)
        self.assertEqual(capacity["announced_remaining_download_bytes"], 6)
        self.assertEqual(capacity["expected_download_bytes"], 6)

    def test_capacity_recounts_complete_destination_with_wrong_locked_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            raw_root = workspace / "raw"
            destination = raw_root / "lidar" / "tile.laz"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"evil")
            capacity = _assert_capacity(
                workspace,
                [
                    {
                        "id": "lidar",
                        "dataset": "lidar",
                        "url": "https://example.invalid/tile.laz",
                        "relative_path": "tile.laz",
                        "content_length_bytes": 4,
                        "download": {
                            "sha256": hashlib.sha256(b"good").hexdigest()
                        },
                    }
                ],
                minimum_free_gib=0,
                raw_root=raw_root,
            )
        self.assertEqual(capacity["expected_download_bytes"], 4)

    def test_fast_resume_skips_payload_reads_for_400_verified_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            entries: list[dict[str, object]] = []
            for index in range(400):
                payload = f"tile-{index:03d}".encode("ascii")
                entry: dict[str, object] = {
                    "id": f"lidar-{index:03d}",
                    "dataset": "lidar",
                    "url": f"https://example.invalid/{index:03d}.laz",
                    "relative_path": f"{index:03d}.laz",
                    "content_length_bytes": len(payload),
                    "size_measurement": "range_content_range_0_0",
                }
                destination = raw_root / "lidar" / f"{index:03d}.laz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                stat = destination.stat()
                entry["download"] = {
                    "state": "downloaded",
                    "relpath": f"{index:03d}.laz",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "file_identity": {
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                    },
                    "source_fingerprint_sha256": (
                        _source_measurement_fingerprint(entry)
                    ),
                }
                entries.append(entry)
            checkpoints: list[int] = []
            with patch(
                "fireviewer_sdg.zone_scenes.sha256",
                side_effect=AssertionError("verified payload was read"),
            ) as digest:
                _download_direct_entries(
                    entries,
                    raw_root=raw_root,
                    timeout=1,
                    max_workers=128,
                    retries=1,
                    checkpoint=lambda: checkpoints.append(1),
                )
        digest.assert_not_called()
        self.assertEqual(checkpoints, [])

    def test_legacy_resume_hashes_once_then_records_fast_fingerprint(self) -> None:
        payload = b"legacy-complete"
        entry: dict[str, object] = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "size_measurement": "range_content_range_0_0",
            "download": {
                "state": "downloaded",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "tile.laz"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(payload)
            checkpoints: list[int] = []
            with patch(
                "fireviewer_sdg.zone_scenes.sha256",
                side_effect=_digest,
            ) as digest:
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=1,
                    max_workers=1,
                    retries=1,
                    checkpoint=lambda: checkpoints.append(1),
                )
            self.assertEqual(digest.call_count, 1)
            self.assertEqual(checkpoints, [1])
            self.assertIn("file_identity", entry["download"])
            self.assertEqual(
                entry["download"]["source_fingerprint_sha256"],
                _source_measurement_fingerprint(entry),
            )
            with patch(
                "fireviewer_sdg.zone_scenes.sha256",
                side_effect=AssertionError("migrated payload was read again"),
            ) as digest:
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=1,
                    max_workers=1,
                    retries=1,
                    checkpoint=lambda: checkpoints.append(1),
                )
            digest.assert_not_called()
            self.assertEqual(checkpoints, [1])

    def test_fast_resume_mtime_mutation_forces_one_rehash(self) -> None:
        payload = b"stable-content"
        entry: dict[str, object] = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "size_measurement": "range_content_range_0_0",
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "tile.laz"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(payload)
            stat = destination.stat()
            entry["download"] = {
                "state": "downloaded",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "file_identity": {
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                },
                "source_fingerprint_sha256": (
                    _source_measurement_fingerprint(entry)
                ),
            }
            changed_mtime = stat.st_mtime_ns + 1_000_000_000
            os.utime(
                destination,
                ns=(stat.st_atime_ns, changed_mtime),
            )
            checkpoints: list[int] = []
            with patch(
                "fireviewer_sdg.zone_scenes.sha256",
                side_effect=_digest,
            ) as digest:
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=1,
                    max_workers=1,
                    retries=1,
                    checkpoint=lambda: checkpoints.append(1),
                )
            self.assertEqual(digest.call_count, 1)
            self.assertEqual(checkpoints, [1])
            self.assertEqual(
                entry["download"]["file_identity"]["mtime_ns"],
                destination.stat().st_mtime_ns,
            )

    def test_fast_resume_size_mutation_is_not_trusted(self) -> None:
        payload = b"expected"
        entry: dict[str, object] = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "size_measurement": "range_content_range_0_0",
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "tile.laz"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(payload)
            stat = destination.stat()
            entry["download"] = {
                "state": "downloaded",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "file_identity": {
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                },
                "source_fingerprint_sha256": (
                    _source_measurement_fingerprint(entry)
                ),
            }
            with destination.open("ab") as stream:
                stream.write(b"!")
            with (
                patch(
                    "fireviewer_sdg.zone_scenes.sha256",
                    side_effect=AssertionError("wrong-size payload was read"),
                ) as digest,
                self.assertRaisesRegex(
                    RuntimeError, "size differs from its locked measurement"
                ),
            ):
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=1,
                    max_workers=1,
                    retries=1,
                    checkpoint=lambda: None,
                )
            digest.assert_not_called()

    def test_fast_resume_source_fingerprint_mutation_fails_closed(self) -> None:
        payload = b"expected"
        entry: dict[str, object] = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/original/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": len(payload),
            "size_measurement": "range_content_range_0_0",
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory)
            destination = raw_root / "lidar" / "tile.laz"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(payload)
            stat = destination.stat()
            entry["download"] = {
                "state": "downloaded",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "file_identity": {
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                },
                "source_fingerprint_sha256": (
                    _source_measurement_fingerprint(entry)
                ),
            }
            entry["url"] = "https://example.invalid/changed/tile.laz"
            with (
                patch(
                    "fireviewer_sdg.zone_scenes.sha256",
                    side_effect=AssertionError(
                        "mismatched source payload was read"
                    ),
                ) as digest,
                self.assertRaisesRegex(
                    RuntimeError, "another URL/measurement"
                ),
            ):
                _download_direct_entries(
                    [entry],
                    raw_root=raw_root,
                    timeout=1,
                    max_workers=1,
                    retries=1,
                    checkpoint=lambda: None,
                )
            digest.assert_not_called()

    def test_capacity_reserves_ranges_plus_atomic_assembly(self) -> None:
        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
            "relative_path": "tile.laz",
            "content_length_bytes": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            capacity = _assert_capacity(
                workspace,
                [entry],
                minimum_free_gib=0,
                raw_root=workspace / "raw",
                segmented_staging_entries=[entry],
            )
        self.assertEqual(capacity["segmented_staging_bytes"], 20)
        self.assertEqual(capacity["expected_download_bytes"], 20)

    def test_direct_download_retries_a_resumable_size_mismatch(self) -> None:
        entry = {
            "id": "lidar",
            "dataset": "lidar",
            "url": "https://example.invalid/tile.laz",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "fireviewer_sdg.zone_scenes._download",
                side_effect=[RuntimeError("size mismatch"), None],
            ) as download,
            patch("fireviewer_sdg.zone_scenes.time.sleep") as sleep,
        ):
            _download_with_retries(
                entry,
                Path(directory) / "tile.laz",
                timeout=2.0,
                retries=3,
            )
        self.assertEqual(download.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertEqual(entry["download_attempts"], 2)

    def test_vector_wfs_request_keeps_epsg2154_and_pagination_in_the_lock_url(self) -> None:
        url = _vector_request_url(
            layer="BDTOPO_V3:batiment",
            bbox=(878000, 6399000, 898000, 6419000),
            start_index=3000,
        )
        self.assertIn("TYPENAMES=BDTOPO_V3%3Abatiment", url)
        self.assertIn("SRSNAME=EPSG%3A2154", url)
        self.assertIn("STARTINDEX=3000", url)
        self.assertIn("COUNT=1000", url)

    def test_vector_layer_continues_full_page_without_number_matched(self) -> None:
        first = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": f"building.{index}", "geometry": None}
                for index in range(1000)
            ],
        }
        second = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": f"building.{1000 + index}",
                    "geometry": None,
                }
                for index in range(3)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_root = Path(directory) / "raw"
            with patch(
                "fireviewer_sdg.zone_scenes._download_vector_page",
                side_effect=[first, second],
            ) as download_page:
                receipt = _download_vector_layer(
                    raw_root=raw_root,
                    name="buildings",
                    layer="BDTOPO_V3:batiment",
                    bbox=(878000, 6399000, 898000, 6419000),
                    timeout=1.0,
                )
            merged = json.loads(
                (raw_root / receipt["download"]["relpath"]).read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(receipt["feature_count"], 1003)
        self.assertEqual(len(merged["features"]), 1003)
        self.assertEqual(download_page.call_count, 2)
        second_url = download_page.call_args_list[1].kwargs["url"]
        self.assertEqual(parse_qs(urlparse(second_url).query)["STARTINDEX"], ["1000"])

    def test_vector_layer_stops_after_empty_page_when_total_is_unknown(self) -> None:
        full = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": f"road.{index}", "geometry": None}
                for index in range(1000)
            ],
        }
        empty = {"type": "FeatureCollection", "features": []}
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "fireviewer_sdg.zone_scenes._download_vector_page",
                side_effect=[full, empty],
            ) as download_page:
                receipt = _download_vector_layer(
                    raw_root=Path(directory) / "raw",
                    name="roads",
                    layer="BDTOPO_V3:troncon_de_route",
                    bbox=(878000, 6399000, 898000, 6419000),
                    timeout=1.0,
                )
        self.assertEqual(receipt["feature_count"], 1000)
        self.assertEqual(len(receipt["urls"]), 2)
        self.assertEqual(download_page.call_count, 2)

    def test_vector_layer_rejects_duplicate_page_without_progress(self) -> None:
        full = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": f"water.{index}", "geometry": None}
                for index in range(1000)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "fireviewer_sdg.zone_scenes._download_vector_page",
                side_effect=[full, full],
            ):
                with self.assertRaisesRegex(RuntimeError, "no progress.*duplicate page"):
                    _download_vector_layer(
                        raw_root=Path(directory) / "raw",
                        name="hydrology",
                        layer="BDTOPO_V3:cours_d_eau",
                        bbox=(878000, 6399000, 898000, 6419000),
                        timeout=1.0,
                    )

    def test_vector_layer_rejects_feature_repeated_across_pages(self) -> None:
        first = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": f"forest.{index}", "geometry": None}
                for index in range(1000)
            ],
        }
        second = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": "forest.999", "geometry": None},
                {"type": "Feature", "id": "forest.1000", "geometry": None},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "fireviewer_sdg.zone_scenes._download_vector_page",
                side_effect=[first, second],
            ):
                with self.assertRaisesRegex(RuntimeError, "duplicate feature"):
                    _download_vector_layer(
                        raw_root=Path(directory) / "raw",
                        name="vegetation",
                        layer="BDTOPO_V3:zone_de_vegetation",
                        bbox=(878000, 6399000, 898000, 6419000),
                        timeout=1.0,
                    )

    def test_acquisition_lock_refuses_concurrent_owner_and_recovers_dead_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            zone_root.mkdir()
            with _acquisition_lock(zone_root):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with _acquisition_lock(zone_root):
                        pass
            stale = zone_root / ".acquisition.lock"
            stale.write_text('{"pid": 999999999}', encoding="utf-8")
            with _acquisition_lock(zone_root):
                self.assertTrue(stale.is_file())
            self.assertFalse(stale.exists())

    def test_build_does_not_spawn_composer_until_review_is_explicitly_requested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            production = ZoneSceneProduction(catalog_root=catalog, workspace_root=workspace, zone_id="Z16")
            production.preflight()
            state = _load_state(workspace)
            state["zones"]["Z16"]["phase"] = "sources_acquired"
            write_json(workspace / "zone-scenes" / "production-state.json", state)
            expected = {"root_usd": {"path": "build/Z16_root.usdc"}}
            with patch("fireviewer_sdg.zone_scenes.subprocess.run") as native_build:
                with patch.object(production, "register_build", return_value=expected):
                    with patch.object(production, "open_review") as review:
                        result = production.build(timeout=1)
            self.assertIs(result, expected)
            native_build.assert_called_once()
            review.assert_not_called()

    def test_validated_scene_launches_composer_without_claiming_human_qa(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            production = ZoneSceneProduction(catalog_root=catalog, workspace_root=workspace, zone_id="Z16")
            production.preflight()
            root = production.zone_root / "build" / "Z16_root.usdc"
            root.parent.mkdir()
            root.write_bytes(b"usdc")
            editor = base / "FireViewer USD Composer" / "fireviewer_usd_composer.kit.bat"
            editor.parent.mkdir()
            editor.write_text("@echo off\n", encoding="utf-8")
            state = _load_state(workspace)
            zone_state = state["zones"]["Z16"]
            zone_state["phase"] = "scene_built"
            zone_state["history"].append(
                {
                    "phase": "scene_built",
                    "root_usd": "build/Z16_root.usdc",
                    "root_usd_sha256": _digest(root),
                }
            )
            write_json(workspace / "zone-scenes" / "production-state.json", state)
            process = type("Process", (), {"pid": 812})()
            with (
                patch.dict(
                    os.environ,
                    {"FW_SDG_REVIEW_EDITOR": str(editor)},
                    clear=False,
                ),
                patch("fireviewer_sdg.zone_scenes.sys.platform", "win32"),
                patch(
                    "fireviewer_sdg.zone_scenes.subprocess.Popen",
                    return_value=process,
                ) as launch,
            ):
                receipt = production.open_review()
            self.assertEqual(receipt["editor_kind"], "fireviewer_usd_composer_via_omniverse_hub")
            self.assertIn("pending", receipt["human_review"])
            self.assertTrue((production.zone_root / "review-launch.json").is_file())
            launch_args = launch.call_args.args
            launch_kwargs = launch.call_args.kwargs
            expected_command = "call " + subprocess.list2cmdline(
                [
                    str(editor),
                    "--no-ros-env",
                    "--exec",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "tools"
                        / "open-zone-scene-in-composer.py"
                    ),
                ]
            )
            self.assertEqual(launch_args[0], expected_command)
            self.assertTrue(launch_kwargs["shell"])
            recorded = _load_state(workspace)["zones"]["Z16"]
            self.assertEqual(recorded["phase"], "review_launch_requested")
            self.assertEqual(recorded["root_usd"], "build/Z16_root.usdc")
            with (
                patch.dict(
                    os.environ,
                    {"FW_SDG_REVIEW_EDITOR": str(editor)},
                    clear=False,
                ),
                patch("fireviewer_sdg.zone_scenes.sys.platform", "win32"),
                patch(
                    "fireviewer_sdg.zone_scenes._process_is_running",
                    return_value=True,
                ),
                patch(
                    "fireviewer_sdg.zone_scenes._review_process_matches",
                    return_value=True,
                ),
                patch("fireviewer_sdg.zone_scenes.subprocess.Popen") as relaunch,
            ):
                reused = production.open_review()
            self.assertTrue(reused["reused_existing_process"])
            self.assertEqual(reused["launcher_pid"], 812)
            relaunch.assert_not_called()

    def test_validated_scene_uses_argv_without_a_shell_on_linux(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            production = ZoneSceneProduction(
                catalog_root=catalog,
                workspace_root=workspace,
                zone_id="Z16",
            )
            production.preflight()
            root = production.zone_root / "build" / "Z16_root.usdc"
            root.parent.mkdir()
            root.write_bytes(b"usdc")
            editor = base / "kit" / "fireviewer_usd_composer.kit.sh"
            editor.parent.mkdir()
            editor.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            state = _load_state(workspace)
            zone = state["zones"]["Z16"]
            zone.update(
                {
                    "phase": "scene_built",
                    "root_usd": "build/Z16_root.usdc",
                    "root_usd_sha256": _digest(root),
                }
            )
            write_json(workspace / "zone-scenes" / "production-state.json", state)
            process = type("Process", (), {"pid": 913})()
            with (
                patch.dict(
                    os.environ,
                    {"FW_SDG_REVIEW_EDITOR": str(editor)},
                    clear=False,
                ),
                patch("fireviewer_sdg.zone_scenes.sys.platform", "linux"),
                patch(
                    "fireviewer_sdg.zone_scenes.subprocess.Popen",
                    return_value=process,
                ) as launch,
            ):
                receipt = production.open_review()
            command = launch.call_args.args[0]
            self.assertIsInstance(command, list)
            self.assertEqual(command[0], str(editor))
            self.assertFalse(launch.call_args.kwargs["shell"])
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.assertEqual(
                receipt["editor_kind"],
                "fireviewer_usd_composer_linux_x11",
            )

    def test_cleanup_refuses_missing_archive_and_only_targets_raw_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            production = ZoneSceneProduction(catalog_root=catalog, workspace_root=workspace, zone_id="Z16")
            production.preflight()
            raw_file = production.zone_root / "raw" / "mnt" / "tile.tif"
            raw_file.parent.mkdir(parents=True)
            raw_file.write_bytes(b"raw")
            unrelated = production.zone_root / "build" / "scene.usda"
            unrelated.parent.mkdir()
            unrelated.write_text("#usda 1.0", encoding="utf-8")
            state = _load_state(workspace)
            state["zones"]["Z16"].update(
                phase="archived",
                archive="zone-scenes/Z16/archive/absent.zip",
                archive_sha256="0" * 64,
            )
            write_json(workspace / "zone-scenes" / "production-state.json", state)
            with self.assertRaisesRegex(RuntimeError, "archive"):
                production.cleanup(confirmation="Z16")
            self.assertTrue(raw_file.exists())
            self.assertTrue(unrelated.exists())

    def test_full_build_receipt_requires_exactly_400_terrain_and_detail_payloads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            terrain_short = _write_full_build_receipt(
                zone_root, terrain_count=399
            )
            with self.assertRaisesRegex(
                ValueError, "exactly 400 one-kilometre payloads"
            ):
                _validate_build_receipt(
                    terrain_short, zone_root=zone_root, zone_id="Z16"
                )

        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            details_short = _write_full_build_receipt(
                zone_root, detail_count=399
            )
            with self.assertRaisesRegex(
                ValueError, "exactly 400 HERO detail payloads"
            ):
                _validate_build_receipt(
                    details_short, zone_root=zone_root, zone_id="Z16"
                )

    def test_full_build_receipt_accepts_complete_tiled_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            receipt_path = _write_full_build_receipt(zone_root)
            validated = _validate_build_receipt(
                receipt_path, zone_root=zone_root, zone_id="Z16"
            )
        self.assertEqual(len(validated["payloads"]), 400)
        self.assertEqual(len(validated["detail_payloads"]), 400)
        self.assertEqual(len(validated["detail_mid_payloads"]), 400)
        self.assertEqual(len(validated["detail_far_payloads"]), 400)
        self.assertEqual(validated["lidar_quality"]["path"], "build/lidar-evidence.json")

    def test_full_build_receipt_rejects_header_only_usd_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            zone_root = Path(directory) / "Z16"
            receipt_path = _write_full_build_receipt(
                zone_root, header_only_terrain=237
            )
            with self.assertRaisesRegex(
                ValueError, "payload contains no material USD prim"
            ):
                _validate_build_receipt(
                    receipt_path, zone_root=zone_root, zone_id="Z16"
                )

    @patch("fireviewer_sdg.zone_scenes._request_collection")
    def test_resolver_keeps_unpublished_tiles_explicitly_unresolved(self, request: object) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = _write_catalog(base / "catalog")
            workspace = base / "workspace"
            production = ZoneSceneProduction(catalog_root=catalog, workspace_root=workspace, zone_id="Z16")
            production.preflight()
            request.return_value = {"type": "FeatureCollection", "features": []}  # type: ignore[attr-defined]
            lock = production.resolve(timeout=1, retries=1)
        self.assertEqual(len(lock["entries"]), 2400)
        self.assertEqual(sum(item["resolution_status"] == "unresolved" for item in lock["entries"] if item["dataset"] == "mnt"), 400)


if __name__ == "__main__":
    unittest.main()
