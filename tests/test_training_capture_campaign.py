from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fireviewer_sdg import training_capture_campaign as capture  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _record(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _exr_attribute(name: str, type_name: str, value: bytes) -> bytes:
    return (
        name.encode("ascii")
        + b"\x00"
        + type_name.encode("ascii")
        + b"\x00"
        + struct.pack("<I", len(value))
        + value
    )


def _write_valid_exr(
    path: Path,
    *,
    width_px: int = 3840,
    height_px: int = 2160,
    modality: str = "normal_rgb",
    rgb_value: float = 0.25,
    thermal_minimum_k: float = 300.0,
    thermal_hotspot_k: float = 900.0,
    hotspot_pixel_xy: tuple[int, int] = (1920, 1080),
    geometry_id_offset: float = 0.0,
) -> None:
    if modality in {"normal_rgb", "negative"}:
        channel_names = ("B", "G", "R")
        pixel_type = 1
        sample_size = 2
    elif modality == "thermal_hotspot":
        channel_names = ("T",)
        pixel_type = 2
        sample_size = 4
    elif modality == "registration_geometry_id":
        channel_names = ("ID",)
        pixel_type = 2
        sample_size = 4
    else:
        raise AssertionError(f"unsupported test modality: {modality}")
    channels = bytearray()
    for channel in channel_names:
        channels.extend(channel.encode("ascii") + b"\x00")
        channels.extend(
            struct.pack("<iB3xii", pixel_type, 0, 1, 1)
        )
    channels.extend(b"\x00")
    window = struct.pack("<iiii", 0, 0, width_px - 1, height_px - 1)
    header = b"".join(
        (
            struct.pack("<II", 20_000_630, 2),
            _exr_attribute("channels", "chlist", bytes(channels)),
            _exr_attribute("compression", "compression", b"\x03"),
            _exr_attribute("dataWindow", "box2i", window),
            _exr_attribute("displayWindow", "box2i", window),
            _exr_attribute("lineOrder", "lineOrder", b"\x00"),
            _exr_attribute(
                "pixelAspectRatio",
                "float",
                struct.pack("<f", 1.0),
            ),
            _exr_attribute(
                "screenWindowCenter",
                "v2f",
                struct.pack("<ff", 0.0, 0.0),
            ),
            _exr_attribute(
                "screenWindowWidth",
                "float",
                struct.pack("<f", 1.0),
            ),
            (
                _exr_attribute(
                    "chromaticities",
                    "chromaticities",
                    struct.pack(
                        "<ffffffff",
                        0.713,
                        0.293,
                        0.165,
                        0.830,
                        0.128,
                        0.044,
                        0.32168,
                        0.33767,
                    ),
                )
                if modality in {"normal_rgb", "negative"}
                else b""
            ),
            b"\x00",
        )
    )
    chunks: list[bytes] = []
    for y_coordinate in range(0, height_px, 16):
        scanlines = min(16, height_px - y_coordinate)
        if modality in {"normal_rgb", "negative"}:
            raw = np.full(
                width_px * len(channel_names) * scanlines,
                rgb_value,
                dtype="<f2",
            ).tobytes()
        elif modality == "thermal_hotspot":
            values = np.full(
                (scanlines, width_px),
                thermal_minimum_k,
                dtype="<f4",
            )
            hotspot_x, hotspot_y = hotspot_pixel_xy
            if y_coordinate <= hotspot_y < y_coordinate + scanlines:
                values[hotspot_y - y_coordinate, hotspot_x] = (
                    thermal_hotspot_k
                )
            raw = values.tobytes()
        else:
            rows = np.arange(
                y_coordinate,
                y_coordinate + scanlines,
                dtype="<f4",
            )[:, None]
            columns = np.arange(width_px, dtype="<f4")[None, :]
            raw = (
                rows * float(width_px)
                + columns
                + geometry_id_offset
            ).astype("<f4").tobytes()
        assert len(raw) == width_px * len(channel_names) * sample_size * scanlines
        reordered = bytearray(raw[0::2] + raw[1::2])
        predicted = bytearray(len(reordered))
        predicted[0] = reordered[0]
        for index in range(1, len(reordered)):
            predicted[index] = (
                reordered[index] - reordered[index - 1] + 128
            ) & 0xFF
        compressed = zlib.compress(bytes(predicted), level=6)
        packed = compressed if len(compressed) < len(raw) else raw
        chunks.append(
            struct.pack("<iI", y_coordinate, len(packed)) + packed
        )
    first_chunk = len(header) + len(chunks) * 8
    offsets: list[int] = []
    cursor = first_chunk
    for chunk in chunks:
        offsets.append(cursor)
        cursor += len(chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        for chunk in chunks:
            stream.write(chunk)


def _camera(view_index: int) -> dict[str, object]:
    if view_index == 1:
        view_class = "top_down"
        position = [1_025.0, 1_015.0, 900.0]
    elif view_index == 2:
        view_class = "top_down"
        position = [1_040.0, 1_010.0, 1_200.0]
    elif view_index == 3:
        view_class = "plunging_oblique"
        position = [400.0, 1_000.0, 350.0]
    elif view_index == 4:
        view_class = "plunging_oblique"
        position = [1_650.0, 1_000.0, 450.0]
    elif view_index == 5:
        view_class = "satellite_high_altitude"
        position = [1_000.0, 1_000.0, 6_000.0]
    elif view_index == 6:
        view_class = "satellite_high_altitude"
        position = [2_500.0, 1_000.0, 7_000.0]
    else:
        slot = view_index - 7
        angle = math.tau * slot / 34.0
        if view_index <= 18:
            view_class = "ground_observer"
            radius, altitude = 650.0, 30.0
        elif view_index <= 29:
            view_class = "ridge_observer"
            radius, altitude = 800.0, 180.0
        else:
            view_class = "airborne_oblique"
            radius, altitude = 1_000.0, 500.0
        position = [
            1_030.0 + radius * math.cos(angle),
            1_015.0 + radius * math.sin(angle),
            altitude,
        ]
    # The source camera pose is authored once for the full temporal sequence.
    # The campaign must preserve this rotation while the fire evolves.
    anchor = [1_008.0, 1_001.4, 10.0]
    orientation = capture._look_at_quaternion_xyzw(
        position_m=position,
        target_m=anchor,
    )
    return {
        "camera_id": f"VIEW-{view_index:02d}",
        "view_class": view_class,
        "pose_local": {
            "position_m": position,
            "orientation_xyzw": orientation,
        },
        "intrinsics": {
            "model": "pinhole",
            "width_px": 320,
            "height_px": 180,
            "fx_px": 233.33333333333334,
            "fy_px": 233.33333333333334,
            "cx_px": 160.0,
            "cy_px": 90.0,
            "near_clip_m": 0.1,
            "far_clip_m": 30_000.0,
            "focal_length_mm": 26.25,
            "horizontal_aperture_mm": 36.0,
            "vertical_aperture_mm": 20.25,
            "f_stop": 8.0,
        },
    }


def _day(day_index: int) -> dict[str, object]:
    hours: list[dict[str, object]] = []
    for capture_hour in capture.CAPTURE_HOURS:
        hour = int(capture_hour[:2])
        burned = (day_index - 1) * 1_000.0 + float(hour)
        hours.append(
            {
                "capture_hour": capture_hour,
                "fire": {
                    "state": "active",
                    "simulation_time_seconds": (
                        (day_index - 1) * 86_400 + hour * 3_600
                    ),
                    "active_area_m2": burned * 0.25,
                    "active_flame_centroid_local_m": [
                        1_000.0 + day_index * 4.0 + hour * 0.5,
                        1_000.0 + day_index * 3.0 - hour * 0.2,
                        10.0,
                    ],
                    "active_flame_front_radius_m": 40.0 + burned * 0.001,
                    "burned_area_m2": burned,
                    "fire_front_length_m": 100.0 + burned * 0.02,
                    "max_flame_height_m": 4.0 + hour * 0.1,
                    "smoke_column_height_m": 250.0 + hour * 3.0,
                },
                "weather": {
                    "air_temperature_c": 18.0 + hour * 0.3,
                    "relative_humidity_percent": 52.0 - hour * 0.5,
                    "wind_speed_m_s": 3.0 + day_index * 0.1,
                    "wind_direction_degrees": float(
                        (110 + day_index + hour) % 360
                    ),
                    "precipitation_mm_h": 0.0,
                    "cloud_cover_fraction": 0.25,
                    "pressure_hpa": 1_015.0,
                },
            }
        )
    return {"day_index": day_index, "hours": hours}


def _fixture(
    root: Path,
    *,
    gate_state: str = capture.SIMULATION_ALLOWED_STATE,
) -> dict[str, Path]:
    volume = root.resolve()
    campaign_id = "fireviewer-omniverse-20-photoreal-simulations-v1"
    simulations: list[dict[str, object]] = []
    source_scenes: list[dict[str, object]] = []
    for scene_index, (scene_id, duration) in enumerate(
        zip(
            capture.SCENE_IDS,
            capture.SCENE_DURATIONS_DAYS,
            strict=True,
        ),
        start=1,
    ):
        scene_root = (
            volume
            / "variant-scenes"
            / scene_id
            / "build"
            / "root.usdc"
        )
        scene_root.parent.mkdir(parents=True)
        scene_root.write_bytes(f"native-usd:{scene_id}".encode())
        root_relative = scene_root.relative_to(volume).as_posix()
        simulations.append(
            {
                "simulation_id": scene_id,
                "scene_binding": {"root_usd": root_relative},
            }
        )
        source_scenes.append(
            {
                "scene_id": scene_id,
                "variant_id": f"VARIANT-{scene_index:02d}",
                "duration_days": duration,
                "scene_root": _record(scene_root, root=volume),
                "scene_origin_epsg2154_ign69_m": [
                    700_000.0 + scene_index * 25_000.0,
                    6_600_000.0 + scene_index * 25_000.0,
                    0.0,
                ],
                "cameras": [
                    _camera(view_index)
                    for view_index in range(
                        1,
                        capture.CAMERAS_PER_SCENE + 1,
                    )
                ],
                "days": [
                    _day(day_index)
                    for day_index in range(1, duration + 1)
                ],
            }
        )
    campaign = volume / "campaign-index.json"
    _write_json(
        campaign,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "simulations": simulations,
        },
    )
    gate_input_root = volume / "gate-inputs"
    runtime_preflight = gate_input_root / "runtime-preflight.json"
    asset_manifest = gate_input_root / "asset-manifest.json"
    build_receipt = gate_input_root / "build-receipt.json"
    scene_auto_validation = gate_input_root / "scene-auto-validation.json"
    _write_json(runtime_preflight, {"state": "RUNTIME_PREFLIGHT_PASSED"})
    _write_json(asset_manifest, {"state": "ASSETS_MATERIALIZED"})
    _write_json(build_receipt, {"state": "BUILD_COMPLETE"})
    _write_json(scene_auto_validation, {"state": "AUTO_VALIDATED"})
    georeference_root = gate_input_root / "georeference"
    for source_scene in source_scenes:
        scene_id = str(source_scene["scene_id"])
        scene_root = volume / source_scene["scene_root"]["path"]
        origin = source_scene["scene_origin_epsg2154_ign69_m"]
        provenance = georeference_root / f"{scene_id}.json"
        _write_json(
            provenance,
            {
                "schema_version": 1,
                "state": capture.GEOREFERENCE_AUTHENTICATED_STATE,
                "scene_id": scene_id,
                "horizontal_crs": "EPSG:2154",
                "vertical_datum": "IGN69",
                "axis_order": list(capture.GEOREFERENCE_AXIS_ORDER),
                "local_axes": list(capture.GEOREFERENCE_LOCAL_AXES),
                "axis_mapping": dict(
                    capture.GEOREFERENCE_AXIS_MAPPING
                ),
                "units": "metres",
                "origin_epsg2154_ign69_m": origin,
                "scene_root_sha256": _sha256(scene_root),
                "build_receipt_sha256": _sha256(build_receipt),
            },
        )
        source_scene["georeference"] = {
            "horizontal_crs": "EPSG:2154",
            "vertical_datum": "IGN69",
            "axis_order": list(capture.GEOREFERENCE_AXIS_ORDER),
            "local_axes": list(capture.GEOREFERENCE_LOCAL_AXES),
            "axis_mapping": dict(capture.GEOREFERENCE_AXIS_MAPPING),
            "units": "metres",
            "origin_epsg2154_ign69_m": origin,
            "scene_root_sha256": _sha256(scene_root),
            "build_receipt_sha256": _sha256(build_receipt),
            "provenance_receipt": _record(provenance, root=volume),
        }
    sim01_root = volume / simulations[0]["scene_binding"]["root_usd"]
    gate_bindings = {
        "runtime_preflight_sha256": _sha256(runtime_preflight),
        "campaign_index_sha256": _sha256(campaign),
        "asset_manifest_sha256": _sha256(asset_manifest),
        "asset_content_sha256": "1" * 64,
        "root_usd_sha256": _sha256(sim01_root),
        "build_receipt_sha256": _sha256(build_receipt),
        "scene_auto_validation_sha256": _sha256(scene_auto_validation),
        "build_artifact_content_sha256": "2" * 64,
        "scene_layer_content_sha256": "3" * 64,
    }
    pending_review = gate_input_root / "editor-review-pending.json"
    _write_json(
        pending_review,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "scene_id": "SIM-01",
            "status": "AWAITING_EDITOR_REVIEW",
            "human_review": "pending",
            "bindings": gate_bindings,
        },
    )
    editor_opened = gate_input_root / "editor-opened.json"
    _write_json(
        editor_opened,
        {
            "state": "opened_for_human_review",
            "human_review": "pending",
            "root_usd_sha256": _sha256(sim01_root),
            "pending_review_sha256": _sha256(pending_review),
        },
    )
    acceptance = gate_input_root / "editor-review-accepted.json"
    _write_json(
        acceptance,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "scene_id": "SIM-01",
            "zone_id": None,
            "decision": "accepted",
            "reviewer": "capture-contract-test",
            "reviewed_at": "2026-07-29T00:00:00Z",
            "status": "EDITOR_REVIEW_ACCEPTED",
            "bindings": gate_bindings,
            "pending_review_sha256": _sha256(pending_review),
            "editor_opened_sha256": _sha256(editor_opened),
            "acknowledgement": capture.EDITOR_REVIEW_ACKNOWLEDGEMENT,
        },
    )
    gate = volume / "fire-simulation-allowed.json"
    _write_json(
        gate,
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "state": gate_state,
            "acceptance_sha256": _sha256(acceptance),
            "bindings": gate_bindings,
        },
    )
    sources = volume / "training-capture-sources.json"
    _write_json(
        sources,
        {
            "schema_version": 1,
            "state": capture.SOURCE_STATE,
            "campaign_id": campaign_id,
            "campaign_index": _record(campaign, root=volume),
            "simulation_gate_inputs": {
                "pending_review": _record(pending_review, root=volume),
                "editor_opened": _record(editor_opened, root=volume),
                "editor_acceptance": _record(acceptance, root=volume),
                "runtime_preflight": _record(
                    runtime_preflight,
                    root=volume,
                ),
                "asset_manifest": _record(asset_manifest, root=volume),
                "root_usd": _record(sim01_root, root=volume),
                "build_receipt": _record(build_receipt, root=volume),
                "scene_auto_validation": _record(
                    scene_auto_validation,
                    root=volume,
                ),
            },
            "scenes": source_scenes,
        },
    )
    output = volume / "training-capture"
    return {
        "volume": volume,
        "campaign": campaign,
        "gate": gate,
        "pending_review": pending_review,
        "editor_opened": editor_opened,
        "acceptance": acceptance,
        "runtime_preflight": runtime_preflight,
        "asset_manifest": asset_manifest,
        "sim01_root": sim01_root,
        "build_receipt": build_receipt,
        "scene_auto_validation": scene_auto_validation,
        "sources": sources,
        "output": output,
        "plan": output / "capture-plan.json",
    }


def _prepare(fixture: dict[str, Path]) -> dict[str, object]:
    return capture.prepare_training_capture_campaign_plan(
        volume_root=fixture["volume"],
        simulation_allowed_receipt_path=fixture["gate"],
        source_manifest_path=fixture["sources"],
        output_root=fixture["output"],
        plan_path=fixture["plan"],
    )


def _shrink_fixture_cameras(
    fixture: dict[str, Path],
    *,
    width_px: int = 64,
    height_px: int = 32,
) -> None:
    source = json.loads(fixture["sources"].read_text(encoding="utf-8"))
    for scene in source["scenes"]:
        for camera in scene["cameras"]:
            intrinsics = camera["intrinsics"]
            intrinsics["width_px"] = width_px
            intrinsics["height_px"] = height_px
            intrinsics["fx_px"] = (
                intrinsics["focal_length_mm"]
                / intrinsics["horizontal_aperture_mm"]
                * width_px
            )
            intrinsics["fy_px"] = (
                intrinsics["focal_length_mm"]
                / intrinsics["vertical_aperture_mm"]
                * height_px
            )
            intrinsics["cx_px"] = width_px / 2.0
            intrinsics["cy_px"] = height_px / 2.0
    _write_json(fixture["sources"], source)


def _thermal_evidence(
    metadata: dict[str, object],
) -> dict[str, object]:
    hotspot_local = metadata["look_at"]["local"]["position_m"]
    hotspot_x, hotspot_y, hotspot_depth = (
        capture._project_local_point_to_pixel(
            point_local_m=hotspot_local,
            pose_local=metadata["pose"]["local"],
            intrinsics=metadata["intrinsics"],
        )
    )
    return {
        "render_source": (
            "fire_temperature_heat_field_plus_surface_emissivity"
        ),
        "temperature_unit": "kelvin",
        "heat_field_sha256": "a" * 64,
        "surface_emissivity": 0.95,
        "min_temperature_k": 300.0,
        "max_temperature_k": 900.0,
        "hotspot_temperature_k": 900.0,
        "hotspot_pixel_xy": [round(hotspot_x), round(hotspot_y)],
        "hotspot_local_m": hotspot_local,
        "hotspot_epsg2154_ign69_m": metadata["look_at"][
            "epsg2154_ign69"
        ]["position_m"],
        "depth_at_hotspot_m": hotspot_depth,
        "hotspot_in_active_front": True,
        "hotspot_in_fov": True,
        "registration_contract_sha256": metadata["pixel_registration"][
            "registration_contract_sha256"
        ],
    }


def _registration_evidence(
    *,
    fixture: dict[str, Path],
    task: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    session_id = f"render-{metadata['observation_id']}"
    registration = metadata["pixel_registration"][
        "registration_contract_sha256"
    ]
    projection = metadata["pixel_registration"][
        "camera_projection_contract_sha256"
    ]
    witness_paths = {
        modality: Path(path)
        for modality, path in task[
            "registration_witness_output_paths"
        ].items()
    }
    width = int(metadata["intrinsics"]["width_px"])
    height = int(metadata["intrinsics"]["height_px"])
    for path in witness_paths.values():
        _write_valid_exr(
            path,
            modality="registration_geometry_id",
            width_px=width,
            height_px=height,
        )
    products: dict[str, object] = {}
    for product_modality, witness_modality in {
        "normal_rgb": "normal_rgb_geometry_id",
        "thermal_hotspot": "thermal_hotspot_geometry_id",
    }.items():
        product_id = f"{metadata['observation_id']}:{product_modality}"
        witness = task["registration_witness_contracts"][
            witness_modality
        ]
        products[product_modality] = {
            "render_product_id": product_id,
            "source_image_artifact_id": task[
                "image_artifact_contracts"
            ][product_modality]["artifact_id"],
            "camera_projection_contract_sha256": projection,
            "geometry_id_aov": {
                **capture._file_record(
                    root=fixture["output"],
                    path=witness_paths[witness_modality],
                ),
                "witness_id": witness["witness_id"],
                "modality": witness_modality,
                "semantic": "nonuniform_geometry_id_registration_aov",
                "delivery_artifact": False,
                "render_session_id": session_id,
                "render_product_id": product_id,
                "registration_contract_sha256": registration,
                "camera_projection_contract_sha256": projection,
            },
        }
    return {
        "render_session_id": session_id,
        "registration_contract_sha256": registration,
        "camera_projection_contract_sha256": projection,
        "render_products": products,
        "pixel_identical_geometry_id": True,
    }


def _write_rendered_observation(
    *,
    fixture: dict[str, Path],
    plan: dict[str, object],
    observation_id: str,
) -> dict[str, object]:
    task = capture.prepare_training_capture_frame_task(
        volume_root=fixture["volume"],
        plan_path=fixture["plan"],
        simulation_allowed_receipt_path=fixture["gate"],
        observation_id=observation_id,
    )
    metadata = Path(task["metadata_output_path"])
    metadata.parent.mkdir(parents=True)
    rendered_metadata = dict(
        task["metadata_contract_without_images_or_thermal_evidence"]
    )
    thermal_evidence = _thermal_evidence(rendered_metadata)
    hotspot_pixel = tuple(thermal_evidence["hotspot_pixel_xy"])
    image_paths = {
        modality: Path(path)
        for modality, path in task["image_output_paths"].items()
    }
    width = int(rendered_metadata["intrinsics"]["width_px"])
    height = int(rendered_metadata["intrinsics"]["height_px"])
    _write_valid_exr(
        image_paths["normal_rgb"],
        modality="normal_rgb",
        rgb_value=0.25,
        width_px=width,
        height_px=height,
    )
    _write_valid_exr(
        image_paths["negative"],
        modality="negative",
        rgb_value=0.75,
        width_px=width,
        height_px=height,
    )
    _write_valid_exr(
        image_paths["thermal_hotspot"],
        modality="thermal_hotspot",
        hotspot_pixel_xy=hotspot_pixel,
        width_px=width,
        height_px=height,
    )
    registration = rendered_metadata["pixel_registration"][
        "registration_contract_sha256"
    ]
    rendered_metadata["images"] = {
        modality: capture._registered_image_record(
            output_root=fixture["output"],
            path=image_paths[modality],
            artifact=task["image_artifact_contracts"][modality],
            registration_contract_sha256=registration,
            expected_metadata=rendered_metadata,
        )
        for modality in capture.IMAGE_MODALITIES
    }
    rendered_metadata["thermal_evidence"] = thermal_evidence
    rendered_metadata["rendered_registration_evidence"] = (
        _registration_evidence(
            fixture=fixture,
            task=task,
            metadata=rendered_metadata,
        )
    )
    _write_json(metadata, rendered_metadata)
    receipt = capture.record_training_capture_observation_completion(
        volume_root=fixture["volume"],
        plan_path=fixture["plan"],
        simulation_allowed_receipt_path=fixture["gate"],
        observation_id=observation_id,
    )
    assert receipt["plan_sha256"] == plan["plan_sha256"]
    return receipt


class TrainingCaptureCampaignTests(unittest.TestCase):
    def test_plan_has_exact_schedule_metadata_hashes_and_no_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            observations = plan["observations"]
            task = capture.prepare_training_capture_frame_task(
                volume_root=fixture["volume"],
                plan_path=fixture["plan"],
                simulation_allowed_receipt_path=fixture["gate"],
                observation_id=observations[0]["observation_id"],
            )

            self.assertEqual(plan["scene_ids"], list(capture.SCENE_IDS))
            self.assertEqual(
                plan["scene_durations_days"],
                list(capture.SCENE_DURATIONS_DAYS),
            )
            self.assertEqual(sum(plan["scene_durations_days"]), 153)
            self.assertEqual(plan["camera_count_per_scene"], 40)
            self.assertEqual(
                plan["capture_hours"],
                ["08:00", "14:00", "20:00"],
            )
            self.assertEqual(len(observations), 18_360)
            self.assertEqual(
                plan["image_artifact_count"],
                capture.EXPECTED_IMAGE_ARTIFACT_COUNT,
            )
            self.assertEqual(
                plan["image_modalities"],
                list(capture.IMAGE_MODALITIES),
            )
            all_artifacts = [
                artifact
                for observation in observations
                for artifact in observation["image_artifacts"].values()
            ]
            self.assertEqual(len(all_artifacts), 55_080)
            self.assertEqual(
                len({artifact["artifact_id"] for artifact in all_artifacts}),
                55_080,
            )
            self.assertEqual(
                len({artifact["path"] for artifact in all_artifacts}),
                55_080,
            )
            all_registration_witnesses = [
                witness
                for observation in observations
                for witness in observation[
                    "registration_witnesses"
                ].values()
            ]
            self.assertEqual(
                len(all_registration_witnesses),
                36_720,
            )
            self.assertTrue(
                all(
                    witness["path"].startswith("registration/")
                    for witness in all_registration_witnesses
                )
            )
            self.assertEqual(
                len(
                    {
                        item["observation_id"]
                        for item in observations
                    }
                ),
                18_360,
            )
            self.assertEqual(
                [len(scene["cameras"]) for scene in plan["scenes"]],
                [40] * 20,
            )
            first_classes = [
                camera["view_class"]
                for camera in plan["scenes"][0]["cameras"]
            ]
            self.assertEqual(first_classes.count("top_down"), 2)
            self.assertEqual(first_classes.count("plunging_oblique"), 2)
            self.assertEqual(
                first_classes.count("satellite_high_altitude"),
                2,
            )
            self.assertEqual(
                len(
                    {
                        tuple(camera["position_local_m"])
                        for camera in plan["scenes"][0]["cameras"]
                    }
                ),
                40,
            )
            self.assertFalse(plan["simulation_execution_performed"])
            self.assertFalse(plan["render_execution_performed"])
            self.assertFalse((fixture["output"] / "frames").exists())
            self.assertFalse((fixture["output"] / "metadata").exists())
            self.assertFalse((fixture["output"] / "receipts").exists())
            self.assertEqual(task["state"], capture.FRAME_READY_STATE)
            metadata = task[
                "metadata_contract_without_images_or_thermal_evidence"
            ]
            self.assertIn("local", metadata["pose"])
            self.assertIn("epsg2154_ign69", metadata["pose"])
            self.assertIn("local", metadata["target"])
            self.assertIn("epsg2154_ign69", metadata["target"])
            self.assertEqual(
                metadata["target"]["semantic"],
                "active_flame_front_truth_reference",
            )
            self.assertEqual(
                metadata["target"]["camera_control"],
                "not_a_camera_reaim",
            )
            self.assertEqual(metadata["intrinsics"]["model"], "pinhole")
            self.assertEqual(
                metadata["camera_calibration"]["projection"],
                "pinhole_perspective",
            )
            self.assertEqual(
                metadata["pose"]["epsg2154_ign69"]["horizontal_crs"],
                "EPSG:2154",
            )
            self.assertEqual(
                metadata["pose"]["epsg2154_ign69"]["vertical_datum"],
                "IGN69",
            )
            self.assertEqual(
                metadata["georeference"]["local_axes"],
                ["east", "north", "up"],
            )
            self.assertEqual(
                metadata["georeference"]["axis_order"],
                ["easting_m", "northing_m", "altitude_m"],
            )
            self.assertEqual(
                metadata["georeference"]["axis_mapping"],
                {"X": "east", "Y": "north", "Z": "up"},
            )
            self.assertEqual(
                metadata["georeference"]["scene_root_sha256"],
                metadata["hashes"]["scene_root_sha256"],
            )
            self.assertEqual(
                metadata["georeference"]["build_receipt_sha256"],
                metadata["hashes"]["build_receipt_sha256"],
            )
            self.assertEqual(metadata["day_id"], "DAY-001")
            self.assertEqual(metadata["view_id"], "SIM-01-VIEW-01")
            self.assertEqual(metadata["variant_id"], "VARIANT-01")
            self.assertEqual(
                set(metadata["image_artifacts"]),
                set(capture.IMAGE_MODALITIES),
            )
            self.assertIn("fire", metadata)
            self.assertIn("weather", metadata)
            self.assertFalse(
                task["required_rendered_registration_evidence"][
                    "delivery_artifact"
                ]
            )
            self.assertEqual(
                set(metadata["hashes"]),
                {
                    "plan_sha256",
                    "observation_contract_sha256",
                    "simulation_allowed_receipt_sha256",
                    "source_manifest_sha256",
                    "campaign_index_sha256",
                    "pending_review_sha256",
                    "editor_opened_sha256",
                    "editor_acceptance_sha256",
                    "runtime_preflight_sha256",
                    "asset_manifest_sha256",
                    "build_receipt_sha256",
                    "scene_auto_validation_sha256",
                    "scene_root_sha256",
                    "camera_contract_sha256",
                    "fire_weather_sha256",
                    "aim_contract_sha256",
                    "intrinsics_sha256",
                    "registration_contract_sha256",
                },
            )
            local_x = metadata["pose"]["local"]["position_m"][0]
            origin_x = plan["scenes"][0][
                "scene_origin_epsg2154_ign69_m"
            ][0]
            absolute_x = metadata["pose"]["epsg2154_ign69"][
                "position_m"
            ][0]
            self.assertEqual(absolute_x, origin_x + local_x)
            local_target = metadata["target"]["local"]["position_m"]
            absolute_target = metadata["target"]["epsg2154_ign69"][
                "position_m"
            ]
            self.assertEqual(absolute_target[0], origin_x + local_target[0])
            camera_to_target = capture._unit(
                capture._vector_subtract(
                    local_target,
                    metadata["pose"]["local"]["position_m"],
                ),
                label="test target direction",
            )
            optical_forward = capture._unit(
                capture._rotate_by_quaternion(
                    [0.0, 0.0, -1.0],
                    metadata["pose"]["local"]["orientation_xyzw"],
                ),
                label="test optical forward",
            )
            self.assertGreater(
                capture._dot(camera_to_target, optical_forward),
                1.0 - 1.0e-9,
            )
            first_instant = observations[:40]
            first_radius = plan["scenes"][0]["days"][0]["hours"][0][
                "fire"
            ]["active_flame_front_radius_m"]
            maximum_target_separation = max(
                capture._length(
                    capture._vector_subtract(
                        left["target_local_m"],
                        right["target_local_m"],
                    )
                )
                for left_index, left in enumerate(first_instant)
                for right in first_instant[left_index + 1 :]
            )
            self.assertLessEqual(
                maximum_target_separation,
                first_radius
                * capture.TARGET_GROUP_DIAMETER_RADIUS_FRACTION,
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "regular paths below",
            ):
                capture.prepare_training_capture_campaign_plan(
                    volume_root=fixture["volume"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    source_manifest_path=fixture["sources"],
                    output_root=fixture["output"],
                    plan_path=fixture["output"] / "frames" / "plan.json",
                )

    def test_refuses_plan_or_frame_without_current_allowed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(
                Path(directory),
                gate_state="AWAITING_EDITOR_REVIEW",
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "blocked without a current FIRE_SIMULATION_ALLOWED",
            ):
                _prepare(fixture)

        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            gate = json.loads(fixture["gate"].read_text(encoding="utf-8"))
            gate["state"] = "AWAITING_EDITOR_REVIEW"
            _write_json(fixture["gate"], gate)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "blocked without a current FIRE_SIMULATION_ALLOWED",
            ):
                capture.prepare_training_capture_frame_task(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=plan["observations"][0][
                        "observation_id"
                    ],
                )

    def test_rehashes_every_direct_simulation_gate_artifact(self) -> None:
        artifact_keys = (
            "pending_review",
            "editor_opened",
            "acceptance",
            "runtime_preflight",
            "asset_manifest",
            "sim01_root",
            "build_receipt",
            "scene_auto_validation",
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            for artifact_key in artifact_keys:
                with self.subTest(artifact=artifact_key):
                    original = fixture[artifact_key].read_bytes()
                    fixture[artifact_key].write_bytes(
                        f"tampered:{artifact_key}".encode()
                    )
                    try:
                        with self.assertRaisesRegex(
                            capture.TrainingCaptureContractError,
                            "SHA-256 or size lock drifted",
                        ):
                            capture.prepare_training_capture_frame_task(
                                volume_root=fixture["volume"],
                                plan_path=fixture["plan"],
                                simulation_allowed_receipt_path=fixture["gate"],
                                observation_id=plan["observations"][0][
                                    "observation_id"
                                ],
                            )
                    finally:
                        fixture[artifact_key].write_bytes(original)

    def test_refuses_a_fabricated_editor_acceptance_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            acceptance = json.loads(
                fixture["acceptance"].read_text(encoding="utf-8")
            )
            acceptance["decision"] = "not_reviewed"
            _write_json(fixture["acceptance"], acceptance)
            sources = json.loads(
                fixture["sources"].read_text(encoding="utf-8")
            )
            sources["simulation_gate_inputs"]["editor_acceptance"] = _record(
                fixture["acceptance"],
                root=fixture["volume"],
            )
            _write_json(fixture["sources"], sources)
            gate = json.loads(fixture["gate"].read_text(encoding="utf-8"))
            gate["acceptance_sha256"] = _sha256(fixture["acceptance"])
            _write_json(fixture["gate"], gate)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "accepted SIM-01 Editor review",
            ):
                _prepare(fixture)

    def test_rejects_duration_camera_and_weather_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            original = fixture["sources"].read_bytes()

            sources = json.loads(original)
            sources["scenes"][0]["duration_days"] = 5
            _write_json(fixture["sources"], sources)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "duration or identity",
            ):
                _prepare(fixture)

            fixture["sources"].write_bytes(original)
            sources = json.loads(original)
            sources["scenes"][0]["cameras"].pop()
            _write_json(fixture["sources"], sources)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "exactly 40 fixed cameras",
            ):
                _prepare(fixture)

            fixture["sources"].write_bytes(original)
            sources = json.loads(original)
            sources["scenes"][0]["days"][0]["hours"][0]["weather"][
                "relative_humidity_percent"
            ] = 101.0
            _write_json(fixture["sources"], sources)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "physically invalid",
            ):
                _prepare(fixture)

    def test_requires_special_view_classes_and_their_real_geometry(
        self,
    ) -> None:
        mutations = (
            (
                "missing top-down minimum",
                0,
                lambda camera: camera.update(
                    {"view_class": "ground_observer"}
                ),
                "requires at least 2 top_down",
            ),
            (
                "invalid top-down geometry",
                0,
                lambda camera: camera["pose_local"].update(
                    {"position_m": [300.0, 1_000.0, 200.0]}
                ),
                "top-down geometry",
            ),
            (
                "invalid oblique geometry",
                2,
                lambda camera: camera["pose_local"].update(
                    {"position_m": [1_010.0, 1_000.0, 30.0]}
                ),
                "plunging-oblique",
            ),
            (
                "invalid satellite altitude",
                4,
                lambda camera: camera["pose_local"].update(
                    {"position_m": [1_000.0, 1_000.0, 1_000.0]}
                ),
                "satellite camera",
            ),
        )
        for label, camera_index, mutation, message in mutations:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = _fixture(Path(directory))
                    source = json.loads(
                        fixture["sources"].read_text(encoding="utf-8")
                    )
                    mutation(
                        source["scenes"][0]["cameras"][camera_index]
                    )
                    _write_json(fixture["sources"], source)
                    with self.assertRaisesRegex(
                        capture.TrainingCaptureContractError,
                        message,
                    ):
                        _prepare(fixture)

    def test_rejects_incoherent_distant_or_out_of_frame_active_front(
        self,
    ) -> None:
        mutations = (
            (
                "radius cannot contain active area",
                lambda source: source["scenes"][0]["days"][0]["hours"][
                    0
                ]["fire"].update(
                    {"active_flame_front_radius_m": 0.1}
                ),
                "fire metrics are incoherent",
            ),
            (
                "front beyond far clip",
                lambda source: source["scenes"][0]["cameras"][6][
                    "intrinsics"
                ].update({"far_clip_m": 100.0}),
                "outside the camera clip range",
            ),
            (
                "front outside narrowed FOV",
                lambda source: source["scenes"][0]["cameras"][6][
                    "intrinsics"
                ].update(
                    {
                        "fx_px": 50_000.0,
                        "fy_px": 50_000.0,
                        "focal_length_mm": 5_625.0,
                    }
                ),
                "does not fit in the FOV",
            ),
        )
        for label, mutation, message in mutations:
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = _fixture(Path(directory))
                    source = json.loads(
                        fixture["sources"].read_text(encoding="utf-8")
                    )
                    mutation(source)
                    _write_json(fixture["sources"], source)
                    with self.assertRaisesRegex(
                        capture.TrainingCaptureContractError,
                        message,
                    ):
                        _prepare(fixture)

    def test_each_instant_keeps_the_complete_camera_pose_fixed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            first = plan["observations"][0]
            six_hours_later = plan["observations"][40]
            next_day = plan["observations"][120]
            self.assertEqual(
                first["pose_local"]["position_m"],
                six_hours_later["pose_local"]["position_m"],
            )
            self.assertEqual(
                first["pose_local"]["position_m"],
                next_day["pose_local"]["position_m"],
            )
            self.assertEqual(
                first["intrinsics_sha256"],
                six_hours_later["intrinsics_sha256"],
            )
            self.assertEqual(
                first["intrinsics_sha256"],
                next_day["intrinsics_sha256"],
            )
            self.assertEqual(
                first["camera_contract_sha256"],
                next_day["camera_contract_sha256"],
            )
            self.assertEqual(
                first["pose_local"]["orientation_xyzw"],
                six_hours_later["pose_local"]["orientation_xyzw"],
            )
            self.assertEqual(
                first["pose_local"]["orientation_xyzw"],
                next_day["pose_local"]["orientation_xyzw"],
            )
            self.assertNotEqual(
                first["target_local_m"],
                six_hours_later["target_local_m"],
            )
            self.assertEqual(
                first["target_offset_from_active_centroid_m"],
                0.0,
            )

    def test_rejects_camera_targets_that_do_not_share_one_active_front(
        self,
    ) -> None:
        aims = [
            {"target_local_m": [0.0, 0.0, 0.0]}
            for _index in range(capture.CAMERAS_PER_SCENE)
        ]
        aims[-1] = {"target_local_m": [18.0, 0.0, 0.0]}
        with self.assertRaisesRegex(
            capture.TrainingCaptureContractError,
            "do not converge",
        ):
            capture._validate_convergent_aims(
                aims=aims,
                active_front_radius_m=100.0,
                label="test instant",
            )

    def test_negative_is_pixelwise_derived_and_thermal_is_not_rgb(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normal_path = root / "normal.exr"
            negative_path = root / "negative.exr"
            fake_thermal_path = root / "fake-thermal.exr"
            _write_valid_exr(
                normal_path,
                width_px=32,
                height_px=16,
                modality="normal_rgb",
                rgb_value=0.25,
            )
            _write_valid_exr(
                negative_path,
                width_px=32,
                height_px=16,
                modality="negative",
                rgb_value=0.25,
            )
            _write_valid_exr(
                fake_thermal_path,
                width_px=32,
                height_px=16,
                modality="normal_rgb",
                rgb_value=0.5,
            )
            normal = capture._validate_openexr_capture(
                path=normal_path,
                width_px=32,
                height_px=16,
                modality="normal_rgb",
                collect_pixels=True,
            )
            negative = capture._validate_openexr_capture(
                path=negative_path,
                width_px=32,
                height_px=16,
                modality="negative",
                collect_pixels=True,
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "deterministic transform",
            ):
                capture._validate_negative_pixels(
                    normal_capture=normal,
                    negative_capture=negative,
                )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "temperature channel T",
            ):
                capture._validate_openexr_capture(
                    path=fake_thermal_path,
                    width_px=32,
                    height_px=16,
                    modality="thermal_hotspot",
                )

    def test_thermal_hotspot_must_stay_inside_front_and_registered_fov(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            observation = plan["observations"][0]
            metadata = capture._metadata_contract(
                plan=plan,
                observation=observation,
            )
            evidence = _thermal_evidence(metadata)
            centroid = metadata["fire"][
                "active_flame_centroid_local_m"
            ]
            radius = metadata["fire"]["active_flame_front_radius_m"]
            evidence["hotspot_local_m"] = [
                centroid[0] + radius * 1.1,
                centroid[1],
                centroid[2],
            ]
            origin = capture._vector_subtract(
                metadata["pose"]["epsg2154_ign69"]["position_m"],
                metadata["pose"]["local"]["position_m"],
            )
            evidence["hotspot_epsg2154_ign69_m"] = (
                capture._vector_add(origin, evidence["hotspot_local_m"])
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "outside the active flame front",
            ):
                capture._validate_thermal_evidence(
                    expected_metadata=metadata,
                    evidence=evidence,
                    thermal_capture={"decoded_chunks": []},
                )

    def test_records_one_frame_resumes_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            observation_id = plan["observations"][0]["observation_id"]
            receipt = _write_rendered_observation(
                fixture=fixture,
                plan=plan,
                observation_id=observation_id,
            )
            progress = capture.inspect_training_capture_resume(
                volume_root=fixture["volume"],
                plan_path=fixture["plan"],
                simulation_allowed_receipt_path=fixture["gate"],
            )
            self.assertEqual(receipt["state"], capture.FRAME_COMPLETE_STATE)
            self.assertEqual(receipt["image_artifact_count"], 3)
            self.assertEqual(
                set(receipt["images"]),
                set(capture.IMAGE_MODALITIES),
            )
            required_registration_fields = {
                "scene_id",
                "variant_id",
                "day_id",
                "day_index",
                "view_id",
                "view_index",
                "camera_id",
                "observation_id",
                "timestamp",
                "capture_hour",
                "fire_state",
                "active_flame_centroid_local_m",
                "active_flame_front_radius_m",
                "fire",
                "weather",
                "camera_pose_local",
                "camera_pose_epsg2154_ign69",
                "look_at_local",
                "look_at_epsg2154_ign69",
                "intrinsics",
                "camera_calibration",
                "modality",
            }
            shared_registration: dict[str, object] | None = None
            for modality, image_record in receipt["images"].items():
                self.assertEqual(image_record["modality"], modality)
                self.assertEqual(
                    set(image_record["registration"]),
                    required_registration_fields,
                )
                self.assertEqual(
                    image_record["registration"]["modality"],
                    modality,
                )
                common = {
                    key: value
                    for key, value in image_record["registration"].items()
                    if key != "modality"
                }
                if shared_registration is None:
                    shared_registration = common
                else:
                    self.assertEqual(common, shared_registration)
            self.assertEqual(progress["completed_observations"], 1)
            self.assertEqual(progress["completed_image_artifacts"], 3)
            self.assertEqual(progress["total_image_artifacts"], 55_080)
            self.assertEqual(progress["pending_observations"], 18_359)
            self.assertEqual(progress["partial_observations"], 0)
            self.assertEqual(
                progress["next_observation_id"],
                plan["observations"][1]["observation_id"],
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "requires exactly 18,360 complete",
            ):
                capture.verify_training_capture_campaign(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                )
            image = (
                fixture["output"]
                / plan["observations"][0]["image_artifacts"]["normal_rgb"][
                    "path"
                ]
            )
            image.write_bytes(b"tampered-after-receipt")
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "OpenEXR",
            ):
                capture.inspect_training_capture_resume(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                )

    def test_partial_frame_is_reported_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            observation_id = plan["observations"][0]["observation_id"]
            task = capture.prepare_training_capture_frame_task(
                volume_root=fixture["volume"],
                plan_path=fixture["plan"],
                simulation_allowed_receipt_path=fixture["gate"],
                observation_id=observation_id,
            )
            image = Path(task["image_output_paths"]["normal_rgb"])
            image.parent.mkdir(parents=True)
            image.write_bytes(b"unreceipted-partial")
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "overwrite is forbidden",
            ):
                capture.prepare_training_capture_frame_task(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=observation_id,
                )
            progress = capture.inspect_training_capture_resume(
                volume_root=fixture["volume"],
                plan_path=fixture["plan"],
                simulation_allowed_receipt_path=fixture["gate"],
            )
            self.assertEqual(progress["partial_observations"], 1)
            self.assertEqual(
                progress["partial_observation_ids"],
                [observation_id],
            )
            self.assertIsNone(progress["next_observation_id"])

    def test_refuses_non_exr_and_wrong_resolution_frame_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            observation_id = plan["observations"][0]["observation_id"]
            task = capture.prepare_training_capture_frame_task(
                volume_root=fixture["volume"],
                plan_path=fixture["plan"],
                simulation_allowed_receipt_path=fixture["gate"],
                observation_id=observation_id,
            )
            _write_rendered_observation(
                fixture=fixture,
                plan=plan,
                observation_id=observation_id,
            )
            Path(task["completion_receipt_path"]).unlink()
            image = Path(task["image_output_paths"]["normal_rgb"])
            metadata = Path(task["metadata_output_path"])
            image.write_bytes(b"not-an-openexr")
            rendered_metadata = json.loads(
                metadata.read_text(encoding="utf-8")
            )
            rendered_metadata["images"]["normal_rgb"] = (
                capture._registered_image_record(
                    output_root=fixture["output"],
                    path=image,
                    artifact=task["image_artifact_contracts"][
                        "normal_rgb"
                    ],
                    registration_contract_sha256=rendered_metadata[
                        "pixel_registration"
                    ]["registration_contract_sha256"],
                    expected_metadata={
                        key: value
                        for key, value in rendered_metadata.items()
                        if key not in {"images", "thermal_evidence"}
                    },
                )
            )
            _write_json(metadata, rendered_metadata)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "OpenEXR",
            ):
                capture.record_training_capture_observation_completion(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=observation_id,
                )

            _write_valid_exr(
                image,
                width_px=160,
                height_px=90,
                modality="normal_rgb",
                rgb_value=0.25,
            )
            rendered_metadata["images"]["normal_rgb"] = (
                capture._registered_image_record(
                    output_root=fixture["output"],
                    path=image,
                    artifact=task["image_artifact_contracts"][
                        "normal_rgb"
                    ],
                    registration_contract_sha256=rendered_metadata[
                        "pixel_registration"
                    ]["registration_contract_sha256"],
                    expected_metadata={
                        key: value
                        for key, value in rendered_metadata.items()
                        if key not in {"images", "thermal_evidence"}
                    },
                )
            )
            _write_json(metadata, rendered_metadata)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "dimensions differ",
            ):
                capture.record_training_capture_observation_completion(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=observation_id,
                )

    def test_authenticates_lambert93_ign69_origin_and_provenance(
        self,
    ) -> None:
        mutations = ("zero_origin", "wrong_axes", "tampered_receipt")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    fixture = _fixture(Path(directory))
                    source = json.loads(
                        fixture["sources"].read_text(encoding="utf-8")
                    )
                    scene = source["scenes"][0]
                    georeference = scene["georeference"]
                    provenance_path = fixture["volume"] / (
                        georeference["provenance_receipt"]["path"]
                    )
                    if mutation == "zero_origin":
                        scene["scene_origin_epsg2154_ign69_m"] = [
                            0.0,
                            0.0,
                            0.0,
                        ]
                        georeference["origin_epsg2154_ign69_m"] = [
                            0.0,
                            0.0,
                            0.0,
                        ]
                        provenance = json.loads(
                            provenance_path.read_text(encoding="utf-8")
                        )
                        provenance["origin_epsg2154_ign69_m"] = [
                            0.0,
                            0.0,
                            0.0,
                        ]
                        _write_json(provenance_path, provenance)
                        georeference["provenance_receipt"] = _record(
                            provenance_path,
                            root=fixture["volume"],
                        )
                        expected = "outside the French Lambert-93"
                    elif mutation == "wrong_axes":
                        georeference["local_axes"] = [
                            "north",
                            "east",
                            "down",
                        ]
                        expected = "not bound to its root and build receipt"
                    else:
                        provenance_path.write_text(
                            provenance_path.read_text(encoding="utf-8")
                            + " ",
                            encoding="utf-8",
                        )
                        expected = "SHA-256 or size lock drifted"
                    _write_json(fixture["sources"], source)
                    with self.assertRaisesRegex(
                        capture.TrainingCaptureContractError,
                        expected,
                    ):
                        _prepare(fixture)

    def test_frame_and_record_use_primary_key_task_index_without_rebuild(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            observation_id = plan["observations"][777][
                "observation_id"
            ]
            index_path = fixture["plan"].with_name(
                f"{fixture['plan'].stem}.tasks.sqlite3"
            )
            self.assertTrue(index_path.is_file())
            with (
                mock.patch.object(
                    capture,
                    "_validated_plan",
                    side_effect=AssertionError("full plan rebuilt"),
                ),
                mock.patch.object(
                    capture,
                    "_build_plan_payload",
                    side_effect=AssertionError("observations rebuilt"),
                ),
                mock.patch.object(
                    capture,
                    "_observation_lookup",
                    side_effect=AssertionError("linear lookup used"),
                ),
            ):
                task = capture.prepare_training_capture_frame_task(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=observation_id,
                )
                self.assertEqual(
                    task["observation"]["observation_id"],
                    observation_id,
                )
                with self.assertRaisesRegex(
                    capture.TrainingCaptureContractError,
                    "capture metadata",
                ):
                    capture.record_training_capture_observation_completion(
                        volume_root=fixture["volume"],
                        plan_path=fixture["plan"],
                        simulation_allowed_receipt_path=fixture["gate"],
                        observation_id=observation_id,
                    )
            fixture["plan"].write_text(
                fixture["plan"].read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "not bound to the current plan file",
            ):
                capture.prepare_training_capture_frame_task(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=observation_id,
                )

    def test_rejects_thermal_from_another_camera_via_geometry_id_aov(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            _shrink_fixture_cameras(fixture)
            plan = _prepare(fixture)
            observation_id = plan["observations"][0]["observation_id"]
            task = capture.prepare_training_capture_frame_task(
                volume_root=fixture["volume"],
                plan_path=fixture["plan"],
                simulation_allowed_receipt_path=fixture["gate"],
                observation_id=observation_id,
            )
            _write_rendered_observation(
                fixture=fixture,
                plan=plan,
                observation_id=observation_id,
            )
            Path(task["completion_receipt_path"]).unlink()
            thermal_aov = Path(
                task["registration_witness_output_paths"][
                    "thermal_hotspot_geometry_id"
                ]
            )
            _write_valid_exr(
                thermal_aov,
                width_px=64,
                height_px=32,
                modality="registration_geometry_id",
                geometry_id_offset=100_000.0,
            )
            metadata_path = Path(task["metadata_output_path"])
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            aov_record = metadata[
                "rendered_registration_evidence"
            ]["render_products"]["thermal_hotspot"]["geometry_id_aov"]
            aov_record.update(
                capture._file_record(
                    root=fixture["output"],
                    path=thermal_aov,
                )
            )
            _write_json(metadata_path, metadata)
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "geometry-ID AOV pixels are not identical",
            ):
                capture.record_training_capture_observation_completion(
                    volume_root=fixture["volume"],
                    plan_path=fixture["plan"],
                    simulation_allowed_receipt_path=fixture["gate"],
                    observation_id=observation_id,
                )

    def test_exact_completion_inventory_requires_all_18360_frames(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _fixture(Path(directory))
            plan = _prepare(fixture)
            gate_sha = plan["source_locks"][
                "simulation_allowed_receipt"
            ]["sha256"]
            receipts = [
                {
                    "state": capture.FRAME_COMPLETE_STATE,
                    "observation_id": observation["observation_id"],
                    "plan_sha256": plan["plan_sha256"],
                    "observation_contract_sha256": observation[
                        "observation_contract_sha256"
                    ],
                    "simulation_allowed_receipt_sha256": gate_sha,
                    "image_artifact_count": 3,
                    "images": {
                        modality: {
                            "artifact_id": artifact["artifact_id"],
                            "modality": modality,
                            "registration_contract_sha256": observation[
                                "registration_contract_sha256"
                            ],
                        }
                        for modality, artifact in observation[
                            "image_artifacts"
                        ].items()
                    },
                }
                for observation in plan["observations"]
            ]
            capture._validate_completion_inventory(
                plan=plan,
                receipts=receipts,
                require_all=True,
            )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "exactly 18,360",
            ):
                capture._validate_completion_inventory(
                    plan=plan,
                    receipts=receipts[:-1],
                    require_all=True,
                )
            with self.assertRaisesRegex(
                capture.TrainingCaptureContractError,
                "duplicated, unknown or stale",
            ):
                capture._validate_completion_inventory(
                    plan=plan,
                    receipts=[*receipts[:-1], receipts[0]],
                    require_all=True,
                )


if __name__ == "__main__":
    unittest.main()
