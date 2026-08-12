"""Deterministic post-review capture contract for the 20-scene campaign.

This module never starts Kit, advances a simulation, or renders an image.  It
only prepares and verifies immutable co-registered image-triplet contracts
after a current
``FIRE_SIMULATION_ALLOWED`` receipt exists.  Every observation is independently
addressable and independently receipted so an interrupted campaign resumes at
the first unverified frame without trusting directory presence alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import uuid
import zlib
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SIMULATION_ALLOWED_STATE = "FIRE_SIMULATION_ALLOWED"
SOURCE_STATE = "TRAINING_CAPTURE_SOURCES_LOCKED"
PLAN_STATE = "TRAINING_CAPTURE_CAMPAIGN_PLANNED"
FRAME_READY_STATE = "TRAINING_CAPTURE_FRAME_READY"
FRAME_RENDERED_STATE = "TRAINING_CAPTURE_OBSERVATION_RENDERED"
FRAME_COMPLETE_STATE = "TRAINING_CAPTURE_OBSERVATION_COMPLETE"
TASK_INDEX_STATE = "TRAINING_CAPTURE_TASK_INDEX_MATERIALIZED"
CAMPAIGN_VERIFIED_STATE = "TRAINING_CAPTURE_CAMPAIGN_VERIFIED"
SCENE_IDS = tuple(f"SIM-{index:02d}" for index in range(1, 21))
SCENE_DURATIONS_DAYS = (
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    4,
    5,
)
CAMERAS_PER_SCENE = 40
CAPTURE_HOURS = ("08:00", "14:00", "20:00")
EXPECTED_OBSERVATION_COUNT = 18_360
IMAGE_MODALITIES = ("normal_rgb", "negative", "thermal_hotspot")
REGISTRATION_WITNESS_MODALITIES = (
    "normal_rgb_geometry_id",
    "thermal_hotspot_geometry_id",
)
EXPECTED_IMAGE_ARTIFACT_COUNT = (
    EXPECTED_OBSERVATION_COUNT * len(IMAGE_MODALITIES)
)
CAPTURE_FORMAT = {
    "container": "OpenEXR",
    "extension": ".exr",
    "channels": ["R", "G", "B"],
    "pixel_type": "float16",
    "color_space": "lin_ap1_scene",
    "compression": "ZIP",
}
CAPTURE_FORMATS = {
    "normal_rgb": {
        **CAPTURE_FORMAT,
        "semantic": "final_linear_rgb",
    },
    "negative": {
        **CAPTURE_FORMAT,
        "semantic": "deterministic_negative_of_final_linear_rgb",
        "derivation": "one_minus_clamped_linear_ap1_v1",
    },
    "thermal_hotspot": {
        "container": "OpenEXR",
        "extension": ".exr",
        "channels": ["T"],
        "pixel_type": "float32",
        "color_space": "temperature_kelvin_linear",
        "compression": "ZIP",
        "semantic": "fire_heat_field_temperature_emissivity",
    },
}
REGISTRATION_WITNESS_FORMAT = {
    "container": "OpenEXR",
    "extension": ".exr",
    "channels": ["ID"],
    "pixel_type": "float32",
    "semantic": "nonuniform_geometry_id_registration_aov",
    "delivery_artifact": False,
}
COORDINATE_CONTRACT = (
    "root_local_z_up_metres_plus_epsg2154_xy_ign69_z"
)
GEOREFERENCE_AUTHENTICATED_STATE = "SCENE_GEOREFERENCE_AUTHENTICATED"
LAMBERT93_EASTING_RANGE_M = (0.0, 1_300_000.0)
LAMBERT93_NORTHING_RANGE_M = (6_000_000.0, 7_200_000.0)
IGN69_ALTITUDE_RANGE_M = (-100.0, 5_000.0)
GEOREFERENCE_AXIS_ORDER = (
    "easting_m",
    "northing_m",
    "altitude_m",
)
GEOREFERENCE_LOCAL_AXES = ("east", "north", "up")
GEOREFERENCE_AXIS_MAPPING = {"X": "east", "Y": "north", "Z": "up"}
EDITOR_REVIEW_ACKNOWLEDGEMENT = (
    "I inspected the scene in FireViewer USD Composer"
)
GATE_ARTIFACT_KEYS = (
    "pending_review",
    "editor_opened",
    "editor_acceptance",
    "runtime_preflight",
    "asset_manifest",
    "root_usd",
    "build_receipt",
    "scene_auto_validation",
)
GATE_BINDING_KEYS = frozenset(
    {
        "runtime_preflight_sha256",
        "campaign_index_sha256",
        "asset_manifest_sha256",
        "asset_content_sha256",
        "root_usd_sha256",
        "build_receipt_sha256",
        "scene_auto_validation_sha256",
        "build_artifact_content_sha256",
        "scene_layer_content_sha256",
    }
)
_DIRECT_GATE_BINDINGS = {
    "runtime_preflight_sha256": "runtime_preflight",
    "asset_manifest_sha256": "asset_manifest",
    "build_receipt_sha256": "build_receipt",
    "scene_auto_validation_sha256": "scene_auto_validation",
}
_OPENEXR_MAGIC = 20_000_630
_OPENEXR_VERSION = 2
_OPENEXR_ZIP_COMPRESSION = 3
_OPENEXR_HALF = 1
_OPENEXR_FLOAT = 2
_OPENEXR_ZIP_LINES_PER_CHUNK = 16
_OPENEXR_AP1_CHROMATICITIES = (
    0.713,
    0.293,
    0.165,
    0.830,
    0.128,
    0.044,
    0.32168,
    0.33767,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_FIRE_FIELDS = frozenset(
    {
        "state",
        "simulation_time_seconds",
        "active_area_m2",
        "burned_area_m2",
        "fire_front_length_m",
        "max_flame_height_m",
        "smoke_column_height_m",
        "active_flame_centroid_local_m",
        "active_flame_front_radius_m",
    }
)
_WEATHER_FIELDS = frozenset(
    {
        "air_temperature_c",
        "relative_humidity_percent",
        "wind_speed_m_s",
        "wind_direction_degrees",
        "precipitation_mm_h",
        "cloud_cover_fraction",
        "pressure_hpa",
    }
)
_INTRINSIC_FIELDS = frozenset(
    {
        "model",
        "width_px",
        "height_px",
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "near_clip_m",
        "far_clip_m",
        "focal_length_mm",
        "horizontal_aperture_mm",
        "vertical_aperture_mm",
        "f_stop",
    }
)
VIEW_CLASSES = (
    "ground_observer",
    "ridge_observer",
    "airborne_oblique",
    "top_down",
    "plunging_oblique",
    "satellite_high_altitude",
)
SPECIAL_VIEW_CLASS_MINIMUMS = {
    "top_down": 2,
    "plunging_oblique": 2,
    "satellite_high_altitude": 2,
}
NON_SPECIAL_VIEW_CLASSES = frozenset(
    {"ground_observer", "ridge_observer", "airborne_oblique"}
)
TARGET_MAX_OFFSET_RADIUS_FRACTION = 0.08
TARGET_MAX_VERTICAL_OFFSET_RADIUS_FRACTION = 0.025
TARGET_GROUP_DIAMETER_RADIUS_FRACTION = 0.17
FOV_USABLE_HALF_ANGLE_FRACTION = 0.90
TOP_DOWN_MAX_NADIR_ANGLE_DEGREES = 12.0
TOP_DOWN_MIN_ALTITUDE_M = 100.0
OBLIQUE_DOWN_MIN_ANGLE_DEGREES = 20.0
OBLIQUE_DOWN_MAX_ANGLE_DEGREES = 70.0
OBLIQUE_DOWN_MIN_ALTITUDE_M = 50.0
SATELLITE_MIN_ALTITUDE_M = 5_000.0
SATELLITE_MIN_SLANT_DISTANCE_M = 5_000.0
MIN_CAMERA_SEPARATION_M = 1.0


class TrainingCaptureContractError(ValueError):
    """Raised when planning or capture evidence cannot be proven current."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingCaptureContractError(
            f"{label} is absent or invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise TrainingCaptureContractError(f"{label} must be an object")
    return value


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw.replace("\\", "/"))
    if (
        not raw
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TrainingCaptureContractError(f"{label} is unsafe")
    return path


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if not _SHA256.fullmatch(digest):
        raise TrainingCaptureContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return digest


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TrainingCaptureContractError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingCaptureContractError(
            f"{label} must be finite"
        ) from exc
    if not math.isfinite(result):
        raise TrainingCaptureContractError(f"{label} must be finite")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingCaptureContractError(
            f"{label} must be a positive integer"
        )
    return value


def _regular_file_below(
    *,
    volume_root: Path,
    path: Path,
    label: str,
) -> Path:
    raw = path
    resolved = raw.resolve()
    if (
        raw.is_symlink()
        or not resolved.is_file()
        or not _inside(volume_root, resolved)
        or resolved.stat().st_size <= 0
    ):
        raise TrainingCaptureContractError(
            f"{label} is absent, empty, unsafe or outside the volume"
        )
    return resolved


def _locked_file(
    *,
    volume_root: Path,
    record: object,
    label: str,
) -> tuple[Path, dict[str, object]]:
    if not isinstance(record, Mapping):
        raise TrainingCaptureContractError(f"{label} lock is absent")
    relative = _safe_relative_path(record.get("path"), label=f"{label} path")
    path = volume_root.joinpath(*relative.parts)
    resolved = _regular_file_below(
        volume_root=volume_root,
        path=path,
        label=label,
    )
    expected_sha256 = _require_sha256(
        record.get("sha256"),
        label=label,
    )
    expected_size = record.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or resolved.stat().st_size != expected_size
        or _sha256_file(resolved) != expected_sha256
    ):
        raise TrainingCaptureContractError(
            f"{label} SHA-256 or size lock drifted"
        )
    return resolved, {
        "path": relative.as_posix(),
        "sha256": expected_sha256,
        "size_bytes": expected_size,
    }


def _file_record(*, root: Path, path: Path) -> dict[str, object]:
    resolved = _regular_file_below(
        volume_root=root,
        path=path,
        label="capture artifact",
    )
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _read_exact(stream: BinaryIO, size: int, *, label: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise TrainingCaptureContractError(
            f"captured OpenEXR is truncated at {label}"
        )
    return value


def _read_exr_cstring(
    stream: BinaryIO,
    *,
    label: str,
    maximum_bytes: int = 255,
) -> str:
    value = bytearray()
    for _index in range(maximum_bytes + 1):
        byte = stream.read(1)
        if byte == b"":
            raise TrainingCaptureContractError(
                f"captured OpenEXR is truncated at {label}"
            )
        if byte == b"\x00":
            try:
                return value.decode("ascii")
            except UnicodeDecodeError as exc:
                raise TrainingCaptureContractError(
                    f"captured OpenEXR {label} is not ASCII"
                ) from exc
        value.extend(byte)
    raise TrainingCaptureContractError(
        f"captured OpenEXR {label} exceeds {maximum_bytes} bytes"
    )


def _exr_attribute(
    attributes: Mapping[str, tuple[str, bytes]],
    *,
    name: str,
    type_name: str,
    size: int | None = None,
) -> bytes:
    record = attributes.get(name)
    if record is None or record[0] != type_name:
        raise TrainingCaptureContractError(
            f"captured OpenEXR requires {name}:{type_name}"
        )
    value = record[1]
    if size is not None and len(value) != size:
        raise TrainingCaptureContractError(
            f"captured OpenEXR {name} has an invalid size"
        )
    return value


def _parse_exr_channels(value: bytes) -> dict[str, tuple[int, int, int]]:
    channels: dict[str, tuple[int, int, int]] = {}
    offset = 0
    while True:
        terminator = value.find(b"\x00", offset)
        if terminator < 0:
            raise TrainingCaptureContractError(
                "captured OpenEXR channel list is unterminated"
            )
        raw_name = value[offset:terminator]
        offset = terminator + 1
        if not raw_name:
            if offset != len(value):
                raise TrainingCaptureContractError(
                    "captured OpenEXR channel list has trailing bytes"
                )
            break
        if len(value) - offset < 16:
            raise TrainingCaptureContractError(
                "captured OpenEXR channel entry is truncated"
            )
        try:
            name = raw_name.decode("ascii")
        except UnicodeDecodeError as exc:
            raise TrainingCaptureContractError(
                "captured OpenEXR channel name is not ASCII"
            ) from exc
        pixel_type = struct.unpack_from("<i", value, offset)[0]
        linear = value[offset + 4]
        reserved = value[offset + 5 : offset + 8]
        x_sampling, y_sampling = struct.unpack_from("<ii", value, offset + 8)
        offset += 16
        if (
            name in channels
            or reserved != b"\x00\x00\x00"
            or linear not in {0, 1}
        ):
            raise TrainingCaptureContractError(
                "captured OpenEXR channel entry is invalid"
            )
        channels[name] = (pixel_type, x_sampling, y_sampling)
    return channels


def _inverse_openexr_zip_preprocess(value: bytes) -> bytes:
    """Undo OpenEXR ZIP byte prediction and even/odd byte reordering."""

    if not value:
        return value
    predicted = bytearray(value)
    for index in range(1, len(predicted)):
        predicted[index] = (
            predicted[index - 1] + predicted[index] - 128
        ) & 0xFF
    first_count = (len(predicted) + 1) // 2
    restored = bytearray(len(predicted))
    restored[0::2] = predicted[:first_count]
    restored[1::2] = predicted[first_count:]
    return bytes(restored)


def _validate_openexr_capture(
    *,
    path: Path,
    width_px: int,
    height_px: int,
    modality: str = "normal_rgb",
    collect_pixels: bool = False,
) -> dict[str, object]:
    """Validate one modality and optionally return decoded scanline samples."""

    file_size = path.stat().st_size
    decoded_chunks: list[dict[str, object]] = []
    try:
        with path.open("rb") as stream:
            magic, version_flags = struct.unpack(
                "<II",
                _read_exact(stream, 8, label="preamble"),
            )
            version = version_flags & 0xFF
            unsupported_flags = version_flags & (
                0x00000200 | 0x00000800 | 0x00001000
            )
            if (
                magic != _OPENEXR_MAGIC
                or version != _OPENEXR_VERSION
                or unsupported_flags
            ):
                raise TrainingCaptureContractError(
                    "captured image is not a single-part scanline OpenEXR v2"
                )

            attributes: dict[str, tuple[str, bytes]] = {}
            while True:
                name = _read_exr_cstring(stream, label="attribute name")
                if not name:
                    break
                if name in attributes:
                    raise TrainingCaptureContractError(
                        "captured OpenEXR has duplicate attributes"
                    )
                type_name = _read_exr_cstring(
                    stream,
                    label=f"{name} type",
                )
                if not type_name:
                    raise TrainingCaptureContractError(
                        f"captured OpenEXR {name} has no type"
                    )
                size = struct.unpack(
                    "<I",
                    _read_exact(stream, 4, label=f"{name} size"),
                )[0]
                if size > 8 * 1024 * 1024:
                    raise TrainingCaptureContractError(
                        f"captured OpenEXR {name} is unreasonably large"
                    )
                attributes[name] = (
                    type_name,
                    _read_exact(stream, size, label=name),
                )

            channels = _parse_exr_channels(
                _exr_attribute(
                    attributes,
                    name="channels",
                    type_name="chlist",
                )
            )
            if modality in {"normal_rgb", "negative"}:
                if set(channels) != {"R", "G", "B"} or any(
                    specification != (_OPENEXR_HALF, 1, 1)
                    for specification in channels.values()
                ):
                    raise TrainingCaptureContractError(
                        f"{modality} OpenEXR must contain exactly RGB HALF "
                        "channels with unit sampling"
                    )
            elif modality == "thermal_hotspot":
                if channels != {"T": (_OPENEXR_FLOAT, 1, 1)}:
                    raise TrainingCaptureContractError(
                        "thermal_hotspot OpenEXR must contain exactly one "
                        "FLOAT temperature channel T"
                    )
            elif modality == "registration_geometry_id":
                if channels != {"ID": (_OPENEXR_FLOAT, 1, 1)}:
                    raise TrainingCaptureContractError(
                        "registration geometry-ID OpenEXR must contain "
                        "exactly one FLOAT channel ID"
                    )
            else:
                raise TrainingCaptureContractError(
                    "captured OpenEXR modality is unknown"
                )
            compression = _exr_attribute(
                attributes,
                name="compression",
                type_name="compression",
                size=1,
            )[0]
            if compression != _OPENEXR_ZIP_COMPRESSION:
                raise TrainingCaptureContractError(
                    "captured OpenEXR must use ZIP compression"
                )
            data_window = struct.unpack(
                "<iiii",
                _exr_attribute(
                    attributes,
                    name="dataWindow",
                    type_name="box2i",
                    size=16,
                ),
            )
            display_window = struct.unpack(
                "<iiii",
                _exr_attribute(
                    attributes,
                    name="displayWindow",
                    type_name="box2i",
                    size=16,
                ),
            )
            xmin, ymin, xmax, ymax = data_window
            if (
                display_window != data_window
                or xmax < xmin
                or ymax < ymin
                or xmax - xmin + 1 != width_px
                or ymax - ymin + 1 != height_px
            ):
                raise TrainingCaptureContractError(
                    "captured OpenEXR dimensions differ from camera intrinsics"
                )
            line_order = _exr_attribute(
                attributes,
                name="lineOrder",
                type_name="lineOrder",
                size=1,
            )[0]
            if line_order not in {0, 1, 2}:
                raise TrainingCaptureContractError(
                    "captured OpenEXR line order is invalid"
                )
            if modality in {"normal_rgb", "negative"}:
                chromaticities = struct.unpack(
                    "<ffffffff",
                    _exr_attribute(
                        attributes,
                        name="chromaticities",
                        type_name="chromaticities",
                        size=32,
                    ),
                )
                if any(
                    not math.isclose(
                        observed,
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-5,
                    )
                    for observed, expected in zip(
                        chromaticities,
                        _OPENEXR_AP1_CHROMATICITIES,
                        strict=True,
                    )
                ):
                    raise TrainingCaptureContractError(
                        f"{modality} OpenEXR chromaticities are not linear "
                        "AP1"
                    )
            bytes_per_pixel = sum(
                2 if specification[0] == _OPENEXR_HALF else 4
                for specification in channels.values()
            )
            if bytes_per_pixel <= 0:
                raise TrainingCaptureContractError(
                    "captured OpenEXR has no supported samples"
                )

            chunk_count = (
                height_px + _OPENEXR_ZIP_LINES_PER_CHUNK - 1
            ) // _OPENEXR_ZIP_LINES_PER_CHUNK
            table_start = stream.tell()
            table_size = chunk_count * 8
            offsets = struct.unpack(
                f"<{chunk_count}Q",
                _read_exact(stream, table_size, label="chunk offset table"),
            )
            table_end = table_start + table_size
            intervals: list[tuple[int, int]] = []
            bytes_per_scanline = width_px * bytes_per_pixel
            for chunk_index, chunk_offset in enumerate(offsets):
                if chunk_offset < table_end or chunk_offset + 8 > file_size:
                    raise TrainingCaptureContractError(
                        "captured OpenEXR chunk offset is outside the file"
                    )
                stream.seek(chunk_offset)
                y_coordinate, packed_size = struct.unpack(
                    "<iI",
                    _read_exact(
                        stream,
                        8,
                        label=f"chunk {chunk_index} header",
                    ),
                )
                expected_y = (
                    ymin
                    + chunk_index * _OPENEXR_ZIP_LINES_PER_CHUNK
                )
                scanline_count = min(
                    _OPENEXR_ZIP_LINES_PER_CHUNK,
                    ymax - expected_y + 1,
                )
                expected_unpacked_size = (
                    bytes_per_scanline * scanline_count
                )
                chunk_end = chunk_offset + 8 + packed_size
                if (
                    y_coordinate != expected_y
                    or packed_size <= 0
                    or packed_size > expected_unpacked_size
                    or chunk_end > file_size
                ):
                    raise TrainingCaptureContractError(
                        "captured OpenEXR ZIP chunk is structurally invalid"
                    )
                packed = _read_exact(
                    stream,
                    packed_size,
                    label=f"chunk {chunk_index} payload",
                )
                if packed_size < expected_unpacked_size:
                    try:
                        decoder = zlib.decompressobj()
                        unpacked = decoder.decompress(
                            packed,
                            expected_unpacked_size + 1,
                        )
                        if decoder.unconsumed_tail:
                            raise zlib.error("unbounded ZIP payload")
                        unpacked += decoder.flush()
                    except zlib.error as exc:
                        raise TrainingCaptureContractError(
                            "captured OpenEXR ZIP chunk cannot be decoded"
                        ) from exc
                    if (
                        len(unpacked) != expected_unpacked_size
                        or not decoder.eof
                        or decoder.unused_data
                    ):
                        raise TrainingCaptureContractError(
                            "captured OpenEXR ZIP chunk has the wrong RGB size"
                        )
                    raw_samples = _inverse_openexr_zip_preprocess(unpacked)
                else:
                    raw_samples = packed
                if collect_pixels:
                    decoded_chunks.append(
                        {
                            "y_coordinate": y_coordinate,
                            "scanline_count": scanline_count,
                            "samples": raw_samples,
                        }
                    )
                intervals.append((chunk_offset, chunk_end))

            cursor = table_end
            for chunk_start, chunk_end in sorted(intervals):
                if chunk_start != cursor:
                    raise TrainingCaptureContractError(
                        "captured OpenEXR chunks overlap or leave gaps"
                    )
                cursor = chunk_end
            if cursor != file_size:
                raise TrainingCaptureContractError(
                    "captured OpenEXR contains unreferenced trailing data"
                )
    except OSError as exc:
        raise TrainingCaptureContractError(
            "captured OpenEXR cannot be inspected"
        ) from exc
    return {
        "modality": modality,
        "width_px": width_px,
        "height_px": height_px,
        "data_window": list(data_window),
        "channel_order": list(channels),
        "channels": {
            name: list(specification)
            for name, specification in channels.items()
        },
        "decoded_chunks": decoded_chunks,
    }


def _stable_id(value: object, *, label: str) -> str:
    identifier = str(value)
    if not _STABLE_ID.fullmatch(identifier):
        raise TrainingCaptureContractError(f"{label} is not a stable ID")
    return identifier


def _normalize_pose(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "position_m",
        "orientation_xyzw",
    }:
        raise TrainingCaptureContractError(f"{label} pose is incomplete")
    position = value["position_m"]
    orientation = value["orientation_xyzw"]
    if (
        not isinstance(position, list)
        or len(position) != 3
        or not isinstance(orientation, list)
        or len(orientation) != 4
    ):
        raise TrainingCaptureContractError(
            f"{label} pose dimensions are invalid"
        )
    normalized_position = [
        _finite(item, label=f"{label} position")
        for item in position
    ]
    normalized_orientation = [
        _finite(item, label=f"{label} orientation")
        for item in orientation
    ]
    norm = math.sqrt(
        math.fsum(item * item for item in normalized_orientation)
    )
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise TrainingCaptureContractError(
            f"{label} orientation quaternion is not normalized"
        )
    return {
        "position_m": normalized_position,
        "orientation_xyzw": normalized_orientation,
    }


def _normalize_position(value: object, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise TrainingCaptureContractError(
            f"{label} must contain exactly three coordinates"
        )
    return [
        _finite(item, label=f"{label} coordinate")
        for item in value
    ]


def _vector_subtract(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    return [
        float(left[index]) - float(right[index])
        for index in range(3)
    ]


def _vector_add(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    return [
        float(left[index]) + float(right[index])
        for index in range(3)
    ]


def _vector_scale(value: Sequence[float], scale: float) -> list[float]:
    return [float(component) * scale for component in value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(
        float(left[index]) * float(right[index])
        for index in range(3)
    )


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2])
        - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0])
        - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1])
        - float(left[1]) * float(right[0]),
    ]


def _length(value: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(item) ** 2 for item in value))


def _unit(value: Sequence[float], *, label: str) -> list[float]:
    norm = _length(value)
    if norm <= 1.0e-9:
        raise TrainingCaptureContractError(f"{label} has zero length")
    return _vector_scale(value, 1.0 / norm)


def _quaternion_from_rotation_rows(
    rows: Sequence[Sequence[float]],
) -> list[float]:
    m00, m01, m02 = (float(item) for item in rows[0])
    m10, m11, m12 = (float(item) for item in rows[1])
    m20, m21, m22 = (float(item) for item in rows[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    quaternion = [x, y, z, w]
    norm = math.sqrt(math.fsum(item * item for item in quaternion))
    if norm <= 1.0e-9:
        raise TrainingCaptureContractError(
            "look-at orientation quaternion has zero length"
        )
    return [item / norm for item in quaternion]


def _rotate_by_quaternion(
    value: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> list[float]:
    vector_part = [float(item) for item in quaternion_xyzw[:3]]
    w = float(quaternion_xyzw[3])
    twice_cross = _vector_scale(_cross(vector_part, value), 2.0)
    return _vector_add(
        _vector_add(value, _vector_scale(twice_cross, w)),
        _cross(vector_part, twice_cross),
    )


def _look_at_quaternion_xyzw(
    *,
    position_m: Sequence[float],
    target_m: Sequence[float],
) -> list[float]:
    """Orient an OpenUSD camera (-Z forward, +Y up) toward ``target_m``."""

    forward = _unit(
        _vector_subtract(target_m, position_m),
        label="camera-to-flame vector",
    )
    world_up = [0.0, 0.0, 1.0]
    if abs(_dot(forward, world_up)) > 0.985:
        world_up = [0.0, 1.0, 0.0]
    right = _unit(
        _cross(forward, world_up),
        label="look-at right axis",
    )
    camera_up = _unit(
        _cross(right, forward),
        label="look-at up axis",
    )
    camera_back = _vector_scale(forward, -1.0)
    quaternion = _quaternion_from_rotation_rows(
        (
            (right[0], camera_up[0], camera_back[0]),
            (right[1], camera_up[1], camera_back[1]),
            (right[2], camera_up[2], camera_back[2]),
        )
    )
    observed_forward = _unit(
        _rotate_by_quaternion([0.0, 0.0, -1.0], quaternion),
        label="rotated camera forward",
    )
    if _dot(observed_forward, forward) < 1.0 - 1.0e-9:
        raise TrainingCaptureContractError(
            "look-at quaternion does not face the active flame target"
        )
    return quaternion


def _normalize_intrinsics(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _INTRINSIC_FIELDS:
        raise TrainingCaptureContractError(
            f"{label} intrinsics are incomplete"
        )
    if value.get("model") != "pinhole":
        raise TrainingCaptureContractError(
            f"{label} camera model must be pinhole"
        )
    width = _positive_integer(value["width_px"], label=f"{label} width")
    height = _positive_integer(value["height_px"], label=f"{label} height")
    fx = _finite(value["fx_px"], label=f"{label} fx")
    fy = _finite(value["fy_px"], label=f"{label} fy")
    cx = _finite(value["cx_px"], label=f"{label} cx")
    cy = _finite(value["cy_px"], label=f"{label} cy")
    near = _finite(value["near_clip_m"], label=f"{label} near clip")
    far = _finite(value["far_clip_m"], label=f"{label} far clip")
    focal_length_mm = _finite(
        value["focal_length_mm"],
        label=f"{label} focal length",
    )
    horizontal_aperture_mm = _finite(
        value["horizontal_aperture_mm"],
        label=f"{label} horizontal aperture",
    )
    vertical_aperture_mm = _finite(
        value["vertical_aperture_mm"],
        label=f"{label} vertical aperture",
    )
    f_stop = _finite(value["f_stop"], label=f"{label} f-stop")
    if (
        fx <= 0.0
        or fy <= 0.0
        or not 0.0 <= cx < width
        or not 0.0 <= cy < height
        or near <= 0.0
        or far <= near
        or focal_length_mm <= 0.0
        or horizontal_aperture_mm <= 0.0
        or vertical_aperture_mm <= 0.0
        or f_stop <= 0.0
        or not math.isclose(
            fx,
            focal_length_mm / horizontal_aperture_mm * width,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
        or not math.isclose(
            fy,
            focal_length_mm / vertical_aperture_mm * height,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
    ):
        raise TrainingCaptureContractError(
            f"{label} intrinsics are physically invalid"
        )
    return {
        "model": "pinhole",
        "width_px": width,
        "height_px": height,
        "fx_px": fx,
        "fy_px": fy,
        "cx_px": cx,
        "cy_px": cy,
        "near_clip_m": near,
        "far_clip_m": far,
        "focal_length_mm": focal_length_mm,
        "horizontal_aperture_mm": horizontal_aperture_mm,
        "vertical_aperture_mm": vertical_aperture_mm,
        "f_stop": f_stop,
    }


def _normalize_fire(
    value: object,
    *,
    label: str,
    expected_time_seconds: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _FIRE_FIELDS:
        raise TrainingCaptureContractError(f"{label} fire state is incomplete")
    state = str(value.get("state", "")).strip()
    if not state:
        raise TrainingCaptureContractError(f"{label} fire state is empty")
    simulation_time = _finite(
        value["simulation_time_seconds"],
        label=f"{label} simulation time",
    )
    active_centroid = _normalize_position(
        value["active_flame_centroid_local_m"],
        label=f"{label} active flame centroid local",
    )
    numeric = {
        field: _finite(value[field], label=f"{label} {field}")
        for field in _FIRE_FIELDS
        if field
        not in {
            "state",
            "simulation_time_seconds",
            "active_flame_centroid_local_m",
        }
    }
    front_radius = numeric["active_flame_front_radius_m"]
    if (
        not math.isclose(
            simulation_time,
            expected_time_seconds,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or any(number < 0.0 for number in numeric.values())
        or front_radius <= 0.0
        or numeric["active_area_m2"] > numeric["burned_area_m2"] + 1.0e-6
        or numeric["active_area_m2"]
        > math.pi * front_radius * front_radius + 1.0e-6
    ):
        raise TrainingCaptureContractError(
            f"{label} fire metrics are incoherent"
        )
    return {
        "state": state,
        "simulation_time_seconds": simulation_time,
        "active_flame_centroid_local_m": active_centroid,
        **{field: numeric[field] for field in sorted(numeric)},
    }


def _normalize_weather(
    value: object,
    *,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _WEATHER_FIELDS:
        raise TrainingCaptureContractError(
            f"{label} weather state is incomplete"
        )
    weather = {
        field: _finite(value[field], label=f"{label} {field}")
        for field in sorted(_WEATHER_FIELDS)
    }
    if (
        not 0.0 <= weather["relative_humidity_percent"] <= 100.0
        or weather["wind_speed_m_s"] < 0.0
        or not 0.0 <= weather["wind_direction_degrees"] < 360.0
        or weather["precipitation_mm_h"] < 0.0
        or not 0.0 <= weather["cloud_cover_fraction"] <= 1.0
        or weather["pressure_hpa"] <= 0.0
    ):
        raise TrainingCaptureContractError(
            f"{label} weather metrics are physically invalid"
        )
    return weather


def _validate_view_geometry(
    *,
    view_class: str,
    camera_position_m: Sequence[float],
    active_centroid_m: Sequence[float],
    label: str,
) -> None:
    camera_above_target = (
        float(camera_position_m[2]) - float(active_centroid_m[2])
    )
    horizontal_distance = math.hypot(
        float(camera_position_m[0]) - float(active_centroid_m[0]),
        float(camera_position_m[1]) - float(active_centroid_m[1]),
    )
    slant_distance = _length(
        _vector_subtract(active_centroid_m, camera_position_m)
    )
    if view_class == "top_down":
        if camera_above_target <= 0.0:
            raise TrainingCaptureContractError(
                f"{label} top-down camera is not above the active flames"
            )
        nadir_angle = math.degrees(
            math.atan2(horizontal_distance, camera_above_target)
        )
        if (
            camera_above_target < TOP_DOWN_MIN_ALTITUDE_M
            or nadir_angle > TOP_DOWN_MAX_NADIR_ANGLE_DEGREES
        ):
            raise TrainingCaptureContractError(
                f"{label} top-down geometry is not quasi-vertical"
            )
    elif view_class == "plunging_oblique":
        if camera_above_target <= 0.0 or horizontal_distance <= 0.0:
            raise TrainingCaptureContractError(
                f"{label} plunging-oblique camera geometry is invalid"
            )
        downward_angle = math.degrees(
            math.atan2(camera_above_target, horizontal_distance)
        )
        if (
            camera_above_target < OBLIQUE_DOWN_MIN_ALTITUDE_M
            or downward_angle < OBLIQUE_DOWN_MIN_ANGLE_DEGREES
            or downward_angle > OBLIQUE_DOWN_MAX_ANGLE_DEGREES
        ):
            raise TrainingCaptureContractError(
                f"{label} plunging-oblique geometry is outside its "
                "inclination "
                "contract"
            )
    elif view_class == "satellite_high_altitude" and (
        camera_above_target < SATELLITE_MIN_ALTITUDE_M
        or slant_distance < SATELLITE_MIN_SLANT_DISTANCE_M
    ):
        raise TrainingCaptureContractError(
            f"{label} satellite camera is not at high altitude"
        )


def _derive_observation_aim(
    *,
    scene_id: str,
    day_index: int,
    capture_hour: str,
    camera: Mapping[str, object],
    fire: Mapping[str, object],
) -> dict[str, object]:
    camera_id = str(camera["camera_id"])
    label = (
        f"{scene_id} day {day_index} {capture_hour} {camera_id}"
    )
    view_class = str(camera["view_class"])
    pose_local = camera["pose_local"]
    camera_position = pose_local["position_m"]
    active_centroid = fire["active_flame_centroid_local_m"]
    active_radius = float(fire["active_flame_front_radius_m"])
    _validate_view_geometry(
        view_class=view_class,
        camera_position_m=camera_position,
        active_centroid_m=active_centroid,
        label=label,
    )
    # Camera framing is a scene-level decision made before simulation.  It is
    # intentionally never re-aimed as the fire evolves: a temporal sequence
    # is comparable only when position, rotation and intrinsics are identical
    # at every capture instant.  The active centroid below is a truth reference
    # used for clip/FOV/visibility checks, not a look-at command.
    target = [float(item) for item in active_centroid]
    target_offset = 0.0
    target_vector = _vector_subtract(target, camera_position)
    optical_forward = _unit(
        _rotate_by_quaternion(
            [0.0, 0.0, -1.0],
            pose_local["orientation_xyzw"],
        ),
        label=f"{label} optical forward",
    )
    if _dot(optical_forward, target_vector) <= 0.0:
        raise TrainingCaptureContractError(
            f"{label} active flame target is not in front of the camera"
        )

    intrinsics = camera["intrinsics"]
    near_clip = float(intrinsics["near_clip_m"])
    far_clip = float(intrinsics["far_clip_m"])
    centroid_vector = _vector_subtract(active_centroid, camera_position)
    centroid_distance = _length(centroid_vector)
    if (
        centroid_distance <= active_radius
        or centroid_distance - active_radius <= near_clip
        or centroid_distance + active_radius >= far_clip
    ):
        raise TrainingCaptureContractError(
            f"{label} active flame front is outside the camera clip range"
        )
    centroid_direction = _unit(
        centroid_vector,
        label=f"{label} active centroid direction",
    )
    center_separation = math.acos(
        max(-1.0, min(1.0, _dot(optical_forward, centroid_direction)))
    )
    angular_radius = math.asin(
        min(1.0, active_radius / centroid_distance)
    )
    fx = float(intrinsics["fx_px"])
    fy = float(intrinsics["fy_px"])
    width = float(intrinsics["width_px"])
    height = float(intrinsics["height_px"])
    cx = float(intrinsics["cx_px"])
    cy = float(intrinsics["cy_px"])
    sensor_half_angles = (
        math.atan(cx / fx),
        math.atan((width - cx) / fx),
        math.atan(cy / fy),
        math.atan((height - cy) / fy),
    )
    usable_half_angle = (
        min(sensor_half_angles) * FOV_USABLE_HALF_ANGLE_FRACTION
    )
    required_half_angle = center_separation + angular_radius
    if required_half_angle > usable_half_angle:
        raise TrainingCaptureContractError(
            f"{label} active flame front does not fit in the FOV with margin"
        )
    aim: dict[str, object] = {
        "camera_id": camera_id,
        "view_class": view_class,
        "pose_local": {
            "position_m": [float(item) for item in camera_position],
            "orientation_xyzw": [
                float(item) for item in pose_local["orientation_xyzw"]
            ],
        },
        "target_local_m": target,
        "target_offset_from_active_centroid_m": target_offset,
        "active_front_framing": {
            "centroid_distance_m": centroid_distance,
            "front_radius_m": active_radius,
            "required_half_angle_degrees": math.degrees(
                required_half_angle
            ),
            "usable_half_angle_degrees": math.degrees(
                usable_half_angle
            ),
            "near_edge_distance_m": centroid_distance - active_radius,
            "far_edge_distance_m": centroid_distance + active_radius,
        },
    }
    aim["aim_contract_sha256"] = _canonical_sha256(aim)
    return aim


def _validate_convergent_aims(
    *,
    aims: Sequence[Mapping[str, object]],
    active_front_radius_m: float,
    label: str,
) -> None:
    if len(aims) != CAMERAS_PER_SCENE:
        raise TrainingCaptureContractError(
            f"{label} requires exactly 40 convergent camera aims"
        )
    maximum_distance = (
        active_front_radius_m
        * TARGET_GROUP_DIAMETER_RADIUS_FRACTION
    )
    for left_index, left in enumerate(aims):
        left_target = left.get("target_local_m")
        if not isinstance(left_target, list) or len(left_target) != 3:
            raise TrainingCaptureContractError(
                f"{label} camera target is malformed"
            )
        for right in aims[left_index + 1 :]:
            right_target = right.get("target_local_m")
            if not isinstance(right_target, list) or len(right_target) != 3:
                raise TrainingCaptureContractError(
                    f"{label} camera target is malformed"
                )
            if (
                _length(_vector_subtract(left_target, right_target))
                > maximum_distance + 1.0e-6
            ):
                raise TrainingCaptureContractError(
                    f"{label} camera targets do not converge on the same "
                    "active flame front"
                )


def _validate_gate(
    *,
    volume_root: Path,
    simulation_allowed_receipt_path: Path,
    campaign_id: str,
    campaign_index_sha256: str,
    raw_gate_artifacts: object,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    if (
        not isinstance(raw_gate_artifacts, Mapping)
        or set(raw_gate_artifacts) != set(GATE_ARTIFACT_KEYS)
    ):
        raise TrainingCaptureContractError(
            "training capture sources must lock every simulation gate artifact"
        )
    gate_artifacts: dict[str, dict[str, object]] = {}
    gate_paths: dict[str, Path] = {}
    for key in GATE_ARTIFACT_KEYS:
        path, lock = _locked_file(
            volume_root=volume_root,
            record=raw_gate_artifacts.get(key),
            label=f"simulation gate {key}",
        )
        gate_paths[key] = path
        gate_artifacts[key] = lock

    pending = _read_json(
        gate_paths["pending_review"],
        label="pending Editor review receipt",
    )
    opened = _read_json(
        gate_paths["editor_opened"],
        label="Editor opened receipt",
    )
    acceptance = _read_json(
        gate_paths["editor_acceptance"],
        label="Editor acceptance receipt",
    )
    acceptance_bindings = acceptance.get("bindings")
    if (
        acceptance.get("schema_version") != SCHEMA_VERSION
        or acceptance.get("campaign_id") != campaign_id
        or acceptance.get("scene_id") != "SIM-01"
        or acceptance.get("decision") != "accepted"
        or acceptance.get("status") != "EDITOR_REVIEW_ACCEPTED"
        or not str(acceptance.get("reviewer", "")).strip()
        or not str(acceptance.get("reviewed_at", "")).strip()
        or acceptance.get("acknowledgement")
        != EDITOR_REVIEW_ACKNOWLEDGEMENT
        or not isinstance(acceptance_bindings, Mapping)
    ):
        raise TrainingCaptureContractError(
            "capture is blocked without the current accepted SIM-01 Editor "
            "review receipt"
        )
    pending_bindings = pending.get("bindings")
    if (
        pending.get("schema_version") != SCHEMA_VERSION
        or pending.get("campaign_id") != campaign_id
        or pending.get("scene_id") != "SIM-01"
        or pending.get("status") != "AWAITING_EDITOR_REVIEW"
        or pending.get("human_review") != "pending"
        or not isinstance(pending_bindings, Mapping)
        or dict(pending_bindings) != dict(acceptance_bindings)
        or opened.get("state") != "opened_for_human_review"
        or opened.get("human_review") != "pending"
        or opened.get("root_usd_sha256")
        != gate_artifacts["root_usd"]["sha256"]
        or opened.get("pending_review_sha256")
        != gate_artifacts["pending_review"]["sha256"]
        or acceptance.get("pending_review_sha256")
        != gate_artifacts["pending_review"]["sha256"]
        or acceptance.get("editor_opened_sha256")
        != gate_artifacts["editor_opened"]["sha256"]
    ):
        raise TrainingCaptureContractError(
            "accepted SIM-01 review is not bound to the pending and real "
            "Editor-open evidence"
        )

    receipt_path = _regular_file_below(
        volume_root=volume_root,
        path=simulation_allowed_receipt_path,
        label="FIRE_SIMULATION_ALLOWED receipt",
    )
    payload = _read_json(
        receipt_path,
        label="FIRE_SIMULATION_ALLOWED receipt",
    )
    bindings = payload.get("bindings")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("state") != SIMULATION_ALLOWED_STATE
        or payload.get("campaign_id") != campaign_id
        or payload.get("acceptance_sha256")
        != gate_artifacts["editor_acceptance"]["sha256"]
        or not isinstance(bindings, Mapping)
        or set(bindings) != GATE_BINDING_KEYS
        or dict(bindings) != dict(acceptance_bindings)
    ):
        raise TrainingCaptureContractError(
            "capture is blocked without a current FIRE_SIMULATION_ALLOWED "
            "receipt bound to this campaign"
        )
    for key in sorted(GATE_BINDING_KEYS):
        value = bindings[key]
        _require_sha256(value, label=f"simulation gate {key}")
    expected_direct_bindings = {
        "campaign_index_sha256": campaign_index_sha256,
        "root_usd_sha256": gate_artifacts["root_usd"]["sha256"],
        **{
            binding: gate_artifacts[artifact]["sha256"]
            for binding, artifact in _DIRECT_GATE_BINDINGS.items()
        },
    }
    if any(
        bindings.get(key) != expected
        for key, expected in expected_direct_bindings.items()
    ):
        raise TrainingCaptureContractError(
            "FIRE_SIMULATION_ALLOWED direct artifact hashes are stale"
        )
    lock = _file_record(root=volume_root, path=receipt_path)
    return dict(payload), lock, gate_artifacts


def _normalized_scene(
    *,
    volume_root: Path,
    raw_scene: object,
    scene_id: str,
    duration_days: int,
    campaign_binding: Mapping[str, object],
    build_receipt: Mapping[str, object],
) -> dict[str, object]:
    required_fields = {
        "scene_id",
        "variant_id",
        "duration_days",
        "scene_root",
        "scene_origin_epsg2154_ign69_m",
        "georeference",
        "cameras",
        "days",
    }
    if not isinstance(raw_scene, Mapping) or set(raw_scene) != required_fields:
        raise TrainingCaptureContractError(
            f"{scene_id} capture source is incomplete"
        )
    if (
        raw_scene.get("scene_id") != scene_id
        or raw_scene.get("duration_days") != duration_days
    ):
        raise TrainingCaptureContractError(
            f"{scene_id} duration or identity differs from the campaign"
        )
    variant_id = _stable_id(
        raw_scene.get("variant_id"),
        label=f"{scene_id} variant",
    )
    scene_path, scene_lock = _locked_file(
        volume_root=volume_root,
        record=raw_scene.get("scene_root"),
        label=f"{scene_id} root USD",
    )
    scene_binding = campaign_binding.get("scene_binding")
    if (
        not isinstance(scene_binding, Mapping)
        or scene_lock["path"] != scene_binding.get("root_usd")
        or scene_path.suffix.casefold() not in {".usd", ".usda", ".usdc"}
    ):
        raise TrainingCaptureContractError(
            f"{scene_id} root USD differs from the campaign index"
        )
    raw_origin = raw_scene.get("scene_origin_epsg2154_ign69_m")
    if not isinstance(raw_origin, list) or len(raw_origin) != 3:
        raise TrainingCaptureContractError(
            f"{scene_id} has no EPSG:2154/IGN69 origin"
        )
    declared_origin = [
        _finite(value, label=f"{scene_id} origin")
        for value in raw_origin
    ]
    raw_georeference = raw_scene.get("georeference")
    expected_georeference_fields = {
        "horizontal_crs",
        "vertical_datum",
        "axis_order",
        "local_axes",
        "axis_mapping",
        "units",
        "origin_epsg2154_ign69_m",
        "scene_root_sha256",
        "build_receipt_sha256",
        "provenance_receipt",
    }
    if (
        not isinstance(raw_georeference, Mapping)
        or set(raw_georeference) != expected_georeference_fields
        or raw_georeference.get("horizontal_crs") != "EPSG:2154"
        or raw_georeference.get("vertical_datum") != "IGN69"
        or raw_georeference.get("axis_order")
        != list(GEOREFERENCE_AXIS_ORDER)
        or raw_georeference.get("local_axes")
        != list(GEOREFERENCE_LOCAL_AXES)
        or raw_georeference.get("axis_mapping")
        != GEOREFERENCE_AXIS_MAPPING
        or raw_georeference.get("units") != "metres"
        or raw_georeference.get("scene_root_sha256")
        != scene_lock["sha256"]
        or raw_georeference.get("build_receipt_sha256")
        != build_receipt.get("sha256")
    ):
        raise TrainingCaptureContractError(
            f"{scene_id} georeference is not bound to its root and build "
            "receipt"
        )
    origin = _normalize_position(
        raw_georeference.get("origin_epsg2154_ign69_m"),
        label=f"{scene_id} authenticated georeference origin",
    )
    if declared_origin != origin:
        raise TrainingCaptureContractError(
            f"{scene_id} declared origin differs from authenticated "
            "georeference"
        )
    if (
        origin == [0.0, 0.0, 0.0]
        or not LAMBERT93_EASTING_RANGE_M[0]
        <= origin[0]
        <= LAMBERT93_EASTING_RANGE_M[1]
        or not LAMBERT93_NORTHING_RANGE_M[0]
        <= origin[1]
        <= LAMBERT93_NORTHING_RANGE_M[1]
        or not IGN69_ALTITUDE_RANGE_M[0]
        <= origin[2]
        <= IGN69_ALTITUDE_RANGE_M[1]
    ):
        raise TrainingCaptureContractError(
            f"{scene_id} origin is outside the French Lambert-93/IGN69 "
            "domain"
        )
    provenance_path, provenance_lock = _locked_file(
        volume_root=volume_root,
        record=raw_georeference.get("provenance_receipt"),
        label=f"{scene_id} georeference provenance receipt",
    )
    provenance = _read_json(
        provenance_path,
        label=f"{scene_id} georeference provenance receipt",
    )
    expected_provenance = {
        "schema_version": SCHEMA_VERSION,
        "state": GEOREFERENCE_AUTHENTICATED_STATE,
        "scene_id": scene_id,
        "horizontal_crs": "EPSG:2154",
        "vertical_datum": "IGN69",
        "axis_order": list(GEOREFERENCE_AXIS_ORDER),
        "local_axes": list(GEOREFERENCE_LOCAL_AXES),
        "axis_mapping": dict(GEOREFERENCE_AXIS_MAPPING),
        "units": "metres",
        "origin_epsg2154_ign69_m": origin,
        "scene_root_sha256": scene_lock["sha256"],
        "build_receipt_sha256": build_receipt["sha256"],
    }
    if provenance != expected_provenance:
        raise TrainingCaptureContractError(
            f"{scene_id} georeference provenance is stale or unauthenticated"
        )
    georeference = {
        **expected_provenance,
        "provenance_receipt": provenance_lock,
    }
    raw_cameras = raw_scene.get("cameras")
    if (
        not isinstance(raw_cameras, list)
        or len(raw_cameras) != CAMERAS_PER_SCENE
    ):
        raise TrainingCaptureContractError(
            f"{scene_id} requires exactly 40 fixed cameras"
        )
    cameras: list[dict[str, object]] = []
    for view_index, raw_camera in enumerate(raw_cameras, start=1):
        camera_id = f"VIEW-{view_index:02d}"
        if (
            not isinstance(raw_camera, Mapping)
            or set(raw_camera)
            != {
                "camera_id",
                "view_class",
                "pose_local",
                "intrinsics",
            }
            or raw_camera.get("camera_id") != camera_id
        ):
            raise TrainingCaptureContractError(
                f"{scene_id} camera {view_index} identity is invalid"
            )
        view_class = str(raw_camera.get("view_class", ""))
        if view_class not in VIEW_CLASSES:
            raise TrainingCaptureContractError(
                f"{scene_id} {camera_id} view_class is invalid"
            )
        source_pose = _normalize_pose(
            raw_camera.get("pose_local"),
            label=f"{scene_id} {camera_id}",
        )
        camera = {
            "camera_id": camera_id,
            "view_class": view_class,
            "pose_local": source_pose,
            # Kept as an explicit, read-only index field for existing plan
            # consumers.  The authoritative temporal pose remains pose_local
            # (position plus orientation); this alias must never be used to
            # author a per-observation camera transform.
            "position_local_m": list(source_pose["position_m"]),
            "intrinsics": _normalize_intrinsics(
                raw_camera.get("intrinsics"),
                label=f"{scene_id} {camera_id}",
            ),
        }
        camera["intrinsics_sha256"] = _canonical_sha256(
            camera["intrinsics"]
        )
        camera["camera_contract_sha256"] = _canonical_sha256(camera)
        cameras.append(camera)
    for left_index, left in enumerate(cameras):
        for right in cameras[left_index + 1 :]:
            if (
                _length(
                    _vector_subtract(
                        left["pose_local"]["position_m"],
                        right["pose_local"]["position_m"],
                    )
                )
                < MIN_CAMERA_SEPARATION_M
            ):
                raise TrainingCaptureContractError(
                    f"{scene_id} fixed camera positions are not distinct"
                )
    view_class_counts = {
        view_class: sum(
            camera["view_class"] == view_class
            for camera in cameras
        )
        for view_class in VIEW_CLASSES
    }
    for view_class, minimum in SPECIAL_VIEW_CLASS_MINIMUMS.items():
        if view_class_counts[view_class] < minimum:
            raise TrainingCaptureContractError(
                f"{scene_id} requires at least {minimum} {view_class} "
                "cameras"
            )
    non_special_count = sum(
        view_class_counts[view_class]
        for view_class in NON_SPECIAL_VIEW_CLASSES
    )
    non_special_diversity = sum(
        view_class_counts[view_class] > 0
        for view_class in NON_SPECIAL_VIEW_CLASSES
    )
    if non_special_count < 34 or non_special_diversity < 3:
        raise TrainingCaptureContractError(
            f"{scene_id} requires 34 diverse non-special viewpoints across "
            "ground, ridge and airborne classes"
        )

    raw_days = raw_scene.get("days")
    if not isinstance(raw_days, list) or len(raw_days) != duration_days:
        raise TrainingCaptureContractError(
            f"{scene_id} day schedule does not match its exact duration"
        )
    days: list[dict[str, object]] = []
    previous_burned_area = -1.0
    for day_index, raw_day in enumerate(raw_days, start=1):
        if (
            not isinstance(raw_day, Mapping)
            or set(raw_day) != {"day_index", "hours"}
            or raw_day.get("day_index") != day_index
        ):
            raise TrainingCaptureContractError(
                f"{scene_id} day {day_index} is malformed"
            )
        raw_hours = raw_day.get("hours")
        if not isinstance(raw_hours, list) or len(raw_hours) != len(
            CAPTURE_HOURS
        ):
            raise TrainingCaptureContractError(
                f"{scene_id} day {day_index} requires 08:00, 14:00 and 20:00"
            )
        hours: list[dict[str, object]] = []
        for capture_hour, raw_hour in zip(
            CAPTURE_HOURS,
            raw_hours,
            strict=True,
        ):
            if (
                not isinstance(raw_hour, Mapping)
                or set(raw_hour) != {"capture_hour", "fire", "weather"}
                or raw_hour.get("capture_hour") != capture_hour
            ):
                raise TrainingCaptureContractError(
                    f"{scene_id} day {day_index} hour schedule drifted"
                )
            hour_value = int(capture_hour[:2])
            condition = {
                "capture_hour": capture_hour,
                "fire": _normalize_fire(
                    raw_hour.get("fire"),
                    label=f"{scene_id} day {day_index} {capture_hour}",
                    expected_time_seconds=(
                        (day_index - 1) * 86_400 + hour_value * 3_600
                    ),
                ),
                "weather": _normalize_weather(
                    raw_hour.get("weather"),
                    label=f"{scene_id} day {day_index} {capture_hour}",
                ),
            }
            aims = [
                _derive_observation_aim(
                    scene_id=scene_id,
                    day_index=day_index,
                    capture_hour=capture_hour,
                    camera=camera,
                    fire=condition["fire"],
                )
                for camera in cameras
            ]
            _validate_convergent_aims(
                aims=aims,
                active_front_radius_m=float(
                    condition["fire"]["active_flame_front_radius_m"]
                ),
                label=f"{scene_id} day {day_index} {capture_hour}",
            )
            condition["camera_aims"] = aims
            burned_area = float(condition["fire"]["burned_area_m2"])
            if burned_area + 1.0e-6 < previous_burned_area:
                raise TrainingCaptureContractError(
                    f"{scene_id} cumulative burned area decreases over time"
                )
            previous_burned_area = burned_area
            condition["fire_weather_sha256"] = _canonical_sha256(condition)
            hours.append(condition)
        days.append({"day_index": day_index, "hours": hours})
    return {
        "scene_id": scene_id,
        "variant_id": variant_id,
        "duration_days": duration_days,
        "scene_root": scene_lock,
        "scene_origin_epsg2154_ign69_m": origin,
        "georeference": georeference,
        "cameras": cameras,
        "days": days,
    }


def _validated_sources(
    *,
    volume_root: Path,
    simulation_allowed_receipt_path: Path,
    source_manifest_path: Path,
) -> dict[str, object]:
    volume = volume_root.resolve()
    if not volume.is_dir() or volume_root.is_symlink():
        raise TrainingCaptureContractError(
            "persistent volume root must be a real directory"
        )
    source_path = _regular_file_below(
        volume_root=volume,
        path=source_manifest_path,
        label="training capture source manifest",
    )
    source = _read_json(
        source_path,
        label="training capture source manifest",
    )
    if (
        source.get("schema_version") != SCHEMA_VERSION
        or source.get("state") != SOURCE_STATE
    ):
        raise TrainingCaptureContractError(
            "training capture source manifest is not locked"
        )
    campaign_id = _stable_id(
        source.get("campaign_id"),
        label="training capture campaign",
    )
    campaign_path, campaign_lock = _locked_file(
        volume_root=volume,
        record=source.get("campaign_index"),
        label="campaign index",
    )
    campaign = _read_json(campaign_path, label="campaign index")
    raw_simulations = campaign.get("simulations")
    if (
        campaign.get("campaign_id") != campaign_id
        or not isinstance(raw_simulations, list)
        or len(raw_simulations) != len(SCENE_IDS)
    ):
        raise TrainingCaptureContractError(
            "campaign index does not contain the exact 20-scene campaign"
        )
    simulation_by_id: dict[str, Mapping[str, object]] = {}
    for raw_simulation in raw_simulations:
        if not isinstance(raw_simulation, Mapping):
            raise TrainingCaptureContractError(
                "campaign simulation slot is malformed"
            )
        simulation_id = str(raw_simulation.get("simulation_id", ""))
        if simulation_id in simulation_by_id:
            raise TrainingCaptureContractError(
                "campaign simulation IDs are duplicated"
            )
        simulation_by_id[simulation_id] = raw_simulation
    if set(simulation_by_id) != set(SCENE_IDS):
        raise TrainingCaptureContractError(
            "campaign simulation IDs differ from SIM-01..SIM-20"
        )
    _gate_payload, gate_lock, gate_artifacts = _validate_gate(
        volume_root=volume,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
        campaign_id=campaign_id,
        campaign_index_sha256=str(campaign_lock["sha256"]),
        raw_gate_artifacts=source.get("simulation_gate_inputs"),
    )
    raw_scenes = source.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != len(SCENE_IDS):
        raise TrainingCaptureContractError(
            "source manifest requires exactly 20 scenes"
        )
    source_by_id: dict[str, object] = {}
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping):
            raise TrainingCaptureContractError(
                "source scene entry is malformed"
            )
        scene_id = str(raw_scene.get("scene_id", ""))
        if scene_id in source_by_id:
            raise TrainingCaptureContractError(
                "source scene IDs are duplicated"
            )
        source_by_id[scene_id] = raw_scene
    if set(source_by_id) != set(SCENE_IDS):
        raise TrainingCaptureContractError(
            "source scene IDs differ from SIM-01..SIM-20"
        )
    scenes = [
        _normalized_scene(
            volume_root=volume,
            raw_scene=source_by_id[scene_id],
            scene_id=scene_id,
            duration_days=duration,
            campaign_binding=simulation_by_id[scene_id],
            build_receipt=gate_artifacts["build_receipt"],
        )
        for scene_id, duration in zip(
            SCENE_IDS,
            SCENE_DURATIONS_DAYS,
            strict=True,
        )
    ]
    if scenes[0]["scene_root"] != gate_artifacts["root_usd"]:
        raise TrainingCaptureContractError(
            "accepted SIM-01 root differs from the capture source root"
        )
    source_lock = _file_record(root=volume, path=source_path)
    return {
        "campaign_id": campaign_id,
        "campaign_index": campaign_lock,
        "simulation_allowed_receipt": gate_lock,
        "simulation_gate_inputs": gate_artifacts,
        "source_manifest": source_lock,
        "scenes": scenes,
    }


def _observation_id(
    *,
    scene_id: str,
    day_index: int,
    view_index: int,
    capture_hour: str,
) -> str:
    return (
        f"{scene_id}-D{day_index:03d}-V{view_index:02d}-"
        f"H{capture_hour.replace(':', '')}"
    )


def _build_observation_record(
    *,
    scene: Mapping[str, object],
    day_index: int,
    condition: Mapping[str, object],
    view_index: int,
    camera: Mapping[str, object],
    sequence_index: int,
) -> dict[str, object]:
    scene_id = str(scene["scene_id"])
    capture_hour = str(condition["capture_hour"])
    hour_token = capture_hour.replace(":", "")
    camera_id = str(camera["camera_id"])
    view_id = f"{scene_id}-{camera_id}"
    day_id = f"DAY-{day_index:03d}"
    aim = condition["camera_aims"][view_index - 1]
    if aim.get("camera_id") != camera_id:
        raise TrainingCaptureContractError(
            "camera aim order differs from the fixed cameras"
        )
    observation_id = _observation_id(
        scene_id=scene_id,
        day_index=day_index,
        view_index=view_index,
        capture_hour=capture_hour,
    )
    stem = (
        f"{scene_id}/day-{day_index:03d}/"
        f"{hour_token}/{camera_id}"
    )
    image_artifacts = {
        modality: {
            "artifact_id": f"{observation_id}:{modality}",
            "modality": modality,
            "path": f"frames/{modality}/{stem}.exr",
        }
        for modality in IMAGE_MODALITIES
    }
    registration_witnesses = {
        modality: {
            "witness_id": f"{observation_id}:{modality}",
            "modality": modality,
            "path": f"registration/{modality}/{stem}.exr",
        }
        for modality in REGISTRATION_WITNESS_MODALITIES
    }
    registration_contract_sha256 = _canonical_sha256(
        {
            "observation_id": observation_id,
            "scene_id": scene_id,
            "variant_id": scene["variant_id"],
            "day_id": day_id,
            "day_index": day_index,
            "view_index": view_index,
            "view_id": view_id,
            "camera_id": camera_id,
            "capture_hour": capture_hour,
            "simulation_time_seconds": condition["fire"][
                "simulation_time_seconds"
            ],
            "pose_local": aim["pose_local"],
            "target_local_m": aim["target_local_m"],
            "intrinsics": camera["intrinsics"],
            "georeference_sha256": _canonical_sha256(
                scene["georeference"]
            ),
            "fire": condition["fire"],
            "weather": condition["weather"],
        }
    )
    core: dict[str, object] = {
        "observation_id": observation_id,
        "sequence_index": sequence_index,
        "scene_id": scene_id,
        "variant_id": scene["variant_id"],
        "duration_days": scene["duration_days"],
        "day_id": day_id,
        "day_index": day_index,
        "view_index": view_index,
        "view_id": view_id,
        "camera_id": camera_id,
        "view_class": camera["view_class"],
        "capture_hour": capture_hour,
        "simulation_time_seconds": condition["fire"][
            "simulation_time_seconds"
        ],
        "pose_local": aim["pose_local"],
        "target_local_m": aim["target_local_m"],
        "target_offset_from_active_centroid_m": aim[
            "target_offset_from_active_centroid_m"
        ],
        "active_front_framing": aim["active_front_framing"],
        "image_artifacts": image_artifacts,
        "registration_witnesses": registration_witnesses,
        "metadata_path": f"metadata/{stem}.json",
        "completion_receipt_path": f"receipts/{stem}.json",
        "scene_root_sha256": scene["scene_root"]["sha256"],
        "camera_contract_sha256": camera["camera_contract_sha256"],
        "intrinsics_sha256": camera["intrinsics_sha256"],
        "fire_weather_sha256": condition["fire_weather_sha256"],
        "aim_contract_sha256": aim["aim_contract_sha256"],
        "registration_contract_sha256": registration_contract_sha256,
    }
    core["observation_contract_sha256"] = _canonical_sha256(core)
    return core


def _build_plan_payload(
    *,
    sources: Mapping[str, object],
    output_root_relative: PurePosixPath,
) -> dict[str, object]:
    observations: list[dict[str, object]] = []
    sequence_index = 0
    scenes = sources["scenes"]
    if not isinstance(scenes, list):
        raise TrainingCaptureContractError("normalized scenes are absent")
    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise TrainingCaptureContractError(
                "normalized scene is malformed"
            )
        scene_id = str(scene["scene_id"])
        for raw_day in scene["days"]:
            day_index = int(raw_day["day_index"])
            for condition in raw_day["hours"]:
                for view_index, camera in enumerate(
                    scene["cameras"],
                    start=1,
                ):
                    sequence_index += 1
                    observations.append(
                        _build_observation_record(
                            scene=scene,
                            day_index=day_index,
                            condition=condition,
                            view_index=view_index,
                            camera=camera,
                            sequence_index=sequence_index,
                        )
                    )
    if sequence_index != EXPECTED_OBSERVATION_COUNT:
        raise TrainingCaptureContractError(
            "campaign schedule did not produce exactly 18,360 observations"
        )
    artifact_contracts = [
        artifact
        for observation in observations
        for artifact in observation["image_artifacts"].values()
    ]
    artifact_ids = {
        str(artifact["artifact_id"])
        for artifact in artifact_contracts
    }
    artifact_paths = {
        str(artifact["path"])
        for artifact in artifact_contracts
    }
    if (
        len(artifact_contracts) != EXPECTED_IMAGE_ARTIFACT_COUNT
        or len(artifact_ids) != EXPECTED_IMAGE_ARTIFACT_COUNT
        or len(artifact_paths) != EXPECTED_IMAGE_ARTIFACT_COUNT
    ):
        raise TrainingCaptureContractError(
            "campaign requires exactly 55,080 distinct image artifacts"
        )
    temporal_camera_contracts: dict[
        tuple[str, str],
        tuple[str, str, int],
    ] = {}
    for observation in observations:
        key = (
            str(observation["scene_id"]),
            str(observation["camera_id"]),
        )
        pose_sha256 = _canonical_sha256(observation["pose_local"])
        intrinsics_sha256 = str(observation["intrinsics_sha256"])
        prior = temporal_camera_contracts.get(key)
        if prior is None:
            temporal_camera_contracts[key] = (
                pose_sha256,
                intrinsics_sha256,
                1,
            )
        elif prior[:2] != (pose_sha256, intrinsics_sha256):
            raise TrainingCaptureContractError(
                "camera pose or intrinsics drifted across capture times"
            )
        else:
            temporal_camera_contracts[key] = (
                prior[0],
                prior[1],
                prior[2] + 1,
            )
    for scene in scenes:
        expected_per_camera = int(scene["duration_days"]) * len(
            CAPTURE_HOURS
        )
        for camera in scene["cameras"]:
            key = (str(scene["scene_id"]), str(camera["camera_id"]))
            if temporal_camera_contracts.get(key, ((), "", 0))[2] != (
                expected_per_camera
            ):
                raise TrainingCaptureContractError(
                    "camera temporal comparison inventory is incomplete"
                )
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "state": PLAN_STATE,
        "campaign_id": sources["campaign_id"],
        "scene_ids": list(SCENE_IDS),
        "scene_durations_days": list(SCENE_DURATIONS_DAYS),
        "camera_count_per_scene": CAMERAS_PER_SCENE,
        "camera_aim_contract": {
            "fixed_components": [
                "position_local_m",
                "orientation_xyzw",
                "intrinsics",
            ],
            "orientation_policy": (
                "fixed_source_pose_openusd_minus_z_forward"
            ),
            "active_front_reference_policy": (
                "per_observation_truth_reference_without_camera_reaim"
            ),
            "view_classes": list(VIEW_CLASSES),
            "special_view_class_minimums": dict(
                SPECIAL_VIEW_CLASS_MINIMUMS
            ),
            "non_special_viewpoint_minimum": 34,
            "non_special_required_classes": sorted(
                NON_SPECIAL_VIEW_CLASSES
            ),
            "target_max_offset_radius_fraction": (
                TARGET_MAX_OFFSET_RADIUS_FRACTION
            ),
            "target_group_diameter_radius_fraction": (
                TARGET_GROUP_DIAMETER_RADIUS_FRACTION
            ),
            "fov_usable_half_angle_fraction": (
                FOV_USABLE_HALF_ANGLE_FRACTION
            ),
        },
        "capture_hours": list(CAPTURE_HOURS),
        "observation_count": EXPECTED_OBSERVATION_COUNT,
        "image_modalities": list(IMAGE_MODALITIES),
        "image_artifact_count": EXPECTED_IMAGE_ARTIFACT_COUNT,
        "image_artifact_inventory_sha256": _canonical_sha256(
            artifact_contracts
        ),
        "capture_formats": {
            modality: dict(CAPTURE_FORMATS[modality])
            for modality in IMAGE_MODALITIES
        },
        "coordinate_contract": COORDINATE_CONTRACT,
        "output_root": output_root_relative.as_posix(),
        "simulation_execution_performed": False,
        "render_execution_performed": False,
        "source_locks": {
            "campaign_index": sources["campaign_index"],
            "simulation_allowed_receipt": sources[
                "simulation_allowed_receipt"
            ],
            "simulation_gate_inputs": sources[
                "simulation_gate_inputs"
            ],
            "source_manifest": sources["source_manifest"],
        },
        "scenes": scenes,
        "observations": observations,
        "observations_sha256": _canonical_sha256(observations),
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _task_index_path(plan_path: Path) -> Path:
    return plan_path.with_name(f"{plan_path.stem}.tasks.sqlite3")


def _task_index_metadata(
    *,
    plan: Mapping[str, object],
    plan_path: Path,
) -> dict[str, str]:
    plan_record = _file_record(root=plan_path.parent, path=plan_path)
    stat = plan_path.stat()
    return {
        "schema_version": str(SCHEMA_VERSION),
        "state": TASK_INDEX_STATE,
        "plan_sha256": str(plan["plan_sha256"]),
        "plan_file_sha256": str(plan_record["sha256"]),
        "plan_file_size_bytes": str(plan_record["size_bytes"]),
        "plan_file_mtime_ns": str(stat.st_mtime_ns),
        "campaign_id": str(plan["campaign_id"]),
        "output_root": str(plan["output_root"]),
        "observation_count": str(EXPECTED_OBSERVATION_COUNT),
        "observations_sha256": str(plan["observations_sha256"]),
        "source_locks_json": json.dumps(
            plan["source_locks"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "capture_formats_json": json.dumps(
            plan["capture_formats"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    }


def _materialize_addressable_task_index(
    *,
    plan: Mapping[str, object],
    plan_path: Path,
) -> Path:
    """Materialize an indexed O(log N) frame-task lookup beside the plan."""

    index_path = _task_index_path(plan_path)
    temporary = index_path.with_name(
        f".{index_path.name}.{uuid.uuid4().hex}.tmp"
    )
    if temporary.exists():
        temporary.unlink()
    scenes = plan.get("scenes")
    observations = plan.get("observations")
    if not isinstance(scenes, list) or not isinstance(observations, list):
        raise TrainingCaptureContractError(
            "task index requires normalized scenes and observations"
        )
    try:
        connection = sqlite3.connect(temporary)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=FULL;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY NOT NULL,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE scenes (
                    scene_id TEXT PRIMARY KEY NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE cameras (
                    scene_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY (scene_id, camera_id)
                ) WITHOUT ROWID;
                CREATE TABLE conditions (
                    scene_id TEXT NOT NULL,
                    day_index INTEGER NOT NULL,
                    capture_hour TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    PRIMARY KEY (scene_id, day_index, capture_hour)
                ) WITHOUT ROWID;
                CREATE TABLE tasks (
                    observation_id TEXT PRIMARY KEY NOT NULL,
                    scene_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    day_index INTEGER NOT NULL,
                    capture_hour TEXT NOT NULL,
                    view_index INTEGER NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    observation_contract_sha256 TEXT NOT NULL,
                    plan_sha256 TEXT NOT NULL
                ) WITHOUT ROWID;
                """
            )
            metadata = _task_index_metadata(
                plan=plan,
                plan_path=plan_path,
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            for scene in scenes:
                if not isinstance(scene, Mapping):
                    raise TrainingCaptureContractError(
                        "task index scene is malformed"
                    )
                scene_core = {
                    key: value
                    for key, value in scene.items()
                    if key not in {"cameras", "days"}
                }
                scene_json = json.dumps(
                    scene_core,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                scene_id = str(scene["scene_id"])
                connection.execute(
                    "INSERT INTO scenes VALUES (?, ?, ?)",
                    (
                        scene_id,
                        scene_json,
                        _canonical_sha256(scene_core),
                    ),
                )
                for camera in scene["cameras"]:
                    camera_json = json.dumps(
                        camera,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    connection.execute(
                        "INSERT INTO cameras VALUES (?, ?, ?, ?)",
                        (
                            scene_id,
                            str(camera["camera_id"]),
                            camera_json,
                            _canonical_sha256(camera),
                        ),
                    )
                for day in scene["days"]:
                    for condition in day["hours"]:
                        condition_json = json.dumps(
                            condition,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        connection.execute(
                            "INSERT INTO conditions VALUES (?, ?, ?, ?, ?)",
                            (
                                scene_id,
                                int(day["day_index"]),
                                str(condition["capture_hour"]),
                                condition_json,
                                _canonical_sha256(condition),
                            ),
                        )
            connection.executemany(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        str(observation["observation_id"]),
                        str(observation["scene_id"]),
                        str(observation["camera_id"]),
                        int(observation["day_index"]),
                        str(observation["capture_hour"]),
                        int(observation["view_index"]),
                        int(observation["sequence_index"]),
                        str(
                            observation[
                                "observation_contract_sha256"
                            ]
                        ),
                        str(plan["plan_sha256"]),
                    )
                    for observation in observations
                ),
            )
            connection.commit()
            observed_count = connection.execute(
                "SELECT COUNT(*) FROM tasks"
            ).fetchone()[0]
            integrity = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            if (
                observed_count != EXPECTED_OBSERVATION_COUNT
                or integrity != "ok"
            ):
                raise TrainingCaptureContractError(
                    "addressable task index failed materialization"
                )
        finally:
            connection.close()
        os.replace(temporary, index_path)
    except (OSError, sqlite3.Error) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise TrainingCaptureContractError(
            "addressable task index could not be materialized"
        ) from exc
    return index_path


def prepare_training_capture_campaign_plan(
    *,
    volume_root: Path,
    simulation_allowed_receipt_path: Path,
    source_manifest_path: Path,
    output_root: Path,
    plan_path: Path,
) -> dict[str, object]:
    """Write the deterministic campaign plan without running any frame."""

    volume = volume_root.resolve()
    raw_output = output_root
    output = raw_output.resolve()
    raw_plan = plan_path
    plan = raw_plan.resolve()
    if (
        raw_output.is_symlink()
        or output == volume
        or not _inside(volume, output)
        or not _inside(output, plan)
        or raw_plan.is_symlink()
        or plan.suffix.casefold() != ".json"
        or (
            plan != output
            and plan.relative_to(output).parts[0]
            in {"frames", "registration", "metadata", "receipts"}
        )
    ):
        raise TrainingCaptureContractError(
            "capture output and plan paths must be regular paths below the "
            "persistent volume"
        )
    if not output.exists():
        if not output.parent.is_dir() or output.parent.is_symlink():
            raise TrainingCaptureContractError(
                "capture output parent is absent or unsafe"
            )
        output.mkdir()
    if not output.is_dir() or output.is_symlink():
        raise TrainingCaptureContractError(
            "capture output root must be a real directory"
        )
    sources = _validated_sources(
        volume_root=volume,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
        source_manifest_path=source_manifest_path,
    )
    expected = _build_plan_payload(
        sources=sources,
        output_root_relative=PurePosixPath(
            output.relative_to(volume).as_posix()
        ),
    )
    if plan.exists():
        existing = _read_json(plan, label="training capture plan")
        if existing != expected:
            raise TrainingCaptureContractError(
                "existing training capture plan differs from current locks"
            )
        _materialize_addressable_task_index(
            plan=expected,
            plan_path=plan,
        )
        return expected
    if any(
        (output / name).exists()
        for name in ("frames", "registration", "metadata", "receipts")
    ):
        raise TrainingCaptureContractError(
            "unbound capture outputs exist before the campaign plan"
        )
    _atomic_write_json(plan, expected)
    _materialize_addressable_task_index(
        plan=expected,
        plan_path=plan,
    )
    return expected


def _validated_plan(
    *,
    volume_root: Path,
    plan_path: Path,
    simulation_allowed_receipt_path: Path,
) -> tuple[dict[str, object], Path]:
    volume = volume_root.resolve()
    path = _regular_file_below(
        volume_root=volume,
        path=plan_path,
        label="training capture plan",
    )
    plan = _read_json(path, label="training capture plan")
    source_locks = plan.get("source_locks")
    if not isinstance(source_locks, Mapping):
        raise TrainingCaptureContractError(
            "training capture plan lacks source locks"
        )
    source_relative = _safe_relative_path(
        source_locks.get("source_manifest", {}).get("path")
        if isinstance(source_locks.get("source_manifest"), Mapping)
        else "",
        label="capture source manifest",
    )
    output_relative = _safe_relative_path(
        plan.get("output_root"),
        label="capture output root",
    )
    output_root = volume.joinpath(*output_relative.parts).resolve()
    if not output_root.is_dir() or output_root.is_symlink():
        raise TrainingCaptureContractError(
            "training capture output root is absent or unsafe"
        )
    sources = _validated_sources(
        volume_root=volume,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
        source_manifest_path=volume.joinpath(*source_relative.parts),
    )
    expected = _build_plan_payload(
        sources=sources,
        output_root_relative=output_relative,
    )
    if plan != expected:
        raise TrainingCaptureContractError(
            "training capture plan is stale, tampered or structurally invalid"
        )
    return expected, output_root


def _validated_addressed_task(
    *,
    volume_root: Path,
    plan_path: Path,
    simulation_allowed_receipt_path: Path,
    observation_id: str,
) -> tuple[
    dict[str, object],
    Path,
    Mapping[str, object],
    dict[str, object],
]:
    """Resolve one task by SQLite primary key without rebuilding the campaign."""

    volume = volume_root.resolve()
    plan_file = _regular_file_below(
        volume_root=volume,
        path=plan_path,
        label="training capture plan",
    )
    index_file = _regular_file_below(
        volume_root=volume,
        path=_task_index_path(plan_file),
        label="addressable training capture task index",
    )
    try:
        connection = sqlite3.connect(index_file)
        connection.execute("PRAGMA query_only=ON")
        try:
            metadata = dict(
                connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            )
            row = connection.execute(
                """
                SELECT scene_id, camera_id, day_index, capture_hour,
                       view_index, sequence_index,
                       observation_contract_sha256, plan_sha256
                FROM tasks
                WHERE observation_id = ?
                """,
                (observation_id,),
            ).fetchone()
            if row is None:
                raise TrainingCaptureContractError(
                    "requested capture observation is unknown"
                )
            (
                scene_id,
                camera_id,
                day_index,
                capture_hour,
                view_index,
                sequence_index,
                observation_contract_sha256,
                row_plan_sha256,
            ) = row
            scene_id = str(scene_id)
            camera_id = str(camera_id)
            day_index = int(day_index)
            capture_hour = str(capture_hour)
            view_index = int(view_index)
            sequence_index = int(sequence_index)
            if (
                _observation_id(
                    scene_id=scene_id,
                    day_index=day_index,
                    view_index=view_index,
                    capture_hour=capture_hour,
                )
                != observation_id
                or row_plan_sha256 != metadata.get("plan_sha256")
                or not _SHA256.fullmatch(
                    str(observation_contract_sha256)
                )
            ):
                raise TrainingCaptureContractError(
                    "addressed capture task is stale or tampered"
                )
            scene_row = connection.execute(
                """
                SELECT payload_json, payload_sha256 FROM scenes
                WHERE scene_id = ?
                """,
                (scene_id,),
            ).fetchone()
            camera_row = connection.execute(
                """
                SELECT payload_json, payload_sha256 FROM cameras
                WHERE scene_id = ? AND camera_id = ?
                """,
                (scene_id, camera_id),
            ).fetchone()
            condition_row = connection.execute(
                """
                SELECT payload_json, payload_sha256 FROM conditions
                WHERE scene_id = ? AND day_index = ? AND capture_hour = ?
                """,
                (scene_id, day_index, capture_hour),
            ).fetchone()
        finally:
            connection.close()
    except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        raise TrainingCaptureContractError(
            "addressable training capture task index is invalid"
        ) from exc
    plan_stat = plan_file.stat()
    expected_metadata_fields = {
        "schema_version",
        "state",
        "plan_sha256",
        "plan_file_sha256",
        "plan_file_size_bytes",
        "plan_file_mtime_ns",
        "campaign_id",
        "output_root",
        "observation_count",
        "observations_sha256",
        "source_locks_json",
        "capture_formats_json",
    }
    if (
        set(metadata) != expected_metadata_fields
        or metadata.get("schema_version") != str(SCHEMA_VERSION)
        or metadata.get("state") != TASK_INDEX_STATE
        or metadata.get("observation_count")
        != str(EXPECTED_OBSERVATION_COUNT)
        or metadata.get("plan_file_size_bytes")
        != str(plan_stat.st_size)
        or metadata.get("plan_file_mtime_ns")
        != str(plan_stat.st_mtime_ns)
        or not _SHA256.fullmatch(metadata.get("plan_file_sha256", ""))
        or not _SHA256.fullmatch(metadata.get("plan_sha256", ""))
        or not _SHA256.fullmatch(metadata.get("observations_sha256", ""))
    ):
        raise TrainingCaptureContractError(
            "addressable task index is not bound to the current plan file"
        )
    if scene_row is None or camera_row is None or condition_row is None:
        raise TrainingCaptureContractError(
            "addressed task lacks its scene, camera or condition binding"
        )

    def decoded_index_record(
        row_value: Sequence[object],
        *,
        label: str,
    ) -> dict[str, object]:
        try:
            payload = json.loads(str(row_value[0]))
        except json.JSONDecodeError as exc:
            raise TrainingCaptureContractError(
                f"{label} indexed payload is invalid"
            ) from exc
        if (
            not isinstance(payload, dict)
            or _canonical_sha256(payload) != row_value[1]
        ):
            raise TrainingCaptureContractError(
                f"{label} indexed payload is stale or tampered"
            )
        return payload

    scene_core = decoded_index_record(scene_row, label="scene")
    camera = decoded_index_record(camera_row, label="camera")
    condition = decoded_index_record(condition_row, label="condition")
    try:
        source_locks = json.loads(metadata["source_locks_json"])
        capture_formats = json.loads(metadata["capture_formats_json"])
    except json.JSONDecodeError as exc:
        raise TrainingCaptureContractError(
            "addressable task plan context is invalid"
        ) from exc
    if not isinstance(source_locks, dict) or not isinstance(
        capture_formats,
        dict,
    ):
        raise TrainingCaptureContractError(
            "addressable task plan context is malformed"
        )
    current_receipt = simulation_allowed_receipt_path.resolve()
    for label, record in (
        ("campaign index", source_locks.get("campaign_index")),
        (
            "simulation allowed receipt",
            source_locks.get("simulation_allowed_receipt"),
        ),
        ("source manifest", source_locks.get("source_manifest")),
    ):
        try:
            locked_path, _locked = _locked_file(
                volume_root=volume,
                record=record,
                label=label,
            )
        except TrainingCaptureContractError as exc:
            if label == "simulation allowed receipt":
                raise TrainingCaptureContractError(
                    "capture is blocked without a current "
                    "FIRE_SIMULATION_ALLOWED receipt bound to this plan"
                ) from exc
            raise
        if label == "simulation allowed receipt" and (
            locked_path != current_receipt
        ):
            raise TrainingCaptureContractError(
                "addressed task uses another simulation allowed receipt"
            )
    raw_gate_inputs = source_locks.get("simulation_gate_inputs")
    if not isinstance(raw_gate_inputs, Mapping) or set(
        raw_gate_inputs
    ) != set(GATE_ARTIFACT_KEYS):
        raise TrainingCaptureContractError(
            "addressed task simulation gate locks are incomplete"
        )
    for key in GATE_ARTIFACT_KEYS:
        _locked_file(
            volume_root=volume,
            record=raw_gate_inputs[key],
            label=f"simulation gate {key}",
        )
    output_relative = _safe_relative_path(
        metadata["output_root"],
        label="capture output root",
    )
    output_root = volume.joinpath(*output_relative.parts).resolve()
    if not output_root.is_dir() or output_root.is_symlink():
        raise TrainingCaptureContractError(
            "training capture output root is absent or unsafe"
        )
    scene = {
        **scene_core,
        "cameras": [camera],
        "days": [
            {
                "day_index": day_index,
                "hours": [condition],
            }
        ],
    }
    plan_context: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "state": PLAN_STATE,
        "campaign_id": metadata["campaign_id"],
        "plan_sha256": metadata["plan_sha256"],
        "output_root": metadata["output_root"],
        "capture_formats": capture_formats,
        "source_locks": source_locks,
        "scenes": [scene],
        "addressable_task_index": {
            "state": TASK_INDEX_STATE,
            "path": index_file.relative_to(volume).as_posix(),
            "plan_file_sha256": metadata["plan_file_sha256"],
            "observations_sha256": metadata["observations_sha256"],
            "lookup": "sqlite_primary_key",
        },
    }
    observation = _build_observation_record(
        scene=scene,
        day_index=day_index,
        condition=condition,
        view_index=view_index,
        camera=camera,
        sequence_index=sequence_index,
    )
    if (
        observation["observation_id"] != observation_id
        or observation["observation_contract_sha256"]
        != observation_contract_sha256
    ):
        raise TrainingCaptureContractError(
            "addressed capture task does not reconstruct its planned hash"
        )
    expected_metadata = _metadata_contract(
        plan=plan_context,
        observation=observation,
    )
    return plan_context, output_root, observation, expected_metadata


def _observation_lookup(
    plan: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    observations = plan.get("observations")
    if (
        not isinstance(observations, list)
        or len(observations) != EXPECTED_OBSERVATION_COUNT
    ):
        raise TrainingCaptureContractError(
            "capture plan does not contain exactly 18,360 observations"
        )
    result: dict[str, Mapping[str, object]] = {}
    for record in observations:
        if not isinstance(record, Mapping):
            raise TrainingCaptureContractError(
                "capture observation is malformed"
            )
        identifier = str(record.get("observation_id", ""))
        if identifier in result:
            raise TrainingCaptureContractError(
                "capture observation IDs are duplicated"
            )
        result[identifier] = record
    return result


def _scene_camera_condition(
    *,
    plan: Mapping[str, object],
    observation: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    scenes = plan.get("scenes")
    if not isinstance(scenes, list):
        raise TrainingCaptureContractError("capture plan scenes are absent")
    matching_scenes = [
        scene
        for scene in scenes
        if isinstance(scene, Mapping)
        and scene.get("scene_id") == observation.get("scene_id")
    ]
    if len(matching_scenes) != 1:
        raise TrainingCaptureContractError(
            "capture observation scene binding is invalid"
        )
    scene = matching_scenes[0]
    matching_cameras = [
        camera
        for camera in scene["cameras"]
        if camera.get("camera_id") == observation.get("camera_id")
    ]
    matching_days = [
        day
        for day in scene["days"]
        if day.get("day_index") == observation.get("day_index")
    ]
    if len(matching_cameras) != 1 or len(matching_days) != 1:
        raise TrainingCaptureContractError(
            "capture observation camera/day binding is invalid"
        )
    matching_conditions = [
        condition
        for condition in matching_days[0]["hours"]
        if condition.get("capture_hour")
        == observation.get("capture_hour")
    ]
    if len(matching_conditions) != 1:
        raise TrainingCaptureContractError(
            "capture observation hour binding is invalid"
        )
    return scene, matching_cameras[0], matching_conditions[0]


def _metadata_contract(
    *,
    plan: Mapping[str, object],
    observation: Mapping[str, object],
) -> dict[str, object]:
    scene, camera, condition = _scene_camera_condition(
        plan=plan,
        observation=observation,
    )
    local_pose = observation["pose_local"]
    georeference = scene["georeference"]
    origin = georeference["origin_epsg2154_ign69_m"]
    local_position = local_pose["position_m"]
    absolute_position = [
        float(origin[index]) + float(local_position[index])
        for index in range(3)
    ]
    local_target = observation["target_local_m"]
    absolute_target = [
        float(origin[index]) + float(local_target[index])
        for index in range(3)
    ]
    active_centroid = condition["fire"][
        "active_flame_centroid_local_m"
    ]
    absolute_active_centroid = [
        float(origin[index]) + float(active_centroid[index])
        for index in range(3)
    ]
    active_front_radius = float(
        condition["fire"]["active_flame_front_radius_m"]
    )
    intrinsics = camera["intrinsics"]
    camera_calibration = {
        "projection": "pinhole_perspective",
        "resolution_px": [
            int(intrinsics["width_px"]),
            int(intrinsics["height_px"]),
        ],
        "focal_length_px": [
            float(intrinsics["fx_px"]),
            float(intrinsics["fy_px"]),
        ],
        "focal_length_mm": float(intrinsics["focal_length_mm"]),
        "field_of_view_degrees": {
            "horizontal": math.degrees(
                2.0
                * math.atan(
                    float(intrinsics["width_px"])
                    / (2.0 * float(intrinsics["fx_px"]))
                )
            ),
            "vertical": math.degrees(
                2.0
                * math.atan(
                    float(intrinsics["height_px"])
                    / (2.0 * float(intrinsics["fy_px"]))
                )
            ),
        },
        "principal_point_px": [
            float(intrinsics["cx_px"]),
            float(intrinsics["cy_px"]),
        ],
        "sensor_aperture_mm": [
            float(intrinsics["horizontal_aperture_mm"]),
            float(intrinsics["vertical_aperture_mm"]),
        ],
        "f_stop": float(intrinsics["f_stop"]),
        "near_far_clip_m": [
            float(intrinsics["near_clip_m"]),
            float(intrinsics["far_clip_m"]),
        ],
    }
    local_pose_contract = {
        **local_pose,
        "axes": "OpenUSD_Z_up_camera_minus_Z_forward_plus_Y_up",
        "units": "metres",
    }
    absolute_pose_contract = {
        "position_m": absolute_position,
        "orientation_xyzw": local_pose["orientation_xyzw"],
        "horizontal_crs": "EPSG:2154",
        "vertical_datum": "IGN69",
        "axis_order": [
            "easting_m",
            "northing_m",
            "altitude_m",
        ],
    }
    look_at_contract = {
        "semantic": "active_flame_front_truth_reference",
        "camera_control": "not_a_camera_reaim",
        "local": {
            "position_m": local_target,
            "axes": "scene_local_Z_up",
            "units": "metres",
        },
        "epsg2154_ign69": {
            "position_m": absolute_target,
            "horizontal_crs": "EPSG:2154",
            "vertical_datum": "IGN69",
            "axis_order": [
                "easting_m",
                "northing_m",
                "altitude_m",
            ],
        },
        "active_centroid_local": {"position_m": active_centroid},
        "active_centroid_epsg2154_ign69": {
            "position_m": absolute_active_centroid,
            "horizontal_crs": "EPSG:2154",
            "vertical_datum": "IGN69",
        },
        "offset_from_active_centroid_m": observation[
            "target_offset_from_active_centroid_m"
        ],
        "active_front_radius_m": active_front_radius,
        "group_convergence_tolerance_m": (
            active_front_radius
            * TARGET_GROUP_DIAMETER_RADIUS_FRACTION
        ),
    }
    source_locks = plan["source_locks"]
    camera_projection_contract = {
        "pose_local": local_pose_contract,
        "intrinsics": intrinsics,
        "camera_calibration": camera_calibration,
        "resolution_px": camera_calibration["resolution_px"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "state": FRAME_RENDERED_STATE,
        "campaign_id": plan["campaign_id"],
        "observation_id": observation["observation_id"],
        "scene_id": observation["scene_id"],
        "variant_id": observation["variant_id"],
        "day_id": observation["day_id"],
        "day_index": observation["day_index"],
        "view_index": observation["view_index"],
        "view_id": observation["view_id"],
        "camera_id": observation["camera_id"],
        "view_class": observation["view_class"],
        "capture_hour": observation["capture_hour"],
        "timestamp": {
            "time_basis": "simulation_elapsed_seconds",
            "simulation_time_seconds": observation[
                "simulation_time_seconds"
            ],
            "capture_hour": observation["capture_hour"],
        },
        "coordinate_contract": COORDINATE_CONTRACT,
        "georeference": georeference,
        "pose": {
            "local": local_pose_contract,
            "epsg2154_ign69": absolute_pose_contract,
        },
        "target": look_at_contract,
        "look_at": look_at_contract,
        "active_front_framing": observation["active_front_framing"],
        "intrinsics": intrinsics,
        "camera_calibration": camera_calibration,
        "fire": condition["fire"],
        "weather": condition["weather"],
        "image_artifacts": observation["image_artifacts"],
        "registration_witnesses": observation["registration_witnesses"],
        "image_modalities": list(IMAGE_MODALITIES),
        "capture_formats": plan["capture_formats"],
        "pixel_registration": {
            "state": "REQUIRED_IDENTICAL_POSE_TIME_INTRINSICS_RESOLUTION",
            "registration_contract_sha256": observation[
                "registration_contract_sha256"
            ],
            "camera_projection_contract_sha256": _canonical_sha256(
                camera_projection_contract
            ),
            "witness_format": dict(REGISTRATION_WITNESS_FORMAT),
            "witness_modalities": list(
                REGISTRATION_WITNESS_MODALITIES
            ),
            "artifact_count": len(IMAGE_MODALITIES),
        },
        "negative_derivation": {
            "algorithm": "one_minus_clamped_linear_ap1_v1",
            "formula": "negative_rgb=1-clamp(normal_rgb,0,1)",
            "source_artifact_id": observation["image_artifacts"][
                "normal_rgb"
            ]["artifact_id"],
            "output_artifact_id": observation["image_artifacts"][
                "negative"
            ]["artifact_id"],
            "new_viewpoint": False,
        },
        "thermal_render_contract": {
            "renderer_input": (
                "fire_temperature_heat_field_plus_surface_emissivity"
            ),
            "raw_channel": "T",
            "temperature_unit": "kelvin",
            "pixel_type": "float32",
            "rgb_colorization_forbidden": True,
            "requires_hotspot_in_active_front_and_fov": True,
        },
        "hashes": {
            "plan_sha256": plan["plan_sha256"],
            "observation_contract_sha256": observation[
                "observation_contract_sha256"
            ],
            "simulation_allowed_receipt_sha256": source_locks[
                "simulation_allowed_receipt"
            ]["sha256"],
            "source_manifest_sha256": source_locks[
                "source_manifest"
            ]["sha256"],
            "campaign_index_sha256": source_locks[
                "campaign_index"
            ]["sha256"],
            "pending_review_sha256": source_locks[
                "simulation_gate_inputs"
            ]["pending_review"]["sha256"],
            "editor_opened_sha256": source_locks[
                "simulation_gate_inputs"
            ]["editor_opened"]["sha256"],
            "editor_acceptance_sha256": source_locks[
                "simulation_gate_inputs"
            ]["editor_acceptance"]["sha256"],
            "runtime_preflight_sha256": source_locks[
                "simulation_gate_inputs"
            ]["runtime_preflight"]["sha256"],
            "asset_manifest_sha256": source_locks[
                "simulation_gate_inputs"
            ]["asset_manifest"]["sha256"],
            "build_receipt_sha256": source_locks[
                "simulation_gate_inputs"
            ]["build_receipt"]["sha256"],
            "scene_auto_validation_sha256": source_locks[
                "simulation_gate_inputs"
            ]["scene_auto_validation"]["sha256"],
            "scene_root_sha256": observation["scene_root_sha256"],
            "camera_contract_sha256": observation[
                "camera_contract_sha256"
            ],
            "intrinsics_sha256": observation["intrinsics_sha256"],
            "fire_weather_sha256": observation[
                "fire_weather_sha256"
            ],
            "aim_contract_sha256": observation[
                "aim_contract_sha256"
            ],
            "registration_contract_sha256": observation[
                "registration_contract_sha256"
            ],
        },
    }


def _output_path(
    *,
    output_root: Path,
    relative_value: object,
    label: str,
) -> Path:
    relative = _safe_relative_path(relative_value, label=label)
    path = output_root.joinpath(*relative.parts).resolve()
    if not _inside(output_root, path):
        raise TrainingCaptureContractError(f"{label} escapes output root")
    return path


def _image_artifact_contracts(
    observation: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = observation.get("image_artifacts")
    if not isinstance(raw, Mapping) or set(raw) != set(IMAGE_MODALITIES):
        raise TrainingCaptureContractError(
            "observation does not declare exactly three image modalities"
        )
    result: dict[str, Mapping[str, object]] = {}
    for modality in IMAGE_MODALITIES:
        artifact = raw.get(modality)
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"artifact_id", "modality", "path"}
            or artifact.get("modality") != modality
            or artifact.get("artifact_id")
            != f"{observation['observation_id']}:{modality}"
        ):
            raise TrainingCaptureContractError(
                f"{modality} image artifact contract is malformed"
            )
        _safe_relative_path(
            artifact.get("path"),
            label=f"{modality} image path",
        )
        result[modality] = artifact
    if len({str(item["path"]) for item in result.values()}) != len(
        IMAGE_MODALITIES
    ):
        raise TrainingCaptureContractError(
            "co-registered image modality paths must be distinct"
        )
    return result


def _registration_witness_contracts(
    observation: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = observation.get("registration_witnesses")
    if (
        not isinstance(raw, Mapping)
        or set(raw) != set(REGISTRATION_WITNESS_MODALITIES)
    ):
        raise TrainingCaptureContractError(
            "observation does not declare both registration AOV witnesses"
        )
    result: dict[str, Mapping[str, object]] = {}
    for modality in REGISTRATION_WITNESS_MODALITIES:
        witness = raw.get(modality)
        if (
            not isinstance(witness, Mapping)
            or set(witness) != {"witness_id", "modality", "path"}
            or witness.get("modality") != modality
            or witness.get("witness_id")
            != f"{observation['observation_id']}:{modality}"
        ):
            raise TrainingCaptureContractError(
                f"{modality} registration witness contract is malformed"
            )
        _safe_relative_path(
            witness.get("path"),
            label=f"{modality} registration witness path",
        )
        result[modality] = witness
    if len({str(item["path"]) for item in result.values()}) != 2:
        raise TrainingCaptureContractError(
            "registration AOV witness paths must be distinct"
        )
    return result


def _image_paths(
    *,
    output_root: Path,
    observation: Mapping[str, object],
) -> dict[str, Path]:
    return {
        modality: _output_path(
            output_root=output_root,
            relative_value=artifact["path"],
            label=f"{modality} captured image",
        )
        for modality, artifact in _image_artifact_contracts(
            observation
        ).items()
    }


def _registration_witness_paths(
    *,
    output_root: Path,
    observation: Mapping[str, object],
) -> dict[str, Path]:
    return {
        modality: _output_path(
            output_root=output_root,
            relative_value=witness["path"],
            label=f"{modality} registration AOV witness",
        )
        for modality, witness in _registration_witness_contracts(
            observation
        ).items()
    }


def _registered_image_record(
    *,
    output_root: Path,
    path: Path,
    artifact: Mapping[str, object],
    registration_contract_sha256: str,
    expected_metadata: Mapping[str, object],
) -> dict[str, object]:
    modality = str(artifact["modality"])
    registration = {
        "scene_id": expected_metadata["scene_id"],
        "variant_id": expected_metadata["variant_id"],
        "day_id": expected_metadata["day_id"],
        "day_index": expected_metadata["day_index"],
        "view_id": expected_metadata["view_id"],
        "view_index": expected_metadata["view_index"],
        "camera_id": expected_metadata["camera_id"],
        "observation_id": expected_metadata["observation_id"],
        "timestamp": expected_metadata["timestamp"],
        "capture_hour": expected_metadata["capture_hour"],
        "fire_state": expected_metadata["fire"]["state"],
        "active_flame_centroid_local_m": expected_metadata["fire"][
            "active_flame_centroid_local_m"
        ],
        "active_flame_front_radius_m": expected_metadata["fire"][
            "active_flame_front_radius_m"
        ],
        "fire": expected_metadata["fire"],
        "weather": expected_metadata["weather"],
        "camera_pose_local": expected_metadata["pose"]["local"],
        "camera_pose_epsg2154_ign69": expected_metadata["pose"][
            "epsg2154_ign69"
        ],
        "look_at_local": expected_metadata["look_at"]["local"],
        "look_at_epsg2154_ign69": expected_metadata["look_at"][
            "epsg2154_ign69"
        ],
        "intrinsics": expected_metadata["intrinsics"],
        "camera_calibration": expected_metadata["camera_calibration"],
        "modality": modality,
    }
    return {
        "artifact_id": artifact["artifact_id"],
        "modality": modality,
        "registration_contract_sha256": registration_contract_sha256,
        "registration": registration,
        **_file_record(root=output_root, path=path),
    }


def _validate_negative_pixels(
    *,
    normal_capture: Mapping[str, object],
    negative_capture: Mapping[str, object],
) -> None:
    normal_chunks = normal_capture.get("decoded_chunks")
    negative_chunks = negative_capture.get("decoded_chunks")
    if (
        not isinstance(normal_chunks, list)
        or not isinstance(negative_chunks, list)
        or len(normal_chunks) != len(negative_chunks)
    ):
        raise TrainingCaptureContractError(
            "negative and normal RGB chunk inventories are not aligned"
        )
    for normal_chunk, negative_chunk in zip(
        normal_chunks,
        negative_chunks,
        strict=True,
    ):
        if (
            normal_chunk.get("y_coordinate")
            != negative_chunk.get("y_coordinate")
            or normal_chunk.get("scanline_count")
            != negative_chunk.get("scanline_count")
        ):
            raise TrainingCaptureContractError(
                "negative and normal RGB scanlines are not pixel-aligned"
            )
        normal = np.frombuffer(
            normal_chunk["samples"],
            dtype="<f2",
        ).astype(np.float32)
        negative = np.frombuffer(
            negative_chunk["samples"],
            dtype="<f2",
        ).astype(np.float32)
        if (
            normal.shape != negative.shape
            or not np.isfinite(normal).all()
            or not np.isfinite(negative).all()
            or not np.allclose(
                negative,
                1.0 - np.clip(normal, 0.0, 1.0),
                rtol=0.0,
                atol=2.0e-3,
            )
        ):
            raise TrainingCaptureContractError(
                "negative pixels are not the deterministic transform of "
                "the final normal RGB pixels"
            )


def _project_local_point_to_pixel(
    *,
    point_local_m: Sequence[float],
    pose_local: Mapping[str, object],
    intrinsics: Mapping[str, object],
) -> tuple[float, float, float]:
    offset = _vector_subtract(
        point_local_m,
        pose_local["position_m"],
    )
    orientation = pose_local["orientation_xyzw"]
    inverse_orientation = [
        -float(orientation[0]),
        -float(orientation[1]),
        -float(orientation[2]),
        float(orientation[3]),
    ]
    camera_local = _rotate_by_quaternion(offset, inverse_orientation)
    depth = -float(camera_local[2])
    if depth <= 0.0:
        raise TrainingCaptureContractError(
            "thermal hotspot is behind the registered camera"
        )
    pixel_x = (
        float(intrinsics["fx_px"]) * float(camera_local[0]) / depth
        + float(intrinsics["cx_px"])
    )
    pixel_y = (
        float(intrinsics["cy_px"])
        - float(intrinsics["fy_px"]) * float(camera_local[1]) / depth
    )
    return pixel_x, pixel_y, depth


def _validate_thermal_evidence(
    *,
    expected_metadata: Mapping[str, object],
    evidence: object,
    thermal_capture: Mapping[str, object],
) -> dict[str, object]:
    required_fields = {
        "render_source",
        "temperature_unit",
        "heat_field_sha256",
        "surface_emissivity",
        "min_temperature_k",
        "max_temperature_k",
        "hotspot_temperature_k",
        "hotspot_pixel_xy",
        "hotspot_local_m",
        "hotspot_epsg2154_ign69_m",
        "depth_at_hotspot_m",
        "hotspot_in_active_front",
        "hotspot_in_fov",
        "registration_contract_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_fields:
        raise TrainingCaptureContractError(
            "thermal hotspot evidence is incomplete"
        )
    if (
        evidence.get("render_source")
        != "fire_temperature_heat_field_plus_surface_emissivity"
        or evidence.get("temperature_unit") != "kelvin"
        or evidence.get("hotspot_in_active_front") is not True
        or evidence.get("hotspot_in_fov") is not True
        or evidence.get("registration_contract_sha256")
        != expected_metadata["pixel_registration"][
            "registration_contract_sha256"
        ]
    ):
        raise TrainingCaptureContractError(
            "thermal hotspot is not bound to the registered heat-field render"
        )
    _require_sha256(
        evidence.get("heat_field_sha256"),
        label="thermal heat field",
    )
    emissivity = _finite(
        evidence.get("surface_emissivity"),
        label="thermal surface emissivity",
    )
    minimum = _finite(
        evidence.get("min_temperature_k"),
        label="thermal minimum temperature",
    )
    maximum = _finite(
        evidence.get("max_temperature_k"),
        label="thermal maximum temperature",
    )
    hotspot_temperature = _finite(
        evidence.get("hotspot_temperature_k"),
        label="thermal hotspot temperature",
    )
    if (
        not 0.0 < emissivity <= 1.0
        or minimum <= 0.0
        or maximum <= minimum
        or not math.isclose(
            hotspot_temperature,
            maximum,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
    ):
        raise TrainingCaptureContractError(
            "thermal temperature or emissivity statistics are invalid"
        )
    hotspot_local = _normalize_position(
        evidence.get("hotspot_local_m"),
        label="thermal hotspot local coordinate",
    )
    hotspot_absolute = _normalize_position(
        evidence.get("hotspot_epsg2154_ign69_m"),
        label="thermal hotspot EPSG:2154 coordinate",
    )
    origin = _vector_subtract(
        expected_metadata["pose"]["epsg2154_ign69"]["position_m"],
        expected_metadata["pose"]["local"]["position_m"],
    )
    expected_absolute = _vector_add(origin, hotspot_local)
    if any(
        not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-5,
        )
        for observed, expected in zip(
            hotspot_absolute,
            expected_absolute,
            strict=True,
        )
    ):
        raise TrainingCaptureContractError(
            "thermal hotspot local and EPSG:2154 coordinates disagree"
        )
    active_centroid = expected_metadata["fire"][
        "active_flame_centroid_local_m"
    ]
    active_radius = float(
        expected_metadata["fire"]["active_flame_front_radius_m"]
    )
    if (
        _length(_vector_subtract(hotspot_local, active_centroid))
        > active_radius + 1.0e-6
    ):
        raise TrainingCaptureContractError(
            "thermal hotspot is outside the active flame front"
        )
    pixel = evidence.get("hotspot_pixel_xy")
    if (
        not isinstance(pixel, list)
        or len(pixel) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel)
    ):
        raise TrainingCaptureContractError(
            "thermal hotspot pixel coordinate is invalid"
        )
    projected_x, projected_y, projected_depth = (
        _project_local_point_to_pixel(
            point_local_m=hotspot_local,
            pose_local=expected_metadata["pose"]["local"],
            intrinsics=expected_metadata["intrinsics"],
        )
    )
    width = int(expected_metadata["intrinsics"]["width_px"])
    height = int(expected_metadata["intrinsics"]["height_px"])
    depth = _finite(
        evidence.get("depth_at_hotspot_m"),
        label="thermal hotspot depth",
    )
    if (
        not 0 <= pixel[0] < width
        or not 0 <= pixel[1] < height
        or abs(float(pixel[0]) - projected_x) > 1.0
        or abs(float(pixel[1]) - projected_y) > 1.0
        or not math.isclose(
            depth,
            projected_depth,
            rel_tol=1.0e-5,
            abs_tol=1.0e-3,
        )
    ):
        raise TrainingCaptureContractError(
            "thermal hotspot coordinate is outside the registered FOV"
        )
    decoded_chunks = thermal_capture.get("decoded_chunks")
    if not isinstance(decoded_chunks, list) or not decoded_chunks:
        raise TrainingCaptureContractError(
            "thermal raw temperature samples are absent"
        )
    observed_minimum = math.inf
    observed_maximum = -math.inf
    observed_hotspot: float | None = None
    data_window = thermal_capture["data_window"]
    xmin, ymin = int(data_window[0]), int(data_window[1])
    for chunk in decoded_chunks:
        values = np.frombuffer(chunk["samples"], dtype="<f4")
        rows = int(chunk["scanline_count"])
        if values.size != width * rows or not np.isfinite(values).all():
            raise TrainingCaptureContractError(
                "thermal raw temperature samples are malformed"
            )
        grid = values.reshape((rows, width))
        observed_minimum = min(observed_minimum, float(grid.min()))
        observed_maximum = max(observed_maximum, float(grid.max()))
        local_row = pixel[1] - int(chunk["y_coordinate"])
        local_column = pixel[0] - xmin
        if 0 <= local_row < rows and 0 <= local_column < width:
            observed_hotspot = float(grid[local_row, local_column])
    if (
        ymin != 0
        or observed_hotspot is None
        or not math.isclose(
            observed_minimum,
            minimum,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
        or not math.isclose(
            observed_maximum,
            maximum,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
        or not math.isclose(
            observed_hotspot,
            hotspot_temperature,
            rel_tol=0.0,
            abs_tol=1.0e-3,
        )
    ):
        raise TrainingCaptureContractError(
            "thermal metadata does not match raw temperature pixels"
        )
    return dict(evidence)


def _validate_rendered_registration_evidence(
    *,
    output_root: Path,
    observation: Mapping[str, object],
    expected_metadata: Mapping[str, object],
    evidence: object,
) -> dict[str, object]:
    """Prove RGB/thermal co-registration from two rendered geometry-ID AOVs."""

    if not isinstance(evidence, Mapping):
        raise TrainingCaptureContractError(
            "rendered RGB/thermal registration evidence is absent"
        )
    expected_fields = {
        "render_session_id",
        "registration_contract_sha256",
        "camera_projection_contract_sha256",
        "render_products",
        "pixel_identical_geometry_id",
    }
    if set(evidence) != expected_fields:
        raise TrainingCaptureContractError(
            "rendered RGB/thermal registration evidence is incomplete"
        )
    render_session_id = _stable_id(
        evidence.get("render_session_id"),
        label="registration render session",
    )
    expected_registration = expected_metadata["pixel_registration"][
        "registration_contract_sha256"
    ]
    expected_projection = expected_metadata["pixel_registration"][
        "camera_projection_contract_sha256"
    ]
    if (
        evidence.get("registration_contract_sha256")
        != expected_registration
        or evidence.get("camera_projection_contract_sha256")
        != expected_projection
        or evidence.get("pixel_identical_geometry_id") is not True
    ):
        raise TrainingCaptureContractError(
            "rendered registration evidence is bound to another camera"
        )
    render_products = evidence.get("render_products")
    expected_product_modalities = {
        "normal_rgb": "normal_rgb_geometry_id",
        "thermal_hotspot": "thermal_hotspot_geometry_id",
    }
    if (
        not isinstance(render_products, Mapping)
        or set(render_products) != set(expected_product_modalities)
    ):
        raise TrainingCaptureContractError(
            "rendered registration products are incomplete"
        )
    witness_paths = _registration_witness_paths(
        output_root=output_root,
        observation=observation,
    )
    width = int(expected_metadata["intrinsics"]["width_px"])
    height = int(expected_metadata["intrinsics"]["height_px"])
    captures: dict[str, dict[str, object]] = {}
    verified_products: dict[str, dict[str, object]] = {}
    product_ids: set[str] = set()
    for product_modality, witness_modality in (
        expected_product_modalities.items()
    ):
        product = render_products[product_modality]
        if not isinstance(product, Mapping) or set(product) != {
            "render_product_id",
            "source_image_artifact_id",
            "camera_projection_contract_sha256",
            "geometry_id_aov",
        }:
            raise TrainingCaptureContractError(
                "rendered registration product binding is incomplete"
            )
        product_id = _stable_id(
            product.get("render_product_id"),
            label=f"{product_modality} render product",
        )
        product_ids.add(product_id)
        expected_artifact_id = observation["image_artifacts"][
            product_modality
        ]["artifact_id"]
        if (
            product.get("source_image_artifact_id")
            != expected_artifact_id
            or product.get("camera_projection_contract_sha256")
            != expected_projection
        ):
            raise TrainingCaptureContractError(
                "rendered registration product is bound to another camera "
                "or image"
            )
        path = _regular_file_below(
            volume_root=output_root,
            path=witness_paths[witness_modality],
            label=f"{witness_modality} registration AOV",
        )
        capture = _validate_openexr_capture(
            path=path,
            width_px=width,
            height_px=height,
            modality="registration_geometry_id",
            collect_pixels=True,
        )
        expected_aov_record = {
            **_file_record(root=output_root, path=path),
            "witness_id": observation["registration_witnesses"][
                witness_modality
            ]["witness_id"],
            "modality": witness_modality,
            "semantic": "nonuniform_geometry_id_registration_aov",
            "delivery_artifact": False,
            "render_session_id": render_session_id,
            "render_product_id": product_id,
            "registration_contract_sha256": expected_registration,
            "camera_projection_contract_sha256": expected_projection,
        }
        if product.get("geometry_id_aov") != expected_aov_record:
            raise TrainingCaptureContractError(
                "rendered geometry-ID AOV record is stale"
            )
        captures[product_modality] = capture
        verified_products[product_modality] = {
            **dict(product),
            "geometry_id_aov": expected_aov_record,
        }
    if len(product_ids) != 2:
        raise TrainingCaptureContractError(
            "RGB and thermal must be distinct render products"
        )
    normal_chunks = captures["normal_rgb"]["decoded_chunks"]
    thermal_chunks = captures["thermal_hotspot"]["decoded_chunks"]
    if len(normal_chunks) != len(thermal_chunks) or not normal_chunks:
        raise TrainingCaptureContractError(
            "RGB and thermal registration AOV inventories differ"
        )
    minimum = math.inf
    maximum = -math.inf
    for normal_chunk, thermal_chunk in zip(
        normal_chunks,
        thermal_chunks,
        strict=True,
    ):
        if (
            normal_chunk["y_coordinate"]
            != thermal_chunk["y_coordinate"]
            or normal_chunk["scanline_count"]
            != thermal_chunk["scanline_count"]
        ):
            raise TrainingCaptureContractError(
                "RGB and thermal registration AOV scanlines differ"
            )
        normal_ids = np.frombuffer(normal_chunk["samples"], dtype="<f4")
        thermal_ids = np.frombuffer(thermal_chunk["samples"], dtype="<f4")
        if (
            normal_ids.size == 0
            or normal_ids.size != thermal_ids.size
            or not np.isfinite(normal_ids).all()
            or not np.array_equal(normal_ids, thermal_ids)
        ):
            raise TrainingCaptureContractError(
                "RGB and thermal geometry-ID AOV pixels are not identical"
            )
        minimum = min(minimum, float(normal_ids.min()))
        maximum = max(maximum, float(normal_ids.max()))
    if not maximum > minimum:
        raise TrainingCaptureContractError(
            "registration geometry-ID AOV is uniform and proves no geometry"
        )
    return {
        "render_session_id": render_session_id,
        "registration_contract_sha256": expected_registration,
        "camera_projection_contract_sha256": expected_projection,
        "render_products": verified_products,
        "pixel_identical_geometry_id": True,
    }


def _validate_registered_triplet(
    *,
    output_root: Path,
    observation: Mapping[str, object],
    expected_metadata: Mapping[str, object],
    thermal_evidence: object,
    registration_evidence: object,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    artifacts = _image_artifact_contracts(observation)
    paths = _image_paths(
        output_root=output_root,
        observation=observation,
    )
    width = int(expected_metadata["intrinsics"]["width_px"])
    height = int(expected_metadata["intrinsics"]["height_px"])
    captures = {
        modality: _validate_openexr_capture(
            path=_regular_file_below(
                volume_root=output_root,
                path=paths[modality],
                label=f"{modality} OpenEXR",
            ),
            width_px=width,
            height_px=height,
            modality=modality,
            collect_pixels=True,
        )
        for modality in IMAGE_MODALITIES
    }
    expected_window = [0, 0, width - 1, height - 1]
    if any(
        capture["data_window"] != expected_window
        for capture in captures.values()
    ):
        raise TrainingCaptureContractError(
            "image triplet is not pixel-aligned at the same resolution"
        )
    _validate_negative_pixels(
        normal_capture=captures["normal_rgb"],
        negative_capture=captures["negative"],
    )
    verified_thermal = _validate_thermal_evidence(
        expected_metadata=expected_metadata,
        evidence=thermal_evidence,
        thermal_capture=captures["thermal_hotspot"],
    )
    verified_registration = _validate_rendered_registration_evidence(
        output_root=output_root,
        observation=observation,
        expected_metadata=expected_metadata,
        evidence=registration_evidence,
    )
    registration = str(
        expected_metadata["pixel_registration"][
            "registration_contract_sha256"
        ]
    )
    records = {
        modality: _registered_image_record(
            output_root=output_root,
            path=paths[modality],
            artifact=artifacts[modality],
            registration_contract_sha256=registration,
            expected_metadata=expected_metadata,
        )
        for modality in IMAGE_MODALITIES
    }
    if (
        len({record["path"] for record in records.values()}) != 3
        or len({record["sha256"] for record in records.values()}) != 3
    ):
        raise TrainingCaptureContractError(
            "image triplet files and contents must be distinct"
        )
    return records, verified_thermal, verified_registration


def _validated_completion_receipt(
    *,
    plan: Mapping[str, object],
    output_root: Path,
    observation: Mapping[str, object],
    expected_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    receipt_path = _output_path(
        output_root=output_root,
        relative_value=observation["completion_receipt_path"],
        label="completion receipt",
    )
    receipt = _read_json(
        _regular_file_below(
            volume_root=output_root,
            path=receipt_path,
            label="completion receipt",
        ),
        label="completion receipt",
    )
    metadata_path = _output_path(
        output_root=output_root,
        relative_value=observation["metadata_path"],
        label="capture metadata",
    )
    expected_metadata = (
        dict(expected_metadata)
        if expected_metadata is not None
        else _metadata_contract(
            plan=plan,
            observation=observation,
        )
    )
    metadata_path = _regular_file_below(
        volume_root=output_root,
        path=metadata_path,
        label="capture metadata",
    )
    metadata = _read_json(metadata_path, label="capture metadata")
    metadata_core = dict(metadata)
    images_in_metadata = metadata_core.pop("images", None)
    thermal_evidence = metadata_core.pop("thermal_evidence", None)
    registration_evidence = metadata_core.pop(
        "rendered_registration_evidence",
        None,
    )
    if (
        metadata_core != expected_metadata
    ):
        raise TrainingCaptureContractError(
            f"{observation['observation_id']} metadata differs from its "
            "camera, fire, weather or image evidence"
        )
    (
        image_records,
        verified_thermal,
        verified_registration,
    ) = _validate_registered_triplet(
        output_root=output_root,
        observation=observation,
        expected_metadata=expected_metadata,
        thermal_evidence=thermal_evidence,
        registration_evidence=registration_evidence,
    )
    if (
        images_in_metadata != image_records
        or thermal_evidence != verified_thermal
        or registration_evidence != verified_registration
    ):
        raise TrainingCaptureContractError(
            f"{observation['observation_id']} image triplet metadata is stale"
        )
    metadata_record = _file_record(root=output_root, path=metadata_path)
    expected_receipt = {
        "schema_version": SCHEMA_VERSION,
        "state": FRAME_COMPLETE_STATE,
        "campaign_id": plan["campaign_id"],
        "observation_id": observation["observation_id"],
        "plan_sha256": plan["plan_sha256"],
        "observation_contract_sha256": observation[
            "observation_contract_sha256"
        ],
        "simulation_allowed_receipt_sha256": plan["source_locks"][
            "simulation_allowed_receipt"
        ]["sha256"],
        "image_artifact_count": len(IMAGE_MODALITIES),
        "images": image_records,
        "thermal_evidence_sha256": _canonical_sha256(verified_thermal),
        "rendered_registration_evidence_sha256": _canonical_sha256(
            verified_registration
        ),
        "metadata": metadata_record,
    }
    if receipt != expected_receipt:
        raise TrainingCaptureContractError(
            f"{observation['observation_id']} completion receipt is stale"
        )
    return receipt


def prepare_training_capture_frame_task(
    *,
    volume_root: Path,
    plan_path: Path,
    simulation_allowed_receipt_path: Path,
    observation_id: str,
) -> dict[str, object]:
    """Return one render handoff, but never run the renderer."""

    (
        plan,
        output_root,
        observation,
        expected_metadata,
    ) = _validated_addressed_task(
        volume_root=volume_root,
        plan_path=plan_path,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
        observation_id=observation_id,
    )
    image_paths = _image_paths(
        output_root=output_root,
        observation=observation,
    )
    registration_witness_paths = _registration_witness_paths(
        output_root=output_root,
        observation=observation,
    )
    metadata_path = _output_path(
        output_root=output_root,
        relative_value=observation["metadata_path"],
        label="capture metadata",
    )
    receipt_path = _output_path(
        output_root=output_root,
        relative_value=observation["completion_receipt_path"],
        label="completion receipt",
    )
    if receipt_path.exists():
        receipt = _validated_completion_receipt(
            plan=plan,
            output_root=output_root,
            observation=observation,
            expected_metadata=expected_metadata,
        )
        return {
            "state": FRAME_COMPLETE_STATE,
            "observation_id": observation_id,
            "completion_receipt": receipt,
        }
    if (
        any(path.exists() for path in image_paths.values())
        or any(path.exists() for path in registration_witness_paths.values())
        or metadata_path.exists()
    ):
        raise TrainingCaptureContractError(
            f"{observation_id} has unreceipted partial output; overwrite is "
            "forbidden"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": FRAME_READY_STATE,
        "observation": dict(observation),
        "image_output_paths": {
            modality: str(image_paths[modality])
            for modality in IMAGE_MODALITIES
        },
        "image_artifact_contracts": {
            modality: dict(artifact)
            for modality, artifact in _image_artifact_contracts(
                observation
            ).items()
        },
        "registration_witness_output_paths": {
            modality: str(registration_witness_paths[modality])
            for modality in REGISTRATION_WITNESS_MODALITIES
        },
        "registration_witness_contracts": {
            modality: dict(witness)
            for modality, witness in _registration_witness_contracts(
                observation
            ).items()
        },
        "metadata_output_path": str(metadata_path),
        "completion_receipt_path": str(receipt_path),
        "metadata_contract_without_images_or_thermal_evidence": dict(
            expected_metadata
        ),
        "required_image_record_fields": [
            "artifact_id",
            "modality",
            "registration_contract_sha256",
            "registration",
            "path",
            "sha256",
            "size_bytes",
        ],
        "required_thermal_evidence_fields": [
            "render_source",
            "temperature_unit",
            "heat_field_sha256",
            "surface_emissivity",
            "min_temperature_k",
            "max_temperature_k",
            "hotspot_temperature_k",
            "hotspot_pixel_xy",
            "hotspot_local_m",
            "hotspot_epsg2154_ign69_m",
            "depth_at_hotspot_m",
            "hotspot_in_active_front",
            "hotspot_in_fov",
            "registration_contract_sha256",
        ],
        "required_rendered_registration_evidence": {
            "aov": "nonuniform_geometry_id_registration_aov",
            "modalities": list(REGISTRATION_WITNESS_MODALITIES),
            "pixel_relation": "exactly_identical",
            "delivery_artifact": False,
            "camera_projection_contract_sha256": (
                expected_metadata["pixel_registration"][
                    "camera_projection_contract_sha256"
                ]
            ),
        },
        "simulation_execution_performed": False,
        "render_execution_performed": False,
    }


def record_training_capture_observation_completion(
    *,
    volume_root: Path,
    plan_path: Path,
    simulation_allowed_receipt_path: Path,
    observation_id: str,
) -> dict[str, object]:
    """Validate and receipt one RGB/negative/thermal observation triplet."""

    (
        plan,
        output_root,
        observation,
        expected_metadata,
    ) = _validated_addressed_task(
        volume_root=volume_root,
        plan_path=plan_path,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
        observation_id=observation_id,
    )
    receipt_path = _output_path(
        output_root=output_root,
        relative_value=observation["completion_receipt_path"],
        label="completion receipt",
    )
    if receipt_path.exists():
        return _validated_completion_receipt(
            plan=plan,
            output_root=output_root,
            observation=observation,
            expected_metadata=expected_metadata,
        )
    metadata_path = _output_path(
        output_root=output_root,
        relative_value=observation["metadata_path"],
        label="capture metadata",
    )
    metadata = _read_json(
        _regular_file_below(
            volume_root=output_root,
            path=metadata_path,
            label="capture metadata",
        ),
        label="capture metadata",
    )
    metadata_core = dict(metadata)
    images_in_metadata = metadata_core.pop("images", None)
    thermal_evidence = metadata_core.pop("thermal_evidence", None)
    registration_evidence = metadata_core.pop(
        "rendered_registration_evidence",
        None,
    )
    if (
        metadata_core != expected_metadata
    ):
        raise TrainingCaptureContractError(
            f"{observation_id} rendered metadata does not match its plan"
        )
    (
        image_records,
        verified_thermal,
        verified_registration,
    ) = _validate_registered_triplet(
        output_root=output_root,
        observation=observation,
        expected_metadata=expected_metadata,
        thermal_evidence=thermal_evidence,
        registration_evidence=registration_evidence,
    )
    if (
        images_in_metadata != image_records
        or thermal_evidence != verified_thermal
        or registration_evidence != verified_registration
    ):
        raise TrainingCaptureContractError(
            f"{observation_id} image triplet metadata does not match files"
        )
    metadata_record = _file_record(root=output_root, path=metadata_path)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "state": FRAME_COMPLETE_STATE,
        "campaign_id": plan["campaign_id"],
        "observation_id": observation_id,
        "plan_sha256": plan["plan_sha256"],
        "observation_contract_sha256": observation[
            "observation_contract_sha256"
        ],
        "simulation_allowed_receipt_sha256": plan["source_locks"][
            "simulation_allowed_receipt"
        ]["sha256"],
        "image_artifact_count": len(IMAGE_MODALITIES),
        "images": image_records,
        "thermal_evidence_sha256": _canonical_sha256(verified_thermal),
        "rendered_registration_evidence_sha256": _canonical_sha256(
            verified_registration
        ),
        "metadata": metadata_record,
    }
    _atomic_write_json(receipt_path, receipt)
    return _validated_completion_receipt(
        plan=plan,
        output_root=output_root,
        observation=observation,
        expected_metadata=expected_metadata,
    )


def _validate_completion_inventory(
    *,
    plan: Mapping[str, object],
    receipts: Sequence[Mapping[str, object]],
    require_all: bool,
) -> None:
    """Pure exact-set gate shared by resume, final verification and tests."""

    observations = _observation_lookup(plan)
    seen: set[str] = set()
    seen_image_artifact_ids: set[str] = set()
    for receipt in receipts:
        observation_id = str(receipt.get("observation_id", ""))
        observation = observations.get(observation_id)
        images = receipt.get("images")
        expected_artifacts = (
            _image_artifact_contracts(observation)
            if observation is not None
            else {}
        )
        valid_images = (
            observation is not None
            and isinstance(images, Mapping)
            and set(images) == set(IMAGE_MODALITIES)
            and receipt.get("image_artifact_count")
            == len(IMAGE_MODALITIES)
        )
        if valid_images:
            for modality in IMAGE_MODALITIES:
                image = images[modality]
                artifact = expected_artifacts[modality]
                if (
                    not isinstance(image, Mapping)
                    or image.get("artifact_id") != artifact["artifact_id"]
                    or image.get("modality") != modality
                    or image.get("registration_contract_sha256")
                    != observation.get("registration_contract_sha256")
                    or image.get("artifact_id")
                    in seen_image_artifact_ids
                ):
                    valid_images = False
                    break
                seen_image_artifact_ids.add(str(image["artifact_id"]))
        if (
            observation is None
            or observation_id in seen
            or receipt.get("state") != FRAME_COMPLETE_STATE
            or receipt.get("plan_sha256") != plan.get("plan_sha256")
            or receipt.get("observation_contract_sha256")
            != observation.get("observation_contract_sha256")
            or receipt.get("simulation_allowed_receipt_sha256")
            != plan.get("source_locks", {})
            .get("simulation_allowed_receipt", {})
            .get("sha256")
            or not valid_images
        ):
            raise TrainingCaptureContractError(
                "completion inventory is duplicated, unknown or stale"
            )
        seen.add(observation_id)
    if require_all and (
        len(receipts) != EXPECTED_OBSERVATION_COUNT
        or seen != set(observations)
        or len(seen_image_artifact_ids) != EXPECTED_IMAGE_ARTIFACT_COUNT
    ):
        raise TrainingCaptureContractError(
            "complete campaign requires exactly 18,360 unique observations "
            "and 55,080 registered image artifacts"
        )


def _unexpected_capture_files(
    *,
    output_root: Path,
    observations: Iterable[Mapping[str, object]],
) -> list[str]:
    expected: set[str] = set()
    for observation in observations:
        expected.update(
            str(artifact["path"])
            for artifact in _image_artifact_contracts(
                observation
            ).values()
        )
        expected.update(
            str(witness["path"])
            for witness in _registration_witness_contracts(
                observation
            ).values()
        )
        expected.add(str(observation["metadata_path"]))
        expected.add(str(observation["completion_receipt_path"]))
    unexpected: list[str] = []
    for directory in ("frames", "registration", "metadata", "receipts"):
        root = output_root / directory
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            unexpected.append(directory)
            continue
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(output_root).as_posix()
                if relative not in expected:
                    unexpected.append(relative)
    return sorted(unexpected)


def inspect_training_capture_resume(
    *,
    volume_root: Path,
    plan_path: Path,
    simulation_allowed_receipt_path: Path,
) -> dict[str, object]:
    """Rehash complete frames and report the first safe pending observation."""

    plan, output_root = _validated_plan(
        volume_root=volume_root,
        plan_path=plan_path,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
    )
    observations = list(_observation_lookup(plan).values())
    completed_receipts: list[dict[str, object]] = []
    completion_locks: list[dict[str, object]] = []
    pending: list[str] = []
    partial: list[str] = []
    for observation in observations:
        receipt_path = _output_path(
            output_root=output_root,
            relative_value=observation["completion_receipt_path"],
            label="completion receipt",
        )
        image_paths = _image_paths(
            output_root=output_root,
            observation=observation,
        )
        witness_paths = _registration_witness_paths(
            output_root=output_root,
            observation=observation,
        )
        metadata_path = _output_path(
            output_root=output_root,
            relative_value=observation["metadata_path"],
            label="capture metadata",
        )
        if receipt_path.exists():
            completed_receipts.append(
                _validated_completion_receipt(
                    plan=plan,
                    output_root=output_root,
                    observation=observation,
                )
            )
            completion_locks.append(
                _file_record(root=output_root, path=receipt_path)
            )
        elif (
            any(path.exists() for path in image_paths.values())
            or any(path.exists() for path in witness_paths.values())
            or metadata_path.exists()
        ):
            partial.append(str(observation["observation_id"]))
        else:
            pending.append(str(observation["observation_id"]))
    _validate_completion_inventory(
        plan=plan,
        receipts=completed_receipts,
        require_all=False,
    )
    unexpected = _unexpected_capture_files(
        output_root=output_root,
        observations=observations,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": (
            CAMPAIGN_VERIFIED_STATE
            if len(completed_receipts) == EXPECTED_OBSERVATION_COUNT
            and not partial
            and not unexpected
            else "TRAINING_CAPTURE_CAMPAIGN_INCOMPLETE"
        ),
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "total_observations": EXPECTED_OBSERVATION_COUNT,
        "total_image_artifacts": EXPECTED_IMAGE_ARTIFACT_COUNT,
        "planned_image_artifact_inventory_sha256": plan[
            "image_artifact_inventory_sha256"
        ],
        "completed_observations": len(completed_receipts),
        "completed_image_artifacts": (
            len(completed_receipts) * len(IMAGE_MODALITIES)
        ),
        "pending_observations": len(pending),
        "pending_image_artifacts": (
            len(pending) * len(IMAGE_MODALITIES)
        ),
        "partial_observations": len(partial),
        "next_observation_id": pending[0] if pending and not partial else None,
        "pending_observation_preview": pending[:100],
        "partial_observation_ids": partial,
        "unexpected_paths": unexpected,
        "completion_receipt_inventory_sha256": _canonical_sha256(
            sorted(completion_locks, key=lambda value: str(value["path"]))
        ),
    }


def verify_training_capture_campaign(
    *,
    volume_root: Path,
    plan_path: Path,
    simulation_allowed_receipt_path: Path,
    verification_receipt_path: Path | None = None,
) -> dict[str, object]:
    """Require 18,360 observations and exactly 55,080 registered images."""

    plan, output_root = _validated_plan(
        volume_root=volume_root,
        plan_path=plan_path,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
    )
    progress = inspect_training_capture_resume(
        volume_root=volume_root,
        plan_path=plan_path,
        simulation_allowed_receipt_path=simulation_allowed_receipt_path,
    )
    if (
        progress["completed_observations"] != EXPECTED_OBSERVATION_COUNT
        or progress["pending_observations"] != 0
        or progress["partial_observations"] != 0
        or progress["unexpected_paths"]
    ):
        raise TrainingCaptureContractError(
            "campaign verification requires exactly 18,360 complete, current "
            "observations and no partial or unexpected files"
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "state": CAMPAIGN_VERIFIED_STATE,
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "simulation_allowed_receipt_sha256": plan["source_locks"][
            "simulation_allowed_receipt"
        ]["sha256"],
        "observation_count": EXPECTED_OBSERVATION_COUNT,
        "image_artifact_count": EXPECTED_IMAGE_ARTIFACT_COUNT,
        "planned_image_artifact_inventory_sha256": plan[
            "image_artifact_inventory_sha256"
        ],
        "completion_receipt_inventory_sha256": progress[
            "completion_receipt_inventory_sha256"
        ],
        "simulation_execution_performed_by_this_module": False,
        "render_execution_performed_by_this_module": False,
    }
    if verification_receipt_path is not None:
        target = verification_receipt_path.resolve()
        relative_target = (
            target.relative_to(output_root)
            if _inside(output_root, target)
            else None
        )
        if (
            relative_target is None
            or verification_receipt_path.is_symlink()
            or target.suffix.casefold() != ".json"
            or (
                relative_target.parts
                and relative_target.parts[0]
                in {"frames", "metadata", "receipts"}
            )
        ):
            raise TrainingCaptureContractError(
                "campaign verification receipt must stay below output root"
            )
        if target.exists():
            if _read_json(target, label="campaign verification receipt") != receipt:
                raise TrainingCaptureContractError(
                    "existing campaign verification receipt drifted"
                )
        else:
            _atomic_write_json(target, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan 18,360 post-review observations and verify their 55,080 "
            "co-registered images; "
            "never execute simulation or rendering"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--volume-root", required=True, type=Path)
    prepare.add_argument("--simulation-allowed", required=True, type=Path)
    prepare.add_argument("--source-manifest", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--plan", required=True, type=Path)
    frame = commands.add_parser("frame-task")
    frame.add_argument("--volume-root", required=True, type=Path)
    frame.add_argument("--simulation-allowed", required=True, type=Path)
    frame.add_argument("--plan", required=True, type=Path)
    frame.add_argument("--observation", required=True)
    record = commands.add_parser("record-frame")
    record.add_argument("--volume-root", required=True, type=Path)
    record.add_argument("--simulation-allowed", required=True, type=Path)
    record.add_argument("--plan", required=True, type=Path)
    record.add_argument("--observation", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--volume-root", required=True, type=Path)
    resume.add_argument("--simulation-allowed", required=True, type=Path)
    resume.add_argument("--plan", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--volume-root", required=True, type=Path)
    verify.add_argument("--simulation-allowed", required=True, type=Path)
    verify.add_argument("--plan", required=True, type=Path)
    verify.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_training_capture_campaign_plan(
            volume_root=args.volume_root,
            simulation_allowed_receipt_path=args.simulation_allowed,
            source_manifest_path=args.source_manifest,
            output_root=args.output_root,
            plan_path=args.plan,
        )
        result = {
            "state": result["state"],
            "campaign_id": result["campaign_id"],
            "observation_count": result["observation_count"],
            "image_artifact_count": result["image_artifact_count"],
            "plan_sha256": result["plan_sha256"],
            "plan_path": str(args.plan.resolve()),
        }
    elif args.command == "frame-task":
        result = prepare_training_capture_frame_task(
            volume_root=args.volume_root,
            plan_path=args.plan,
            simulation_allowed_receipt_path=args.simulation_allowed,
            observation_id=args.observation,
        )
    elif args.command == "record-frame":
        result = record_training_capture_observation_completion(
            volume_root=args.volume_root,
            plan_path=args.plan,
            simulation_allowed_receipt_path=args.simulation_allowed,
            observation_id=args.observation,
        )
    elif args.command == "resume":
        result = inspect_training_capture_resume(
            volume_root=args.volume_root,
            plan_path=args.plan,
            simulation_allowed_receipt_path=args.simulation_allowed,
        )
    else:
        result = verify_training_capture_campaign(
            volume_root=args.volume_root,
            plan_path=args.plan,
            simulation_allowed_receipt_path=args.simulation_allowed,
            verification_receipt_path=args.receipt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CAMERAS_PER_SCENE",
    "CAMPAIGN_VERIFIED_STATE",
    "CAPTURE_FORMAT",
    "CAPTURE_FORMATS",
    "CAPTURE_HOURS",
    "EDITOR_REVIEW_ACKNOWLEDGEMENT",
    "EXPECTED_OBSERVATION_COUNT",
    "EXPECTED_IMAGE_ARTIFACT_COUNT",
    "FRAME_COMPLETE_STATE",
    "FRAME_READY_STATE",
    "FRAME_RENDERED_STATE",
    "GEOREFERENCE_AUTHENTICATED_STATE",
    "GEOREFERENCE_AXIS_ORDER",
    "GEOREFERENCE_AXIS_MAPPING",
    "GEOREFERENCE_LOCAL_AXES",
    "GATE_ARTIFACT_KEYS",
    "GATE_BINDING_KEYS",
    "IMAGE_MODALITIES",
    "REGISTRATION_WITNESS_FORMAT",
    "REGISTRATION_WITNESS_MODALITIES",
    "PLAN_STATE",
    "SCENE_DURATIONS_DAYS",
    "SCENE_IDS",
    "SIMULATION_ALLOWED_STATE",
    "SPECIAL_VIEW_CLASS_MINIMUMS",
    "SOURCE_STATE",
    "TARGET_GROUP_DIAMETER_RADIUS_FRACTION",
    "TARGET_MAX_OFFSET_RADIUS_FRACTION",
    "TrainingCaptureContractError",
    "VIEW_CLASSES",
    "build_parser",
    "inspect_training_capture_resume",
    "main",
    "prepare_training_capture_campaign_plan",
    "prepare_training_capture_frame_task",
    "record_training_capture_observation_completion",
    "verify_training_capture_campaign",
]


if __name__ == "__main__":
    raise SystemExit(main())
