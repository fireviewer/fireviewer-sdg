"""Install a portable, photoreal USD asset bundle without weakening its locks.

The network transfer is deliberately owned by the RunPod setup shell script.
This module only consumes a local archive that is already on the persistent
volume, verifies its SHA-256, extracts regular files safely, and converts the
portable manifest's dependency paths into paths relative to the volume root.

Native USD loading and visual-quality validation remain the responsibility of
``fireviewer_sdg.omniverse_pod validate-assets --native-usd-quality``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

from fireviewer_sdg.simready_assets import (
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
)


SHA256_LENGTH = 64
INSTALL_MARKER = ".fireviewer-asset-bundle.json"
DEFAULT_MAX_FILES = 100_000
DEFAULT_MAX_UNPACKED_BYTES = 500 * 1024**3
DEFAULT_MIN_FREE_AFTER_INSTALL_BYTES = 100 * 1024**3
LOD_LEVELS = ("HERO", "MID", "FAR")
REQUIRED_ACTOR_CLASSES = (
    "sdis_vehicle",
    "canadair",
    "dash",
    "securite_civile_helicopter",
    "hard_negative_construction_truck",
    "hard_negative_crop_duster",
    "hard_negative_utility_helicopter",
)
PBR_MATERIAL_ROLES = (
    "forest_floor",
    "grass",
    "soil",
    "rock",
    "asphalt",
    "gravel",
    "water",
)
PBR_REQUIRED_TEXTURES = ("base_color", "normal", "roughness")
PBR_OPTIONAL_TEXTURES = ("displacement",)
MIN_PBR_TEXTURE_DIMENSION = 2_048
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
MATERIAL_SUFFIXES = USD_SUFFIXES
TEXTURE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dds", ".exr"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _inside(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return (
        resolved_candidate == resolved_root
        or resolved_root in resolved_candidate.parents
    )


def _safe_relative_path(raw: str, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    if "\\" in raw or "\x00" in raw or any(ord(char) < 32 for char in raw):
        raise ValueError(f"{label} contains forbidden characters")
    value = PurePosixPath(raw)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise ValueError(f"{label} must stay below the bundle root")
    if value.parts[0] in {"", "."}:
        raise ValueError(f"{label} must be normalized")
    return value


def _validate_destination(
    *, volume_root: Path, destination_root: Path, expected_sha256: str
) -> tuple[Path, Path, str]:
    volume = volume_root.resolve()
    destination = destination_root.resolve()
    digest = expected_sha256.strip().lower()
    if (
        len(digest) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("asset bundle expected_sha256 must be exactly 64 hex digits")
    if (
        not volume.is_dir()
        or volume.is_symlink()
        or not _inside(volume, destination)
        or destination == volume
    ):
        raise ValueError(
            "asset bundle destination must be a child of the persistent volume"
        )
    return volume, destination, digest


def _copy_exact(
    source: BinaryIO,
    destination: Path,
    *,
    expected_size: int,
) -> None:
    written = 0
    with destination.open("xb") as output:
        while block := source.read(4 * 1024 * 1024):
            written += len(block)
            if written > expected_size:
                raise ValueError(
                    f"archive member exceeds its declared size: {destination.name}"
                )
            output.write(block)
    if written != expected_size:
        raise ValueError(
            f"archive member size mismatch for {destination.name}: "
            f"expected={expected_size} actual={written}"
        )
    destination.chmod(0o640)


def _prepare_member_paths(
    members: Iterable[tuple[str, int]],
    *,
    destination: Path,
    max_files: int,
    max_unpacked_bytes: int,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    casefolded: set[str] = set()
    total_size = 0
    for index, (raw_name, size) in enumerate(members, start=1):
        if index > max_files:
            raise ValueError(
                f"asset bundle exceeds the {max_files} member safety limit"
            )
        if size < 0:
            raise ValueError(f"archive member has a negative size: {raw_name}")
        total_size += size
        if total_size > max_unpacked_bytes:
            raise ValueError(
                "asset bundle exceeds the configured uncompressed-size safety limit"
            )
        relative = _safe_relative_path(raw_name.rstrip("/"), label="archive member")
        normalized = relative.as_posix()
        key = normalized.casefold()
        if key in casefolded:
            raise ValueError(
                f"asset bundle contains duplicate or case-colliding path: {normalized}"
            )
        casefolded.add(key)
        target = destination.joinpath(*relative.parts)
        if not _inside(destination, target):
            raise ValueError(f"archive member escapes the bundle root: {raw_name}")
        paths[raw_name] = target
    return paths


def _extract_zip(
    archive: Path,
    destination: Path,
    *,
    max_files: int,
    max_unpacked_bytes: int,
) -> None:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for info in infos:
            unix_type = (info.external_attr >> 16) & 0o170000
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted archive members are forbidden: {info.filename}")
            if unix_type == stat.S_IFLNK:
                raise ValueError(f"archive links are forbidden: {info.filename}")
            if not (info.is_dir() or unix_type in {0, stat.S_IFREG}):
                raise ValueError(
                    f"archive member is not a regular file or directory: {info.filename}"
                )
        paths = _prepare_member_paths(
            ((info.filename, info.file_size) for info in infos),
            destination=destination,
            max_files=max_files,
            max_unpacked_bytes=max_unpacked_bytes,
        )
        for info in infos:
            target = paths[info.filename]
            if info.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError(
                        f"archive directory collides with a file: {info.filename}"
                    )
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(0o750)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info, "r") as source:
                _copy_exact(source, target, expected_size=info.file_size)


def _extract_tar(
    archive: Path,
    destination: Path,
    *,
    max_files: int,
    max_unpacked_bytes: int,
) -> None:
    with tarfile.open(archive, mode="r:*") as bundle:
        members = bundle.getmembers()
        for member in members:
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"archive links and special files are forbidden: {member.name}")
        paths = _prepare_member_paths(
            ((member.name, member.size) for member in members),
            destination=destination,
            max_files=max_files,
            max_unpacked_bytes=max_unpacked_bytes,
        )
        for member in members:
            target = paths[member.name]
            if member.isdir():
                if target.exists() and not target.is_dir():
                    raise ValueError(
                        f"archive directory collides with a file: {member.name}"
                    )
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(0o750)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"unable to read archive member: {member.name}")
            with source:
                _copy_exact(source, target, expected_size=member.size)


def _extract_archive(
    archive: Path,
    destination: Path,
    *,
    max_files: int,
    max_unpacked_bytes: int,
) -> None:
    if zipfile.is_zipfile(archive):
        _extract_zip(
            archive,
            destination,
            max_files=max_files,
            max_unpacked_bytes=max_unpacked_bytes,
        )
    elif tarfile.is_tarfile(archive):
        _extract_tar(
            archive,
            destination,
            max_files=max_files,
            max_unpacked_bytes=max_unpacked_bytes,
        )
    else:
        raise ValueError("asset bundle must be a valid ZIP or TAR archive")


def _archive_declared_usage(
    archive: Path,
    *,
    destination: Path,
    max_files: int,
    max_unpacked_bytes: int,
) -> tuple[int, int]:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            members = [
                (info.filename, info.file_size)
                for info in bundle.infolist()
            ]
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, mode="r:*") as bundle:
            members = [
                (member.name, member.size)
                for member in bundle.getmembers()
            ]
    else:
        raise ValueError("asset bundle must be a valid ZIP or TAR archive")
    _prepare_member_paths(
        members,
        destination=destination,
        max_files=max_files,
        max_unpacked_bytes=max_unpacked_bytes,
    )
    return len(members), sum(size for _name, size in members)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"asset bundle manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("asset bundle manifest must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"asset bundle manifest schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    if payload.get("profile") != MANIFEST_PROFILE:
        raise ValueError(f"asset bundle profile must be {MANIFEST_PROFILE}")
    if payload.get("family_minimums") != PHOTOREAL_FAMILY_MINIMUMS:
        raise ValueError("asset bundle family minimums were weakened")
    if payload.get("library_policy") != PHOTOREAL_LIBRARY_POLICY:
        raise ValueError("asset bundle photoreal library policy was weakened")
    discovery = payload.get("discovery")
    if (
        not isinstance(discovery, dict)
        or discovery.get("mode")
        != "materialized_photoreal_asset_library_v3"
        or list(discovery.get("missing_environment") or [])
    ):
        raise ValueError("asset bundle discovery contract is incomplete")
    return payload


def _environment_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    environment = payload.get("environment")
    if not isinstance(environment, dict) or set(environment) != set(
        PHOTOREAL_FAMILY_MINIMUMS
    ):
        raise ValueError("asset bundle has an invalid environment family tree")
    entries: list[tuple[str, dict[str, Any]]] = []
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        families = environment.get(kind)
        if not isinstance(families, dict) or set(families) != set(family_minimums):
            raise ValueError(f"asset bundle environment.{kind} family tree is invalid")
        for family, minimum in family_minimums.items():
            assets = families.get(family)
            role = f"{kind}.{family}"
            if not isinstance(assets, list) or len(assets) < minimum:
                raise ValueError(
                    f"asset bundle requires at least {minimum} assets in {role}"
                )
            for index, entry in enumerate(assets):
                if not isinstance(entry, dict):
                    raise ValueError(f"{role}[{index}] must be a JSON object")
                entries.append((f"{role}[{index}]", entry))
    return entries


def _actor_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    actors = payload.get("actors")
    actor_keys = set(actors) if isinstance(actors, dict) else set()
    if not isinstance(actors, dict) or actor_keys != set(REQUIRED_ACTOR_CLASSES):
        missing = sorted(set(REQUIRED_ACTOR_CLASSES) - actor_keys)
        unexpected = sorted(actor_keys - set(REQUIRED_ACTOR_CLASSES))
        raise ValueError(
            "asset bundle requires the exact reviewed actor classes: "
            f"missing={missing}, unexpected={unexpected}"
        )
    entries: list[tuple[str, dict[str, Any]]] = []
    for class_id in REQUIRED_ACTOR_CLASSES:
        entry = actors[class_id]
        if not isinstance(entry, dict):
            raise ValueError(f"actors.{class_id} must be a JSON object")
        entries.append((f"actors.{class_id}", entry))
    return entries


def _asset_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [*_environment_entries(payload), *_actor_entries(payload)]


def _portable_file(
    manifest_parent: Path,
    raw: object,
    *,
    label: str,
) -> tuple[Path, PurePosixPath]:
    relative = _safe_relative_path(str(raw or ""), label=label)
    path = manifest_parent.joinpath(*relative.parts)
    if not _inside(manifest_parent, path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file inside the bundle")
    return path, relative


def _locked_portable_file(
    *,
    manifest_parent: Path,
    record: object,
    label: str,
    allowed_suffixes: frozenset[str],
) -> tuple[Path, PurePosixPath]:
    if not isinstance(record, dict):
        raise ValueError(f"{label} lock must be an object")
    path, relative = _portable_file(
        manifest_parent,
        record.get("path"),
        label=label,
    )
    if path.suffix.casefold() not in allowed_suffixes:
        raise ValueError(f"{label} has an unsupported file type: {path.suffix}")
    actual_sha = _sha256(path)
    actual_size = path.stat().st_size
    if (
        str(record.get("sha256", "")).strip().lower() != actual_sha
        or record.get("size_bytes") != actual_size
    ):
        raise ValueError(f"{label} SHA-256 or size lock does not match")
    return path, relative


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if (
        len(header) < 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG texture header: {path}")
    return struct.unpack(">II", header[16:24])


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8" or data[-2:] != b"\xff\xd9":
        raise ValueError(f"invalid JPEG texture framing: {path}")
    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = struct.unpack(">H", data[offset : offset + 2])[0]
        if length < 2 or offset + length > len(data):
            raise ValueError(f"invalid JPEG segment length: {path}")
        if marker in start_of_frame:
            if length < 8:
                raise ValueError(f"invalid JPEG start-of-frame segment: {path}")
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return width, height
        offset += length
    raise ValueError(f"JPEG texture has no image dimensions: {path}")


def _tiff_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 8 or data[:2] not in {b"II", b"MM"}:
        raise ValueError(f"invalid TIFF texture header: {path}")
    endian = "<" if data[:2] == b"II" else ">"
    if struct.unpack(f"{endian}H", data[2:4])[0] != 42:
        raise ValueError(f"BigTIFF and invalid TIFF textures are unsupported: {path}")
    offset = struct.unpack(f"{endian}I", data[4:8])[0]
    if offset + 2 > len(data):
        raise ValueError(f"invalid TIFF IFD offset: {path}")
    count = struct.unpack(f"{endian}H", data[offset : offset + 2])[0]
    values: dict[int, int] = {}
    cursor = offset + 2
    for _index in range(count):
        if cursor + 12 > len(data):
            raise ValueError(f"truncated TIFF IFD: {path}")
        tag, value_type, item_count = struct.unpack(
            f"{endian}HHI", data[cursor : cursor + 8]
        )
        raw = data[cursor + 8 : cursor + 12]
        if item_count == 1 and value_type in {3, 4}:
            values[tag] = struct.unpack(
                f"{endian}{'H' if value_type == 3 else 'I'}",
                raw[: 2 if value_type == 3 else 4],
            )[0]
        cursor += 12
    if 256 not in values or 257 not in values:
        raise ValueError(f"TIFF texture has no width/height tags: {path}")
    return values[256], values[257]


def _dds_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"DDS ":
        raise ValueError(f"invalid DDS texture header: {path}")
    height, width = struct.unpack("<II", header[12:20])
    return width, height


def _exr_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 16 or data[:4] != b"\x76\x2f\x31\x01":
        raise ValueError(f"invalid OpenEXR texture header: {path}")
    offset = 8
    while offset < len(data):
        name_end = data.find(b"\x00", offset)
        if name_end < 0:
            break
        if name_end == offset:
            break
        name = data[offset:name_end].decode("ascii", errors="strict")
        offset = name_end + 1
        type_end = data.find(b"\x00", offset)
        if type_end < 0 or type_end + 5 > len(data):
            break
        attribute_type = data[offset:type_end].decode("ascii", errors="strict")
        offset = type_end + 1
        size = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        if offset + size > len(data):
            raise ValueError(f"truncated OpenEXR attribute: {path}")
        value = data[offset : offset + size]
        offset += size
        if name == "dataWindow" and attribute_type == "box2i" and size == 16:
            minimum_x, minimum_y, maximum_x, maximum_y = struct.unpack("<iiii", value)
            width = maximum_x - minimum_x + 1
            height = maximum_y - minimum_y + 1
            return width, height
    raise ValueError(f"OpenEXR texture has no valid dataWindow: {path}")


def _image_dimensions(path: Path) -> tuple[int, int]:
    suffix = path.suffix.casefold()
    if suffix == ".png":
        return _png_dimensions(path)
    if suffix in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(path)
    if suffix in {".tif", ".tiff"}:
        return _tiff_dimensions(path)
    if suffix == ".dds":
        return _dds_dimensions(path)
    if suffix == ".exr":
        return _exr_dimensions(path)
    raise ValueError(f"unsupported PBR texture format: {path.suffix}")


def _validate_pbr_materials(
    *,
    payload: dict[str, Any],
    manifest_parent: Path,
) -> dict[str, Any]:
    materials = payload.get("pbr_materials")
    if not isinstance(materials, dict) or set(materials) != set(PBR_MATERIAL_ROLES):
        raise ValueError(
            "asset bundle pbr_materials must contain exactly "
            + ", ".join(PBR_MATERIAL_ROLES)
        )
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    seen_hashes: set[str] = set()
    summary: dict[str, Any] = {}
    for role in PBR_MATERIAL_ROLES:
        material = materials.get(role)
        if not isinstance(material, dict):
            raise ValueError(f"pbr_materials.{role} must be an object")
        material_id = str(material.get("material_id", "")).strip()
        if not material_id or material_id in seen_ids:
            raise ValueError(f"pbr_materials.{role} requires a unique material_id")
        seen_ids.add(material_id)
        material_file, material_relative = _locked_portable_file(
            manifest_parent=manifest_parent,
            record=material.get("material_file"),
            label=f"pbr_materials.{role}.material_file",
            allowed_suffixes=MATERIAL_SUFFIXES,
        )
        material_key = material_relative.as_posix().casefold()
        material_hash = _sha256(material_file)
        if material_key in seen_files or material_hash in seen_hashes:
            raise ValueError("PBR material files and textures must be unique")
        seen_files.add(material_key)
        seen_hashes.add(material_hash)
        material_prim_path = str(material.get("material_prim_path", "")).strip()
        if (
            not material_prim_path.startswith("/")
            or "//" in material_prim_path
            or ".." in material_prim_path.split("/")
        ):
            raise ValueError(
                f"pbr_materials.{role}.material_prim_path must be an absolute "
                "USD prim path"
            )
        try:
            metres_per_tile = float(material["metres_per_uv_tile"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"pbr_materials.{role}.metres_per_uv_tile is invalid"
            ) from exc
        if not 0.1 <= metres_per_tile <= 100.0:
            raise ValueError(
                f"pbr_materials.{role}.metres_per_uv_tile is implausible"
            )
        textures = material.get("textures")
        if not isinstance(textures, dict) or not set(PBR_REQUIRED_TEXTURES).issubset(
            textures
        ) or set(textures) - set(PBR_REQUIRED_TEXTURES + PBR_OPTIONAL_TEXTURES):
            raise ValueError(
                f"pbr_materials.{role} requires base_color, normal and roughness "
                "with optional displacement"
            )
        dimensions: tuple[int, int] | None = None
        texture_summary: dict[str, Any] = {}
        for texture_role in (
            *PBR_REQUIRED_TEXTURES,
            *(
                optional
                for optional in PBR_OPTIONAL_TEXTURES
                if optional in textures
            ),
        ):
            texture_record = textures[texture_role]
            texture, texture_relative = _locked_portable_file(
                manifest_parent=manifest_parent,
                record=texture_record,
                label=f"pbr_materials.{role}.textures.{texture_role}",
                allowed_suffixes=TEXTURE_SUFFIXES,
            )
            texture_key = texture_relative.as_posix().casefold()
            texture_hash = _sha256(texture)
            if texture_key in seen_files or texture_hash in seen_hashes:
                raise ValueError("PBR material files and textures must be unique")
            seen_files.add(texture_key)
            seen_hashes.add(texture_hash)
            actual_dimensions = _image_dimensions(texture)
            try:
                declared_dimensions = (
                    int(texture_record["width_px"]),
                    int(texture_record["height_px"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"pbr_materials.{role}.textures.{texture_role} dimensions "
                    "are invalid"
                ) from exc
            if (
                actual_dimensions != declared_dimensions
                or min(actual_dimensions) < MIN_PBR_TEXTURE_DIMENSION
                or actual_dimensions[0] != actual_dimensions[1]
            ):
                raise ValueError(
                    f"pbr_materials.{role}.textures.{texture_role} must be a "
                    f"square >= {MIN_PBR_TEXTURE_DIMENSION}px texture matching "
                    "its declared dimensions"
                )
            color_space = str(texture_record.get("color_space", "")).casefold()
            expected_spaces = (
                {"srgb"}
                if texture_role == "base_color"
                else {"raw", "linear"}
            )
            if color_space not in expected_spaces:
                raise ValueError(
                    f"pbr_materials.{role}.textures.{texture_role} has an "
                    "invalid color_space"
                )
            if dimensions is None:
                dimensions = actual_dimensions
            elif dimensions != actual_dimensions:
                raise ValueError(
                    f"pbr_materials.{role} texture dimensions must match"
                )
            texture_summary[texture_role] = {
                "path": texture_relative.as_posix(),
                "sha256": texture_hash,
                "width_px": actual_dimensions[0],
                "height_px": actual_dimensions[1],
                "color_space": color_space,
            }
        summary[role] = {
            "material_id": material_id,
            "material_file": material_relative.as_posix(),
            "material_file_sha256": _sha256(material_file),
            "material_prim_path": material_prim_path,
            "metres_per_uv_tile": metres_per_tile,
            "textures": texture_summary,
        }
    return summary


def _validate_asset_lod_paths(
    *,
    role: str,
    entry: dict[str, Any],
    manifest_parent: Path,
) -> dict[str, dict[str, object]]:
    lod_paths = entry.get("lod_paths")
    if not isinstance(lod_paths, dict) or set(lod_paths) != set(LOD_LEVELS):
        raise ValueError(f"{role} requires local HERO, MID and FAR lod_paths")
    summary: dict[str, dict[str, object]] = {}
    asset_id = str(entry.get("asset_id", "")).strip()
    identity = entry.get("identity")
    source_identity = (
        str(identity.get("source_identity", "")).strip()
        if isinstance(identity, dict)
        else ""
    )
    expected_lineage = hashlib.sha256(
        f"{asset_id}\0{source_identity}".encode("utf-8")
    ).hexdigest()
    lineage = str(entry.get("lod_lineage_id", "")).strip().lower()
    if not asset_id or not source_identity or lineage != expected_lineage:
        raise ValueError(f"{role} has no locked common source lineage for its LODs")
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for level in LOD_LEVELS:
        lod_record = lod_paths[level]
        if (
            not isinstance(lod_record, dict)
            or str(lod_record.get("lineage_id", "")).strip().lower() != lineage
        ):
            raise ValueError(f"{role}.lod_paths.{level} breaks the common lineage")
        path, relative = _locked_portable_file(
            manifest_parent=manifest_parent,
            record=lod_record,
            label=f"{role}.lod_paths.{level}",
            allowed_suffixes=USD_SUFFIXES,
        )
        key = relative.as_posix().casefold()
        digest = _sha256(path)
        if key in seen_paths or digest in seen_hashes:
            raise ValueError(f"{role} HERO, MID and FAR wrappers must be distinct")
        seen_paths.add(key)
        seen_hashes.add(digest)
        summary[level] = {
            "path": relative.as_posix(),
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "lineage_id": lineage,
        }
    hero = lod_paths["HERO"]
    if (
        str(entry.get("path", "")).casefold()
        != str(hero.get("path", "")).casefold()
        or str(entry.get("sha256", "")).lower()
        != str(hero.get("sha256", "")).lower()
    ):
        raise ValueError(f"{role} primary wrapper must be the locked HERO wrapper")
    return summary


def _validate_asset_lod_library(
    *,
    payload: dict[str, Any],
    manifest_parent: Path,
) -> dict[str, dict[str, dict[str, object]]]:
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for role, entry in _asset_entries(payload):
        summary = _validate_asset_lod_paths(
            role=role,
            entry=entry,
            manifest_parent=manifest_parent,
        )
        for level in LOD_LEVELS:
            path_key = str(summary[level]["path"]).casefold()
            digest = str(summary[level]["sha256"])
            if path_key in seen_paths or digest in seen_hashes:
                raise ValueError(
                    "every HERO, MID and FAR representation in the bundle must "
                    "be a unique local USD wrapper"
                )
            seen_paths.add(path_key)
            seen_hashes.add(digest)
        summaries[role] = summary
    return summaries


def _normalize_manifest(
    *,
    payload: dict[str, Any],
    staging_manifest: Path,
    staging_root: Path,
    destination_root: Path,
    volume_root: Path,
) -> dict[str, Any]:
    manifest_parent = staging_manifest.parent
    _validate_pbr_materials(payload=payload, manifest_parent=manifest_parent)
    _validate_asset_lod_library(
        payload=payload,
        manifest_parent=manifest_parent,
    )
    destination_manifest_parent = destination_root / staging_manifest.relative_to(
        staging_root
    ).parent
    for role, entry in _asset_entries(payload):
        _portable_file(manifest_parent, entry.get("path"), label=f"{role} wrapper")
        source, _source_relative = _portable_file(
            manifest_parent,
            entry.get("source_cache_path"),
            label=f"{role} source cache",
        )
        files = entry.get("materialized_files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{role} must inventory every materialized dependency")
        normalized_files: list[dict[str, object]] = []
        seen: set[str] = set()
        source_seen = False
        for index, file_entry in enumerate(files):
            if not isinstance(file_entry, dict):
                raise ValueError(f"{role} materialized_files[{index}] must be an object")
            path, relative = _portable_file(
                manifest_parent,
                file_entry.get("path"),
                label=f"{role} materialized_files[{index}]",
            )
            key = relative.as_posix().casefold()
            if key in seen:
                raise ValueError(f"{role} contains duplicate dependency paths")
            seen.add(key)
            actual_sha = _sha256(path)
            actual_size = path.stat().st_size
            if (
                str(file_entry.get("sha256", "")).strip().lower() != actual_sha
                or file_entry.get("size_bytes") != actual_size
            ):
                raise ValueError(f"{role} dependency lock does not match {relative}")
            source_seen = source_seen or path.resolve() == source.resolve()
            final_path = destination_manifest_parent.joinpath(*relative.parts)
            if not _inside(destination_root, final_path):
                raise ValueError(f"{role} dependency escapes the installed bundle")
            normalized_files.append(
                {
                    "path": final_path.relative_to(volume_root).as_posix(),
                    "sha256": actual_sha,
                    "size_bytes": actual_size,
                }
            )
        portable_lock = hashlib.sha256(
            json.dumps(files, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if str(entry.get("content_lock_sha256", "")).strip().lower() != portable_lock:
            raise ValueError(f"{role} portable dependency content lock is invalid")
        if not source_seen:
            raise ValueError(f"{role} source cache is absent from materialized_files")
        entry["materialized_files"] = normalized_files
        entry["content_lock_sha256"] = hashlib.sha256(
            json.dumps(normalized_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return payload


def _inventory(root: Path) -> tuple[list[dict[str, object]], int]:
    entries: list[dict[str, object]] = []
    total = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if path.name == INSTALL_MARKER:
            continue
        size = path.stat().st_size
        total += size
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": size,
            }
        )
    return entries, total


def _validate_reuse(
    *,
    destination: Path,
    expected_sha256: str,
    manifest_relative: PurePosixPath,
) -> dict[str, Any]:
    marker_path = destination / INSTALL_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"asset bundle destination exists without a valid install marker: {destination}"
        ) from exc
    manifest = destination.joinpath(*manifest_relative.parts)
    if (
        not isinstance(marker, dict)
        or marker.get("state") != "ASSET_BUNDLE_INSTALLED"
        or marker.get("bundle_sha256") != expected_sha256
        or marker.get("manifest_relative") != manifest_relative.as_posix()
        or not manifest.is_file()
        or marker.get("runtime_manifest_sha256") != _sha256(manifest)
    ):
        raise RuntimeError("persisted asset bundle does not match its install marker")
    payload = _read_manifest(manifest)
    pbr_summary = _validate_pbr_materials(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    lod_summary = _validate_asset_lod_library(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    inventory, unpacked_bytes = _inventory(destination)
    if (
        marker.get("file_count") != len(inventory)
        or marker.get("unpacked_bytes") != unpacked_bytes
        or marker.get("content_inventory_sha256") != _canonical_sha256(inventory)
        or marker.get("pbr_materials_sha256") != _canonical_sha256(pbr_summary)
        or marker.get("asset_lod_library_sha256") != _canonical_sha256(lod_summary)
    ):
        raise RuntimeError("persisted asset bundle content drifted from its install marker")
    return marker


def install_asset_bundle(
    *,
    archive_path: Path,
    expected_sha256: str,
    volume_root: Path,
    destination_root: Path,
    manifest_relative: str,
    receipt_path: Path,
    max_files: int = DEFAULT_MAX_FILES,
    max_unpacked_bytes: int = DEFAULT_MAX_UNPACKED_BYTES,
    minimum_free_after_install_bytes: int = DEFAULT_MIN_FREE_AFTER_INSTALL_BYTES,
) -> dict[str, Any]:
    """Install an immutable portable asset bundle on a persistent volume."""

    volume, destination, expected = _validate_destination(
        volume_root=volume_root,
        destination_root=destination_root,
        expected_sha256=expected_sha256,
    )
    archive = archive_path.resolve()
    receipt = receipt_path.resolve()
    if (
        not archive.is_file()
        or archive.is_symlink()
        or not _inside(volume, archive)
        or not _inside(volume, receipt)
        or max_files <= 0
        or max_unpacked_bytes <= 0
        or minimum_free_after_install_bytes < 0
    ):
        raise ValueError(
            "asset bundle archive, receipt, and safety limits must stay inside the volume"
        )
    if _sha256(archive) != expected:
        raise RuntimeError("asset bundle archive SHA-256 mismatch")
    relative_manifest = _safe_relative_path(
        manifest_relative, label="asset bundle manifest"
    )
    if destination.exists():
        result = _validate_reuse(
            destination=destination,
            expected_sha256=expected,
            manifest_relative=relative_manifest,
        )
        _atomic_write_json(receipt, result)
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    declared_files, declared_unpacked_bytes = _archive_declared_usage(
        archive,
        destination=destination,
        max_files=max_files,
        max_unpacked_bytes=max_unpacked_bytes,
    )
    free_bytes = shutil.disk_usage(destination.parent).free
    required_free_bytes = (
        declared_unpacked_bytes + minimum_free_after_install_bytes
    )
    if free_bytes < required_free_bytes:
        raise RuntimeError(
            "persistent volume has insufficient free space for asset bundle "
            f"extraction and safety margin: free={free_bytes} "
            f"required={required_free_bytes}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        _extract_archive(
            archive,
            staging,
            max_files=max_files,
            max_unpacked_bytes=max_unpacked_bytes,
        )
        staging_manifest = staging.joinpath(*relative_manifest.parts)
        if not staging_manifest.is_file() or staging_manifest.is_symlink():
            raise ValueError(
                f"asset bundle manifest is absent: {relative_manifest.as_posix()}"
            )
        source_manifest_sha256 = _sha256(staging_manifest)
        payload = _read_manifest(staging_manifest)
        normalized = _normalize_manifest(
            payload=payload,
            staging_manifest=staging_manifest,
            staging_root=staging,
            destination_root=destination,
            volume_root=volume,
        )
        _atomic_write_json(staging_manifest, normalized)
        pbr_summary = _validate_pbr_materials(
            payload=normalized,
            manifest_parent=staging_manifest.parent,
        )
        lod_summary = _validate_asset_lod_library(
            payload=normalized,
            manifest_parent=staging_manifest.parent,
        )
        inventory, unpacked_bytes = _inventory(staging)
        marker = {
            "schema_version": 1,
            "state": "ASSET_BUNDLE_INSTALLED",
            "bundle_sha256": expected,
            "archive": archive.relative_to(volume).as_posix(),
            "install_root": destination.relative_to(volume).as_posix(),
            "manifest_relative": relative_manifest.as_posix(),
            "source_manifest_sha256": source_manifest_sha256,
            "runtime_manifest_sha256": _sha256(staging_manifest),
            "file_count": len(inventory),
            "unpacked_bytes": unpacked_bytes,
            "archive_declared_member_count": declared_files,
            "archive_declared_unpacked_bytes": declared_unpacked_bytes,
            "minimum_free_after_install_bytes": minimum_free_after_install_bytes,
            "content_inventory_sha256": _canonical_sha256(inventory),
            "pbr_material_roles": list(PBR_MATERIAL_ROLES),
            "pbr_materials_sha256": _canonical_sha256(pbr_summary),
            "asset_lod_levels": list(LOD_LEVELS),
            "asset_lod_library_sha256": _canonical_sha256(lod_summary),
            "proof_boundary": (
                "archive, PBR texture locks and relative-path installation only; "
                "native USD and geometric LOD quality must pass separately"
            ),
        }
        _atomic_write_json(staging / INSTALL_MARKER, marker)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _atomic_write_json(receipt, marker)
    return marker


def _bundle_locked_paths(
    *,
    bundle_root: Path,
    manifest_path: Path,
) -> set[Path]:
    root = bundle_root.resolve()
    manifest = manifest_path.resolve()
    if (
        not root.is_dir()
        or root.is_symlink()
        or not _inside(root, manifest)
    ):
        raise ValueError("native bundle manifest must stay below its install root")
    marker_path = root / INSTALL_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("native bundle validation requires its install marker") from exc
    manifest_relative = manifest.relative_to(root)
    _validate_reuse(
        destination=root,
        expected_sha256=str(marker.get("bundle_sha256", "")),
        manifest_relative=_safe_relative_path(
            manifest_relative.as_posix(),
            label="installed bundle manifest",
        ),
    )
    inventory, _total = _inventory(root)
    return {
        root.joinpath(
            *_safe_relative_path(
                str(entry["path"]),
                label="locked bundle inventory path",
            ).parts
        ).resolve()
        for entry in inventory
    }


def _native_used_layers(
    *,
    stage: object,
    bundle_root: Path,
    locked_paths: set[Path],
    label: str,
) -> list[str]:
    get_session_layer = getattr(stage, "GetSessionLayer", None)
    session_layer = get_session_layer() if callable(get_session_layer) else None
    session_identifier = (
        str(getattr(session_layer, "identifier", "") or "")
        if session_layer is not None
        else ""
    )
    used: list[str] = []
    for layer in stage.GetUsedLayers():
        identifier = str(layer.identifier)
        if (
            session_layer is not None
            and (
                layer is session_layer
                or (session_identifier and identifier == session_identifier)
            )
        ):
            continue
        real_path = str(getattr(layer, "realPath", "") or "")
        if "://" in identifier or "://" in real_path or not real_path:
            raise RuntimeError(f"{label} resolves a non-local USD layer: {identifier}")
        path = Path(real_path).resolve()
        if not _inside(bundle_root, path) or path not in locked_paths:
            raise RuntimeError(
                f"{label} resolves a USD layer outside its locked bundle: {path}"
            )
        used.append(str(path))
    if not used:
        raise RuntimeError(f"{label} has no locked USD layers")
    return sorted(used)


def _native_usd_metrics(
    path: Path,
    *,
    bundle_root: Path,
    locked_paths: set[Path],
) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "native HERO/MID/FAR validation requires the pinned Isaac/Kit Python"
        ) from exc

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to open local LOD wrapper as USD: {path}")
    used_layers = _native_used_layers(
        stage=stage,
        bundle_root=bundle_root,
        locked_paths=locked_paths,
        label="LOD wrapper",
    )

    forbidden_types = {
        "Capsule",
        "Cone",
        "Cube",
        "Cylinder",
        "Plane",
        "Sphere",
    }
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    geometry_prims = 0
    material_bound_prims = 0
    complexity = 0
    face_complexity = 0
    asset_dependencies: set[Path] = set()
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for prim in stage.Traverse():
        if not prim.IsActive() or not prim.IsDefined():
            continue
        if prim.GetTypeName() in forbidden_types:
            raise RuntimeError(
                f"primitive fallback is forbidden in {path}: {prim.GetPath()}"
            )
        for attribute in prim.GetAttributes():
            for asset_path in _asset_paths_from_value(attribute.Get()):
                asset_dependencies.add(
                    _resolved_asset_path(
                        asset_path=asset_path,
                        material_file=path,
                        bundle_root=bundle_root,
                        locked_paths=locked_paths,
                    )
                )
        point_count = 0
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            point_count = len(points) if points is not None else 0
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_complexity += (
                len(face_vertex_counts)
                if face_vertex_counts is not None
                else 0
            )
        elif prim.IsA(UsdGeom.BasisCurves):
            points = UsdGeom.BasisCurves(prim).GetPointsAttr().Get()
            point_count = len(points) if points is not None else 0
        elif prim.IsA(UsdGeom.Points):
            points = UsdGeom.Points(prim).GetPointsAttr().Get()
            point_count = len(points) if points is not None else 0
        elif prim.IsA(UsdGeom.PointInstancer):
            positions = UsdGeom.PointInstancer(prim).GetPositionsAttr().Get()
            point_count = len(positions) if positions is not None else 0
        if point_count <= 0:
            continue
        geometry_prims += 1
        complexity += point_count
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        if material and material.GetPrim().IsValid():
            material_bound_prims += 1
        aligned_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if aligned_range.IsEmpty():
            continue
        range_minimum = aligned_range.GetMin()
        range_maximum = aligned_range.GetMax()
        for axis in range(3):
            minimum[axis] = min(minimum[axis], float(range_minimum[axis]))
            maximum[axis] = max(maximum[axis], float(range_maximum[axis]))
    if (
        geometry_prims <= 0
        or complexity < 4
        or material_bound_prims <= 0
        or any(not math.isfinite(value) for value in (*minimum, *maximum))
    ):
        raise RuntimeError(
            f"LOD wrapper has no material-bound renderable geometry: {path}"
        )
    dimensions = [
        maximum[axis] - minimum[axis]
        for axis in range(3)
    ]
    if any(value <= 0.001 or not math.isfinite(value) for value in dimensions):
        raise RuntimeError(f"LOD wrapper has invalid world bounds: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "used_layer_count": len(used_layers),
        "used_layers": used_layers,
        "geometry_prim_count": geometry_prims,
        "material_bound_prim_count": material_bound_prims,
        "locked_asset_dependencies": sorted(
            str(dependency) for dependency in asset_dependencies
        ),
        "geometry_point_count": complexity,
        "geometry_face_count": face_complexity,
        "world_bounds": {
            "minimum": minimum,
            "maximum": maximum,
            "dimensions": dimensions,
        },
    }


def validate_native_lod_quality(
    *,
    manifest_path: Path,
    volume_root: Path,
    bundle_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Open every HERO/MID/FAR wrapper and prove geometric LOD behavior."""

    volume = volume_root.resolve()
    manifest = manifest_path.resolve()
    receipt = receipt_path.resolve()
    if (
        not manifest.is_file()
        or manifest.is_symlink()
        or not _inside(volume, manifest)
        or not _inside(volume, receipt)
    ):
        raise ValueError("native LOD manifest and receipt must stay inside the volume")
    payload = _read_manifest(manifest)
    bundle = bundle_root.resolve()
    if not _inside(volume, bundle):
        raise ValueError("native LOD bundle root must stay inside the volume")
    locked_paths = _bundle_locked_paths(
        bundle_root=bundle,
        manifest_path=manifest,
    )
    install_marker = json.loads(
        (bundle / INSTALL_MARKER).read_text(encoding="utf-8")
    )
    _validate_pbr_materials(payload=payload, manifest_parent=manifest.parent)
    structural = _validate_asset_lod_library(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    assets: list[dict[str, Any]] = []
    for role, entry in _asset_entries(payload):
        metrics: dict[str, dict[str, Any]] = {}
        for level in LOD_LEVELS:
            relative = _safe_relative_path(
                str(entry["lod_paths"][level]["path"]),
                label=f"{role}.lod_paths.{level}",
            )
            metrics[level] = _native_usd_metrics(
                manifest.parent.joinpath(*relative.parts),
                bundle_root=bundle,
                locked_paths=locked_paths,
            )
        face_complexities = [
            int(metrics[level].get("geometry_face_count", 0))
            for level in LOD_LEVELS
        ]
        if all(complexity > 0 for complexity in face_complexities):
            hero_complexity, mid_complexity, far_complexity = face_complexities
        elif any(face_complexities):
            raise RuntimeError(
                f"{role} has incomplete mesh-face metrics across HERO/MID/FAR"
            )
        else:
            hero_complexity, mid_complexity, far_complexity = [
                int(metrics[level]["geometry_point_count"])
                for level in LOD_LEVELS
            ]
        if not hero_complexity > mid_complexity > far_complexity:
            raise RuntimeError(
                f"{role} is not a real decreasing HERO/MID/FAR geometric LOD chain"
            )
        hero_dimensions = metrics["HERO"]["world_bounds"]["dimensions"]
        for level in ("MID", "FAR"):
            level_dimensions = metrics[level]["world_bounds"]["dimensions"]
            ratios = [
                level_dimensions[axis] / hero_dimensions[axis]
                for axis in range(3)
            ]
            if any(not 0.65 <= ratio <= 1.35 for ratio in ratios):
                raise RuntimeError(
                    f"{role} {level} bounds do not represent the same asset as HERO"
                )
        assets.append(
            {
                "role": role,
                "asset_id": entry.get("asset_id"),
                "lods": metrics,
            }
        )
    result = {
        "schema_version": 1,
        "state": "NATIVE_ASSET_LODS_VALIDATED",
        "manifest": manifest.relative_to(volume).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "lod_levels": list(LOD_LEVELS),
        "asset_count": len(assets),
        "structural_lod_library_sha256": _canonical_sha256(structural),
        "native_metrics_sha256": _canonical_sha256(assets),
        "bundle_content_inventory_sha256": install_marker[
            "content_inventory_sha256"
        ],
        "assets": assets,
    }
    _atomic_write_json(receipt, result)
    return result


def _texture_semantic(name: str) -> str | None:
    normalized = "".join(character for character in name.casefold() if character.isalnum())
    if any(token in normalized for token in ("basecolor", "diffuse", "albedo")):
        return "base_color"
    if "roughness" in normalized:
        return "roughness"
    if "normal" in normalized:
        return "normal"
    if any(token in normalized for token in ("displacement", "height")):
        return "displacement"
    return None


def _asset_paths_from_value(value: object) -> list[object]:
    try:
        from pxr import Sdf
    except ImportError as exc:
        raise RuntimeError(
            "native PBR validation requires the pinned Isaac/Kit Python"
        ) from exc
    if isinstance(value, Sdf.AssetPath):
        if not str(value.path or "") and not str(value.resolvedPath or ""):
            return []
        return [value]
    if isinstance(value, (list, tuple)):
        values: list[object] = []
        for item in value:
            values.extend(_asset_paths_from_value(item))
        return values
    return []


def _resolved_asset_path(
    *,
    asset_path: object,
    material_file: Path,
    bundle_root: Path,
    locked_paths: set[Path],
) -> Path:
    raw = str(getattr(asset_path, "path", "") or "")
    resolved = str(getattr(asset_path, "resolvedPath", "") or "")
    if "://" in raw or "://" in resolved:
        raise RuntimeError("remote PBR texture references are forbidden")
    if resolved:
        path = Path(resolved).resolve()
    else:
        relative = _safe_relative_path(raw, label="native PBR texture reference")
        path = material_file.parent.joinpath(*relative.parts).resolve()
    if not _inside(bundle_root, path) or path not in locked_paths:
        raise RuntimeError(
            f"PBR shader resolves a texture outside its locked bundle: {path}"
        )
    return path


def _connected_source(property_: object) -> object | None:
    from pxr import UsdShade

    try:
        source = UsdShade.ConnectableAPI.GetConnectedSource(property_)
    except Exception:
        return None
    if not source:
        return None
    connectable = source[0]
    if not connectable or not connectable.GetPrim().IsValid():
        return None
    return connectable


def _native_material_metrics(
    *,
    material_file: Path,
    material_prim_path: str,
    expected_textures: dict[str, Path],
    bundle_root: Path,
    locked_paths: set[Path],
) -> dict[str, Any]:
    try:
        from pxr import Usd, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "native PBR validation requires the pinned Isaac/Kit Python"
        ) from exc

    stage = Usd.Stage.Open(str(material_file), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to open PBR material USD: {material_file}")
    used_layers = _native_used_layers(
        stage=stage,
        bundle_root=bundle_root,
        locked_paths=locked_paths,
        label="PBR material",
    )
    prim = stage.GetPrimAtPath(material_prim_path)
    if not prim.IsValid() or not prim.IsA(UsdShade.Material):
        raise RuntimeError(
            f"PBR material prim is absent or not UsdShade.Material: "
            f"{material_file}{material_prim_path}"
        )
    material = UsdShade.Material(prim)
    reachable: dict[str, set[Path]] = {
        semantic: set() for semantic in (*PBR_REQUIRED_TEXTURES, *PBR_OPTIONAL_TEXTURES)
    }
    all_reachable_textures: set[Path] = set()
    visited: set[tuple[str, str | None]] = set()

    def walk(connectable: object, inherited_semantic: str | None) -> None:
        connectable_prim = connectable.GetPrim()
        key = (str(connectable_prim.GetPath()), inherited_semantic)
        if key in visited:
            return
        visited.add(key)
        for input_ in connectable.GetInputs():
            semantic = _texture_semantic(str(input_.GetBaseName())) or inherited_semantic
            for asset_path in _asset_paths_from_value(input_.Get()):
                resolved = _resolved_asset_path(
                    asset_path=asset_path,
                    material_file=material_file,
                    bundle_root=bundle_root,
                    locked_paths=locked_paths,
                )
                if not resolved.is_file() or resolved.is_symlink():
                    raise RuntimeError(
                        f"PBR shader references a missing texture: {resolved}"
                    )
                all_reachable_textures.add(resolved)
                if semantic is not None:
                    reachable[semantic].add(resolved)
            upstream = _connected_source(input_)
            if upstream is not None:
                walk(upstream, semantic)

    surface_roots = 0
    displacement_roots = 0
    for output in material.GetOutputs():
        output_name = str(output.GetBaseName())
        output_semantic = _texture_semantic(output_name)
        is_surface = "surface" in output_name.casefold()
        is_displacement = output_semantic == "displacement"
        if not is_surface and not is_displacement:
            continue
        source = _connected_source(output)
        if source is None:
            continue
        if is_surface:
            surface_roots += 1
        if is_displacement:
            displacement_roots += 1
        walk(source, "displacement" if is_displacement else None)
    if surface_roots <= 0:
        raise RuntimeError(f"PBR material has no connected surface output: {material_file}")
    for semantic, expected in expected_textures.items():
        if expected.resolve() not in reachable[semantic]:
            raise RuntimeError(
                f"PBR {semantic} texture is not connected to the matching "
                f"surface branch: {material_file}"
            )
    if "displacement" in expected_textures and displacement_roots <= 0:
        raise RuntimeError(
            f"PBR displacement texture has no connected displacement output: {material_file}"
        )
    return {
        "material_file": str(material_file),
        "material_file_sha256": _sha256(material_file),
        "material_prim_path": material_prim_path,
        "used_layer_count": len(used_layers),
        "used_layers": used_layers,
        "reachable_shader_prim_count": len({path for path, _semantic in visited}),
        "connected_surface_output_count": surface_roots,
        "connected_displacement_output_count": displacement_roots,
        "connected_textures": {
            semantic: sorted(str(path) for path in paths)
            for semantic, paths in reachable.items()
            if paths
        },
        "all_reachable_textures": sorted(
            str(path) for path in all_reachable_textures
        ),
    }


def validate_native_pbr_quality(
    *,
    manifest_path: Path,
    volume_root: Path,
    bundle_root: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Prove that locked PBR textures participate in native shader graphs."""

    volume = volume_root.resolve()
    manifest = manifest_path.resolve()
    receipt = receipt_path.resolve()
    if (
        not manifest.is_file()
        or manifest.is_symlink()
        or not _inside(volume, manifest)
        or not _inside(volume, receipt)
    ):
        raise ValueError(
            "native PBR manifest and receipt must stay inside the persistent volume"
        )
    payload = _read_manifest(manifest)
    bundle = bundle_root.resolve()
    if not _inside(volume, bundle):
        raise ValueError("native PBR bundle root must stay inside the volume")
    locked_paths = _bundle_locked_paths(
        bundle_root=bundle,
        manifest_path=manifest,
    )
    install_marker = json.loads(
        (bundle / INSTALL_MARKER).read_text(encoding="utf-8")
    )
    structural = _validate_pbr_materials(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    materials: dict[str, dict[str, Any]] = {}
    for role in PBR_MATERIAL_ROLES:
        entry = payload["pbr_materials"][role]
        material_relative = _safe_relative_path(
            str(entry["material_file"]["path"]),
            label=f"pbr_materials.{role}.material_file",
        )
        expected_textures = {
            texture_role: manifest.parent.joinpath(
                *_safe_relative_path(
                    str(texture_record["path"]),
                    label=f"pbr_materials.{role}.textures.{texture_role}",
                ).parts
            )
            for texture_role, texture_record in entry["textures"].items()
        }
        materials[role] = _native_material_metrics(
            material_file=manifest.parent.joinpath(*material_relative.parts),
            material_prim_path=str(entry["material_prim_path"]),
            expected_textures=expected_textures,
            bundle_root=bundle,
            locked_paths=locked_paths,
        )
    result = {
        "schema_version": 1,
        "state": "NATIVE_PBR_MATERIALS_VALIDATED",
        "manifest": manifest.relative_to(volume).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "material_roles": list(PBR_MATERIAL_ROLES),
        "structural_materials_sha256": _canonical_sha256(structural),
        "native_material_metrics_sha256": _canonical_sha256(materials),
        "bundle_content_inventory_sha256": install_marker[
            "content_inventory_sha256"
        ],
        "materials": materials,
    }
    _atomic_write_json(receipt, result)
    return result


def verify_native_quality_receipts(
    *,
    manifest_path: Path,
    volume_root: Path,
    bundle_root: Path,
    native_lod_receipt: Path,
    native_pbr_receipt: Path,
) -> dict[str, Any]:
    """Fail closed when a later pilot run reuses stale native-quality gates."""

    volume = volume_root.resolve()
    manifest = manifest_path.resolve()
    bundle = bundle_root.resolve()
    lod_path = native_lod_receipt.resolve()
    pbr_path = native_pbr_receipt.resolve()
    for path, label in (
        (manifest, "bundle manifest"),
        (lod_path, "native LOD receipt"),
        (pbr_path, "native PBR receipt"),
    ):
        if not path.is_file() or path.is_symlink() or not _inside(volume, path):
            raise RuntimeError(f"{label} is absent from the persistent volume")
    locked_paths = _bundle_locked_paths(
        bundle_root=bundle,
        manifest_path=manifest,
    )
    install_marker = json.loads(
        (bundle / INSTALL_MARKER).read_text(encoding="utf-8")
    )
    payload = _read_manifest(manifest)
    pbr_structural = _validate_pbr_materials(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    lod_structural = _validate_asset_lod_library(
        payload=payload,
        manifest_parent=manifest.parent,
    )
    try:
        lod = json.loads(lod_path.read_text(encoding="utf-8"))
        pbr = json.loads(pbr_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("native asset quality receipt is not valid JSON") from exc
    manifest_sha = _sha256(manifest)
    content_sha = install_marker.get("content_inventory_sha256")
    lod_assets = lod.get("assets") if isinstance(lod, dict) else None
    expected_roles = {
        role for role, _entry in _asset_entries(payload)
    }
    if (
        not isinstance(lod, dict)
        or lod.get("state") != "NATIVE_ASSET_LODS_VALIDATED"
        or lod.get("manifest_sha256") != manifest_sha
        or lod.get("bundle_content_inventory_sha256") != content_sha
        or lod.get("lod_levels") != list(LOD_LEVELS)
        or lod.get("asset_count") != len(_asset_entries(payload))
        or lod.get("structural_lod_library_sha256")
        != _canonical_sha256(lod_structural)
        or not isinstance(lod_assets, list)
        or lod.get("native_metrics_sha256") != _canonical_sha256(lod_assets)
        or {str(asset.get("role")) for asset in lod_assets if isinstance(asset, dict)}
        != expected_roles
    ):
        raise RuntimeError("native HERO/MID/FAR receipt is stale or incomplete")
    for asset in lod_assets:
        if not isinstance(asset, dict):
            raise RuntimeError("native HERO/MID/FAR receipt has an invalid asset")
        role = str(asset["role"])
        lods = asset.get("lods")
        if not isinstance(lods, dict) or set(lods) != set(LOD_LEVELS):
            raise RuntimeError(f"{role} native LOD metrics are incomplete")
        point_complexities: list[int] = []
        face_complexities: list[int] = []
        for level in LOD_LEVELS:
            metrics = lods[level]
            if not isinstance(metrics, dict):
                raise RuntimeError(f"{role} {level} native metrics are invalid")
            wrapper = Path(str(metrics.get("path", ""))).resolve()
            used_layers = metrics.get("used_layers")
            if (
                wrapper not in locked_paths
                or str(metrics.get("sha256", "")).lower() != _sha256(wrapper)
                or not isinstance(used_layers, list)
                or not used_layers
                or any(Path(str(path)).resolve() not in locked_paths for path in used_layers)
                or int(metrics.get("geometry_prim_count", 0)) <= 0
                or int(metrics.get("material_bound_prim_count", 0)) <= 0
            ):
                raise RuntimeError(f"{role} {level} native metrics are stale")
            point_complexities.append(
                int(metrics.get("geometry_point_count", 0))
            )
            face_complexities.append(
                int(metrics.get("geometry_face_count", 0))
            )
        if all(complexity > 0 for complexity in face_complexities):
            complexities = face_complexities
            minimum_complexity = 1
        elif any(face_complexities):
            raise RuntimeError(
                f"{role} native mesh-face LOD metrics are incomplete"
            )
        else:
            complexities = point_complexities
            minimum_complexity = 4
        if not (
            complexities[0]
            > complexities[1]
            > complexities[2]
            >= minimum_complexity
        ):
            raise RuntimeError(f"{role} native geometric LOD chain is not decreasing")

    pbr_materials = pbr.get("materials") if isinstance(pbr, dict) else None
    if (
        not isinstance(pbr, dict)
        or pbr.get("state") != "NATIVE_PBR_MATERIALS_VALIDATED"
        or pbr.get("manifest_sha256") != manifest_sha
        or pbr.get("bundle_content_inventory_sha256") != content_sha
        or pbr.get("material_roles") != list(PBR_MATERIAL_ROLES)
        or pbr.get("structural_materials_sha256")
        != _canonical_sha256(pbr_structural)
        or not isinstance(pbr_materials, dict)
        or set(pbr_materials) != set(PBR_MATERIAL_ROLES)
        or pbr.get("native_material_metrics_sha256")
        != _canonical_sha256(pbr_materials)
    ):
        raise RuntimeError("native PBR material receipt is stale or incomplete")
    for role in PBR_MATERIAL_ROLES:
        metrics = pbr_materials[role]
        expected = payload["pbr_materials"][role]
        if not isinstance(metrics, dict):
            raise RuntimeError(f"{role} native PBR metrics are invalid")
        material_file = Path(str(metrics.get("material_file", ""))).resolve()
        used_layers = metrics.get("used_layers")
        connected = metrics.get("connected_textures")
        if (
            material_file not in locked_paths
            or str(metrics.get("material_file_sha256", "")).lower()
            != _sha256(material_file)
            or metrics.get("material_prim_path")
            != expected.get("material_prim_path")
            or int(metrics.get("connected_surface_output_count", 0)) <= 0
            or not isinstance(used_layers, list)
            or not used_layers
            or any(Path(str(path)).resolve() not in locked_paths for path in used_layers)
            or not isinstance(connected, dict)
        ):
            raise RuntimeError(f"{role} native PBR metrics are stale")
        for semantic in expected["textures"]:
            expected_path = manifest.parent.joinpath(
                *_safe_relative_path(
                    str(expected["textures"][semantic]["path"]),
                    label=f"pbr_materials.{role}.textures.{semantic}",
                ).parts
            ).resolve()
            connected_paths = connected.get(semantic)
            if (
                expected_path not in locked_paths
                or not isinstance(connected_paths, list)
                or expected_path
                not in {Path(str(path)).resolve() for path in connected_paths}
            ):
                raise RuntimeError(
                    f"{role} {semantic} native PBR connection is stale"
                )
    return {
        "state": "NATIVE_ASSET_BUNDLE_RECEIPTS_CURRENT",
        "manifest_sha256": manifest_sha,
        "bundle_content_inventory_sha256": content_sha,
        "native_lod_receipt_sha256": _sha256(lod_path),
        "native_pbr_receipt_sha256": _sha256(pbr_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install a hash-locked portable FireViewer USD asset bundle"
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--volume-root", required=True, type=Path)
    parser.add_argument("--destination-root", required=True, type=Path)
    parser.add_argument("--manifest-relative", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    parser.add_argument(
        "--max-unpacked-gib",
        type=float,
        default=DEFAULT_MAX_UNPACKED_BYTES / 1024**3,
    )
    parser.add_argument(
        "--minimum-free-after-install-gib",
        type=float,
        default=DEFAULT_MIN_FREE_AFTER_INSTALL_BYTES / 1024**3,
    )
    parser.add_argument("--native-lod-receipt", type=Path)
    parser.add_argument("--native-pbr-receipt", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = install_asset_bundle(
        archive_path=args.archive,
        expected_sha256=args.sha256,
        volume_root=args.volume_root,
        destination_root=args.destination_root,
        manifest_relative=args.manifest_relative,
        receipt_path=args.receipt,
        max_files=args.max_files,
        max_unpacked_bytes=int(args.max_unpacked_gib * 1024**3),
        minimum_free_after_install_bytes=int(
            args.minimum_free_after_install_gib * 1024**3
        ),
    )
    if args.native_lod_receipt is not None:
        manifest_relative = _safe_relative_path(
            args.manifest_relative,
            label="asset bundle manifest",
        )
        result["native_lod_quality"] = validate_native_lod_quality(
            manifest_path=args.destination_root.joinpath(*manifest_relative.parts),
            volume_root=args.volume_root,
            bundle_root=args.destination_root,
            receipt_path=args.native_lod_receipt,
        )
    if args.native_pbr_receipt is not None:
        manifest_relative = _safe_relative_path(
            args.manifest_relative,
            label="asset bundle manifest",
        )
        result["native_pbr_quality"] = validate_native_pbr_quality(
            manifest_path=args.destination_root.joinpath(*manifest_relative.parts),
            volume_root=args.volume_root,
            bundle_root=args.destination_root,
            receipt_path=args.native_pbr_receipt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
