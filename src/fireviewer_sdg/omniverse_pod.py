"""Fail-closed RunPod contract for the FireViewer Omniverse campaign.

This module prepares evidence and gates.  It deliberately does not start a
fire simulation.  A scene can move from automated validation to human review,
but only a human-authored acceptance receipt bound to the current runtime,
catalog, assets and USD root can make the future simulation gate pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from fireviewer_sdg.simready_assets import (
    MANIFEST_PROFILE,
    MANIFEST_SCHEMA_VERSION,
    PHOTOREAL_FAMILY_MINIMUMS,
    PHOTOREAL_LIBRARY_POLICY,
    PHOTOREAL_MIN_LOD_LEVELS,
)
from fireviewer_sdg.zone_scenes import (
    ZONE_ORDER,
    _zone_manifest,
    _zone_rows,
    validate_catalog,
)


SCHEMA_VERSION = 1
CAMPAIGN_ID = "fireviewer-omniverse-20-photoreal-simulations-v1"
EXPECTED_SIMULATION_COUNT = 20
BASE_SCENE_COUNT = 4
VARIANTS_PER_BASE = 5
DEFAULT_BASE_ZONES = tuple(ZONE_ORDER[:BASE_SCENE_COUNT])
KIT_TEMPLATE_COMMIT = "483e364a4176f102f2d3c3aaf9f301a103d61d69"
KIT_TEMPLATE_ORIGIN = "https://github.com/NVIDIA-Omniverse/kit-app-template.git"
MIN_DRIVER_VERSION = (550, 54, 15)
BLACKWELL_MIN_DRIVER_VERSION = (570, 158, 1)
PRODUCTION_MIN_VRAM_MIB = 90_000
PRODUCTION_MIN_SYSTEM_RAM_MIB = 138_000
PRODUCTION_MIN_STORAGE_BYTES = 1_500_000_000_000
PRODUCTION_GPU_NAME = "RTX PRO 6000 Blackwell Server Edition"
SIMREADY_PROFILE = MANIFEST_PROFILE
MATERIALIZED_ASSET_MODE = "materialized_photoreal_asset_library_v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMPUTE_ONLY_GPU_MARKERS = ("A100", "H100", "H200", "B100", "B200")
RENDER_GPU_MARKERS = ("RTX", "L40", "A40", "A6000")
NETWORK_FILESYSTEM_TYPES = frozenset(
    {
        "9p",
        "ceph",
        "cifs",
        "fuse.rclone",
        "fuse.s3fs",
        "nfs",
        "nfs4",
        "smb3",
    }
)
REVIEW_ACKNOWLEDGEMENT = "I inspected the scene in FireViewer USD Composer"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
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
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_gpu_name(value: object) -> str:
    normalized = " ".join(str(value).strip().casefold().split())
    if normalized.startswith("nvidia "):
        normalized = normalized.removeprefix("nvidia ")
    return normalized


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _inside(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _relative_file(
    *,
    root: Path,
    base: Path,
    raw: object,
    label: str,
) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"{label} path is required")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must use a safe relative path")
    path = (base / relative).resolve()
    if not _inside(root, path) or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is not a materialized file inside the volume: {path}")
    return path


def _asset_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("SimReady manifest requires an environment object")
    entries: list[tuple[str, dict[str, Any]]] = []
    if set(environment) != set(PHOTOREAL_FAMILY_MINIMUMS):
        raise ValueError(
            "photoreal environment must contain exactly vegetation and buildings"
        )
    for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items():
        families = environment.get(kind)
        if not isinstance(families, dict):
            raise ValueError(f"environment.{kind} must be an object")
        if set(families) != set(family_minimums):
            raise ValueError(
                f"environment.{kind} must contain exactly "
                f"{', '.join(family_minimums)}"
            )
        for family, minimum in family_minimums.items():
            assets = families.get(family)
            role = f"{kind}.{family}"
            if not isinstance(assets, list) or len(assets) < minimum:
                raise ValueError(
                    f"SimReady manifest requires at least {minimum} assets in {role}"
                )
            for index, item in enumerate(assets):
                if not isinstance(item, dict):
                    raise ValueError(f"{role}[{index}] must be an object")
                entries.append((f"{role}[{index}]", item))
    return entries


_DIMENSION_LIMITS_M: dict[str, tuple[float, float, float, float]] = {
    "vegetation.trees": (0.1, 80.0, 2.0, 90.0),
    "vegetation.shrubs": (0.05, 30.0, 0.1, 15.0),
    "vegetation.understory": (0.01, 20.0, 0.01, 6.0),
    "buildings.habitat": (1.5, 250.0, 2.0, 100.0),
    "buildings.agricultural": (2.0, 450.0, 2.0, 100.0),
    "buildings.industrial": (2.0, 700.0, 2.0, 150.0),
    "buildings.annex": (0.5, 200.0, 1.0, 60.0),
}


def _validate_visual_asset_contract(
    *,
    role: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    family = role.split("[", 1)[0]
    if entry.get("family") != family:
        raise ValueError(f"{role} family metadata does not match its manifest slot")
    asset_id = str(entry.get("asset_id", "")).strip()
    if not asset_id.startswith(f"{family}:"):
        raise ValueError(f"{role} has no family-scoped asset_id")
    identity = entry.get("identity")
    if (
        not isinstance(identity, dict)
        or not str(identity.get("source_name", "")).strip()
        or not str(identity.get("source_identity", "")).strip()
    ):
        raise ValueError(f"{role} has incomplete visual identity metadata")

    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{role} provenance must be structured")
    if (
        not str(provenance.get("provider", "")).strip()
        or not str(provenance.get("provider_version", "")).strip()
    ):
        raise ValueError(f"{role} provenance provider identity is incomplete")
    source_uri = str(provenance.get("source_uri", "")).strip()
    source_url = urlparse(source_uri)
    if source_url.scheme != "https" or not source_url.hostname:
        raise ValueError(f"{role} source provenance must be HTTPS")
    provider_hash = str(provenance.get("provider_hash", "")).strip().lower()
    if not SHA256_RE.fullmatch(provider_hash):
        raise ValueError(f"{role} provenance provider_hash must be SHA-256")
    if (
        str(entry.get("source_uri", "")).strip() != source_uri
        or str(entry.get("provider_hash", "")).strip().lower() != provider_hash
    ):
        raise ValueError(f"{role} structured provenance disagrees with its lock")

    licence = entry.get("license")
    licence_url = urlparse(str(licence.get("uri", ""))) if isinstance(
        licence, dict
    ) else urlparse("")
    if (
        not isinstance(licence, dict)
        or not str(licence.get("id", "")).strip()
        or licence_url.scheme != "https"
        or not licence_url.hostname
    ):
        raise ValueError(f"{role} requires an explicit HTTPS license contract")
    if str(entry.get("license_id", "")).strip() != licence.get("id"):
        raise ValueError(f"{role} license metadata disagrees with its lock")

    dimensions = entry.get("native_dimensions_m")
    if not isinstance(dimensions, dict):
        raise ValueError(f"{role} native_dimensions_m must be an object")
    try:
        x = float(dimensions["x"])
        y = float(dimensions["y"])
        z = float(dimensions["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} native dimensions are invalid") from exc
    horizontal_min, horizontal_max, vertical_min, vertical_max = (
        _DIMENSION_LIMITS_M[family]
    )
    if (
        any(not math.isfinite(value) for value in (x, y, z))
        or not horizontal_min <= x <= horizontal_max
        or not horizontal_min <= y <= horizontal_max
        or not vertical_min <= z <= vertical_max
    ):
        raise ValueError(f"{role} native dimensions are implausible for {family}")
    try:
        meters_per_unit = float(entry["source_meters_per_unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} source unit metadata is invalid") from exc
    if (
        not math.isfinite(meters_per_unit)
        or not 0.000001 <= meters_per_unit <= 1000.0
        or entry.get("source_up_axis") != "Z"
    ):
        raise ValueError(f"{role} source unit/up-axis metadata is invalid")

    anchor = entry.get("ground_anchor_m")
    anchor_validation = entry.get("anchor_validation")
    if (
        not isinstance(anchor, list)
        or len(anchor) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in anchor
        )
        or not isinstance(anchor_validation, dict)
        or anchor_validation.get("state") != "passed"
        or anchor_validation.get("policy") != "native_bbox_bottom_center"
    ):
        raise ValueError(f"{role} has no validated native ground anchor")

    placement = entry.get("placement")
    try:
        minimum_scale = float(placement["minimum_uniform_scale"])
        maximum_scale = float(placement["maximum_uniform_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} placement scale bounds are invalid") from exc
    if (
        not isinstance(placement, dict)
        or placement.get("grounding") != "native_anchor"
        or placement.get("scale_policy") != "uniform_only"
        or placement.get("non_uniform_scale_allowed") is not False
        or minimum_scale < 0.5
        or maximum_scale > 1.5
        or minimum_scale > maximum_scale
    ):
        raise ValueError(
            f"{role} must preserve native proportions with bounded uniform scaling"
        )

    lod = entry.get("lod")
    minimum_lods = PHOTOREAL_MIN_LOD_LEVELS[family]
    if not isinstance(lod, dict) or lod.get("state") != "passed":
        raise ValueError(f"{role} has no passed LOD validation")
    levels = lod.get("levels")
    if (
        not isinstance(levels, list)
        or len(levels) < minimum_lods
        or any(not str(level).strip() for level in levels)
        or len({str(level) for level in levels}) != len(levels)
        or lod.get("level_count") != len(levels)
        or lod.get("strategy") not in {
            "native_variant_set",
            "native_prim_hierarchy",
            "scene_optimizer_decimateMeshes",
        }
    ):
        raise ValueError(
            f"{role} requires at least {minimum_lods} distinct native LOD levels"
        )

    materials = entry.get("materials")
    try:
        material_count = int(materials["material_prim_count"])
        bound_material_count = int(materials["bound_material_prim_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} material metrics are invalid") from exc
    if (
        not isinstance(materials, dict)
        or materials.get("state") != "passed"
        or material_count < 1
        or bound_material_count < 1
        or list(materials.get("unresolved_dependencies") or [])
    ):
        raise ValueError(f"{role} has no passed bound-material validation")

    contract = {
        key: entry.get(key)
        for key in (
            "native_dimensions_m",
            "ground_anchor_m",
            "anchor_validation",
            "lod",
            "materials",
            "placement",
        )
    }
    expected_metadata_sha = str(
        entry.get("metadata_validation_sha256", "")
    ).strip().lower()
    if (
        not SHA256_RE.fullmatch(expected_metadata_sha)
        or _canonical_sha256(contract) != expected_metadata_sha
    ):
        raise RuntimeError(f"{role} visual metadata validation hash drifted")
    if entry.get("quality_validation") != "native_metadata_passed":
        raise RuntimeError(f"{role} native metadata quality has not passed")
    return {
        "asset_id": asset_id,
        "family": family,
        "native_dimensions_m": {"x": x, "y": y, "z": z},
        "lod_level_count": len(levels),
        "material_prim_count": material_count,
        "bound_material_prim_count": bound_material_count,
        "metadata_validation_sha256": expected_metadata_sha,
    }


def _verify_artifact_entry(
    entry: object,
    *,
    base: Path,
    allowed_root: Path,
    label: str,
    path_key: str = "path",
    sha_key: str = "sha256",
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"{label} must be an object")
    raw_path = str(entry.get(path_key, "")).strip()
    expected_sha = str(entry.get(sha_key, "")).strip().lower()
    if not raw_path or not SHA256_RE.fullmatch(expected_sha):
        raise ValueError(f"{label} must contain a path and SHA-256")
    candidate = Path(raw_path)
    path = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    if (
        not _inside(allowed_root, path)
        or not path.is_file()
        or path.is_symlink()
    ):
        raise RuntimeError(f"{label} escaped or is absent: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")
    return {
        "path": path.relative_to(allowed_root.resolve()).as_posix(),
        "sha256": actual_sha,
        "size_bytes": path.stat().st_size,
    }


def _verify_build_artifacts(
    *,
    build_payload: dict[str, Any],
    build_receipt: Path,
    root_usd: Path,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    build = build_receipt.resolve()
    root = root_usd.resolve()
    build_root = build.parent
    if root.parent != build_root:
        raise RuntimeError("root USD and build receipt must share the build directory")
    scene_root = build_root.parent
    artifact_root = (
        allowed_root.resolve() if allowed_root is not None else scene_root
    )
    if not _inside(artifact_root, scene_root):
        raise RuntimeError("scene build escapes its allowed artifact root")
    variant_build = build_payload.get("scene_kind") == "fictive_variant"
    verified: list[dict[str, Any]] = []

    root_entry = _verify_artifact_entry(
        build_payload.get("root_usd"),
        base=scene_root,
        allowed_root=artifact_root,
        label="build root USD",
    )
    if root_entry["sha256"] != _sha256(root):
        raise RuntimeError("build artifact inventory references a different root USD")
    verified.append(root_entry)
    collection_names = (
        (
            "payloads",
            "detail_payloads",
            "detail_mid_payloads",
            "detail_far_payloads",
            "water_payloads",
        )
        if variant_build
        else ("payloads", "aggregates_5km")
    )
    for collection_name in collection_names:
        collection = build_payload.get(collection_name)
        if not isinstance(collection, list) or not collection:
            raise RuntimeError(f"build receipt has no {collection_name} artifacts")
        if (
            variant_build
            and collection_name != "water_payloads"
            and len(collection) != 400
        ):
            raise RuntimeError(
                f"variant build must lock exactly 400 {collection_name} artifacts"
            )
        collection_paths: set[str] = set()
        for index, entry in enumerate(collection):
            raw_path = (
                str(entry.get("path", "")).strip()
                if isinstance(entry, dict)
                else ""
            )
            if not raw_path or raw_path in collection_paths:
                raise RuntimeError(
                    f"{collection_name} artifact paths must be present and unique"
                )
            collection_paths.add(raw_path)
            verified.append(
                _verify_artifact_entry(
                    entry,
                    base=scene_root,
                    allowed_root=artifact_root,
                    label=f"{collection_name}[{index}]",
                )
            )
    required_single_artifacts = (
        ("cameras", "asset_lock")
        if variant_build
        else ("cameras", "source_lock", "asset_lock")
    )
    for key in required_single_artifacts:
        verified.append(
            _verify_artifact_entry(
                build_payload.get(key),
                base=scene_root,
                allowed_root=artifact_root,
                label=f"build {key}",
            )
        )
    if variant_build:
        ground = build_payload.get("ground_material")
        if (
            not isinstance(ground, dict)
            or ground.get("topology")
            != "payload_tiled_materials_shared_pbr_library"
            or ground.get("binding_scope")
            != "per_terrain_tile_stronger_than_descendants"
        ):
            raise RuntimeError(
                "variant build has no tiled object-free ground contract"
            )
        verified.append(
            _verify_artifact_entry(
                ground.get("index"),
                base=scene_root,
                allowed_root=artifact_root,
                label="build ground_material.index",
            )
        )
        ground_tiles = ground.get("tile_material_payloads")
        if not isinstance(ground_tiles, list) or len(ground_tiles) != 400:
            raise RuntimeError(
                "variant build must lock exactly 400 ground material payloads"
            )
        ground_ids: set[str] = set()
        ground_paths: set[str] = set()
        for index, entry in enumerate(ground_tiles):
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"ground material payload[{index}] is malformed"
                )
            tile_id = str(entry.get("tile_id", "")).strip()
            raw_path = str(entry.get("path", "")).strip()
            bounds = entry.get("tile_bounds_m")
            if (
                not tile_id
                or tile_id in ground_ids
                or not raw_path
                or raw_path in ground_paths
                or not isinstance(bounds, list)
                or len(bounds) != 4
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in bounds
                )
                or float(bounds[0]) >= float(bounds[2])
                or float(bounds[1]) >= float(bounds[3])
            ):
                raise RuntimeError(
                    "variant ground material tile IDs, paths and bounds must "
                    "be complete and unique"
                )
            ground_ids.add(tile_id)
            ground_paths.add(raw_path)
            verified.append(
                _verify_artifact_entry(
                    entry,
                    base=scene_root,
                    allowed_root=artifact_root,
                    label=f"ground material payload[{index}]",
                )
            )
        layers = build_payload.get("layers")
        terrain_layer = (
            layers.get("terrain") if isinstance(layers, dict) else None
        )
        if (
            not isinstance(terrain_layer, dict)
            or terrain_layer.get("ground_material_topology")
            != ground.get("topology")
            or terrain_layer.get("ground_material_payload_count") != 400
            or terrain_layer.get("global_ground_material_binding") is not False
        ):
            raise RuntimeError(
                "variant terrain layer does not prove 400 payload-streamed "
                "ground bindings"
            )
        identity = build_payload.get("identity_contract")
        if (
            not isinstance(identity, dict)
            or identity.get("numeric_ids_preserved") is not True
            or identity.get("stable_ids_preserved") is not True
            or identity.get(
                "source_namespace_may_differ_from_destination_tile"
            )
            is not True
            or not SHA256_RE.fullmatch(
                str(identity.get("source_identity_sha256", "")).lower()
            )
            or identity.get("source_identity_sha256")
            != identity.get("authored_identity_sha256")
        ):
            raise RuntimeError(
                "variant build has no source-preserving identity contract"
            )
    layers = build_payload.get("layers")
    if isinstance(layers, dict):
        optional_artifacts = (
            ("terrain", "visual_surface"),
            ("imagery", "continuous_texture"),
        )
        for layer_name, artifact_name in optional_artifacts:
            layer = layers.get(layer_name)
            entry = layer.get(artifact_name) if isinstance(layer, dict) else None
            if entry is not None:
                verified.append(
                    _verify_artifact_entry(
                        entry,
                        base=scene_root,
                        allowed_root=artifact_root,
                        label=f"layer {layer_name}.{artifact_name}",
                    )
                )
    asset_lock = build_payload.get("asset_lock")
    locked_assets = asset_lock.get("assets") if isinstance(asset_lock, dict) else None
    if not isinstance(locked_assets, list) or not locked_assets:
        raise RuntimeError("build receipt has no locked asset inventory")
    for index, asset in enumerate(locked_assets):
        if not isinstance(asset, dict) or not asset.get("packaged_path"):
            continue
        verified.append(
            _verify_artifact_entry(
                asset,
                base=build_root,
                allowed_root=artifact_root,
                label=f"packaged build asset[{index}]",
                path_key="packaged_path",
                sha_key="packaged_sha256",
            )
        )
    return {
        "artifact_count": len(verified),
        "artifact_content_sha256": _canonical_sha256(verified),
    }


def _verify_scene_validation_layers(
    *,
    scene_payload: dict[str, Any],
    volume_root: Path,
) -> dict[str, Any]:
    used_layers = scene_payload.get("used_layers")
    if not isinstance(used_layers, list) or not used_layers:
        raise RuntimeError("scene auto-validation has no composed layer inventory")
    verified = [
        _verify_artifact_entry(
            entry,
            base=volume_root,
            allowed_root=volume_root,
            label=f"scene used_layers[{index}]",
        )
        for index, entry in enumerate(used_layers)
    ]
    return {
        "layer_count": len(verified),
        "layer_content_sha256": _canonical_sha256(verified),
    }


def validate_materialized_assets(
    *,
    manifest_path: Path,
    volume_root: Path,
) -> dict[str, Any]:
    """Require local, hash-locked USD wrappers and their materialized sources."""

    volume = volume_root.resolve()
    manifest = manifest_path.resolve()
    if not _inside(volume, manifest) or not manifest.is_file():
        raise RuntimeError("SimReady asset manifest must be a file inside the volume")
    payload = _read_json(manifest, label="SimReady asset manifest")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported SimReady asset manifest schema_version")
    if payload.get("profile") != SIMREADY_PROFILE:
        raise ValueError(f"SimReady asset profile must be {SIMREADY_PROFILE}")
    discovery = payload.get("discovery")
    if not isinstance(discovery, dict):
        raise ValueError("SimReady manifest requires discovery metadata")
    if discovery.get("mode") != MATERIALIZED_ASSET_MODE:
        raise RuntimeError(
            "remote-reference assets are forbidden for the pod; every USD and dependency "
            "must be materialized"
        )
    if list(discovery.get("missing_environment") or []):
        raise RuntimeError("SimReady environment asset discovery is incomplete")
    if payload.get("family_minimums") != PHOTOREAL_FAMILY_MINIMUMS:
        raise ValueError("SimReady photoreal family minimums were weakened")
    if payload.get("library_policy") != PHOTOREAL_LIBRARY_POLICY:
        raise ValueError("SimReady photoreal library policy was weakened")

    locked: list[dict[str, Any]] = []
    seen_wrapper_hashes: set[str] = set()
    seen_source_paths: set[str] = set()
    seen_asset_ids: set[str] = set()
    for role, entry in _asset_entries(payload):
        visual_contract = _validate_visual_asset_contract(
            role=role,
            entry=entry,
        )
        asset_id = visual_contract["asset_id"]
        if asset_id in seen_asset_ids:
            raise ValueError("environment asset_id values must be unique")
        seen_asset_ids.add(asset_id)
        wrapper = _relative_file(
            root=volume,
            base=manifest.parent,
            raw=entry.get("path"),
            label=f"{role} wrapper",
        )
        expected = str(entry.get("sha256", "")).lower()
        actual = _sha256(wrapper)
        if not SHA256_RE.fullmatch(expected) or expected != actual:
            raise RuntimeError(f"{role} wrapper SHA-256 mismatch")
        wrapper_source = wrapper.read_text(encoding="utf-8", errors="replace")
        if "@http://" in wrapper_source or "@https://" in wrapper_source:
            raise RuntimeError(f"{role} wrapper still references a remote asset")
        source = _relative_file(
            root=volume,
            base=manifest.parent,
            raw=entry.get("source_cache_path"),
            label=f"{role} source cache",
        )
        content_lock = str(entry.get("content_lock_sha256", "")).lower()
        if not SHA256_RE.fullmatch(content_lock):
            raise RuntimeError(f"{role} has no materialized dependency content lock")
        dependency_count = entry.get("dependency_count")
        if not isinstance(dependency_count, int) or dependency_count < 0:
            raise ValueError(f"{role} dependency_count is invalid")
        materialized_files = entry.get("materialized_files")
        if not isinstance(materialized_files, list) or not materialized_files:
            raise RuntimeError(
                f"{role} has no explicit materialized dependency inventory"
            )
        verified_files: list[dict[str, Any]] = []
        seen_dependency_paths: set[str] = set()
        for file_index, file_entry in enumerate(materialized_files):
            if not isinstance(file_entry, dict):
                raise ValueError(
                    f"{role} materialized_files[{file_index}] must be an object"
                )
            dependency = _relative_file(
                root=volume,
                base=volume,
                raw=file_entry.get("path"),
                label=f"{role} materialized_files[{file_index}]",
            )
            relative_dependency = dependency.relative_to(volume).as_posix()
            dependency_key = relative_dependency.casefold()
            if dependency_key in seen_dependency_paths:
                raise ValueError(f"{role} dependency inventory contains duplicates")
            seen_dependency_paths.add(dependency_key)
            expected_dependency_sha = str(
                file_entry.get("sha256", "")
            ).strip().lower()
            expected_dependency_size = file_entry.get("size_bytes")
            actual_dependency_sha = _sha256(dependency)
            actual_dependency_size = dependency.stat().st_size
            if (
                not SHA256_RE.fullmatch(expected_dependency_sha)
                or expected_dependency_sha != actual_dependency_sha
            ):
                raise RuntimeError(
                    f"{role} dependency SHA-256 mismatch: {relative_dependency}"
                )
            if (
                not isinstance(expected_dependency_size, int)
                or expected_dependency_size < 0
                or expected_dependency_size != actual_dependency_size
            ):
                raise RuntimeError(
                    f"{role} dependency size mismatch: {relative_dependency}"
                )
            verified_files.append(
                {
                    "path": relative_dependency,
                    "sha256": actual_dependency_sha,
                    "size_bytes": actual_dependency_size,
                }
            )
        recalculated_content_lock = hashlib.sha256(
            json.dumps(verified_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if recalculated_content_lock != content_lock:
            raise RuntimeError(f"{role} materialized dependency content lock drifted")
        if dependency_count not in {
            len(verified_files),
            max(0, len(verified_files) - 1),
        }:
            raise RuntimeError(f"{role} dependency_count disagrees with its inventory")
        source_relative_to_volume = source.relative_to(volume).as_posix()
        if source_relative_to_volume.casefold() not in seen_dependency_paths:
            raise RuntimeError(f"{role} source cache is absent from its dependency lock")
        source_lock = next(
            item
            for item in verified_files
            if item["path"].casefold() == source_relative_to_volume.casefold()
        )
        if source_lock["sha256"] != entry["provider_hash"]:
            raise RuntimeError(f"{role} provenance hash disagrees with source content")
        expected_reference = (
            os.path.relpath(source, wrapper.parent)
            .replace("\\", "/")
            .replace("@", "%40")
        )
        if f"@{expected_reference}@" not in wrapper_source:
            raise RuntimeError(
                f"{role} wrapper does not reference its locked local source"
            )
        source_uri = str(entry.get("source_uri", "")).strip()
        licence = str(entry.get("license_id", "")).strip()
        if actual in seen_wrapper_hashes:
            raise ValueError("environment USD wrappers must be unique")
        source_key = str(source).casefold()
        if source_key in seen_source_paths:
            raise ValueError("environment source assets must be unique")
        seen_wrapper_hashes.add(actual)
        seen_source_paths.add(source_key)
        locked.append(
            {
                "role": role,
                **visual_contract,
                "wrapper": wrapper.relative_to(volume).as_posix(),
                "wrapper_sha256": actual,
                "source_cache": source.relative_to(volume).as_posix(),
                "content_lock_sha256": content_lock,
                "dependency_count": dependency_count,
                "materialized_files": verified_files,
                "license_id": licence,
                "source_uri": source_uri,
            }
        )

    asset_content_sha256 = _canonical_sha256(
        locked
    )
    family_counts = {
        kind: {
            family: sum(
                1
                for item in locked
                if item["family"] == f"{kind}.{family}"
            )
            for family in family_minimums
        }
        for kind, family_minimums in PHOTOREAL_FAMILY_MINIMUMS.items()
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "validated_at": _utc_now(),
        "state": "ASSETS_LOCKED",
        "profile": payload.get("profile"),
        "manifest": manifest.relative_to(volume).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "materialization_mode": MATERIALIZED_ASSET_MODE,
        "asset_content_sha256": asset_content_sha256,
        "family_counts": family_counts,
        "vegetation_assets": sum(family_counts["vegetation"].values()),
        "building_assets": sum(family_counts["buildings"].values()),
        "asset_count": len(locked),
        "library_policy": dict(PHOTOREAL_LIBRARY_POLICY),
        "assets": locked,
    }
    return receipt


def validate_native_asset_quality(
    *,
    materialized_receipt: dict[str, Any],
    volume_root: Path,
) -> dict[str, Any]:
    """Load every environment wrapper in Isaac and apply photoreal thresholds."""

    from fireviewer_sdg.ign_catalog import (
        _assert_asset_quality,
        _validate_usd_assets,
    )

    volume = volume_root.resolve()
    entries = materialized_receipt.get("assets")
    minimum_assets = sum(
        sum(families.values())
        for families in PHOTOREAL_FAMILY_MINIMUMS.values()
    )
    if not isinstance(entries, list) or len(entries) < minimum_assets:
        raise ValueError("materialized asset receipt is incomplete")
    paths = [(volume / str(entry["wrapper"])).resolve() for entry in entries]
    if any(not _inside(volume, path) or not path.is_file() for path in paths):
        raise RuntimeError("materialized asset wrapper escaped before native validation")
    report = _validate_usd_assets(paths)
    quality: list[dict[str, Any]] = []
    for entry, path in zip(entries, paths, strict=True):
        role = str(entry["role"])
        asset_family = str(entry["family"])
        validator_family = (
            "rural_building"
            if asset_family.startswith("buildings.")
            else "vegetation"
        )
        metrics = _assert_asset_quality(
            report,
            entry={"path": path, "role": role},
            family=validator_family,
        )
        quality.append(
            {
                "role": role,
                "asset_id": entry["asset_id"],
                "family": asset_family,
                **metrics,
            }
        )
    return {
        "validator": "fireviewer_native_usd_photoreal_quality_v2",
        "validated_assets": len(quality),
        "family_counts": materialized_receipt["family_counts"],
        "assets": quality,
    }


def _zone_source_signature(catalog_root: Path, zone_id: str) -> tuple[str, list[int]]:
    rows = _zone_rows(catalog_root, zone_id)
    if len(rows) != 400:
        raise ValueError(f"{zone_id} must contain exactly 400 one-kilometre tiles")
    bounds = [
        min(int(row["xmin"]) for row in rows),
        min(int(row["ymin"]) for row in rows),
        max(int(row["xmax"]) for row in rows),
        max(int(row["ymax"]) for row in rows),
    ]
    signature_payload = [
        [
            row["tile_ref"],
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        ]
        for row in sorted(rows, key=lambda item: item["tile_ref"])
    ]
    return _canonical_sha256(signature_payload), bounds


def build_campaign_index(
    *,
    catalog_root: Path,
    asset_manifest: Path,
    volume_root: Path,
    output_path: Path,
    pilot_zone: str | None = None,
    base_zones: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Bind four source scenes to five compositions each."""

    catalog = catalog_root.resolve()
    if base_zones is None:
        raise ValueError(
            "campaign base zones must be supplied explicitly; "
            "automatic catalog selection is forbidden"
        )
    requested_bases = tuple(str(value).strip() for value in base_zones)
    if (
        len(requested_bases) != BASE_SCENE_COUNT
        or len(set(requested_bases)) != BASE_SCENE_COUNT
        or any(zone not in ZONE_ORDER for zone in requested_bases)
    ):
        raise ValueError("campaign requires exactly four distinct catalog base zones")
    # The composition engine also orders bases by stable ID.  Canonicalising
    # here keeps SIM-01..SIM-20 bound to the same base independently of the
    # environment-variable ordering.
    bases = tuple(sorted(requested_bases))
    selected_pilot = pilot_zone or requested_bases[0]
    if selected_pilot not in requested_bases:
        raise ValueError(f"unsupported pilot zone: {pilot_zone}")
    catalog_receipt = validate_catalog(catalog)
    asset_receipt = validate_materialized_assets(
        manifest_path=asset_manifest,
        volume_root=volume_root,
    )

    source_scenes: list[dict[str, Any]] = []
    source_signatures: set[str] = set()
    for zone_id in bases:
        manifest_path, _manifest = _zone_manifest(catalog, zone_id)
        signature, bounds = _zone_source_signature(catalog, zone_id)
        if signature in source_signatures:
            raise ValueError("catalog source zones must have distinct spatial signatures")
        source_signatures.add(signature)
        source_scenes.append(
            {
                "zone_id": zone_id,
                "manifest": manifest_path.relative_to(catalog).as_posix(),
                "manifest_sha256": _sha256(manifest_path),
                "crs": "EPSG:2154",
                "tile_count": 400,
                "bounds": bounds,
                "source_spatial_signature": signature,
                "build_state": "pending",
            }
        )

    simulations: list[dict[str, Any]] = []
    seeds: set[int] = set()
    slot_index = 0
    for source_scene in source_scenes:
        zone_id = str(source_scene["zone_id"])
        for variant_index in range(1, VARIANTS_PER_BASE + 1):
            slot_index += 1
            slot_id = f"SIM-{slot_index:02d}"
            seed = int(
                hashlib.sha256(
                    f"{CAMPAIGN_ID}:{zone_id}:{variant_index}".encode("ascii")
                ).hexdigest()[:15],
                16,
            )
            if seed in seeds:
                raise RuntimeError("simulation seed collision")
            seeds.add(seed)
            simulations.append(
                {
                    "simulation_id": slot_id,
                    "base_zone_id": zone_id,
                    "variant_index": variant_index,
                    "seed": seed,
                    "state": "blocked_pending_editor_review",
                    "scene_binding": {
                        "source_spatial_signature": source_scene[
                            "source_spatial_signature"
                        ],
                        "root_usd": (
                            f"variant-scenes/{slot_id}/build/root.usdc"
                        ),
                        "build_receipt": (
                            f"variant-scenes/{slot_id}/build/build-receipt.json"
                        ),
                        "composition_plan": (
                            f"variant-plan/{slot_id}/variant.json"
                        ),
                        "portfolio_authoring_receipt": (
                            "variant-scenes/authoring-receipt.json"
                        ),
                        "acceptance": "current_editor_acceptance_required",
                    },
                    "composition_policy": (
                        "preserve_counts_assets_ids_rearrange_forest_buildings_routes"
                    ),
                    "photoreal_contract": (
                        "usd_assets_object_free_pbr_lidar_rtx"
                    ),
                }
            )

    if (
        len(source_scenes) != BASE_SCENE_COUNT
        or len(simulations) != EXPECTED_SIMULATION_COUNT
    ):
        raise RuntimeError(
            "campaign contract must contain four bases and twenty variants"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "created_at": _utc_now(),
        "state": "SCENE_BUILD_PENDING",
        "fire_simulation_status": "blocked_pending_editor_review",
        "pilot_zone": selected_pilot,
        "catalog": {
            "receipt_sha256": _canonical_sha256(catalog_receipt),
            "zone_count": len(source_scenes),
            "tile_count": sum(scene["tile_count"] for scene in source_scenes),
        },
        "assets": {
            "manifest_sha256": asset_receipt["manifest_sha256"],
            "content_sha256": asset_receipt["asset_content_sha256"],
            "materialization_mode": asset_receipt["materialization_mode"],
            "family_counts": asset_receipt["family_counts"],
            "vegetation_assets": asset_receipt["vegetation_assets"],
            "building_assets": asset_receipt["building_assets"],
            "asset_count": asset_receipt["asset_count"],
            "library_policy": asset_receipt["library_policy"],
        },
        "source_scenes": source_scenes,
        "simulations": simulations,
        "composition": {
            "base_scene_count": BASE_SCENE_COUNT,
            "variants_per_base": VARIANTS_PER_BASE,
            "algorithm": "fireviewer-photoreal-scene-variants-v1",
            "preserve_exact_counts": True,
            "rearranged_layers": ["trees", "buildings", "routes"],
            "preserved_layers": ["terrain", "water"],
            "ground_surface": (
                "object_free_pbr_or_object_removed_orthomosaic"
            ),
        },
        "manual_gate": {
            "required": True,
            "editor": "FireViewer USD Composer",
            "acceptance_binds": [
                "runtime_preflight_sha256",
                "campaign_index_sha256",
                "asset_manifest_sha256",
                "root_usd_sha256",
                "build_receipt_sha256",
                "scene_auto_validation_sha256",
            ],
        },
    }
    _atomic_write_json(output_path, payload)
    return payload


def _run_text(command: list[str], *, timeout: float = 60.0) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(item) for item in parts[:4])


def _container_memory_limit(
    *,
    cgroup_limit_paths: Iterable[Path] = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ),
) -> dict[str, Any]:
    """Return the finite container cgroup limit without using host MemTotal."""

    finite_limits: list[tuple[Path, int]] = []
    inspected_paths: list[str] = []
    for limit_path in cgroup_limit_paths:
        inspected_paths.append(str(limit_path))
        try:
            raw = limit_path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"cgroup memory limit is malformed: {limit_path}"
            ) from exc
        if limit <= 0:
            raise RuntimeError(f"cgroup memory limit is invalid: {limit_path}")
        finite_limits.append((limit_path, limit))
    if not finite_limits:
        raise RuntimeError(
            "container has no finite cgroup memory limit; "
            f"inspected {', '.join(inspected_paths)}"
        )
    source_path, limit_bytes = min(finite_limits, key=lambda item: item[1])
    return {
        "limit_bytes": limit_bytes,
        "effective_mib": limit_bytes // (1024 * 1024),
        "source": str(source_path),
        "measurement": "finite_container_cgroup_limit",
        "host_proc_meminfo_used": False,
    }


def _effective_system_ram_mib(
    *,
    cgroup_limit_paths: Iterable[Path] = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ),
) -> int:
    """Compatibility helper returning the authoritative cgroup limit in MiB."""

    return int(
        _container_memory_limit(cgroup_limit_paths=cgroup_limit_paths)[
            "effective_mib"
        ]
    )


def _extension_versions(release_root: Path, prefix: str) -> list[str]:
    results: list[str] = []
    for folder in (release_root / "extscache", release_root / "exts"):
        if not folder.is_dir():
            continue
        results.extend(path.name for path in folder.glob(f"{prefix}-*") if path.is_dir())
        results.extend(path.name for path in folder.glob(prefix) if path.is_dir())
    return sorted(set(results))


def _filesystem_identity(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(
            _run_text(
                [
                    "findmnt",
                    "--json",
                    "--target",
                    str(path),
                    "--output",
                    "SOURCE,FSTYPE,TARGET",
                ]
            )
        )
    except (json.JSONDecodeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"workspace filesystem identity is unavailable: {path}"
        ) from exc
    rows = payload.get("filesystems") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError(
            f"workspace filesystem identity is ambiguous: {path}"
        )
    return {
        "source": str(rows[0].get("source", "")),
        "type": str(rows[0].get("fstype", "")).casefold(),
        "target": str(rows[0].get("target", "")),
    }


def write_runtime_preflight(
    *,
    workspace_mount: Path,
    volume_root: Path,
    storage_mode: str,
    editor_launcher: Path,
    app_kit: Path,
    kit_checkout: Path,
    template_playback: Path,
    editor_build_stamp: Path,
    output_path: Path,
    minimum_free_gib: float,
    minimum_storage_gb: float,
    minimum_vram_mib: int,
    minimum_system_ram_mib: int,
    required_gpu_name: str,
) -> dict[str, Any]:
    """Record actual pod, GPU, Vulkan and built-Editor evidence."""

    if os.name != "posix":
        raise RuntimeError("RunPod runtime preflight is Linux-only")
    mount = workspace_mount.resolve()
    volume = volume_root.resolve()
    selected_storage_mode = storage_mode.strip()
    if selected_storage_mode not in {"persistent-volume", "ephemeral-nvme"}:
        raise ValueError(
            "storage_mode must be persistent-volume or ephemeral-nvme"
        )
    workspace_is_mount = mount.is_mount()
    if selected_storage_mode == "persistent-volume" and not workspace_is_mount:
        raise RuntimeError(f"persistent workspace is not a mount point: {mount}")
    filesystem = _filesystem_identity(mount)
    if (
        selected_storage_mode == "ephemeral-nvme"
        and filesystem["type"] in NETWORK_FILESYSTEM_TYPES
    ):
        raise RuntimeError(
            "ephemeral-nvme mode refuses network-backed workspace storage: "
            f"{filesystem['type']}"
        )
    if volume_root.is_symlink() or not _inside(mount, volume):
        raise RuntimeError("FireViewer volume must be a non-symlink path inside /workspace")
    volume.mkdir(parents=True, exist_ok=True)
    probe = volume / ".fireviewer-write-probe"
    probe.write_text("ok\n", encoding="ascii")
    probe.unlink()
    disk_usage = shutil.disk_usage(volume)
    capacity_bytes = disk_usage.total
    free_gib = disk_usage.free / (1024**3)
    minimum_storage_bytes = int(minimum_storage_gb * 1_000_000_000)
    if (
        selected_storage_mode == "ephemeral-nvme"
        and capacity_bytes < minimum_storage_bytes
    ):
        raise RuntimeError(
            f"ephemeral NVMe exposes {capacity_bytes} bytes; "
            f"{minimum_storage_bytes} bytes are required"
        )
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"workspace storage has {free_gib:.2f} GiB free; "
            f"{minimum_free_gib:.2f} GiB are required before measured acquisition"
        )
    memory_limit = _container_memory_limit()
    system_ram_mib = int(memory_limit["effective_mib"])
    if system_ram_mib < minimum_system_ram_mib:
        raise RuntimeError(
            f"container exposes {system_ram_mib} MiB system RAM; "
            f"{minimum_system_ram_mib} MiB are required"
        )

    checkout = kit_checkout.resolve()
    playback = template_playback.resolve()
    if not playback.is_file():
        raise RuntimeError("versioned Kit template playback is absent")
    build_stamp_path = editor_build_stamp.resolve()
    if not _inside(volume, build_stamp_path):
        raise RuntimeError("Kit Editor build stamp must stay inside the workspace volume")
    build_stamp = _read_json(build_stamp_path, label="Kit Editor build stamp")
    commit = _run_text(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if commit != KIT_TEMPLATE_COMMIT:
        raise RuntimeError(f"Kit App Template commit drifted: {commit}")
    origin = _run_text(["git", "-C", str(checkout), "remote", "get-url", "origin"])
    if origin.rstrip("/") not in {
        KIT_TEMPLATE_ORIGIN.rstrip("/"),
        KIT_TEMPLATE_ORIGIN.removesuffix(".git").rstrip("/"),
    }:
        raise RuntimeError(f"unexpected Kit App Template origin: {origin}")

    launcher = editor_launcher.resolve()
    application = app_kit.resolve()
    if not launcher.is_file() or not application.is_file():
        raise RuntimeError("built FireViewer USD Composer launcher or application is absent")
    release_root = launcher.parent
    kit_binary = release_root / "kit" / "kit"
    if not kit_binary.is_file():
        raise RuntimeError("built Kit executable is absent")
    build_version = release_root / "VERSION"
    if not build_version.is_file():
        raise RuntimeError("built Editor VERSION file is absent")
    source_application = checkout / "source" / "apps" / "fireviewer_usd_composer.kit"
    if (
        build_stamp.get("state") != "KIT_EDITOR_BUILT"
        or build_stamp.get("template_commit") != commit
        or build_stamp.get("playback_sha256") != _sha256(playback)
        or build_stamp.get("source_application_sha256")
        != _sha256(source_application)
        or build_stamp.get("built_application_sha256") != _sha256(application)
        or build_stamp.get("launcher_sha256") != _sha256(launcher)
    ):
        raise RuntimeError("Kit Editor build stamp is stale for the built application")

    gpu_rows = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    if not gpu_rows:
        raise RuntimeError("nvidia-smi returned no GPU")
    gpu_name, driver_version, memory_text = [
        item.strip() for item in gpu_rows[0].rsplit(",", 2)
    ]
    required_gpu = required_gpu_name.strip()
    if (
        required_gpu
        and _normalized_gpu_name(required_gpu)
        != _normalized_gpu_name(gpu_name)
    ):
        raise RuntimeError(
            f"pod GPU is {gpu_name}; required exact GPU is {required_gpu!r}"
        )
    if any(marker in gpu_name.upper() for marker in COMPUTE_ONLY_GPU_MARKERS):
        raise RuntimeError(f"compute-only GPU is not supported by the Editor: {gpu_name}")
    if not any(marker in gpu_name.upper() for marker in RENDER_GPU_MARKERS):
        raise RuntimeError(f"GPU is not in the validated RTX rendering families: {gpu_name}")
    if _version_tuple(driver_version) < MIN_DRIVER_VERSION:
        raise RuntimeError(f"NVIDIA driver is too old for the pinned Kit build: {driver_version}")
    if (
        "RTX PRO 6000" in gpu_name.upper()
        and _version_tuple(driver_version) < BLACKWELL_MIN_DRIVER_VERSION
    ):
        raise RuntimeError(
            "RTX PRO 6000 Blackwell requires a validated Linux R570 data-center "
            f"driver or newer; found {driver_version}"
        )
    memory_mib = int(float(memory_text))
    if memory_mib < minimum_vram_mib:
        raise RuntimeError(
            f"GPU exposes {memory_mib} MiB VRAM; {minimum_vram_mib} MiB are required"
        )
    vulkan_summary = _run_text(["vulkaninfo", "--summary"], timeout=120)
    if gpu_name.split()[0].lower() not in vulkan_summary.lower() and "nvidia" not in vulkan_summary.lower():
        raise RuntimeError("Vulkan summary does not expose an NVIDIA device")

    scene_optimizer = _extension_versions(release_root, "omni.scene.optimizer.bundle")
    asset_validator = _extension_versions(release_root, "omni.asset_validator.core")
    if not scene_optimizer or not asset_validator:
        raise RuntimeError("built Editor is missing Scene Optimizer or Asset Validator")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": _utc_now(),
        "state": "SETUP_PREFLIGHT_PASSED",
        "workspace": {
            "mount": str(mount),
            "volume": str(volume),
            "storage_mode": selected_storage_mode,
            "workspace_is_mount_point": workspace_is_mount,
            "filesystem": filesystem,
            "capacity_bytes": capacity_bytes,
            "minimum_capacity_bytes": minimum_storage_bytes,
            "free_gib": round(free_gib, 2),
            "minimum_free_gib": minimum_free_gib,
        },
        "storage": {
            "mode": selected_storage_mode,
            "capacity_bytes": capacity_bytes,
            "minimum_capacity_gb_decimal": minimum_storage_gb,
            "free_bytes": disk_usage.free,
            "automatic_stop_allowed": False,
            "durability": (
                "ephemeral_until_explicit_pod_termination"
                if selected_storage_mode == "ephemeral-nvme"
                else "persistent_volume"
            ),
        },
        "gpu": {
            "name": gpu_name,
            "driver_version": driver_version,
            "memory_mib": memory_mib,
            "minimum_memory_mib": minimum_vram_mib,
            "required_name_exact": required_gpu,
            "vulkan_summary_sha256": hashlib.sha256(
                vulkan_summary.encode("utf-8")
            ).hexdigest(),
        },
        "system_memory": {
            "effective_mib": system_ram_mib,
            "limit_bytes": memory_limit["limit_bytes"],
            "minimum_effective_mib": minimum_system_ram_mib,
            "measurement": memory_limit["measurement"],
            "source": memory_limit["source"],
            "host_proc_meminfo_used": memory_limit["host_proc_meminfo_used"],
        },
        "editor": {
            "template_origin": KIT_TEMPLATE_ORIGIN,
            "template_commit": commit,
            "template_playback": str(playback),
            "template_playback_sha256": _sha256(playback),
            "build_stamp": str(build_stamp_path),
            "build_stamp_sha256": _sha256(build_stamp_path),
            "overlay_sha256": build_stamp.get("overlay_sha256"),
            "source_application_sha256": _sha256(source_application),
            "application": str(application),
            "application_sha256": _sha256(application),
            "launcher": str(launcher),
            "launcher_sha256": _sha256(launcher),
            "kit_binary_sha256": _sha256(kit_binary),
            "build_version": build_version.read_text(encoding="utf-8").strip(),
            "scene_optimizer": scene_optimizer,
            "asset_validator": asset_validator,
        },
        "proof_boundary": (
            "hardware and built runtime only; no scene has been human-reviewed"
        ),
    }
    _atomic_write_json(output_path, payload)
    return payload


def create_review_pending(
    *,
    zone_id: str | None = None,
    scene_id: str | None = None,
    root_usd: Path,
    runtime_preflight: Path,
    campaign_index: Path,
    asset_manifest: Path,
    volume_root: Path,
    build_receipt: Path,
    scene_auto_validation: Path,
    internal_qa_receipt: Path | None = None,
    output_path: Path,
) -> dict[str, Any]:
    """Create a pending gate receipt without manufacturing a review decision."""

    selected_scene = (scene_id or zone_id or "").strip()
    if zone_id and scene_id and zone_id != scene_id:
        raise ValueError("zone_id and scene_id identify different review scenes")
    if selected_scene not in ZONE_ORDER and not re.fullmatch(
        r"SIM-(?:0[1-9]|1[0-9]|20)", selected_scene
    ):
        raise ValueError(f"unsupported review scene: {selected_scene}")
    root = root_usd.resolve()
    runtime = runtime_preflight.resolve()
    campaign = campaign_index.resolve()
    assets = asset_manifest.resolve()
    asset_receipt = validate_materialized_assets(
        manifest_path=assets,
        volume_root=volume_root,
    )
    build = build_receipt.resolve()
    scene_validation = scene_auto_validation.resolve()
    for path, label in (
        (root, "root USD"),
        (runtime, "runtime preflight"),
        (campaign, "campaign index"),
        (assets, "asset manifest"),
        (build, "build receipt"),
        (scene_validation, "scene auto-validation receipt"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is absent: {path}")
    variant_binding_inventory: dict[str, Any] | None = None
    internal_qa_path: Path | None = None
    runtime_payload = _read_json(runtime, label="runtime preflight")
    if runtime_payload.get("state") != "SETUP_PREFLIGHT_PASSED":
        raise RuntimeError("runtime preflight is not passed")
    runtime_gpu = runtime_payload.get("gpu")
    runtime_memory = runtime_payload.get("system_memory")
    runtime_storage = runtime_payload.get("storage")
    if (
        not isinstance(runtime_gpu, dict)
        or _normalized_gpu_name(runtime_gpu.get("name"))
        != _normalized_gpu_name(PRODUCTION_GPU_NAME)
        or int(runtime_gpu.get("memory_mib", 0)) < PRODUCTION_MIN_VRAM_MIB
    ):
        raise RuntimeError(
            "runtime preflight is not bound to the RTX PRO 6000 Blackwell "
            "Server Edition 96 GB profile"
        )
    if (
        not isinstance(runtime_memory, dict)
        or int(runtime_memory.get("effective_mib", 0))
        < PRODUCTION_MIN_SYSTEM_RAM_MIB
        or runtime_memory.get("measurement")
        != "finite_container_cgroup_limit"
        or runtime_memory.get("host_proc_meminfo_used") is not False
        or not str(runtime_memory.get("source", "")).startswith(
            "/sys/fs/cgroup/"
        )
    ):
        raise RuntimeError(
            "runtime preflight is not bound to a finite container cgroup RAM limit"
        )
    if (
        not isinstance(runtime_storage, dict)
        or runtime_storage.get("mode") != "ephemeral-nvme"
        or int(runtime_storage.get("capacity_bytes", 0))
        < PRODUCTION_MIN_STORAGE_BYTES
        or runtime_storage.get("automatic_stop_allowed") is not False
    ):
        raise RuntimeError(
            "runtime preflight is not bound to the 1500 GB ephemeral NVMe contract"
        )
    campaign_payload = _read_json(campaign, label="campaign index")
    if campaign_payload.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("campaign index has the wrong campaign_id")
    if campaign_payload.get("fire_simulation_status") != "blocked_pending_editor_review":
        raise RuntimeError("campaign must still be blocked before editor review")
    selected_simulation: dict[str, Any] | None = None
    if selected_scene not in ZONE_ORDER:
        if internal_qa_receipt is None:
            raise RuntimeError(
                "SIM-01 review pending requires the current internal QA receipt"
            )
        internal_qa_path = internal_qa_receipt.resolve()
        if (
            not internal_qa_path.is_file()
            or not _inside(volume_root.resolve(), internal_qa_path)
        ):
            raise RuntimeError(
                "SIM-01 internal QA receipt is absent from the production volume"
            )
        simulations = campaign_payload.get("simulations")
        if not isinstance(simulations, list):
            raise RuntimeError("campaign index has no simulation slots")
        matching_variants = [
            item
            for item in simulations
            if isinstance(item, dict)
            and item.get("simulation_id") == selected_scene
        ]
        if (
            len(matching_variants) != 1
            or matching_variants[0].get("state")
            != "blocked_pending_editor_review"
        ):
            raise RuntimeError(
                "review scene is not one blocked variant in the campaign index"
            )
        selected_simulation = matching_variants[0]
        scene_binding = selected_simulation.get("scene_binding")
        if not isinstance(scene_binding, dict):
            raise RuntimeError("campaign simulation slot has no scene binding")
        bound_root = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("root_usd"),
            label="campaign variant root USD",
        )
        bound_build = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("build_receipt"),
            label="campaign variant build receipt",
        )
        plan_metadata = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("composition_plan"),
            label="campaign variant composition plan",
        )
        authoring_receipt = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("portfolio_authoring_receipt"),
            label="portfolio authoring receipt",
        )
        if bound_root != root or bound_build != build:
            raise RuntimeError(
                "campaign simulation slot is bound to another scene artifact"
            )
        plan_payload = _read_json(
            plan_metadata, label="variant composition plan metadata"
        )
        if (
            plan_payload.get("simulation_id") != selected_scene
            or plan_payload.get("base_scene_id")
            != selected_simulation.get("base_zone_id")
            or plan_payload.get("variant_index")
            != selected_simulation.get("variant_index")
            or plan_payload.get("fire_simulation_status")
            != "blocked_pending_editor_review"
        ):
            raise RuntimeError(
                "variant composition plan differs from its campaign slot"
            )
        authoring_payload = _read_json(
            authoring_receipt, label="portfolio authoring receipt"
        )
        authored_variants = authoring_payload.get("variants")
        authored_match = [
            item
            for item in authored_variants or []
            if isinstance(item, dict)
            and item.get("simulation_id") == selected_scene
        ]
        if (
            authoring_payload.get("state") != "VARIANT_USD_AUTHORED"
            or authoring_payload.get("simulation_count")
            != EXPECTED_SIMULATION_COUNT
            or not isinstance(authored_variants, list)
            or len(authored_variants) != EXPECTED_SIMULATION_COUNT
            or len(authored_match) != 1
            or authored_match[0].get("base_scene_id")
            != selected_simulation.get("base_zone_id")
            or authored_match[0].get("variant_index")
            != selected_simulation.get("variant_index")
            or authored_match[0].get("fire_simulation_status")
            != "blocked_pending_editor_review"
            or authoring_payload.get("fire_simulation_status")
            != "blocked_pending_editor_review"
        ):
            raise RuntimeError(
                "portfolio authoring receipt differs from its campaign slot"
            )
        authored_artifacts = authored_match[0].get("artifacts")
        authored_build = (
            authored_artifacts.get("composer_build_receipt")
            if isinstance(authored_artifacts, dict)
            else None
        )
        verified_authored_build = _verify_artifact_entry(
            authored_build,
            base=build.parent.parent,
            allowed_root=volume_root,
            label="authored Composer build receipt",
        )
        if (
            Path(volume_root, verified_authored_build["path"]).resolve()
            != build
            or verified_authored_build["sha256"] != _sha256(build)
        ):
            raise RuntimeError(
                "portfolio authoring receipt is stale for the review build"
            )
        variant_binding_inventory = {
            "composition_plan_sha256": _sha256(plan_metadata),
            "portfolio_authoring_receipt_sha256": _sha256(authoring_receipt),
        }
    asset_sha = _sha256(assets)
    if campaign_payload.get("assets", {}).get("manifest_sha256") != asset_sha:
        raise RuntimeError("campaign index is not bound to the current asset manifest")
    asset_content_sha = asset_receipt["asset_content_sha256"]
    if campaign_payload.get("assets", {}).get("content_sha256") != asset_content_sha:
        raise RuntimeError(
            "campaign index is not bound to the current materialized asset content"
        )
    build_payload = _read_json(build, label="build receipt")
    build_root = build_payload.get("root_usd")
    if (
        not isinstance(build_root, dict)
        or build_root.get("sha256") != _sha256(root)
        or build_payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
    ):
        raise RuntimeError("build receipt is not bound to the blocked review root")
    if selected_simulation is not None and (
        build_payload.get("scene_kind") != "fictive_variant"
        or build_payload.get("zone_id") != selected_scene
        or build_payload.get("base_scene_id")
        != selected_simulation.get("base_zone_id")
        or build_payload.get("variant_index")
        != selected_simulation.get("variant_index")
    ):
        raise RuntimeError(
            "variant build identity differs from its campaign simulation slot"
        )
    build_inventory = _verify_build_artifacts(
        build_payload=build_payload,
        build_receipt=build,
        root_usd=root,
        allowed_root=volume_root,
    )
    scene_payload = _read_json(
        scene_validation, label="scene auto-validation receipt"
    )
    if (
        scene_payload.get("state") != "AUTO_VALIDATED"
        or scene_payload.get("fire_simulation_status")
        != "blocked_pending_editor_review"
        or scene_payload.get("root_usd_sha256") != _sha256(root)
        or scene_payload.get("build_receipt_sha256") != _sha256(build)
        or scene_payload.get("asset_manifest_sha256") != asset_sha
    ):
        raise RuntimeError(
            "scene auto-validation is stale for the current build or asset manifest"
        )
    scene_inventory = _verify_scene_validation_layers(
        scene_payload=scene_payload,
        volume_root=volume_root,
    )
    internal_qa_sha: str | None = None
    if internal_qa_path is not None:
        internal_qa_payload = _read_json(
            internal_qa_path,
            label="SIM-01 internal QA receipt",
        )
        internal_qa_bindings = internal_qa_payload.get("bindings")
        internal_qa_inputs = (
            internal_qa_bindings.get("inputs")
            if isinstance(internal_qa_bindings, dict)
            else None
        )
        internal_scene_input = (
            internal_qa_inputs.get("scene_auto_validation")
            if isinstance(internal_qa_inputs, dict)
            else None
        )
        if (
            internal_qa_payload.get("state") != "SIM01_INTERNAL_QA_PASSED"
            or internal_qa_payload.get("simulation_id") != "SIM-01"
            or internal_qa_payload.get("review_handoff_ready") is not True
            or internal_qa_payload.get("fire_simulation_status")
            != "blocked_pending_editor_review"
            or not isinstance(internal_qa_bindings, dict)
            or internal_qa_bindings.get("root_usd_sha256") != _sha256(root)
            or internal_qa_bindings.get("build_receipt_sha256") != _sha256(build)
            or internal_qa_bindings.get("asset_manifest_sha256") != asset_sha
            or not isinstance(internal_scene_input, dict)
            or internal_scene_input.get("sha256") != _sha256(scene_validation)
        ):
            raise RuntimeError(
                "SIM-01 internal QA receipt is stale or bound to another scene"
            )
        internal_qa_sha = _sha256(internal_qa_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "scene_id": selected_scene,
        "zone_id": selected_scene if selected_scene in ZONE_ORDER else None,
        "created_at": _utc_now(),
        "status": "AWAITING_EDITOR_REVIEW",
        "human_review": "pending",
        "fire_simulation_status": "blocked_pending_editor_review",
        "bindings": {
            "runtime_preflight_sha256": _sha256(runtime),
            "campaign_index_sha256": _sha256(campaign),
            "asset_manifest_sha256": asset_sha,
            "asset_content_sha256": asset_content_sha,
            "root_usd_sha256": _sha256(root),
            "build_receipt_sha256": _sha256(build),
            "scene_auto_validation_sha256": _sha256(scene_validation),
            "build_artifact_content_sha256": build_inventory[
                "artifact_content_sha256"
            ],
            "scene_layer_content_sha256": scene_inventory[
                "layer_content_sha256"
            ],
            **(
                {"internal_qa_receipt_sha256": internal_qa_sha}
                if internal_qa_sha is not None
                else {}
            ),
            **(variant_binding_inventory or {}),
        },
        "root_usd": str(root),
        "checklist": [
            "terrain_relief_and_textures",
            "forest_density_distribution_and_no_origin_pile",
            "photoreal_asset_family_diversity_and_no_prototype_dominance",
            "native_asset_proportions_grounding_lods_and_materials",
            "no_primitive_or_procedural_asset_fallbacks",
            "inclined_camera_tile_coverage",
            "interactive_editor_stability",
        ],
    }
    if "decision" in payload:
        raise AssertionError("pending review receipt must never contain a decision")
    if output_path.exists():
        existing = _read_json(
            output_path.resolve(),
            label="existing pending review receipt",
        )
        expected_without_time = dict(payload)
        existing_without_time = dict(existing)
        expected_without_time.pop("created_at", None)
        existing_without_time.pop("created_at", None)
        if existing_without_time != expected_without_time:
            raise RuntimeError(
                "existing pending review receipt is stale for current evidence"
            )
        return existing
    _atomic_write_json(output_path, payload)
    return payload


def accept_review(
    *,
    pending_path: Path,
    opened_path: Path,
    output_path: Path,
    reviewer: str,
    acknowledgement: str,
) -> dict[str, Any]:
    """Record an explicit human acceptance after the real Editor opened."""

    reviewer_name = reviewer.strip()
    if not reviewer_name:
        raise ValueError("reviewer is required")
    if acknowledgement != REVIEW_ACKNOWLEDGEMENT:
        raise ValueError("the exact manual-review acknowledgement is required")
    pending = _read_json(pending_path, label="pending review receipt")
    opened = _read_json(opened_path, label="Editor opened receipt")
    if (
        pending.get("status") != "AWAITING_EDITOR_REVIEW"
        or pending.get("human_review") != "pending"
    ):
        raise RuntimeError("review receipt is not pending")
    if (
        opened.get("state") != "opened_for_human_review"
        or opened.get("human_review") != "pending"
    ):
        raise RuntimeError("real Editor-open evidence is absent")
    bindings = pending.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("pending review receipt has no bindings")
    if opened.get("root_usd_sha256") != bindings.get("root_usd_sha256"):
        raise RuntimeError("Editor opened a different root USD")
    if opened.get("pending_review_sha256") != _sha256(pending_path):
        raise RuntimeError("Editor-open evidence is not bound to the pending receipt")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": pending.get("campaign_id"),
        "scene_id": pending.get("scene_id", pending.get("zone_id")),
        "zone_id": pending.get("zone_id"),
        "decision": "accepted",
        "reviewer": reviewer_name,
        "reviewed_at": _utc_now(),
        "status": "EDITOR_REVIEW_ACCEPTED",
        "bindings": dict(bindings),
        "pending_review_sha256": _sha256(pending_path),
        "editor_opened_sha256": _sha256(opened_path),
        "acknowledgement": acknowledgement,
    }
    _atomic_write_json(output_path, payload)
    return payload


def assert_simulation_allowed(
    *,
    acceptance_path: Path,
    runtime_preflight: Path,
    campaign_index: Path,
    asset_manifest: Path,
    volume_root: Path,
    root_usd: Path,
    build_receipt: Path,
    scene_auto_validation: Path,
    internal_qa_receipt: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the current artifacts match a human acceptance."""

    acceptance = _read_json(acceptance_path, label="Editor acceptance receipt")
    if (
        acceptance.get("decision") != "accepted"
        or acceptance.get("status") != "EDITOR_REVIEW_ACCEPTED"
    ):
        raise RuntimeError("fire simulation is blocked pending Editor acceptance")
    asset_receipt = validate_materialized_assets(
        manifest_path=asset_manifest,
        volume_root=volume_root,
    )
    build_payload = _read_json(build_receipt, label="build receipt")
    build_inventory = _verify_build_artifacts(
        build_payload=build_payload,
        build_receipt=build_receipt,
        root_usd=root_usd,
        allowed_root=volume_root,
    )
    scene_payload = _read_json(
        scene_auto_validation, label="scene auto-validation receipt"
    )
    scene_inventory = _verify_scene_validation_layers(
        scene_payload=scene_payload,
        volume_root=volume_root,
    )
    expected = {
        "runtime_preflight_sha256": _sha256(runtime_preflight),
        "campaign_index_sha256": _sha256(campaign_index),
        "asset_manifest_sha256": _sha256(asset_manifest),
        "asset_content_sha256": asset_receipt["asset_content_sha256"],
        "root_usd_sha256": _sha256(root_usd),
        "build_receipt_sha256": _sha256(build_receipt),
        "scene_auto_validation_sha256": _sha256(scene_auto_validation),
        "build_artifact_content_sha256": build_inventory[
            "artifact_content_sha256"
        ],
        "scene_layer_content_sha256": scene_inventory["layer_content_sha256"],
    }
    accepted_scene = acceptance.get("scene_id", acceptance.get("zone_id"))
    if accepted_scene not in ZONE_ORDER:
        if internal_qa_receipt is None or not internal_qa_receipt.is_file():
            raise RuntimeError(
                "accepted SIM-01 scene requires its current internal QA receipt"
            )
        campaign_payload = _read_json(campaign_index, label="campaign index")
        simulations = campaign_payload.get("simulations")
        matching_variants = [
            item
            for item in simulations or []
            if isinstance(item, dict)
            and item.get("simulation_id") == accepted_scene
        ]
        if len(matching_variants) != 1:
            raise RuntimeError(
                "Editor acceptance is not bound to one campaign variant"
            )
        scene_binding = matching_variants[0].get("scene_binding")
        if not isinstance(scene_binding, dict):
            raise RuntimeError(
                "accepted campaign variant has no scene binding"
            )
        bound_root = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("root_usd"),
            label="accepted campaign variant root USD",
        )
        bound_build = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("build_receipt"),
            label="accepted campaign variant build receipt",
        )
        composition_plan = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("composition_plan"),
            label="accepted campaign variant composition plan",
        )
        authoring_receipt = _relative_file(
            root=volume_root,
            base=volume_root,
            raw=scene_binding.get("portfolio_authoring_receipt"),
            label="accepted portfolio authoring receipt",
        )
        if bound_root != root_usd.resolve() or bound_build != build_receipt.resolve():
            raise RuntimeError(
                "Editor acceptance is bound to another campaign variant artifact"
            )
        expected.update(
            {
                "composition_plan_sha256": _sha256(composition_plan),
                "portfolio_authoring_receipt_sha256": _sha256(
                    authoring_receipt
                ),
                "internal_qa_receipt_sha256": _sha256(
                    internal_qa_receipt
                ),
            }
        )
    if acceptance.get("bindings") != expected:
        raise RuntimeError("Editor acceptance is stale for the current runtime or scene")
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "checked_at": _utc_now(),
        "state": "FIRE_SIMULATION_ALLOWED",
        "acceptance_sha256": _sha256(acceptance_path),
        "bindings": expected,
    }


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FireViewer RunPod Omniverse setup contract and manual gate"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    assets = commands.add_parser("validate-assets")
    assets.add_argument("--manifest", required=True, type=Path)
    assets.add_argument("--volume-root", required=True, type=Path)
    assets.add_argument("--receipt", required=True, type=Path)
    assets.add_argument("--native-usd-quality", action="store_true")

    campaign = commands.add_parser("campaign-index")
    campaign.add_argument("--catalog-root", required=True, type=Path)
    campaign.add_argument("--asset-manifest", required=True, type=Path)
    campaign.add_argument("--volume-root", required=True, type=Path)
    campaign.add_argument("--output", required=True, type=Path)
    campaign.add_argument("--pilot-zone", choices=ZONE_ORDER)
    campaign.add_argument(
        "--base-zone",
        action="append",
        dest="base_zones",
        choices=ZONE_ORDER,
        help="repeat exactly four times; automatic base selection is forbidden",
    )

    preflight = commands.add_parser("runtime-preflight")
    preflight.add_argument("--workspace-mount", default="/workspace", type=Path)
    preflight.add_argument("--volume-root", required=True, type=Path)
    preflight.add_argument(
        "--storage-mode",
        choices=("persistent-volume", "ephemeral-nvme"),
        default="ephemeral-nvme",
    )
    preflight.add_argument("--editor-launcher", required=True, type=Path)
    preflight.add_argument("--app-kit", required=True, type=Path)
    preflight.add_argument("--kit-checkout", required=True, type=Path)
    preflight.add_argument("--template-playback", required=True, type=Path)
    preflight.add_argument("--editor-build-stamp", required=True, type=Path)
    preflight.add_argument("--output", required=True, type=Path)
    preflight.add_argument("--minimum-free-gib", type=float, default=300.0)
    preflight.add_argument("--minimum-storage-gb", type=float, default=1500.0)
    preflight.add_argument("--minimum-vram-mib", type=int, default=24_000)
    preflight.add_argument(
        "--minimum-system-ram-mib",
        type=int,
        default=138_000,
    )
    preflight.add_argument("--required-gpu-name", default="")

    pending = commands.add_parser("review-pending")
    pending.add_argument(
        "--scene",
        "--zone",
        dest="scene_id",
        required=True,
        help="one catalog base ID or one authored SIM-01..SIM-20 scene",
    )
    pending.add_argument("--root-usd", required=True, type=Path)
    pending.add_argument("--runtime-preflight", required=True, type=Path)
    pending.add_argument("--campaign-index", required=True, type=Path)
    pending.add_argument("--asset-manifest", required=True, type=Path)
    pending.add_argument("--volume-root", required=True, type=Path)
    pending.add_argument("--build-receipt", required=True, type=Path)
    pending.add_argument("--scene-auto-validation", required=True, type=Path)
    pending.add_argument("--internal-qa-receipt", type=Path)
    pending.add_argument("--output", required=True, type=Path)

    accept = commands.add_parser("accept-review")
    accept.add_argument("--pending", required=True, type=Path)
    accept.add_argument("--opened", required=True, type=Path)
    accept.add_argument("--output", required=True, type=Path)
    accept.add_argument("--reviewer", required=True)
    accept.add_argument("--acknowledge", required=True)

    gate = commands.add_parser("simulation-gate")
    gate.add_argument("--acceptance", required=True, type=Path)
    gate.add_argument("--runtime-preflight", required=True, type=Path)
    gate.add_argument("--campaign-index", required=True, type=Path)
    gate.add_argument("--asset-manifest", required=True, type=Path)
    gate.add_argument("--volume-root", required=True, type=Path)
    gate.add_argument("--root-usd", required=True, type=Path)
    gate.add_argument("--build-receipt", required=True, type=Path)
    gate.add_argument("--scene-auto-validation", required=True, type=Path)
    gate.add_argument("--internal-qa-receipt", type=Path)
    gate.add_argument("--receipt", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "validate-assets":
        result = validate_materialized_assets(
            manifest_path=args.manifest,
            volume_root=args.volume_root,
        )
        if args.native_usd_quality:
            result["usd_quality"] = validate_native_asset_quality(
                materialized_receipt=result,
                volume_root=args.volume_root,
            )
        _atomic_write_json(args.receipt, result)
    elif args.command == "campaign-index":
        result = build_campaign_index(
            catalog_root=args.catalog_root,
            asset_manifest=args.asset_manifest,
            volume_root=args.volume_root,
            output_path=args.output,
            pilot_zone=args.pilot_zone,
            base_zones=args.base_zones,
        )
    elif args.command == "runtime-preflight":
        result = write_runtime_preflight(
            workspace_mount=args.workspace_mount,
            volume_root=args.volume_root,
            storage_mode=args.storage_mode,
            editor_launcher=args.editor_launcher,
            app_kit=args.app_kit,
            kit_checkout=args.kit_checkout,
            template_playback=args.template_playback,
            editor_build_stamp=args.editor_build_stamp,
            output_path=args.output,
            minimum_free_gib=args.minimum_free_gib,
            minimum_storage_gb=args.minimum_storage_gb,
            minimum_vram_mib=args.minimum_vram_mib,
            minimum_system_ram_mib=args.minimum_system_ram_mib,
            required_gpu_name=args.required_gpu_name,
        )
    elif args.command == "review-pending":
        result = create_review_pending(
            scene_id=args.scene_id,
            root_usd=args.root_usd,
            runtime_preflight=args.runtime_preflight,
            campaign_index=args.campaign_index,
            asset_manifest=args.asset_manifest,
            volume_root=args.volume_root,
            build_receipt=args.build_receipt,
            scene_auto_validation=args.scene_auto_validation,
            internal_qa_receipt=args.internal_qa_receipt,
            output_path=args.output,
        )
    elif args.command == "accept-review":
        result = accept_review(
            pending_path=args.pending,
            opened_path=args.opened,
            output_path=args.output,
            reviewer=args.reviewer,
            acknowledgement=args.acknowledge,
        )
    else:
        result = assert_simulation_allowed(
            acceptance_path=args.acceptance,
            runtime_preflight=args.runtime_preflight,
            campaign_index=args.campaign_index,
            asset_manifest=args.asset_manifest,
            volume_root=args.volume_root,
            root_usd=args.root_usd,
            build_receipt=args.build_receipt,
            scene_auto_validation=args.scene_auto_validation,
            internal_qa_receipt=args.internal_qa_receipt,
        )
        if args.receipt:
            _atomic_write_json(args.receipt, result)
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
