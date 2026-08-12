"""Incremental OpenUSD gate for a tiled photoreal FireViewer review scene.

This module is intentionally executed with the pinned Kit/Isaac Python.  It
opens the review root with ``LoadNone`` and then inspects one terrain or detail
payload at a time.  At no point are the 400 one-kilometre detail payloads kept
open together.

The gate is structural and geometric.  It does not claim RTX image quality or
replace the mandatory human review in USD Composer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import struct
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
REQUIRED_TERRAIN_LODS = frozenset({"LOD1", "LOD2", "LOD3"})
ALLOWED_TERRAIN_LODS = frozenset({"LOD0", "LOD1", "LOD2", "LOD3"})
DETAIL_COUNT_KEYS = frozenset(
    {"buildings", "roads", "hydrology", "vegetation"}
)
FORBIDDEN_DIRECT_PRIM_TYPES = frozenset(
    {"Capsule", "Cone", "Cube", "Cylinder", "Sphere"}
)
FORBIDDEN_FALLBACK_MARKERS = (
    "dense_forest_fill_only",
    "fallback",
    "placeholder",
    "procedural",
    "proxy_box",
    "proxy_cone",
    "proxy_cube",
)
MAXIMUM_PROTOTYPE_SHARE = 0.25
INSTANCE_FAMILY_CODES = {
    "buildings": 1,
    "trees": 2,
    "shrubs": 3,
    "understory": 4,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SIMULATION_ID_RE = re.compile(r"^SIM-(?:0[1-9]|1[0-9]|20)$")
BASE_SCENE_ID_RE = re.compile(r"^Z[0-9]{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_below(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _nonnegative_count(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if count < 0 or str(value).strip() not in {str(count), f"{count}.0"}:
        raise ValueError(f"{label} must be a non-negative integer")
    return count


def _replicator_semantic(prim: Any) -> str:
    applied = {str(value) for value in prim.GetAppliedSchemas()}
    if "SemanticsAPI:Semantics" not in applied:
        raise RuntimeError(f"{prim.GetPath()} has no applied SemanticsAPI")
    for name in (
        "semantic:Semantics:params:semanticData",
        "semantics:Semantics:semanticData",
    ):
        attribute = prim.GetAttribute(name)
        value = str(attribute.Get() if attribute else "").strip()
        if value:
            return value.lower()
    raise RuntimeError(f"{prim.GetPath()} has no Replicator semanticData")


def _semantic(prim: Any) -> str:
    custom = prim.GetCustomData()
    standard = _replicator_semantic(prim)
    values = (
        standard,
        str(custom.get("fireviewer:semantic_class", "")),
        str(custom.get("fireviewer", "")),
        str(prim.GetPath()),
    )
    return " ".join(values).lower()


def _world_positions(
    *,
    prim: Any,
    positions: Iterable[Any],
    xform_cache: Any,
    gf: Any,
) -> Iterable[Any]:
    transform = xform_cache.GetLocalToWorldTransform(prim)
    for point in positions:
        yield transform.Transform(
            gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        )


def _infer_zone_root(
    *,
    build_receipt: Path,
    root_usd: Path,
    root_record: object,
) -> Path:
    if not isinstance(root_record, dict):
        raise ValueError("build receipt root_usd record is malformed")
    raw = str(root_record.get("path", "")).strip()
    if not raw:
        raise ValueError("build receipt root_usd.path is absent")
    declared = Path(raw)
    if declared.is_absolute():
        if declared.resolve() != root_usd:
            raise RuntimeError("build receipt points at a different root USD")
        return root_usd.parent
    candidates = (build_receipt.parent, *build_receipt.parents)
    matches = [
        candidate.resolve()
        for candidate in candidates
        if (candidate / declared).resolve() == root_usd
    ]
    if not matches:
        raise RuntimeError("could not infer the zone root from the locked root USD")
    # The closest matching ancestor is the zone root.  Duplicate values can
    # occur because build_receipt.parent is also the first parent.
    return matches[0]


def _resolve_locked_artifact(
    *,
    zone_root: Path,
    record: object,
    label: str,
    suffixes: frozenset[str] | None = None,
    allowed_root: Path | None = None,
) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record is malformed")
    raw = str(record.get("path", "")).strip()
    expected_sha = str(record.get("sha256", "")).strip().lower()
    if not raw or len(expected_sha) != 64:
        raise ValueError(f"{label} path or SHA-256 is absent")
    candidate = Path(raw)
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (zone_root / candidate).resolve()
    )
    boundary = (
        allowed_root.resolve()
        if allowed_root is not None
        else zone_root.resolve()
    )
    if not _is_below(boundary, path):
        raise RuntimeError(f"{label} escapes the allowed artifact root")
    if not path.is_file():
        raise RuntimeError(f"{label} is absent: {path}")
    if suffixes is not None and path.suffix.lower() not in suffixes:
        raise RuntimeError(f"{label} has an unsupported file type")
    if _sha256(path) != expected_sha:
        raise RuntimeError(f"{label} checksum does not match")
    return path


def _variant_build_contract(
    build_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate the immutable identity/topology envelope of a final variant."""

    scene_kind = build_payload.get("scene_kind")
    if scene_kind in {None, "", "native_base"}:
        return None
    if scene_kind != "fictive_variant":
        raise ValueError(f"unsupported scene_kind: {scene_kind}")
    scene_id = str(build_payload.get("zone_id", "")).strip()
    base_scene_id = str(build_payload.get("base_scene_id", "")).strip()
    variant_index = build_payload.get("variant_index")
    if (
        not SIMULATION_ID_RE.fullmatch(scene_id)
        or not BASE_SCENE_ID_RE.fullmatch(base_scene_id)
        or not isinstance(variant_index, int)
        or isinstance(variant_index, bool)
        or not 1 <= variant_index <= 5
    ):
        raise ValueError("variant build identity is incomplete")
    identity = build_payload.get("identity_contract")
    if not isinstance(identity, dict):
        raise ValueError("variant build identity contract is absent")
    source_identity = str(
        identity.get("source_identity_sha256", "")
    ).lower()
    authored_identity = str(
        identity.get("authored_identity_sha256", "")
    ).lower()
    if (
        identity.get("numeric_ids_preserved") is not True
        or identity.get("stable_ids_preserved") is not True
        or identity.get(
            "source_namespace_may_differ_from_destination_tile"
        )
        is not True
        or not SHA256_RE.fullmatch(source_identity)
        or authored_identity != source_identity
    ):
        raise ValueError(
            "variant build does not preserve the exact source object identity"
        )
    topology = build_payload.get("route_topology")
    if not isinstance(topology, dict):
        raise ValueError("variant route topology contract is absent")
    source_membership = str(
        topology.get("source_membership_sha256", "")
    ).lower()
    result_membership = str(
        topology.get("result_membership_sha256", "")
    ).lower()
    source_components = topology.get("source_component_count")
    result_components = topology.get("result_component_count")
    if (
        topology.get("algorithm")
        != "segment-connectivity-components-v1"
        or topology.get("exact_membership_preserved") is not True
        or not isinstance(source_components, int)
        or isinstance(source_components, bool)
        or source_components < 0
        or result_components != source_components
        or not SHA256_RE.fullmatch(source_membership)
        or result_membership != source_membership
    ):
        raise ValueError(
            "variant build changed route component membership"
        )
    return {
        "scene_id": scene_id,
        "base_scene_id": base_scene_id,
        "variant_index": variant_index,
        "identity_sha256": source_identity,
        "route_membership_sha256": source_membership,
        "route_component_count": source_components,
    }


def _coverage_contract(build_payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the 400-tile receipt shape without importing OpenUSD."""

    variant_contract = _variant_build_contract(build_payload)
    if build_payload.get("schema_version") != 2:
        raise ValueError("scene build receipt schema 2 is required")
    if build_payload.get("source_profile") != "full":
        raise ValueError("photoreal scene gate requires the full source profile")
    payloads = build_payload.get("payloads")
    details_by_level = {
        "HERO": build_payload.get("detail_payloads"),
        "MID": build_payload.get("detail_mid_payloads"),
        "FAR": build_payload.get("detail_far_payloads"),
    }
    coverage = build_payload.get("tile_coverage")
    if not isinstance(payloads, list) or len(payloads) != 400:
        raise ValueError("build receipt must declare exactly 400 terrain payloads")
    for level, details in details_by_level.items():
        if not isinstance(details, list) or len(details) != 400:
            raise ValueError(
                f"build receipt must declare exactly 400 {level} detail payloads"
            )
    if not isinstance(coverage, list) or len(coverage) != 400:
        raise ValueError("build receipt must declare exactly 400 tile coverage records")

    terrain_records: dict[str, dict[str, Any]] = {}
    detail_records_by_level: dict[str, dict[str, dict[str, Any]]] = {
        "HERO": {},
        "MID": {},
        "FAR": {},
    }
    catalogs: list[
        tuple[str, list[dict[str, Any]], dict[str, dict[str, Any]]]
    ] = [("terrain", payloads, terrain_records)]
    catalogs.extend(
        (
            f"{level} detail",
            details_by_level[level],
            detail_records_by_level[level],
        )
        for level in ("HERO", "MID", "FAR")
    )
    for label, records, destination in catalogs:
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{label} payload record is malformed")
            path = str(record.get("path", "")).strip()
            if not path or path in destination:
                raise ValueError(f"{label} payload paths must be present and unique")
            destination[path] = record
    all_detail_paths = {
        path
        for records in detail_records_by_level.values()
        for path in records
    }
    if len(all_detail_paths) != 1200:
        raise ValueError("HERO/MID/FAR detail payload paths must be distinct")

    by_tile: dict[str, dict[str, Any]] = {}
    totals = Counter({key: 0 for key in DETAIL_COUNT_KEYS})
    lod0_tiles = 0
    namespaces: set[int] = set()
    for record in coverage:
        if not isinstance(record, dict):
            raise ValueError("tile coverage record is malformed")
        tile_ref = str(record.get("tile_ref", "")).strip()
        if not tile_ref or tile_ref in by_tile:
            raise ValueError("tile coverage references must be present and unique")
        terrain_path = str(record.get("terrain_payload", "")).strip()
        if terrain_path not in terrain_records:
            raise ValueError(f"{tile_ref} does not bind a locked terrain payload")
        detail_lods = record.get("detail_lods")
        if not isinstance(detail_lods, dict) or set(detail_lods) != {
            "HERO",
            "MID",
            "FAR",
        }:
            raise ValueError(f"{tile_ref} detail LOD contract is incomplete")
        normalized_detail_paths = {
            level: str(detail_lods[level]).strip()
            for level in ("HERO", "MID", "FAR")
        }
        for level, detail_path in normalized_detail_paths.items():
            if detail_path not in detail_records_by_level[level]:
                raise ValueError(
                    f"{tile_ref} does not bind a locked {level} detail payload"
                )
        if str(record.get("detail_payload", "")).strip() != (
            normalized_detail_paths["HERO"]
        ):
            raise ValueError(f"{tile_ref} legacy detail path must equal HERO")
        lods = record.get("terrain_lods")
        if not isinstance(lods, list):
            raise ValueError(f"{tile_ref} terrain LOD contract is malformed")
        lod_names = {str(value) for value in lods}
        if (
            not REQUIRED_TERRAIN_LODS.issubset(lod_names)
            or not lod_names.issubset(ALLOWED_TERRAIN_LODS)
            or len(lod_names) != len(lods)
        ):
            raise ValueError(f"{tile_ref} terrain LOD contract is incomplete")
        if "LOD0" in lod_names:
            lod0_tiles += 1
        if record.get("collision_lods") != ["NEAR", "FAR"]:
            raise ValueError(f"{tile_ref} collision LOD contract is incomplete")
        lod_counts = record.get("detail_lod_counts")
        if not isinstance(lod_counts, dict) or set(lod_counts) != {
            "HERO",
            "MID",
            "FAR",
        }:
            raise ValueError(f"{tile_ref} detail LOD counts are incomplete")
        normalized_lod_counts: dict[str, dict[str, int]] = {}
        for level in ("HERO", "MID", "FAR"):
            counts = lod_counts[level]
            if not isinstance(counts, dict) or set(counts) != DETAIL_COUNT_KEYS:
                raise ValueError(f"{tile_ref} {level} detail counts are incomplete")
            normalized_lod_counts[level] = {
                key: _nonnegative_count(
                    counts[key], label=f"{tile_ref}.{level}.{key}"
                )
                for key in DETAIL_COUNT_KEYS
            }
        declared_hero = record.get("detail_counts")
        if declared_hero != normalized_lod_counts["HERO"]:
            raise ValueError(f"{tile_ref} legacy detail counts must equal HERO")
        normalized_counts = normalized_lod_counts["HERO"]
        instance_namespace = _nonnegative_count(
            record.get("instance_namespace"),
            label=f"{tile_ref}.instance_namespace",
        )
        if (
            instance_namespace <= 0
            or instance_namespace in namespaces
            or instance_namespace >= (1 << 20)
        ):
            raise ValueError(
                f"{tile_ref} instance namespace must be positive and globally unique"
            )
        namespaces.add(instance_namespace)
        totals.update(normalized_counts)
        by_tile[tile_ref] = {
            "terrain_path": terrain_path,
            "detail_paths": normalized_detail_paths,
            "lods": lod_names,
            "counts": normalized_counts,
            "lod_counts": normalized_lod_counts,
            "instance_namespace": instance_namespace,
        }
    if lod0_tiles <= 0:
        raise ValueError("the review camera working set exposes no LOD0 terrain")

    layers = build_payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("scene build layers are absent")
    for key in DETAIL_COUNT_KEYS:
        layer = layers.get(key)
        if not isinstance(layer, dict):
            raise ValueError(f"scene layer is absent: {key}")
        declared = _nonnegative_count(
            layer.get("prim_count"), label=f"layers.{key}.prim_count"
        )
        if declared != totals[key]:
            raise ValueError(
                f"layers.{key}.prim_count={declared} differs from "
                f"tile total {totals[key]}"
            )
    collision_layer = layers.get("collisions")
    streaming_layer = layers.get("detail_streaming")
    if not isinstance(collision_layer, dict) or _nonnegative_count(
        collision_layer.get("prim_count"),
        label="layers.collisions.prim_count",
    ) != 400:
        raise ValueError("scene must declare one collision mesh per terrain tile")
    if (
        collision_layer.get("levels") != ["NEAR", "FAR"]
        or float(collision_layer.get("near_spacing_m", 0.0)) > 4.0
        or float(collision_layer.get("far_spacing_m", 0.0)) > 32.0
    ):
        raise ValueError("collision LOD contract must expose 4 m NEAR and 32 m FAR")
    if not isinstance(streaming_layer, dict) or _nonnegative_count(
        streaming_layer.get("prim_count"),
        label="layers.detail_streaming.prim_count",
    ) != 400:
        raise ValueError("scene must declare 400 camera-streamed detail payloads")
    if streaming_layer.get("levels") != ["HERO", "MID", "FAR"]:
        raise ValueError("detail streaming must expose HERO/MID/FAR")
    if not bool(streaming_layer.get("terrain_is_never_unloaded_for_detail_streaming")):
        raise ValueError("detail streaming must preserve full-zone terrain visibility")
    return {
        "scene_kind": (
            "fictive_variant"
            if variant_contract is not None
            else "native_base"
        ),
        "variant_contract": variant_contract,
        "terrain_records": terrain_records,
        "detail_records_by_level": detail_records_by_level,
        "by_tile": by_tile,
        "totals": dict(totals),
        "lod0_tiles": lod0_tiles,
    }


def _prototype_share_gate(
    usage: Counter[str],
    *,
    label: str,
) -> dict[str, Any]:
    total = sum(usage.values())
    if total <= 0:
        return {
            "instances": 0,
            "prototype_count_used": 0,
            "maximum_share": 0.0,
        }
    maximum_key, maximum_count = max(usage.items(), key=lambda item: item[1])
    maximum_share = maximum_count / float(total)
    # Samples smaller than four cannot mathematically distribute an instance
    # over four 25%-capped prototypes.  Such families are still reported, while
    # every statistically meaningful family is gated.
    if total >= 4 and maximum_share > MAXIMUM_PROTOTYPE_SHARE + 1e-9:
        raise RuntimeError(
            f"one {label} prototype dominates {maximum_share:.1%}: {maximum_key}"
        )
    return {
        "instances": total,
        "prototype_count_used": len(usage),
        "maximum_share": maximum_share,
        "most_used_prototype": maximum_key,
    }


def _iter_prim_specs(layer: Any) -> Iterable[Any]:
    def descend(spec: Any) -> Iterable[Any]:
        yield spec
        children = getattr(spec, "nameChildren", ())
        if hasattr(children, "values"):
            children = children.values()
        for child in children:
            yield from descend(child)

    for root in layer.rootPrims:
        yield from descend(root)


def _direct_layer_summary(stage: Any) -> dict[str, Any]:
    """Inspect only specs authored by this payload, not referenced asset internals."""

    semantic_counts = Counter()
    semantic_paths: list[str] = []
    instancer_paths: list[str] = []
    forbidden: list[str] = []
    root_layer = stage.GetRootLayer()
    for spec in _iter_prim_specs(root_layer):
        path = str(spec.path)
        type_name = str(getattr(spec, "typeName", ""))
        custom = dict(getattr(spec, "customData", {}) or {})
        semantic = str(custom.get("fireviewer:semantic_class", "")).lower()
        if semantic:
            semantic_counts[semantic] += 1
            semantic_paths.append(path)
        if type_name == "PointInstancer":
            instancer_paths.append(path)
        if type_name in FORBIDDEN_DIRECT_PRIM_TYPES:
            forbidden.append(f"{path} ({type_name})")
        custom_text = json.dumps(
            custom, ensure_ascii=True, sort_keys=True, default=str
        ).lower()
        if any(marker in custom_text for marker in FORBIDDEN_FALLBACK_MARKERS):
            forbidden.append(f"{path} (fallback marker)")
    if forbidden:
        raise RuntimeError(
            "payload contains forbidden primitive/procedural fallbacks: "
            + ", ".join(forbidden[:8])
        )
    return {
        "semantic_counts": semantic_counts,
        "semantic_paths": semantic_paths,
        "instancer_paths": instancer_paths,
    }


def _reference_items(prim: Any, *, root_layer: Any) -> list[tuple[Any, str]]:
    result: list[tuple[Any, str]] = []
    for spec in prim.GetPrimStack():
        if spec.layer != root_layer:
            continue
        operation = getattr(spec, "referenceList", None)
        if operation is None:
            continue
        items: list[Any] = []
        for attribute in (
            "explicitItems",
            "prependedItems",
            "appendedItems",
            "addedItems",
        ):
            items.extend(list(getattr(operation, attribute, ()) or ()))
        seen: set[tuple[str, str]] = set()
        for item in items:
            asset_path = str(getattr(item, "assetPath", "")).strip()
            prim_path = str(getattr(item, "primPath", "")).strip()
            identity = (asset_path, prim_path)
            if asset_path and identity not in seen:
                result.append((spec.layer, asset_path))
                seen.add(identity)
    return result


def _payload_items(prim: Any, *, root_layer: Any) -> list[tuple[Any, str]]:
    result: list[tuple[Any, str]] = []
    for spec in prim.GetPrimStack():
        if spec.layer != root_layer:
            continue
        operation = getattr(spec, "payloadList", None)
        if operation is None:
            continue
        items: list[Any] = []
        for attribute in (
            "explicitItems",
            "prependedItems",
            "appendedItems",
            "addedItems",
        ):
            items.extend(list(getattr(operation, attribute, ()) or ()))
        seen: set[tuple[str, str]] = set()
        for item in items:
            asset_path = str(getattr(item, "assetPath", "")).strip()
            prim_path = str(getattr(item, "primPath", "")).strip()
            identity = (asset_path, prim_path)
            if asset_path and identity not in seen:
                result.append((spec.layer, asset_path))
                seen.add(identity)
    return result


def _resolved_reference(
    *, layer: Any, asset_path: str, sdf: Any
) -> Path | None:
    if "://" in asset_path:
        return None
    try:
        resolved = str(sdf.ComputeAssetPathRelativeToLayer(layer, asset_path))
    except (AttributeError, RuntimeError):
        resolved = str(
            (Path(str(layer.realPath)).parent / asset_path).resolve()
        )
    return Path(resolved).resolve()


def _family_from_semantic(semantic: str) -> str | None:
    if "building" in semantic:
        return "buildings"
    for family in ("trees", "shrubs", "understory"):
        if f"vegetation_{family}_" in semantic:
            return family
    return None


def _inspect_instancer(
    *,
    stage: Any,
    prim_path: str,
    detail_path: Path,
    expected_tile_bounds: tuple[float, float, float, float],
    expected_instance_namespace: int,
    scene_kind: str,
    origin: tuple[float, float],
    gf: Any,
    sdf: Any,
    usd: Any,
    usd_geom: Any,
) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid() or not prim.IsA(usd_geom.PointInstancer):
        raise RuntimeError(f"{detail_path.name}: invalid PointInstancer {prim_path}")
    instancer = usd_geom.PointInstancer(prim)
    positions = list(instancer.GetPositionsAttr().Get() or [])
    proto_indices = list(instancer.GetProtoIndicesAttr().Get() or [])
    scales = list(instancer.GetScalesAttr().Get() or [])
    orientations = list(instancer.GetOrientationsAttr().Get() or [])
    ids = list(instancer.GetIdsAttr().Get() or [])
    if len(proto_indices) != len(positions):
        raise RuntimeError(f"{prim_path} has incoherent position/prototype arrays")
    if positions and (
        len(scales) != len(positions)
        or len(orientations) != len(positions)
        or len(ids) != len(positions)
    ):
        raise RuntimeError(f"{prim_path} has incomplete transform or stable-ID arrays")
    if len(set(int(value) for value in ids)) != len(ids):
        raise RuntimeError(f"{prim_path} has duplicate stable instance IDs")
    identity_arrays: dict[str, list[Any]] = {}
    identity_names = [
        "fireviewer_stable_id",
        "fireviewer_footprint_radius_m",
        "fireviewer_group_id",
    ]
    if scene_kind == "fictive_variant":
        identity_names.append("fireviewer_source_instance_namespace")
    for name in identity_names:
        attribute = prim.GetAttribute(f"primvars:{name}")
        values = (
            list(attribute.Get() or [])
            if attribute
            and attribute.IsValid()
            and attribute.HasAuthoredValue()
            else []
        )
        if len(values) != len(positions):
            raise RuntimeError(
                f"{prim_path} has no complete per-instance primvars:{name}"
            )
        identity_arrays[name] = values
    stable_ids = [str(value).strip() for value in identity_arrays[
        "fireviewer_stable_id"
    ]]
    group_ids = [str(value).strip() for value in identity_arrays[
        "fireviewer_group_id"
    ]]
    footprint_radii = [
        float(value)
        for value in identity_arrays["fireviewer_footprint_radius_m"]
    ]
    expected_identity_contract = (
        "ids+stable_id+footprint_radius_m+group_id"
        "+source_instance_namespace"
        if scene_kind == "fictive_variant"
        else "ids+stable_id+footprint_radius_m+group_id"
    )
    if (
        any(not value for value in stable_ids)
        or len(set(stable_ids)) != len(stable_ids)
        or any(not value for value in group_ids)
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in footprint_radii
        )
        or prim.GetCustomDataByKey(
            "fireviewer:instance_identity_contract"
        )
        != expected_identity_contract
    ):
        raise RuntimeError(f"{prim_path} has an invalid variant identity contract")
    source_namespaces = (
        [
            int(value)
            for value in identity_arrays[
                "fireviewer_source_instance_namespace"
            ]
        ]
        if scene_kind == "fictive_variant"
        else [expected_instance_namespace] * len(ids)
    )

    targets = list(instancer.GetPrototypesRel().GetTargets())
    if not targets:
        raise RuntimeError(f"{prim_path} has no photoreal prototypes")
    prefix = f"{prim_path}/"
    if any(not str(target).startswith(prefix) for target in targets):
        raise RuntimeError(
            f"{prim_path} exposes sibling prototypes; origin piles are forbidden"
        )
    if any(index < 0 or index >= len(targets) for index in proto_indices):
        raise RuntimeError(f"{prim_path} contains an invalid protoIndex")

    semantic = _semantic(prim)
    family = _family_from_semantic(semantic)
    if family is None:
        raise RuntimeError(f"{prim_path} has an unsupported instance semantic")
    family_code = INSTANCE_FAMILY_CODES[family]
    for index, value in enumerate(ids):
        encoded = int(value)
        encoded_namespace = encoded >> 43
        if encoded < 0 or ((encoded >> 39) & 0xF) != family_code:
            raise RuntimeError(
                f"{prim_path} has an instance ID outside its tile/family namespace"
            )
        if scene_kind == "fictive_variant":
            if (
                source_namespaces[index] <= 0
                or source_namespaces[index] != encoded_namespace
            ):
                raise RuntimeError(
                    f"{prim_path} lost its source instance namespace"
                )
        elif encoded_namespace != expected_instance_namespace:
            raise RuntimeError(
                f"{prim_path} has an instance ID outside its tile namespace"
            )
    prototype_keys: list[str] = []
    root_layer = stage.GetRootLayer()
    for target in targets:
        prototype = stage.GetPrimAtPath(target)
        if not prototype or not prototype.IsValid():
            raise RuntimeError(f"{prim_path} references an absent prototype {target}")
        references = _reference_items(prototype, root_layer=root_layer)
        if len(references) != 1:
            raise RuntimeError(
                f"{target} must bind exactly one materialized photoreal USD reference"
            )
        layer, raw_reference = references[0]
        resolved = _resolved_reference(
            layer=layer, asset_path=raw_reference, sdf=sdf
        )
        if resolved is None or not resolved.is_file():
            raise RuntimeError(
                f"{target} references an absent local photoreal USD: {raw_reference}"
            )
        custom = prototype.GetCustomData()
        source_asset = str(custom.get("fireviewer:source_asset", "")).strip()
        asset_family = str(custom.get("fireviewer:asset_family", "")).strip()
        expected_family = (
            "buildings." if family == "buildings" else f"vegetation.{family}"
        )
        if not source_asset or not asset_family.startswith(expected_family):
            raise RuntimeError(f"{target} has no compatible photoreal family identity")
        if str(custom.get("fireviewer:lod_role", "")) != "source_identity_lod_chain":
            raise RuntimeError(f"{target} has no source-preserving LOD contract")
        prototype_keys.append(f"{asset_family}:{resolved}")

    usage = Counter(prototype_keys[index] for index in proto_indices)
    xform_cache = usd_geom.XformCache(usd.TimeCode.Default())
    world_positions = list(
        _world_positions(
            prim=prim,
            positions=positions,
            xform_cache=xform_cache,
            gf=gf,
        )
    )
    xmin, ymin, xmax, ymax = expected_tile_bounds
    origin_x, origin_y = origin
    minimum_local_x = xmin - origin_x - 2.0
    maximum_local_x = xmax - origin_x + 2.0
    minimum_local_y = ymin - origin_y - 2.0
    maximum_local_y = ymax - origin_y + 2.0
    unique_xy: set[tuple[int, int]] = set()
    for position, scale in zip(world_positions, scales, strict=True):
        xyz = tuple(float(position[index]) for index in range(3))
        if not all(math.isfinite(value) for value in xyz):
            raise RuntimeError(f"{prim_path} contains a non-finite world position")
        if not (
            minimum_local_x <= xyz[0] <= maximum_local_x
            and minimum_local_y <= xyz[1] <= maximum_local_y
        ):
            raise RuntimeError(f"{prim_path} contains an instance outside its tile")
        scale_values = tuple(float(scale[index]) for index in range(3))
        if (
            not all(math.isfinite(value) and value > 0.0 for value in scale_values)
            or max(scale_values) - min(scale_values) > 1e-4
        ):
            raise RuntimeError(f"{prim_path} contains invalid non-uniform scaling")
        unique_xy.add((round(xyz[0] * 10), round(xyz[1] * 10)))
    if len(positions) >= 50 and len(unique_xy) < math.ceil(len(positions) * 0.98):
        raise RuntimeError(f"{prim_path} is abnormally concentrated on duplicate XY")
    identity_rows = sorted(
        (
            int(ids[index]),
            stable_ids[index],
            round(footprint_radii[index], 6),
            group_ids[index],
            source_namespaces[index],
            round(float(world_positions[index][0]), 6),
            round(float(world_positions[index][1]), 6),
            tuple(
                round(float(scales[index][axis]), 6)
                for axis in range(3)
            ),
            str(orientations[index]),
        )
        for index in range(len(ids))
    )
    return {
        "path": prim_path,
        "semantic": semantic,
        "family": family,
        "instances": len(positions),
        "usage": usage,
        "world_positions": world_positions,
        "unique_xy": len(unique_xy),
        "identity_primvars": {
            "stable_ids": len(stable_ids),
            "footprint_radii": len(footprint_radii),
            "group_ids": len(group_ids),
        },
        "identity_transform_sha256": hashlib.sha256(
            json.dumps(
                identity_rows,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _asset_value_path(value: Any) -> str:
    return str(getattr(value, "path", value) or "").strip()


def _inspect_terrain_payload(
    *,
    path: Path,
    tile_ref: str,
    expected_lods: set[str],
    usd: Any,
    usd_geom: Any,
) -> dict[str, Any]:
    stage = usd.Stage.Open(str(path), load=usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open terrain payload {path}")
    try:
        summary = _direct_layer_summary(stage)
        if summary["instancer_paths"]:
            raise RuntimeError(f"{path.name} terrain unexpectedly contains instances")
        tile = stage.GetPrimAtPath("/Tile")
        if not tile or not tile.IsValid():
            raise RuntimeError(f"{path.name} has no /Tile root")
        if str(tile.GetCustomData().get("fireviewer:tile_ref", "")) != tile_ref:
            raise RuntimeError(f"{path.name} tile identity differs from the receipt")
        collision_variant_set = tile.GetVariantSets().GetVariantSet(
            "collisionLOD"
        )
        collision_variant_names = set(
            collision_variant_set.GetVariantNames()
        )
        if collision_variant_names != {"NEAR", "FAR"}:
            raise RuntimeError(
                f"{path.name} collision variants are incomplete: "
                f"{sorted(collision_variant_names)}"
            )
        if collision_variant_set.GetVariantSelection() != "FAR":
            raise RuntimeError(
                f"{path.name} default collision variant must be FAR"
            )

        variant_set = tile.GetVariantSets().GetVariantSet("terrainLOD")
        variant_names = set(variant_set.GetVariantNames())
        if variant_names != expected_lods:
            raise RuntimeError(
                f"{path.name} terrain variants {sorted(variant_names)} differ "
                f"from receipt {sorted(expected_lods)}"
            )
        if variant_set.GetVariantSelection() != "LOD1":
            raise RuntimeError(f"{path.name} default terrain variant must be LOD1")
        stage.SetEditTarget(stage.GetSessionLayer())
        collision_point_counts: dict[str, int] = {}
        for collision_lod in ("NEAR", "FAR"):
            if not collision_variant_set.SetVariantSelection(collision_lod):
                raise RuntimeError(
                    f"{path.name} cannot select collision {collision_lod}"
                )
            collision = stage.GetPrimAtPath("/Tile/Collision")
            if not collision or not collision.IsA(usd_geom.Mesh):
                raise RuntimeError(
                    f"{path.name} {collision_lod} has no source-backed "
                    "collision mesh"
                )
            points = list(
                usd_geom.Mesh(collision).GetPointsAttr().Get() or []
            )
            if not points:
                raise RuntimeError(
                    f"{path.name} {collision_lod} collision mesh is empty"
                )
            if "terrain_collision" not in _semantic(collision):
                raise RuntimeError(
                    f"{path.name} {collision_lod} collision semantic is absent"
                )
            if (
                collision.GetCustomData().get("fireviewer:collision_lod")
                != collision_lod
            ):
                raise RuntimeError(
                    f"{path.name} {collision_lod} collision identity is absent"
                )
            collision_point_counts[collision_lod] = len(points)
        if (
            collision_point_counts["NEAR"]
            <= collision_point_counts["FAR"]
        ):
            raise RuntimeError(
                f"{path.name} NEAR collision does not improve on FAR"
            )

        point_counts: dict[str, int] = {}
        for lod in sorted(expected_lods):
            if not variant_set.SetVariantSelection(lod):
                raise RuntimeError(f"{path.name} cannot select {lod}")
            terrain = stage.GetPrimAtPath("/Tile/Terrain")
            if not terrain or not terrain.IsA(usd_geom.Mesh):
                raise RuntimeError(f"{path.name} {lod} is not a terrain mesh")
            _replicator_semantic(terrain)
            if str(
                terrain.GetCustomData().get("fireviewer:terrain_lod", "")
            ) != lod:
                raise RuntimeError(f"{path.name} {lod} has no matching LOD identity")
            mesh = usd_geom.Mesh(terrain)
            points = list(mesh.GetPointsAttr().Get() or [])
            if not points:
                raise RuntimeError(f"{path.name} {lod} terrain mesh is empty")
            normals = list(mesh.GetNormalsAttr().Get() or [])
            if (
                len(normals) != len(points)
                or mesh.GetNormalsInterpolation() != usd_geom.Tokens.vertex
            ):
                raise RuntimeError(
                    f"{path.name} {lod} has no complete smooth vertex normals"
                )
            material_binding = terrain.GetRelationship("material:binding")
            if (
                not material_binding
                or not list(material_binding.GetTargets() or [])
            ):
                raise RuntimeError(
                    f"{path.name} {lod} has no bound terrain material"
                )
            point_counts[lod] = len(points)
            imagery_name = "fireviewer:ortho20" if lod == "LOD0" else "fireviewer:ortho50"
            imagery = terrain.GetAttribute(imagery_name)
            imagery_path = _asset_value_path(imagery.Get() if imagery else None)
            if not imagery_path:
                raise RuntimeError(f"{path.name} {lod} has no orthophoto binding")
            source = (
                Path(imagery_path).resolve()
                if Path(imagery_path).is_absolute()
                else (path.parent / imagery_path).resolve()
            )
            if not source.is_file():
                raise RuntimeError(
                    f"{path.name} {lod} orthophoto source is absent: {source}"
                )
        if not (
            point_counts["LOD1"]
            > point_counts["LOD2"]
            > point_counts["LOD3"]
        ):
            raise RuntimeError(f"{path.name} terrain LOD reductions are not monotonic")
        if "LOD0" in point_counts and point_counts["LOD0"] <= point_counts["LOD1"]:
            raise RuntimeError(f"{path.name} LOD0 does not improve on LOD1")
        return {
            "tile_ref": tile_ref,
            "path": str(path),
            "lod_point_counts": point_counts,
            "collision_point_counts": collision_point_counts,
        }
    finally:
        del stage


def _detail_bounds(detail: Any, *, path: Path) -> tuple[float, float, float, float]:
    raw = str(
        detail.GetCustomData().get("fireviewer:epsg2154_bounds", "")
    ).strip()
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as exc:
        raise RuntimeError(f"{path.name} has invalid EPSG:2154 bounds") from exc
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        raise RuntimeError(f"{path.name} has invalid EPSG:2154 bounds")
    if not (
        math.isclose(values[2] - values[0], 1000.0, abs_tol=0.01)
        and math.isclose(values[3] - values[1], 1000.0, abs_tol=0.01)
    ):
        raise RuntimeError(f"{path.name} detail payload is not one kilometre")
    return values  # type: ignore[return-value]


def _inspect_detail_payload(
    *,
    path: Path,
    tile_ref: str,
    expected_counts: dict[str, int],
    expected_instance_namespace: int,
    expected_detail_level: str,
    scene_kind: str,
    origin: tuple[float, float],
    gf: Any,
    sdf: Any,
    usd: Any,
    usd_geom: Any,
) -> dict[str, Any]:
    stage = usd.Stage.Open(str(path), load=usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open detail payload {path}")
    try:
        summary = _direct_layer_summary(stage)
        detail = stage.GetPrimAtPath("/Detail")
        if not detail or not detail.IsValid():
            raise RuntimeError(f"{path.name} has no /Detail root")
        custom = detail.GetCustomData()
        if str(custom.get("fireviewer:tile_ref", "")) != tile_ref:
            raise RuntimeError(f"{path.name} tile identity differs from the receipt")
        if int(custom.get("fireviewer:instance_namespace", 0)) != (
            expected_instance_namespace
        ):
            raise RuntimeError(f"{path.name} instance namespace differs from receipt")
        if str(custom.get("fireviewer:detail_level", "")) != (
            expected_detail_level
        ):
            raise RuntimeError(f"{path.name} detail level differs from receipt")
        try:
            authored_counts = json.loads(
                str(custom.get("fireviewer:layer_counts", ""))
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path.name} layer-count metadata is invalid") from exc
        if not isinstance(authored_counts, dict) or set(authored_counts) != DETAIL_COUNT_KEYS:
            raise RuntimeError(f"{path.name} layer-count metadata is incomplete")
        normalized_authored = {
            key: _nonnegative_count(
                authored_counts[key], label=f"{path.name}.{key}"
            )
            for key in DETAIL_COUNT_KEYS
        }
        if normalized_authored != expected_counts:
            raise RuntimeError(f"{path.name} layer counts differ from the build receipt")
        bounds = _detail_bounds(detail, path=path)
        for semantic_path in summary["semantic_paths"]:
            semantic_prim = stage.GetPrimAtPath(semantic_path)
            if not semantic_prim or not semantic_prim.IsValid():
                raise RuntimeError(
                    f"{path.name} semantic prim is absent: {semantic_path}"
                )
            _replicator_semantic(semantic_prim)

        instancer_results = [
            _inspect_instancer(
                stage=stage,
                prim_path=prim_path,
                detail_path=path,
                expected_tile_bounds=bounds,
                expected_instance_namespace=expected_instance_namespace,
                scene_kind=scene_kind,
                origin=origin,
                gf=gf,
                sdf=sdf,
                usd=usd,
                usd_geom=usd_geom,
            )
            for prim_path in summary["instancer_paths"]
        ]
        actual_buildings = sum(
            item["instances"]
            for item in instancer_results
            if item["family"] == "buildings"
        )
        actual_vegetation = sum(
            item["instances"]
            for item in instancer_results
            if item["family"] in {"trees", "shrubs", "understory"}
        )
        semantic_counts: Counter[str] = summary["semantic_counts"]
        actual_roads = semantic_counts["road"]
        actual_hydrology = (
            semantic_counts["water"] + semantic_counts["watercourse"]
        )
        actual_counts = {
            "buildings": actual_buildings,
            "roads": actual_roads,
            "hydrology": actual_hydrology,
            "vegetation": actual_vegetation,
        }
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"{path.name} authored geometry counts {actual_counts} differ "
                f"from receipt {expected_counts}"
            )
        return {
            "tile_ref": tile_ref,
            "path": str(path),
            "counts": actual_counts,
            "instancers": instancer_results,
        }
    finally:
        del stage


def _root_origin(world: Any) -> tuple[float, float]:
    raw = str(world.GetCustomData().get("fireviewer:epsg2154_origin", ""))
    try:
        values = tuple(float(value) for value in raw.split(","))
    except ValueError as exc:
        raise RuntimeError("review root has an invalid EPSG:2154 origin") from exc
    if len(values) < 2 or not all(math.isfinite(value) for value in values[:2]):
        raise RuntimeError("review root has an invalid EPSG:2154 origin")
    return values[0], values[1]


def _inspect_root_stage(
    *,
    root: Path,
    expected_tile_refs: set[str],
    required_reference_paths: set[Path],
    scene_kind: str,
    usd: Any,
    usd_geom: Any,
) -> dict[str, Any]:
    stage = usd.Stage.Open(str(root), load=usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError("OpenUSD could not compose the review root")
    try:
        terrain = stage.GetPrimAtPath("/World/Terrain")
        if not terrain or not terrain.IsValid():
            raise RuntimeError("review scene has no terrain root")
        world = stage.GetPrimAtPath("/World")
        if not world or not world.IsValid():
            raise RuntimeError("review scene has no /World root")
        fire = stage.GetPrimAtPath("/World/FireAndSmoke")
        if not fire or not fire.IsValid():
            raise RuntimeError("review scene has no locked fire layer")
        if fire.HasAuthoredPayloads():
            raise RuntimeError("review scene must not compose Flow before Editor acceptance")
        if usd_geom.Imageable(fire).ComputeVisibility() != usd_geom.Tokens.invisible:
            raise RuntimeError("fire/smoke must remain invisible before Editor acceptance")

        tile_refs: set[str] = set()
        terrain_arcs = 0
        detail_arcs = 0
        ground_material_arcs = 0
        required_detail_children = (
            ("Details", "DetailsMid", "DetailsFar")
            if scene_kind == "fictive_variant"
            else ("Details",)
        )
        for prim in stage.TraverseAll():
            custom = prim.GetCustomData()
            tile_ref = str(custom.get("fireviewer:tile_ref", "")).strip()
            if not tile_ref:
                continue
            if tile_ref in tile_refs:
                raise RuntimeError(f"root composition duplicates tile {tile_ref}")
            tile_refs.add(tile_ref)
            terrain_arc = prim.GetChild("Terrain")
            if (
                not terrain_arc
                or not terrain_arc.IsValid()
                or not terrain_arc.HasAuthoredPayloads()
            ):
                raise RuntimeError(f"root tile {tile_ref} has no terrain payload arc")
            for child_name in required_detail_children:
                detail_arc = prim.GetChild(child_name)
                if (
                    not detail_arc
                    or not detail_arc.IsValid()
                    or not detail_arc.HasAuthoredPayloads()
                ):
                    raise RuntimeError(
                        f"root tile {tile_ref} has no {child_name} payload arc"
                    )
                detail_arcs += 1
            if scene_kind == "fictive_variant":
                ground_material = terrain_arc.GetChild("GroundMaterial")
                if (
                    not ground_material
                    or not ground_material.IsValid()
                    or not ground_material.HasAuthoredPayloads()
                ):
                    raise RuntimeError(
                        f"root tile {tile_ref} has no tiled ground material payload"
                    )
                binding = terrain_arc.GetRelationship("material:binding")
                if not binding or list(binding.GetTargets()) != [
                    ground_material.GetPath()
                ]:
                    raise RuntimeError(
                        f"root tile {tile_ref} has no exact ground material binding"
                    )
                ground_material_arcs += 1
            terrain_arcs += 1
        if tile_refs != expected_tile_refs:
            raise RuntimeError("root composition does not expose all 400 locked tiles")
        expected_detail_arcs = (
            1200 if scene_kind == "fictive_variant" else 400
        )
        if (
            terrain_arcs != 400
            or detail_arcs != expected_detail_arcs
            or (
                scene_kind == "fictive_variant"
                and ground_material_arcs != 400
            )
        ):
            raise RuntimeError(
                "root composition has an incomplete terrain/detail/ground "
                "payload topology"
            )

        used_local_layers = {
            Path(str(layer.realPath)).resolve()
            for layer in stage.GetUsedLayers()
            if str(layer.realPath or "")
        }
        missing_references = required_reference_paths - used_local_layers
        if missing_references:
            raise RuntimeError(
                "root composition omits locked reference layers: "
                + ", ".join(path.name for path in sorted(missing_references)[:8])
            )
        return {
            "origin": _root_origin(world),
            "terrain_payload_arcs": terrain_arcs,
            "detail_payload_arcs": detail_arcs,
            "ground_material_payload_arcs": ground_material_arcs,
            "used_layer_count": len(used_local_layers),
        }
    finally:
        del stage


def _inspect_aggregate_bindings(
    *,
    aggregate_paths: set[Path],
    by_tile: dict[str, dict[str, Any]],
    terrain_paths: dict[str, Path],
    detail_paths_by_level: dict[str, dict[str, Path]],
    sdf: Any,
    usd: Any,
) -> dict[str, Any]:
    """Bind every lightweight aggregate arc to its exact locked tile file."""

    seen_tiles: set[str] = set()
    for aggregate_path in sorted(aggregate_paths):
        stage = usd.Stage.Open(str(aggregate_path), load=usd.Stage.LoadNone)
        if stage is None:
            raise RuntimeError(f"OpenUSD could not open aggregate {aggregate_path}")
        try:
            root_layer = stage.GetRootLayer()
            for prim in stage.TraverseAll():
                tile_ref = str(
                    prim.GetCustomData().get("fireviewer:tile_ref", "")
                ).strip()
                if not tile_ref:
                    continue
                if tile_ref in seen_tiles or tile_ref not in by_tile:
                    raise RuntimeError(
                        f"aggregate composition has an invalid tile identity: {tile_ref}"
                    )
                terrain_arc = prim.GetChild("Terrain")
                expected = by_tile[tile_ref]
                arcs = (
                    (
                        "terrain",
                        prim.GetChild("Terrain"),
                        terrain_paths[expected["terrain_path"]],
                    ),
                    (
                        "HERO detail",
                        prim.GetChild("Details"),
                        detail_paths_by_level["HERO"][
                            expected["detail_paths"]["HERO"]
                        ],
                    ),
                    (
                        "MID detail",
                        prim.GetChild("DetailsMid"),
                        detail_paths_by_level["MID"][
                            expected["detail_paths"]["MID"]
                        ],
                    ),
                    (
                        "FAR detail",
                        prim.GetChild("DetailsFar"),
                        detail_paths_by_level["FAR"][
                            expected["detail_paths"]["FAR"]
                        ],
                    ),
                )
                for label, arc, expected_path in arcs:
                    if not arc or not arc.IsValid():
                        raise RuntimeError(
                            f"{aggregate_path.name} {tile_ref} has no {label} arc"
                        )
                    items = _payload_items(arc, root_layer=root_layer)
                    if len(items) != 1:
                        raise RuntimeError(
                            f"{aggregate_path.name} {tile_ref} must bind exactly "
                            f"one {label} payload"
                        )
                    layer, raw_path = items[0]
                    resolved = _resolved_reference(
                        layer=layer, asset_path=raw_path, sdf=sdf
                    )
                    if resolved != expected_path:
                        raise RuntimeError(
                            f"{aggregate_path.name} {tile_ref} {label} arc "
                            "differs from the locked build receipt"
                        )
                seen_tiles.add(tile_ref)
        finally:
            del stage
    if seen_tiles != set(by_tile):
        raise RuntimeError("aggregate layers do not bind every locked tile exactly once")
    return {
        "aggregate_count": len(aggregate_paths),
        "bound_tile_count": len(seen_tiles),
        "terrain_arcs": len(seen_tiles),
        "detail_arcs": len(seen_tiles) * 3,
    }


def _inspect_variant_root_bindings(
    *,
    root: Path,
    by_tile: dict[str, dict[str, Any]],
    terrain_paths: dict[str, Path],
    detail_paths_by_level: dict[str, dict[str, Path]],
    ground_paths_by_tile: dict[str, Path],
    water_paths: set[Path],
    sdf: Any,
    usd: Any,
) -> dict[str, Any]:
    """Prove every final SIM root payload arc against its locked receipt."""

    stage = usd.Stage.Open(str(root), load=usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError("OpenUSD could not open the final variant root")
    try:
        root_layer = stage.GetRootLayer()
        seen_tiles: set[str] = set()
        for prim in stage.TraverseAll():
            tile_ref = str(
                prim.GetCustomData().get("fireviewer:tile_ref", "")
            ).strip()
            if not tile_ref:
                continue
            if tile_ref in seen_tiles or tile_ref not in by_tile:
                raise RuntimeError(
                    f"variant root has an invalid tile identity: {tile_ref}"
                )
            expected = by_tile[tile_ref]
            terrain_arc = prim.GetChild("Terrain")
            arcs = (
                (
                    "terrain",
                    terrain_arc,
                    terrain_paths[expected["terrain_path"]],
                ),
                (
                    "HERO detail",
                    prim.GetChild("Details"),
                    detail_paths_by_level["HERO"][
                        expected["detail_paths"]["HERO"]
                    ],
                ),
                (
                    "MID detail",
                    prim.GetChild("DetailsMid"),
                    detail_paths_by_level["MID"][
                        expected["detail_paths"]["MID"]
                    ],
                ),
                (
                    "FAR detail",
                    prim.GetChild("DetailsFar"),
                    detail_paths_by_level["FAR"][
                        expected["detail_paths"]["FAR"]
                    ],
                ),
                (
                    "ground material",
                    terrain_arc.GetChild("GroundMaterial"),
                    ground_paths_by_tile[tile_ref],
                ),
            )
            for label, arc, expected_path in arcs:
                if not arc or not arc.IsValid():
                    raise RuntimeError(
                        f"variant root {tile_ref} has no {label} arc"
                    )
                items = _payload_items(arc, root_layer=root_layer)
                if len(items) != 1:
                    raise RuntimeError(
                        f"variant root {tile_ref} must bind exactly one "
                        f"{label} payload"
                    )
                layer, raw_path = items[0]
                resolved = _resolved_reference(
                    layer=layer,
                    asset_path=raw_path,
                    sdf=sdf,
                )
                if resolved != expected_path:
                    raise RuntimeError(
                        f"variant root {tile_ref} {label} differs from "
                        "the locked build receipt"
                    )
            seen_tiles.add(tile_ref)
        if seen_tiles != set(by_tile):
            raise RuntimeError(
                "variant root does not bind every locked tile exactly once"
            )

        water_root = stage.GetPrimAtPath("/World/Water")
        if not water_root or not water_root.IsValid():
            raise RuntimeError("variant root has no water layer")
        if (
            water_root.HasAuthoredPayloads()
            or any(
                child.HasAuthoredPayloads()
                for child in water_root.GetChildren()
            )
        ):
            raise RuntimeError(
                "variant root must not duplicate source water over the tiled "
                "HERO/MID/FAR hydrology"
            )
        return {
            "aggregate_count": 0,
            "bound_tile_count": len(seen_tiles),
            "terrain_arcs": len(seen_tiles),
            "detail_arcs": len(seen_tiles) * 3,
            "ground_material_arcs": len(seen_tiles),
            "water_arcs": 0,
            "locked_water_source_payloads": len(water_paths),
        }
    finally:
        del stage


def validate_scene(
    *,
    root_usd: Path,
    build_receipt: Path,
    asset_manifest: Path,
    output_path: Path,
    minimum_tree_instances: int,
    minimum_building_instances: int,
    minimum_forest_span_metres: float,
    volume_root: Path | None = None,
) -> dict[str, Any]:
    if minimum_tree_instances <= 0 or minimum_building_instances <= 0:
        raise ValueError("minimum instance counts must be positive")
    if not math.isfinite(minimum_forest_span_metres) or minimum_forest_span_metres <= 0:
        raise ValueError("minimum forest span must be positive")
    root = root_usd.resolve()
    build = build_receipt.resolve()
    assets = asset_manifest.resolve()
    if not root.is_file() or root.suffix.lower() not in USD_SUFFIXES:
        raise RuntimeError("review root USD is absent")
    build_payload = _read_json(build, label="build receipt")
    if not assets.is_file():
        raise RuntimeError("materialized asset manifest is absent")
    contract = _coverage_contract(build_payload)
    zone_root = _infer_zone_root(
        build_receipt=build,
        root_usd=root,
        root_record=build_payload.get("root_usd"),
    )
    if contract["scene_kind"] == "fictive_variant":
        if volume_root is None:
            raise ValueError(
                "final variant validation requires an explicit persistent "
                "volume root"
            )
        artifact_root = volume_root.resolve()
        if (
            not artifact_root.is_dir()
            or not _is_below(artifact_root, zone_root)
            or not _is_below(artifact_root, assets)
        ):
            raise RuntimeError(
                "variant scene or assets escape the persistent volume"
            )
    else:
        artifact_root = zone_root

    locked_root = _resolve_locked_artifact(
        zone_root=zone_root,
        record=build_payload.get("root_usd"),
        label="root USD",
        suffixes=USD_SUFFIXES,
        allowed_root=artifact_root,
    )
    if locked_root != root:
        raise RuntimeError("build receipt is bound to a different root USD")
    if build_payload.get("fire_simulation_status") != "blocked_pending_editor_review":
        raise RuntimeError("scene build did not preserve the manual simulation gate")
    asset_lock = build_payload.get("asset_lock")
    asset_entries = asset_lock.get("assets") if isinstance(asset_lock, dict) else None
    if contract["scene_kind"] == "fictive_variant":
        shared_manifest = (
            asset_lock.get("shared_manifest")
            if isinstance(asset_lock, dict)
            else None
        )
        if (
            not isinstance(asset_entries, list)
            or not asset_entries
            or not isinstance(shared_manifest, dict)
            or shared_manifest.get("sha256") != _sha256(assets)
        ):
            raise RuntimeError(
                "variant build is not bound to the current shared photoreal "
                "asset manifest"
            )
        locked_shared_manifest = _resolve_locked_artifact(
            zone_root=zone_root,
            record=shared_manifest,
            label="shared photoreal asset manifest",
            allowed_root=artifact_root,
        )
        if locked_shared_manifest != assets:
            raise RuntimeError(
                "variant asset lock resolves to another manifest"
            )
    else:
        shared_asset = next(
            (
                item
                for item in asset_entries or []
                if isinstance(item, dict)
                and item.get("id")
                == "shared-materialized-simready-environment"
            ),
            None,
        )
        if (
            not isinstance(shared_asset, dict)
            or shared_asset.get("manifest_sha256") != _sha256(assets)
        ):
            raise RuntimeError(
                "scene build is not bound to the current photoreal asset "
                "manifest"
            )

    terrain_paths = {
        raw: _resolve_locked_artifact(
            zone_root=zone_root,
            record=record,
            label=f"terrain payload {index + 1}",
            suffixes=USD_SUFFIXES,
            allowed_root=artifact_root,
        )
        for index, (raw, record) in enumerate(
            contract["terrain_records"].items()
        )
    }
    detail_paths_by_level = {
        level: {
            raw: _resolve_locked_artifact(
                zone_root=zone_root,
                record=record,
                label=f"{level} detail payload {index + 1}",
                suffixes=USD_SUFFIXES,
                allowed_root=artifact_root,
            )
            for index, (raw, record) in enumerate(
                contract["detail_records_by_level"][level].items()
            )
        }
        for level in ("HERO", "MID", "FAR")
    }
    locked_aggregates: list[tuple[Path, str]] = []
    aggregate_paths: set[Path] = set()
    ground_index_path: Path | None = None
    ground_index_sha = ""
    ground_paths_by_tile: dict[str, Path] = {}
    locked_ground_materials: list[tuple[Path, str]] = []
    locked_waters: list[tuple[Path, str]] = []
    water_paths: set[Path] = set()
    if contract["scene_kind"] == "fictive_variant":
        ground = build_payload.get("ground_material")
        ground_tiles = (
            ground.get("tile_material_payloads")
            if isinstance(ground, dict)
            else None
        )
        if (
            not isinstance(ground, dict)
            or ground.get("topology")
            != "payload_tiled_materials_shared_pbr_library"
            or ground.get("binding_scope")
            != "per_terrain_tile_stronger_than_descendants"
            or not isinstance(ground_tiles, list)
            or len(ground_tiles) != 400
        ):
            raise ValueError(
                "variant build has no exact 400-tile ground material contract"
            )
        ground_index_path = _resolve_locked_artifact(
            zone_root=zone_root,
            record=ground.get("index"),
            label="ground material index",
            suffixes=USD_SUFFIXES,
            allowed_root=artifact_root,
        )
        ground_index_sha = str(
            ground.get("index", {}).get("sha256", "")
        ).lower()
        for index, record in enumerate(ground_tiles):
            if not isinstance(record, dict):
                raise ValueError(
                    f"ground material payload {index + 1} is malformed"
                )
            tile_id = str(record.get("tile_id", "")).strip()
            if not tile_id or tile_id in ground_paths_by_tile:
                raise ValueError(
                    "ground material tile identities must be present and unique"
                )
            path = _resolve_locked_artifact(
                zone_root=zone_root,
                record=record,
                label=f"ground material payload {index + 1}",
                suffixes=USD_SUFFIXES,
                allowed_root=artifact_root,
            )
            ground_paths_by_tile[tile_id] = path
            locked_ground_materials.append(
                (path, str(record.get("sha256", "")).lower())
            )
        if set(ground_paths_by_tile) != set(contract["by_tile"]):
            raise ValueError(
                "ground material payloads do not cover the exact terrain tiles"
            )
        water_records = build_payload.get("water_payloads")
        if not isinstance(water_records, list) or not water_records:
            raise ValueError("variant build has no locked water payloads")
        locked_waters = [
            (
                _resolve_locked_artifact(
                    zone_root=zone_root,
                    record=record,
                    label=f"water payload {index + 1}",
                    suffixes=USD_SUFFIXES,
                    allowed_root=artifact_root,
                ),
                str(record.get("sha256", "")).lower(),
            )
            for index, record in enumerate(water_records)
        ]
        water_paths = {path for path, _sha in locked_waters}
        if len(water_paths) != len(locked_waters):
            raise ValueError("water payload paths must be unique")
    else:
        aggregate_records = build_payload.get("aggregates_5km")
        if not isinstance(aggregate_records, list) or not aggregate_records:
            raise ValueError("build receipt has no aggregate payload catalog")
        locked_aggregates = [
            (
                _resolve_locked_artifact(
                    zone_root=zone_root,
                    record=record,
                    label=f"aggregate {index + 1}",
                    suffixes=USD_SUFFIXES,
                    allowed_root=artifact_root,
                ),
                str(record.get("sha256", "")).lower(),
            )
            for index, record in enumerate(aggregate_records)
        ]
        aggregate_paths = {path for path, _sha in locked_aggregates}
        if len(aggregate_paths) != len(aggregate_records):
            raise ValueError("aggregate paths must be unique")
    camera_record = build_payload.get("cameras")
    if (
        not isinstance(camera_record, dict)
        or _nonnegative_count(
            camera_record.get("count"), label="cameras.count"
        )
        <= 0
    ):
        raise ValueError("build receipt has no review camera catalog")
    cameras_path = _resolve_locked_artifact(
        zone_root=zone_root,
        record=camera_record,
        label="review cameras",
        suffixes=USD_SUFFIXES,
        allowed_root=artifact_root,
    )

    from pxr import Gf, Sdf, Usd, UsdGeom

    root_result = _inspect_root_stage(
        root=root,
        expected_tile_refs=set(contract["by_tile"]),
        required_reference_paths=aggregate_paths | {cameras_path},
        scene_kind=contract["scene_kind"],
        usd=Usd,
        usd_geom=UsdGeom,
    )
    if contract["scene_kind"] == "fictive_variant":
        aggregate_result = _inspect_variant_root_bindings(
            root=root,
            by_tile=contract["by_tile"],
            terrain_paths=terrain_paths,
            detail_paths_by_level=detail_paths_by_level,
            ground_paths_by_tile=ground_paths_by_tile,
            water_paths=water_paths,
            sdf=Sdf,
            usd=Usd,
        )
    else:
        aggregate_result = _inspect_aggregate_bindings(
            aggregate_paths=aggregate_paths,
            by_tile=contract["by_tile"],
            terrain_paths=terrain_paths,
            detail_paths_by_level=detail_paths_by_level,
            sdf=Sdf,
            usd=Usd,
        )
    origin = root_result["origin"]
    terrain_results: list[dict[str, Any]] = []
    detail_tile_receipts: list[dict[str, Any]] = []
    family_counts = Counter()
    prototype_usage: dict[str, Counter[str]] = {
        "buildings": Counter(),
        "trees": Counter(),
        "shrubs": Counter(),
        "understory": Counter(),
    }
    forest_min = [math.inf, math.inf, math.inf]
    forest_max = [-math.inf, -math.inf, -math.inf]
    forest_near_origin = 0
    forest_unique_xy_count = 0
    spatial_digest = hashlib.sha256()

    for tile_index, (tile_ref, tile) in enumerate(
        sorted(contract["by_tile"].items())
    ):
        terrain_result = _inspect_terrain_payload(
            path=terrain_paths[tile["terrain_path"]],
            tile_ref=tile_ref,
            expected_lods=tile["lods"],
            usd=Usd,
            usd_geom=UsdGeom,
        )
        terrain_results.append(terrain_result)
        level_results = {
            level: _inspect_detail_payload(
                path=detail_paths_by_level[level][
                    tile["detail_paths"][level]
                ],
                tile_ref=tile_ref,
                expected_counts=tile["lod_counts"][level],
                expected_instance_namespace=tile["instance_namespace"],
                expected_detail_level=level,
                scene_kind=contract["scene_kind"],
                origin=origin,
                gf=Gf,
                sdf=Sdf,
                usd=Usd,
                usd_geom=UsdGeom,
            )
            for level in ("HERO", "MID", "FAR")
        }
        level_identity = {
            level: {
                (
                    item["family"],
                    item["path"].split("/Prototypes/", 1)[0],
                ): item["identity_transform_sha256"]
                for item in level_results[level]["instancers"]
            }
            for level in ("HERO", "MID", "FAR")
        }
        if not (
            level_identity["HERO"]
            == level_identity["MID"]
            == level_identity["FAR"]
        ):
            raise RuntimeError(
                f"{tile_ref} HERO/MID/FAR changed object identity, "
                "placement or scale"
            )
        detail_result = level_results["HERO"]
        compact_instancers: list[dict[str, Any]] = []
        for instancer in detail_result["instancers"]:
            family = instancer["family"]
            family_counts[family] += instancer["instances"]
            prototype_usage[family].update(instancer["usage"])
            compact_instancers.append(
                {
                    "path": instancer["path"],
                    "family": family,
                    "instances": instancer["instances"],
                    "unique_xy": instancer["unique_xy"],
                }
            )
            if family in {"trees", "shrubs", "understory"}:
                forest_unique_xy_count += int(instancer["unique_xy"])
                for position in instancer["world_positions"]:
                    xyz = tuple(float(position[index]) for index in range(3))
                    for axis in range(3):
                        forest_min[axis] = min(forest_min[axis], xyz[axis])
                        forest_max[axis] = max(forest_max[axis], xyz[axis])
                    if math.hypot(xyz[0], xyz[1]) <= 25.0:
                        forest_near_origin += 1
                    spatial_digest.update(struct.pack("<ddd", *xyz))
        detail_tile_receipts.append(
            {
                "tile_ref": tile_ref,
                "path": detail_result["path"],
                "counts": detail_result["counts"],
                "lod_counts": {
                    level: level_results[level]["counts"]
                    for level in ("HERO", "MID", "FAR")
                },
                "instancers": compact_instancers,
            }
        )
        del level_results
        # Ensure Kit releases each stage and referenced asset graph before the
        # next working set.  Only compact numeric summaries remain resident.
        if (tile_index + 1) % 16 == 0:
            gc.collect()

    vegetation_count = (
        family_counts["trees"]
        + family_counts["shrubs"]
        + family_counts["understory"]
    )
    if vegetation_count != contract["totals"]["vegetation"]:
        raise RuntimeError("vegetation instance total differs from the build receipt")
    if family_counts["buildings"] != contract["totals"]["buildings"]:
        raise RuntimeError("building instance total differs from the build receipt")
    if vegetation_count < minimum_tree_instances:
        raise RuntimeError(
            f"scene exposes {vegetation_count} vegetation instances; "
            f"{minimum_tree_instances} required"
        )
    if family_counts["buildings"] < minimum_building_instances:
        raise RuntimeError(
            f"scene exposes {family_counts['buildings']} buildings; "
            f"{minimum_building_instances} required"
        )
    required_vegetation_families = (
        ("trees",)
        if contract["scene_kind"] == "fictive_variant"
        else ("trees", "shrubs", "understory")
    )
    missing_families = [
        family
        for family in required_vegetation_families
        if family_counts[family] <= 0
    ]
    if missing_families:
        raise RuntimeError(
            "scene is missing vegetation instance families: "
            + ", ".join(missing_families)
        )
    if forest_unique_xy_count < math.ceil(vegetation_count * 0.98):
        raise RuntimeError("forest contains excessive duplicate world XY positions")
    if forest_near_origin > max(8, int(vegetation_count * 0.002)):
        raise RuntimeError("forest is abnormally concentrated at the scene origin")
    forest_span = [
        forest_max[index] - forest_min[index] for index in range(3)
    ]
    if min(forest_span[:2]) < minimum_forest_span_metres:
        raise RuntimeError(
            f"forest covers only {forest_span[0]:.1f} x "
            f"{forest_span[1]:.1f} metres"
        )
    prototype_results = {
        family: _prototype_share_gate(usage, label=family)
        for family, usage in prototype_usage.items()
    }

    receipt = {
        "schema_version": 2,
        "validated_at": datetime.now(UTC).isoformat(),
        # Existing orchestration consumes this machine-gate state.  The
        # separate fire_simulation_status and validation_scope keep it
        # impossible to mistake for Editor acceptance.
        "state": "AUTO_VALIDATED",
        "scene_kind": contract["scene_kind"],
        "variant_contract": contract["variant_contract"],
        "validation_scope": (
            "incremental_structural_geometric_gate_not_rtx_or_human_review"
        ),
        "root_usd": str(root),
        "root_usd_sha256": _sha256(root),
        "build_receipt_sha256": _sha256(build),
        "asset_manifest_sha256": _sha256(assets),
        "fire_simulation_status": "blocked_pending_editor_review",
        "streaming": {
            "root_initial_load_set": "LoadNone",
            "terrain_payloads_inspected_incrementally": len(terrain_results),
            "detail_payloads_inspected_incrementally": len(detail_tile_receipts),
            "simultaneously_retained_detail_stages": 1,
            "root_terrain_payload_arcs": root_result["terrain_payload_arcs"],
            "root_detail_payload_arcs": root_result["detail_payload_arcs"],
            "root_ground_material_payload_arcs": root_result[
                "ground_material_payload_arcs"
            ],
            "aggregate_bindings": aggregate_result,
        },
        "terrain": {
            "payload_count": len(terrain_results),
            "lod0_tile_count": contract["lod0_tiles"],
            "tiles": terrain_results,
        },
        "details": {
            "payload_count": len(detail_tile_receipts),
            "expected_totals": contract["totals"],
            "tiles": detail_tile_receipts,
        },
        "vegetation_instances": vegetation_count,
        "vegetation_family_instances": {
            family: family_counts[family]
            for family in ("trees", "shrubs", "understory")
        },
        "building_instances": family_counts["buildings"],
        "prototype_usage": prototype_results,
        "forest_world_bounds": {
            "minimum": forest_min,
            "maximum": forest_max,
            "span_metres": forest_span,
        },
        "forest_near_origin_instances": forest_near_origin,
        "forest_unique_xy": forest_unique_xy_count,
        "spatial_signature": spatial_digest.hexdigest(),
        "used_layers": [
            {
                "path": str(root),
                "sha256": str(build_payload["root_usd"]["sha256"]).lower(),
            }
        ]
        + [
            {"path": str(path), "sha256": sha}
            for path, sha in sorted(locked_aggregates)
        ]
        + [
            {
                "path": str(cameras_path),
                "sha256": str(camera_record["sha256"]).lower(),
            }
        ]
        + (
            [
                {
                    "path": str(ground_index_path),
                    "sha256": ground_index_sha,
                    "role": "ground_material_index",
                }
            ]
            + [
                {
                    "path": str(path),
                    "sha256": sha,
                    "role": "ground_material_tile",
                }
                for path, sha in sorted(locked_ground_materials)
            ]
            + [
                {
                    "path": str(path),
                    "sha256": sha,
                    "role": "water_payload",
                }
                for path, sha in sorted(locked_waters)
            ]
            if contract["scene_kind"] == "fictive_variant"
            else []
        )
        + [
            {
                "path": str(terrain_paths[raw]),
                "sha256": str(record["sha256"]).lower(),
            }
            for raw, record in sorted(contract["terrain_records"].items())
        ]
        + [
            {
                "path": str(detail_paths_by_level[level][raw]),
                "sha256": str(record["sha256"]).lower(),
                "detail_level": level,
            }
            for level in ("HERO", "MID", "FAR")
            for raw, record in sorted(
                contract["detail_records_by_level"][level].items()
            )
        ],
    }
    _atomic_json(output_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally validate one tiled FireViewer photoreal review scene"
    )
    parser.add_argument("--root-usd", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument(
        "--volume-root",
        type=Path,
        help=(
            "persistent artifact root; mandatory for final SIM variants "
            "that reference shared base payloads"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-tree-instances", type=int, default=25_000)
    parser.add_argument("--minimum-building-instances", type=int, default=1)
    parser.add_argument("--minimum-forest-span-metres", type=float, default=2_000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = validate_scene(
        root_usd=args.root_usd,
        build_receipt=args.build_receipt,
        asset_manifest=args.asset_manifest,
        output_path=args.output,
        minimum_tree_instances=args.minimum_tree_instances,
        minimum_building_instances=args.minimum_building_instances,
        minimum_forest_span_metres=args.minimum_forest_span_metres,
        volume_root=args.volume_root,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
