"""Restore one Level-1 site input from a previously exported Z16 lock.

The historical pod export deliberately excluded the large raw source tree.
This utility restores exactly the sources recorded in ``source-lock.json``
into a fresh site input, preserving the original SHA-256 locks and resumable
partial transfers.  A compact LiDAR-raster profile derives one 4 km × 4 km
study site from the historical full lock, retaining MNT/MNS/MNH rasters and
orthophotos while excluding raw point clouds.  It never substitutes actor
assets: the accompanying asset freeze requirement is valid only when the
original wrapper files reappear with their recorded digests.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

# The restorer is intentionally executable directly from a background pod or
# Windows process.  Add this repository's src root only for that direct-script
# entrypoint; installed/module execution keeps its normal import resolution.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fireviewer_sdg import zone_scenes
from fireviewer_sdg.artifacts import sha256, write_json


SCHEMA_VERSION = 1
SOURCE_LOCK_NAME = "source-lock.json"
FULL_SOURCE_LOCK_NAME = "source-lock.full-raw-lidar.json"
ASSET_REQUIREMENTS_NAME = "asset-freeze-requirements.json"
STATUS_NAME = "restore-status.json"
RAW_SOURCE_DATASETS = frozenset({"lidar"})
FULL_SOURCE_PROFILE = "full"
COMPACT_LIDAR_RASTER_SOURCE_PROFILE = "compact_lidar_raster_1m_v1"
COMPACT_SITE_TILE_SIDE = 4
COMPACT_SITE_TILE_COUNT = 16
COMPACT_SITE_RASTER_DATASETS = frozenset({"mnt", "mns", "mnh", "ortho20", "ortho50"})
COMPACT_SITE_ENTRY_COUNT = COMPACT_SITE_TILE_COUNT * len(COMPACT_SITE_RASTER_DATASETS)
_TILE_REF = re.compile(r"^L93_(?P<x>\d{4})_(?P<y>\d{4})$")

# Four non-overlapping 4 km × 4 km study sites inside the historical Z16
# envelope.  One site is activated for the SIM-01 pilot; the remaining three
# are independent candidates for the later 20-scene campaign.
COMPACT_STUDY_SITES: tuple[tuple[str, int, int], ...] = (
    ("Z16-base-01", 879, 6402),
    ("Z16-base-02", 884, 6407),
    ("Z16-base-03", 889, 6411),
    ("Z16-base-04", 894, 6415),
)


class Level1RestoreError(RuntimeError):
    """Raised when a Level-1 input cannot be restored faithfully."""


def _source_profile(lock: Mapping[str, Any]) -> str:
    acquisition = lock.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise Level1RestoreError("source lock has no acquisition contract")
    profile = str(acquisition.get("source_profile", "")).strip()
    if profile not in {FULL_SOURCE_PROFILE, COMPACT_LIDAR_RASTER_SOURCE_PROFILE}:
        raise Level1RestoreError(f"unsupported source profile: {profile!r}")
    return profile


def _tile_coordinates(tile_ref: str) -> tuple[int, int]:
    match = _TILE_REF.fullmatch(tile_ref)
    if match is None:
        raise Level1RestoreError(f"invalid Lambert-93 tile reference: {tile_ref}")
    return int(match.group("x")), int(match.group("y"))


def _ortho50_bounds(entry: Mapping[str, Any]) -> list[int]:
    """Read the authoritative Lambert-93 tile bounds from locked WMS data.

    The historical ``L93_x_y`` tile name encodes the northern kilometre edge,
    not the southern edge.  Deriving both edges from that suffix shifted each
    compact site by one kilometre north.  The locked BD ORTHO request already
    carries the exact EPSG:2154 BBOX, so use it rather than recreating a
    coordinate convention from the filename.
    """

    if entry.get("dataset") != "ortho50":
        raise Level1RestoreError("compact tile geometry must come from ortho50")
    url = entry.get("url")
    if not isinstance(url, str):
        raise Level1RestoreError("compact ortho50 entry has no source URL")
    raw_bbox = parse_qs(urlparse(url).query).get("BBOX", [])
    if len(raw_bbox) != 1:
        raise Level1RestoreError("compact ortho50 entry has no EPSG:2154 BBOX")
    try:
        values = [float(value) for value in raw_bbox[0].split(",")]
    except ValueError as exc:
        raise Level1RestoreError("compact ortho50 BBOX is invalid") from exc
    if len(values) != 4 or any(not value.is_integer() for value in values):
        raise Level1RestoreError("compact ortho50 BBOX must use integer metre bounds")
    xmin, ymin, xmax, ymax = (int(value) for value in values)
    if xmax - xmin != 1000 or ymax - ymin != 1000:
        raise Level1RestoreError("compact ortho50 BBOX is not one kilometre square")
    return [xmin, ymin, xmax, ymax]


def _compact_site_records(entries: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    entry_list = list(entries)
    available = {str(entry.get("tile_ref", "")) for entry in entry_list}
    ortho50_by_tile = {
        str(entry.get("tile_ref", "")): entry
        for entry in entry_list
        if entry.get("dataset") == "ortho50"
    }
    sites: dict[str, dict[str, Any]] = {}
    for base_id, origin_x, origin_y in COMPACT_STUDY_SITES:
        tile_refs = [
            f"L93_{x:04d}_{y:04d}"
            for y in range(origin_y, origin_y + COMPACT_SITE_TILE_SIDE)
            for x in range(origin_x, origin_x + COMPACT_SITE_TILE_SIDE)
        ]
        missing = sorted(set(tile_refs) - available)
        if missing:
            raise Level1RestoreError(
                f"compact site {base_id} has missing locked tiles: {', '.join(missing)}"
            )
        if set(tile_refs) - set(ortho50_by_tile):
            raise Level1RestoreError(
                f"compact site {base_id} has missing locked ortho50 bounds"
            )
        tile_bounds = {
            tile_ref: _ortho50_bounds(ortho50_by_tile[tile_ref])
            for tile_ref in tile_refs
        }
        xmin = min(bounds[0] for bounds in tile_bounds.values())
        ymin = min(bounds[1] for bounds in tile_bounds.values())
        xmax = max(bounds[2] for bounds in tile_bounds.values())
        ymax = max(bounds[3] for bounds in tile_bounds.values())
        if (xmax - xmin, ymax - ymin) != (4000, 4000):
            raise Level1RestoreError(
                f"compact site {base_id} does not form one 4 km by 4 km rectangle"
            )
        sites[base_id] = {
            "base_id": base_id,
            "tile_ref_origin": f"L93_{origin_x:04d}_{origin_y:04d}",
            "tile_count": len(tile_refs),
            "tile_refs": tile_refs,
            "extent_m": [xmin, ymin, xmax, ymax],
            "tile_bounds_epsg2154": tile_bounds,
        }
    return sites


def _absolute_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise Level1RestoreError(f"{label} is absent or unsafe: {path}")
    return path


def _site_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_absolute():
        raise Level1RestoreError("site root must be absolute")
    if root.exists() and root.is_symlink():
        raise Level1RestoreError("site root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _is_below(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Level1RestoreError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise Level1RestoreError(f"{label} must be a JSON object: {path}")
    return payload


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != sha256(source):
            raise Level1RestoreError(
                f"refusing to replace different site-input artifact: {destination}"
            )
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _locked_entries(lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = lock.get("entries")
    profile = _source_profile(lock)
    expected_entries = (
        2400 if profile == FULL_SOURCE_PROFILE else COMPACT_SITE_ENTRY_COUNT
    )
    if not isinstance(entries, list) or len(entries) != expected_entries:
        raise Level1RestoreError(
            f"{profile} source lock must contain exactly {expected_entries:,} source entries"
        )
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise Level1RestoreError("source lock contains a malformed entry")
        entry_id = str(entry.get("id", "")).strip()
        dataset = str(entry.get("dataset", "")).strip()
        url = str(entry.get("url", "")).strip()
        download = entry.get("download")
        if (
            not entry_id
            or entry_id in seen_ids
            or dataset not in zone_scenes.ALL_DATASETS
            or not url.startswith("https://")
            or not isinstance(download, dict)
            or not isinstance(download.get("sha256"), str)
            or len(str(download["sha256"])) != 64
        ):
            raise Level1RestoreError(f"source lock entry is incomplete: {entry_id!r}")
        seen_ids.add(entry_id)
        normalized.append(entry)
    counts: dict[str, int] = {}
    tiles_by_dataset: dict[str, set[str]] = {}
    for entry in normalized:
        dataset = str(entry["dataset"])
        tile_ref = str(entry.get("tile_ref", ""))
        _tile_coordinates(tile_ref)
        counts[dataset] = counts.get(dataset, 0) + 1
        tiles_by_dataset.setdefault(dataset, set()).add(tile_ref)
    if profile == FULL_SOURCE_PROFILE:
        if set(counts) != set(zone_scenes.ALL_DATASETS) or any(
            count != 400 for count in counts.values()
        ):
            raise Level1RestoreError(
                "full source lock must contain 400 entries for every source dataset"
            )
    else:
        if set(counts) != COMPACT_SITE_RASTER_DATASETS or any(
            count != COMPACT_SITE_TILE_COUNT for count in counts.values()
        ):
            raise Level1RestoreError(
                "compact LiDAR-raster source lock must contain 16 entries for every retained dataset"
            )
        tile_sets = list(tiles_by_dataset.values())
        if not tile_sets or any(tile_set != tile_sets[0] for tile_set in tile_sets[1:]):
            raise Level1RestoreError(
                "compact LiDAR-raster source datasets must cover the same 16 tiles"
            )
        acquisition = lock.get("acquisition")
        if not isinstance(acquisition, Mapping) or acquisition.get(
            "raw_point_cloud_policy"
        ) != "forbidden":
            raise Level1RestoreError(
                "compact LiDAR-raster source lock must forbid raw point-cloud LiDAR"
            )
        site = acquisition.get("selected_tile_site")
        if not isinstance(site, Mapping) or set(site.get("tile_refs", [])) != tile_sets[0]:
            raise Level1RestoreError(
                "compact LiDAR-raster source lock has an invalid selected-site contract"
            )
        if acquisition.get("terrain_height_resolution_m") != 1.0:
            raise Level1RestoreError(
                "compact LiDAR-raster source lock must set 1 m terrain resolution"
            )
    return normalized


def _compact_lidar_raster_lock(
    full_lock: Mapping[str, Any], *, full_lock_sha256: str, base_id: str
) -> dict[str, Any]:
    """Derive one compact 4 km × 4 km 1 m raster-source lock."""

    if _source_profile(full_lock) != FULL_SOURCE_PROFILE:
        raise Level1RestoreError("compact LiDAR-raster profile requires a full historical lock")
    full_entries = _locked_entries(full_lock)
    sites = _compact_site_records(full_entries)
    site = sites.get(base_id)
    if site is None:
        raise Level1RestoreError(f"unknown compact study site: {base_id}")
    selected_tiles = set(site["tile_refs"])
    compact_entries = [
        copy.deepcopy(entry)
        for entry in full_entries
        if str(entry["dataset"]) in COMPACT_SITE_RASTER_DATASETS
        and str(entry["tile_ref"]) in selected_tiles
    ]
    compact_entries.sort(key=lambda entry: (str(entry["tile_ref"]), str(entry["dataset"])))
    if len(compact_entries) != COMPACT_SITE_ENTRY_COUNT:
        raise Level1RestoreError("compact LiDAR-raster profile did not select 80 sources")
    result = copy.deepcopy(dict(full_lock))
    acquisition = result.get("acquisition")
    if not isinstance(acquisition, dict):
        raise Level1RestoreError("full source lock acquisition contract is malformed")
    acquisition.update(
        {
            "source_profile": COMPACT_LIDAR_RASTER_SOURCE_PROFILE,
            "raw_point_cloud_policy": "forbidden",
            "parent_full_source_lock_sha256": full_lock_sha256,
            "lod0_selection": "single_4km_x_4km_1m_lidar_raster_training_site",
            "lod0_tiles": sorted(selected_tiles),
            "selected_tile_site": site,
            "selected_tile_count": COMPACT_SITE_TILE_COUNT,
            "selected_source_entry_count": COMPACT_SITE_ENTRY_COUNT,
            "terrain_height_resolution_m": 1.0,
            "terrain_height_source": "MNT_MNS_MNH_lidar_derived_rasters",
        }
    )
    result["entries"] = compact_entries
    _locked_entries(result)
    return result


def _asset_requirements(actor_contract: Path) -> dict[str, Any]:
    contract = _read_json(actor_contract, label="actor deployment contract")
    library = contract.get("asset_library")
    scope = contract.get("asset_scope")
    if not isinstance(library, dict) or not isinstance(scope, dict) or len(library) != 5:
        raise Level1RestoreError("actor deployment contract must define exactly five assets")
    assets: list[dict[str, Any]] = []
    for selection_id, record in sorted(library.items()):
        if not isinstance(record, dict):
            raise Level1RestoreError(f"actor record is malformed: {selection_id}")
        wrapper_path = str(record.get("wrapper_path", "")).replace("\\", "/")
        wrapper_sha256 = str(record.get("wrapper_sha256", "")).lower()
        content_lock_sha256 = str(record.get("content_lock_sha256", "")).lower()
        if (
            not wrapper_path
            or wrapper_path.startswith("/")
            or ".." in Path(wrapper_path).parts
            or len(wrapper_sha256) != 64
            or len(content_lock_sha256) != 64
        ):
            raise Level1RestoreError(f"actor wrapper lock is invalid: {selection_id}")
        assets.append(
            {
                "selection_id": str(selection_id),
                "source_name": str(record.get("source_name", "")),
                "operational_role": str(record.get("operational_role", "")),
                "license": str(record.get("license", "")),
                "wrapper_path": wrapper_path,
                "wrapper_sha256": wrapper_sha256,
                "content_lock_sha256": content_lock_sha256,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "ASSET_FREEZE_REQUIREMENTS_RECORDED",
        "source_inventory_path": str(scope.get("source_inventory_path", "")),
        "source_inventory_sha256": str(scope.get("source_inventory_sha256", "")),
        "group_id": str(scope.get("group_id", "")),
        "assets": assets,
        "materialization_policy": "exact_locked_wrappers_only_no_substitution",
    }


def _status(
    *,
    lock: Mapping[str, Any],
    raw_root: Path,
    asset_requirements: Mapping[str, Any],
    state: str,
) -> dict[str, Any]:
    entries = _locked_entries(lock)
    profile = _source_profile(lock)
    completed = 0
    completed_bytes = 0
    expected_bytes = 0
    for entry in entries:
        destination = zone_scenes._destination(raw_root, entry)
        download = entry["download"]
        expected = int(download.get("bytes", 0))
        expected_bytes += expected
        if destination.is_file() and not destination.is_symlink():
            if destination.stat().st_size == expected and sha256(destination) == download["sha256"]:
                completed += 1
                completed_bytes += expected
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "source_profile": profile,
        "raw_point_cloud_lidar": (
            "excluded"
            if profile == COMPACT_LIDAR_RASTER_SOURCE_PROFILE
            else "required"
        ),
        "source_entries": len(entries),
        "source_entries_complete": completed,
        "source_bytes_expected": expected_bytes,
        "source_bytes_complete": completed_bytes,
        "source_lock_sha256": sha256(raw_root.parent / "inputs" / SOURCE_LOCK_NAME),
        "asset_freeze": {
            "state": "PENDING_EXACT_WRAPPERS",
            "required_wrapper_count": len(asset_requirements["assets"]),
            "substitution_allowed": False,
        },
    }


def activate_compact_lidar_raster_profile(
    *, site_root: Path, base_id: str
) -> dict[str, Any]:
    """Activate one compact 1 m raster site without raw point clouds.

    The original full source lock is retained beside the active lock for audit.
    Only the active lock is changed; raw raster files are not copied or removed.
    """

    site_root = _site_root(site_root)
    inputs = site_root / "inputs"
    active_lock_path = inputs / SOURCE_LOCK_NAME
    requirements_path = inputs / ASSET_REQUIREMENTS_NAME
    active_lock = _read_json(active_lock_path, label="prepared source lock")
    requirements = _read_json(requirements_path, label="asset freeze requirements")
    if not isinstance(requirements.get("assets"), list) or len(requirements["assets"]) != 5:
        raise Level1RestoreError("prepared asset freeze requirements are invalid")
    full_lock_path = inputs / FULL_SOURCE_LOCK_NAME
    if full_lock_path.exists():
        full_lock = _read_json(full_lock_path, label="historical full source lock")
        if _source_profile(full_lock) != FULL_SOURCE_PROFILE:
            raise Level1RestoreError("historical source lock is not a full source lock")
        if _source_profile(active_lock) == FULL_SOURCE_PROFILE and active_lock != full_lock:
            raise Level1RestoreError("active full source lock differs from historical lock")
    else:
        if _source_profile(active_lock) != FULL_SOURCE_PROFILE:
            raise Level1RestoreError(
                "cannot activate compact profile without the historical full source lock"
            )
        _atomic_copy(active_lock_path, full_lock_path)
        full_lock = _read_json(full_lock_path, label="copied historical full source lock")
    compact_lock = _compact_lidar_raster_lock(
        full_lock,
        full_lock_sha256=sha256(full_lock_path),
        base_id=base_id,
    )
    write_json(active_lock_path, compact_lock)
    raw_root = site_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    result = _status(
        lock=compact_lock,
        raw_root=raw_root,
        asset_requirements=requirements,
        state="COMPACT_LIDAR_RASTER_SOURCES_READY_PENDING_ASSET_FREEZE",
    )
    if result["source_entries_complete"] != COMPACT_SITE_ENTRY_COUNT:
        raise Level1RestoreError(
            "compact LiDAR-raster profile has missing or stale selected sources"
        )
    write_json(site_root / STATUS_NAME, result)
    return result


def prepare_site_input(
    *,
    source_lock_path: Path,
    source_resolution_path: Path,
    actor_contract_path: Path,
    site_root: Path,
) -> dict[str, Any]:
    """Create an isolated, resumable Level-1 site input without downloading."""

    source_lock_path = _absolute_file(source_lock_path, label="source lock")
    source_resolution_path = _absolute_file(
        source_resolution_path, label="source resolution"
    )
    actor_contract_path = _absolute_file(
        actor_contract_path, label="actor deployment contract"
    )
    site_root = _site_root(site_root)
    lock = _read_json(source_lock_path, label="source lock")
    _locked_entries(lock)
    requirements = _asset_requirements(actor_contract_path)
    inputs = site_root / "inputs"
    _atomic_copy(source_lock_path, inputs / SOURCE_LOCK_NAME)
    _atomic_copy(source_resolution_path, inputs / "source-resolution.csv")
    _atomic_copy(actor_contract_path, inputs / "actor-deployments.json")
    requirements_path = inputs / ASSET_REQUIREMENTS_NAME
    if requirements_path.exists():
        if _read_json(requirements_path, label="asset freeze requirements") != requirements:
            raise Level1RestoreError("asset freeze requirements drifted")
    else:
        write_json(requirements_path, requirements)
    raw_root = site_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    result = _status(
        lock=_read_json(inputs / SOURCE_LOCK_NAME, label="copied source lock"),
        raw_root=raw_root,
        asset_requirements=requirements,
        state="LEVEL1_INPUT_PREPARED",
    )
    write_json(site_root / STATUS_NAME, result)
    return result


def _restore_vector_sources(
    *,
    raw_root: Path,
    lock: Mapping[str, Any],
    timeout: float,
) -> None:
    raw_vectors = lock.get("vector_sources")
    if not isinstance(raw_vectors, dict):
        raise Level1RestoreError("source lock has no vector source locks")
    for records in raw_vectors.values():
        if not isinstance(records, list):
            raise Level1RestoreError("vector source records are malformed")
        for record in records:
            if not isinstance(record, dict):
                raise Level1RestoreError("vector source record is malformed")
            download = record.get("download")
            urls = record.get("urls")
            if not isinstance(download, dict) or not isinstance(urls, list) or not urls:
                raise Level1RestoreError("vector source lock has no request pages")
            relative = str(download.get("relpath", "")).replace("\\", "/")
            destination = (raw_root / relative).resolve()
            if not _is_below(raw_root, destination):
                raise Level1RestoreError("vector source destination escapes site input")
            expected_sha256 = str(download.get("sha256", "")).lower()
            if destination.is_file():
                if sha256(destination) != expected_sha256:
                    raise Level1RestoreError(f"vector source digest differs: {relative}")
                continue
            pages_root = destination.parent / f".{destination.stem}.restore-pages"
            features: list[object] = []
            for index, url in enumerate(urls):
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise Level1RestoreError("vector source request escaped HTTPS")
                page = zone_scenes._download_vector_page(
                    url=url,
                    destination=pages_root / f"{index:08d}.geojson",
                    timeout=timeout,
                )
                page_features = page.get("features")
                if not isinstance(page_features, list):
                    raise Level1RestoreError("vector page has no feature list")
                features.extend(page_features)
            payload = {
                "type": "FeatureCollection",
                "name": str(record.get("layer", "")),
                "crs": {"type": "name", "properties": {"name": "EPSG:2154"}},
                "features": features,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.partial"
            )
            try:
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
                actual_sha256 = sha256(temporary)
                if actual_sha256 != expected_sha256:
                    raise Level1RestoreError(
                        f"vector source changed from its locked digest: {relative}"
                    )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def restore_sources(
    *,
    site_root: Path,
    timeout: float,
    direct_workers: int,
    raster_workers: int,
    retries: int,
) -> dict[str, Any]:
    """Resume verified source restoration into a prepared Level-1 input."""

    site_root = _site_root(site_root)
    inputs = site_root / "inputs"
    lock_path = inputs / SOURCE_LOCK_NAME
    requirements_path = inputs / ASSET_REQUIREMENTS_NAME
    lock = _read_json(lock_path, label="prepared source lock")
    entries = _locked_entries(lock)
    requirements = _read_json(requirements_path, label="asset freeze requirements")
    if not isinstance(requirements.get("assets"), list) or len(requirements["assets"]) != 5:
        raise Level1RestoreError("prepared asset freeze requirements are invalid")
    if timeout <= 0 or not 1 <= direct_workers <= 16 or not 1 <= raster_workers <= 16:
        raise Level1RestoreError("invalid bounded restore transfer settings")
    if not 1 <= retries <= 10:
        raise Level1RestoreError("restore retries must be between 1 and 10")
    raw_root = site_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    if raw_root.is_symlink():
        raise Level1RestoreError("raw source root may not be a symlink")
    # Historical locks recorded the accepted target filename only under the
    # nested download receipt.  Make that destination explicit in the copied
    # input lock before handing it to the current resumable downloader.  This
    # is path normalization, not a source/provenance change: URL, expected
    # byte count and SHA-256 all remain untouched.
    for entry in entries:
        download = entry["download"]
        relative = str(download.get("relpath", "")).replace("\\", "/")
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or len(candidate.parts) != 1
        ):
            raise Level1RestoreError(
                f"historical source target is unsafe: {entry['id']}"
            )
        entry["relative_path"] = candidate.name
    zone_scenes._assert_unique_destinations(entries, raw_root=raw_root)
    checkpoint_count = 0

    def checkpoint() -> None:
        nonlocal checkpoint_count
        checkpoint_count += 1
        if checkpoint_count % 25 == 0:
            write_json(lock_path, copy.deepcopy(lock))

    pacer = zone_scenes._RequestStartPacer(
        zone_scenes.DIRECT_REQUEST_START_INTERVAL_SECONDS
    )
    direct = [entry for entry in entries if entry["dataset"] in RAW_SOURCE_DATASETS]
    raster = [entry for entry in entries if entry["dataset"] not in RAW_SOURCE_DATASETS]
    zone_scenes._download_direct_entries(
        direct,
        raw_root=raw_root,
        timeout=timeout,
        max_workers=direct_workers,
        retries=retries,
        checkpoint=checkpoint,
        request_pacer=pacer,
    )
    zone_scenes._download_raster_entries(
        raster,
        raw_root=raw_root,
        timeout=timeout,
        max_workers=raster_workers,
        retries=retries,
        checkpoint=checkpoint,
        request_pacer=pacer,
    )
    _restore_vector_sources(raw_root=raw_root, lock=lock, timeout=timeout)
    write_json(lock_path, lock)
    profile = _source_profile(lock)
    result = _status(
        lock=lock,
        raw_root=raw_root,
        asset_requirements=requirements,
        state=(
            "COMPACT_LIDAR_RASTER_SOURCES_READY_PENDING_ASSET_FREEZE"
            if profile == COMPACT_LIDAR_RASTER_SOURCE_PROFILE
            else "SOURCES_REHYDRATED_PENDING_ASSET_FREEZE"
        ),
    )
    if result["source_entries_complete"] != result["source_entries"]:
        raise Level1RestoreError("source restoration did not produce every locked entry")
    write_json(site_root / STATUS_NAME, result)
    return result


def verify_asset_freeze(*, site_root: Path, asset_root: Path) -> dict[str, Any]:
    """Verify exact actor wrapper recovery without accepting substitutes."""

    site_root = _site_root(site_root)
    asset_root = Path(asset_root).expanduser().resolve()
    if not asset_root.is_dir() or asset_root.is_symlink():
        raise Level1RestoreError("asset root is absent or unsafe")
    requirements = _read_json(
        site_root / "inputs" / ASSET_REQUIREMENTS_NAME,
        label="asset freeze requirements",
    )
    assets = requirements.get("assets")
    if not isinstance(assets, list):
        raise Level1RestoreError("asset freeze requirements are malformed")
    verified: list[dict[str, str]] = []
    missing: list[str] = []
    for asset in assets:
        if not isinstance(asset, dict):
            raise Level1RestoreError("asset freeze requirement is malformed")
        path = (asset_root / str(asset["wrapper_path"])).resolve()
        if not _is_below(asset_root, path) or not path.is_file() or path.is_symlink():
            missing.append(str(asset["selection_id"]))
            continue
        actual = sha256(path)
        if actual != asset["wrapper_sha256"]:
            raise Level1RestoreError(
                f"actor wrapper digest differs: {asset['selection_id']}"
            )
        verified.append(
            {
                "selection_id": str(asset["selection_id"]),
                "wrapper_path": str(asset["wrapper_path"]),
                "sha256": actual,
            }
        )
    if missing:
        raise Level1RestoreError(
            "exact actor wrappers are still missing: " + ", ".join(sorted(missing))
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "state": "ASSET_FREEZE_MATERIALIZED",
        "asset_root": str(asset_root),
        "wrappers": verified,
    }
    write_json(site_root / "assets" / "asset-freeze-receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore one exact Level-1 FireViewer site input"
    )
    parser.add_argument("--site-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-lock", required=True, type=Path)
    prepare.add_argument("--source-resolution", required=True, type=Path)
    prepare.add_argument("--actor-contract", required=True, type=Path)
    restore = subparsers.add_parser("restore-sources")
    restore.add_argument("--timeout", type=float, default=180.0)
    restore.add_argument("--direct-workers", type=int, default=8)
    restore.add_argument("--raster-workers", type=int, default=8)
    restore.add_argument("--retries", type=int, default=5)
    asset = subparsers.add_parser("verify-asset-freeze")
    asset.add_argument("--asset-root", required=True, type=Path)
    compact = subparsers.add_parser("activate-compact-lidar-raster-profile")
    compact.add_argument("--base-id", choices=[item[0] for item in COMPACT_STUDY_SITES], default="Z16-base-01")
    subparsers.add_parser("status")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        result = prepare_site_input(
            source_lock_path=args.source_lock,
            source_resolution_path=args.source_resolution,
            actor_contract_path=args.actor_contract,
            site_root=args.site_root,
        )
    elif args.command == "restore-sources":
        result = restore_sources(
            site_root=args.site_root,
            timeout=args.timeout,
            direct_workers=args.direct_workers,
            raster_workers=args.raster_workers,
            retries=args.retries,
        )
    elif args.command == "verify-asset-freeze":
        result = verify_asset_freeze(site_root=args.site_root, asset_root=args.asset_root)
    elif args.command == "activate-compact-lidar-raster-profile":
        result = activate_compact_lidar_raster_profile(
            site_root=args.site_root,
            base_id=args.base_id,
        )
    else:
        status_path = _site_root(args.site_root) / STATUS_NAME
        result = _read_json(status_path, label="restore status")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Level1RestoreError as exc:
        print(f"LEVEL1_RESTORE_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
