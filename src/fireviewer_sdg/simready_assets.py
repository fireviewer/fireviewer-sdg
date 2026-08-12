"""Discover and lock official NVIDIA USD assets without inventing semantics.

The discovery phase is intentionally conservative:

* only the pinned public Isaac Sim asset host is crawled automatically;
* every selected remote layer is referenced through a local, hashable USDA wrapper;
* generic vehicles are never promoted to French response-vehicle classes;
* missing exact response assets are reported and keep production blocked.

Owned or separately licensed community assets remain supported through the manual
manifest override handled by :mod:`fireviewer_sdg.ign_catalog`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from fireviewer_sdg.preparation_progress import write_progress


DEFAULT_NVIDIA_ASSET_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0"
)
OFFICIAL_NVIDIA_ASSET_HOSTS = frozenset(
    {"omniverse-content-production.s3-us-west-2.amazonaws.com"}
)
DISCOVERY_SUBROOTS = (
    "Isaac/Environments/Outdoor",
    "Isaac/Environments/Terrains",
    "Isaac/Props",
    "Isaac/SimReady",
)
OFFICIAL_INDEX_PATHS = (
    "Isaac/SimReady/manifest.csv",
    "Isaac/Environments/Outdoor/Rivermark/dsready_content/file_list.txt",
)
RIVERMARK_CONTENT_PATH = (
    "Isaac/Environments/Outdoor/Rivermark/dsready_content"
)
PREFERRED_VEGETATION_SUFFIXES = (
    "/assets/vegetation/trees/norway_spruce.usd",
    "/assets/vegetation/trees/douglas_fir.usd",
    "/assets/vegetation/trees/lombardy_poplar.usd",
    "/assets/vegetation/trees/common_apple.usd",
    "/assets/vegetation/trees/hawthorn.usd",
    "/assets/vegetation/shrub/juniper.usd",
)
PREFERRED_RURAL_BUILDING_SUFFIXES = (
    "/props_structures/jb_gardens_01/jb_gardens_01.usd",
    "/props_structures/kasa_house_01/kasa_house_01.usd",
    "/props_structures/kasa_house_02/kasa_house_02.usd",
    "/props_structures/kasa_house_03/kasa_house_03.usd",
    "/props_structures/wyatt_house/wyatt_house.usd",
)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MAX_DISCOVERY_DEPTH = 7
MAX_DISCOVERY_DIRECTORIES = 5_000
MAX_DISCOVERY_ASSETS = 30_000
MANIFEST_SCHEMA_VERSION = 3
MANIFEST_PROFILE = "fireviewer_simready_photoreal_hd_v3"
PHOTOREAL_FAMILY_MINIMUMS: dict[str, dict[str, int]] = {
    "vegetation": {
        "trees": 8,
        "shrubs": 4,
        "understory": 4,
    },
    "buildings": {
        "habitat": 4,
        "agricultural": 2,
        "industrial": 2,
        "annex": 2,
    },
}
PHOTOREAL_MIN_LOD_LEVELS: dict[str, int] = {
    "vegetation.trees": 3,
    "vegetation.shrubs": 3,
    "vegetation.understory": 2,
    "buildings.habitat": 2,
    "buildings.agricultural": 2,
    "buildings.industrial": 2,
    "buildings.annex": 2,
}
PHOTOREAL_LIBRARY_POLICY = {
    "primitive_fallbacks": "forbidden",
    "procedural_asset_fallbacks": "forbidden",
    "non_uniform_asset_scaling": "forbidden",
    "prototype_selection": "family_weighted_without_single_asset_dominance",
    "maximum_single_prototype_share": 0.25,
}
NVIDIA_ASSET_LICENSE_ID = (
    "LicenseRef-NVIDIA-Isaac-Sim-Additional-Software-and-Materials"
)
NVIDIA_ASSET_LICENSE_URI = (
    "https://docs.isaacsim.omniverse.nvidia.com/latest/common/"
    "license-isaac-sim-additional.html"
)
NVIDIA_OMNIVERSE_LICENSE_ID = "LicenseRef-NVIDIA-Omniverse-Content"
NVIDIA_OMNIVERSE_LICENSE_URI = (
    "https://docs.omniverse.nvidia.com/connect/latest/common/"
    "NVIDIA_Omniverse_License_Agreement.html"
)
NVIDIA_BUCKET_ORIGIN = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
)
NVIDIA_VEGETATION_PREFIX = "Assets/Vegetation/"
_SKIP_PATH_PARTS = frozenset(
    {
        ".thumbs",
        "demo",
        "demos",
        "example",
        "examples",
        "placeholder",
        "preview",
        "previews",
        "sample",
        "samples",
        "test",
        "tests",
        "thumbnail",
        "thumbnails",
    }
)
_VEGETATION_TERMS = frozenset(
    {
        "birch",
        "bush",
        "fir",
        "forest",
        "oak",
        "pine",
        "plant",
        "shrub",
        "spruce",
        "tree",
        "vegetation",
    }
)
_VEGETATION_FAMILY_TERMS: dict[str, frozenset[str]] = {
    "trees": frozenset(
        {
            "apple",
            "birch",
            "cedar",
            "cypress",
            "fir",
            "oak",
            "pine",
            "poplar",
            "spruce",
            "tree",
        }
    ),
    "shrubs": frozenset(
        {
            "bush",
            "hawthorn",
            "hedge",
            "juniper",
            "scrub",
            "shrub",
        }
    ),
    "understory": frozenset(
        {
            "fern",
            "flower",
            "grass",
            "groundcover",
            "herb",
            "plant",
            "reed",
            "understory",
            "weed",
        }
    ),
}
_INDOOR_VEGETATION_TERMS = frozenset(
    {"bonsai", "indoor", "office", "potted", "warehouse"}
)
_BUILDING_FAMILY_TERMS: dict[str, frozenset[str]] = {
    "habitat": frozenset(
        {
            "apartment",
            "cabin",
            "cottage",
            "farmhouse",
            "house",
            "residence",
            "residential",
            "villa",
        }
    ),
    "agricultural": frozenset(
        {
            "agricultural",
            "barn",
            "farm",
            "farmhouse",
            "granary",
            "silo",
            "stable",
        }
    ),
    "industrial": frozenset(
        {
            "factory",
            "hangar",
            "industrial",
            "plant",
            "warehouse",
            "workshop",
        }
    ),
    "annex": frozenset(
        {
            "annex",
            "carport",
            "garage",
            "outbuilding",
            "parking",
            "shed",
        }
    ),
}
_ACTOR_MATCHERS: dict[str, Callable[[str, set[str]], bool]] = {
    "sdis_vehicle": lambda normalized, tokens: (
        "sdis" in tokens
        or "camion_citerne_feux" in normalized
        or "camion_citerne_foret" in normalized
        or (
            "ccf" in tokens
            and any(
                context in normalized
                for context in ("fire", "feu", "forest", "foret", "pompier")
            )
        )
    ),
    "canadair": lambda normalized, tokens: (
        "canadair" in tokens
        or "cl415" in tokens
        or "cl_415" in normalized
    ),
    "dash": lambda normalized, tokens: (
        any(
            name in normalized
            for name in ("dash_8", "dash8", "q400")
        )
        and any(
            context in normalized
            for context in (
                "air_tanker",
                "airtanker",
                "firefight",
                "securite_civile",
                "water_bomber",
            )
        )
    ),
    "securite_civile_helicopter": lambda normalized, tokens: (
        "securite_civile" in normalized
        and any(
            name in normalized
            for name in ("dragon", "ec145", "h145", "helicopter", "helicoptere")
        )
    ),
    "hard_negative_construction_truck": lambda normalized, tokens: (
        any(term in normalized for term in ("construction", "dump_truck", "dumptruck"))
        and any(term in normalized for term in ("truck", "camion"))
    ),
    "hard_negative_crop_duster": lambda normalized, tokens: (
        "crop_duster" in normalized
        or (
            any(term in normalized for term in ("agricultural", "agriculture"))
            and any(term in normalized for term in ("aircraft", "plane", "avion"))
        )
    ),
    "hard_negative_utility_helicopter": lambda normalized, tokens: (
        "utility" in tokens
        and any(term in normalized for term in ("helicopter", "helicoptere", "rotorcraft"))
    ),
}


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _normalize_uri(uri: str) -> tuple[str, set[str]]:
    decoded = urllib.parse.unquote(uri).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", decoded).strip("_")
    return normalized, {token for token in normalized.split("_") if token}


def _validate_official_root(asset_root: str) -> str:
    root = asset_root.strip().rstrip("/")
    parsed = urllib.parse.urlparse(root)
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_NVIDIA_ASSET_HOSTS:
        raise ValueError(
            "automatic asset discovery is restricted to the official NVIDIA "
            "Isaac Sim asset host; use a reviewed manual manifest for other sources"
        )
    if not parsed.path.rstrip("/").endswith("/Assets/Isaac/6.0"):
        raise ValueError("automatic asset discovery requires the pinned Isaac 6.0 root")
    return root


def _omni_client_list(uri: str) -> list[dict[str, Any]]:
    try:
        import omni.client
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "automatic NVIDIA asset discovery requires omni.client from Isaac Sim"
        ) from exc
    omni.client.bypass_list_cache(True)
    result, entries = omni.client.list(uri)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"omni.client.list failed for {uri}: {result}")
    folder_flag = int(omni.client.ItemFlags.CAN_HAVE_CHILDREN)
    return [
        {
            "relative_path": str(entry.relative_path),
            "is_folder": bool(int(entry.flags) & folder_flag),
            "provider_hash": str(entry.hash or ""),
            "provider_version": str(entry.version or ""),
            "size_bytes": int(entry.size or 0),
        }
        for entry in entries
    ]


def _omni_client_read_bytes(uri: str) -> bytes:
    try:
        import omni.client
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "official NVIDIA index access requires omni.client from Isaac Sim"
        ) from exc
    result, _version, content = omni.client.read_file(uri)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"omni.client.read_file failed for {uri}: {result}")
    return memoryview(content).tobytes()


def _http_read_bytes(uri: str) -> bytes:
    parsed = urllib.parse.urlparse(uri)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in OFFICIAL_NVIDIA_ASSET_HOSTS
    ):
        raise ValueError("NVIDIA asset download escaped the official content host")
    last_error: OSError | None = None
    for attempt in range(4):
        request = urllib.request.Request(
            uri,
            headers={"User-Agent": "FireViewer-SDG-Asset-Lock/2.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep((2, 5, 15)[attempt])
    raise RuntimeError(f"NVIDIA asset download failed: {uri}: {last_error}")


def _http_download_file(uri: str, destination: Path) -> None:
    """Stream and resume a large official NVIDIA object into an atomic cache file."""

    parsed = urllib.parse.urlparse(uri)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in OFFICIAL_NVIDIA_ASSET_HOSTS
    ):
        raise ValueError("NVIDIA asset download escaped the official content host")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.download")
    last_error: OSError | None = None
    for attempt in range(4):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "FireViewer-SDG-Asset-Lock/2.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(uri, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = int(getattr(response, "status", 200))
                resumed = offset > 0 and status == 206
                if not resumed:
                    offset = 0
                content_length = int(response.headers.get("Content-Length", "0") or 0)
                expected_size = offset + content_length if content_length else 0
                with partial.open("ab" if resumed else "wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                actual_size = partial.stat().st_size
                if expected_size and actual_size != expected_size:
                    raise OSError(
                        f"incomplete NVIDIA object: {actual_size}/{expected_size} bytes"
                    )
            os.replace(partial, destination)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 416 and partial.is_file():
                partial.unlink()
            if attempt == 3:
                break
            time.sleep((2, 5, 15)[attempt])
        except OSError as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep((2, 5, 15)[attempt])
    raise RuntimeError(f"NVIDIA asset download failed: {uri}: {last_error}")


def cache_official_vegetation_inventory(
    *,
    volume_root: Path,
    opener: Callable[[str], bytes] = _http_read_bytes,
) -> dict[str, Any]:
    """Cache the official bucket inventory for NVIDIA's vegetation mount."""

    volume = volume_root.resolve()
    cache_root = volume / "input" / "nvidia-index-cache"
    destination = cache_root / "vegetation-s3-inventory.json"
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and payload.get("prefix") == NVIDIA_VEGETATION_PREFIX
            and isinstance(payload.get("objects"), list)
        ):
            return payload

    objects: list[dict[str, Any]] = []
    continuation = ""
    page = 0
    namespace = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        page += 1
        query = {
            "list-type": "2",
            "prefix": NVIDIA_VEGETATION_PREFIX,
            "max-keys": "1000",
        }
        if continuation:
            query["continuation-token"] = continuation
        uri = f"{NVIDIA_BUCKET_ORIGIN}/?{urllib.parse.urlencode(query)}"
        write_progress(
            volume,
            phase="official_nvidia_vegetation_inventory",
            message=f"Inventaire NVIDIA Vegetation : page {page}.",
            inventory_pages_completed=page - 1,
            vegetation_objects=len(objects),
        )
        document = ET.fromstring(opener(uri))
        for content in document.findall(f"{namespace}Contents"):
            key = str(content.findtext(f"{namespace}Key") or "").strip()
            if not key:
                continue
            objects.append(
                {
                    "key": key,
                    "etag": str(
                        content.findtext(f"{namespace}ETag") or ""
                    ).strip('"'),
                    "last_modified": str(
                        content.findtext(f"{namespace}LastModified") or ""
                    ),
                    "size_bytes": int(
                        content.findtext(f"{namespace}Size") or 0
                    ),
                }
            )
        continuation = str(
            document.findtext(f"{namespace}NextContinuationToken") or ""
        )
        if not continuation:
            break
        if page >= 10:
            raise RuntimeError("NVIDIA vegetation inventory exceeded 10 pages")

    payload = {
        "schema_version": 1,
        "source_uri": f"{NVIDIA_BUCKET_ORIGIN}/?list-type=2",
        "prefix": NVIDIA_VEGETATION_PREFIX,
        "objects": sorted(objects, key=lambda entry: str(entry["key"])),
    }
    _atomic_write_json(destination, payload)
    write_progress(
        volume,
        phase="official_nvidia_vegetation_inventory",
        state="completed",
        message=(
            f"Inventaire NVIDIA Vegetation verrouillé : {len(objects)} objets."
        ),
        inventory_pages_completed=page,
        vegetation_objects=len(objects),
        vegetation_inventory_sha256=_sha256(destination),
    )
    return payload


def cache_official_nvidia_indexes(
    *,
    volume_root: Path,
    asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    reader: Callable[[str], bytes] = _http_read_bytes,
) -> dict[str, Any]:
    """Cache the two bounded official indexes used for environment discovery."""

    volume = volume_root.resolve()
    root = _validate_official_root(asset_root)
    cache_root = volume / "input" / "nvidia-index-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for index, relative in enumerate(OFFICIAL_INDEX_PATHS, start=1):
        write_progress(
            volume,
            phase="official_nvidia_index_download",
            message=f"Lecture de l'index NVIDIA {index}/{len(OFFICIAL_INDEX_PATHS)}.",
            indexes_completed=index - 1,
            indexes_total=len(OFFICIAL_INDEX_PATHS),
            current_index=relative,
        )
        uri = f"{root}/{relative}"
        content = reader(uri)
        destination = cache_root / Path(relative).name
        _atomic_write_text(
            destination,
            content.decode("utf-8-sig"),
        )
        entries.append(
            {
                "source_uri": uri,
                "path": os.path.relpath(destination, volume).replace("\\", "/"),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
    report = {
        "schema_version": 1,
        "asset_root": root,
        "indexes": entries,
    }
    report_path = cache_root / "index-lock.json"
    _atomic_write_json(report_path, report)
    write_progress(
        volume,
        phase="official_nvidia_index_download",
        state="completed",
        message="Index officiels NVIDIA téléchargés et hashés.",
        indexes_completed=len(entries),
        indexes_total=len(OFFICIAL_INDEX_PATHS),
        index_lock=os.path.relpath(report_path, volume).replace("\\", "/"),
    )
    return {
        "index_lock": report_path,
        "indexes": entries,
    }


def _verified_index_paths(
    *,
    volume_root: Path,
    asset_root: str,
) -> tuple[Path, Path, dict[str, Any]]:
    volume = volume_root.resolve()
    lock_path = volume / "input" / "nvidia-index-cache" / "index-lock.json"
    if not lock_path.is_file():
        cache_official_nvidia_indexes(
            volume_root=volume,
            asset_root=asset_root,
        )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("asset_root") != _validate_official_root(asset_root)
    ):
        raise RuntimeError("official NVIDIA index lock is invalid or belongs to another root")
    locked: dict[str, Path] = {}
    for entry in payload.get("indexes", []):
        if not isinstance(entry, dict):
            raise RuntimeError("official NVIDIA index lock contains an invalid entry")
        relative = str(entry.get("path", "")).strip()
        candidate = (volume / relative).resolve()
        if volume not in candidate.parents or not candidate.is_file():
            raise RuntimeError(f"locked NVIDIA index is absent: {relative}")
        expected = str(entry.get("sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or _sha256(candidate) != expected:
            raise RuntimeError(f"locked NVIDIA index hash mismatch: {relative}")
        locked[candidate.name] = candidate
    try:
        return locked["manifest.csv"], locked["file_list.txt"], payload
    except KeyError as exc:
        raise RuntimeError("official NVIDIA index lock is incomplete") from exc


def _is_canonical_indexed_usd(relative: str) -> bool:
    normalized = relative.strip().replace("\\", "/")
    if Path(normalized).suffix.lower() not in USD_SUFFIXES:
        return False
    lowered = normalized.lower()
    if any(f"/{part}/" in lowered for part in _SKIP_PATH_PARTS):
        return False
    stem = Path(lowered).stem
    if (
        stem.endswith(("_base", "_inst", "_inst_base", "_tagged", "_unmatched"))
        or "/nv_core/" in lowered
    ):
        return False
    return True


def discover_official_nvidia_assets_from_indexes(
    *,
    volume_root: Path,
    asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    vegetation_inventory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic inventory from the two official bounded indexes."""

    root = _validate_official_root(asset_root)
    manifest_path, rivermark_path, lock = _verified_index_paths(
        volume_root=volume_root,
        asset_root=root,
    )
    lock_identity = hashlib.sha256(
        json.dumps(lock, sort_keys=True).encode("utf-8")
    ).hexdigest()
    candidates: dict[str, dict[str, Any]] = {}

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relative = str(row.get("_Path", "")).strip().replace("\\", "/")
            if not _is_canonical_indexed_usd(relative):
                continue
            uri = f"{root}/Isaac/{urllib.parse.quote(relative, safe='/%:@+-._~')}"
            candidates[uri] = {
                "uri": uri,
                "provider_hash": "",
                "provider_version": f"Isaac-6.0-index-{lock_identity[:16]}",
                "size_bytes": 0,
                "source_index": "Isaac/SimReady/manifest.csv",
            }

    for raw_line in rivermark_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        relative = raw_line.strip().replace("\\", "/")
        if not _is_canonical_indexed_usd(relative):
            continue
        uri = (
            f"{root}/{RIVERMARK_CONTENT_PATH}"
            f"{urllib.parse.quote(relative, safe='/%:@+-._~')}"
        )
        candidates[uri] = {
            "uri": uri,
            "provider_hash": "",
            "provider_version": f"Isaac-6.0-index-{lock_identity[:16]}",
            "size_bytes": 0,
            "source_index": (
                "Isaac/Environments/Outdoor/Rivermark/"
                "dsready_content/file_list.txt"
            ),
        }

    if vegetation_inventory is None:
        vegetation_inventory = cache_official_vegetation_inventory(
            volume_root=volume_root
        )
    for item in vegetation_inventory["objects"]:
        key = str(item["key"])
        if not _is_canonical_indexed_usd(key):
            continue
        uri = (
            f"{NVIDIA_BUCKET_ORIGIN}/"
            f"{urllib.parse.quote(key, safe='/%:@+-._~')}"
        )
        candidates[uri] = {
            "uri": uri,
            "provider_hash": str(item.get("etag", "")),
            "provider_version": str(item.get("last_modified", "")),
            "size_bytes": int(item.get("size_bytes", 0)),
            "source_index": "Assets/Vegetation S3 inventory",
            "bucket_key": key,
            "license_id": NVIDIA_OMNIVERSE_LICENSE_ID,
            "license_uri": NVIDIA_OMNIVERSE_LICENSE_URI,
        }

    inventory = [candidates[uri] for uri in sorted(candidates)]
    write_progress(
        volume_root,
        phase="official_nvidia_index_selection",
        message=(
            f"{len(inventory)} couches USD canoniques lues depuis les index "
            "NVIDIA; sélection rurale en cours."
        ),
        candidates_indexed=len(inventory),
        indexes_completed=2,
        indexes_total=2,
    )
    return inventory


def _rivermark_relative_path(uri: str, *, asset_root: str) -> str:
    prefix = f"{_validate_official_root(asset_root)}/{RIVERMARK_CONTENT_PATH}"
    decoded = urllib.parse.unquote(uri)
    if not decoded.startswith(f"{prefix}/"):
        raise ValueError(f"asset is outside the indexed Rivermark content root: {uri}")
    relative = decoded[len(prefix) :].lstrip("/")
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"asset has an unsafe indexed relative path: {uri}")
    return "/".join(parts)


def _metadata_validation_sha256(metadata: dict[str, Any]) -> str:
    validated = {
        key: metadata[key]
        for key in (
            "native_dimensions_m",
            "ground_anchor_m",
            "anchor_validation",
            "lod",
            "materials",
            "placement",
        )
    }
    return hashlib.sha256(
        json.dumps(
            validated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _inspect_local_usd_metadata(path: Path) -> dict[str, Any]:
    try:
        import isaacsim  # noqa: F401 - exposes the bundled pxr modules
        from pxr import Usd, UsdGeom, UsdShade, UsdUtils
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "local NVIDIA USD inspection requires the pinned Isaac runtime"
        ) from exc
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"materialized NVIDIA USD could not be opened: {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"materialized NVIDIA USD has no default prim: {path}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if not 0.000001 <= meters_per_unit <= 1000.0:
        raise RuntimeError(
            f"materialized NVIDIA USD has invalid metersPerUnit: {meters_per_unit}"
        )
    if up_axis not in {"Y", "Z"}:
        raise RuntimeError(f"materialized NVIDIA USD has invalid upAxis: {up_axis}")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bounds = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    root_values = [
        *(float(minimum[index]) for index in range(3)),
        *(float(maximum[index]) for index in range(3)),
    ]
    if bounds.IsEmpty() or any(
        not math.isfinite(value) or abs(value) >= 1.0e30
        for value in root_values
    ):
        # Some official vegetation layers author a parent Mesh with an empty
        # sentinel extent and place the actual meshes below it.  BBoxCache
        # legally stops at that parent Boundable, so recover the union from
        # renderable descendants instead of rejecting a populated asset.
        fallback_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=False,
        )
        recovered_minimum = [math.inf, math.inf, math.inf]
        recovered_maximum = [-math.inf, -math.inf, -math.inf]
        recovered_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Gprim):
                continue
            candidate = fallback_cache.ComputeWorldBound(
                prim
            ).ComputeAlignedRange()
            if candidate.IsEmpty():
                continue
            candidate_minimum = candidate.GetMin()
            candidate_maximum = candidate.GetMax()
            values = [
                *(float(candidate_minimum[index]) for index in range(3)),
                *(float(candidate_maximum[index]) for index in range(3)),
            ]
            if any(
                not math.isfinite(value) or abs(value) >= 1.0e30
                for value in values
            ):
                continue
            recovered_count += 1
            for index in range(3):
                recovered_minimum[index] = min(
                    recovered_minimum[index],
                    float(candidate_minimum[index]),
                )
                recovered_maximum[index] = max(
                    recovered_maximum[index],
                    float(candidate_maximum[index]),
                )
        if recovered_count == 0:
            raise RuntimeError(
                f"materialized NVIDIA USD has no usable renderable bounds: {path}"
            )
        minimum = recovered_minimum
        maximum = recovered_maximum
    source_dimensions = [
        float(maximum[index] - minimum[index]) * meters_per_unit
        for index in range(3)
    ]
    if up_axis == "Z":
        dimensions = source_dimensions
        # Bottom-centre expressed in the wrapper's final Z-up metre space.
        anchor = [
            float(minimum[0] + maximum[0]) * 0.5 * meters_per_unit,
            float(minimum[1] + maximum[1]) * 0.5 * meters_per_unit,
            float(minimum[2]) * meters_per_unit,
        ]
    else:
        # +90 degrees around X maps source +Y to wrapper +Z and source +Z to
        # wrapper -Y.  Dimension and anchor validation must use that same
        # transform; treating source Z as height made Y-up assets float and
        # produced meaningless family compatibility checks.
        dimensions = [
            source_dimensions[0],
            source_dimensions[2],
            source_dimensions[1],
        ]
        anchor = [
            float(minimum[0] + maximum[0]) * 0.5 * meters_per_unit,
            -float(minimum[2] + maximum[2]) * 0.5 * meters_per_unit,
            float(minimum[1]) * meters_per_unit,
        ]
    if (
        any(not math.isfinite(value) or value <= 0.0001 for value in dimensions)
        or any(not math.isfinite(value) for value in anchor)
    ):
        raise RuntimeError(
            f"materialized NVIDIA USD has unusable native bounds: {path}"
        )

    variant_lods: set[str] = set()
    hierarchy_lods: set[str] = set()
    material_prim_count = 0
    bound_material_prim_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_prim_count += 1
        if prim.IsA(UsdGeom.Gprim):
            material, _relationship = (
                UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            )
            if material and material.GetPrim().IsValid():
                bound_material_prim_count += 1
        for variant_name in prim.GetVariantSets().GetNames():
            if "lod" not in variant_name.casefold():
                continue
            variant_set = prim.GetVariantSet(variant_name)
            for value in variant_set.GetVariantNames():
                variant_lods.add(f"{prim.GetPath()}:{variant_name}={value}")
        if re.fullmatch(r"(?i)lod[_-]?\d+", prim.GetName()):
            hierarchy_lods.add(str(prim.GetPath()))
    if variant_lods:
        lod_strategy = "native_variant_set"
        lod_levels = sorted(variant_lods)
    elif hierarchy_lods:
        lod_strategy = "native_prim_hierarchy"
        lod_levels = sorted(hierarchy_lods)
    else:
        lod_strategy = "source_default_only"
        lod_levels = [str(default_prim.GetPath())]

    _layers, resolved_assets, unresolved_assets = UsdUtils.ComputeAllDependencies(
        str(path)
    )
    unresolved = sorted(
        {
            str(asset)
            for asset in unresolved_assets
            if str(asset).strip()
            and not _is_builtin_omniverse_mdl_module(str(asset))
        }
    )
    metadata: dict[str, Any] = {
        "source_meters_per_unit": meters_per_unit,
        "source_up_axis": up_axis,
        "native_dimensions_m": {
            "x": dimensions[0],
            "y": dimensions[1],
            "z": dimensions[2],
        },
        "ground_anchor_m": anchor,
        "anchor_validation": {
            "state": "passed",
            "policy": "native_bbox_bottom_center",
        },
        "lod": {
            "state": "passed",
            "strategy": lod_strategy,
            "levels": lod_levels,
            "level_count": len(lod_levels),
        },
        "materials": {
            "state": (
                "passed"
                if (
                    material_prim_count > 0
                    and bound_material_prim_count > 0
                    and not unresolved
                )
                else "failed"
            ),
            "material_prim_count": material_prim_count,
            "bound_material_prim_count": bound_material_prim_count,
            "resolved_asset_dependency_count": len(resolved_assets),
            "unresolved_dependencies": unresolved,
        },
        "placement": {
            "grounding": "native_anchor",
            "scale_policy": "uniform_only",
            "non_uniform_scale_allowed": False,
            "minimum_uniform_scale": 0.8,
            "maximum_uniform_scale": 1.25,
        },
    }
    metadata["metadata_validation_sha256"] = _metadata_validation_sha256(metadata)
    return metadata


def _is_builtin_omniverse_mdl_module(raw_path: str) -> bool:
    """Identify MDL modules supplied by every supported Kit runtime.

    ``UsdUtils.ComputeAllDependencies`` reports bare MDL module names as
    unresolved filesystem paths because MDL search paths are resolved by Kit,
    not by USD.  Only the explicit NVIDIA core module used by the Rivermark
    thumbnail rigs is accepted here; project or asset-relative MDL files must
    still be present in the locked standalone cache.
    """

    normalized = raw_path.strip().replace("\\", "/")
    return normalized == "OmniPBR.mdl"


_TECHNICAL_RENDER_PRIM_NAMES = frozenset(
    {
        "tagging",
        "thumbrig",
        "thumbnail",
        "thumbnails",
    }
)


def _is_technical_render_prim_path(raw_path: object) -> bool:
    parts = {
        part.casefold()
        for part in PurePosixPath(str(raw_path)).parts
        if part != "/"
    }
    return bool(parts & _TECHNICAL_RENDER_PRIM_NAMES)


def _inspect_editable_nvidia_payload(path: Path) -> dict[str, Any]:
    """Prove that a USD exposes editable, material-bound provider geometry."""

    try:
        import isaacsim  # noqa: F401 - exposes the bundled pxr modules
        from pxr import Usd, UsdGeom, UsdShade
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "editable NVIDIA payload inspection requires the pinned Isaac runtime"
        ) from exc
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        return {
            "state": "rejected",
            "reason": "open_failed",
            "editable_mesh_count": 0,
            "material_bound_mesh_count": 0,
            "face_count": 0,
        }
    default_prim = stage.GetDefaultPrim()
    editable_mesh_count = 0
    material_bound_mesh_count = 0
    face_count = 0
    for prim in stage.Traverse():
        if (
            not prim.IsActive()
            or not prim.IsDefined()
            or not prim.IsA(UsdGeom.Mesh)
            or _is_technical_render_prim_path(prim.GetPath())
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        if (
            points is None
            or len(points) < 4
            or face_vertex_counts is None
            or len(face_vertex_counts) < 1
        ):
            continue
        editable_mesh_count += 1
        face_count += len(face_vertex_counts)
        material, _relationship = UsdShade.MaterialBindingAPI(
            prim
        ).ComputeBoundMaterial()
        has_material = bool(material and material.GetPrim().IsValid())
        if not has_material:
            for subset in UsdShade.MaterialBindingAPI(
                prim
            ).GetMaterialBindSubsets():
                subset_material, _ = UsdShade.MaterialBindingAPI(
                    subset.GetPrim()
                ).ComputeBoundMaterial()
                if subset_material and subset_material.GetPrim().IsValid():
                    has_material = True
                    break
        material_bound_mesh_count += int(has_material)
    passed = bool(
        default_prim
        and default_prim.IsValid()
        and editable_mesh_count > 0
        and material_bound_mesh_count > 0
        and face_count >= 4
    )
    return {
        "state": "passed" if passed else "rejected",
        "reason": "" if passed else "no_editable_material_bound_geometry",
        "default_prim": (
            str(default_prim.GetPath())
            if default_prim and default_prim.IsValid()
            else ""
        ),
        "editable_mesh_count": editable_mesh_count,
        "material_bound_mesh_count": material_bound_mesh_count,
        "face_count": face_count,
    }


def _select_editable_nvidia_payload(
    *,
    main_path: Path,
    inspections: dict[Path, dict[str, Any]],
    usd_dependencies: dict[Path, set[Path]],
) -> Path:
    """Choose the single editable composition root, never a backing layer."""

    main = main_path.resolve()
    if inspections.get(main, {}).get("state") == "passed":
        return main
    qualified = {
        path.resolve()
        for path, inspection in inspections.items()
        if inspection.get("state") == "passed"
    }
    referenced_backing_layers = {
        dependency.resolve()
        for candidate in qualified
        for dependency in usd_dependencies.get(candidate, set())
        if dependency.resolve() in qualified and dependency.resolve() != candidate
    }
    composition_roots = qualified - referenced_backing_layers
    if len(composition_roots) != 1:
        diagnostics = {
            str(path): {
                "inspection": inspections[path],
                "usd_dependencies": sorted(
                    str(dependency)
                    for dependency in usd_dependencies.get(path, set())
                ),
            }
            for path in sorted(inspections, key=str)
        }
        raise RuntimeError(
            "selected NVIDIA layer exposes no unique editable renderable "
            f"composition root: main={main}, diagnostics="
            f"{json.dumps(diagnostics, sort_keys=True)}"
        )
    return next(iter(composition_roots))


def materialize_indexed_nvidia_assets(
    *,
    volume_root: Path,
    assets: Iterable[dict[str, Any]],
    asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    reader: Callable[[str], bytes] = _http_read_bytes,
) -> list[dict[str, Any]]:
    """Cache selected immutable NVIDIA folders and hash every dependency."""

    try:
        import isaacsim  # noqa: F401 - exposes the bundled pxr modules
        from pxr import UsdUtils
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "indexed NVIDIA materialization requires the pinned Isaac runtime"
        ) from exc

    volume = volume_root.resolve()
    _manifest_path, file_list_path, _lock = _verified_index_paths(
        volume_root=volume,
        asset_root=asset_root,
    )
    indexed_files = [
        line.strip().replace("\\", "/").lstrip("/")
        for line in file_list_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    indexed_file_set = set(indexed_files)
    base_uri = (
        f"{_validate_official_root(asset_root)}/{RIVERMARK_CONTENT_PATH}"
    )
    cache_root = volume / "input" / "nvidia-asset-cache"
    materialized: list[dict[str, Any]] = []
    selected = list(assets)

    def download_relative(relative: str) -> Path:
        if relative not in indexed_file_set:
            raise RuntimeError(
                "referenced NVIDIA dependency is absent from the locked "
                f"Rivermark index: {relative}"
            )
        destination = (cache_root / Path(relative)).resolve()
        if cache_root.resolve() not in destination.parents:
            raise RuntimeError(f"unsafe NVIDIA dependency path: {relative}")
        if not destination.is_file():
            remote_uri = (
                f"{base_uri}/"
                f"{urllib.parse.quote(relative, safe='/%:@+-._~')}"
            )
            _atomic_write_bytes(destination, reader(remote_uri))
        return destination

    for asset_index, asset in enumerate(selected, start=1):
        relative_main = _rivermark_relative_path(
            str(asset["uri"]),
            asset_root=asset_root,
        )
        folder_prefix = f"{str(PurePosixPath(relative_main).parent)}/"
        dependencies = sorted(
            {
                relative
                for relative in indexed_files
                if relative.startswith(folder_prefix)
                and "/.thumbs/" not in f"/{relative.lower()}"
            }
        )
        if relative_main not in dependencies:
            raise RuntimeError(
                f"selected NVIDIA asset is absent from its locked index: {relative_main}"
            )
        materialized_paths: set[Path] = set()
        for file_index, relative in enumerate(dependencies, start=1):
            write_progress(
                volume,
                phase="official_nvidia_asset_download",
                message=(
                    f"Asset NVIDIA {asset_index}/{len(selected)}, fichier "
                    f"{file_index}/{len(dependencies)}."
                ),
                current_asset=PurePosixPath(relative_main).stem,
                assets_completed=asset_index - 1,
                assets_total=len(selected),
                files_completed=file_index - 1,
                files_total=len(dependencies),
            )
            materialized_paths.add(download_relative(relative))
        main_path = (cache_root / Path(relative_main)).resolve()

        # Rivermark's folder listing is not a dependency graph.  Complete
        # structures reference shared nv_core MDL files outside their own
        # folder, so resolve the USD graph iteratively against the same locked
        # provider index.  This keeps the final bundle standalone instead of
        # silently accepting unresolved materials.
        for _iteration in range(6):
            _layers, resolved_assets, unresolved_assets = (
                UsdUtils.ComputeAllDependencies(str(main_path))
            )
            for raw_path in resolved_assets:
                dependency = Path(str(raw_path)).resolve()
                if dependency.is_file() and (
                    dependency == cache_root.resolve()
                    or cache_root.resolve() in dependency.parents
                ):
                    materialized_paths.add(dependency)
            missing_relatives: list[str] = []
            for raw_path in unresolved_assets:
                if _is_builtin_omniverse_mdl_module(str(raw_path)):
                    continue
                dependency = Path(str(raw_path)).resolve()
                if cache_root.resolve() not in dependency.parents:
                    raise RuntimeError(
                        "indexed NVIDIA dependency escaped the local cache: "
                        f"{raw_path}"
                    )
                relative = dependency.relative_to(cache_root.resolve()).as_posix()
                if relative not in indexed_file_set:
                    raise RuntimeError(
                        "indexed NVIDIA dependency is absent from the locked "
                        f"provider index: {relative}"
                    )
                if not dependency.is_file():
                    missing_relatives.append(relative)
            if not missing_relatives:
                break
            with ThreadPoolExecutor(max_workers=8) as pool:
                materialized_paths.update(
                    pool.map(
                        download_relative,
                        sorted(set(missing_relatives)),
                    )
                )
        else:
            raise RuntimeError(
                "indexed NVIDIA dependency resolution did not converge: "
                f"{relative_main}"
            )

        sibling_usd_paths = sorted(
            {
                path.resolve()
                for path in materialized_paths
                if path.parent == main_path.parent
                and path.suffix.casefold() in {".usd", ".usda", ".usdc"}
            },
            key=str,
        )
        inspections = {
            path: _inspect_editable_nvidia_payload(path)
            for path in sibling_usd_paths
        }
        usd_dependencies: dict[Path, set[Path]] = {}
        for path, inspection in inspections.items():
            if inspection.get("state") != "passed":
                usd_dependencies[path] = set()
                continue
            layers, _resolved_assets, _unresolved_assets = (
                UsdUtils.ComputeAllDependencies(str(path))
            )
            usd_dependencies[path] = {
                Path(str(getattr(layer, "realPath", "") or "")).resolve()
                for layer in layers
                if str(getattr(layer, "realPath", "") or "").strip()
            }
        editable_main_path = _select_editable_nvidia_payload(
            main_path=main_path,
            inspections=inspections,
            usd_dependencies=usd_dependencies,
        )
        # The originally indexed layer may only contain provider tagging and
        # therefore cannot reveal external dependencies of the editable
        # composition root. Resolve that selected graph before computing the
        # standalone content lock.
        for _iteration in range(6):
            _layers, resolved_assets, unresolved_assets = (
                UsdUtils.ComputeAllDependencies(str(editable_main_path))
            )
            for raw_path in resolved_assets:
                dependency = Path(str(raw_path)).resolve()
                if dependency.is_file() and (
                    dependency == cache_root.resolve()
                    or cache_root.resolve() in dependency.parents
                ):
                    materialized_paths.add(dependency)
            missing_relatives = []
            for raw_path in unresolved_assets:
                if _is_builtin_omniverse_mdl_module(str(raw_path)):
                    continue
                dependency = Path(str(raw_path)).resolve()
                if cache_root.resolve() not in dependency.parents:
                    raise RuntimeError(
                        "editable NVIDIA payload dependency escaped the local "
                        f"cache: {raw_path}"
                    )
                relative = dependency.relative_to(cache_root.resolve()).as_posix()
                if relative not in indexed_file_set:
                    raise RuntimeError(
                        "editable NVIDIA payload dependency is absent from the "
                        f"locked provider index: {relative}"
                    )
                if not dependency.is_file():
                    missing_relatives.append(relative)
            if not missing_relatives:
                break
            with ThreadPoolExecutor(max_workers=8) as pool:
                materialized_paths.update(
                    pool.map(
                        download_relative,
                        sorted(set(missing_relatives)),
                    )
                )
        else:
            raise RuntimeError(
                "editable NVIDIA payload dependency resolution did not "
                f"converge: {editable_main_path}"
            )
        locked_files = [
            {
                "path": os.path.relpath(path, volume).replace("\\", "/"),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(materialized_paths)
        ]
        content_lock = hashlib.sha256(
            json.dumps(locked_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
        materialized.append(
            {
                **asset,
                **_inspect_local_usd_metadata(editable_main_path),
                "local_path": editable_main_path,
                "provider_hash": _sha256(editable_main_path),
                "provider_size_bytes": editable_main_path.stat().st_size,
                "dependency_count": len(locked_files),
                "content_lock_sha256": content_lock,
                "materialized_files": locked_files,
            }
        )
    write_progress(
        volume,
        phase="official_nvidia_asset_download",
        state="completed",
        message="Assets NVIDIA et dépendances mis en cache local et hashés.",
        assets_completed=len(materialized),
        assets_total=len(selected),
        files_completed=sum(
            int(asset.get("dependency_count", 0)) for asset in materialized
        ),
    )
    return materialized


def materialize_official_vegetation_assets(
    *,
    volume_root: Path,
    assets: Iterable[dict[str, Any]],
    reader: Callable[[str], bytes] = _http_read_bytes,
) -> list[dict[str, Any]]:
    """Materialize selected NVIDIA vegetation plus referenced MDL/PBR resources."""

    try:
        import isaacsim  # noqa: F401 - exposes the bundled pxr modules
        from pxr import UsdUtils
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "NVIDIA vegetation materialization requires the pinned Isaac runtime"
        ) from exc

    volume = volume_root.resolve()
    cache_root = (
        volume / "input" / "nvidia-asset-cache" / "official-content"
    ).resolve()
    selected = list(assets)
    for asset in selected:
        key = str(asset.get("bucket_key", "")).strip()
        if not key.startswith(NVIDIA_VEGETATION_PREFIX):
            raise RuntimeError(f"vegetation asset has no official bucket key: {key}")
    materialized: list[dict[str, Any]] = []

    def download_key(key: str) -> Path:
        parts = PurePosixPath(key).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise RuntimeError(f"unsafe NVIDIA vegetation key: {key}")
        destination = (cache_root / Path(*parts)).resolve()
        if cache_root not in destination.parents:
            raise RuntimeError(f"NVIDIA vegetation key escaped cache: {key}")
        if not destination.is_file():
            uri = (
                f"{NVIDIA_BUCKET_ORIGIN}/"
                f"{urllib.parse.quote(key, safe='/%:@+-._~')}"
            )
            if reader is _http_read_bytes:
                _http_download_file(uri, destination)
            else:
                _atomic_write_bytes(destination, reader(uri))
        return destination

    main_paths: dict[str, Path] = {}
    write_progress(
        volume,
        phase="official_nvidia_asset_download",
        message=(
            f"Téléchargement local des {len(selected)} végétaux NVIDIA "
            "sélectionnés."
        ),
        main_assets_completed=0,
        main_assets_total=len(selected),
        assets_completed=0,
        assets_total=len(selected),
    )
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(selected)))) as pool:
        futures = {
            pool.submit(
                download_key,
                str(asset.get("bucket_key", "")).strip(),
            ): asset
            for asset in selected
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            asset = futures[future]
            main_paths[str(asset["uri"])] = future.result()
            write_progress(
                volume,
                phase="official_nvidia_asset_download",
                message=(
                    f"Fichier principal NVIDIA {completed}/{len(selected)} : "
                    f"{PurePosixPath(str(asset['bucket_key'])).stem}."
                ),
                current_asset=PurePosixPath(str(asset["bucket_key"])).stem,
                main_assets_completed=completed,
                main_assets_total=len(selected),
                assets_completed=0,
                assets_total=len(selected),
            )

    for asset_index, asset in enumerate(selected, start=1):
        key = str(asset.get("bucket_key", "")).strip()
        write_progress(
            volume,
            phase="official_nvidia_asset_download",
            message=(
                f"Végétation NVIDIA {asset_index}/{len(selected)} : "
                f"{PurePosixPath(key).stem}."
            ),
            current_asset=PurePosixPath(key).stem,
            assets_completed=asset_index - 1,
            assets_total=len(selected),
        )
        main_path = main_paths[str(asset["uri"])]
        downloaded: set[Path] = {main_path}
        for _iteration in range(4):
            _layers, _resolved, unresolved = UsdUtils.ComputeAllDependencies(
                str(main_path)
            )
            missing = []
            for raw_path in unresolved:
                dependency = Path(str(raw_path)).resolve()
                if cache_root not in dependency.parents:
                    raise RuntimeError(
                        f"vegetation dependency escaped NVIDIA cache: {raw_path}"
                    )
                if not dependency.is_file():
                    missing.append(dependency)
            if not missing:
                break
            relative_dependencies = [
                dependency.relative_to(cache_root).as_posix()
                for dependency in sorted(set(missing))
            ]
            with ThreadPoolExecutor(max_workers=4) as pool:
                downloaded.update(pool.map(download_key, relative_dependencies))
        else:
            raise RuntimeError(
                f"vegetation dependency resolution did not converge: {key}"
            )

        texture_pattern = re.compile(
            r"""["']([^"']+\.(?:exr|jpeg|jpg|png|tga|tif|tiff))["']""",
            re.IGNORECASE,
        )
        texture_keys: set[str] = set()
        for mdl_path in sorted(main_path.parent.rglob("*.mdl")):
            downloaded.add(mdl_path)
            source = mdl_path.read_text(encoding="utf-8", errors="replace")
            for match in texture_pattern.finditer(source):
                reference = match.group(1).replace("\\", "/")
                if "://" in reference or reference.startswith("/"):
                    raise RuntimeError(
                        f"unsupported absolute MDL resource in {mdl_path}: {reference}"
                    )
                texture_path = (mdl_path.parent / Path(reference)).resolve()
                if cache_root not in texture_path.parents:
                    raise RuntimeError(
                        f"MDL texture escaped NVIDIA cache: {reference}"
                    )
                texture_keys.add(texture_path.relative_to(cache_root).as_posix())
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloaded.update(pool.map(download_key, sorted(texture_keys)))

        thumbnail_key = (
            f"{str(PurePosixPath(key).parent)}/.thumbs/256x256/"
            f"{PurePosixPath(key).name}.png"
        )
        thumbnail_path = download_key(thumbnail_key)
        downloaded.add(thumbnail_path)
        locked_files = [
            {
                "path": os.path.relpath(path, volume).replace("\\", "/"),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(downloaded)
        ]
        content_lock = hashlib.sha256(
            json.dumps(locked_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
        materialized.append(
            {
                **asset,
                **_inspect_local_usd_metadata(main_path),
                "local_path": main_path,
                "provider_hash": _sha256(main_path),
                "provider_size_bytes": main_path.stat().st_size,
                "dependency_count": len(locked_files) - 1,
                "content_lock_sha256": content_lock,
                "materialized_files": locked_files,
                "thumbnail_path": thumbnail_path,
            }
        )
    return materialized


def discover_official_nvidia_assets(
    asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    *,
    lister: Callable[[str], Iterable[dict[str, Any]]] = _omni_client_list,
    volume_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a bounded, deterministic inventory of official NVIDIA USD layers."""

    root = _validate_official_root(asset_root)
    if volume_root is not None and lister is _omni_client_list:
        return discover_official_nvidia_assets_from_indexes(
            volume_root=volume_root,
            asset_root=root,
        )
    queue: deque[tuple[str, int]] = deque(
        (f"{root}/{subroot}", 0) for subroot in DISCOVERY_SUBROOTS
    )
    visited: set[str] = set()
    assets: list[dict[str, Any]] = []
    while queue:
        directory, depth = queue.popleft()
        if directory in visited:
            continue
        visited.add(directory)
        if len(visited) > MAX_DISCOVERY_DIRECTORIES:
            raise RuntimeError("NVIDIA asset discovery exceeded its directory safety bound")
        entries = sorted(
            lister(directory),
            key=lambda entry: str(entry.get("relative_path", "")).lower(),
        )
        for entry in entries:
            relative = str(entry.get("relative_path", "")).strip().replace("\\", "/")
            if not relative or relative in {".", ".."}:
                continue
            path_parts = {
                part.lower()
                for part in Path(relative).parts
                if part not in {"/", "\\"}
            }
            if path_parts & _SKIP_PATH_PARTS:
                continue
            child = f"{directory.rstrip('/')}/{urllib.parse.quote(relative, safe='/%:@+-._~')}"
            if bool(entry.get("is_folder")):
                if depth < MAX_DISCOVERY_DEPTH:
                    queue.append((child, depth + 1))
                continue
            if Path(relative).suffix.lower() not in USD_SUFFIXES:
                continue
            assets.append(
                {
                    "uri": child,
                    "provider_hash": str(entry.get("provider_hash", "")).strip(),
                    "provider_version": str(
                        entry.get("provider_version", "")
                    ).strip(),
                    "size_bytes": max(0, int(entry.get("size_bytes", 0))),
                }
            )
            if len(assets) > MAX_DISCOVERY_ASSETS:
                raise RuntimeError("NVIDIA asset discovery exceeded its asset safety bound")
    unique = {asset["uri"]: asset for asset in assets}
    return [unique[uri] for uri in sorted(unique)]


def _environment_score(asset: dict[str, Any], terms: set[str]) -> int:
    normalized, tokens = _normalize_uri(str(asset["uri"]))
    score = 100 * len(tokens & terms)
    if "/outdoor/" in urllib.parse.unquote(str(asset["uri"])).lower():
        score += 40
    if "/simready/" in urllib.parse.unquote(str(asset["uri"])).lower():
        score += 25
    if "high" in tokens or "detailed" in tokens:
        score += 10
    score += min(int(asset.get("size_bytes", 0)) // 10_000_000, 20)
    return score


def _preferred_rank(uri: str, suffixes: tuple[str, ...]) -> int:
    decoded = urllib.parse.unquote(uri).lower()
    for index, suffix in enumerate(suffixes):
        if decoded.endswith(suffix):
            return index
    return len(suffixes)


def _asset_family_identity(asset: dict[str, Any]) -> str:
    provider_hash = str(asset.get("provider_hash", "")).strip()
    if provider_hash:
        return provider_hash
    decoded = urllib.parse.unquote(str(asset["uri"])).lower()
    stem = Path(decoded).stem
    stem = re.sub(r"_(?:inst(?:_base)?|base)$", "", stem)
    return f"{str(Path(decoded).parent).lower()}/{stem}"


def _classify_vegetation_family(
    normalized: str,
    tokens: set[str],
    decoded_uri: str,
) -> str | None:
    if tokens & _INDOOR_VEGETATION_TERMS:
        return None
    # NVIDIA's official vegetation library stores several ground-cover
    # species (ferns, grasses and lilies) below ``Shrub`` or
    # ``Plant_Tropical``.  The canonical asset name is more precise than the
    # broad storage folder and prevents stale Rivermark stand-ins from being
    # selected merely to fill the understory quota.
    if tokens & {
        "fern",
        "flower",
        "grass",
        "groundcover",
        "herb",
        "lily",
        "reed",
        "understory",
        "weed",
    }:
        return "understory"
    if "/trees/" in decoded_uri:
        return "trees"
    if "/shrub/" in decoded_uri or "/shrubs/" in decoded_uri:
        return "shrubs"
    if any(
        marker in decoded_uri
        for marker in ("/grass/", "/herb/", "/plants/", "/understory/")
    ):
        return "understory"
    # ``Plant_Tropical`` is only a provider storage taxonomy.  Treating its
    # generic ``plant`` token as a semantic family promoted palms and ornamental
    # trees to ground cover.  Only the explicit ground-cover names handled
    # above are suitable for the understory contract.
    if "/plant_tropical/" in decoded_uri:
        return None
    scores = {
        family: len(tokens & terms)
        for family, terms in _VEGETATION_FAMILY_TERMS.items()
    }
    best = max(scores, key=lambda family: scores[family])
    return best if scores[best] > 0 else None


def _classify_building_family(
    normalized: str,
    tokens: set[str],
    decoded_uri: str,
) -> str | None:
    structure_path = any(
        marker in decoded_uri
        for marker in (
            "/architecture/",
            "/building/",
            "/buildings/",
            "/props_structures/",
            "/structure/",
            "/structures/",
        )
    )
    structure_nouns = frozenset(
        {
            "annex",
            "apartment",
            "barn",
            "building",
            "cabin",
            "carport",
            "cottage",
            "factory",
            "farmhouse",
            "garage",
            "hangar",
            "house",
            "outbuilding",
            "residence",
            "shed",
            "silo",
            "stable",
            "structure",
            "villa",
            "warehouse",
            "workshop",
        }
    )
    # A directory named ``Warehouse`` contains hundreds of rails, pallets,
    # pipes and barriers.  Classifying those child props as industrial
    # buildings produced a formally complete but visibly absurd library.  The
    # canonical USD filename itself must identify a complete structure.
    stem = Path(urllib.parse.unquote(decoded_uri)).stem
    ordered_stem_tokens = [
        token
        for token in re.sub(r"[^a-z0-9]+", "_", stem).split("_")
        if token
    ]
    stem_tokens = set(ordered_stem_tokens)
    non_building_components = frozenset(
        {
            "barrier",
            "beam",
            "box",
            "column",
            "container",
            "cube",
            "door",
            "equipment",
            "fence",
            "gate",
            "pallet",
            "panel",
            "pipe",
            "rail",
            "roof",
            "shelf",
            "stair",
            "wall",
            "window",
        }
    )
    if stem_tokens & non_building_components:
        return None
    descriptive_suffixes = frozenset(
        {
            "black",
            "brick",
            "concrete",
            "large",
            "metal",
            "modern",
            "old",
            "red",
            "small",
            "stone",
            "white",
            "wood",
            "wooden",
        }
    )
    substantive = [
        token
        for token in ordered_stem_tokens
        if token not in {"asset", "mesh", "model", "prop", "sm", "usd"}
        and token not in descriptive_suffixes
        and not re.fullmatch(r"[a-z]?\d+", token)
    ]
    if not substantive or substantive[-1] not in structure_nouns:
        return None
    if not structure_path and not tokens & structure_nouns:
        return None
    scores = {
        family: len(tokens & terms)
        + sum(1 for term in terms if term in normalized)
        for family, terms in _BUILDING_FAMILY_TERMS.items()
    }
    for priority in ("industrial", "agricultural", "annex", "habitat"):
        if scores[priority] == max(scores.values()) and scores[priority] > 0:
            return priority
    return None


def select_simready_assets(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Select a diverse, family-complete photoreal environment library."""

    inventory = sorted(candidates, key=lambda asset: str(asset["uri"]))
    selected_uris: set[str] = set()
    selected_identities: set[str] = set()
    vegetation_candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {
        family: [] for family in PHOTOREAL_FAMILY_MINIMUMS["vegetation"]
    }
    building_candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {
        family: [] for family in PHOTOREAL_FAMILY_MINIMUMS["buildings"]
    }
    for asset in inventory:
        normalized, tokens = _normalize_uri(str(asset["uri"]))
        decoded_uri = urllib.parse.unquote(str(asset["uri"])).lower()
        # The official Norway Spruce layer authors material:binding
        # relationships without applying MaterialBindingAPI.  Current USD
        # therefore ignores every one of its 41 bindings.  Prefer another
        # real official tree instead of accepting a visibly unshaded asset or
        # mutating NVIDIA's source layer.
        if decoded_uri.endswith("/norway_spruce.usd"):
            continue
        if any(forbidden in normalized for forbidden in _SKIP_PATH_PARTS):
            continue
        is_official_vegetation = "/assets/vegetation/" in decoded_uri
        if (
            (tokens & _VEGETATION_TERMS or is_official_vegetation)
            and not tokens & _INDOOR_VEGETATION_TERMS
        ):
            family = _classify_vegetation_family(
                normalized,
                tokens,
                decoded_uri,
            )
            if family is not None:
                vegetation_candidates[family].append(
                    (_environment_score(asset, set(_VEGETATION_TERMS)), asset)
                )
        building_family = _classify_building_family(
            normalized,
            tokens,
            decoded_uri,
        )
        if building_family is not None:
            building_candidates[building_family].append(
                (
                    _environment_score(
                        asset,
                        set(_BUILDING_FAMILY_TERMS[building_family]),
                    ),
                    asset,
                )
            )

    environment: dict[str, dict[str, list[dict[str, Any]]]] = {
        "vegetation": {
            family: [] for family in PHOTOREAL_FAMILY_MINIMUMS["vegetation"]
        },
        "buildings": {
            family: [] for family in PHOTOREAL_FAMILY_MINIMUMS["buildings"]
        },
    }
    for kind, candidates_by_family, preferred_suffixes in (
        ("vegetation", vegetation_candidates, PREFERRED_VEGETATION_SUFFIXES),
        ("buildings", building_candidates, PREFERRED_RURAL_BUILDING_SUFFIXES),
    ):
        for family, minimum in PHOTOREAL_FAMILY_MINIMUMS[kind].items():
            for _score, asset in sorted(
                candidates_by_family[family],
                key=lambda item: (
                    (
                        0
                        if (
                            kind == "vegetation"
                            and item[1].get("source_index")
                            == "Assets/Vegetation S3 inventory"
                        )
                        else 1
                    ),
                    _preferred_rank(str(item[1]["uri"]), preferred_suffixes),
                    -item[0],
                    str(item[1]["uri"]),
                ),
            ):
                identity = _asset_family_identity(asset)
                if (
                    str(asset["uri"]) in selected_uris
                    or identity in selected_identities
                ):
                    continue
                environment[kind][family].append(asset)
                selected_uris.add(str(asset["uri"]))
                selected_identities.add(identity)
                if len(environment[kind][family]) == minimum:
                    break

    missing_environment = [
        f"{kind}.{family}"
        for kind, families in PHOTOREAL_FAMILY_MINIMUMS.items()
        for family, minimum in families.items()
        if len(environment[kind][family]) < minimum
    ]
    actors: dict[str, dict[str, Any]] = {}
    for class_id, matcher in _ACTOR_MATCHERS.items():
        matches = []
        for asset in inventory:
            identity = _asset_family_identity(asset)
            if (
                str(asset["uri"]) in selected_uris
                or identity in selected_identities
            ):
                continue
            normalized, tokens = _normalize_uri(str(asset["uri"]))
            if matcher(normalized, tokens):
                matches.append(asset)
        if matches:
            selected = sorted(
                matches,
                key=lambda asset: (
                    -int(asset.get("size_bytes", 0)),
                    str(asset["uri"]),
                ),
            )[0]
            identity = _asset_family_identity(selected)
            selected_uris.add(str(selected["uri"]))
            selected_identities.add(identity)
            actors[class_id] = selected
    return {
        "environment": environment,
        "actors": actors,
        "missing_environment": missing_environment,
        "missing_actor_classes": sorted(set(_ACTOR_MATCHERS) - set(actors)),
    }


def _wrapper_entry(
    *,
    role: str,
    family: str,
    asset: dict[str, Any],
    wrapper_root: Path,
    manifest_root: Path,
) -> dict[str, Any]:
    remote_uri = str(asset["uri"])
    normalized_role = re.sub(r"[^a-z0-9_-]+", "-", role.lower()).strip("-")
    identity = hashlib.sha256(remote_uri.encode("utf-8")).hexdigest()[:16]
    wrapper = wrapper_root / f"{normalized_role}-{identity}.usda"
    local_path = asset.get("local_path")
    if local_path is not None:
        reference = os.path.relpath(Path(local_path), wrapper.parent).replace(
            "\\", "/"
        )
    else:
        reference = remote_uri
    source_for_usda = reference.replace("@", "%40")
    source_for_string = remote_uri.replace("\\", "\\\\").replace('"', '\\"')
    provider_hash = str(asset.get("provider_hash", "")).replace('"', '\\"')
    provider_version = str(asset.get("provider_version", "")).replace('"', '\\"')
    source_meters_per_unit = float(asset.get("source_meters_per_unit", 1.0))
    source_up_axis = str(asset.get("source_up_axis", "Z"))
    if source_up_axis not in {"Y", "Z"}:
        raise RuntimeError(
            f"automatic NVIDIA wrapper has an unsupported source up axis: "
            f"{remote_uri}: {source_up_axis}"
        )
    raw_anchor = asset.get("ground_anchor_m")
    if (
        isinstance(raw_anchor, list)
        and len(raw_anchor) == 3
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in raw_anchor
        )
    ):
        normalized_anchor = [float(value) for value in raw_anchor]
    else:
        normalized_anchor = [0.0, 0.0, 0.0]
    rotation_block = (
        '        double xformOp:rotateX = 90\n'
        if source_up_axis == "Y"
        else ""
    )
    source_xform_order = (
        '["xformOp:rotateX", "xformOp:scale"]'
        if source_up_axis == "Y"
        else '["xformOp:scale"]'
    )
    _atomic_write_text(
        wrapper,
        f'''#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string fireviewer_discovery = "official_nvidia_isaac_6_0"
        string fireviewer_provider_hash = "{provider_hash}"
        string fireviewer_provider_version = "{provider_version}"
        string fireviewer_source_uri = "{source_for_string}"
    }}
)
def Xform "Asset"
{{
    double3 xformOp:translate = ({-normalized_anchor[0]:.9g}, {-normalized_anchor[1]:.9g}, {-normalized_anchor[2]:.9g})
    uniform token[] xformOpOrder = ["xformOp:translate"]
    def Xform "Source" (
        prepend references = @{source_for_usda}@
    )
    {{
{rotation_block}        float3 xformOp:scale = ({source_meters_per_unit:.9g}, {source_meters_per_unit:.9g}, {source_meters_per_unit:.9g})
        uniform token[] xformOpOrder = {source_xform_order}
    }}
}}
''',
    )
    licence_id = str(asset.get("license_id") or NVIDIA_ASSET_LICENSE_ID)
    licence_uri = str(asset.get("license_uri") or NVIDIA_ASSET_LICENSE_URI)
    source_anchor = list(normalized_anchor)
    metadata = {
        "native_dimensions_m": dict(
            asset.get(
                "native_dimensions_m",
                {"x": 0.0, "y": 0.0, "z": 0.0},
            )
        ),
        # The wrapper has already moved the validated source bottom-centre to
        # its local origin.  Scene instancers therefore place at terrain Z
        # without reapplying a provider-specific pivot offset.
        "ground_anchor_m": (
            [0.0, 0.0, 0.0]
            if str(
                asset.get("anchor_validation", {}).get("state", "")
            )
            == "passed"
            else []
        ),
        "anchor_validation": dict(
            asset.get(
                "anchor_validation",
                {"state": "pending_native_validation"},
            )
        ),
        "lod": dict(
            asset.get(
                "lod",
                {
                    "state": "pending_native_validation",
                    "strategy": "unknown",
                    "levels": [],
                    "level_count": 0,
                },
            )
        ),
        "materials": dict(
            asset.get(
                "materials",
                {
                    "state": "pending_native_validation",
                    "material_prim_count": 0,
                    "bound_material_prim_count": 0,
                    "resolved_asset_dependency_count": 0,
                    "unresolved_dependencies": [],
                },
            )
        ),
        "placement": dict(
            asset.get(
                "placement",
                {
                    "grounding": "native_anchor",
                    "scale_policy": "uniform_only",
                    "non_uniform_scale_allowed": False,
                    "minimum_uniform_scale": 0.8,
                    "maximum_uniform_scale": 1.25,
                },
            )
        ),
    }
    provider_metadata_validation = str(
        asset.get("metadata_validation_sha256", "")
    )
    # The wrapper changes the coordinate contract (metres, Z-up, bottom-centre
    # at the origin), so the entry hash must bind the normalized metadata, not
    # the provider-stage metadata inspected before wrapping.
    metadata_validation = _metadata_validation_sha256(metadata)
    entry = {
        "asset_id": f"{family}:{identity}",
        "family": family,
        "identity": {
            "source_name": Path(urllib.parse.unquote(remote_uri)).stem,
            "source_identity": _asset_family_identity(asset),
        },
        "path": os.path.relpath(wrapper, manifest_root).replace("\\", "/"),
        "sha256": _sha256(wrapper),
        "quality_validation": (
            "native_metadata_passed"
            if (
                metadata["anchor_validation"].get("state") == "passed"
                and metadata["lod"].get("state") == "passed"
                and metadata["materials"].get("state") == "passed"
            )
            else "pending_native_validation"
        ),
        "placement_validation": "pending_console_review",
        "provenance": {
            "provider": "NVIDIA Omniverse",
            "source_uri": remote_uri,
            "provider_hash": str(asset.get("provider_hash", "")),
            "provider_version": str(asset.get("provider_version", "")),
            "discovery": "official_nvidia_isaac_6_0",
        },
        "license": {
            "id": licence_id,
            "uri": licence_uri,
            "redistribution": (
                "nvidia_asset_not_bundled_verify_output_use_before_release"
            ),
        },
        "source_uri": remote_uri,
        "provider_hash": str(asset.get("provider_hash", "")),
        "provider_version": str(asset.get("provider_version", "")),
        "provider_size_bytes": int(
            asset.get("provider_size_bytes", asset.get("size_bytes", 0))
        ),
        "source_meters_per_unit": 1.0,
        "source_up_axis": "Z",
        "provider_source_meters_per_unit": source_meters_per_unit,
        "provider_source_up_axis": source_up_axis,
        "provider_ground_anchor_m": source_anchor,
        "provider_metadata_validation_sha256": provider_metadata_validation,
        "dependency_count": int(asset.get("dependency_count", 0)),
        "content_lock_sha256": str(asset.get("content_lock_sha256", "")),
        **metadata,
        "metadata_validation_sha256": metadata_validation,
        "materialized_files": [
            {
                "path": str(item.get("path", "")),
                "sha256": str(item.get("sha256", "")),
                "size_bytes": int(item.get("size_bytes", -1)),
            }
            for item in asset.get("materialized_files", [])
            if isinstance(item, dict)
        ],
        "source_cache_path": (
            os.path.relpath(Path(local_path), manifest_root).replace("\\", "/")
            if local_path is not None
            else ""
        ),
        "thumbnail_path": (
            os.path.relpath(
                Path(asset["thumbnail_path"]),
                manifest_root,
            ).replace("\\", "/")
            if asset.get("thumbnail_path")
            else ""
        ),
        "license_id": licence_id,
        "license_uri": licence_uri,
    }
    return entry


def provision_official_nvidia_manifest(
    *,
    volume_root: Path,
    manifest_path: Path,
    asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    lister: Callable[[str], Iterable[dict[str, Any]]] = _omni_client_list,
) -> dict[str, Any]:
    """Discover, wrap and lock usable official assets inside the production volume."""

    volume = volume_root.resolve()
    manifest = manifest_path.resolve()
    if manifest != volume and volume not in manifest.parents:
        raise ValueError("automatic asset lockfile must remain inside the production volume")
    candidates = discover_official_nvidia_assets(
        asset_root,
        lister=lister,
        volume_root=volume,
    )
    selection = select_simready_assets(candidates)
    environment_selection = selection["environment"]
    selected_environment = [
        asset
        for kind in ("vegetation", "buildings")
        for family in PHOTOREAL_FAMILY_MINIMUMS[kind]
        for asset in environment_selection[kind][family]
    ]
    locked_environment: list[dict[str, Any]] = []
    vegetation_assets = [
        asset
        for asset in selected_environment
        if asset.get("source_index") == "Assets/Vegetation S3 inventory"
    ]
    rivermark_assets = [
        asset
        for asset in selected_environment
        if asset.get("source_index")
        == (
            "Isaac/Environments/Outdoor/Rivermark/"
            "dsready_content/file_list.txt"
        )
    ]
    materialized_by_uri: dict[str, dict[str, Any]] = {}
    if vegetation_assets:
        for asset in materialize_official_vegetation_assets(
            volume_root=volume,
            assets=vegetation_assets,
        ):
            materialized_by_uri[str(asset["uri"])] = asset
    if rivermark_assets:
        for asset in materialize_indexed_nvidia_assets(
            volume_root=volume,
            assets=rivermark_assets,
            asset_root=asset_root,
        ):
            materialized_by_uri[str(asset["uri"])] = asset
    if selected_environment:
        locked_environment = [
            materialized_by_uri.get(str(asset["uri"]), asset)
            for asset in selected_environment
        ]
        locked_by_uri = {
            str(asset["uri"]): asset for asset in locked_environment
        }
        for kind in ("vegetation", "buildings"):
            for family in PHOTOREAL_FAMILY_MINIMUMS[kind]:
                environment_selection[kind][family] = [
                    locked_by_uri.get(str(asset["uri"]), asset)
                    for asset in environment_selection[kind][family]
                ]
    family_counts = {
        kind: {
            family: len(environment_selection[kind][family])
            for family in PHOTOREAL_FAMILY_MINIMUMS[kind]
        }
        for kind in ("vegetation", "buildings")
    }
    write_progress(
        volume,
        phase="official_nvidia_asset_lock",
        message=(
            "Sélection NVIDIA terminée; création du lockfile avec provenance, "
            "licence et empreintes locales."
        ),
        candidates_indexed=len(candidates),
        family_counts=family_counts,
        family_minimums=PHOTOREAL_FAMILY_MINIMUMS,
        missing_environment=selection["missing_environment"],
    )
    wrapper_root = manifest.parent / "nvidia-simready-lock"
    environment: dict[str, dict[str, list[dict[str, Any]]]] = {
        "vegetation": {},
        "buildings": {},
    }
    for kind in ("vegetation", "buildings"):
        for family in PHOTOREAL_FAMILY_MINIMUMS[kind]:
            family_id = f"{kind}.{family}"
            environment[kind][family] = [
                _wrapper_entry(
                    role=f"{kind}-{family}-{index:02d}",
                    family=family_id,
                    asset=asset,
                    wrapper_root=wrapper_root,
                    manifest_root=manifest.parent,
                )
                for index, asset in enumerate(
                    environment_selection[kind][family]
                )
            ]
    actors = {
        class_id: _wrapper_entry(
            role=f"actor-{class_id}",
            family=f"actors.{class_id}",
            asset=asset,
            wrapper_root=wrapper_root,
            manifest_root=manifest.parent,
        )
        for class_id, asset in selection["actors"].items()
    }
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile": MANIFEST_PROFILE,
        "library_policy": dict(PHOTOREAL_LIBRARY_POLICY),
        "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
        "discovery": {
            "asset_root": _validate_official_root(asset_root),
            "candidate_count": len(candidates),
            "mode": (
                "materialized_photoreal_asset_library_v3"
                if selected_environment
                and all(asset.get("local_path") for asset in locked_environment)
                else "official_nvidia_remote_reference_lock"
            ),
            "missing_environment": selection["missing_environment"],
            "missing_actor_classes": selection["missing_actor_classes"],
            "semantic_policy": (
                "exact_response_identity_only_no_generic_vehicle_promotion"
            ),
        },
        "environment": environment,
        "actors": actors,
    }
    _atomic_write_json(manifest, payload)
    report_path = manifest.with_name("simready-discovery-report.json")
    _atomic_write_json(
        report_path,
        {
            "schema_version": 1,
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "candidate_count": len(candidates),
            "selected_environment_assets": sum(
                len(entries)
                for families in environment.values()
                for entries in families.values()
            ),
            "family_counts": family_counts,
            "family_minimums": PHOTOREAL_FAMILY_MINIMUMS,
            "selected_actor_assets": len(actors),
            "missing_environment": selection["missing_environment"],
            "missing_actor_classes": selection["missing_actor_classes"],
            "production_ready": (
                not selection["missing_environment"]
                and not selection["missing_actor_classes"]
                and bool(selected_environment)
                and all(asset.get("local_path") for asset in locked_environment)
                and all(
                    asset.get("materials", {}).get("state") == "passed"
                    and asset.get("lod", {}).get("state") == "passed"
                    for asset in locked_environment
                )
            ),
        },
    )
    write_progress(
        volume,
        phase="official_nvidia_asset_lock",
        state=(
            "blocked"
            if selection["missing_environment"]
            else "completed"
        ),
        message=(
            "Lockfile NVIDIA environnement créé; revue visuelle et validation "
            "USD restent obligatoires."
            if not selection["missing_environment"]
            else "Inventaire NVIDIA incomplet pour l'environnement pilote."
        ),
        candidates_indexed=len(candidates),
        assets_locked=sum(
            len(entries)
            for families in environment.values()
            for entries in families.values()
        ),
        family_counts=family_counts,
        missing_environment=selection["missing_environment"],
        manifest=os.path.relpath(manifest, volume).replace("\\", "/"),
        report=os.path.relpath(report_path, volume).replace("\\", "/"),
    )
    return {
        "manifest": manifest,
        "report": report_path,
        "candidate_count": len(candidates),
        "missing_environment": selection["missing_environment"],
        "missing_actor_classes": selection["missing_actor_classes"],
    }


__all__ = [
    "DEFAULT_NVIDIA_ASSET_ROOT",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_PROFILE",
    "PHOTOREAL_FAMILY_MINIMUMS",
    "PHOTOREAL_LIBRARY_POLICY",
    "PHOTOREAL_MIN_LOD_LEVELS",
    "cache_official_nvidia_indexes",
    "discover_official_nvidia_assets",
    "discover_official_nvidia_assets_from_indexes",
    "materialize_indexed_nvidia_assets",
    "provision_official_nvidia_manifest",
    "select_simready_assets",
]
