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

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import csv
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
MIN_VEGETATION_VARIANTS = 6
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_PROFILE = "fireviewer_simready_photoreal_hd_v2"
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
_INDOOR_VEGETATION_TERMS = frozenset(
    {"bonsai", "indoor", "office", "potted", "warehouse"}
)
_RURAL_BUILDING_TERMS = frozenset(
    {
        "agricultural_building",
        "barn",
        "cabin",
        "cottage",
        "farm_building",
        "farm_house",
        "farmhouse",
        "house",
        "rural_building",
        "shed",
        "stable",
    }
)
_NON_RURAL_BUILDING_TERMS = frozenset(
    {"industrial", "office", "skyscraper", "warehouse"}
)
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
    reader: Callable[[str], bytes] = _omni_client_read_bytes,
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


def _inspect_local_usd_metadata(path: Path) -> dict[str, Any]:
    try:
        import isaacsim  # noqa: F401 - exposes the bundled pxr modules
        from pxr import Usd, UsdGeom
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
    return {
        "source_meters_per_unit": meters_per_unit,
        "source_up_axis": up_axis,
    }


def materialize_indexed_nvidia_assets(
    *,
    volume_root: Path,
    assets: Iterable[dict[str, Any]],
    asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    reader: Callable[[str], bytes] = _http_read_bytes,
) -> list[dict[str, Any]]:
    """Cache selected immutable NVIDIA folders and hash every dependency."""

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
    base_uri = (
        f"{_validate_official_root(asset_root)}/{RIVERMARK_CONTENT_PATH}"
    )
    cache_root = volume / "input" / "nvidia-asset-cache"
    materialized: list[dict[str, Any]] = []
    selected = list(assets)
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
        locked_files: list[dict[str, Any]] = []
        for file_index, relative in enumerate(dependencies, start=1):
            destination = (cache_root / Path(relative)).resolve()
            if cache_root.resolve() not in destination.parents:
                raise RuntimeError(f"unsafe NVIDIA dependency path: {relative}")
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
            if not destination.is_file():
                remote_uri = (
                    f"{base_uri}/"
                    f"{urllib.parse.quote(relative, safe='/%:@+-._~')}"
                )
                _atomic_write_bytes(destination, reader(remote_uri))
            locked_files.append(
                {
                    "path": os.path.relpath(destination, volume).replace("\\", "/"),
                    "sha256": _sha256(destination),
                    "size_bytes": destination.stat().st_size,
                }
            )
        main_path = (cache_root / Path(relative_main)).resolve()
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


def select_simready_assets(
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Select unique environment assets and exact-semantic response assets."""

    inventory = sorted(candidates, key=lambda asset: str(asset["uri"]))
    selected_uris: set[str] = set()
    selected_identities: set[str] = set()
    vegetation_candidates: list[tuple[int, dict[str, Any]]] = []
    building_candidates: list[tuple[int, dict[str, Any]]] = []
    for asset in inventory:
        normalized, tokens = _normalize_uri(str(asset["uri"]))
        decoded_uri = urllib.parse.unquote(str(asset["uri"])).lower()
        if any(forbidden in normalized for forbidden in _SKIP_PATH_PARTS):
            continue
        is_official_vegetation = (
            "/assets/vegetation/" in decoded_uri
            and any(
                family in decoded_uri
                for family in ("/shrub/", "/trees/")
            )
        )
        if (
            (tokens & _VEGETATION_TERMS or is_official_vegetation)
            and not tokens & _INDOOR_VEGETATION_TERMS
        ):
            vegetation_candidates.append(
                (_environment_score(asset, set(_VEGETATION_TERMS)), asset)
            )
        if (
            (
                bool(
                    tokens
                    & {
                        "barn",
                        "cabin",
                        "cottage",
                        "farmhouse",
                        "house",
                        "shed",
                        "stable",
                    }
                )
                or any(
                    term in normalized
                    for term in (
                        "agricultural_building",
                        "farm_building",
                        "farm_house",
                        "rural_building",
                    )
                )
            )
            and not tokens & _NON_RURAL_BUILDING_TERMS
        ):
            building_candidates.append(
                (_environment_score(asset, set(_RURAL_BUILDING_TERMS)), asset)
            )
    vegetation = []
    for _score, asset in sorted(
        vegetation_candidates,
        key=lambda item: (
            _preferred_rank(
                str(item[1]["uri"]),
                PREFERRED_VEGETATION_SUFFIXES,
            ),
            -item[0],
            str(item[1]["uri"]),
        ),
    ):
        identity = _asset_family_identity(asset)
        if asset["uri"] in selected_uris or identity in selected_identities:
            continue
        vegetation.append(asset)
        selected_uris.add(str(asset["uri"]))
        selected_identities.add(identity)
        if len(vegetation) == MIN_VEGETATION_VARIANTS:
            break
    rural_building = None
    for _score, asset in sorted(
        building_candidates,
        key=lambda item: (
            _preferred_rank(
                str(item[1]["uri"]),
                PREFERRED_RURAL_BUILDING_SUFFIXES,
            ),
            -item[0],
            str(item[1]["uri"]),
        ),
    ):
        identity = _asset_family_identity(asset)
        if asset["uri"] not in selected_uris and identity not in selected_identities:
            rural_building = asset
            selected_uris.add(str(asset["uri"]))
            selected_identities.add(identity)
            break
    actors: dict[str, dict[str, Any]] = {}
    for class_id, matcher in _ACTOR_MATCHERS.items():
        matches = []
        for asset in inventory:
            identity = str(asset.get("provider_hash") or asset["uri"])
            if asset["uri"] in selected_uris or identity in selected_identities:
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
            actors[class_id] = selected
            selected_uris.add(str(selected["uri"]))
            selected_identities.add(
                str(selected.get("provider_hash") or selected["uri"])
            )
    return {
        "vegetation": vegetation,
        "rural_building": rural_building,
        "actors": actors,
        "missing_environment": [
            *(
                ["vegetation"]
                if len(vegetation) < MIN_VEGETATION_VARIANTS
                else []
            ),
            *(["rural_building"] if rural_building is None else []),
        ],
        "missing_actor_classes": sorted(set(_ACTOR_MATCHERS) - set(actors)),
    }


def _wrapper_entry(
    *,
    role: str,
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
    if source_up_axis != "Z":
        raise RuntimeError(
            f"automatic NVIDIA wrapper requires a Z-up source: {remote_uri}"
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
def Xform "Asset" (
    prepend references = @{source_for_usda}@
) {{
    float3 xformOp:scale = ({source_meters_per_unit:.9g}, {source_meters_per_unit:.9g}, {source_meters_per_unit:.9g})
    uniform token[] xformOpOrder = ["xformOp:scale"]
}}
''',
    )
    return {
        "path": os.path.relpath(wrapper, manifest_root).replace("\\", "/"),
        "sha256": _sha256(wrapper),
        "quality_validation": "pending_console_review",
        "placement_validation": "pending_console_review",
        "provenance": "nvidia_simready",
        "source_uri": remote_uri,
        "provider_hash": str(asset.get("provider_hash", "")),
        "provider_version": str(asset.get("provider_version", "")),
        "provider_size_bytes": int(
            asset.get("provider_size_bytes", asset.get("size_bytes", 0))
        ),
        "source_meters_per_unit": source_meters_per_unit,
        "source_up_axis": source_up_axis,
        "dependency_count": int(asset.get("dependency_count", 0)),
        "content_lock_sha256": str(asset.get("content_lock_sha256", "")),
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
        "license_id": str(
            asset.get("license_id") or NVIDIA_ASSET_LICENSE_ID
        ),
        "license_uri": str(
            asset.get("license_uri") or NVIDIA_ASSET_LICENSE_URI
        ),
        "redistribution": (
            "nvidia_asset_not_bundled_verify_output_use_before_release"
        ),
    }


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
    selected_environment = [
        *selection["vegetation"],
        *(
            [selection["rural_building"]]
            if selection["rural_building"] is not None
            else []
        ),
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
        vegetation_count = len(selection["vegetation"])
        selection["vegetation"] = locked_environment[:vegetation_count]
        selection["rural_building"] = locked_environment[vegetation_count]
    write_progress(
        volume,
        phase="official_nvidia_asset_lock",
        message=(
            "Sélection NVIDIA terminée; création du lockfile avec provenance, "
            "licence et empreintes locales."
        ),
        candidates_indexed=len(candidates),
        vegetation_selected=len(selection["vegetation"]),
        vegetation_required=MIN_VEGETATION_VARIANTS,
        rural_building_selected=int(selection["rural_building"] is not None),
        missing_environment=selection["missing_environment"],
    )
    wrapper_root = manifest.parent / "nvidia-simready-lock"
    environment: dict[str, Any] = {"vegetation": []}
    for index, asset in enumerate(selection["vegetation"]):
        environment["vegetation"].append(
            _wrapper_entry(
                role=f"vegetation-{index:02d}",
                asset=asset,
                wrapper_root=wrapper_root,
                manifest_root=manifest.parent,
            )
        )
    if selection["rural_building"] is not None:
        environment["rural_building"] = _wrapper_entry(
            role="rural-building",
            asset=selection["rural_building"],
            wrapper_root=wrapper_root,
            manifest_root=manifest.parent,
        )
    actors = {
        class_id: _wrapper_entry(
            role=f"actor-{class_id}",
            asset=asset,
            wrapper_root=wrapper_root,
            manifest_root=manifest.parent,
        )
        for class_id, asset in selection["actors"].items()
    }
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "profile": MANIFEST_PROFILE,
        "discovery": {
            "asset_root": _validate_official_root(asset_root),
            "candidate_count": len(candidates),
            "mode": (
                "official_nvidia_materialized_lock_v2"
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
            "selected_environment_assets": len(environment["vegetation"])
            + int("rural_building" in environment),
            "selected_actor_assets": len(actors),
            "missing_environment": selection["missing_environment"],
            "missing_actor_classes": selection["missing_actor_classes"],
            "production_ready": not selection["missing_environment"]
            and not selection["missing_actor_classes"],
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
        assets_locked=len(environment["vegetation"])
        + int("rural_building" in environment),
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
    "MANIFEST_PROFILE",
    "cache_official_nvidia_indexes",
    "discover_official_nvidia_assets",
    "discover_official_nvidia_assets_from_indexes",
    "materialize_indexed_nvidia_assets",
    "provision_official_nvidia_manifest",
    "select_simready_assets",
]
