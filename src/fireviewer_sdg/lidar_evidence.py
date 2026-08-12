"""Deterministic, local-only PDAL evidence for LiDAR scene sources.

This module deliberately executes PDAL as an external process.  The production
Kit/Isaac Python environment is therefore not polluted with a second GDAL/PROJ
stack.  A receipt is useful only while every source byte, spatial reference and
classification statistic still matches a fresh PDAL probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


CONTRACT = "fireviewer.lidar-evidence"
SCHEMA_VERSION = 1
REQUIRED_DIMENSIONS = ("X", "Y", "Z", "Classification")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_COUNT = re.compile(
    r"^\s*([+-]?\d+(?:\.0+)?)\s*(?:/|:|=|,)\s*(\d+)\s*$"
)
_DEFAULT_WORKERS = 1
_MAX_WORKERS = 32


class LidarEvidenceError(RuntimeError):
    """Raised when LiDAR provenance cannot be proved."""


def _canonical_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise LidarEvidenceError(f"LiDAR evidence is not canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: object) -> None:
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(_canonical_bytes(payload))
    temporary.replace(destination)


def _pdal_binary(configured: str | os.PathLike[str] | None = None) -> str:
    value = (
        os.fspath(configured)
        if configured is not None
        else os.environ.get("FW_SDG_PDAL_BIN", "")
    )
    value = value.strip()
    if not value:
        raise LidarEvidenceError("FW_SDG_PDAL_BIN must identify the external PDAL binary")
    if "\x00" in value or "\r" in value or "\n" in value:
        raise LidarEvidenceError("FW_SDG_PDAL_BIN contains an invalid control character")
    return value


def _execute_pdal(
    pdal_bin: str,
    arguments: Sequence[str],
    *,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [pdal_bin, *arguments]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            input=stdin,
            timeout=3600,
        )
    except FileNotFoundError as exc:
        raise LidarEvidenceError(f"external PDAL binary is absent: {pdal_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LidarEvidenceError(
            f"external PDAL command timed out: {' '.join(arguments[:2])}"
        ) from exc
    except OSError as exc:
        raise LidarEvidenceError(f"external PDAL command could not start: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 1000:
            detail = detail[-1000:]
        raise LidarEvidenceError(
            f"external PDAL command failed with exit code {completed.returncode}: {detail}"
        )
    return completed


def _run_pdal(
    pdal_bin: str,
    arguments: Sequence[str],
    *,
    expect_json: bool,
) -> Any:
    completed = _execute_pdal(pdal_bin, arguments)
    output = completed.stdout.strip()
    if not expect_json:
        version = " ".join(output.split())
        if not version:
            raise LidarEvidenceError("external PDAL returned an empty version")
        return version
    try:
        decoded = json.loads(output)
    except json.JSONDecodeError as exc:
        raise LidarEvidenceError("external PDAL returned invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise LidarEvidenceError("external PDAL JSON root must be an object")
    return decoded


def _run_pdal_classification_counts(
    pdal_bin: str,
    source: Path,
) -> Mapping[str, Any]:
    """Return exact Classification counts from one streamable PDAL pass.

    ``pdal info --enumerate=Classification`` only returns the distinct scalar
    values on PDAL 2.10.2.  The ``count`` option of ``filters.stats`` is the
    corresponding exact histogram operation and emits a count for every value.
    Pipeline metadata is written to a unique temporary file because PDAL 2.10.2
    does not reliably emit ``--metadata=/dev/stdout``.
    """

    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(source)},
            {
                "type": "filters.stats",
                "dimensions": "Classification",
                "count": "Classification",
            },
        ]
    }
    stdin = json.dumps(
        pipeline,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with tempfile.TemporaryDirectory(prefix="fireviewer-pdal-counts-") as directory:
        metadata_path = Path(directory) / "metadata.json"
        _execute_pdal(
            pdal_bin,
            (
                "pipeline",
                "--stdin",
                "--stream",
                f"--metadata={metadata_path}",
            ),
            stdin=stdin,
        )
        try:
            decoded = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LidarEvidenceError(
                "external PDAL returned no Classification count metadata"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LidarEvidenceError(
                "external PDAL returned invalid Classification count metadata"
            ) from exc
    if not isinstance(decoded, Mapping):
        raise LidarEvidenceError(
            "external PDAL Classification count metadata root must be an object"
        )
    return decoded


def _as_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise LidarEvidenceError(f"{label} must be a positive integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise LidarEvidenceError(f"{label} must be a positive integer") from exc
    if number <= 0:
        raise LidarEvidenceError(f"{label} must be a positive integer")
    if isinstance(value, float) and not value.is_integer():
        raise LidarEvidenceError(f"{label} must be a positive integer")
    return number


def _as_finite_float(value: object, *, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise LidarEvidenceError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise LidarEvidenceError(f"{label} must be finite")
    return number


def _walk_mappings(value: object) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _summary_object(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = payload.get("summary", payload)
    if not isinstance(summary, Mapping):
        raise LidarEvidenceError("PDAL summary is absent")
    return summary


def _extract_point_count(summary: Mapping[str, Any]) -> int:
    for key in ("num_points", "point_count", "count", "points"):
        if key in summary:
            return _as_positive_int(summary[key], label="PDAL point count")
    metadata = summary.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("count", "num_points", "point_count"):
            if key in metadata:
                return _as_positive_int(metadata[key], label="PDAL point count")
    raise LidarEvidenceError("PDAL summary is missing its point count")


def _dimension_names(raw: object) -> list[str]:
    if isinstance(raw, str):
        values: Iterable[object] = re.split(r"[,;]", raw)
    elif isinstance(raw, list):
        values = raw
    else:
        return []
    dimensions: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("name")
        name = str(value).strip() if value is not None else ""
        if name:
            dimensions.append(name)
    unique = {name.casefold(): name for name in dimensions}
    for required in REQUIRED_DIMENSIONS:
        if required.casefold() in unique:
            unique[required.casefold()] = required
    return sorted(unique.values(), key=str.casefold)


def _extract_dimensions(summary: Mapping[str, Any]) -> list[str]:
    for mapping in _walk_mappings(summary):
        if "dimensions" in mapping:
            dimensions = _dimension_names(mapping["dimensions"])
            if dimensions:
                missing = [
                    name
                    for name in REQUIRED_DIMENSIONS
                    if name.casefold() not in {item.casefold() for item in dimensions}
                ]
                if missing:
                    raise LidarEvidenceError(
                        "PDAL source is missing required dimensions: " + ", ".join(missing)
                    )
                return dimensions
    raise LidarEvidenceError("PDAL summary is missing its dimensions")


def _extract_bounds(summary: Mapping[str, Any]) -> dict[str, float]:
    required = ("minx", "miny", "minz", "maxx", "maxy", "maxz")
    candidates = list(_walk_mappings(summary.get("bounds")))
    candidates.extend(_walk_mappings(summary))
    for candidate in candidates:
        lowered = {str(key).casefold(): value for key, value in candidate.items()}
        if all(key in lowered for key in required):
            bounds = {
                key: _as_finite_float(lowered[key], label=f"PDAL bound {key}")
                for key in required
            }
            for minimum, maximum in (("minx", "maxx"), ("miny", "maxy"), ("minz", "maxz")):
                if bounds[minimum] > bounds[maximum]:
                    raise LidarEvidenceError(f"PDAL bounds invert {minimum}/{maximum}")
            return bounds
    raise LidarEvidenceError("PDAL summary is missing finite XYZ bounds")


def _first_nonempty(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_srs(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw: object | None = summary.get("srs")
    if raw is None:
        metadata = summary.get("metadata")
        if isinstance(metadata, Mapping):
            raw = (
                metadata.get("srs")
                or metadata.get("comp_spatialreference")
                or metadata.get("spatialreference")
            )
    if isinstance(raw, str):
        wkt = raw.strip()
        if not wkt:
            raise LidarEvidenceError("PDAL source has no spatial reference")
        return {"wkt": wkt, "wkt_sha256": hashlib.sha256(wkt.encode("utf-8")).hexdigest()}
    if not isinstance(raw, Mapping):
        raise LidarEvidenceError("PDAL source has no spatial reference")

    wkt = _first_nonempty(
        raw,
        ("compoundwkt", "wkt", "horizontal", "prettycompoundwkt", "prettywkt"),
    )
    authority = _first_nonempty(raw, ("authority", "auth", "epsg"))
    proj4 = _first_nonempty(raw, ("proj4", "proj"))
    vertical = _first_nonempty(raw, ("vertical",))
    if not any((wkt, authority, proj4)):
        raise LidarEvidenceError("PDAL source has no spatial reference")
    result: dict[str, Any] = {}
    if authority:
        result["authority"] = authority
    if proj4:
        result["proj4"] = proj4
    if vertical:
        result["vertical"] = vertical
    if wkt:
        result["wkt"] = wkt
        result["wkt_sha256"] = hashlib.sha256(wkt.encode("utf-8")).hexdigest()
    for key in ("isgeocentric", "isgeographic"):
        if isinstance(raw.get(key), bool):
            result[key] = raw[key]
    return result


def _classification_value(value: object) -> int:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise LidarEvidenceError(f"invalid PDAL Classification value: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise LidarEvidenceError(f"invalid PDAL Classification value: {value!r}")
    result = int(numeric)
    if result < 0 or result > 255:
        raise LidarEvidenceError(f"PDAL Classification is outside uint8: {result}")
    return result


def _classification_counts(raw: object) -> dict[str, int]:
    items: Iterable[tuple[object, object]]
    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, list):
        pairs: list[tuple[object, object]] = []
        for item in raw:
            if isinstance(item, Mapping):
                value = item.get("value", item.get("class"))
                count = item.get("count")
                pairs.append((value, count))
            elif isinstance(item, str):
                match = _CLASS_COUNT.fullmatch(item)
                if not match:
                    raise LidarEvidenceError(
                        f"invalid PDAL Classification count entry: {item!r}"
                    )
                pairs.append((match.group(1), match.group(2)))
            else:
                raise LidarEvidenceError("PDAL Classification counts have an invalid entry")
        items = pairs
    else:
        return {}
    result: dict[str, int] = {}
    for value, count in items:
        classification = str(_classification_value(value))
        parsed_count = _as_positive_int(
            count, label=f"PDAL Classification {classification} count"
        )
        result[classification] = result.get(classification, 0) + parsed_count
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _extract_classification_histogram(
    payload: Mapping[str, Any], *, point_count: int
) -> dict[str, int]:
    statistics: list[Mapping[str, Any]] = []
    for mapping in _walk_mappings(payload.get("stats", payload)):
        raw_statistics = mapping.get("statistic")
        if isinstance(raw_statistics, list):
            statistics.extend(
                item for item in raw_statistics if isinstance(item, Mapping)
            )
    for statistic in statistics:
        name = str(statistic.get("name", statistic.get("dimension", ""))).strip()
        if name.casefold() != "classification":
            continue
        for key in ("counts", "enumeration", "values"):
            histogram = _classification_counts(statistic.get(key))
            if histogram:
                if sum(histogram.values()) != point_count:
                    raise LidarEvidenceError(
                        "PDAL Classification histogram does not cover every point"
                    )
                return histogram
        raise LidarEvidenceError("PDAL Classification statistic has no enumerated classes")
    raise LidarEvidenceError("PDAL source has no Classification statistic")


def _source_format(path: Path) -> str:
    lowered = path.name.casefold()
    if lowered.endswith(".copc.laz"):
        return "copc_laz"
    if lowered.endswith(".laz"):
        return "laz"
    raise LidarEvidenceError(f"LiDAR source is not a LAZ/COPC file: {path.name}")


def _resolve_source(root: Path, source: Path) -> tuple[Path, str]:
    candidate = source if source.is_absolute() else root / source
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LidarEvidenceError(f"LiDAR source is absent: {candidate}") from exc
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise LidarEvidenceError(f"LiDAR source escapes its declared root: {candidate}") from exc
    if not resolved.is_file():
        raise LidarEvidenceError(f"LiDAR source is not a file: {candidate}")
    _source_format(resolved)
    return resolved, relative.as_posix()


def discover_sources(source_root: Path) -> list[Path]:
    """Discover every local LAZ/COPC source below ``source_root``."""

    try:
        root = source_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LidarEvidenceError(f"LiDAR source root is absent: {source_root}") from exc
    if not root.is_dir():
        raise LidarEvidenceError(f"LiDAR source root is not a directory: {root}")
    sources = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name.casefold().endswith(".laz")),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    if not sources:
        raise LidarEvidenceError(f"LiDAR source root contains no LAZ/COPC file: {root}")
    return sources


def probe_source(
    source: Path,
    *,
    relative_path: str,
    pdal_bin: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Probe one already-confined local source through external PDAL."""

    resolved = source.resolve(strict=True)
    source_format = _source_format(resolved)
    binary = _pdal_binary(pdal_bin)
    summary_payload = _run_pdal(
        binary, ("info", "--summary", str(resolved)), expect_json=True
    )
    summary = _summary_object(summary_payload)
    point_count = _extract_point_count(summary)
    dimensions = _extract_dimensions(summary)
    bounds = _extract_bounds(summary)
    srs = _extract_srs(summary)
    statistics = _run_pdal_classification_counts(binary, resolved)
    histogram = _extract_classification_histogram(
        statistics, point_count=point_count
    )
    return {
        "path": relative_path,
        "format": source_format,
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
        "point_count": point_count,
        "bounds": bounds,
        "dimensions": dimensions,
        "srs": srs,
        "classification_histogram": histogram,
    }


def _normalize_required_classes(values: Iterable[int]) -> list[int]:
    result: set[int] = set()
    for value in values:
        parsed = _classification_value(value)
        result.add(parsed)
    return sorted(result)


def _worker_count(configured: int | None = None) -> int:
    raw: object = (
        configured
        if configured is not None
        else os.environ.get("FW_SDG_LIDAR_EVIDENCE_WORKERS", str(_DEFAULT_WORKERS))
    )
    if isinstance(raw, bool):
        raise LidarEvidenceError(
            "LiDAR evidence worker count must be an integer between "
            f"1 and {_MAX_WORKERS}"
        )
    try:
        workers = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise LidarEvidenceError(
            "LiDAR evidence worker count must be an integer between "
            f"1 and {_MAX_WORKERS}"
        ) from exc
    if workers < 1 or workers > _MAX_WORKERS:
        raise LidarEvidenceError(
            "LiDAR evidence worker count must be an integer between "
            f"1 and {_MAX_WORKERS}"
        )
    return workers


def _probe_sources(
    resolved_sources: Sequence[tuple[Path, str]],
    *,
    pdal_bin: str,
    workers: int,
) -> list[dict[str, Any]]:
    if workers == 1:
        return [
            probe_source(path, relative_path=relative, pdal_bin=pdal_bin)
            for path, relative in resolved_sources
        ]

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="fireviewer-lidar-evidence",
    )
    futures = [
        executor.submit(
            probe_source,
            path,
            relative_path=relative,
            pdal_bin=pdal_bin,
        )
        for path, relative in resolved_sources
    ]
    probed: list[dict[str, Any]] = []
    try:
        for future in as_completed(futures):
            probed.append(future.result())
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    return probed


def _assemble_receipt(
    *,
    resolved_sources: Sequence[tuple[Path, str]],
    pdal_bin: str,
    required_classes: Sequence[int],
    scope: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    version = _run_pdal(pdal_bin, ("--version",), expect_json=False)
    probed = _probe_sources(
        resolved_sources,
        pdal_bin=pdal_bin,
        workers=workers,
    )
    probed.sort(key=lambda item: str(item["path"]).casefold())

    spatial_references: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    aggregate_histogram: dict[str, int] = {}
    for item in probed:
        source = dict(item)
        srs = source.pop("srs")
        srs_sha256 = _canonical_sha256(srs)
        spatial_references[srs_sha256] = srs
        source["srs_sha256"] = srs_sha256
        sources.append(source)
        for classification, count in source["classification_histogram"].items():
            aggregate_histogram[classification] = (
                aggregate_histogram.get(classification, 0) + int(count)
            )

    present_classes = {int(value) for value in aggregate_histogram}
    missing_classes = sorted(set(required_classes) - present_classes)
    if missing_classes:
        raise LidarEvidenceError(
            "LiDAR source set is missing required classifications: "
            + ", ".join(str(value) for value in missing_classes)
        )

    inventory_binding = {
        "sources": sources,
        "spatial_references": spatial_references,
    }
    return {
        "contract": CONTRACT,
        "schema_version": SCHEMA_VERSION,
        "pdal": {"version": version},
        "requirements": {
            "dimensions": list(REQUIRED_DIMENSIONS),
            "classifications": list(required_classes),
        },
        "scope": dict(scope),
        "sources": sources,
        "spatial_references": spatial_references,
        "summary": {
            "source_count": len(sources),
            "total_bytes": sum(int(item["bytes"]) for item in sources),
            "total_points": sum(int(item["point_count"]) for item in sources),
            "classification_histogram": dict(
                sorted(aggregate_histogram.items(), key=lambda item: int(item[0]))
            ),
        },
        "inventory_sha256": _canonical_sha256(inventory_binding),
    }


def build_receipt(
    *,
    source_root: Path,
    output_path: Path,
    sources: Sequence[Path] | None = None,
    required_classes: Iterable[int] = (),
    pdal_bin: str | os.PathLike[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Probe a source set and atomically write its canonical evidence receipt."""

    try:
        root = source_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LidarEvidenceError(f"LiDAR source root is absent: {source_root}") from exc
    if not root.is_dir():
        raise LidarEvidenceError(f"LiDAR source root is not a directory: {root}")

    if sources is None:
        selected = discover_sources(root)
        scope: Mapping[str, Any] = {"kind": "recursive_laz"}
    else:
        if not sources:
            raise LidarEvidenceError("LiDAR source list is empty")
        selected = list(sources)
        scope = {"kind": "explicit"}

    resolved_sources = [_resolve_source(root, Path(source)) for source in selected]
    relative_paths = [relative for _path, relative in resolved_sources]
    if len(relative_paths) != len(set(relative_paths)):
        raise LidarEvidenceError("LiDAR source list contains duplicates")
    resolved_sources.sort(key=lambda item: item[1].casefold())

    receipt = _assemble_receipt(
        resolved_sources=resolved_sources,
        pdal_bin=_pdal_binary(pdal_bin),
        required_classes=_normalize_required_classes(required_classes),
        scope=scope,
        workers=_worker_count(workers),
    )
    _atomic_write(output_path, receipt)
    return receipt


def _read_canonical_receipt(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.resolve(strict=True).read_bytes()
    except FileNotFoundError as exc:
        raise LidarEvidenceError(f"LiDAR evidence receipt is absent: {path}") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LidarEvidenceError("LiDAR evidence receipt is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise LidarEvidenceError("LiDAR evidence receipt root must be an object")
    if raw != _canonical_bytes(decoded):
        raise LidarEvidenceError("LiDAR evidence receipt is not canonical JSON")
    return decoded, raw


def _validate_receipt_structure(receipt: Mapping[str, Any]) -> None:
    if receipt.get("contract") != CONTRACT or receipt.get("schema_version") != SCHEMA_VERSION:
        raise LidarEvidenceError("LiDAR evidence receipt has an unsupported contract")
    pdal = receipt.get("pdal")
    if not isinstance(pdal, Mapping) or not str(pdal.get("version", "")).strip():
        raise LidarEvidenceError("LiDAR evidence receipt is missing its PDAL version")
    requirements = receipt.get("requirements")
    if not isinstance(requirements, Mapping):
        raise LidarEvidenceError("LiDAR evidence receipt is missing requirements")
    if requirements.get("dimensions") != list(REQUIRED_DIMENSIONS):
        raise LidarEvidenceError("LiDAR evidence receipt has invalid required dimensions")
    raw_classes = requirements.get("classifications")
    if not isinstance(raw_classes, list):
        raise LidarEvidenceError("LiDAR evidence receipt has invalid required classifications")
    required_classes = _normalize_required_classes(raw_classes)
    if raw_classes != required_classes:
        raise LidarEvidenceError("LiDAR evidence required classifications are not canonical")

    scope = receipt.get("scope")
    if not isinstance(scope, Mapping) or scope.get("kind") not in {
        "explicit",
        "recursive_laz",
    }:
        raise LidarEvidenceError("LiDAR evidence receipt has an invalid source scope")
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        raise LidarEvidenceError("LiDAR evidence receipt contains no source")
    paths: list[str] = []
    total_bytes = 0
    total_points = 0
    aggregate: dict[str, int] = {}
    spatial_references = receipt.get("spatial_references")
    if not isinstance(spatial_references, Mapping) or not spatial_references:
        raise LidarEvidenceError("LiDAR evidence receipt contains no spatial reference")
    for fingerprint, srs in spatial_references.items():
        if not _SHA256.fullmatch(str(fingerprint)) or not isinstance(srs, Mapping):
            raise LidarEvidenceError("LiDAR evidence receipt has an invalid spatial reference")
        if _canonical_sha256(srs) != fingerprint:
            raise LidarEvidenceError("LiDAR spatial reference fingerprint does not match")

    for raw_source in sources:
        if not isinstance(raw_source, Mapping):
            raise LidarEvidenceError("LiDAR evidence source entry must be an object")
        path = str(raw_source.get("path", ""))
        if not path or "\\" in path or Path(path).is_absolute():
            raise LidarEvidenceError("LiDAR evidence source path is not canonical")
        paths.append(path)
        if raw_source.get("format") not in {"laz", "copc_laz"}:
            raise LidarEvidenceError("LiDAR evidence source format is invalid")
        digest = str(raw_source.get("sha256", ""))
        if not _SHA256.fullmatch(digest):
            raise LidarEvidenceError("LiDAR evidence source SHA-256 is invalid")
        srs_sha256 = str(raw_source.get("srs_sha256", ""))
        if srs_sha256 not in spatial_references:
            raise LidarEvidenceError("LiDAR evidence source has no bound spatial reference")
        dimensions = raw_source.get("dimensions")
        if not isinstance(dimensions, list) or any(
            required.casefold() not in {str(item).casefold() for item in dimensions}
            for required in REQUIRED_DIMENSIONS
        ):
            raise LidarEvidenceError("LiDAR evidence source is missing required dimensions")
        source_bytes = _as_positive_int(
            raw_source.get("bytes"), label="LiDAR evidence source bytes"
        )
        point_count = _as_positive_int(
            raw_source.get("point_count"), label="LiDAR evidence point count"
        )
        bounds = raw_source.get("bounds")
        if not isinstance(bounds, Mapping):
            raise LidarEvidenceError("LiDAR evidence source bounds are absent")
        _extract_bounds({"bounds": bounds})
        histogram = _classification_counts(
            raw_source.get("classification_histogram")
        )
        if not histogram or histogram != raw_source.get("classification_histogram"):
            raise LidarEvidenceError("LiDAR evidence Classification histogram is invalid")
        if sum(histogram.values()) != point_count:
            raise LidarEvidenceError(
                "LiDAR evidence Classification histogram does not cover every point"
            )
        total_bytes += source_bytes
        total_points += point_count
        for classification, count in histogram.items():
            aggregate[classification] = aggregate.get(classification, 0) + count

    if paths != sorted(paths, key=str.casefold) or len(paths) != len(set(paths)):
        raise LidarEvidenceError("LiDAR evidence source paths are not unique and sorted")
    missing_classes = sorted(set(required_classes) - {int(value) for value in aggregate})
    if missing_classes:
        raise LidarEvidenceError(
            "LiDAR evidence is missing required classifications: "
            + ", ".join(str(value) for value in missing_classes)
        )
    summary = receipt.get("summary")
    expected_summary = {
        "source_count": len(sources),
        "total_bytes": total_bytes,
        "total_points": total_points,
        "classification_histogram": dict(
            sorted(aggregate.items(), key=lambda item: int(item[0]))
        ),
    }
    if summary != expected_summary:
        raise LidarEvidenceError("LiDAR evidence aggregate summary does not match sources")
    expected_inventory = _canonical_sha256(
        {"sources": sources, "spatial_references": spatial_references}
    )
    if receipt.get("inventory_sha256") != expected_inventory:
        raise LidarEvidenceError("LiDAR evidence inventory fingerprint does not match")


def verify_receipt(
    *,
    receipt_path: Path,
    source_root: Path,
    pdal_bin: str | os.PathLike[str] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Re-probe every source and require byte-for-byte receipt equivalence."""

    receipt, _raw = _read_canonical_receipt(receipt_path)
    _validate_receipt_structure(receipt)
    try:
        root = source_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LidarEvidenceError(f"LiDAR source root is absent: {source_root}") from exc
    if not root.is_dir():
        raise LidarEvidenceError(f"LiDAR source root is not a directory: {root}")

    recorded_paths = [Path(str(item["path"])) for item in receipt["sources"]]
    if receipt["scope"]["kind"] == "recursive_laz":
        discovered = discover_sources(root)
        discovered_relative = [
            path.resolve().relative_to(root).as_posix() for path in discovered
        ]
        if discovered_relative != [path.as_posix() for path in recorded_paths]:
            raise LidarEvidenceError(
                "LiDAR recursive source set differs from the recorded receipt"
            )
    resolved_sources = [_resolve_source(root, path) for path in recorded_paths]
    for (resolved, relative), recorded in zip(resolved_sources, receipt["sources"]):
        if resolved.stat().st_size != int(recorded["bytes"]):
            raise LidarEvidenceError(f"LiDAR source size changed: {relative}")
        if _file_sha256(resolved) != recorded["sha256"]:
            raise LidarEvidenceError(f"LiDAR source SHA-256 changed: {relative}")

    rebuilt = _assemble_receipt(
        resolved_sources=resolved_sources,
        pdal_bin=_pdal_binary(pdal_bin),
        required_classes=receipt["requirements"]["classifications"],
        scope=receipt["scope"],
        workers=_worker_count(workers),
    )
    if rebuilt != receipt:
        raise LidarEvidenceError(
            "fresh PDAL evidence differs from the recorded LiDAR receipt"
        )
    return receipt


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify canonical local PDAL evidence for LAZ/COPC sources"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="probe sources and create a receipt")
    create.add_argument("--source-root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument(
        "--source",
        action="append",
        type=Path,
        help="source relative to --source-root; omit to inventory every *.laz recursively",
    )
    create.add_argument(
        "--require-class",
        action="append",
        type=int,
        default=[],
        help="classification that must occur in the aggregate source set",
    )
    create.add_argument(
        "--workers",
        type=int,
        help=(
            "bounded concurrent source probes; defaults to "
            "FW_SDG_LIDAR_EVIDENCE_WORKERS or 1"
        ),
    )
    verify = subparsers.add_parser("verify", help="re-probe and verify an existing receipt")
    verify.add_argument("--source-root", required=True, type=Path)
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument(
        "--workers",
        type=int,
        help=(
            "bounded concurrent source probes; defaults to "
            "FW_SDG_LIDAR_EVIDENCE_WORKERS or 1"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if arguments.command == "create":
        receipt = build_receipt(
            source_root=arguments.source_root,
            output_path=arguments.output,
            sources=arguments.source,
            required_classes=arguments.require_class,
            workers=arguments.workers,
        )
    else:
        receipt = verify_receipt(
            receipt_path=arguments.receipt,
            source_root=arguments.source_root,
            workers=arguments.workers,
        )
    print(
        json.dumps(
            {
                "contract": receipt["contract"],
                "inventory_sha256": receipt["inventory_sha256"],
                "source_count": receipt["summary"]["source_count"],
                "total_points": receipt["summary"]["total_points"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
