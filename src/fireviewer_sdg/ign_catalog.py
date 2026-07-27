"""Build a fail-closed 512-fire catalogue from official French terrain rasters.

The preparation step deliberately creates only three spatial sites.  Fire events,
progression and camera geometry vary independently inside those sites.  This keeps
the corpus representative of the operational constraint (few deployed cameras per
site) without cloning one event or inventing 512 unrelated locations.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from fireviewer_sdg.geometry import assert_visible, camera_contract, project_point
from fireviewer_sdg.preparation_progress import write_progress
from fireviewer_sdg.real_world import (
    LOCAL_RENDER_HEIGHT,
    LOCAL_RENDER_WIDTH,
    RENDER_HEIGHT,
    RENDER_PROFILE,
    RENDER_WIDTH,
)
from fireviewer_sdg.simready_assets import (
    DEFAULT_NVIDIA_ASSET_ROOT,
    MANIFEST_PROFILE,
    provision_official_nvidia_manifest,
)


CATALOG_NAME = "event-catalog-4096-hd-v2.json"
CATALOG_SCHEMA = 1
PREPARATION_VERSION = "ign-simready-flow-hd-v2"
ASSET_MANIFEST_NAME = "simready-assets-hd-v2.json"
ASSET_MANIFEST_PROFILE = MANIFEST_PROFILE
READINESS_REPORT_NAME = "input-readiness-hd-v2.json"
IGN_WMS = "https://data.geopf.fr/wms-r/wms"
ORTHO_LAYER = "ORTHOIMAGERY.ORTHOPHOTOS"
MNT_LAYER = "IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
SLOT_PATTERN = (4, 6, 10, 12)
SITE_SPAN_M = 2000.0
ORTHOPHOTO_PIXELS = 4096
MNT_PIXELS = 2048
TERRAIN_GRID = 257
ACTOR_CLASSES = (
    "sdis_vehicle",
    "canadair",
    "dash",
    "securite_civile_helicopter",
    "hard_negative_construction_truck",
    "hard_negative_crop_duster",
    "hard_negative_utility_helicopter",
)
SITES = (
    {
        "id": "montmaur",
        "label": "Montmaur-en-Diois",
        "profile": "rural_mountain",
        "bounds": (887576.5, 6399287.5, 889576.5, 6401287.5),
    },
    {
        "id": "barsac",
        "label": "Barsac - vallee de la Drome",
        "profile": "rural_agricultural",
        "bounds": (880380.5, 6405209.5, 882380.5, 6407209.5),
    },
    {
        "id": "ausson",
        "label": "Ausson - versant agricole",
        "profile": "mountain_agricultural",
        "bounds": (888056.0, 6404524.0, 890056.0, 6406524.0),
    },
)
PHASES = (
    "advancing_flame_zone",
    "front_split",
    "reignition",
    "initial_growth",
    "partial_suppression",
    "multi_front_spread",
    "decay",
)
PROGRESSION_PROFILES = tuple(
    permutation
    for fourth in PHASES[3:]
    for permutation in itertools.permutations((*PHASES[:3], fourth))
)
MAX_FRAMING_DISTANCE_M = {
    "near": 110.0,
    "medium": 240.0,
    "far": 600.0,
    "very_far": 950.0,
}
IGN_DOWNLOAD_ATTEMPTS = 4
IGN_RETRY_DELAYS_S = (2.0, 5.0, 15.0)
IGN_RETRYABLE_HTTP_STATUS = frozenset({400, 408, 425, 429, 500, 502, 503, 504})
ASSET_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
MIN_VEGETATION_VARIANTS = 6
FORBIDDEN_ASSET_NAME_PARTS = frozenset(
    {"demo", "lowpoly", "placeholder", "playmobil", "primitive", "proxy", "sample"}
)
ASSET_QUALITY_THRESHOLDS = {
    "vegetation": {
        "mesh_points": 1_000,
        "materials": 1,
        "textures": 1,
        "minimum_height_m": 0.25,
        "maximum_height_m": 40.0,
        "maximum_horizontal_span_m": 35.0,
    },
    "rural_building": {
        "mesh_points": 5_000,
        "materials": 2,
        "textures": 1,
        "minimum_height_m": 2.0,
        "maximum_height_m": 50.0,
        "maximum_horizontal_span_m": 150.0,
    },
    "actor": {
        "mesh_points": 5_000,
        "materials": 2,
        "textures": 1,
        "minimum_height_m": 0.5,
        "maximum_height_m": 30.0,
        "maximum_horizontal_span_m": 100.0,
    },
}
VEGETATION_SPECIES = (
    "norway_spruce",
    "douglas_fir",
    "lombardy_poplar",
    "common_apple",
    "hawthorn",
    "juniper",
)
VEGETATION_SCALE_RANGES = (
    (0.75, 1.15),
    (0.85, 1.35),
    (0.75, 1.20),
    (0.70, 1.05),
    (0.65, 1.00),
    (0.70, 1.35),
)
VEGETATION_PROFILES = {
    "rural_mountain": {
        "natural_count": 1_800,
        "weights": (0.44, 0.32, 0.02, 0.01, 0.08, 0.13),
        "clustered_fraction": 0.82,
        "orchard_rows": 0,
        "orchard_columns": 0,
        "hedgerow_count": 0,
    },
    "rural_agricultural": {
        "natural_count": 500,
        "weights": (0.03, 0.03, 0.24, 0.18, 0.32, 0.20),
        "clustered_fraction": 0.20,
        "orchard_rows": 12,
        "orchard_columns": 18,
        "hedgerow_count": 300,
    },
    "mountain_agricultural": {
        "natural_count": 1_000,
        "weights": (0.27, 0.22, 0.11, 0.09, 0.18, 0.13),
        "clustered_fraction": 0.64,
        "orchard_rows": 8,
        "orchard_columns": 15,
        "hedgerow_count": 180,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _asset_entry(
    payload: object,
    *,
    role: str,
    manifest_root: Path,
    volume_root: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"asset {role} must be an object")
    raw_path = str(payload.get("path", "")).strip()
    if not raw_path:
        raise ValueError(f"asset {role}.path is required")
    candidate = Path(raw_path)
    path = (candidate if candidate.is_absolute() else manifest_root / candidate).resolve()
    if path != volume_root and volume_root not in path.parents:
        raise ValueError(f"asset {role} must remain inside the production volume")
    if not path.is_file() or path.suffix.lower() not in ASSET_SUFFIXES:
        raise ValueError(f"asset {role} must be an existing USD asset")
    lowered_name = path.stem.lower()
    if any(part in lowered_name for part in FORBIDDEN_ASSET_NAME_PARTS):
        raise ValueError(f"asset {role} has a forbidden placeholder-style filename")
    expected = str(payload.get("sha256", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"asset {role}.sha256 must be a lowercase SHA-256")
    actual = _sha256(path)
    if not hmac.compare_digest(expected, actual):
        raise ValueError(f"asset {role}.sha256 does not match its file")
    quality_validation = str(payload.get("quality_validation", "")).strip()
    if quality_validation not in {
        "pending_console_review",
        "simready_asset_human_approved",
    }:
        raise ValueError(
            f"asset {role} requires a pending or approved SimReady quality review"
        )
    placement_validation = str(payload.get("placement_validation", "")).strip()
    if placement_validation not in {
        "pending_console_review",
        "usd_z_up_meters_grounded_human_approved",
    }:
        raise ValueError(
            f"asset {role} requires a pending or approved Z-up placement review"
        )
    provenance = str(payload.get("provenance", "")).strip()
    if provenance not in {"nvidia_simready", "licensed_third_party", "owned_original"}:
        raise ValueError(f"asset {role}.provenance is unsupported")
    if not str(payload.get("source_uri", "")).strip():
        raise ValueError(f"asset {role}.source_uri is required")
    if not str(payload.get("license_id", "")).strip():
        raise ValueError(f"asset {role}.license_id is required")
    if provenance == "nvidia_simready":
        parsed_source = urllib.parse.urlparse(str(payload["source_uri"]))
        if (
            parsed_source.scheme != "https"
            or parsed_source.hostname
            != "omniverse-content-production.s3-us-west-2.amazonaws.com"
            or not (
                "/Assets/Isaac/6.0/" in parsed_source.path
                or "/Assets/Vegetation/" in parsed_source.path
            )
        ):
            raise ValueError(
                f"asset {role} claims NVIDIA provenance outside the pinned official root"
            )
        if not (
            str(payload.get("provider_hash", "")).strip()
            or str(payload.get("provider_version", "")).strip()
        ):
            raise ValueError(
                f"asset {role} requires an NVIDIA provider hash or version"
            )
    return {
        **payload,
        "role": role,
        "path": path,
        "sha256": actual,
        "provenance": provenance,
    }


def _load_simready_asset_manifest(
    path: Path, *, volume_root: Path
) -> dict[str, Any]:
    volume = volume_root.resolve()
    manifest = path.resolve()
    if manifest != volume and volume not in manifest.parents:
        raise ValueError("SimReady asset manifest must remain inside the production volume")
    if not manifest.is_file():
        raise RuntimeError(
            "photoreal SimReady asset manifest is absent; refusing to author "
            f"primitive placeholder scenery: {manifest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported SimReady asset manifest schema_version")
    if payload.get("profile") != ASSET_MANIFEST_PROFILE:
        raise ValueError(
            f"SimReady asset manifest profile must be {ASSET_MANIFEST_PROFILE}"
        )
    environment = payload.get("environment")
    actors = payload.get("actors")
    if not isinstance(environment, dict) or not isinstance(actors, dict):
        raise ValueError("SimReady asset manifest requires environment and actors")
    vegetation_payload = environment.get("vegetation")
    if (
        not isinstance(vegetation_payload, list)
        or len(vegetation_payload) < MIN_VEGETATION_VARIANTS
    ):
        raise ValueError(
            f"SimReady manifest requires at least {MIN_VEGETATION_VARIANTS} "
            "reviewed vegetation variants"
        )
    vegetation = [
        _asset_entry(
            item,
            role=f"vegetation[{index}]",
            manifest_root=manifest.parent,
            volume_root=volume,
        )
        for index, item in enumerate(vegetation_payload)
    ]
    rural_building = _asset_entry(
        environment.get("rural_building"),
        role="rural_building",
        manifest_root=manifest.parent,
        volume_root=volume,
    )
    unknown_actor_classes = set(actors) - set(ACTOR_CLASSES)
    if unknown_actor_classes:
        raise ValueError(
            "SimReady actor manifest contains unsupported response classes: "
            + ", ".join(sorted(unknown_actor_classes))
        )
    actor_entries = {
        class_id: _asset_entry(
            actors[class_id],
            role=f"actor:{class_id}",
            manifest_root=manifest.parent,
            volume_root=volume,
        )
        for class_id in ACTOR_CLASSES
        if class_id in actors
    }
    all_entries = [*vegetation, rural_building, *actor_entries.values()]
    hashes = [entry["sha256"] for entry in all_entries]
    if len(set(hashes)) != len(hashes):
        raise ValueError("every SimReady environment and actor asset must be unique")
    return {
        "path": manifest,
        "sha256": _sha256(manifest),
        "vegetation": vegetation,
        "rural_building": rural_building,
        "actors": actor_entries,
        "missing_actor_classes": sorted(set(ACTOR_CLASSES) - set(actor_entries)),
    }


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / base
    value = index
    while value > 0:
        result += fraction * (value % base)
        value //= base
        fraction /= base
    return result


_USDA_DECLARATION = (
    r'(?:def\s+[A-Za-z_][A-Za-z0-9_]*\s+"|'
    r'(?:uniform\s+)?token(?:\[\])?\s+[A-Za-z_]|'
    r'(?:asset|bool|color3f|double3|float(?:2|3)?|int\[\]|matrix4d|'
    r'point3f\[\]|quatf|rel|string|texCoord2f\[\])\s+[A-Za-z_])'
)


def _format_generated_usda(value: str) -> str:
    """Put every generated USDA declaration and closing brace on its own line."""

    value = re.sub(
        rf'(?<!uniform)(?<=\S)[ \t]+(?={_USDA_DECLARATION})',
        "\n",
        value,
    )
    value = re.sub(r'(?<=\S)[ \t]+(?=\})', "\n", value)
    return value


def _write_text(path: Path, value: str) -> None:
    if path.suffix.lower() in {".usd", ".usda"}:
        value = _format_generated_usda(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _download(url: str, destination: Path) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "data.geopf.fr":
        raise RuntimeError("terrain source must remain on the official IGN host")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, IGN_DOWNLOAD_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "image/tiff,image/geotiff,application/octet-stream",
                "User-Agent": "FireViewer-SDG-IGN/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response, partial.open(
                "wb"
            ) as output:
                content_type = response.headers.get_content_type()
                if content_type not in {
                    "image/tiff",
                    "image/geotiff",
                    "application/octet-stream",
                }:
                    excerpt = response.read(256).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"IGN returned {content_type!r} instead of GeoTIFF: "
                        f"{excerpt!r}"
                    )
                shutil.copyfileobj(response, output, length=1024 * 1024)
            os.replace(partial, destination)
            return
        except urllib.error.HTTPError as exc:
            excerpt = exc.read(512).decode("utf-8", errors="replace").strip()
            can_retry = (
                exc.code in IGN_RETRYABLE_HTTP_STATUS
                and attempt < IGN_DOWNLOAD_ATTEMPTS
            )
            if not can_retry:
                raise RuntimeError(
                    f"IGN WMS request failed after {attempt} attempt(s): "
                    f"HTTP {exc.code}; response={excerpt[:256]!r}"
                ) from exc
            delay = IGN_RETRY_DELAYS_S[attempt - 1]
            print(
                "fireviewer inputs: IGN WMS transient response "
                f"HTTP {exc.code}; retry={attempt + 1}/{IGN_DOWNLOAD_ATTEMPTS} "
                f"in={delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt >= IGN_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"IGN WMS transport failed after {attempt} attempt(s): {exc}"
                ) from exc
            delay = IGN_RETRY_DELAYS_S[attempt - 1]
            print(
                "fireviewer inputs: IGN WMS transport retry "
                f"retry={attempt + 1}/{IGN_DOWNLOAD_ATTEMPTS} in={delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
        finally:
            partial.unlink(missing_ok=True)


def _wms_url(layer: str, bounds: tuple[float, float, float, float], pixels: int) -> str:
    query = urllib.parse.urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": layer,
            "STYLES": "",
            "CRS": "EPSG:2154",
            "BBOX": ",".join(f"{value:.3f}" for value in bounds),
            "WIDTH": str(pixels),
            "HEIGHT": str(pixels),
            "FORMAT": "image/geotiff",
        }
    )
    return f"{IGN_WMS}?{query}"


def _validated_raster(path: Path, *, pixels: int, kind: str) -> Image.Image:
    try:
        image = Image.open(path)
        image.load()
    except Exception as exc:
        raise RuntimeError(f"{kind} is not a readable IGN raster: {path}") from exc
    if image.size != (pixels, pixels):
        raise RuntimeError(
            f"{kind} dimensions differ from the controlled WMS request: {image.size}"
        )
    extrema = image.getextrema()
    if (
        not extrema
        or (
            isinstance(extrema[0], tuple)
            and not any(float(high) > float(low) for low, high in extrema)
        )
        or (
            not isinstance(extrema[0], tuple)
            and float(extrema[1]) <= float(extrema[0])
        )
    ):
        raise RuntimeError(f"{kind} contains no usable variation")
    return image


def _prepare_site(
    root: Path,
    site: dict[str, Any],
    *,
    fetcher: Callable[[str, Path], None],
    vegetation_assets: list[Path],
    building_asset: Path,
) -> dict[str, Any]:
    site_root = root / "sites" / site["id"]
    site_root.mkdir(parents=True, exist_ok=True)
    ortho_tif = site_root / "orthophoto-ign.tif"
    mnt_tif = site_root / "mnt-ign.tif"
    if not ortho_tif.is_file():
        fetcher(_wms_url(ORTHO_LAYER, site["bounds"], ORTHOPHOTO_PIXELS), ortho_tif)
    if not mnt_tif.is_file():
        fetcher(_wms_url(MNT_LAYER, site["bounds"], MNT_PIXELS), mnt_tif)
    ortho = _validated_raster(
        ortho_tif, pixels=ORTHOPHOTO_PIXELS, kind="orthophoto"
    ).convert("RGB")
    mnt = _validated_raster(mnt_tif, pixels=MNT_PIXELS, kind="MNT")
    ortho_jpg = site_root / "orthophoto-render.jpg"
    if not ortho_jpg.is_file():
        ortho.save(ortho_jpg, format="JPEG", quality=94, subsampling=0)
    preview = site_root / "mnt-preview.png"
    if not preview.is_file():
        values = mnt.convert("F").resize((512, 512), Image.Resampling.BILINEAR)
        low, high = values.getextrema()
        span = max(1e-6, float(high) - float(low))
        shaded = Image.new("L", values.size)
        shaded.putdata(
            [
                max(0, min(255, round((float(value) - float(low)) * 255.0 / span)))
                for value in values.getdata()
            ]
        )
        shaded.save(preview)
    scene = site_root / "terrain-simready.usda"
    _write_site_usda(
        scene,
        site=site,
        mnt=mnt,
        texture=ortho_jpg,
        vegetation_assets=vegetation_assets,
        building_asset=building_asset,
    )
    return {
        **site,
        "root": site_root,
        "ortho": ortho_tif,
        "ortho_render": ortho_jpg,
        "mnt": mnt_tif,
        "mnt_preview": preview,
        "scene": scene,
        "base_elevation": _center_elevation(mnt),
        "mnt_image": mnt.convert("F"),
    }


def _center_elevation(mnt: Image.Image) -> float:
    values = mnt.convert("F")
    width, height = values.size
    sample = values.crop(
        (width // 2 - 8, height // 2 - 8, width // 2 + 8, height // 2 + 8)
    )
    finite = [float(value) for value in sample.getdata() if math.isfinite(float(value))]
    if not finite:
        raise RuntimeError("MNT centre has no finite elevation")
    return sum(finite) / len(finite)


def _terrain_elevation(site: dict[str, Any], x: float, y: float) -> float:
    """Bilinearly sample local Z from the same MNT used to author the USD mesh."""
    values = site["mnt_image"]
    if not isinstance(values, Image.Image):
        raise RuntimeError("prepared site is missing its MNT sampler")
    return _mnt_local_elevation(values, float(site["base_elevation"]), x, y)


def _mnt_local_elevation(
    values: Image.Image, base_elevation: float, x: float, y: float
) -> float:
    half = SITE_SPAN_M / 2.0
    if not -half <= x <= half or not -half <= y <= half:
        raise RuntimeError("terrain sample escapes the authored USD heightfield")
    pixel_x = (x + half) * (values.width - 1) / SITE_SPAN_M
    pixel_y = (half - y) * (values.height - 1) / SITE_SPAN_M
    left = int(math.floor(pixel_x))
    top = int(math.floor(pixel_y))
    right = min(values.width - 1, left + 1)
    bottom = min(values.height - 1, top + 1)
    dx = pixel_x - left
    dy = pixel_y - top
    samples = [
        float(values.getpixel((left, top))),
        float(values.getpixel((right, top))),
        float(values.getpixel((left, bottom))),
        float(values.getpixel((right, bottom))),
    ]
    if any(not math.isfinite(value) for value in samples):
        raise RuntimeError("MNT terrain sample contains a non-finite elevation")
    absolute = (
        samples[0] * (1.0 - dx) * (1.0 - dy)
        + samples[1] * dx * (1.0 - dy)
        + samples[2] * (1.0 - dx) * dy
        + samples[3] * dx * dy
    )
    return absolute - base_elevation


def _vegetation_layout(
    *,
    site_id: str,
    profile: str,
) -> list[tuple[int, float, float, float, float]]:
    """Return a deterministic, profile-specific landscape layout.

    The six prototype slots deliberately match ``PREFERRED_VEGETATION_SUFFIXES``
    in the NVIDIA lock: spruce, fir, poplar, apple, hawthorn and juniper.
    """

    profile_config = VEGETATION_PROFILES.get(profile)
    if profile_config is None:
        raise RuntimeError(f"unsupported landscape profile: {profile}")
    rng = random.Random(f"fireviewer-site-{site_id}-vegetation-v2")
    weights = tuple(float(value) for value in profile_config["weights"])
    natural_count = int(profile_config["natural_count"])
    clustered_fraction = float(profile_config["clustered_fraction"])
    placements: list[tuple[int, float, float, float, float]] = []
    cluster_centres = [
        (rng.uniform(-680.0, 680.0), rng.uniform(-680.0, 680.0))
        for _index in range(10)
    ]

    def append(species_index: int, x: float, y: float) -> None:
        x = max(-945.0, min(945.0, x))
        y = max(-945.0, min(945.0, y))
        lower, upper = VEGETATION_SCALE_RANGES[species_index]
        placements.append(
            (
                species_index,
                x,
                y,
                rng.uniform(lower, upper),
                rng.uniform(0.0, 360.0),
            )
        )

    for _index in range(natural_count):
        if rng.random() < clustered_fraction:
            centre_x, centre_y = rng.choice(cluster_centres)
            x = rng.gauss(centre_x, 125.0)
            y = rng.gauss(centre_y, 125.0)
        else:
            x = rng.uniform(-920.0, 920.0)
            y = rng.uniform(-920.0, 920.0)
        species_index = rng.choices(range(len(VEGETATION_SPECIES)), weights=weights)[
            0
        ]
        append(species_index, x, y)

    orchard_rows = int(profile_config["orchard_rows"])
    orchard_columns = int(profile_config["orchard_columns"])
    if orchard_rows and orchard_columns:
        # A deliberately imperfect orchard block: regular enough to read as
        # agriculture, jittered enough to avoid a synthetic checkerboard.
        origin_x = -720.0
        origin_y = -610.0
        row_spacing = 17.5
        column_spacing = 20.0
        for row in range(orchard_rows):
            for column in range(orchard_columns):
                append(
                    VEGETATION_SPECIES.index("common_apple"),
                    origin_x
                    + column * column_spacing
                    + row * 2.8
                    + rng.uniform(-1.6, 1.6),
                    origin_y
                    + row * row_spacing
                    + rng.uniform(-1.6, 1.6),
                )

    hedgerow_count = int(profile_config["hedgerow_count"])
    for index in range(hedgerow_count):
        along = -875.0 + 1_750.0 * index / max(1, hedgerow_count - 1)
        if index % 2:
            x = along
            y = 355.0 + 22.0 * math.sin(index * 0.19)
        else:
            x = 225.0 + 18.0 * math.sin(index * 0.17)
            y = along
        species_index = (
            VEGETATION_SPECIES.index("lombardy_poplar")
            if index % 5 == 0
            else VEGETATION_SPECIES.index("hawthorn")
        )
        append(species_index, x, y)
    return placements


def _write_site_usda(
    path: Path,
    *,
    site: dict[str, Any],
    mnt: Image.Image,
    texture: Path,
    vegetation_assets: list[Path],
    building_asset: Path,
) -> None:
    if len(vegetation_assets) != len(VEGETATION_SPECIES):
        raise RuntimeError(
            "site authoring requires the six ordered NVIDIA vegetation prototypes"
        )
    grid = TERRAIN_GRID
    mnt_float = mnt.convert("F")
    heights = mnt_float.resize((grid, grid), Image.Resampling.BILINEAR)
    raw = [float(value) for value in heights.getdata()]
    finite = [value for value in raw if math.isfinite(value)]
    if len(finite) != len(raw):
        raise RuntimeError(f"MNT contains non-finite samples for {site['id']}")
    base = sum(finite) / len(finite)
    step = SITE_SPAN_M / (grid - 1)
    points = []
    uvs = []
    for row in range(grid):
        for column in range(grid):
            points.append(
                f"({-SITE_SPAN_M / 2.0 + column * step:.4f}, "
                f"{SITE_SPAN_M / 2.0 - row * step:.4f}, "
                f"{raw[row * grid + column] - base:.4f})"
            )
            uvs.append(f"({column / (grid - 1):.6f}, {1.0 - row / (grid - 1):.6f})")
    indices = []
    counts = []
    for row in range(grid - 1):
        for column in range(grid - 1):
            nw = row * grid + column
            sw = (row + 1) * grid + column
            se = sw + 1
            ne = nw + 1
            indices.extend((nw, sw, se, ne))
            counts.append(4)
    placements = _vegetation_layout(
        site_id=str(site["id"]),
        profile=str(site["profile"]),
    )
    vegetation_positions = []
    vegetation_orientations = []
    vegetation_scales = []
    vegetation_prototype_indices = []
    for species_index, x, y, scale, rotation in placements:
        terrain_z = _mnt_local_elevation(mnt_float, base, x, y)
        radians = math.radians(rotation) / 2.0
        vegetation_positions.append(f"({x:.3f}, {y:.3f}, {terrain_z:.3f})")
        vegetation_orientations.append(
            f"({math.cos(radians):.6f}, 0, 0, {math.sin(radians):.6f})"
        )
        vegetation_scales.append(f"({scale:.3f}, {scale:.3f}, {scale:.3f})")
        vegetation_prototype_indices.append(str(species_index))
    vegetation_prototypes = []
    for index, vegetation_asset in enumerate(vegetation_assets):
        vegetation_ref = (
            os.path.relpath(
                vegetation_asset, path.parent
            )
            .replace("\\", "/")
            .replace("@", "%40")
        )
        vegetation_prototypes.append(
            f'''        def Xform "{VEGETATION_SPECIES[index]}" (\n'''
            f'''            instanceable = true\n'''
            f'''            prepend references = @{vegetation_ref}@\n'''
            f'''        ) {{}}'''
        )
    building_positions = {
        "rural_mountain": (610.0, 520.0, 28.0),
        "rural_agricultural": (510.0, -120.0, -12.0),
        "mountain_agricultural": (580.0, 380.0, 16.0),
    }
    building_x, building_y, building_rotation = building_positions[str(site["profile"])]
    building_z = _mnt_local_elevation(
        mnt_float, base, building_x, building_y
    )
    building_ref = (
        os.path.relpath(building_asset, path.parent)
        .replace("\\", "/")
        .replace("@", "%40")
    )
    texture_rel = texture.name
    content = f'''#usda 1.0
(
    defaultPrim = "Site"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        string fireviewer_landscape_profile = "{site["profile"]}"
        string fireviewer_source = "IGN Geoplateforme EPSG:2154"
        int fireviewer_vegetation_instance_count = {len(placements)}
    }}
)

def Xform "Site" (
    prepend variantSets = "lighting"
    variants = {{ string lighting = "light_0" }}
)
{{
    variantSet "lighting" = {{
        "light_0" {{ def DistantLight "Sun" {{ float intensity = 900 color3f color = (1.0, 0.97, 0.92) float angle = 1.0 }} def DomeLight "Sky" {{ float intensity = 150 color3f color = (0.62, 0.72, 0.88) }} }}
        "light_1" {{ def DistantLight "Moon" {{ float intensity = 40 color3f color = (0.48, 0.56, 0.72) float angle = 2.0 }} def DomeLight "Sky" {{ float intensity = 8 color3f color = (0.025, 0.035, 0.07) }} }}
        "light_2" {{ def DistantLight "Dawn" {{ float intensity = 620 color3f color = (1.0, 0.72, 0.48) float angle = 1.4 }} def DomeLight "Sky" {{ float intensity = 95 color3f color = (0.42, 0.46, 0.62) }} }}
        "light_3" {{ def DistantLight "Dusk" {{ float intensity = 520 color3f color = (1.0, 0.66, 0.42) float angle = 1.6 }} def DomeLight "Sky" {{ float intensity = 70 color3f color = (0.28, 0.25, 0.38) }} }}
    }}
    def Scope "Looks" {{
        def Material "Terrain" {{
            token outputs:surface.connect = </Site/Looks/Terrain/Surface.outputs:surface>
            def Shader "Primvar" {{ uniform token info:id = "UsdPrimvarReader_float2" token inputs:varname = "st" float2 outputs:result }}
            def Shader "Texture" {{ uniform token info:id = "UsdUVTexture" asset inputs:file = @{texture_rel}@ token inputs:sourceColorSpace = "sRGB" float2 inputs:st.connect = </Site/Looks/Terrain/Primvar.outputs:result> float3 outputs:rgb }}
            def Shader "Surface" {{ uniform token info:id = "UsdPreviewSurface" color3f inputs:diffuseColor.connect = </Site/Looks/Terrain/Texture.outputs:rgb> float inputs:roughness = 0.82 token outputs:surface }}
        }}
    }}
    def Mesh "Terrain" {{
        int[] faceVertexCounts = [{", ".join(map(str, counts))}]
        int[] faceVertexIndices = [{", ".join(map(str, indices))}]
        point3f[] points = [{", ".join(points)}]
        texCoord2f[] primvars:st = [{", ".join(uvs)}] ( interpolation = "vertex" )
        uniform token subdivisionScheme = "none"
        rel material:binding = </Site/Looks/Terrain>
    }}
    def Scope "VegetationPrototypes" {{
{chr(10).join(vegetation_prototypes)}
    }}
    def PointInstancer "Vegetation" {{
        rel prototypes = [
            {", ".join(f"</Site/VegetationPrototypes/{species}>" for species in VEGETATION_SPECIES)}
        ]
        int[] protoIndices = [{", ".join(vegetation_prototype_indices)}]
        point3f[] positions = [{", ".join(vegetation_positions)}]
        quath[] orientations = [{", ".join(vegetation_orientations)}]
        float3[] scales = [{", ".join(vegetation_scales)}]
    }}
    def Xform "Occluders" {{
        def Xform "FarmBuilding" (
            prepend references = @{building_ref}@
        ) {{
            double3 xformOp:translate = ({building_x:.3f}, {building_y:.3f}, {building_z:.3f})
            double3 xformOp:rotateXYZ = (0, 0, {building_rotation:.3f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        }}
    }}
}}
'''
    _write_text(path, content)


def _usd_heightfield_raycast(
    scene: Path,
    *,
    start: list[float],
    target: list[float],
) -> dict[str, Any]:
    """Test line-of-sight against the exact authored USD terrain mesh."""

    try:
        import isaacsim  # noqa: F401 - exposes bundled pxr
        from pxr import Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError("USD terrain raycast requires the pinned Isaac runtime") from exc
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"site scene could not be opened for raycast: {scene}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Site/Terrain"))
    if not mesh:
        raise RuntimeError(f"site scene has no /Site/Terrain mesh: {scene}")
    points = list(mesh.GetPointsAttr().Get() or [])
    grid = int(round(math.sqrt(len(points))))
    if grid < 2 or grid * grid != len(points):
        raise RuntimeError("USD terrain raycast requires a regular heightfield mesh")

    def terrain_z(x: float, y: float) -> float:
        half = SITE_SPAN_M / 2.0
        column = (x + half) * (grid - 1) / SITE_SPAN_M
        row = (half - y) * (grid - 1) / SITE_SPAN_M
        if not 0.0 <= column <= grid - 1 or not 0.0 <= row <= grid - 1:
            raise RuntimeError("USD terrain raycast escaped the authored mesh")
        left = int(math.floor(column))
        top = int(math.floor(row))
        right = min(grid - 1, left + 1)
        bottom = min(grid - 1, top + 1)
        dx = column - left
        dy = row - top
        samples = [
            float(points[top * grid + left][2]),
            float(points[top * grid + right][2]),
            float(points[bottom * grid + left][2]),
            float(points[bottom * grid + right][2]),
        ]
        return (
            samples[0] * (1.0 - dx) * (1.0 - dy)
            + samples[1] * dx * (1.0 - dy)
            + samples[2] * (1.0 - dx) * dy
            + samples[3] * dx * dy
        )

    minimum_clearance = math.inf
    closest_sample = 0
    sample_count = 256
    for index in range(1, sample_count):
        ratio = index / sample_count
        x = start[0] + (target[0] - start[0]) * ratio
        y = start[1] + (target[1] - start[1]) * ratio
        z = start[2] + (target[2] - start[2]) * ratio
        clearance = z - terrain_z(x, y)
        if clearance < minimum_clearance:
            minimum_clearance = clearance
            closest_sample = index
    return {
        "validation": (
            "usd_heightfield_line_of_sight_passed"
            if minimum_clearance >= 0.35
            else "usd_heightfield_occluded"
        ),
        "terrain_prim_path": "/Site/Terrain",
        "terrain_mesh_point_count": len(points),
        "sample_count": sample_count - 1,
        "minimum_clearance_m": round(minimum_clearance, 6),
        "closest_sample": closest_sample,
        "passed": minimum_clearance >= 0.35,
    }


def _validate_site_references(
    sites: list[dict[str, Any]],
    *,
    volume_root: Path,
) -> list[dict[str, Any]]:
    """Render one measured RTX reference and calibrate a clear ray per site."""

    try:
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "site reference validation requires active Isaac and Replicator"
        ) from exc
    from fireviewer_sdg.case_generation import (
        _frame_quality_failure,
        _frame_quality_metrics,
    )

    application = omni.kit.app.get_app()
    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("Isaac has no active stage for site reference validation")
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Cameras", "Xform")
    camera = rep.create.camera(
        position=(0.0, -180.0, 12.0),
        look_at=(0.0, 0.0, 4.0),
        focal_length=24.0,
        clipping_range=(0.1, 100000.0),
        parent="/World/Cameras",
        name="SetupReferenceCamera",
    )
    product = rep.create.render_product(
        camera,
        (LOCAL_RENDER_WIDTH, LOCAL_RENDER_HEIGHT),
        name="SetupReferenceRenderProduct",
    )
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    profile_azimuth = {
        "rural_mountain": 225.0,
        "rural_agricultural": 135.0,
        "mountain_agricultural": 205.0,
    }
    validations: list[dict[str, Any]] = []
    try:
        for site_index, site in enumerate(sites, start=1):
            write_progress(
                volume_root,
                phase="site_reference_validation",
                message=(
                    f"Raycast USD et rendu RTX du site {site_index}/{len(sites)} : "
                    f"{site['label']}."
                ),
                current_site=site["label"],
                sites_completed=site_index - 1,
                sites_total=len(sites),
            )
            stage.RemovePrim("/World/ReferenceScene")
            reference = stage.DefinePrim("/World/ReferenceScene", "Xform")
            reference.GetReferences().AddReference(str(site["scene"]))
            for _update in range(24):
                application.update()
            azimuth = math.radians(profile_azimuth[str(site["profile"])])
            distance = 180.0
            camera_x = math.cos(azimuth) * distance
            camera_y = math.sin(azimuth) * distance
            target = [
                0.0,
                0.0,
                _terrain_elevation(site, 0.0, 0.0) + 5.0,
            ]
            raycast: dict[str, Any] | None = None
            position: list[float] = []
            for camera_height in (3.0, 6.0, 12.0, 24.0):
                position = [
                    camera_x,
                    camera_y,
                    _terrain_elevation(site, camera_x, camera_y)
                    + camera_height,
                ]
                raycast = _usd_heightfield_raycast(
                    site["scene"],
                    start=position,
                    target=target,
                )
                if raycast["passed"]:
                    break
            if raycast is None or not raycast["passed"]:
                raise RuntimeError(
                    f"no clear USD reference ray could be calibrated for {site['id']}"
                )
            camera_data = camera_contract(
                position=position,
                look_at=target,
                width=LOCAL_RENDER_WIDTH,
                height=LOCAL_RENDER_HEIGHT,
                focal_length_mm=24.0,
                horizontal_aperture_mm=20.955,
            )
            anchors = {
                "active_fire_point": [
                    0.0,
                    0.0,
                    _terrain_elevation(site, 0.0, 0.0) + 1.2,
                ],
                "visible_fire_front_point": [
                    8.0,
                    4.0,
                    _terrain_elevation(site, 8.0, 4.0) + 0.8,
                ],
                "smoke_column_base": [
                    1.0,
                    -1.0,
                    _terrain_elevation(site, 1.0, -1.0) + 1.0,
                ],
            }
            projected = {
                label: project_point(world, camera_data)
                for label, world in anchors.items()
            }
            assert_visible(projected.values(), margin=0.03)
            rep.functional.modify.pose(
                camera,
                position_value=tuple(position),
                look_at_value=tuple(target),
                look_at_up_axis=(0.0, 0.0, 1.0),
                write_to_usd=True,
            )
            for _update in range(16):
                application.update()
            rep.orchestrator.step(delta_time=0.0, rt_subframes=16)
            data = rgb.get_data()
            metrics = _frame_quality_metrics(data)
            failure = _frame_quality_failure(metrics)
            if failure is not None:
                raise RuntimeError(
                    f"site reference render failed {failure}: {site['id']}"
                )
            import numpy as np

            array = np.asarray(data)
            image_path = site["root"] / "setup-reference-720p.jpg"
            Image.fromarray(array[:, :, :3].astype("uint8"), mode="RGB").save(
                image_path,
                format="JPEG",
                quality=97,
                subsampling=0,
                optimize=True,
            )
            result = {
                "site_id": site["id"],
                "label": site["label"],
                "profile": site["profile"],
                "scene": site["scene"].relative_to(volume_root).as_posix(),
                "scene_sha256": _sha256(site["scene"]),
                "orthophoto": site["ortho"].relative_to(volume_root).as_posix(),
                "orthophoto_sha256": _sha256(site["ortho"]),
                "mnt": site["mnt"].relative_to(volume_root).as_posix(),
                "mnt_sha256": _sha256(site["mnt"]),
                "reference_image": image_path.relative_to(volume_root).as_posix(),
                "reference_image_sha256": _sha256(image_path),
                "reference_image_bytes": image_path.stat().st_size,
                "resolution": [LOCAL_RENDER_WIDTH, LOCAL_RENDER_HEIGHT],
                "camera": camera_data,
                "projected_reference_anchors": projected,
                "calibration_validation": "calibrated_project_to_image_passed",
                "raycast": raycast,
                "frame_quality": metrics,
                "automatic_validation": "passed",
                "human_visual_review": "pending_console_review",
            }
            validations.append(result)
            site["reference_validation"] = result
            write_progress(
                volume_root,
                phase="site_reference_validation",
                message=f"Référence RTX validée pour {site['label']}.",
                current_site=site["label"],
                sites_completed=site_index,
                sites_total=len(sites),
            )
    finally:
        rgb.detach([product])
        product.destroy()
        stage.RemovePrim("/World/ReferenceScene")
    catalog = volume_root / "input" / "site-reference-catalog.json"
    _write_json(
        catalog,
        {
            "schema_version": 1,
            "validation": "three_site_rtx_reference_gate_passed",
            "sites": validations,
        },
    )
    return validations


def _discover_flow_fire_preset(runtime_root: Path | None = None) -> Path:
    root = (runtime_root or Path(sys.prefix)).resolve()
    extension_roots: list[Path] = []
    for candidate in root.glob("lib/python*/site-packages/isaacsim/extscache/*flow*"):
        if candidate.is_dir():
            extension_roots.append(candidate)
    candidates: list[tuple[int, Path]] = []
    for extension in extension_roots:
        for pattern in ("*.usd", "*.usda", "*.usdc"):
            for path in extension.rglob(pattern):
                lower = path.as_posix().lower()
                score = 0
                if "fire" in lower:
                    score += 100
                if "preset" in lower:
                    score += 50
                if "sample" in lower or "demo" in lower:
                    score += 10
                if "thumbnail" in lower:
                    score -= 100
                if score > 0:
                    candidates.append((score, path))
    if not candidates:
        raise RuntimeError("no NVIDIA Flow fire preset was found in the pinned Isaac runtime")
    return max(candidates, key=lambda item: (item[0], -len(item[1].as_posix())))[1]


def _write_flow_wrapper(path: Path, preset: Path) -> None:
    preset_value = preset.as_posix().replace("@", "%40")
    _write_text(
        path,
        f'''#usda 1.0
(
    defaultPrim = "OfficialNvidiaFlowFire"
    customLayerData = {{
        string fireviewer_flow_preset_sha256 = "{_sha256(preset)}"
        string fireviewer_flow_preset_source = "pinned Isaac Sim runtime"
    }}
)
def Xform "OfficialNvidiaFlowFire" (
    prepend payload = @{preset_value}@
) {{}}
''',
    )


def _validate_usd_assets(paths: list[Path]) -> dict[str, Any]:
    try:
        import isaacsim  # noqa: F401 - exposes the bundled pxr modules
        from pxr import Gf, Sdf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - Isaac runtime gate
        raise RuntimeError(
            "the pinned Isaac runtime must expose pxr.Usd for prepared asset validation"
        ) from exc
    validated: list[str] = []
    quality: dict[str, dict[str, Any]] = {}
    for path in paths:
        stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"USD stage could not be opened: {path}")
        default_prim = stage.GetDefaultPrim()
        if not default_prim or not default_prim.IsValid():
            raise RuntimeError(f"USD stage has no valid default prim: {path}")
        primitive_count = 0
        mesh_points = 0
        material_count = 0
        texture_assets: set[str] = set()
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        stage_up_axis = str(UsdGeom.GetStageUpAxis(stage))
        bounds_min = [math.inf, math.inf, math.inf]
        bounds_max = [-math.inf, -math.inf, -math.inf]
        for prim in stage.Traverse():
            type_name = str(prim.GetTypeName())
            if type_name in {"Cube", "Cylinder", "Sphere", "Cone", "Capsule"}:
                primitive_count += 1
            if type_name == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                if points is not None:
                    mesh_points += len(points)
                if points:
                    extent = mesh.GetExtentAttr().Get()
                    if extent is None:
                        extent = [
                            Gf.Vec3f(
                                *[
                                    min(
                                        float(point[axis])
                                        for point in points
                                    )
                                    for axis in range(3)
                                ]
                            ),
                            Gf.Vec3f(
                                *[
                                    max(
                                        float(point[axis])
                                        for point in points
                                    )
                                    for axis in range(3)
                                ]
                            ),
                        ]
                    matrix = xform_cache.GetLocalToWorldTransform(prim)
                    for x in (float(extent[0][0]), float(extent[1][0])):
                        for y in (float(extent[0][1]), float(extent[1][1])):
                            for z in (float(extent[0][2]), float(extent[1][2])):
                                world = matrix.Transform(Gf.Vec3d(x, y, z))
                                for axis in range(3):
                                    value = (
                                        float(world[axis])
                                        * stage_meters_per_unit
                                    )
                                    bounds_min[axis] = min(
                                        bounds_min[axis], value
                                    )
                                    bounds_max[axis] = max(
                                        bounds_max[axis], value
                                    )
            if type_name == "Material":
                material_count += 1
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset:
                    value = attribute.Get()
                    asset_path = str(getattr(value, "path", "")).strip()
                    if asset_path:
                        texture_assets.add(asset_path)
                elif attribute.GetTypeName() == Sdf.ValueTypeNames.AssetArray:
                    values = attribute.Get() or []
                    for value in values:
                        asset_path = str(getattr(value, "path", "")).strip()
                        if asset_path:
                            texture_assets.add(asset_path)
        minimum = bounds_min
        maximum = bounds_max
        resolved = str(path.resolve())
        validated.append(resolved)
        quality[resolved] = {
            "primitive_geometry_count": primitive_count,
            "mesh_point_count": mesh_points,
            "material_count": material_count,
            "texture_asset_count": len(texture_assets),
            "meters_per_unit": stage_meters_per_unit,
            "up_axis": stage_up_axis,
            "aabb_min_m": minimum,
            "aabb_max_m": maximum,
        }
    return {"count": len(validated), "assets": validated, "quality": quality}


def _assert_asset_quality(
    report: dict[str, Any],
    *,
    entry: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    thresholds = ASSET_QUALITY_THRESHOLDS[family]
    quality = report.get("quality")
    if not isinstance(quality, dict):
        raise RuntimeError("USD validator returned no per-asset quality report")
    metrics = quality.get(str(entry["path"].resolve()))
    if not isinstance(metrics, dict):
        raise RuntimeError(f"USD quality report is missing {entry['role']}")
    if int(metrics.get("primitive_geometry_count", -1)) != 0:
        raise RuntimeError(
            f"{entry['role']} contains Cube/Cylinder/Sphere-style placeholder geometry"
        )
    for metric, threshold_key in (
        ("mesh_point_count", "mesh_points"),
        ("material_count", "materials"),
        ("texture_asset_count", "textures"),
    ):
        minimum = thresholds[threshold_key]
        if int(metrics.get(metric, -1)) < minimum:
            raise RuntimeError(
                f"{entry['role']} fails photoreal quality gate: "
                f"{metric}={metrics.get(metric)!r}, required>={minimum}"
            )
    meters_per_unit = metrics.get("meters_per_unit")
    if meters_per_unit is not None and not math.isclose(
        float(meters_per_unit), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            f"{entry['role']} is not assembled in meter units: "
            f"metersPerUnit={meters_per_unit!r}"
        )
    up_axis = metrics.get("up_axis")
    if up_axis is not None and str(up_axis) != "Z":
        raise RuntimeError(
            f"{entry['role']} is not assembled Z-up: upAxis={up_axis!r}"
        )
    minimum = metrics.get("aabb_min_m")
    maximum = metrics.get("aabb_max_m")
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 3
        or len(maximum) != 3
        or any(
            not math.isfinite(float(minimum[axis]))
            or not math.isfinite(float(maximum[axis]))
            or float(minimum[axis]) >= float(maximum[axis])
            for axis in range(3)
        )
    ):
        raise RuntimeError(f"{entry['role']} has no valid metric USD bounds")
    dimensions = [
        float(maximum[axis]) - float(minimum[axis])
        for axis in range(3)
    ]
    if not (
        thresholds["minimum_height_m"]
        <= dimensions[2]
        <= thresholds["maximum_height_m"]
    ):
        raise RuntimeError(
            f"{entry['role']} has implausible metric height: "
            f"{dimensions[2]:.3f} m"
        )
    if max(dimensions[0], dimensions[1]) > thresholds[
        "maximum_horizontal_span_m"
    ]:
        raise RuntimeError(
            f"{entry['role']} has implausible horizontal span: "
            f"{max(dimensions[0], dimensions[1]):.3f} m"
        )
    return {
        **metrics,
        "dimensions_m": dimensions,
        "aabb_min_m": [float(value) for value in minimum],
        "aabb_max_m": [float(value) for value in maximum],
    }


def _event_contract(
    *,
    index: int,
    event_root: Path,
    site: dict[str, Any],
    flow_asset: Path,
    actor_assets: dict[str, dict[str, Any]],
    asset_manifest_sha256: str,
    response_engagement_in_scope: bool = True,
) -> dict[str, Any]:
    event_id = f"fire-fr-{index:04d}"
    duration = index % 15 + 1
    rng = random.Random(0xF17E0000 + index)
    site_ordinal = index // len(SITES) + 1
    fire_x = (_halton(site_ordinal, 2) - 0.5) * 420.0 + rng.uniform(-4.0, 4.0)
    fire_y = (_halton(site_ordinal, 3) - 0.5) * 420.0 + rng.uniform(-4.0, 4.0)
    fire_z = _terrain_elevation(site, fire_x, fire_y) + 1.2
    azimuth_base = (index % 12) * 30.0
    # The farthest pilot view must remain operationally distant while still
    # preserving a trainable minimum edge for the smallest exact response actor.
    distances = (45.0, 135.0, 220.0, 330.0)
    bands = ("near", "medium", "far", "very_far")
    occlusions = ("clear", "partial_building", "partial_mountain", "clear")
    poses = []
    for pose_index, (distance, band, occlusion) in enumerate(
        zip(distances, bands, occlusions, strict=True)
    ):
        azimuth = (azimuth_base + pose_index * 90.0) % 360.0
        radians = math.radians(azimuth)
        camera_x = fire_x + math.cos(radians) * distance
        camera_y = fire_y + math.sin(radians) * distance
        position = [
            camera_x,
            camera_y,
            _terrain_elevation(site, camera_x, camera_y) + 2.1,
        ]
        poses.append(
            {
                "id": f"pose-{pose_index:02d}",
                "position": [round(value, 6) for value in position],
                "look_at": [round(fire_x, 6), round(fire_y, 6), round(fire_z + 5.0, 6)],
                "validation": "pending_console_review",
                "viewpoint": {
                    "distance_band": band,
                    "occlusion": occlusion,
                    "azimuth_deg": azimuth,
                    "elevation_deg": round(
                        math.degrees(math.atan2(fire_z + 5.0 - position[2], distance)), 4
                    ),
                    "occlusion_fraction": (0.0, 0.22, 0.36, 0.0)[pose_index],
                    "occluder_prim_path": (
                        ""
                        if occlusion == "clear"
                        else (
                            "/World/RealWorldScene/Occluders/FarmBuilding"
                            if pose_index == 1
                            else "/World/RealWorldScene/Terrain"
                        )
                    ),
                    "line_of_sight_validation": "pending_console_review",
                    "required_anchors_visible": "pending_console_review",
                    "reference_validation": "pending_console_review",
                },
            }
        )
    profile = PROGRESSION_PROFILES[index % len(PROGRESSION_PROFILES)]
    times = ("day", "night", "dawn", "dusk")
    day_slots = (
        1,
        max(1, math.ceil(duration / 3)),
        max(1, math.ceil(duration * 2 / 3)),
        duration,
    )
    flow_states = []
    burned = rng.uniform(180.0, 900.0)
    spread_heading_deg = rng.uniform(0.0, 360.0)
    spread_heading = math.radians(spread_heading_deg)
    wind_heading_deg = (spread_heading_deg + rng.uniform(-65.0, 65.0)) % 360.0
    wind_heading = math.radians(wind_heading_deg)
    wind_speed_mps = rng.uniform(1.2, 12.5)
    spread_step_m = rng.uniform(1.6, 2.8)
    time_cursor = rng.uniform(2.0, 7.0)
    intensity_factor = rng.uniform(0.72, 1.65)
    for state_index, phase in enumerate(profile):
        time_cursor += rng.uniform(5.0, 12.0) * (1.0 + state_index * 0.08)
        burned += rng.uniform(90.0, 480.0) * (state_index + 1)
        fronts = [f"front-{index:04d}-a"]
        if phase in {"front_split", "multi_front_spread"}:
            fronts.append(f"front-{index:04d}-b")
        front_distance = 4.0 + spread_step_m * (state_index + 1)
        lateral = (
            rng.choice((-1.0, 1.0)) * rng.uniform(1.5, 4.5)
            if phase in {"front_split", "multi_front_spread", "reignition"}
            else rng.uniform(-1.5, 1.5)
        )
        front_x = (
            fire_x
            + math.cos(spread_heading) * front_distance
            - math.sin(spread_heading) * lateral
        )
        front_y = (
            fire_y
            + math.sin(spread_heading) * front_distance
            + math.cos(spread_heading) * lateral
        )
        smoke_offset = 0.8 + wind_speed_mps * 0.18
        smoke_x = fire_x + math.cos(wind_heading) * smoke_offset
        smoke_y = fire_y + math.sin(wind_heading) * smoke_offset
        active_flame_area = (
            rng.uniform(45.0, 210.0)
            * (1.0 + state_index * 0.2)
            * intensity_factor
        )
        flow_states.append(
            {
                "id": f"flow-{state_index}",
                "time_seconds": round(time_cursor, 6),
                "event_day": day_slots[state_index],
                "lighting_variant_id": f"light-{state_index}",
                "anchors_world_m": {
                    "active_fire_point": [fire_x, fire_y, fire_z],
                    "visible_fire_front_point": [
                        front_x,
                        front_y,
                        _terrain_elevation(site, front_x, front_y) + 0.8,
                    ],
                    "smoke_column_base": [
                        smoke_x,
                        smoke_y,
                        _terrain_elevation(site, smoke_x, smoke_y) + 1.0,
                    ],
                },
                "progression": {
                    "phase": phase,
                    "front_ids": fronts,
                    "parent_front_ids": (
                        [f"front-{index:04d}-a"] if phase == "front_split" else []
                    ),
                    "advancing_zone_ids": (
                        [f"zone-{index:04d}-advance"]
                        if phase == "advancing_flame_zone"
                        else []
                    ),
                    "reignited_zone_ids": (
                        [f"zone-{index:04d}-reignite"] if phase == "reignition" else []
                    ),
                    "burned_area_m2": round(burned, 3),
                    "active_flame_area_m2": round(active_flame_area, 3),
                    "spread_heading_deg": round(spread_heading_deg, 4),
                    "spread_step_m": round(spread_step_m, 4),
                    "wind_heading_deg": round(wind_heading_deg, 4),
                    "wind_speed_mps": round(wind_speed_mps, 4),
                },
                "validation": "pending_console_review",
            }
        )
    all_anchor_points = [
        point
        for state in flow_states
        for point in state["anchors_world_m"].values()
    ]
    anchor_center = [
        (min(point[axis] for point in all_anchor_points) + max(point[axis] for point in all_anchor_points))
        / 2.0
        for axis in range(3)
    ]
    for pose in poses:
        viewpoint = pose["viewpoint"]
        azimuth = math.radians(float(viewpoint["azimuth_deg"]))
        initial_distance = math.dist(pose["position"][:2], [fire_x, fire_y])
        distance = initial_distance
        maximum_distance = MAX_FRAMING_DISTANCE_M[str(viewpoint["distance_band"])]
        projected: list[dict[str, float]] = []
        while True:
            camera_x = fire_x + math.cos(azimuth) * distance
            camera_y = fire_y + math.sin(azimuth) * distance
            position = [
                camera_x,
                camera_y,
                _terrain_elevation(site, camera_x, camera_y) + 2.1,
            ]
            pose["position"] = [round(value, 6) for value in position]
            pose["look_at"] = [round(value, 6) for value in anchor_center]
            calibrated = camera_contract(
                position=pose["position"],
                look_at=pose["look_at"],
                width=RENDER_WIDTH,
                height=RENDER_HEIGHT,
            )
            try:
                projected = [
                    project_point(point, calibrated) for point in all_anchor_points
                ]
                assert_visible(projected, margin=0.03)
                break
            except (RuntimeError, ValueError) as exc:
                if distance >= maximum_distance:
                    raise RuntimeError(
                        f"{event_id}/{pose['id']} cannot frame every fire anchor "
                        f"inside the {viewpoint['distance_band']} review image"
                    ) from exc
                distance = min(maximum_distance, distance * 1.2)
        viewpoint["distance_m"] = round(distance, 6)
        viewpoint["framing_adjustment_m"] = round(distance - initial_distance, 6)
        viewpoint["elevation_deg"] = round(
            math.degrees(
                math.atan2(anchor_center[2] - pose["position"][2], distance)
            ),
            4,
        )
        pose["validation"] = "calibrated_project_to_image_passed"
        viewpoint["required_anchors_visible"] = True
    manifest_path = event_root / "capture-manifest.json"
    render_slots = [
        {
            "id": f"render-slot-{number:02d}",
            "kind": "required_reference_render_slot",
            "resolution": [RENDER_WIDTH, RENDER_HEIGHT],
            "status": "pending_console_review",
        }
        for number in range(12)
    ]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "site_id": site["id"],
            "event_id": event_id,
            "provenance": "official_ign_terrain_plus_new_omniverse_fire",
            "terrain_sources": [
                {
                    "kind": "official_ign_orthophoto",
                    "path": site["ortho"].relative_to(event_root.parents[2]).as_posix(),
                    "sha256": _sha256(site["ortho"]),
                    "crs": "EPSG:2154",
                },
                {
                    "kind": "official_ign_lidar_hd_mnt",
                    "path": site["mnt"].relative_to(event_root.parents[2]).as_posix(),
                    "sha256": _sha256(site["mnt"]),
                    "crs": "EPSG:2154",
                },
            ],
            "site_reference_validation": site["reference_validation"],
            "reference_render_slots": render_slots,
            "human_review_required": True,
        },
    )
    center_x = (site["bounds"][0] + site["bounds"][2]) / 2.0
    center_y = (site["bounds"][1] + site["bounds"][3]) / 2.0
    actors = []
    positions = (
        (16.0, -12.0, 1.7),
        (28.0, 14.0, 48.0),
        (-34.0, 18.0, 58.0),
        (8.0, 31.0, 42.0),
        (-18.0, -25.0, 1.7),
        (35.0, -28.0, 44.0),
        (-31.0, 32.0, 39.0),
    )
    actor_classes = ACTOR_CLASSES if response_engagement_in_scope else ()
    for class_id, offset in zip(actor_classes, positions, strict=False):
        actor_x = fire_x + offset[0]
        actor_y = fire_y + offset[1]
        actor_z = (
            _terrain_elevation(site, actor_x, actor_y) + offset[2]
            if class_id in {"sdis_vehicle", "hard_negative_construction_truck"}
            else fire_z + offset[2]
        )
        position = [actor_x, actor_y, actor_z]
        asset_entry = actor_assets[class_id]
        asset_quality = asset_entry["usd_quality"]
        local_minimum = asset_quality["aabb_min_m"]
        local_maximum = asset_quality["aabb_max_m"]
        local_center = [
            (local_minimum[axis] + local_maximum[axis]) / 2.0
            for axis in range(3)
        ]
        extent = [
            (local_maximum[axis] - local_minimum[axis]) / 2.0
            for axis in range(3)
        ]
        translation = [
            position[axis] - local_center[axis] for axis in range(3)
        ]
        actors.append(
            {
                "class_id": class_id,
                "asset": os.path.relpath(asset_entry["path"], event_root),
                "asset_sha256": asset_entry["sha256"],
                "asset_manifest_sha256": asset_manifest_sha256,
                "asset_provenance": asset_entry["provenance"],
                "asset_source_uri": asset_entry["source_uri"],
                "asset_license_id": asset_entry["license_id"],
                "asset_quality_metrics": asset_quality,
                "center_world_m": position,
                "aabb_min_world_m": [
                    position[axis] - extent[axis] for axis in range(3)
                ],
                "aabb_max_world_m": [
                    position[axis] + extent[axis] for axis in range(3)
                ],
                "translation_world_m": translation,
                "rotation_xyz_deg": [0.0, 0.0, float((index * 17) % 360)],
                "scale_xyz": [1.0, 1.0, 1.0],
                "camera_pose_ids": [pose["id"] for pose in poses],
                "engagement_context": (
                    "hard_negative_not_engaged"
                    if class_id.startswith("hard_negative")
                    else "wildfire_response_engaged"
                ),
                "identity_validation": "pending_console_review",
                "engagement_validation": "pending_console_review",
                "quality_validation": asset_entry["quality_validation"],
                "placement_validation": asset_entry["placement_validation"],
            }
        )
    return {
        "schema_version": 1,
        "pipeline": "nvidia_omniverse_simready_flow",
        "render_profile": RENDER_PROFILE,
        "site_id": site["id"],
        "event_id": event_id,
        "duration_days": duration,
        "scope": {
            "response_engagement": response_engagement_in_scope,
            "humans": False,
        },
        "capture": {
            "source": "new_synthetic_french_reference",
            "capture_manifest": manifest_path.name,
            "capture_manifest_sha256": _sha256(manifest_path),
            "reference_render_slot_count": 12,
            "minimum_source_resolution": [ORTHOPHOTO_PIXELS, ORTHOPHOTO_PIXELS],
            "terrain_scale_validated": True,
            "materials_validation": "pending_console_review",
            "orthophoto_mnt_coherence_validated": True,
            "site_reference_image": os.path.relpath(
                event_root.parents[3]
                / site["reference_validation"]["reference_image"],
                event_root,
            ),
            "site_reference_image_sha256": site["reference_validation"][
                "reference_image_sha256"
            ],
            "site_reference_raycast": site["reference_validation"]["raycast"],
            "coordinate_convention": "usd_z_up_meters_lambert93",
        },
        "reconstruction": {
            "trainer": "fireviewer/omniverse_usd_terrain",
            "format": "review_gated_usd",
            "asset": os.path.relpath(site["scene"], event_root),
            "asset_sha256": _sha256(site["scene"]),
            "metrics": {"quality_review": "pending_console_review"},
        },
        "composition": {
            "flow_asset": os.path.relpath(flow_asset, event_root),
            "flow_asset_sha256": _sha256(flow_asset),
            "flow_validation": {
                "preset_rendered_and_anchor_verified": "pending_console_review",
                "simulated_frame_count": 4,
                "preset_source": "pinned_nvidia_flow_runtime",
            },
            "camera_poses": poses,
            "lighting_variants": [
                {
                    "id": f"light-{number}",
                    "prim_path": "/World/RealWorldScene",
                    "variant_set": "lighting",
                    "selection": f"light_{number}",
                    "time_of_day": times[number],
                    "validation": "pending_console_review",
                }
                for number in range(4)
            ],
            "flow_states": flow_states,
            "diversity": {
                "selector": "operational_viewpoint_progression_v1",
                "capacity_per_category": 16,
            },
            "actors": actors,
        },
        "geospatial": {
            "crs": "EPSG:2154",
            "country_profile": "FR",
            "landscape_origin": "synthetic_french_reference",
            "landscape_profile": site["profile"],
            "site_context_validation": "pending_console_review",
            "real_world_claim": False,
            "world_axes_aligned_lambert93": True,
            "world_origin_lambert93_m": [center_x, center_y, site["base_elevation"]],
            "orthophoto": os.path.relpath(site["ortho"], event_root),
            "orthophoto_sha256": _sha256(site["ortho"]),
            "mnt": os.path.relpath(site["mnt"], event_root),
            "mnt_sha256": _sha256(site["mnt"]),
            "mnt_preview": os.path.relpath(site["mnt_preview"], event_root),
            "mnt_preview_sha256": _sha256(site["mnt_preview"]),
        },
    }


def prepare_ign_catalog(
    volume_root: Path,
    *,
    runtime_root: Path | None = None,
    asset_manifest: Path | None = None,
    nvidia_asset_root: str = DEFAULT_NVIDIA_ASSET_ROOT,
    include_response_assets: bool = True,
    fetcher: Callable[[str, Path], None] = _download,
    usd_validator: Callable[[list[Path]], dict[str, Any]] = _validate_usd_assets,
    site_reference_validator: Callable[..., list[dict[str, Any]]] = (
        _validate_site_references
    ),
    asset_provisioner: Callable[..., dict[str, Any]] = (
        provision_official_nvidia_manifest
    ),
) -> dict[str, Any]:
    """Prepare reviewed environments first, then effects and response actors."""
    volume = volume_root.resolve()
    catalog = volume / "input" / CATALOG_NAME
    if catalog.is_file():
        write_progress(
            volume,
            phase="catalog_prepared",
            state="completed",
            message="Catalogue pilote existant vérifié; aucune régénération lancée.",
            sites_completed=len(SITES),
            sites_total=len(SITES),
            contracts_completed=512,
            contracts_total=512,
        )
        existing: dict[str, Any] = {
            "state": "existing",
            "catalog": str(catalog),
            "production_scope": "pilot_setup_proof",
            "bulk_allowed": False,
        }
        readiness_path = volume / "input" / READINESS_REPORT_NAME
        if readiness_path.is_file():
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            existing["site_count"] = int(
                readiness.get("environment", {}).get("site_count", len(SITES))
            )
        return existing
    root = volume / "input" / PREPARATION_VERSION
    root.mkdir(parents=True, exist_ok=True)
    write_progress(
        volume,
        phase="asset_manifest",
        message="Chargement du lockfile d'assets et contrôle de provenance.",
        sites_completed=0,
        sites_total=len(SITES),
        contracts_completed=0,
        contracts_total=512,
    )
    manifest_path = asset_manifest or volume / "input" / ASSET_MANIFEST_NAME
    discovery: dict[str, Any] = {
        "mode": "reviewed_manifest_override"
        if asset_manifest is not None
        else "existing_official_nvidia_lock"
    }
    if asset_manifest is None:
        if not manifest_path.is_file():
            print(
                "fireviewer inputs: discovering pinned official NVIDIA USD assets",
                flush=True,
            )
            discovery = {
                "mode": "official_nvidia_auto_discovery",
                **asset_provisioner(
                    volume_root=volume,
                    manifest_path=manifest_path,
                    asset_root=nvidia_asset_root,
                ),
            }
        else:
            locked_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            locked_discovery = locked_manifest.get("discovery", {})
            if not isinstance(locked_discovery, dict):
                raise ValueError(
                    "official NVIDIA asset lock has no discovery contract"
                )
            discovery = {
                "mode": "existing_official_nvidia_lock",
                "manifest": manifest_path,
                "missing_environment": list(
                    locked_discovery.get("missing_environment", [])
                ),
                "missing_actor_classes": list(
                    locked_discovery.get("missing_actor_classes", [])
                ),
            }
        missing_environment = list(discovery.get("missing_environment", []))
        if missing_environment:
            readiness = {
                "schema_version": 1,
                "state": "blocked",
                "phase": "environment_assets",
                "reason": (
                    "official NVIDIA discovery did not find enough exact, "
                    "reviewable rural environment assets"
                ),
                "missing_environment": missing_environment,
                "missing_actor_classes": list(
                    discovery.get("missing_actor_classes", [])
                ),
                "asset_discovery": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in discovery.items()
                },
                "catalog_written": False,
                "synthetic_cases_written": 0,
                "production_scope": "pilot_setup_proof",
                "bulk_allowed": False,
            }
            readiness_path = volume / "input" / READINESS_REPORT_NAME
            _write_json(readiness_path, readiness)
            return {
                **readiness,
                "readiness_report": str(readiness_path),
            }
    simready_assets = _load_simready_asset_manifest(
        manifest_path,
        volume_root=volume,
    )
    print(
        "fireviewer inputs: preparing three IGN terrain-backed environments",
        flush=True,
    )
    vegetation_paths = [
        entry["path"] for entry in simready_assets["vegetation"]
    ]
    building_path = simready_assets["rural_building"]["path"]
    sites = []
    for site_index, site in enumerate(SITES, start=1):
        write_progress(
            volume,
            phase="ign_terrain_download",
            message=(
                f"Préparation IGN du site {site_index}/{len(SITES)} : "
                f"{site['label']}."
            ),
            current_site=site["label"],
            sites_completed=site_index - 1,
            sites_total=len(SITES),
            assets_locked=len(vegetation_paths) + 1,
            contracts_completed=0,
            contracts_total=512,
        )
        sites.append(_prepare_site(
            root,
            site,
            fetcher=fetcher,
            vegetation_assets=vegetation_paths,
            building_asset=building_path,
        ))
        write_progress(
            volume,
            phase="ign_terrain_authoring",
            message=f"Site {site['label']} téléchargé et scène USD écrite.",
            current_site=site["label"],
            sites_completed=site_index,
            sites_total=len(SITES),
            assets_locked=len(vegetation_paths) + 1,
            contracts_completed=0,
            contracts_total=512,
        )
    write_progress(
        volume,
        phase="environment_usd_validation",
        message=(
            "Chargement complet des trois scènes et des assets dans Isaac "
            "pour valider géométrie, PBR, échelle et compatibilité USD."
        ),
        sites_completed=len(sites),
        sites_total=len(SITES),
        assets_validated=0,
        assets_total=len(sites) + len(vegetation_paths) + 1,
        contracts_completed=0,
        contracts_total=512,
    )
    environment_validation = usd_validator(
        [
            *(site["scene"] for site in sites),
            *vegetation_paths,
            building_path,
        ]
    )
    for entry in simready_assets["vegetation"]:
        entry["usd_quality"] = _assert_asset_quality(
            environment_validation,
            entry=entry,
            family="vegetation",
        )
    simready_assets["rural_building"]["usd_quality"] = _assert_asset_quality(
        environment_validation,
        entry=simready_assets["rural_building"],
        family="rural_building",
    )
    write_progress(
        volume,
        phase="environment_usd_validation",
        state="completed",
        message="Trois scènes et sept assets environnement validés dans Isaac.",
        sites_completed=len(sites),
        sites_total=len(SITES),
        assets_validated=int(environment_validation.get("count", 0)),
        assets_total=len(sites) + len(vegetation_paths) + 1,
        contracts_completed=0,
        contracts_total=512,
    )
    site_reference_validation = site_reference_validator(
        sites,
        volume_root=volume,
    )
    if (
        len(site_reference_validation) != len(sites)
        or {
            str(item.get("site_id", ""))
            for item in site_reference_validation
        }
        != {str(site["id"]) for site in sites}
        or any(
            item.get("automatic_validation") != "passed"
            for item in site_reference_validation
        )
    ):
        raise RuntimeError(
            "every pilot site requires a passed RTX reference, calibration and USD raycast"
        )
    references_by_site = {
        str(item["site_id"]): item for item in site_reference_validation
    }
    for site in sites:
        site["reference_validation"] = references_by_site[str(site["id"])]
    readiness_path = volume / "input" / READINESS_REPORT_NAME
    if include_response_assets and simready_assets["missing_actor_classes"]:
        readiness = {
            "schema_version": 1,
            "state": "blocked",
            "phase": "response_assets",
            "reason": (
                "the three environments are technically prepared, but exact "
                "response assets are missing; generic vehicles were not promoted"
            ),
            "environment": {
                "site_count": len(sites),
                "technical_validation": environment_validation,
                "site_reference_validation": site_reference_validation,
                "visual_review": "pending_console_review",
            },
            "fire_and_smoke": {"state": "not_started"},
            "missing_actor_classes": simready_assets["missing_actor_classes"],
            "asset_manifest": str(simready_assets["path"]),
            "asset_manifest_sha256": simready_assets["sha256"],
            "asset_discovery": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in discovery.items()
            },
            "catalog_written": False,
            "synthetic_cases_written": 0,
            "production_scope": "pilot_setup_proof",
            "bulk_allowed": False,
        }
        _write_json(readiness_path, readiness)
        return {
            **readiness,
            "readiness_report": str(readiness_path),
        }
    preset = _discover_flow_fire_preset(runtime_root)
    write_progress(
        volume,
        phase="fire_smoke_validation",
        message="Preset NVIDIA Flow trouvé; validation du feu et de la fumée en cours.",
        sites_completed=len(sites),
        sites_total=len(SITES),
        assets_validated=int(environment_validation.get("count", 0)),
        contracts_completed=0,
        contracts_total=512,
    )
    flow_asset = root / "flow" / "official-nvidia-fire.usda"
    _write_flow_wrapper(flow_asset, preset)
    actor_assets = (
        simready_assets["actors"] if include_response_assets else {}
    )
    effects_and_actors_validation = usd_validator(
        [
            preset,
            flow_asset,
            *(entry["path"] for entry in actor_assets.values()),
        ]
    )
    for entry in actor_assets.values():
        entry["usd_quality"] = _assert_asset_quality(
            effects_and_actors_validation,
            entry=entry,
            family="actor",
        )
    usd_assets = []
    usd_quality: dict[str, Any] = {}
    for report in (environment_validation, effects_and_actors_validation):
        for asset in report.get("assets", []):
            if asset not in usd_assets:
                usd_assets.append(asset)
        quality = report.get("quality", {})
        if isinstance(quality, dict):
            usd_quality.update(quality)
    usd_validation = {
        "count": len(usd_assets),
        "assets": usd_assets,
        "quality": usd_quality,
    }
    events = []
    for index in range(512):
        event_id = f"fire-fr-{index:04d}"
        event_root = root / "events" / event_id
        contract_path = event_root / "contract.json"
        contract = _event_contract(
            index=index,
            event_root=event_root,
            site=sites[index % len(sites)],
            flow_asset=flow_asset,
            actor_assets=actor_assets,
            asset_manifest_sha256=simready_assets["sha256"],
            response_engagement_in_scope=include_response_assets,
        )
        _write_json(contract_path, contract)
        events.append(
            {
                "event_id": event_id,
                "case_slots_per_category": SLOT_PATTERN[index % len(SLOT_PATTERN)],
                "real_world_contract": os.path.relpath(contract_path, volume),
                "real_world_contract_sha256": _sha256(contract_path),
            }
        )
        if (index + 1) % 64 == 0:
            print(f"fireviewer inputs: contracts={index + 1}/512", flush=True)
            write_progress(
                volume,
                phase="event_contract_generation",
                message=f"Contrats de feux écrits : {index + 1}/512.",
                sites_completed=len(sites),
                sites_total=len(SITES),
                contracts_completed=index + 1,
                contracts_total=512,
            )
    payload = {
        "schema_version": CATALOG_SCHEMA,
        "preparation_version": PREPARATION_VERSION,
        "data_origin": "new_synthetic_generation",
        "minimum_fire_events": 512,
        "maximum_fire_duration_days": 15,
        "max_cases_per_fire_per_category": 24,
        "site_scope": "pilot_setup_proof",
        "site_count": len(sites),
        "bulk_allowed": False,
        "events": events,
    }
    _write_json(catalog, payload)
    _write_json(
        readiness_path,
        {
            "schema_version": 1,
            "state": "ready_for_console_review",
            "phase": "catalog_prepared",
            "production_scope": "pilot_setup_proof",
            "bulk_allowed": False,
            "environment": {
                "site_count": len(sites),
                "technical_validation": environment_validation,
                "site_reference_validation": site_reference_validation,
                "visual_review": "pending_console_review",
            },
            "fire_and_smoke": {
                "state": "technically_validated",
                "visual_review": "pending_console_review",
            },
            "response_assets": (
                {
                    "technical_validation": effects_and_actors_validation,
                    "engagement_context_review": "pending_console_review",
                }
                if include_response_assets
                else {
                    "state": "out_of_scope_v1",
                    "vehicles": "excluded",
                    "humans": "excluded",
                }
            ),
            "catalog": str(catalog),
            "catalog_sha256": _sha256(catalog),
            "catalog_written": True,
            "synthetic_cases_written": 0,
        },
    )
    print(
        "fireviewer inputs: catalog ready fires=512 slots_per_category=4096",
        flush=True,
    )
    write_progress(
        volume,
        phase="catalog_prepared",
        state="completed",
        message=(
            "Catalogue pilote prêt : 3 sites, 512 feux, 4096 créneaux par "
            "catégorie; revue visuelle obligatoire avant production."
        ),
        sites_completed=len(sites),
        sites_total=len(SITES),
        assets_validated=int(usd_validation.get("count", 0)),
        contracts_completed=len(events),
        contracts_total=512,
        catalog=os.path.relpath(catalog, volume).replace("\\", "/"),
    )
    return {
        "state": "prepared",
        "catalog": str(catalog),
        "fire_events": 512,
        "site_count": len(sites),
        "production_scope": "pilot_setup_proof",
        "bulk_allowed": False,
        "asset_manifest": str(simready_assets["path"]),
        "asset_manifest_sha256": simready_assets["sha256"],
        "readiness_report": str(readiness_path),
        "flow_preset": str(preset),
        "usd_validation": usd_validation,
    }


__all__ = ["prepare_ign_catalog"]
