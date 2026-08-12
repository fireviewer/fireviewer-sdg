"""Capture the approved Z16 valley-wind Editor scenario with native Kit RTX.

Run this script with the Isaac Sim Python launcher after the human Editor
approval.  The source overlay is opened read-only in practice: all runtime
opinions are authored in its anonymous session layer and no USD layer is saved.

The thermal modality is a 16-bit Kelvin image.  It is derived from the warm,
emissive signal in the synchronous RTX RGB render, spatially constrained by
the camera projection of the authored fire-front centroid and radius.  It is
therefore an image-derived heat modality, not a renamed or empty placeholder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


SCENE_PATH = Path(
    "/workspace/fireviewer-omniverse/production/zone-scenes/"
    "Z16/build/Z16_editor_overlay.usda"
)
SCENARIO_ID = os.getenv(
    "FW_Z16_CAPTURE_SCENARIO_ID", "Z16-VALLEY-WIND-04D-V1"
).strip()
if not SCENARIO_ID:
    raise RuntimeError("FW_Z16_CAPTURE_SCENARIO_ID cannot be empty")
SCENARIO_PRIM_PATH = "/FireViewerEditorOverlay/Scenario"
CAMERA_ROOT_PATH = f"{SCENARIO_PRIM_PATH}/Cameras"
ACTOR_CAMERA_ROOT_PATH = f"{SCENARIO_PRIM_PATH}/ActorCameras"
FIRE_PRIM_PATH = f"{SCENARIO_PRIM_PATH}/Fire"
DEFAULT_SCENARIOS_ROOT = Path(
    "/workspace/fireviewer-omniverse/production/zone-scenes/Z16/scenarios"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/fireviewer-omniverse/production/captures/"
    f"Z16/{SCENARIO_ID}"
)
DEFAULT_RESOLUTION = (3840, 2160)
CAPTURES_PER_DAY = 3
EXPECTED_CAMERAS = 40
EXPECTED_ACTOR_CAMERAS = 5


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return result if result and not result[0].isdigit() else f"_{result}"


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("resolution must have WIDTHxHEIGHT form")
    width, height = (int(match.group(1)), int(match.group(2)))
    if width < 64 or height < 64:
        raise argparse.ArgumentTypeError("resolution must be at least 64x64")
    return width, height


def _scenario_steps(contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scenarios = contract.get("scenarios")
    if not isinstance(scenarios, list):
        raise RuntimeError("fire scenario contract has no scenarios")
    scenario = next(
        (
            item
            for item in scenarios
            if isinstance(item, dict) and item.get("scenario_id") == SCENARIO_ID
        ),
        None,
    )
    if scenario is None:
        raise RuntimeError(f"fire scenario contract has no {SCENARIO_ID}")
    steps: list[dict[str, Any]] = []
    for day in scenario.get("days", []):
        if not isinstance(day, dict):
            continue
        day_index = int(day["day_index"])
        for hour in day.get("hours", []):
            if not isinstance(hour, dict):
                continue
            steps.append(
                {
                    "day_index": day_index,
                    "capture_hour": str(hour["capture_hour"]),
                    "fire": dict(hour["fire"]),
                    "weather": dict(hour["weather"]),
                    "terrain_sample": dict(hour.get("terrain_sample", {})),
                }
            )
    day_count = len(
        [day for day in scenario.get("days", []) if isinstance(day, dict)]
    )
    expected_time_steps = day_count * CAPTURES_PER_DAY
    if day_count < 1 or len(steps) != expected_time_steps:
        raise RuntimeError(
            f"{SCENARIO_ID} must expose exactly three captures per day"
        )
    return scenario, steps


def _camera_records(contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    zone = contract.get("zone")
    cameras = contract.get("cameras")
    if not isinstance(zone, dict):
        raise RuntimeError("camera contract has no Z16 georeference")
    if not isinstance(cameras, list) or len(cameras) != EXPECTED_CAMERAS:
        raise RuntimeError(
            f"camera contract must expose exactly {EXPECTED_CAMERAS} cameras"
        )
    origin = zone.get("origin_epsg2154_ign69_m")
    if (
        zone.get("horizontal_crs") != "EPSG:2154"
        or zone.get("vertical_datum") != "IGN69"
        or not isinstance(origin, list)
        or len(origin) != 3
    ):
        raise RuntimeError("camera contract has no usable EPSG:2154/IGN69 origin")
    return zone, [dict(camera) for camera in cameras]


def _actor_camera_records(stage: Any) -> list[dict[str, Any]]:
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(ACTOR_CAMERA_ROOT_PATH)
    if not root.IsValid():
        raise RuntimeError("selected scenario exposes no actor-camera root")
    result: list[dict[str, Any]] = []
    for prim in root.GetChildren():
        if not prim.IsA(UsdGeom.Camera):
            continue
        camera_id = prim.GetAttribute("fireviewer:cameraId").Get()
        view_class = prim.GetAttribute("fireviewer:viewClass").Get()
        intrinsics_json = prim.GetAttribute("fireviewer:intrinsicsJson").Get()
        selection_id = prim.GetAttribute("fireviewer:selectionId").Get()
        if not all(
            isinstance(value, str) and value
            for value in (camera_id, view_class, intrinsics_json, selection_id)
        ):
            raise RuntimeError(f"actor camera {prim.GetPath()} is incomplete")
        try:
            intrinsics = json.loads(intrinsics_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"actor camera {prim.GetPath()} has invalid intrinsics"
            ) from exc
        result.append(
            {
                "camera_id": camera_id,
                "view_class": view_class,
                "intrinsics": intrinsics,
                "selection_id": selection_id,
                "prim_path": str(prim.GetPath()),
            }
        )
    if len(result) != EXPECTED_ACTOR_CAMERAS:
        raise RuntimeError(
            f"selected scenario must expose {EXPECTED_ACTOR_CAMERAS} actor cameras"
        )
    return result


def _wait_for_loading(
    *, context: Any, application: Any, updates: int = 3_600
) -> None:
    stable = 0
    previous: tuple[str, int, int] | None = None
    for _ in range(updates):
        application.update()
        message, loaded, total = context.get_stage_loading_status()
        current = (str(message), int(loaded), int(total))
        if current[1] < current[2] or current != previous:
            stable = 0
        else:
            stable += 1
            if stable >= 6:
                return
        previous = current
    raise RuntimeError(
        "Z16 overlay did not settle before capture: "
        f"message={previous[0] if previous else ''!r}, "
        f"loaded={previous[1] if previous else 0}, "
        f"total={previous[2] if previous else 0}"
    )


def _rgb_u8(value: Any, *, resolution: tuple[int, int]) -> np.ndarray:
    data = np.asarray(value)
    width, height = resolution
    if data.ndim != 3 or data.shape[:2] != (height, width) or data.shape[2] < 3:
        raise RuntimeError(f"Replicator RGB has unexpected shape {tuple(data.shape)}")
    rgb = np.asarray(data[:, :, :3])
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(rgb)) <= 1.5:
            rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    value = rgb.astype(np.float32) / 255.0
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    encoded = np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(encoded * 255.0), 0.0, 255.0).astype(np.uint8)


def _project_fire(
    *,
    camera_world: Any,
    centroid: Iterable[float],
    intrinsics: dict[str, Any],
    resolution: tuple[int, int],
) -> tuple[float, float, float] | None:
    from pxr import Gf

    point = camera_world.GetInverse().Transform(
        Gf.Vec3d(*[float(component) for component in centroid])
    )
    depth = -float(point[2])
    if not math.isfinite(depth) or depth <= 0.0:
        return None
    source_width = float(intrinsics["width_px"])
    source_height = float(intrinsics["height_px"])
    width, height = resolution
    fx = float(intrinsics["fx_px"]) * width / source_width
    fy = float(intrinsics["fy_px"]) * height / source_height
    cx = float(intrinsics["cx_px"]) * width / source_width
    cy = float(intrinsics["cy_px"]) * height / source_height
    return (
        fx * float(point[0]) / depth + cx,
        cy - fy * float(point[1]) / depth,
        depth,
    )


def _thermal_kelvin(
    *,
    linear_rgb: np.ndarray,
    camera_world: Any,
    fire: dict[str, Any],
    weather: dict[str, Any],
    intrinsics: dict[str, Any],
    resolution: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = linear_rgb.shape[:2]
    red, green, blue = (
        linear_rgb[:, :, 0],
        linear_rgb[:, :, 1],
        linear_rgb[:, :, 2],
    )
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    warm_excess = np.clip(red - 0.55 * green - 0.20 * blue - 0.025, 0.0, 1.0)
    rendered_heat_signal = warm_excess * np.sqrt(np.clip(luminance, 0.0, 1.0))

    projection = _project_fire(
        camera_world=camera_world,
        centroid=fire["active_flame_centroid_local_m"],
        intrinsics=intrinsics,
        resolution=resolution,
    )
    visible = projection is not None
    if projection is not None:
        px, py, depth = projection
        scaled_fx = (
            float(intrinsics["fx_px"])
            * width
            / float(intrinsics["width_px"])
        )
        radius_px = max(
            3.0,
            scaled_fx
            * float(fire["active_flame_front_radius_m"])
            / depth,
        )
        yy, xx = np.ogrid[:height, :width]
        distance2 = (xx - px) ** 2 + (yy - py) ** 2
        fire_region = np.exp(
            -distance2 / (2.0 * max(radius_px * 0.65, 2.0) ** 2)
        ).astype(np.float32)
        rendered_heat_signal *= fire_region
        visible = (
            -radius_px <= px < width + radius_px
            and -radius_px <= py < height + radius_px
        )
    else:
        rendered_heat_signal.fill(0.0)

    peak = float(np.percentile(rendered_heat_signal, 99.95))
    if peak > 1e-7:
        rendered_heat_signal = np.clip(rendered_heat_signal / peak, 0.0, 1.0)
    else:
        rendered_heat_signal.fill(0.0)

    ambient_kelvin = float(weather["air_temperature_c"]) + 273.15
    flame_kelvin = float(
        np.clip(
            1_073.15 + 15.0 * float(fire["max_flame_height_m"]),
            1_073.15,
            1_473.15,
        )
    )
    temperature = ambient_kelvin + np.power(
        rendered_heat_signal, 0.60
    ) * (flame_kelvin - ambient_kelvin)
    kelvin_u16 = np.clip(np.rint(temperature * 10.0), 0, 65_535).astype(
        np.uint16
    )
    return kelvin_u16, {
        "encoding": "uint16_png",
        "scale_kelvin_per_code": 0.1,
        "ambient_kelvin": ambient_kelvin,
        "model_flame_kelvin": flame_kelvin,
        "rendered_signal_peak_before_normalization": peak,
        "projected_fire_front_visible": bool(visible),
        "derivation": (
            "RTX linear-RGB warm emissive response multiplied by the "
            "projected synchronous fire-front support, then mapped to Kelvin"
        ),
    }


def _pose(
    *, camera_prim: Any, time_code: Any, origin: list[Any]
) -> tuple[dict[str, Any], Any]:
    from pxr import UsdGeom

    cache = UsdGeom.XformCache(time_code)
    matrix = cache.GetLocalToWorldTransform(camera_prim)
    position = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    local = [float(position[index]) for index in range(3)]
    orientation = [
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quaternion.GetReal()),
    ]
    return (
        {
            "local_z_up_m": {
                "position_m": local,
                "orientation_xyzw": orientation,
            },
            "epsg2154_ign69_m": {
                "position_m": [
                    float(origin[index]) + local[index] for index in range(3)
                ],
                "orientation_xyzw": orientation,
            },
        },
        matrix,
    )


def _rendered_intrinsics(
    intrinsics: dict[str, Any], resolution: tuple[int, int]
) -> dict[str, Any]:
    width, height = resolution
    x_scale = width / float(intrinsics["width_px"])
    y_scale = height / float(intrinsics["height_px"])
    return {
        **intrinsics,
        "width_px": width,
        "height_px": height,
        "fx_px": float(intrinsics["fx_px"]) * x_scale,
        "fy_px": float(intrinsics["fy_px"]) * y_scale,
        "cx_px": float(intrinsics["cx_px"]) * x_scale,
        "cy_px": float(intrinsics["cy_px"]) * y_scale,
    }


def _capture_observation(
    *,
    rep: Any,
    annotator: Any,
    camera_prim: Any,
    camera: dict[str, Any],
    step: dict[str, Any],
    origin: list[Any],
    output_root: Path,
    resolution: tuple[int, int],
    rt_subframes: int,
) -> None:
    from pxr import Usd

    simulation_seconds = float(step["fire"]["simulation_time_seconds"])
    time_code = Usd.TimeCode(simulation_seconds)
    pose, camera_world = _pose(
        camera_prim=camera_prim, time_code=time_code, origin=origin
    )
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
    rgb = _rgb_u8(annotator.get_data(), resolution=resolution)
    linear_rgb = _srgb_to_linear(rgb)
    negative = _linear_to_srgb(1.0 - np.clip(linear_rgb, 0.0, 1.0))
    thermal, thermal_metadata = _thermal_kelvin(
        linear_rgb=linear_rgb,
        camera_world=camera_world,
        fire=step["fire"],
        weather=step["weather"],
        intrinsics=camera["intrinsics"],
        resolution=resolution,
    )

    day_id = f"D{int(step['day_index']):02d}"
    time_id = str(step["capture_hour"]).replace(":", "")
    view_id = str(camera["camera_id"])
    observation_id = f"{SCENARIO_ID}:{day_id}:{step['capture_hour']}:{view_id}"
    observation_root = output_root / day_id / f"T{time_id}" / view_id
    observation_root.mkdir(parents=True, exist_ok=True)
    normal_path = observation_root / "normal_rgb.png"
    negative_path = observation_root / "negative.png"
    thermal_path = observation_root / "thermal_hotspot.png"
    Image.fromarray(rgb, mode="RGB").save(normal_path)
    Image.fromarray(negative, mode="RGB").save(negative_path)
    Image.fromarray(thermal, mode="I;16").save(thermal_path)

    metadata = {
        "schema_version": 1,
        "observation_id": observation_id,
        "scene_id": "Z16",
        "scenario_id": SCENARIO_ID,
        "day_id": day_id,
        "day_index": int(step["day_index"]),
        "view_id": view_id,
        "view_class": camera["view_class"],
        "time_id": str(step["capture_hour"]),
        "simulation_time_seconds": simulation_seconds,
        "camera": {
            "prim_path": str(camera_prim.GetPath()),
            "pose": pose,
            "intrinsics": _rendered_intrinsics(
                dict(camera["intrinsics"]), resolution
            ),
            "actor_selection_id": camera.get("selection_id"),
        },
        "georeference": {
            "horizontal_crs": "EPSG:2154",
            "vertical_datum": "IGN69",
            "local_axes": ["east", "north", "up"],
            "origin_epsg2154_ign69_m": [float(value) for value in origin],
        },
        "fire": dict(step["fire"]),
        "weather": dict(step["weather"]),
        "terrain_sample": dict(step["terrain_sample"]),
        "modalities": {
            "normal_rgb": {
                "path": normal_path.name,
                "encoding": "uint8_srgb_png",
                "source": "native_omniverse_replicator_rgb_annotator",
            },
            "negative": {
                "path": negative_path.name,
                "encoding": "uint8_srgb_png",
                "source": "normal_rgb",
                "formula": "negative_linear_rgb=1-clamp(normal_linear_rgb,0,1)",
            },
            "thermal_hotspot": {
                "path": thermal_path.name,
                **thermal_metadata,
            },
        },
        "render": {
            "runtime": "native_isaac_sim_omniverse_replicator",
            "renderer": "RayTracedLighting",
            "resolution_px": [resolution[0], resolution[1]],
            "rt_subframes": rt_subframes,
            "source_stage_saved": False,
            "runtime_opinions_layer": "anonymous_session_layer",
        },
    }
    _atomic_json(observation_root / "metadata.json", metadata)
    print(f"CAPTURED {observation_id}", flush=True)


def capture(
    *,
    scenarios_root: Path,
    output_root: Path,
    resolution: tuple[int, int],
    rt_subframes: int,
) -> int:
    if not SCENE_PATH.is_file():
        raise RuntimeError(f"approved Z16 Editor overlay is absent: {SCENE_PATH}")
    output_root.mkdir(parents=True, exist_ok=True)
    fire_contract = _read_json(
        scenarios_root / "z16-fire-scenarios-v1.json",
        label="Z16 fire scenario contract",
    )
    camera_contract = _read_json(
        scenarios_root / "z16-capture-cameras-v1.json",
        label="Z16 camera contract",
    )
    _scenario, steps = _scenario_steps(fire_contract)
    zone, cameras = _camera_records(camera_contract)
    origin = list(zone["origin_epsg2154_ign69_m"])

    from isaacsim.simulation_app import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "renderer": "RayTracedLighting",
            "multi_gpu": False,
            "width": resolution[0],
            "height": resolution[1],
        }
    )
    try:
        import carb.settings
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd

        settings = carb.settings.get_settings()
        settings.set("/renderer/multiGpu/enabled", False)
        settings.set("/rtx/rendermode", "RayTracedLighting")
        settings.set("/rtx/post/aa/op", 3)
        settings.set("/rtx/post/dlss/execMode", 2)
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.flowusd", True)
        if not manager.is_extension_enabled("omni.flowusd"):
            raise RuntimeError("native NVIDIA Flow extension could not be enabled")

        context = omni.usd.get_context()
        if not context.open_stage(str(SCENE_PATH)):
            raise RuntimeError(f"Kit could not open the Z16 overlay: {SCENE_PATH}")
        _wait_for_loading(context=context, application=application)
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("Kit exposes no opened Z16 stage")

        # Variant selection is a runtime opinion only.  No source/root layer is
        # ever selected as an edit target and no layer Save() call is made.
        stage.SetEditTarget(stage.GetSessionLayer())
        scenario_prim = stage.GetPrimAtPath(SCENARIO_PRIM_PATH)
        if not scenario_prim.IsValid():
            raise RuntimeError(
                f"overlay has no scenario prim at {SCENARIO_PRIM_PATH}"
            )
        variants = scenario_prim.GetVariantSets().GetVariantSet("fireScenario")
        variant_name = _identifier(SCENARIO_ID)
        if variant_name not in set(variants.GetVariantNames()):
            raise RuntimeError(f"overlay has no scenario variant {SCENARIO_ID}")
        if not variants.SetVariantSelection(variant_name):
            raise RuntimeError(f"could not select scenario variant {SCENARIO_ID}")
        stage.Load()
        _wait_for_loading(context=context, application=application)

        fire_prim = stage.GetPrimAtPath(FIRE_PRIM_PATH)
        if not fire_prim.IsValid():
            raise RuntimeError("selected scenario exposes no rendered Flow fire")
        selected_id = fire_prim.GetAttribute("fireviewer:scenarioId").Get()
        if selected_id != SCENARIO_ID:
            raise RuntimeError(
                f"selected Flow fire is bound to {selected_id!r}, not {SCENARIO_ID}"
            )
        capture_cameras = [*cameras, *_actor_camera_records(stage)]

        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        rendered_count = 0
        for step_index, step in enumerate(steps, start=1):
            simulation_seconds = float(step["fire"]["simulation_time_seconds"])
            timeline.set_current_time(simulation_seconds)
            for _ in range(4):
                application.update()
            for camera_index, camera in enumerate(capture_cameras, start=1):
                camera_id = str(camera["camera_id"])
                camera_path = str(
                    camera.get(
                        "prim_path",
                        f"{CAMERA_ROOT_PATH}/{_identifier(camera_id)}",
                    )
                )
                camera_prim = stage.GetPrimAtPath(camera_path)
                if not camera_prim.IsValid():
                    raise RuntimeError(
                        f"selected scenario exposes no camera {camera_id}"
                    )
                product = rep.create.render_product(
                    camera_path,
                    resolution,
                    name=(
                        f"Z16_{step_index:02d}_{camera_index:02d}_"
                        f"{_identifier(camera_id)}"
                    ),
                )
                rgb = rep.annotators.get("rgb")
                rgb.attach(product)
                try:
                    _capture_observation(
                        rep=rep,
                        annotator=rgb,
                        camera_prim=camera_prim,
                        camera=camera,
                        step=step,
                        origin=origin,
                        output_root=output_root,
                        resolution=resolution,
                        rt_subframes=rt_subframes,
                    )
                finally:
                    rgb.detach()
                    product.destroy()
                rendered_count += 1
        expected_observations = len(steps) * len(capture_cameras)
        if rendered_count != expected_observations:
            raise RuntimeError(
                f"capture stopped after {rendered_count} observations"
            )
        print(
            json.dumps(
                {
                    "scene": str(SCENE_PATH),
                    "scenario_id": SCENARIO_ID,
                    "time_steps": len(steps),
                    "cameras": len(capture_cameras),
                    "observations": rendered_count,
                    "output_root": str(output_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        # Match the native renderer's process lifecycle.  Kit shutdown can
        # otherwise swallow a propagated render exception before the launcher
        # receives a non-zero exit code.
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture three daily observations from every fixed camera for one "
            "approved Z16 fire-scenario timeline with native Omniverse Replicator"
        )
    )
    parser.add_argument(
        "--scenarios-root",
        type=Path,
        default=Path(
            os.getenv("FW_Z16_SCENARIOS_ROOT", str(DEFAULT_SCENARIOS_ROOT))
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.getenv("FW_Z16_CAPTURE_OUTPUT", str(DEFAULT_OUTPUT_ROOT))
        ),
    )
    parser.add_argument(
        "--resolution",
        type=_resolution,
        default=_resolution(
            os.getenv(
                "FW_Z16_CAPTURE_RESOLUTION",
                f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}",
            )
        ),
    )
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=int(os.getenv("FW_Z16_CAPTURE_RT_SUBFRAMES", "4")),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rt_subframes < 1:
        raise ValueError("--rt-subframes must be positive")
    scenarios_root = args.scenarios_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not scenarios_root.is_dir():
        raise RuntimeError(f"scenario directory is absent: {scenarios_root}")
    return capture(
        scenarios_root=scenarios_root,
        output_root=output_root,
        resolution=args.resolution,
        rt_subframes=args.rt_subframes,
    )


if __name__ == "__main__":
    try:
        status = main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(status)
