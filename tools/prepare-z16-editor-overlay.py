"""Author and open a non-destructive Z16 Editor overlay.

The generated layer composes the existing Z16 root as a read-only sublayer.
All cameras, scenario variants, fire/weather tracks and response actors are
authored in the new overlay; the source scene is never selected as an edit
target or saved.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom


OVERLAY_ROOT = "/FireViewerEditorOverlay"
SCENARIO_ROOT = f"{OVERLAY_ROOT}/Scenario"
LOD_PRIORITY = ("LOD0", "LOD1", "LOD2", "LOD3")
EDITOR_CAMERA_ID = "VIEW-36"
EDITOR_DETAIL_CORRIDOR_PADDING_M = 1_000.0
EDITOR_HERO_TILE_CAP = 2
EDITOR_MID_TILE_CAP = 8
TerrainGrid = tuple[
    float,
    float,
    float,
    float,
    int,
    Any,
    Gf.Matrix4d,
    float,
    float,
    float,
    float,
]


def _required_path(name: str) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise RuntimeError(f"{name} is required")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{name} does not identify a file: {path}")
    return path


def _output_path() -> Path:
    raw = os.getenv("FW_Z16_OVERLAY_OUTPUT", "").strip()
    if not raw:
        raise RuntimeError("FW_Z16_OVERLAY_OUTPUT is required")
    output = Path(raw).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object")
    return value


def _identifier(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not result or result[0].isdigit():
        result = f"_{result}"
    return result


def _asset_path(volume_root: Path, wrapper: str) -> Path:
    path = Path(wrapper)
    if not path.is_absolute():
        path = volume_root / path
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"response actor wrapper is absent: {path}")
    return path


def _string_attr(prim: Usd.Prim, name: str, value: str) -> None:
    prim.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True).Set(value)


def _double_attr(prim: Usd.Prim, name: str) -> Usd.Attribute:
    return prim.CreateAttribute(name, Sdf.ValueTypeNames.Double, custom=True)


def _double3_attr(prim: Usd.Prim, name: str) -> Usd.Attribute:
    return prim.CreateAttribute(name, Sdf.ValueTypeNames.Double3, custom=True)


def _look_at_quaternion(eye: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Quatd:
    delta = target - eye
    if delta.GetLength() <= 1e-6:
        raise RuntimeError("camera target coincides with its position")
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    direction = delta.GetNormalized()
    if abs(direction * up) > 0.995:
        up = Gf.Vec3d(0.0, 1.0, 0.0)
    view = Gf.Matrix4d().SetLookAt(eye, target, up)
    return view.GetInverse().ExtractRotationQuat()


def _select_highest_lods(stage: Usd.Stage) -> None:
    prims = list(stage.TraverseAll())
    for prim in prims:
        sets = prim.GetVariantSets()
        for name in sets.GetNames():
            variant = sets.GetVariantSet(name)
            choices = set(variant.GetVariantNames())
            if name == "terrainLOD":
                selected = next((item for item in LOD_PRIORITY if item in choices), None)
                if selected:
                    variant.SetVariantSelection(selected)
            elif "lod" in name.lower() and "HERO" in choices:
                variant.SetVariantSelection("HERO")
            elif name == "collisionLOD" and "NEAR" in choices:
                variant.SetVariantSelection("NEAR")

    # Each tile exposes exclusive HERO, MID and FAR payload choices. Keep their
    # headers active so the Editor loader can compose exactly one level per
    # tile without duplicating the same vegetation.
    for prim in prims:
        if prim.GetName() in {"Details", "DetailsMid", "DetailsFar"}:
            prim.SetActive(True)


def _load_terrain(stage: Usd.Stage) -> None:
    paths = [
        prim.GetPath()
        for prim in stage.TraverseAll()
        if prim.GetName() == "Terrain" and prim.HasPayload()
    ]
    for path in paths:
        stage.Load(path)


def _preferred_editor_camera(stage: Usd.Stage) -> tuple[Usd.Prim, str]:
    cameras = stage.GetPrimAtPath(f"{SCENARIO_ROOT}/Cameras").GetChildren()
    if not cameras:
        raise RuntimeError("selected scenario exposes no camera")
    for camera in cameras:
        camera_id = camera.GetAttribute("fireviewer:cameraId").Get()
        if camera_id == EDITOR_CAMERA_ID:
            return camera, str(camera_id)
    fallback = cameras[0]
    camera_id = fallback.GetAttribute("fireviewer:cameraId").Get()
    return fallback, str(camera_id or fallback.GetName())


def _tile_local_bounds(detail_prim: Usd.Prim) -> tuple[float, float, float, float]:
    raw = detail_prim.GetParent().GetCustomDataByKey("fireviewer:local_bounds")
    if not isinstance(raw, str):
        raise RuntimeError(
            f"detail payload parent has no local bounds: {detail_prim.GetPath()}"
        )
    try:
        xmin, ymin, xmax, ymax = (float(value) for value in raw.split(","))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"detail payload parent has invalid local bounds: {detail_prim.GetPath()}"
        ) from exc
    return xmin, ymin, xmax, ymax


def _point_to_segment_distance_xy(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    length_squared = segment_x * segment_x + segment_y * segment_y
    if length_squared <= 1.0e-9:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = (
        (point[0] - start[0]) * segment_x
        + (point[1] - start[1]) * segment_y
    ) / length_squared
    ratio = max(0.0, min(1.0, ratio))
    nearest = (
        start[0] + ratio * segment_x,
        start[1] + ratio * segment_y,
    )
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def _detail_lods_for_camera_fire_corridor(
    stage: Usd.Stage,
    *,
    camera: Usd.Prim,
    fire: Usd.Prim,
) -> dict[str, list[Sdf.Path]]:
    time = Usd.TimeCode(0.0)
    camera_position = UsdGeom.XformCache(time).GetLocalToWorldTransform(
        camera
    ).ExtractTranslation()
    fire_position = fire.GetAttribute(
        "fireviewer:activeFlameCentroidLocalM"
    ).Get(time)
    if fire_position is None:
        raise RuntimeError("selected scenario exposes no fire centroid at time 0")
    camera_xy = (float(camera_position[0]), float(camera_position[1]))
    fire_xy = (float(fire_position[0]), float(fire_position[1]))
    candidates: list[
        tuple[Usd.Prim, float, float, float]
    ] = []
    for prim in stage.TraverseAll():
        if prim.GetName() != "Details" or not prim.HasPayload():
            continue
        xmin, ymin, xmax, ymax = _tile_local_bounds(prim)
        centre = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5)
        half_diagonal = 0.5 * math.hypot(xmax - xmin, ymax - ymin)
        corridor_distance = _point_to_segment_distance_xy(
            centre, camera_xy, fire_xy
        )
        if corridor_distance > EDITOR_DETAIL_CORRIDOR_PADDING_M + half_diagonal:
            continue
        candidates.append(
            (
                prim.GetParent(),
                corridor_distance,
                math.dist(centre, camera_xy),
                math.dist(centre, fire_xy),
            )
        )
    if len(candidates) < EDITOR_HERO_TILE_CAP + EDITOR_MID_TILE_CAP:
        raise RuntimeError(
            "camera-to-fire corridor does not expose enough detail tiles"
        )

    def candidate_key(item: tuple[Usd.Prim, float, float, float]) -> str:
        return str(item[0].GetPath())

    hero_keys: set[str] = set()
    endpoint_cap = EDITOR_HERO_TILE_CAP // 2
    for distance_index in (2, 3):
        target_size = min(
            EDITOR_HERO_TILE_CAP, len(hero_keys) + endpoint_cap
        )
        for item in sorted(
            candidates,
            key=lambda candidate: (
                candidate[distance_index],
                candidate[1],
                candidate_key(candidate),
            ),
        ):
            hero_keys.add(candidate_key(item))
            if len(hero_keys) >= target_size:
                break
    if len(hero_keys) < EDITOR_HERO_TILE_CAP:
        for item in sorted(
            candidates,
            key=lambda candidate: (
                candidate[1],
                min(candidate[2], candidate[3]),
                candidate_key(candidate),
            ),
        ):
            hero_keys.add(candidate_key(item))
            if len(hero_keys) == EDITOR_HERO_TILE_CAP:
                break
    remaining = [
        item for item in candidates if candidate_key(item) not in hero_keys
    ]
    mid_keys = {
        candidate_key(item)
        for item in sorted(
            remaining,
            key=lambda candidate: (
                candidate[1],
                min(candidate[2], candidate[3]),
                candidate_key(candidate),
            ),
        )[:EDITOR_MID_TILE_CAP]
    }

    result: dict[str, list[Sdf.Path]] = {
        "HERO": [],
        "MID": [],
        "FAR": [],
    }
    selected_parent_keys: set[str] = set()
    for parent, _distance, _camera_distance, _fire_distance in candidates:
        parent_key = str(parent.GetPath())
        if parent_key in selected_parent_keys:
            raise RuntimeError(f"duplicate detail tile in working set: {parent_key}")
        selected_parent_keys.add(parent_key)
        if parent_key in hero_keys:
            level = "HERO"
            selected = parent.GetChild("Details")
        elif parent_key in mid_keys:
            level = "MID"
            selected = parent.GetChild("DetailsMid")
        else:
            level = "FAR"
            selected = parent.GetChild("DetailsFar")
        if not selected.IsValid() or not selected.HasPayload():
            raise RuntimeError(
                f"tile {parent.GetPath()} exposes no {level} detail payload"
            )
        result[level].append(selected.GetPath())
    for paths in result.values():
        paths.sort(key=str)
    if (
        len(result["HERO"]) != EDITOR_HERO_TILE_CAP
        or len(result["MID"]) != EDITOR_MID_TILE_CAP
        or sum(len(paths) for paths in result.values()) != len(candidates)
    ):
        raise RuntimeError("detail working set violates its exclusive LOD contract")
    return result


def _terrain_samples(stage: Usd.Stage) -> list[TerrainGrid]:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    result: list[TerrainGrid] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if "terrain" not in str(prim.GetPath()).lower():
            continue
        raw = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not raw:
            continue
        side = math.isqrt(len(raw))
        if side * side != len(raw):
            raise RuntimeError(
                f"terrain mesh is not a square grid: {prim.GetPath()}"
            )
        matrix = cache.GetLocalToWorldTransform(prim)
        top_left = Gf.Vec3d(matrix.Transform(raw[0]))
        top_right = Gf.Vec3d(matrix.Transform(raw[side - 1]))
        bottom_left = Gf.Vec3d(matrix.Transform(raw[(side - 1) * side]))
        bottom_right = Gf.Vec3d(matrix.Transform(raw[-1]))
        xs = (
            top_left[0],
            top_right[0],
            bottom_left[0],
            bottom_right[0],
        )
        ys = (
            top_left[1],
            top_right[1],
            bottom_left[1],
            bottom_right[1],
        )
        result.append(
            (
                min(xs),
                min(ys),
                max(xs),
                max(ys),
                side,
                raw,
                matrix,
                (top_left[0] + bottom_left[0]) * 0.5,
                (top_right[0] + bottom_right[0]) * 0.5,
                (top_left[1] + top_right[1]) * 0.5,
                (bottom_left[1] + bottom_right[1]) * 0.5,
            )
        )
    if not result:
        raise RuntimeError("the composed Z16 root exposes no terrain mesh samples")
    return result


def _height_at(
    samples: list[TerrainGrid],
    x: float,
    y: float,
) -> float:
    candidates = [
        sample
        for sample in samples
        for xmin, ymin, xmax, ymax in [sample[:4]]
        if xmin - 1.0 <= x <= xmax + 1.0 and ymin - 1.0 <= y <= ymax + 1.0
    ]
    if not candidates:
        raise RuntimeError(f"no terrain tile covers response position ({x}, {y})")
    nearest: Gf.Vec3d | None = None
    distance = math.inf
    for (
        _xmin,
        _ymin,
        _xmax,
        _ymax,
        side,
        points,
        matrix,
        x_start,
        x_end,
        y_start,
        y_end,
    ) in candidates:
        column_ratio = (
            0.0
            if abs(x_end - x_start) <= 1.0e-9
            else (x - x_start) / (x_end - x_start)
        )
        row_ratio = (
            0.0
            if abs(y_end - y_start) <= 1.0e-9
            else (y - y_start) / (y_end - y_start)
        )
        column = max(0, min(side - 1, round(column_ratio * (side - 1))))
        row = max(0, min(side - 1, round(row_ratio * (side - 1))))
        for nearby_row in range(max(0, row - 2), min(side, row + 3)):
            for nearby_column in range(
                max(0, column - 2), min(side, column + 3)
            ):
                point = Gf.Vec3d(
                    matrix.Transform(points[nearby_row * side + nearby_column])
                )
                current = (point[0] - x) ** 2 + (point[1] - y) ** 2
                if current < distance:
                    distance = current
                    nearest = point
    if nearest is None:
        raise RuntimeError(f"terrain tile at ({x}, {y}) contains no point")
    return float(nearest[2])


def _scenario_steps(response: dict[str, Any], scenario_id: str) -> list[dict[str, Any]]:
    for deployment in response.get("scenario_deployments", []):
        if deployment.get("scenario_id") == scenario_id:
            steps = deployment.get("steps")
            if isinstance(steps, list) and steps:
                return steps
    raise RuntimeError(f"response contract has no steps for {scenario_id}")


def _fire_steps(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for day in scenario.get("days", []):
        for hour in day.get("hours", []):
            result.append(hour)
    if not result:
        raise RuntimeError(f"fire scenario {scenario.get('scenario_id')} has no steps")
    return result


def _author_camera(
    stage: Usd.Stage,
    camera_record: dict[str, Any],
    fire_steps: list[dict[str, Any]],
    contract_to_scene_offset: Gf.Vec3d,
) -> None:
    camera_id = str(camera_record["camera_id"])
    camera = UsdGeom.Camera.Define(
        stage, f"{SCENARIO_ROOT}/Cameras/{_identifier(camera_id)}"
    )
    pose = camera_record["pose_local"]
    position = (
        Gf.Vec3d(*map(float, pose["position_m"]))
        + contract_to_scene_offset
    )
    xform = UsdGeom.Xformable(camera)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(position)
    orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    intrinsics = camera_record["intrinsics"]
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr().Set(float(intrinsics["focal_length_mm"]))
    camera.CreateHorizontalApertureAttr().Set(
        float(intrinsics["horizontal_aperture_mm"])
    )
    camera.CreateVerticalApertureAttr().Set(
        float(intrinsics["vertical_aperture_mm"])
    )
    camera.CreateClippingRangeAttr().Set(
        Gf.Vec2f(
            float(intrinsics["near_clip_m"]),
            float(intrinsics["far_clip_m"]),
        )
    )
    camera.CreateFStopAttr().Set(float(intrinsics["f_stop"]))
    prim = camera.GetPrim()
    _string_attr(prim, "fireviewer:cameraId", camera_id)
    _string_attr(prim, "fireviewer:viewClass", str(camera_record["view_class"]))
    _string_attr(
        prim,
        "fireviewer:intrinsicsJson",
        json.dumps(intrinsics, separators=(",", ":"), sort_keys=True),
    )
    for step in fire_steps:
        fire = step["fire"]
        time = Usd.TimeCode(float(fire["simulation_time_seconds"]))
        target = (
            Gf.Vec3d(*map(float, fire["active_flame_centroid_local_m"]))
            + contract_to_scene_offset
        )
        orient.Set(_look_at_quaternion(position, target), time)


def _author_fire_track(
    stage: Usd.Stage,
    scenario: dict[str, Any],
    fire_steps: list[dict[str, Any]],
    source_root: Path,
    contract_to_scene_offset: Gf.Vec3d,
) -> None:
    fire_prim = UsdGeom.Xform.Define(stage, f"{SCENARIO_ROOT}/Fire").GetPrim()
    flow_asset = (
        source_root.parent / "assets" / "flow-fire-source-yup-centimetres.usda"
    )
    if not flow_asset.is_file():
        raise RuntimeError(f"the packaged NVIDIA Flow fire asset is absent: {flow_asset}")
    fire_prim.GetReferences().AddReference(str(flow_asset), "/Fire")
    fire_xform = UsdGeom.Xformable(fire_prim)
    fire_translate = fire_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    fire_rotate = fire_xform.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble)
    fire_scale = fire_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    fire_rotate.Set(90.0)
    UsdGeom.Imageable(fire_prim).MakeVisible()
    _string_attr(fire_prim, "fireviewer:scenarioId", str(scenario["scenario_id"]))
    state = fire_prim.CreateAttribute(
        "fireviewer:state", Sdf.ValueTypeNames.String, custom=True
    )
    centroid = _double3_attr(fire_prim, "fireviewer:activeFlameCentroidLocalM")
    scalar_fields = {
        "active_area_m2": "activeAreaM2",
        "active_flame_front_radius_m": "activeFlameFrontRadiusM",
        "burned_area_m2": "burnedAreaM2",
        "fire_front_length_m": "fireFrontLengthM",
        "max_flame_height_m": "maxFlameHeightM",
        "smoke_column_height_m": "smokeColumnHeightM",
    }
    scalars = {
        key: _double_attr(fire_prim, f"fireviewer:{name}")
        for key, name in scalar_fields.items()
    }
    weather_prim = UsdGeom.Scope.Define(
        stage, f"{SCENARIO_ROOT}/Weather"
    ).GetPrim()
    weather_attrs: dict[str, Usd.Attribute] = {}
    for step in fire_steps:
        fire = step["fire"]
        time = Usd.TimeCode(float(fire["simulation_time_seconds"]))
        state.Set(str(fire["state"]), time)
        scene_centroid = (
            Gf.Vec3d(*map(float, fire["active_flame_centroid_local_m"]))
            + contract_to_scene_offset
        )
        centroid.Set(scene_centroid, time)
        fire_translate.Set(scene_centroid, time)
        horizontal_scale = max(
            0.5, float(fire["active_flame_front_radius_m"]) / 10.0
        )
        vertical_scale = max(0.2, float(fire["max_flame_height_m"]) / 10.0)
        fire_scale.Set(
            Gf.Vec3d(horizontal_scale, horizontal_scale, vertical_scale), time
        )
        for key, attr in scalars.items():
            attr.Set(float(fire[key]), time)
        for key, value in step["weather"].items():
            attr = weather_attrs.get(key)
            if attr is None:
                attr = _double_attr(
                    weather_prim, f"fireviewer:{_identifier(key)}"
                )
                weather_attrs[key] = attr
            attr.Set(float(value), time)


def _actor_camera_intrinsics(record: dict[str, Any]) -> dict[str, Any]:
    aerial = str(record["placement_class"]) == "aerial"
    width = 1920
    height = 1080
    horizontal_aperture = 36.0
    vertical_aperture = 20.25
    focal_length = 28.0 if aerial else 24.0
    return {
        "model": "pinhole",
        "width_px": width,
        "height_px": height,
        "fx_px": round(focal_length / horizontal_aperture * width, 6),
        "fy_px": round(focal_length / vertical_aperture * height, 6),
        "cx_px": width / 2.0,
        "cy_px": height / 2.0,
        "near_clip_m": 0.1,
        "far_clip_m": 30_000.0,
        "focal_length_mm": focal_length,
        "horizontal_aperture_mm": horizontal_aperture,
        "vertical_aperture_mm": vertical_aperture,
        "f_stop": 5.6 if aerial else 4.0,
        "mount": "onboard_observation" if aerial else "roof_observation",
    }


def _define_actor_camera(
    stage: Usd.Stage,
    *,
    selection_id: str,
    record: dict[str, Any],
) -> tuple[UsdGeom.XformOp, UsdGeom.XformOp, float]:
    source_name = str(record["source_name"])
    camera_id = f"ACTOR_CAM_{_identifier(source_name).upper()}"
    camera = UsdGeom.Camera.Define(
        stage, f"{SCENARIO_ROOT}/ActorCameras/{_identifier(camera_id)}"
    )
    intrinsics = _actor_camera_intrinsics(record)
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr().Set(float(intrinsics["focal_length_mm"]))
    camera.CreateHorizontalApertureAttr().Set(
        float(intrinsics["horizontal_aperture_mm"])
    )
    camera.CreateVerticalApertureAttr().Set(
        float(intrinsics["vertical_aperture_mm"])
    )
    camera.CreateClippingRangeAttr().Set(
        Gf.Vec2f(
            float(intrinsics["near_clip_m"]),
            float(intrinsics["far_clip_m"]),
        )
    )
    camera.CreateFStopAttr().Set(float(intrinsics["f_stop"]))
    prim = camera.GetPrim()
    aerial = str(record["placement_class"]) == "aerial"
    _string_attr(prim, "fireviewer:cameraId", camera_id)
    _string_attr(
        prim,
        "fireviewer:viewClass",
        "onboard_aerial_fire_observation"
        if aerial
        else "roof_ground_fire_observation",
    )
    _string_attr(
        prim,
        "fireviewer:intrinsicsJson",
        json.dumps(intrinsics, separators=(",", ":"), sort_keys=True),
    )
    _string_attr(prim, "fireviewer:selectionId", selection_id)
    xform = UsdGeom.Xformable(camera)
    translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    return translate, orient, 1.5 if aerial else 3.2


def _author_response_actors(
    stage: Usd.Stage,
    *,
    response: dict[str, Any],
    scenario_id: str,
    volume_root: Path,
    terrain: list[TerrainGrid],
    contract_to_scene_offset: Gf.Vec3d,
) -> None:
    library = response["asset_library"]
    steps = _scenario_steps(response, scenario_id)
    actor_ops: dict[str, tuple[UsdGeom.XformOp, UsdGeom.XformOp, Usd.Attribute, Usd.Attribute]] = {}
    camera_ops: dict[
        str, tuple[UsdGeom.XformOp, UsdGeom.XformOp, float]
    ] = {}
    for step in steps:
        time = Usd.TimeCode(float(step["simulation_time_seconds"]))
        for state in step["actor_states"]:
            selection_id = str(state["selection_id"])
            actor_path = f"{SCENARIO_ROOT}/ResponseActors/Actor_{_identifier(selection_id)}"
            if selection_id not in actor_ops:
                record = library[selection_id]
                actor = UsdGeom.Xform.Define(stage, actor_path)
                model = UsdGeom.Xform.Define(stage, f"{actor_path}/Model")
                model.GetPrim().GetReferences().AddReference(
                    str(_asset_path(volume_root, str(record["wrapper_path"]))),
                    "/Asset",
                )
                lod = model.GetPrim().GetVariantSets().GetVariantSet("lodVariant")
                if "HERO" not in lod.GetVariantNames():
                    raise RuntimeError(f"{selection_id} wrapper exposes no HERO variant")
                lod.SetVariantSelection("HERO")
                xform = UsdGeom.Xformable(actor)
                translate = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                heading = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
                visibility = UsdGeom.Imageable(actor).CreateVisibilityAttr()
                operational = actor.GetPrim().CreateAttribute(
                    "fireviewer:operationalState",
                    Sdf.ValueTypeNames.String,
                    custom=True,
                )
                _string_attr(actor.GetPrim(), "fireviewer:selectionId", selection_id)
                _string_attr(
                    actor.GetPrim(), "fireviewer:teamId", str(record["team_id"])
                )
                actor_ops[selection_id] = (
                    translate,
                    heading,
                    visibility,
                    operational,
                )
                camera_ops[selection_id] = _define_actor_camera(
                    stage,
                    selection_id=selection_id,
                    record=record,
                )
            translate, heading, visibility, operational = actor_ops[selection_id]
            camera_translate, camera_orient, camera_mount_height = camera_ops[
                selection_id
            ]
            binding = state["pose_binding"]
            contract_x, contract_y = map(
                float, binding["position_xy_local_m"]
            )
            x = contract_x + contract_to_scene_offset[0]
            y = contract_y + contract_to_scene_offset[1]
            terrain_z = _height_at(terrain, x, y)
            z_binding = binding["z_binding"]
            if "altitude_agl_m" in z_binding:
                z = terrain_z + float(z_binding["altitude_agl_m"])
            else:
                anchor = z_binding.get("asset_ground_anchor_m", [0.0, 0.0, 0.0])
                z = terrain_z - float(anchor[2])
            translate.Set(Gf.Vec3d(x, y, z), time)
            camera_position = Gf.Vec3d(x, y, z + camera_mount_height)
            camera_target = (
                Gf.Vec3d(
                    *map(float, step["active_flame_centroid_local_m"])
                )
                + contract_to_scene_offset
            )
            camera_translate.Set(camera_position, time)
            camera_orient.Set(
                _look_at_quaternion(camera_position, camera_target), time
            )
            heading.Set(float(binding["heading_degrees"]), time)
            visibility.Set(
                UsdGeom.Tokens.inherited
                if bool(state["visible_in_capture"])
                else UsdGeom.Tokens.invisible,
                time,
            )
            operational.Set(str(state["operational_state"]), time)


def _author_overlay() -> Path:
    source_root = _required_path("FW_Z16_SOURCE_ROOT")
    output = _output_path()
    if output == source_root:
        raise RuntimeError("overlay output must differ from the source root")
    scenario_root = Path(
        os.getenv(
            "FW_Z16_SCENARIOS_ROOT",
            str(Path(__file__).resolve().parents[1] / "scenarios"),
        )
    ).resolve()
    volume_root = Path(
        os.getenv("FW_Z16_VOLUME_ROOT", "/workspace/fireviewer-omniverse")
    ).resolve()
    cameras = _read_json(
        scenario_root / "z16-capture-cameras-v1.json", "camera contract"
    )
    fires = _read_json(
        scenario_root / "z16-fire-scenarios-v1.json", "fire scenario contract"
    )
    response = _read_json(
        scenario_root / "z16-response-deployments-v1.json",
        "response deployment contract",
    )
    camera_records = cameras.get("cameras")
    scenarios = fires.get("scenarios")
    if not isinstance(camera_records, list) or len(camera_records) != 40:
        raise RuntimeError("camera contract must expose exactly 40 cameras")
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError("fire scenario contract exposes no scenario")
    local_bounds = cameras.get("zone", {}).get("local_bounds_m", {})
    try:
        contract_to_scene_offset = Gf.Vec3d(
            -0.5
            * (
                float(local_bounds["xmin"])
                + float(local_bounds["xmax"])
            ),
            -0.5
            * (
                float(local_bounds["ymin"])
                + float(local_bounds["ymax"])
            ),
            0.0,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "camera contract has no usable local zone bounds"
        ) from exc

    if output.exists():
        if os.getenv("FW_Z16_REPLACE_OVERLAY", "").strip() != "1":
            raise RuntimeError(
                "overlay already exists; set FW_Z16_REPLACE_OVERLAY=1 to replace it"
            )
        output.unlink()
    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = [Sdf.ComputeAssetPathRelativeToLayer(layer, str(source_root))]
    stage = Usd.Stage.Open(layer, Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError("could not compose the Z16 source root")
    stage.SetEditTarget(layer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(1.0)
    stage.SetStartTimeCode(0.0)
    maximum_time = max(
        float(step["fire"]["simulation_time_seconds"])
        for scenario in scenarios
        for step in _fire_steps(scenario)
    )
    stage.SetEndTimeCode(maximum_time)
    root = UsdGeom.Scope.Define(stage, OVERLAY_ROOT).GetPrim()
    _string_attr(root, "fireviewer:sourceRoot", str(source_root))
    _string_attr(root, "fireviewer:cameraContractId", str(cameras["contract_id"]))
    _string_attr(root, "fireviewer:fireContractId", str(fires["contract_id"]))
    _string_attr(root, "fireviewer:responseContractId", str(response["contract_id"]))
    _string_attr(root, "fireviewer:horizontalCrs", "EPSG:2154")
    _double3_attr(root, "fireviewer:contractToSceneOffsetLocalM").Set(
        contract_to_scene_offset
    )
    root.SetCustomDataByKey("fireviewer:sourceSceneReadOnly", True)

    _select_highest_lods(stage)
    _load_terrain(stage)
    terrain = _terrain_samples(stage)
    scenario_prim = UsdGeom.Xform.Define(stage, SCENARIO_ROOT).GetPrim()
    variants = scenario_prim.GetVariantSets().AddVariantSet("fireScenario")
    for scenario in scenarios:
        scenario_id = str(scenario["scenario_id"])
        variant_name = _identifier(scenario_id)
        variants.AddVariant(variant_name)
        variants.SetVariantSelection(variant_name)
        with variants.GetVariantEditContext():
            fire_steps = _fire_steps(scenario)
            UsdGeom.Scope.Define(stage, f"{SCENARIO_ROOT}/Cameras")
            UsdGeom.Scope.Define(stage, f"{SCENARIO_ROOT}/ActorCameras")
            UsdGeom.Scope.Define(stage, f"{SCENARIO_ROOT}/ResponseActors")
            for record in camera_records:
                _author_camera(
                    stage,
                    record,
                    fire_steps,
                    contract_to_scene_offset,
                )
            _author_fire_track(
                stage,
                scenario,
                fire_steps,
                source_root,
                contract_to_scene_offset,
            )
            _author_response_actors(
                stage,
                response=response,
                scenario_id=scenario_id,
                volume_root=volume_root,
                terrain=terrain,
                contract_to_scene_offset=contract_to_scene_offset,
            )
    requested = os.getenv("FW_Z16_SCENARIO_ID", str(scenarios[0]["scenario_id"]))
    requested_variant = _identifier(requested)
    if requested_variant not in variants.GetVariantNames():
        raise RuntimeError(f"unknown FW_Z16_SCENARIO_ID: {requested}")
    variants.SetVariantSelection(requested_variant)
    stage.GetRootLayer().Save()
    return output


async def _open_overlay(path: Path) -> None:
    import carb
    import omni.kit.app
    import omni.kit.viewport.utility
    import omni.timeline
    import omni.usd

    settings = carb.settings.get_settings()
    settings.set("/app/runLoops/main/rateLimitEnabled", True)
    settings.set("/app/runLoops/main/rateLimitFrequency", 60)
    context = omni.usd.get_context()
    result, error = await context.open_stage_async(
        str(path), load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE
    )
    if not result:
        raise RuntimeError(f"Composer could not open Z16 overlay: {error}")
    for _ in range(4):
        await omni.kit.app.get_app().next_update_async()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Composer exposes no opened overlay stage")
    stage.SetEditTarget(stage.GetSessionLayer())
    _select_highest_lods(stage)
    terrain_paths = [
        prim.GetPath()
        for prim in stage.TraverseAll()
        if prim.HasPayload() and prim.GetName() == "Terrain"
    ]
    for index, payload_path in enumerate(terrain_paths, start=1):
        stage.Load(payload_path)
        if index % 8 == 0:
            await omni.kit.app.get_app().next_update_async()
    selected = stage.GetPrimAtPath(SCENARIO_ROOT).GetVariantSets().GetVariantSet(
        "fireScenario"
    ).GetVariantSelection()
    camera, camera_id = _preferred_editor_camera(stage)
    fire = stage.GetPrimAtPath(f"{SCENARIO_ROOT}/Fire")
    if not fire.IsValid():
        raise RuntimeError(f"selected scenario {selected} exposes no fire")
    detail_paths = _detail_lods_for_camera_fire_corridor(
        stage,
        camera=camera,
        fire=fire,
    )
    if not detail_paths["HERO"] or not detail_paths["MID"]:
        raise RuntimeError("camera-to-fire corridor exposes incomplete detail LODs")
    for level in ("FAR", "MID", "HERO"):
        for index, payload_path in enumerate(detail_paths[level], start=1):
            stage.Load(payload_path)
            if index % (32 if level == "FAR" else 4) == 0:
                await omni.kit.app.get_app().next_update_async()
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    timeline.set_current_time(0.0)
    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is not None:
        viewport.camera_path = str(camera.GetPath())
    print(
        "Z16_EDITOR_STREAMING_READY "
        f"terrain={len(terrain_paths)} "
        f"detailsHero={len(detail_paths['HERO'])} "
        f"detailsMid={len(detail_paths['MID'])} "
        f"detailsFar={len(detail_paths['FAR'])} "
        f"camera={camera_id} scenario={selected} time=0",
        flush=True,
    )


def _task_done(task: asyncio.Task[Any]) -> None:
    error = task.exception()
    if error is not None:
        raise error


def main() -> None:
    if os.getenv("FW_Z16_REUSE_OVERLAY", "").strip() == "1":
        output = _output_path()
        if not output.is_file():
            raise RuntimeError(f"reusable overlay is absent: {output}")
        print(f"Z16_EDITOR_OVERLAY_REUSED path={output}", flush=True)
    else:
        output = _author_overlay()
        print(f"Z16_EDITOR_OVERLAY_AUTHORED path={output}", flush=True)
    if os.getenv("FW_Z16_OPEN_EDITOR", "1").strip() != "0":
        try:
            import omni.usd  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "FW_Z16_OPEN_EDITOR=1 requires execution inside Omniverse Kit"
            ) from exc
        task = asyncio.ensure_future(_open_overlay(output))
        task.add_done_callback(_task_done)


if __name__ == "__main__":
    main()
