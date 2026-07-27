"""Headless Isaac Sim Replicator generation with explicit, validated inputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any


PROCEDURAL_SCENE_MODE = "procedural_fire_landscape"
USD_SCENE_MODE = "usd"
ALLOWED_ENVIRONMENTS = frozenset({"rural", "forest", "mountain"})
ALLOWED_LIGHTING = frozenset({"day", "night"})
ALLOWED_VIEWPOINTS = frozenset({"road", "building", "valley"})
ALLOWED_ANNOTATIONS = frozenset(
    {
        "rgb",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "semantic_segmentation",
        "instance_segmentation",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "pointcloud",
    }
)
DEFAULT_ANNOTATIONS = (
    "rgb",
    "bounding_box_2d_tight",
    "semantic_segmentation",
    "instance_segmentation",
    "distance_to_camera",
    "normals",
)


def _vector(value: Any, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain three numbers")
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def _validate_camera_poses(poses: Any) -> list[dict[str, Any]]:
    if not isinstance(poses, list) or not poses:
        raise ValueError("camera_poses must be a non-empty list")
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict):
            raise ValueError("camera pose entries must be objects")
        _vector(pose.get("position"), field=f"camera_poses[{index}].position")
        _vector(pose.get("look_at"), field=f"camera_poses[{index}].look_at")
        if not str(pose.get("id", "")).strip():
            raise ValueError("camera poses require stable ids")
    return poses


def build_procedural_camera_poses(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a deterministic orbit appropriate for the requested viewpoint."""
    seed = int(scenario["seed"])
    rng = random.Random(seed)
    viewpoint = str(scenario["viewpoint"])
    bases = {
        "road": (-math.pi / 2, 24.0, 4.5),
        "building": (math.pi / 4, 21.0, 11.0),
        "valley": (-math.pi / 3, 33.0, 16.0),
    }
    base_angle, radius, height = bases[viewpoint]
    poses: list[dict[str, Any]] = []
    for index in range(16):
        orbit_offset = ((index % 8) - 3.5) * 0.075
        angle = base_angle + orbit_offset + rng.uniform(-0.02, 0.02)
        sampled_radius = radius + rng.uniform(-1.5, 1.5)
        sampled_height = height + rng.uniform(-0.8, 0.8)
        poses.append(
            {
                "id": f"{viewpoint}-{index:02d}",
                "position": [
                    round(sampled_radius * math.cos(angle), 6),
                    round(sampled_radius * math.sin(angle), 6),
                    round(sampled_height, 6),
                ],
                "look_at": [
                    round(rng.uniform(-0.8, 0.8), 6),
                    round(rng.uniform(-0.8, 0.8), 6),
                    round(3.0 + rng.uniform(-0.3, 0.5), 6),
                ],
            }
        )
    return poses


def load_scenario(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported scenario schema_version")
    scenario_id = str(payload.get("scenario_id", "")).strip()
    if not scenario_id:
        raise ValueError("scenario_id is required")
    output_value = str(payload.get("output_dir", "")).strip()
    if not output_value:
        raise ValueError("output_dir is required")
    frame_count = int(payload.get("frame_count", 0))
    if not 1 <= frame_count <= 10000:
        raise ValueError("frame_count must be between 1 and 10000")
    resolution = payload.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(int(component) < 64 or int(component) > 4096 for component in resolution)
    ):
        raise ValueError("resolution must contain two values between 64 and 4096")
    seed = int(payload.get("seed", -1))
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be between 0 and 4294967295")

    scene_mode = str(payload.get("scene_mode", USD_SCENE_MODE)).strip().lower()
    if scene_mode == PROCEDURAL_SCENE_MODE:
        if payload.get("environment") not in ALLOWED_ENVIRONMENTS:
            raise ValueError("unsupported procedural environment")
        if payload.get("lighting") not in ALLOWED_LIGHTING:
            raise ValueError("unsupported procedural lighting")
        if payload.get("viewpoint") not in ALLOWED_VIEWPOINTS:
            raise ValueError("unsupported procedural viewpoint")
        payload["camera_poses"] = _validate_camera_poses(
            payload.get("camera_poses") or build_procedural_camera_poses(payload)
        )
    elif scene_mode == USD_SCENE_MODE:
        scene_value = str(payload.get("scene_usd", "")).strip()
        if not scene_value:
            raise ValueError("scene_usd is required for USD scene mode")
        scene_path = Path(scene_value).resolve()
        if not scene_path.is_file() or scene_path.suffix.lower() not in {
            ".usd",
            ".usda",
            ".usdc",
        }:
            raise ValueError(f"provisioned USD scene is absent: {scene_path}")
        payload["scene_usd"] = str(scene_path)
        payload["camera_poses"] = _validate_camera_poses(payload.get("camera_poses"))
    else:
        raise ValueError(f"unsupported scene_mode: {scene_mode}")

    annotations = payload.get("annotations", list(DEFAULT_ANNOTATIONS))
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("annotations must be a non-empty list")
    normalized_annotations = [str(value).strip() for value in annotations]
    unsupported = set(normalized_annotations) - ALLOWED_ANNOTATIONS
    if unsupported:
        raise ValueError(f"unsupported annotations: {sorted(unsupported)}")
    if "rgb" not in normalized_annotations or "semantic_segmentation" not in normalized_annotations:
        raise ValueError("rgb and semantic_segmentation annotations are required")

    payload["scene_mode"] = scene_mode
    payload["scenario_id"] = scenario_id
    payload["output_dir"] = str(Path(output_value).resolve())
    payload["frame_count"] = frame_count
    payload["resolution"] = [int(value) for value in resolution]
    payload["seed"] = seed
    payload["annotations"] = normalized_annotations
    return payload


def build_pose_schedule(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one deterministic, explicit camera pose for every generated frame."""
    frame_count = int(scenario["frame_count"])
    poses = scenario["camera_poses"]
    return [dict(poses[index % len(poses)]) for index in range(frame_count)]


def _create_procedural_scene(rep: Any, scenario: dict[str, Any]) -> dict[str, int]:
    """Create a local-only landscape and explicit geometric fire/smoke proxies."""
    rng = random.Random(int(scenario["seed"]))
    create = rep.functional.create
    modify = rep.functional.modify
    create.xform(name="World")
    create.scope(parent="/World", name="Assets")
    create.scope(parent="/World", name="Materials")
    create.scope(parent="/World", name="Lights")
    create.scope(parent="/World", name="Cameras")

    palette = {
        "rural": (0.28, 0.36, 0.12),
        "forest": (0.08, 0.23, 0.07),
        "mountain": (0.32, 0.29, 0.24),
        "road": (0.12, 0.12, 0.12),
        "trunk": (0.25, 0.12, 0.04),
        "vegetation": (0.05, 0.30, 0.08),
        "mountain_rock": (0.30, 0.28, 0.27),
        "building": (0.52, 0.42, 0.30),
        "roof": (0.23, 0.08, 0.05),
        "fire": (1.0, 0.16, 0.01),
        "fire_hot": (1.0, 0.62, 0.02),
        "smoke": (0.15, 0.15, 0.17),
    }
    materials: dict[str, Any] = {}
    for name, color in palette.items():
        materials[name] = create.material(
            mdl="OmniPBR.mdl",
            diffuse_color_constant=color,
            reflection_roughness_constant=0.82,
            metallic_constant=0.0,
            parent="/World/Materials",
            name=f"{name.title().replace('_', '')}Material",
        )

    counts: dict[str, int] = {}

    def primitive(
        kind: str,
        *,
        name: str,
        label: str,
        material: str,
        position: tuple[float, float, float],
        scale: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Any:
        prim = getattr(create, kind)(
            parent="/World/Assets",
            name=name,
            position=position,
            rotation=rotation,
            scale=scale,
            semantics={"class": label},
        )
        modify.material(prim, materials[material])
        counts[label] = counts.get(label, 0) + 1
        return prim

    environment = str(scenario["environment"])
    primitive(
        "plane",
        name="Terrain",
        label="terrain",
        material=environment,
        position=(0.0, 0.0, 0.0),
        scale=(55.0, 55.0, 1.0),
    )
    primitive(
        "cube",
        name="Road",
        label="road",
        material="road",
        position=(0.0, -8.0, 0.08),
        scale=(42.0, 3.5, 0.08),
    )

    mountain_count = 10 if environment == "mountain" else 5
    for index in range(mountain_count):
        angle = math.tau * index / mountain_count + rng.uniform(-0.12, 0.12)
        radius = rng.uniform(32.0, 45.0)
        height = rng.uniform(7.0, 14.0) if environment == "mountain" else rng.uniform(4.0, 8.0)
        primitive(
            "cone",
            name=f"Mountain{index:02d}",
            label="mountain",
            material="mountain_rock",
            position=(radius * math.cos(angle), radius * math.sin(angle), height),
            scale=(height * 0.9, height * 0.9, height),
            rotation=(0.0, 0.0, rng.uniform(0.0, 360.0)),
        )

    tree_target = {"rural": 18, "forest": 44, "mountain": 24}[environment]
    tree_index = 0
    attempts = 0
    while tree_index < tree_target and attempts < tree_target * 20:
        attempts += 1
        x = rng.uniform(-30.0, 30.0)
        y = rng.uniform(-27.0, 30.0)
        if x * x + y * y < 75.0 or (-11.5 < y < -4.5):
            continue
        height = rng.uniform(2.4, 4.4)
        primitive(
            "cylinder",
            name=f"TreeTrunk{tree_index:02d}",
            label="vegetation",
            material="trunk",
            position=(x, y, height * 0.45),
            scale=(0.28, 0.28, height * 0.45),
        )
        primitive(
            "cone",
            name=f"TreeCrown{tree_index:02d}",
            label="vegetation",
            material="vegetation",
            position=(x, y, height + 1.1),
            scale=(1.25, 1.25, height * 0.72),
            rotation=(0.0, 0.0, rng.uniform(0.0, 360.0)),
        )
        tree_index += 1
    if tree_index != tree_target:
        raise RuntimeError("procedural tree placement exhausted deterministic attempts")

    building_positions = [(11.0, 9.0)]
    if environment == "rural":
        building_positions.append((-14.0, 13.0))
    for index, (x, y) in enumerate(building_positions):
        primitive(
            "cube",
            name=f"Building{index:02d}",
            label="building",
            material="building",
            position=(x, y, 2.5),
            scale=(3.8, 3.2, 2.5),
        )
        primitive(
            "cone",
            name=f"Roof{index:02d}",
            label="building",
            material="roof",
            position=(x, y, 6.1),
            scale=(4.5, 4.0, 1.8),
            rotation=(0.0, 0.0, 45.0),
        )

    # These three primitives make the review points a geometric contract rather
    # than an estimate inferred from pixels. The points are projected from the
    # recorded USD camera and remain inside their corresponding visible proxy.
    primitive(
        "cone",
        name="ActiveFireAnchor",
        label="fire",
        material="fire_hot",
        position=(0.0, 0.0, 1.0),
        scale=(0.72, 0.72, 1.0),
    )
    primitive(
        "cone",
        name="VisibleFireFrontAnchor",
        label="fire",
        material="fire",
        position=(2.45, 0.0, 0.95),
        scale=(0.62, 0.62, 0.95),
    )
    primitive(
        "sphere",
        name="SmokeColumnBaseAnchor",
        label="smoke",
        material="smoke",
        position=(0.0, 0.0, 4.15),
        scale=(1.0, 1.0, 1.0),
    )

    for index in range(14):
        angle = rng.uniform(0.0, math.tau)
        radius = rng.uniform(0.0, 3.2)
        height = rng.uniform(1.0, 3.7)
        primitive(
            "cone",
            name=f"Fire{index:02d}",
            label="fire",
            material="fire_hot" if index % 3 == 0 else "fire",
            position=(radius * math.cos(angle), radius * math.sin(angle), height * 0.62),
            scale=(rng.uniform(0.35, 0.9), rng.uniform(0.35, 0.9), height),
            rotation=(rng.uniform(-8.0, 8.0), rng.uniform(-8.0, 8.0), rng.uniform(0.0, 360.0)),
        )
    for index in range(9):
        altitude = 4.0 + index * 1.15
        primitive(
            "sphere",
            name=f"Smoke{index:02d}",
            label="smoke",
            material="smoke",
            position=(
                rng.uniform(-1.2, 1.2) + index * 0.18,
                rng.uniform(-1.0, 1.0),
                altitude,
            ),
            scale=(
                1.0 + index * 0.13,
                1.0 + index * 0.13,
                0.8 + index * 0.11,
            ),
        )

    create.dome_light(
        intensity=1100 if scenario["lighting"] == "day" else 320,
        parent="/World/Lights",
        name="DomeLight",
    )
    return counts


def generate(scenario_path: Path) -> Path:
    scenario = load_scenario(scenario_path)
    import isaacsim  # noqa: F401 - initializes the pip namespace package
    from isaacsim.simulation_app import SimulationApp

    application = SimulationApp({"headless": True})
    try:
        import carb.settings
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.flowusd", True)
        if bool(scenario.get("flow_required")) and not manager.is_extension_enabled(
            "omni.flowusd"
        ):
            raise RuntimeError("scenario requires Flow but omni.flowusd is unavailable")
        rep.orchestrator.set_capture_on_play(False)
        rep.set_global_seed(int(scenario["seed"]))
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        representation: dict[str, Any]
        if scenario["scene_mode"] == USD_SCENE_MODE:
            context = omni.usd.get_context()
            context.open_stage(str(Path(scenario["scene_usd"]).resolve()))
            deadline = time.monotonic() + 120
            while context.is_loading() and time.monotonic() < deadline:
                application.update()
            if context.is_loading():
                raise RuntimeError("USD scene did not finish loading")
            representation = {"kind": "provisioned_usd"}
            camera_parent = None
        else:
            omni.usd.get_context().new_stage()
            rep.settings.set_stage_up_axis("Z")
            rep.settings.set_stage_meters_per_unit(1.0)
            counts = _create_procedural_scene(rep, scenario)
            representation = {
                "kind": "geometric_fire_smoke_proxy",
                "object_counts": counts,
            }
            camera_parent = "/World/Cameras"

        pose_schedule = build_pose_schedule(scenario)
        first = pose_schedule[0]
        camera_arguments: dict[str, Any] = {
            "position": _vector(first["position"], field="position"),
            "look_at": _vector(first["look_at"], field="look_at"),
            "name": "DatasetCamera",
        }
        if camera_parent is not None:
            camera_arguments["parent"] = camera_parent
        camera = rep.functional.create.camera(**camera_arguments)
        resolution = tuple(int(value) for value in scenario["resolution"])
        render_product = rep.create.render_product(camera, resolution, name="DatasetRenderProduct")
        output_dir = Path(str(scenario["output_dir"])).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(output_dir))
        writer = rep.writers.get("BasicWriter")
        writer.initialize(
            backend=backend,
            **{annotation: True for annotation in scenario["annotations"]},
        )
        writer.attach(render_product)

        rt_subframes = int(scenario.get("rt_subframes", 4))
        if not 1 <= rt_subframes <= 64:
            raise ValueError("rt_subframes must be between 1 and 64")
        for index, pose in enumerate(pose_schedule):
            rep.functional.modify.pose(
                camera,
                position_value=_vector(pose["position"], field="position"),
                look_at_value=_vector(pose["look_at"], field="look_at"),
                look_at_up_axis=(0.0, 0.0, 1.0),
                write_to_usd=True,
            )
            rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
            if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(pose_schedule):
                print(
                    f"fireviewer sdg generation: frame={index + 1}/{len(pose_schedule)}",
                    flush=True,
                )
        rep.orchestrator.wait_until_complete()
        writer.detach()
        render_product.destroy()

        manifest = {
            "schema_version": 1,
            "scenario_id": scenario["scenario_id"],
            "scene_mode": scenario["scene_mode"],
            "frame_count": scenario["frame_count"],
            "resolution": scenario["resolution"],
            "seed": scenario["seed"],
            "annotations": scenario["annotations"],
            "representation": representation,
            "frames": [
                {
                    "frame_index": index,
                    "pose_id": pose["id"],
                    "position": pose["position"],
                    "look_at": pose["look_at"],
                }
                for index, pose in enumerate(pose_schedule)
            ],
            "synthetic": True,
            "human_review_required": True,
            "usable_for_training": False,
        }
        for field in ("environment", "lighting", "viewpoint"):
            if field in scenario:
                manifest[field] = scenario[field]
        manifest_path = output_dir / "fireviewer-sdg-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return manifest_path
    finally:
        application.close(
            wait_for_replicator=False,
            skip_cleanup=True,
            exit_code=1 if sys.exc_info()[0] is not None else 0,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    arguments = parser.parse_args()
    print(generate(arguments.scenario), flush=True)


if __name__ == "__main__":
    main()
