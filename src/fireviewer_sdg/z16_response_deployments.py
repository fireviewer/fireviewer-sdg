"""Prepare the Z16 responder timeline from already materialized pod assets.

This module is deliberately authoring-only.  It does not download assets,
open Kit, modify a remote stage, advance the fire simulation, or render.  The
output binds the existing camera and fire contracts to deterministic responder
poses that can be authored after the scene build is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
STATE = "Z16_RESPONSE_DEPLOYMENTS_PREPARED_PRE_ACCEPTANCE"
CONTRACT_ID = "Z16-RESPONSE-DEPLOYMENTS-3-V1"
CAMERA_CONTRACT_ID = "Z16-CAMERA-40-V1"
FIRE_CONTRACT_ID = "Z16-FIRE-SCENARIOS-3-V1"
ASSET_GROUP_ID = "chrome-fire-response-group-2026-07-29"
SOURCE_INVENTORY_SHA256 = (
    "fe7feb434fb19ce03197ac76998ad525"
    "9878a11b6d4f1c845840bd469d7c9f02"
)
VOLUME_ROOT = "/workspace/fireviewer-omniverse"
EXPECTED_CAMERA_COUNT = 40
CAPTURE_HOURS = ("08:00", "14:00", "20:00")
ZONE_BOUNDS = (0.0, 0.0, 20_000.0, 20_000.0)
POSITION_MARGIN_M = 250.0


ASSETS: tuple[dict[str, Any], ...] = (
    {
        "selection_id": "94ef5c37c3c543fd9efbaa571a7a7590",
        "source_name": "Po2",
        "placement_class": "aerial",
        "team_id": "TEAM-AERIAL-RECON-ALPHA",
        "operational_role": "aerial_reconnaissance_visual_actor",
        "wrapper_path": (
            "actor-qa/actor-source-group-19/curated-available/wrappers/"
            "94ef5c37c3c543fd9efbaa571a7a7590/asset.usda"
        ),
        "wrapper_sha256": (
            "89bd15216c10232d64a343e22005e987"
            "e7209b42dbeb5d50e2f53558fbcb9bba"
        ),
        "content_lock_sha256": (
            "d577b37b2cf3d92c4a46ce59a29b497"
            "35f45afadc0f4df4fda6ad54c3caabcec"
        ),
        "license": "CC-BY-4.0",
        "ground_anchor_m": [
            -0.31826408948798335,
            -0.7392887046723262,
            -7.092005015520549e-17,
        ],
        "hero_dimensions_m": [
            8.800489801527345,
            9.023618432248815,
            2.9999998565638752,
        ],
        "orbit_radius_m": 2_400.0,
        "altitude_agl_m": 760.0,
        "orbit_phase_degrees": 25.0,
    },
    {
        "selection_id": "8f62ab4eacbc430186d85a7029d7d156",
        "source_name": "An2",
        "placement_class": "aerial",
        "team_id": "TEAM-AERIAL-RECON-BRAVO",
        "operational_role": "aerial_logistics_and_recon_visual_actor",
        "wrapper_path": (
            "actor-qa/actor-source-group-19/curated-available/wrappers/"
            "8f62ab4eacbc430186d85a7029d7d156/asset.usda"
        ),
        "wrapper_sha256": (
            "19c70a99d6edae4123cefa2bf571a819"
            "c8aa37439bf65cdd6f3bd1e170890ab2"
        ),
        "content_lock_sha256": (
            "aa52c02d8eda58e777dbe08365557fcd"
            "74f28627a04a2090c487e04324579f53"
        ),
        "license": "CC-BY-4.0",
        "ground_anchor_m": [
            1.9475668200122076,
            -0.011771757942023342,
            1.5544575690706706e-18,
        ],
        "hero_dimensions_m": [
            6.193846505790896,
            9.040792206826389,
            2.999999946799649,
        ],
        "orbit_radius_m": 3_250.0,
        "altitude_agl_m": 1_120.0,
        "orbit_phase_degrees": 155.0,
    },
    {
        "selection_id": "6246617aeb874e4793b21d5861eea8c9",
        "source_name": "Sikorsky CH-53E Sea Stallion",
        "placement_class": "aerial",
        "team_id": "TEAM-HEAVY-LIFT-RESCUE",
        "operational_role": "heavy_lift_and_rescue_visual_actor",
        "wrapper_path": (
            "actor-qa/actor-source-group-19/curated-available/wrappers/"
            "6246617aeb874e4793b21d5861eea8c9/asset.usda"
        ),
        "wrapper_sha256": (
            "f747fc745371eb1a2699c7a163172f357"
            "beb5dc6014168c175d27f221b2e8701"
        ),
        "content_lock_sha256": (
            "43ed103f2f6bf17312332d970a67893d"
            "7b6bc4c9fb22618e5f49c414edb8f7a4"
        ),
        "license": "CC-BY-4.0",
        "ground_anchor_m": [
            0.0008251952940554474,
            -1.7517905797409137,
            0.0019750976121025157,
        ],
        "hero_dimensions_m": [
            24.736481380690293,
            9.011486920159605,
            30.881864544111522,
        ],
        "orbit_radius_m": 1_350.0,
        "altitude_agl_m": 380.0,
        "orbit_phase_degrees": 275.0,
    },
    {
        "selection_id": "fc2b5eb692ca40c2b44357b62eb149df",
        "source_name": "Truck",
        "placement_class": "ground",
        "team_id": "TEAM-INCIDENT-LOGISTICS",
        "operational_role": "incident_command_and_logistics_visual_actor",
        "wrapper_path": (
            "actor-qa/actor-source-group-19/curated-available/wrappers/"
            "fc2b5eb692ca40c2b44357b62eb149df/asset.usda"
        ),
        "wrapper_sha256": (
            "f7d118e308ee6732bf7a6276bb18a53c"
            "75e4ed0e81d419aae30bf0414466c3fc"
        ),
        "content_lock_sha256": (
            "dacd25a1aebe45f8d76d06d316b3f5a"
            "92b3ba6516960774dc3c6389c7c15df7a"
        ),
        "license": "CC-BY-4.0",
        "ground_anchor_m": [
            -0.04572027295584835,
            0.6934172070176481,
            2.1468364170853892e-16,
        ],
        "hero_dimensions_m": [
            13.763499689234365,
            12.67946724836247,
            2.9999998859627253,
        ],
        "upwind_offset_m": 2_100.0,
        "flank_offset_m": -320.0,
    },
    {
        "selection_id": "c573303be1f04e0c94cfa245c2f2ddcf",
        "source_name": "Construction truck",
        "placement_class": "ground",
        "team_id": "TEAM-FIREBREAK-ENGINEERING",
        "operational_role": "firebreak_engineering_visual_actor",
        "wrapper_path": (
            "actor-qa/actor-source-group-19/curated-available/wrappers/"
            "c573303be1f04e0c94cfa245c2f2ddcf/asset.usda"
        ),
        "wrapper_sha256": (
            "5be0e0d6e1441d66bca0c5bb818fbab"
            "a068291419ad2bb0e4571fda8a9e6ebdb"
        ),
        "content_lock_sha256": (
            "bfee08f56be883a888cdd76f6172a1f94"
            "f61e5f7c30fe95a7ba75acd8e49eecd"
        ),
        "license": "CC-BY-4.0",
        "ground_anchor_m": [
            -1.7271803818275089e-06,
            1.554462344177665e-06,
            1.6731172872082973e-17,
        ],
        "hero_dimensions_m": [
            11.590914787812002,
            11.568045883250946,
            4.80609583300672,
        ],
        "upwind_offset_m": 780.0,
        "flank_offset_m": 920.0,
    },
)


class Z16ResponseContractError(ValueError):
    """Raised when the camera, fire, asset, or deployment contract is invalid."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _clamp_xy(x: float, y: float) -> list[float]:
    xmin, ymin, xmax, ymax = ZONE_BOUNDS
    return [
        round(min(xmax - POSITION_MARGIN_M, max(xmin + POSITION_MARGIN_M, x)), 3),
        round(min(ymax - POSITION_MARGIN_M, max(ymin + POSITION_MARGIN_M, y)), 3),
    ]


def _heading_degrees(origin_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    return round(
        math.degrees(
            math.atan2(
                float(target_xy[1]) - float(origin_xy[1]),
                float(target_xy[0]) - float(origin_xy[0]),
            )
        )
        % 360.0,
        3,
    )


def _phase_for_hour(hour: str) -> str:
    return {
        "08:00": "morning_recon_and_staging",
        "14:00": "peak_suppression_support",
        "20:00": "evening_monitoring_and_redeployment",
    }[hour]


def _ground_pose(
    asset: Mapping[str, Any],
    *,
    fire_xy: Sequence[float],
    wind_from_degrees: float,
) -> dict[str, Any]:
    angle = math.radians(wind_from_degrees)
    # Meteorological direction is the direction the wind comes from.
    upwind = (math.sin(angle), math.cos(angle))
    right_flank = (math.cos(angle), -math.sin(angle))
    x = (
        float(fire_xy[0])
        + upwind[0] * float(asset["upwind_offset_m"])
        + right_flank[0] * float(asset["flank_offset_m"])
    )
    y = (
        float(fire_xy[1])
        + upwind[1] * float(asset["upwind_offset_m"])
        + right_flank[1] * float(asset["flank_offset_m"])
    )
    xy = _clamp_xy(x, y)
    return {
        "position_xy_local_m": xy,
        "z_binding": {
            "mode": "scene_terrain_surface_at_route_snap_xy_plus_asset_anchor",
            "asset_ground_anchor_m": asset["ground_anchor_m"],
        },
        "horizontal_binding": {
            "mode": "nearest_authored_route",
            "maximum_snap_distance_m": 600.0,
            "fallback": "block_stage_authoring",
        },
        "heading_degrees": _heading_degrees(xy, fire_xy),
        "target_local_xy_m": [round(float(v), 3) for v in fire_xy],
    }


def _aerial_pose(
    asset: Mapping[str, Any],
    *,
    fire_xyz: Sequence[float],
    step_index: int,
) -> dict[str, Any]:
    phase = float(asset["orbit_phase_degrees"])
    angle_degrees = (phase + step_index * 17.0) % 360.0
    angle = math.radians(angle_degrees)
    radius = float(asset["orbit_radius_m"])
    xy = _clamp_xy(
        float(fire_xyz[0]) + math.cos(angle) * radius,
        float(fire_xyz[1]) + math.sin(angle) * radius,
    )
    if asset["source_name"] == "Sikorsky CH-53E Sea Stallion":
        heading = _heading_degrees(xy, fire_xyz[:2])
        motion = "moving_hover_orbit_look_at_active_front"
    else:
        heading = round((angle_degrees + 90.0) % 360.0, 3)
        motion = "clockwise_fixed_wing_orbit"
    return {
        "position_xy_local_m": xy,
        "z_binding": {
            "mode": "scene_terrain_surface_at_xy_plus_agl",
            "altitude_agl_m": float(asset["altitude_agl_m"]),
        },
        "heading_degrees": heading,
        "motion": motion,
        "look_target_local_m": [round(float(v), 3) for v in fire_xyz],
    }


def _validate_inputs(
    camera_contract: Mapping[str, Any],
    fire_contract: Mapping[str, Any],
) -> None:
    if camera_contract.get("contract_id") != CAMERA_CONTRACT_ID:
        raise Z16ResponseContractError("unexpected Z16 camera contract")
    cameras = camera_contract.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != EXPECTED_CAMERA_COUNT:
        raise Z16ResponseContractError("Z16 camera contract must contain 40 cameras")
    if fire_contract.get("contract_id") != FIRE_CONTRACT_ID:
        raise Z16ResponseContractError("unexpected Z16 fire contract")
    if fire_contract.get("camera_contract_id") != CAMERA_CONTRACT_ID:
        raise Z16ResponseContractError("fire contract is not bound to the camera contract")
    if fire_contract.get("response_deployment_contract_id") != CONTRACT_ID:
        raise Z16ResponseContractError(
            "fire contract is not bound to the response deployment contract"
        )
    scenarios = fire_contract.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise Z16ResponseContractError("Z16 requires exactly three prepared scenarios")


def build_contract(
    camera_contract: Mapping[str, Any],
    fire_contract: Mapping[str, Any],
    *,
    camera_file_sha256: str,
    fire_file_sha256: str,
) -> dict[str, Any]:
    """Build the complete responder plan without touching a USD stage."""

    _validate_inputs(camera_contract, fire_contract)
    scenario_deployments: list[dict[str, Any]] = []
    timeline_step_count = 0
    actor_state_count = 0
    for scenario in fire_contract["scenarios"]:
        duration_days = int(scenario["duration_days"])
        expected_observations = duration_days * len(CAPTURE_HOURS) * EXPECTED_CAMERA_COUNT
        if scenario.get("observation_count") != expected_observations:
            raise Z16ResponseContractError(
                f"{scenario.get('scenario_id')} observation count is inconsistent"
            )
        steps: list[dict[str, Any]] = []
        step_index = 0
        for day in scenario["days"]:
            for observation in day["hours"]:
                hour = str(observation["capture_hour"])
                if hour not in CAPTURE_HOURS:
                    raise Z16ResponseContractError(f"unsupported capture hour {hour}")
                fire = observation["fire"]
                weather = observation["weather"]
                fire_xyz = fire["active_flame_centroid_local_m"]
                if (
                    not isinstance(fire_xyz, list)
                    or len(fire_xyz) != 3
                    or any(not _finite(value) for value in fire_xyz)
                ):
                    raise Z16ResponseContractError("active flame centroid is invalid")
                actor_states: list[dict[str, Any]] = []
                for asset in ASSETS:
                    if asset["placement_class"] == "ground":
                        pose = _ground_pose(
                            asset,
                            fire_xy=fire_xyz[:2],
                            wind_from_degrees=float(
                                weather["wind_direction_degrees"]
                            ),
                        )
                    else:
                        pose = _aerial_pose(
                            asset,
                            fire_xyz=fire_xyz,
                            step_index=step_index,
                        )
                    actor_states.append(
                        {
                            "actor_id": (
                                f"{scenario['scenario_id']}:"
                                f"{asset['team_id']}:{asset['selection_id'][:8]}"
                            ),
                            "selection_id": asset["selection_id"],
                            "team_id": asset["team_id"],
                            "operational_role": asset["operational_role"],
                            "operational_state": _phase_for_hour(hour),
                            "visible_in_capture": True,
                            "pose_binding": pose,
                        }
                    )
                steps.append(
                    {
                        "step_id": (
                            f"{scenario['scenario_id']}:"
                            f"D{int(day['day_index']):02d}:{hour}"
                        ),
                        "day_index": int(day["day_index"]),
                        "capture_hour": hour,
                        "simulation_time_seconds": int(
                            fire["simulation_time_seconds"]
                        ),
                        "active_flame_centroid_local_m": [
                            round(float(value), 3) for value in fire_xyz
                        ],
                        "active_flame_front_radius_m": float(
                            fire["active_flame_front_radius_m"]
                        ),
                        "camera_binding": {
                            "camera_contract_id": CAMERA_CONTRACT_ID,
                            "camera_count": EXPECTED_CAMERA_COUNT,
                            "orientation_policy": (
                                "fixed_source_pose_openusd_minus_z_forward"
                            ),
                            "active_front_truth_reference_local_m": [
                                round(float(value), 3) for value in fire_xyz
                            ],
                            "camera_reaimed": False,
                        },
                        "actor_states": actor_states,
                    }
                )
                step_index += 1
                timeline_step_count += 1
                actor_state_count += len(actor_states)
        scenario_deployments.append(
            {
                "scenario_id": scenario["scenario_id"],
                "duration_days": duration_days,
                "timeline_step_count": len(steps),
                "camera_observation_count": expected_observations,
                "actor_state_count": len(steps) * len(ASSETS),
                "steps": steps,
            }
        )

    teams = [
        {
            "team_id": asset["team_id"],
            "display_name": asset["team_id"].replace("TEAM-", "").replace("-", " "),
            "visual_actor_selection_id": asset["selection_id"],
            "operational_role": asset["operational_role"],
            "personnel_geometry": (
                "not_authored_no_already_imported_human_asset"
            ),
        }
        for asset in ASSETS
    ]
    asset_library = {
        asset["selection_id"]: {
            key: value
            for key, value in asset.items()
            if key
            not in {
                "orbit_radius_m",
                "altitude_agl_m",
                "orbit_phase_degrees",
                "upwind_offset_m",
                "flank_offset_m",
            }
        }
        for asset in ASSETS
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": STATE,
        "contract_id": CONTRACT_ID,
        "zone_id": "Z16",
        "camera_contract_id": CAMERA_CONTRACT_ID,
        "fire_scenario_contract_id": FIRE_CONTRACT_ID,
        "source_contracts": {
            "camera_file_sha256": camera_file_sha256,
            "fire_file_sha256": fire_file_sha256,
        },
        "execution_gate": {
            "base_scene_build_required_before_usd_authoring": True,
            "simulation_requires_fire_simulation_allowed_receipt": True,
            "simulation_performed": False,
            "render_performed": False,
            "remote_scene_modified": False,
        },
        "asset_scope": {
            "policy": "already_materialized_on_pod_only",
            "volume_root": VOLUME_ROOT,
            "source_inventory_path": (
                "actor-qa/actor-source-group-19/curated-available/"
                "selected-actor-group-available.json"
            ),
            "source_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "group_id": ASSET_GROUP_ID,
            "selected_asset_count": len(ASSETS),
            "new_downloads_required": False,
            "new_assets_imported": False,
            "source_inventory_campaign_admitted": False,
            "scope_override": (
                "user_requested_available_pod_assets_only; "
                "14 unavailable source-group entries are excluded"
            ),
            "all_wrappers_have_native_hero_mid_far": True,
        },
        "placement_contract": {
            "coordinate_system": "local_z_up_metres_plus_epsg2154_xy_ign69_z",
            "local_axes": {"X": "east", "Y": "north", "Z": "up"},
            "ground_actors": (
                "snap planned XY to an authored route, then sample the final "
                "scene terrain and apply the locked asset ground anchor"
            ),
            "aerial_actors": (
                "sample final scene terrain at planned XY and add locked AGL"
            ),
            "ground_route_snap_failure": "block_stage_authoring",
            "camera_pose_and_intrinsics_remain_fixed": True,
            "active_front_is_truth_reference_not_camera_control": True,
        },
        "teams": teams,
        "asset_library": asset_library,
        "scenario_deployments": scenario_deployments,
        "validation_summary": {
            "scenario_count": len(scenario_deployments),
            "timeline_step_count": timeline_step_count,
            "selected_asset_count": len(ASSETS),
            "actor_state_count": actor_state_count,
            "camera_observation_count": sum(
                item["camera_observation_count"]
                for item in scenario_deployments
            ),
            "all_selected_assets_used_at_every_step": True,
            "unselected_asset_reference_count": 0,
        },
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    validate_contract(payload, camera_contract, fire_contract)
    return payload


def validate_contract(
    payload: Mapping[str, Any],
    camera_contract: Mapping[str, Any],
    fire_contract: Mapping[str, Any],
) -> None:
    """Fail closed on a partial, unbound, or out-of-zone responder plan."""

    _validate_inputs(camera_contract, fire_contract)
    if payload.get("state") != STATE or payload.get("contract_id") != CONTRACT_ID:
        raise Z16ResponseContractError("unexpected response contract identity")
    scope = payload.get("asset_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("policy") != "already_materialized_on_pod_only"
        or scope.get("new_downloads_required") is not False
        or scope.get("new_assets_imported") is not False
    ):
        raise Z16ResponseContractError("response asset scope is not pod-local only")
    library = payload.get("asset_library")
    expected_ids = {asset["selection_id"] for asset in ASSETS}
    if not isinstance(library, Mapping) or set(library) != expected_ids:
        raise Z16ResponseContractError("response asset library differs from pod inventory")
    for item in library.values():
        path = str(item.get("wrapper_path", ""))
        if (
            not path.endswith("/asset.usda")
            or path.startswith("/")
            or "://" in path
            or "\\" in path
        ):
            raise Z16ResponseContractError("asset wrapper path is not pod-relative")
        if item.get("license") != "CC-BY-4.0":
            raise Z16ResponseContractError("asset license lock is missing")
    scenario_ids = {
        str(scenario["scenario_id"]) for scenario in fire_contract["scenarios"]
    }
    deployments = payload.get("scenario_deployments")
    if (
        not isinstance(deployments, list)
        or {str(item.get("scenario_id")) for item in deployments} != scenario_ids
    ):
        raise Z16ResponseContractError("response scenarios differ from fire scenarios")
    total_steps = 0
    total_actor_states = 0
    for deployment in deployments:
        steps = deployment.get("steps")
        if not isinstance(steps, list):
            raise Z16ResponseContractError("response scenario has no timeline")
        for step in steps:
            actor_states = step.get("actor_states")
            if (
                not isinstance(actor_states, list)
                or len(actor_states) != len(ASSETS)
                or {item.get("selection_id") for item in actor_states}
                != expected_ids
            ):
                raise Z16ResponseContractError(
                    "every timeline step must use the five pod assets once"
                )
            for state in actor_states:
                pose = state.get("pose_binding")
                xy = pose.get("position_xy_local_m") if isinstance(pose, Mapping) else None
                if (
                    not isinstance(xy, list)
                    or len(xy) != 2
                    or any(not _finite(value) for value in xy)
                    or not (POSITION_MARGIN_M <= float(xy[0]) <= 19_750.0)
                    or not (POSITION_MARGIN_M <= float(xy[1]) <= 19_750.0)
                ):
                    raise Z16ResponseContractError("actor XY is outside Z16")
                z_binding = pose.get("z_binding")
                if not isinstance(z_binding, Mapping):
                    raise Z16ResponseContractError("actor has no terrain-aware Z binding")
            total_steps += 1
            total_actor_states += len(actor_states)
    summary = payload.get("validation_summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("scenario_count") != 3
        or summary.get("timeline_step_count") != 72
        or summary.get("actor_state_count") != 360
        or summary.get("camera_observation_count") != 2_880
        or total_steps != 72
        or total_actor_states != 360
    ):
        raise Z16ResponseContractError("response deployment totals are inconsistent")
    actual_hash = payload.get("contract_sha256")
    unhashed = dict(payload)
    unhashed.pop("contract_sha256", None)
    if actual_hash != _canonical_sha256(unhashed):
        raise Z16ResponseContractError("response contract checksum is invalid")


def prepare_contract(
    camera_path: Path,
    fire_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    camera_contract = json.loads(camera_path.read_text(encoding="utf-8"))
    fire_contract = json.loads(fire_path.read_text(encoding="utf-8"))
    payload = build_contract(
        camera_contract,
        fire_contract,
        camera_file_sha256=_sha256_file(camera_path),
        fire_file_sha256=_sha256_file(fire_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--fire-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = prepare_contract(
        args.camera_contract.resolve(),
        args.fire_contract.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "state": payload["state"],
                "contract_id": payload["contract_id"],
                "output": str(args.output.resolve()),
                **payload["validation_summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
