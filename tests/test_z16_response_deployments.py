from __future__ import annotations

import json
from pathlib import Path

import pytest

from fireviewer_sdg.z16_response_deployments import (
    ASSETS,
    CONTRACT_ID,
    STATE,
    Z16ResponseContractError,
    prepare_contract,
    validate_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = REPO_ROOT / "scenarios" / "z16-capture-cameras-v1.json"
FIRE = REPO_ROOT / "scenarios" / "z16-fire-scenarios-v1.json"


def test_prepares_complete_pod_local_response_timeline(tmp_path: Path) -> None:
    output = tmp_path / "response.json"
    payload = prepare_contract(CAMERAS, FIRE, output)

    assert payload["state"] == STATE
    assert payload["contract_id"] == CONTRACT_ID
    assert payload["asset_scope"]["new_downloads_required"] is False
    assert payload["asset_scope"]["selected_asset_count"] == 5
    assert len(payload["asset_scope"]["source_inventory_sha256"]) == 64
    assert payload["validation_summary"] == {
        "scenario_count": 3,
        "timeline_step_count": 72,
        "selected_asset_count": 5,
        "actor_state_count": 360,
        "camera_observation_count": 2880,
        "all_selected_assets_used_at_every_step": True,
        "unselected_asset_reference_count": 0,
    }
    expected_ids = {asset["selection_id"] for asset in ASSETS}
    for scenario in payload["scenario_deployments"]:
        for step in scenario["steps"]:
            assert len(step["actor_states"]) == 5
            assert {
                state["selection_id"] for state in step["actor_states"]
            } == expected_ids
            assert step["camera_binding"] == {
                "camera_contract_id": "Z16-CAMERA-40-V1",
                "camera_count": 40,
                "orientation_policy": (
                    "fixed_source_pose_openusd_minus_z_forward"
                ),
                "active_front_truth_reference_local_m": step[
                    "active_flame_centroid_local_m"
                ],
                "camera_reaimed": False,
            }
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_rejects_unselected_asset_reference(tmp_path: Path) -> None:
    output = tmp_path / "response.json"
    payload = prepare_contract(CAMERAS, FIRE, output)
    cameras = json.loads(CAMERAS.read_text(encoding="utf-8"))
    fire = json.loads(FIRE.read_text(encoding="utf-8"))
    payload["scenario_deployments"][0]["steps"][0]["actor_states"][0][
        "selection_id"
    ] = "not-on-pod"

    with pytest.raises(
        Z16ResponseContractError,
        match="five pod assets",
    ):
        validate_contract(payload, cameras, fire)
