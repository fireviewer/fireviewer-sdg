from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fireviewer_sdg import fireviewer_writer as writer  # noqa: E402


class _Prim:
    def __init__(self, data: dict[str, object] | None = None) -> None:
        self.data = data or {}

    def IsValid(self) -> bool:
        return True

    def GetCustomDataByKey(self, key: str) -> object:
        return self.data.get(key)


class _Stage:
    def __init__(self, *, include_visual: bool = True) -> None:
        self.truth = _Prim(
            {
                writer.FIRE_TRUTH_STATE_KEY: "SIM-01-D001-H0800",
                writer.FIRE_TRUTH_TARGETS_KEY: json.dumps(
                    [[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]]
                ),
            }
        )
        self.visual = _Prim()
        self.include_visual = include_visual

    def GetPrimAtPath(self, path: str) -> _Prim | None:
        if path == writer.FIRE_TRUTH_ROOT:
            return self.truth
        if path == writer.FIRE_VISUAL_ROOT and self.include_visual:
            return self.visual
        return None


class _Annotator:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.attached = False

    def attach(self, _render_products: object) -> None:
        self.attached = True

    def detach(self) -> None:
        self.attached = False

    def get_data(self) -> np.ndarray:
        return self.data


class _Registry:
    def __init__(self) -> None:
        self.annotators = {
            "rgb": _Annotator(np.ones((2, 3, 4), dtype=np.uint8)),
            "semantic_segmentation": _Annotator(np.ones((2, 3), dtype=np.uint32)),
            "instance_segmentation": _Annotator(np.full((2, 3), 7, dtype=np.uint32)),
            "distance_to_camera": _Annotator(np.full((2, 3), 9.5, dtype=np.float32)),
        }

    def get_annotator(self, name: str) -> _Annotator:
        return self.annotators[name]


class _Rep:
    AnnotatorRegistry = _Registry()


class FireViewerWriterTests(unittest.TestCase):
    def test_requires_explicit_truth_and_visual_layers(self) -> None:
        with self.assertRaisesRegex(
            writer.FireViewerWriterError,
            "FireVisual layer is absent",
        ):
            writer.validate_fire_layers(_Stage(include_visual=False))

    def test_raycast_visibility_rejects_an_early_blocker(self) -> None:
        calls: list[float] = []

        def raycast(
            _origin: object, _direction: object, distance: float
        ) -> dict[str, object]:
            calls.append(distance)
            return {"hit": True, "distance": distance - 2.0}

        result = writer.raycast_visibility(
            camera_position_m=[0.0, 0.0, 0.0],
            targets_local_m=[[10.0, 0.0, 0.0]],
            raycast_closest=raycast,
        )
        self.assertEqual(calls, [10.0])
        self.assertEqual(result["visible_ray_count"], 0)
        self.assertFalse(result["rays"][0]["visible"])

    def test_writer_publishes_visual_truth_depth_and_raycast_receipt(self) -> None:
        def raycast(
            _origin: object, _direction: object, distance: float
        ) -> dict[str, object]:
            return {"hit": True, "distance": distance}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = _Stage()
            instance = writer.FireViewerReplicatorWriter(
                rep=_Rep(),
                stage=stage,
                output_root=root,
                raycast_closest=raycast,
            )
            instance.attach(object())
            receipt = instance.capture_frame(
                frame_index=4,
                camera_position_m=[0.0, 0.0, 0.0],
                camera_pose={
                    "position_m": [0.0, 0.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                intrinsics={"width_px": 3, "height_px": 2},
            )
            instance.detach()
            self.assertEqual(receipt["fire_truth"]["fire_state_id"], "SIM-01-D001-H0800")
            self.assertEqual(receipt["visibility"]["visible_ray_count"], 2)
            self.assertTrue((root / receipt["fire_visual"]["rgb_path"]).is_file())
            self.assertTrue((root / receipt["depth_path"]).is_file())
            self.assertTrue((root / receipt["semantic_path"]).is_file())
            self.assertTrue((root / receipt["instance_path"]).is_file())
            persisted = json.loads(
                (root / "FireTruth" / "receipts" / "frame-000004.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(persisted["visibility"]["method"], "physx_closest_hit_distance_v1")
            self.assertEqual(
                np.load(root / receipt["depth_path"], allow_pickle=False).shape,
                (2, 3),
            )

    def test_writer_rejects_truth_state_drift_while_attached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage = _Stage()
            instance = writer.FireViewerReplicatorWriter(
                rep=_Rep(),
                stage=stage,
                output_root=Path(directory),
                raycast_closest=lambda _a, _b, distance: {
                    "hit": True,
                    "distance": distance,
                },
            )
            instance.attach(object())
            stage.truth.data[writer.FIRE_TRUTH_STATE_KEY] = "SIM-01-D001-H1400"
            with self.assertRaisesRegex(
                writer.FireViewerWriterError,
                "state changed",
            ):
                instance.capture_frame(
                    frame_index=0,
                    camera_position_m=[0.0, 0.0, 0.0],
                    camera_pose={},
                    intrinsics={},
                )
            instance.detach()


if __name__ == "__main__":
    unittest.main()
