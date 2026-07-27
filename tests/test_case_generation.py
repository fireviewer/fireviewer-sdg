from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.case_generation import (  # noqa: E402
    _composite_quality_failure,
    _composite_quality_metrics,
    _current_stage_ready,
    _frame_quality_failure,
    _frame_quality_metrics,
    _response_objects,
    _setup_reconstruction_renderer,
    _validate_response_box,
    _wait_for_stage_loading,
    _write_batch_progress,
    generate_batch,
)


class ResponseGenerationTests(unittest.TestCase):
    def test_batch_progress_counts_only_written_case_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch"
            records = batch / "records"
            records.mkdir(parents=True)
            record = {
                "case_id": "fid-000000",
                "category": "france_incident_days",
                "data_origin": "new_synthetic_generation",
                "production_stage": "pilot",
                "preview_relpath": "previews/fid-000000.png",
                "overlays": [],
            }
            (records / "fid-000000.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )
            spec = {
                "category": "france_incident_days",
                "case_start": 0,
                "case_count": 4,
                "production_stage": "pilot",
                "batch_root": batch,
                "render_revision": "unused-for-documents",
            }
            _write_batch_progress(
                spec,
                state="running",
                last_completed_record=record,
            )
            progress = json.loads(
                (batch / "batch-progress.json").read_text(encoding="utf-8")
            )

        self.assertEqual(progress["produced"], 1)
        self.assertEqual(progress["remaining"], 3)
        self.assertEqual(
            progress["last_completed"]["case_id"],
            "fid-000000",
        )
    def test_frame_quality_rejects_the_observed_washed_out_pilot(self) -> None:
        import numpy as np

        washed_out = np.full((120, 160, 3), 216, dtype=np.uint8)
        metrics = _frame_quality_metrics(washed_out)
        self.assertEqual(_frame_quality_failure(metrics), "overexposed")

    def test_frame_quality_accepts_a_detailed_outdoor_range(self) -> None:
        import numpy as np

        gradient = np.linspace(25, 210, 160, dtype=np.uint8)
        frame = np.repeat(gradient[None, :, None], 120, axis=0)
        frame = np.repeat(frame, 3, axis=2)
        metrics = _frame_quality_metrics(frame)
        self.assertIsNone(_frame_quality_failure(metrics))

    def test_frame_quality_rejects_a_large_warm_emissive_volume(self) -> None:
        import numpy as np

        frame = np.full((120, 160, 3), (75, 105, 135), dtype=np.uint8)
        frame[:, 60:, :] = (245, 220, 130)
        metrics = _frame_quality_metrics(frame)
        self.assertGreater(metrics["warm_emissive_fraction"], 0.12)
        self.assertEqual(
            _frame_quality_failure(metrics), "oversized_emissive_region"
        )

    def test_composite_quality_rejects_screen_filling_smoke(self) -> None:
        import numpy as np

        gradient = np.linspace(20, 220, 160, dtype=np.uint8)
        background = np.repeat(gradient[None, :, None], 120, axis=0)
        background = np.repeat(background, 3, axis=2)
        composite = np.full_like(background, (118, 118, 118))
        metrics = _composite_quality_metrics(background, composite)
        self.assertGreater(metrics["flow_affected_fraction"], 0.48)
        self.assertEqual(
            _composite_quality_failure(metrics),
            "excessive_smoke_coverage",
        )

    def test_composite_quality_accepts_localized_fire_and_smoke(self) -> None:
        import numpy as np

        gradient = np.linspace(25, 205, 160, dtype=np.uint8)
        background = np.repeat(gradient[None, :, None], 120, axis=0)
        background = np.repeat(background, 3, axis=2)
        composite = background.copy()
        composite[58:75, 76:92, :] = (235, 175, 80)
        composite[35:58, 70:99, :] = (105, 108, 112)
        metrics = _composite_quality_metrics(background, composite)
        self.assertIsNone(
            _composite_quality_failure(
                metrics,
                expected_fire_points=[
                    {"x_normalized": 0.52, "y_normalized": 0.55}
                ],
            )
        )

    def test_composite_quality_accepts_a_column_connected_to_its_base(self) -> None:
        import numpy as np

        background = np.full((120, 160, 3), (80, 110, 135), dtype=np.uint8)
        composite = background.copy()
        composite[66:86, 76:91, :] = (235, 175, 80)
        composite[30:67, 70:98, :] = (120, 120, 120)
        anchors = {
            "active_fire_point": {
                "x_normalized": 0.52,
                "y_normalized": 0.68,
            },
            "visible_fire_front_point": {
                "x_normalized": 0.55,
                "y_normalized": 0.63,
            },
            "smoke_column_base": {
                "x_normalized": 0.52,
                "y_normalized": 0.56,
            },
        }
        metrics = _composite_quality_metrics(
            background,
            composite,
            expected_anchors=anchors,
        )
        self.assertIsNone(
            _composite_quality_failure(metrics, expected_anchors=anchors)
        )
        self.assertEqual(metrics["smoke_column_connected_to_base"], 1.0)

    def test_composite_quality_rejects_a_detached_smoke_column(self) -> None:
        import numpy as np

        background = np.full((120, 160, 3), (80, 110, 135), dtype=np.uint8)
        composite = background.copy()
        composite[76:91, 76:91, :] = (235, 175, 80)
        composite[20:55, 10:35, :] = (120, 120, 120)
        anchors = {
            "active_fire_point": {
                "x_normalized": 0.52,
                "y_normalized": 0.70,
            },
            "visible_fire_front_point": {
                "x_normalized": 0.55,
                "y_normalized": 0.66,
            },
            "smoke_column_base": {
                "x_normalized": 0.52,
                "y_normalized": 0.56,
            },
        }
        metrics = _composite_quality_metrics(
            background,
            composite,
            expected_anchors=anchors,
        )
        self.assertEqual(
            _composite_quality_failure(metrics, expected_anchors=anchors),
            "smoke_column_detached_from_base",
        )

    def test_composite_quality_rejects_a_second_sun_like_flow_blob(self) -> None:
        import numpy as np

        background = np.full((120, 160, 3), (80, 110, 135), dtype=np.uint8)
        composite = background.copy()
        composite[66:86, 76:91, :] = (235, 175, 80)
        composite[30:67, 70:98, :] = (120, 120, 120)
        composite[5:22, 130:150, :] = (245, 195, 80)
        anchors = {
            "active_fire_point": {
                "x_normalized": 0.52,
                "y_normalized": 0.68,
            },
            "visible_fire_front_point": {
                "x_normalized": 0.55,
                "y_normalized": 0.63,
            },
            "smoke_column_base": {
                "x_normalized": 0.52,
                "y_normalized": 0.56,
            },
        }
        metrics = _composite_quality_metrics(
            background,
            composite,
            expected_anchors=anchors,
        )
        self.assertEqual(
            _composite_quality_failure(metrics, expected_anchors=anchors),
            "detached_sun_like_emissive_component",
        )

    def test_composite_quality_rejects_sun_like_blob_away_from_fire(self) -> None:
        import numpy as np

        background = np.full((120, 160, 3), (80, 110, 135), dtype=np.uint8)
        composite = background.copy()
        composite[5:25, 125:150, :] = (245, 195, 80)
        metrics = _composite_quality_metrics(background, composite)
        self.assertEqual(
            _composite_quality_failure(
                metrics,
                expected_fire_points=[
                    {"x_normalized": 0.2, "y_normalized": 0.75}
                ],
            ),
            "emissive_effect_detached_from_fire",
        )

    def test_current_stage_waits_for_stability_without_reopening_context(self) -> None:
        stage = object()

        class Context:
            def __init__(self) -> None:
                self.status_calls = 0

            def get_stage_loading_status(self):
                self.status_calls += 1
                if self.status_calls < 3:
                    return ("loading", 0, 1)
                return ("", 1, 1)

            def get_stage_streaming_status(self):
                return False

            def is_writable(self):
                return True

            def get_stage(self):
                return stage

        context = Context()
        application = SimpleNamespace(update=Mock())
        self.assertIs(_current_stage_ready(context, application), stage)
        self.assertEqual(application.update.call_count, 10)
        self.assertFalse(hasattr(context, "new_stage_with_callback"))

    def test_stage_loading_requires_consecutive_stable_updates(self) -> None:
        statuses = iter(
            [
                ("loading", 0, 1, False),
                ("", 1, 1, False),
                ("streaming", 1, 1, True),
                ("", 1, 1, False),
                ("", 1, 1, False),
            ]
        )

        class Context:
            def __init__(self) -> None:
                self.current = ("", 0, 0, False)

            def get_stage_loading_status(self):
                self.current = next(statuses)
                return self.current[:3]

            def get_stage_streaming_status(self):
                return self.current[3]

        application = SimpleNamespace(update=Mock())
        result = _wait_for_stage_loading(
            Context(), application, max_updates=5, stable_updates=2
        )
        self.assertEqual(result, ("", 1, 1))
        self.assertEqual(application.update.call_count, 5)

    def test_visual_batch_rejects_multiple_fire_events_before_isaac_start(self) -> None:
        spec = {
            "category": "terrestrial_fire_points",
            "case_start": 0,
            "case_count": 2,
            "batch_root": Path("batch"),
            "real_world_catalog": Path("catalog.json"),
            "volume_root": Path("volume"),
            "target_per_category": 4096,
        }
        catalog = {
            "assignments": [
                {"event": {"event_id": "fire-a"}},
                {"event": {"event_id": "fire-b"}},
            ]
        }
        with (
            patch(
                "fireviewer_sdg.case_generation.load_batch_spec",
                return_value=spec,
            ),
            patch(
                "fireviewer_sdg.case_generation.load_event_catalog",
                return_value=catalog,
            ),
            patch("pathlib.Path.mkdir"),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one fire event"):
                generate_batch(Path("batch-spec.json"))

    def test_review_gated_usd_does_not_require_nurec_utilities(self) -> None:
        with patch("fireviewer_sdg.case_generation.importlib.import_module") as importer:
            _setup_reconstruction_renderer(object(), "review_gated_usd")
        importer.assert_not_called()

    def test_nurec_contract_uses_nurec_render_setup(self) -> None:
        nurec_utils = SimpleNamespace(setup_for_rendering=Mock())
        stage = object()
        with patch(
            "fireviewer_sdg.case_generation.importlib.import_module",
            return_value=nurec_utils,
        ) as importer:
            _setup_reconstruction_renderer(stage, "particle_field")
        importer.assert_called_once_with("isaacsim.replicator.nurec_utils")
        nurec_utils.setup_for_rendering.assert_called_once_with(stage)

    def test_response_objects_map_to_stable_usd_prims(self) -> None:
        contract = {
            "composition": {
                "actors": [
                    {
                        "class_id": "sdis_vehicle",
                        "center_world_m": [1.0, 2.0, 3.0],
                        "aabb_min_world_m": [0.0, 1.0, 2.0],
                        "aabb_max_world_m": [2.0, 3.0, 4.0],
                        "positive": True,
                        "asset_sha256": "1" * 64,
                    },
                    {
                        "class_id": "hard_negative_construction_truck",
                        "center_world_m": [5.0, 6.0, 7.0],
                        "aabb_min_world_m": [4.0, 5.0, 6.0],
                        "aabb_max_world_m": [6.0, 7.0, 8.0],
                        "positive": False,
                        "asset_sha256": "2" * 64,
                    },
                ]
            }
        }
        objects = _response_objects(contract)
        self.assertEqual(objects["sdis_vehicle"]["prim_path"], "/World/Actors/Actor00")
        self.assertEqual(
            objects["hard_negative_construction_truck"]["prim_path"],
            "/World/Actors/Actor01",
        )

    def test_response_box_requires_visible_hd_detail_and_margin(self) -> None:
        quality = _validate_response_box(
            {"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.5},
            width=1920,
            height=1080,
            distance_band="near",
        )
        self.assertEqual(quality["width_px"], 384.0)
        with self.assertRaisesRegex(RuntimeError, "too small"):
            _validate_response_box(
                {"x_min": 0.2, "y_min": 0.2, "x_max": 0.205, "y_max": 0.205},
                width=1920,
                height=1080,
                distance_band="near",
            )
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            _validate_response_box(
                {"x_min": 0.0, "y_min": 0.2, "x_max": 0.2, "y_max": 0.5},
                width=1920,
                height=1080,
                distance_band="near",
            )


if __name__ == "__main__":
    unittest.main()
