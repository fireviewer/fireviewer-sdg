from __future__ import annotations

import unittest

import numpy as np

from fireviewer_sdg.native_zone_render import (
    REVIEW_VIEW_COUNT,
    RUNTIME_RESOLUTION,
    _quality,
    _wait_for_loading,
    _view_definitions,
)


class NativeZoneRenderTests(unittest.TestCase):
    def test_quality_accepts_non_uniform_rgb_buffer_at_exact_runtime_resolution(self) -> None:
        width, height = RUNTIME_RESOLUTION
        data = np.zeros((height, width, 4), dtype=np.uint8)
        data[:, width // 2 :, :3] = 128
        result = _quality(data, expected=RUNTIME_RESOLUTION)
        self.assertEqual(result["minimum"], 0)
        self.assertEqual(result["maximum"], 128)

    def test_quality_rejects_uniform_or_wrong_sized_buffers(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "uniform"):
            _quality(np.zeros((720, 1280, 4), dtype=np.uint8), expected=RUNTIME_RESOLUTION)
        with self.assertRaisesRegex(RuntimeError, "unexpected shape"):
            _quality(np.zeros((64, 64, 4), dtype=np.uint8), expected=RUNTIME_RESOLUTION)

    def test_review_contract_has_twelve_separate_camera_definitions(self) -> None:
        views = _view_definitions(elevation=600.0)
        self.assertEqual(len(views), REVIEW_VIEW_COUNT)
        self.assertEqual(len({view["position"] for view in views}), REVIEW_VIEW_COUNT)

    def test_wait_allows_stable_flow_streaming_after_payloads_are_loaded(self) -> None:
        class Application:
            def __init__(self) -> None:
                self.updates = 0

            def update(self) -> None:
                self.updates += 1

        class Context:
            def get_stage_loading_status(self) -> tuple[str, int, int]:
                return ("Flow residency", 400, 400)

            def get_stage_streaming_status(self) -> bool:
                return True

        application = Application()
        _wait_for_loading(context=Context(), application=application, updates=8)
        self.assertGreaterEqual(application.updates, 5)
