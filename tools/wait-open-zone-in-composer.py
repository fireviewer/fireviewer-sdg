"""Keep Composer visible while waiting for a zone build to become openable."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import omni.kit.viewport.utility
import omni.usd


async def _wait_and_open() -> None:
    root = Path(os.environ["FW_SDG_REVIEW_USD"])
    while not root.is_file():
        await asyncio.sleep(2.0)
    await omni.usd.get_context().open_stage_async(str(root))
    viewport = omni.kit.viewport.utility.get_active_viewport()
    if viewport is not None:
        viewport.camera_path = "/World/ReviewCameras/Review06"


asyncio.ensure_future(_wait_and_open())
