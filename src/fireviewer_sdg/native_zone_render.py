"""Native Isaac renderer for a validated geographic OpenUSD package."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fireviewer_sdg.artifacts import sha256, write_json
from fireviewer_sdg.zone_scenes import _is_below, _read_json


RUNTIME_RESOLUTION = (1280, 720)
REVIEW_RESOLUTION = (3840, 2160)
REVIEW_VIEW_COUNT = 12


def _absolute_directory(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"{label} is absent or is not a directory: {path}")
    return path


def _quality(data: np.ndarray, *, expected: tuple[int, int]) -> dict[str, Any]:
    width, height = expected
    if data.ndim != 3 or data.shape[0] != height or data.shape[1] != width or data.shape[2] < 3:
        raise RuntimeError(f"RTX render has unexpected shape {tuple(data.shape)}")
    rgb = data[:, :, :3]
    minimum = int(rgb.min())
    maximum = int(rgb.max())
    variance = float(rgb.var())
    if maximum <= minimum or variance <= 0.25:
        raise RuntimeError("RTX render is empty or visually uniform")
    return {"minimum": minimum, "maximum": maximum, "variance": variance}


def _capture(
    *, rep: Any, camera: Any, resolution: tuple[int, int], path: Path, rt_subframes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    product = rep.create.render_product(camera, resolution, name=path.stem)
    rgb = rep.annotators.get("rgb")
    rgb.attach(product)
    try:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
        data = np.asarray(rgb.get_data()).copy()
        quality = _quality(data, expected=resolution)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(data[:, :, :3].astype(np.uint8), mode="RGB").save(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("RTX capture did not create an image file")
        return data, quality
    finally:
        rgb.detach()
        product.destroy()


def _view_definitions(*, elevation: float) -> list[dict[str, tuple[float, float, float]]]:
    offsets = (
        (-5500, -5500), (-2000, -6000), (2000, -6000), (5500, -5500),
        (-6000, -2000), (-2200, -1800), (2200, -1800), (6000, -2000),
        (-5500, 3500), (-2000, 4000), (2000, 4000), (5500, 3500),
    )
    result: list[dict[str, tuple[float, float, float]]] = []
    for x, y in offsets:
        target = (float(x * 0.35), float(y * 0.35), elevation + 80.0)
        position = (float(x), float(y), elevation + 1600.0)
        result.append({"position": position, "look_at": target})
    return result


def _wait_for_loading(*, context: Any, application: Any, updates: int = 360) -> None:
    """Require a stable, complete payload set before rendering.

    Flow keeps its stage streaming flag enabled while its simulation resources
    are resident.  Treating that persistent flag as an active load makes a
    valid Flow stage wait until timeout even after every USD payload is ready.
    """

    stable = 0
    last: tuple[str, int, int, bool] | None = None
    for _ in range(updates):
        application.update()
        message, loaded, total = context.get_stage_loading_status()
        snapshot = (str(message), int(loaded), int(total), bool(context.get_stage_streaming_status()))
        loading = snapshot[1] < snapshot[2]
        streaming = bool(context.get_stage_streaming_status())
        if loading or snapshot != last:
            stable = 0
        else:
            stable += 1
            if stable >= 4:
                return
        last = snapshot
    raise RuntimeError(
        "USD payloads did not stabilize for render: "
        f"message={last[0] if last else ''!r} loaded={last[1] if last else 0} "
        f"total={last[2] if last else 0} streaming={last[3] if last else False}"
    )


def render_zone(*, workspace_root: Path, zone_id: str) -> dict[str, Any]:
    zone_root = (workspace_root / "zone-scenes" / zone_id).resolve()
    if not _is_below(workspace_root / "zone-scenes", zone_root):
        raise ValueError("zone workspace escapes zone-scenes root")
    build_receipt_path = zone_root / "build" / "build-receipt.json"
    build_receipt = _read_json(build_receipt_path, label="scene build receipt")
    root = (zone_root / str(build_receipt.get("root_usd", {}).get("path", ""))).resolve()
    expected_hash = str(build_receipt.get("root_usd", {}).get("sha256", ""))
    if not _is_below(zone_root, root) or not root.is_file() or sha256(root) != expected_hash:
        raise RuntimeError("render root USD is absent or no longer matches the build receipt")
    render_root = zone_root / "renders"
    render_root.mkdir(parents=True, exist_ok=True)

    from isaacsim.simulation_app import SimulationApp

    application = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "multi_gpu": False})
    try:
        import carb.settings
        import omni.kit.app
        import omni.timeline
        import omni.usd
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom

        settings = carb.settings.get_settings()
        settings.set("/renderer/multiGpu/enabled", False)
        settings.set("/rtx/rendermode", "RayTracedLighting")
        settings.set("/rtx/post/aa/op", 3)
        settings.set("/rtx/post/dlss/execMode", 2)
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.flowusd", True)
        if not manager.is_extension_enabled("omni.flowusd"):
            raise RuntimeError("native Flow extension could not be enabled for render")
        context = omni.usd.get_context()
        if not context.open_stage(str(root)):
            raise RuntimeError("native Isaac renderer could not open the validated root USD")
        _wait_for_loading(context=context, application=application)
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("native Isaac renderer has no open USD stage")
        stage.Load()
        _wait_for_loading(context=context, application=application)
        flow = stage.GetPrimAtPath("/World/FireAndSmoke")
        if not flow or not flow.IsValid():
            raise RuntimeError("validated scene is missing its Flow smoke payload")

        # All render-only additions live in an anonymous session layer: the
        # signed root USD remains byte-identical to the build receipt.
        session = Sdf.Layer.CreateAnonymous("fireviewer-render-session")
        stage.SetEditTarget(session)
        rep.functional.create.dome_light(intensity=800, name="FireViewerRenderDome")
        # The root stores XY origin; get the actual scene elevation from the
        # Flow transform authored by the builder.
        flow_xform = UsdGeom.Xformable(flow)
        matrix = flow_xform.GetLocalTransformation()
        if isinstance(matrix, tuple):
            matrix = matrix[0]
        flow_elevation = float(matrix.ExtractTranslation()[2])
        if not math.isfinite(flow_elevation):
            raise RuntimeError("Flow placement has no finite local elevation")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(90):
            application.update()
        timeline.pause()
        runtime_camera = rep.functional.create.camera(
            position=(650.0, -650.0, flow_elevation + 360.0),
            look_at=(0.0, 0.0, flow_elevation + 30.0),
            name="RuntimeSmokeCamera",
        )
        flow_imageable = UsdGeom.Imageable(flow)
        flow_imageable.MakeInvisible()
        for _ in range(10):
            application.update()
        background, _background_quality = _capture(
            rep=rep,
            camera=runtime_camera,
            resolution=RUNTIME_RESOLUTION,
            path=render_root / "runtime_background_720p.png",
            rt_subframes=2,
        )
        flow_imageable.MakeVisible()
        for _ in range(20):
            application.update()
        runtime_path = render_root / "runtime_smoke_720p.png"
        runtime, runtime_quality = _capture(
            rep=rep,
            camera=runtime_camera,
            resolution=RUNTIME_RESOLUTION,
            path=runtime_path,
            rt_subframes=4,
        )
        flow_difference = float(
            np.mean(np.abs(runtime[:, :, :3].astype(np.int16) - background[:, :, :3].astype(np.int16)))
        )
        if flow_difference < 0.5:
            raise RuntimeError("Flow smoke render did not differ from its clean background")

        reviews: list[dict[str, Any]] = []
        for index, view in enumerate(_view_definitions(elevation=flow_elevation), start=1):
            camera = rep.functional.create.camera(
                position=view["position"], look_at=view["look_at"], name=f"ReviewCamera{index:02d}"
            )
            path = render_root / f"review_{index:02d}_4k.png"
            _data, quality = _capture(
                rep=rep,
                camera=camera,
                resolution=REVIEW_RESOLUTION,
                path=path,
                rt_subframes=2,
            )
            reviews.append(
                {
                    "path": path.relative_to(zone_root).as_posix(),
                    "sha256": sha256(path),
                    "width": REVIEW_RESOLUTION[0],
                    "height": REVIEW_RESOLUTION[1],
                    "camera": view,
                    "quality": quality,
                }
            )
        receipt_path = render_root / "render-receipt.json"
        receipt = {
            "schema_version": 1,
            "zone_id": zone_id,
            "rendered_at": datetime.now(UTC).isoformat(),
            "runtime": "native_isaac_sim_replicator_flow",
            "root_usd_sha256": expected_hash,
            "runtime_720p": {
                "path": runtime_path.relative_to(zone_root).as_posix(),
                "sha256": sha256(runtime_path),
                "width": RUNTIME_RESOLUTION[0],
                "height": RUNTIME_RESOLUTION[1],
                "quality": runtime_quality,
                "flow_background_mean_absolute_difference": flow_difference,
            },
            "review_renders": reviews,
        }
        write_json(receipt_path, receipt)
        return receipt
    finally:
        # The module terminates with os._exit after its receipt is printed.
        # Calling SimulationApp.close() here can terminate Kit before Python
        # propagates a load/render failure, falsely returning success without
        # a render receipt.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one native Isaac/OpenUSD geographic scene")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--zone", required=True)
    args = parser.parse_args(argv)
    receipt = render_zone(workspace_root=_absolute_directory(args.workspace_root, label="workspace root"), zone_id=args.zone)
    print(json.dumps({"zone_id": args.zone, "review_renders": len(receipt["review_renders"]), "runtime_720p": receipt["runtime_720p"]}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


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
