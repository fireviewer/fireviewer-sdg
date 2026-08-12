from __future__ import annotations

"""Capture one deterministic native RTX diagnostic frame for a zone scene.

This is deliberately narrower than the production render phase.  It verifies
the same USD payload and material path that Composer opens, before a reviewer
is asked to inspect the stage visually.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one native RTX zone-scene diagnostic")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument(
        "--sun-intensity",
        type=float,
        default=0.0,
        help="optional session-only sun; zero verifies authored stage lighting",
    )
    parser.add_argument(
        "--probe-dome-intensity",
        type=float,
        default=0.0,
        help="optional session-only dome; zero verifies authored stage lighting",
    )
    parser.add_argument(
        "--authored-dome-intensity",
        type=float,
        default=None,
        help="temporary session override for /World/Lighting/Sky",
    )
    parser.add_argument(
        "--authored-sun-intensity",
        type=float,
        default=None,
        help="temporary session override for /World/Lighting/Sun",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace = Path(args.workspace_root).resolve()
    zone_root = (workspace / "zone-scenes" / args.zone).resolve()
    root = zone_root / "build" / f"{args.zone}_root.usdc"
    output_dir = zone_root / "renders" / "diagnostics"
    if not root.is_file():
        raise FileNotFoundError(f"zone root USD is absent: {root}")

    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "multi_gpu": False})
    try:
        import carb.settings
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Sdf, UsdGeom, UsdLux, UsdShade

        settings = carb.settings.get_settings()
        settings.set("/renderer/multiGpu/enabled", False)
        settings.set("/rtx/rendermode", "RayTracedLighting")
        context = omni.usd.get_context()
        if not context.open_stage(str(root)):
            raise RuntimeError("native RTX probe could not open root USD")
        for _ in range(120):
            app.update()
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("native RTX probe has no stage")
        stage.Load()
        for _ in range(120):
            app.update()
        stage.SetEditTarget(stage.GetSessionLayer())
        if args.authored_dome_intensity is not None:
            sky = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lighting/Sky"))
            if not sky:
                raise RuntimeError("native RTX probe is missing the authored sky")
            sky.GetIntensityAttr().Set(args.authored_dome_intensity)
        if args.authored_sun_intensity is not None:
            sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Lighting/Sun"))
            if not sun:
                raise RuntimeError("native RTX probe is missing the authored sun")
            sun.GetIntensityAttr().Set(args.authored_sun_intensity)
        if args.sun_intensity > 0.0:
            daylight = UsdLux.DistantLight.Define(stage, "/ZoneProbe/Daylight")
            daylight.CreateIntensityAttr(args.sun_intensity)
            daylight.CreateAngleAttr(0.53)
            daylight.AddRotateXYZOp().Set((35.0, -25.0, 35.0))
        for _ in range(12):
            app.update()

        surface = stage.GetPrimAtPath("/World/Terrain/VisualSurface/Surface")
        if not surface or not surface.IsValid():
            raise RuntimeError("visible continuous terrain mesh is absent")
        imageable = UsdGeom.Imageable(surface)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            raise RuntimeError("visible continuous terrain mesh is invisible")
        material, _ = UsdShade.MaterialBindingAPI(surface).ComputeBoundMaterial()
        if not material:
            raise RuntimeError("visible terrain has no bound material")

        # Keep the diagnostic camera away from the terrain origin and point it
        # toward the centre so the frame covers both relief and orthophoto.
        if args.probe_dome_intensity > 0.0:
            rep.functional.create.dome_light(
                intensity=args.probe_dome_intensity, name="ZoneProbeDome"
            )
        camera = rep.functional.create.camera(
            position=(6500.0, -8500.0, 2500.0),
            look_at=(0.0, 0.0, 500.0),
            name="ZoneProbeCamera",
        )
        def capture(name: str) -> np.ndarray:
            product = rep.create.render_product(camera, (1280, 720), name=name)
            rgb = rep.annotators.get("rgb")
            rgb.attach(product)
            try:
                rep.orchestrator.step(delta_time=0.0, rt_subframes=2)
                return np.asarray(rgb.get_data()).copy()
            finally:
                rgb.detach()
                product.destroy()

        pixels = capture("zone_rtx_textured_probe")
        if pixels.ndim != 3 or pixels.shape[:2] != (720, 1280):
            raise RuntimeError(f"unexpected RTX probe dimensions: {pixels.shape}")
        rgb_pixels = pixels[:, :, :3]
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "native_rtx_720p.png"
        Image.fromarray(rgb_pixels.astype(np.uint8), mode="RGB").save(image_path)

        # The second capture deliberately replaces only the material in an
        # anonymous session layer.  A bright emissive surface distinguishes a
        # camera/geometry failure from a texture-network failure without
        # changing the signed scene package.
        flat = UsdShade.Material.Define(stage, "/ZoneProbe/DiffuseTerrain")
        flat_shader = UsdShade.Shader.Define(stage, "/ZoneProbe/DiffuseTerrain/Preview")
        flat_shader.CreateIdAttr("UsdPreviewSurface")
        flat_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.95, 0.08, 0.04))
        flat.CreateSurfaceOutput().ConnectToSource(flat_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(surface).Bind(flat)
        for _ in range(12):
            app.update()
        diffuse_pixels = capture("zone_rtx_diffuse_probe")[:, :, :3]
        diffuse_path = output_dir / "native_rtx_diffuse_720p.png"
        Image.fromarray(diffuse_pixels.astype(np.uint8), mode="RGB").save(diffuse_path)

        emissive = UsdShade.Material.Define(stage, "/ZoneProbe/EmissiveTerrain")
        emissive_shader = UsdShade.Shader.Define(stage, "/ZoneProbe/EmissiveTerrain/Preview")
        emissive_shader.CreateIdAttr("UsdPreviewSurface")
        emissive_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.95, 0.08, 0.04))
        emissive_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.95, 0.08, 0.04))
        emissive.CreateSurfaceOutput().ConnectToSource(emissive_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(surface).Bind(emissive)
        for _ in range(12):
            app.update()
        flat_pixels = capture("zone_rtx_emissive_probe")[:, :, :3]
        flat_path = output_dir / "native_rtx_emissive_720p.png"
        Image.fromarray(flat_pixels.astype(np.uint8), mode="RGB").save(flat_path)
        report = {
            "captured_at": datetime.now(UTC).isoformat(),
            "root_usd": str(root),
            "frame": image_path.name,
            "shape": list(rgb_pixels.shape),
            "minimum": int(rgb_pixels.min()),
            "maximum": int(rgb_pixels.max()),
            "mean": float(rgb_pixels.mean()),
            "variance": float(rgb_pixels.var()),
            "diffuse_frame": diffuse_path.name,
            "diffuse_minimum": int(diffuse_pixels.min()),
            "diffuse_maximum": int(diffuse_pixels.max()),
            "diffuse_mean": float(diffuse_pixels.mean()),
            "diffuse_variance": float(diffuse_pixels.var()),
            "emissive_frame": flat_path.name,
            "emissive_minimum": int(flat_pixels.min()),
            "emissive_maximum": int(flat_pixels.max()),
            "emissive_mean": float(flat_pixels.mean()),
            "emissive_variance": float(flat_pixels.var()),
            "terrain_material": str(material.GetPath()),
            "terrain_visibility": str(imageable.ComputeVisibility()),
            "probe_sun_intensity": args.sun_intensity,
            "probe_dome_intensity": args.probe_dome_intensity,
            "authored_dome_intensity_override": args.authored_dome_intensity,
            "authored_sun_intensity_override": args.authored_sun_intensity,
        }
        (output_dir / "native_rtx_720p.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))
        if report["emissive_maximum"] <= report["emissive_minimum"] or report["emissive_variance"] <= 0.25:
            raise RuntimeError("native RTX probe camera cannot see the terrain mesh")
        if report["maximum"] <= report["minimum"] or report["variance"] <= 0.25:
            raise RuntimeError("native RTX probe is visually uniform")
        return 0
    finally:
        # The Isaac standalone launcher owns process shutdown; a direct close
        # can hide a pending Kit exception on this runtime.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
