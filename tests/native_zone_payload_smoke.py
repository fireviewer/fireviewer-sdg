"""Run manually under C:\\isaacsim\\python.bat to prove native USD payload API use."""

from __future__ import annotations

import tempfile
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from fireviewer_sdg.native_zone_scene import (
    _ElevationGrid,
    _lock_flow_preset,
    _write_aggregate,
    _write_cameras,
    _write_payload,
    _write_root,
)


def main() -> int:
    from isaacsim.simulation_app import SimulationApp

    SimulationApp({"headless": True})
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ortho = root / "ortho.jpg"
        payload = root / "payloads" / "L93_0000_0001.usdc"
        values = np.array(
            [[100.0, 110.0, 120.0, 130.0], [200.0, 210.0, 220.0, 230.0], [300.0, 310.0, 320.0, 330.0], [400.0, 410.0, 420.0, 430.0]],
            dtype=np.float32,
        )
        Image.new("RGB", (4, 4), color=(32, 96, 64)).save(ortho)
        grid = _ElevationGrid()
        _write_payload(
            payload_path=payload,
            tile={"tile_ref": "L93_0000_0001", "xmin": "0", "ymin": "0", "xmax": "1000", "ymax": "1000"},
            values=values,
            ortho_path=ortho,
            ortho_lod0_path=None,
            origin_x=500,
            origin_y=500,
            usd=Usd,
            usd_geom=UsdGeom,
            usd_shade=UsdShade,
            sdf=Sdf,
            gf=Gf,
            elevation_grid=grid,
        )
        reopened = Usd.Stage.Open(str(payload))
        assert reopened is not None
        tile_prim = reopened.GetPrimAtPath("/Tile")
        assert tile_prim.IsValid()
        lods = tile_prim.GetVariantSets().GetVariantSet("terrainLOD")
        assert set(lods.GetVariantNames()) == {"LOD1", "LOD2", "LOD3"}
        assert lods.GetVariantSelection() == "LOD1"
        assert reopened.GetPrimAtPath("/Tile/Terrain").IsValid()
        assert reopened.GetPrimAtPath("/Tile/Collision").IsValid()
        assert reopened.GetPrimAtPath("/Tile/Materials/TerrainContext").IsValid()
        assert grid.elevation(0, 1000, fallback=0.0) == 100.0
        aggregate = root / "aggregates" / "aggregate_5km_0_0.usdc"
        _write_aggregate(
            aggregate_path=aggregate,
            payload_records=[
                (
                    payload,
                    None,
                    {
                        "tile_ref": "L93_0000_0001",
                        "xmin": "0",
                        "ymin": "0",
                        "xmax": "1000",
                        "ymax": "1000",
                    },
                )
            ],
            origin_x=500,
            origin_y=500,
            usd=Usd,
            usd_geom=UsdGeom,
        )
        cameras = root / "review-cameras.usda"
        _write_cameras(
            cameras_path=cameras,
            origin_x=500,
            origin_y=500,
            elevation=250.0,
            usd=Usd,
            usd_geom=UsdGeom,
            gf=Gf,
        )
        source_lock = root / "source-lock.json"
        source_lock.write_text("{}", encoding="utf-8")
        flow_lock = _lock_flow_preset(build_root=root)
        scene = root / "ZTEST_root.usdc"
        _write_root(
            root_path=scene,
            aggregate_paths=[aggregate],
            visual_terrain_path=None,
            cameras_path=cameras,
            zone={"id": "ZTEST", "name": "Native root smoke"},
            source_lock=source_lock,
            flow_asset=root / str(flow_lock["packaged_path"]),
            flow_asset_lock=flow_lock,
            origin_x=500,
            origin_y=500,
            flow_elevation=250.0,
            usd=Usd,
            usd_geom=UsdGeom,
            usd_lux=UsdLux,
        )
        reopened_scene = Usd.Stage.Open(str(scene))
        assert reopened_scene is not None
        assert reopened_scene.GetPrimAtPath("/World/FireAndSmoke").IsValid()
        assert reopened_scene.GetPrimAtPath("/World/Terrain/aggregate_5km_0_0").IsValid()
    print("native_zone_payload_smoke=ok", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(code)
