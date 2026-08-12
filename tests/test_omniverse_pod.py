from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg import omniverse_pod  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _content_lock(files: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_materialized_asset_manifest(
    volume: Path,
) -> tuple[Path, dict[str, object]]:
    manifest = volume / "input" / "simready-assets.json"
    wrappers = manifest.parent / "wrappers"
    sources = manifest.parent / "cache"
    wrappers.mkdir(parents=True)
    sources.mkdir()

    environment: dict[str, dict[str, list[dict[str, object]]]] = {
        "vegetation": {},
        "buildings": {},
    }
    for kind, family_minimums in omniverse_pod.PHOTOREAL_FAMILY_MINIMUMS.items():
        for family, minimum in family_minimums.items():
            family_id = f"{kind}.{family}"
            family_entries: list[dict[str, object]] = []
            for family_index in range(minimum):
                name = f"{kind}-{family}-{family_index:02d}"
                wrapper = wrappers / f"{name}.usda"
                source = sources / f"{name}.usd"
                wrapper.write_text(
                    "#usda 1.0\n"
                    f'def Xform "{name}" (\n'
                    f"    prepend references = @../cache/{name}.usd@\n"
                    ") {}\n",
                    encoding="utf-8",
                )
                source.write_text(
                    f'#usda 1.0\ndef Xform "source-{name}" {{}}\n',
                    encoding="utf-8",
                )
                source_sha = _digest(source)
                materialized_files: list[dict[str, object]] = [
                    {
                        "path": source.relative_to(volume).as_posix(),
                        "sha256": source_sha,
                        "size_bytes": source.stat().st_size,
                    }
                ]
                if family_id.startswith("vegetation.trees"):
                    dimensions = {"x": 4.0, "y": 4.5, "z": 14.0}
                elif family_id.startswith("vegetation.shrubs"):
                    dimensions = {"x": 1.5, "y": 1.4, "z": 1.2}
                elif family_id.startswith("vegetation.understory"):
                    dimensions = {"x": 0.5, "y": 0.4, "z": 0.3}
                else:
                    dimensions = {"x": 12.0, "y": 9.0, "z": 7.0}
                minimum_lods = omniverse_pod.PHOTOREAL_MIN_LOD_LEVELS[
                    family_id
                ]
                metadata = {
                    "native_dimensions_m": dimensions,
                    "ground_anchor_m": [0.0, 0.0, 0.0],
                    "anchor_validation": {
                        "state": "passed",
                        "policy": "native_bbox_bottom_center",
                    },
                    "lod": {
                        "state": "passed",
                        "strategy": "native_variant_set",
                        "levels": [
                            f"LOD{lod_index}"
                            for lod_index in range(minimum_lods)
                        ],
                        "level_count": minimum_lods,
                    },
                    "materials": {
                        "state": "passed",
                        "material_prim_count": 2,
                        "bound_material_prim_count": 4,
                        "resolved_asset_dependency_count": 1,
                        "unresolved_dependencies": [],
                    },
                    "placement": {
                        "grounding": "native_anchor",
                        "scale_policy": "uniform_only",
                        "non_uniform_scale_allowed": False,
                        "minimum_uniform_scale": 0.8,
                        "maximum_uniform_scale": 1.25,
                    },
                }
                source_uri = (
                    "https://omniverse-content-production.s3-us-west-2."
                    f"amazonaws.com/Assets/Isaac/6.0/{name}.usd"
                )
                licence_id = "LicenseRef-NvidiaProprietary"
                family_entries.append(
                    {
                        "asset_id": f"{family_id}:{source_sha[:16]}",
                        "family": family_id,
                        "identity": {
                            "source_name": name,
                            "source_identity": f"{family_id}/{name}",
                        },
                        "path": wrapper.relative_to(manifest.parent).as_posix(),
                        "sha256": _digest(wrapper),
                        "source_cache_path": source.relative_to(
                            manifest.parent
                        ).as_posix(),
                        "content_lock_sha256": _content_lock(
                            materialized_files
                        ),
                        "dependency_count": 1,
                        "materialized_files": materialized_files,
                        "source_uri": source_uri,
                        "provider_hash": source_sha,
                        "provider_version": "6.0-test",
                        "provenance": {
                            "provider": "NVIDIA Omniverse",
                            "source_uri": source_uri,
                            "provider_hash": source_sha,
                            "provider_version": "6.0-test",
                            "discovery": "test_fixture",
                        },
                        "license_id": licence_id,
                        "license_uri": "https://docs.example.test/license",
                        "license": {
                            "id": licence_id,
                            "uri": "https://docs.example.test/license",
                            "redistribution": "test_only",
                        },
                        "quality_validation": "native_metadata_passed",
                        "placement_validation": "pending_console_review",
                        "source_meters_per_unit": 1.0,
                        "source_up_axis": "Z",
                        **metadata,
                        "metadata_validation_sha256": _canonical_digest(
                            metadata
                        ),
                    }
                )
            environment[kind][family] = family_entries

    payload: dict[str, object] = {
        "schema_version": omniverse_pod.MANIFEST_SCHEMA_VERSION,
        "profile": omniverse_pod.SIMREADY_PROFILE,
        "family_minimums": omniverse_pod.PHOTOREAL_FAMILY_MINIMUMS,
        "library_policy": omniverse_pod.PHOTOREAL_LIBRARY_POLICY,
        "discovery": {
            "mode": omniverse_pod.MATERIALIZED_ASSET_MODE,
            "missing_environment": [],
        },
        "environment": environment,
    }
    _write_json(manifest, payload)
    return manifest, payload


def _write_gate_inputs(
    root: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    volume = root / "volume"
    contracts = volume / "contracts"
    scene_root = volume / "production" / "zone-scenes" / "Z16"
    scene_build = scene_root / "build"
    runtime = contracts / "runtime-preflight.json"
    campaign = contracts / "campaign-index.json"
    assets, _payload = _write_materialized_asset_manifest(volume)
    root_usd = scene_build / "Z16_root.usdc"
    build = scene_build / "build-receipt.json"
    scene_validation = scene_root / "scene-auto-validation.json"
    pending = scene_root / "review-pending.json"
    payload = scene_build / "payloads" / "tile.usdc"
    aggregate = scene_build / "aggregates" / "aggregate.usdc"
    cameras = scene_build / "metadata" / "cameras.usda"
    source_lock = scene_root / "source-lock.json"
    asset_lock = scene_build / "metadata" / "asset-lock.json"
    packaged_asset = scene_build / "assets" / "flow.usda"

    _write_json(
        runtime,
        {
            "state": "SETUP_PREFLIGHT_PASSED",
            "gpu": {
                "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "memory_mib": 96_000,
            },
            "system_memory": {
                "effective_mib": 142_000,
                "minimum_effective_mib": 138_000,
                "measurement": "finite_container_cgroup_limit",
                "source": "/sys/fs/cgroup/memory.max",
                "host_proc_meminfo_used": False,
            },
            "storage": {
                "mode": "ephemeral-nvme",
                "capacity_bytes": 1_610_612_736_000,
                "automatic_stop_allowed": False,
            },
        },
    )
    for artifact, content in (
        (root_usd, '#usda 1.0\ndef Xform "World" {}\n'),
        (payload, '#usda 1.0\ndef Xform "Tile" {}\n'),
        (aggregate, '#usda 1.0\ndef Xform "Aggregate" {}\n'),
        (cameras, '#usda 1.0\ndef Xform "Cameras" {}\n'),
        (packaged_asset, '#usda 1.0\ndef Xform "Flow" {}\n'),
    ):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(content, encoding="utf-8")
    _write_json(source_lock, {"state": "locked"})
    _write_json(asset_lock, {"state": "locked"})
    _write_json(
        build,
        {
            "root_usd": {
                "path": root_usd.relative_to(scene_root).as_posix(),
                "sha256": _digest(root_usd),
            },
            "payloads": [
                {
                    "path": payload.relative_to(scene_root).as_posix(),
                    "sha256": _digest(payload),
                }
            ],
            "aggregates_5km": [
                {
                    "path": aggregate.relative_to(scene_root).as_posix(),
                    "sha256": _digest(aggregate),
                }
            ],
            "cameras": {
                "path": cameras.relative_to(scene_root).as_posix(),
                "sha256": _digest(cameras),
            },
            "source_lock": {
                "path": source_lock.relative_to(scene_root).as_posix(),
                "sha256": _digest(source_lock),
            },
            "asset_lock": {
                "path": asset_lock.relative_to(scene_root).as_posix(),
                "sha256": _digest(asset_lock),
                "assets": [
                    {
                        "packaged_path": packaged_asset.relative_to(
                            scene_build
                        ).as_posix(),
                        "packaged_sha256": _digest(packaged_asset),
                    }
                ],
            },
            "layers": {},
            "fire_simulation_status": "blocked_pending_editor_review",
        },
    )
    _write_json(
        scene_validation,
        {
            "state": "AUTO_VALIDATED",
            "fire_simulation_status": "blocked_pending_editor_review",
            "root_usd_sha256": _digest(root_usd),
            "build_receipt_sha256": _digest(build),
            "asset_manifest_sha256": _digest(assets),
            "used_layers": [
                {"path": str(root_usd), "sha256": _digest(root_usd)},
                {"path": str(payload), "sha256": _digest(payload)},
            ],
        },
    )
    asset_receipt = omniverse_pod.validate_materialized_assets(
        manifest_path=assets,
        volume_root=volume,
    )
    _write_json(
        campaign,
        {
            "campaign_id": omniverse_pod.CAMPAIGN_ID,
            "fire_simulation_status": "blocked_pending_editor_review",
            "assets": {
                "manifest_sha256": _digest(assets),
                "content_sha256": asset_receipt["asset_content_sha256"],
            },
        },
    )
    omniverse_pod.create_review_pending(
        zone_id="Z16",
        root_usd=root_usd,
        runtime_preflight=runtime,
        campaign_index=campaign,
        asset_manifest=assets,
        volume_root=volume,
        build_receipt=build,
        scene_auto_validation=scene_validation,
        output_path=pending,
    )
    return (
        volume,
        runtime,
        campaign,
        assets,
        root_usd,
        build,
        scene_validation,
        pending,
    )


class OmniversePodTests(unittest.TestCase):
    def test_effective_system_ram_uses_the_container_cgroup_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cgroup_v2 = root / "memory.max"
            cgroup_v1 = root / "memory.limit_in_bytes"
            cgroup_v2.write_text(
                str(142_000 * 1024 * 1024),
                encoding="ascii",
            )
            cgroup_v1.write_text("max", encoding="ascii")

            effective = omniverse_pod._effective_system_ram_mib(
                cgroup_limit_paths=(cgroup_v2, cgroup_v1),
            )

        self.assertEqual(effective, 142_000)

    def test_container_memory_receipt_never_uses_host_meminfo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            limit_path = Path(directory) / "memory.limit_in_bytes"
            limit_path.write_text(
                str(250_999_996_416),
                encoding="ascii",
            )

            receipt = omniverse_pod._container_memory_limit(
                cgroup_limit_paths=(limit_path,),
            )

        self.assertEqual(receipt["limit_bytes"], 250_999_996_416)
        self.assertEqual(receipt["effective_mib"], 239_372)
        self.assertEqual(receipt["source"], str(limit_path))
        self.assertEqual(receipt["measurement"], "finite_container_cgroup_limit")
        self.assertFalse(receipt["host_proc_meminfo_used"])

    def test_container_memory_rejects_unbounded_cgroup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            limit_path = Path(directory) / "memory.max"
            limit_path.write_text("max", encoding="ascii")

            with self.assertRaisesRegex(
                RuntimeError,
                "no finite cgroup memory limit",
            ):
                omniverse_pod._container_memory_limit(
                    cgroup_limit_paths=(limit_path,),
                )

    def test_materialized_asset_manifest_is_fully_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, _payload = _write_materialized_asset_manifest(volume)

            receipt = omniverse_pod.validate_materialized_assets(
                manifest_path=manifest,
                volume_root=volume,
            )

        self.assertEqual(receipt["state"], "ASSETS_LOCKED")
        self.assertEqual(
            receipt["materialization_mode"],
            omniverse_pod.MATERIALIZED_ASSET_MODE,
        )
        self.assertEqual(
            receipt["family_counts"],
            omniverse_pod.PHOTOREAL_FAMILY_MINIMUMS,
        )
        self.assertEqual(receipt["vegetation_assets"], 16)
        self.assertEqual(receipt["building_assets"], 10)
        self.assertEqual(receipt["asset_count"], 26)
        self.assertEqual(len(receipt["assets"]), 26)

    def test_photoreal_manifest_rejects_poor_family_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, payload = _write_materialized_asset_manifest(volume)
            payload["environment"]["buildings"]["industrial"].pop()
            _write_json(manifest, payload)

            with self.assertRaisesRegex(
                ValueError,
                "at least 2 assets in buildings.industrial",
            ):
                omniverse_pod.validate_materialized_assets(
                    manifest_path=manifest,
                    volume_root=volume,
                )

    def test_photoreal_manifest_rejects_weak_lods_and_materials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, payload = _write_materialized_asset_manifest(volume)
            tree = payload["environment"]["vegetation"]["trees"][0]
            tree["lod"]["levels"] = ["LOD0"]
            tree["lod"]["level_count"] = 1
            metadata = {
                key: tree[key]
                for key in (
                    "native_dimensions_m",
                    "ground_anchor_m",
                    "anchor_validation",
                    "lod",
                    "materials",
                    "placement",
                )
            }
            tree["metadata_validation_sha256"] = _canonical_digest(metadata)
            _write_json(manifest, payload)

            with self.assertRaisesRegex(
                ValueError,
                "requires at least 3 distinct native LOD levels",
            ):
                omniverse_pod.validate_materialized_assets(
                    manifest_path=manifest,
                    volume_root=volume,
                )

    def test_photoreal_manifest_accepts_scene_optimizer_generated_lods(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, payload = _write_materialized_asset_manifest(volume)
            tree = payload["environment"]["vegetation"]["trees"][0]
            tree["lod"]["strategy"] = "scene_optimizer_decimateMeshes"
            metadata = {
                key: tree[key]
                for key in (
                    "native_dimensions_m",
                    "ground_anchor_m",
                    "anchor_validation",
                    "lod",
                    "materials",
                    "placement",
                )
            }
            tree["metadata_validation_sha256"] = _canonical_digest(metadata)
            _write_json(manifest, payload)

            receipt = omniverse_pod.validate_materialized_assets(
                manifest_path=manifest,
                volume_root=volume,
            )
            validated_tree = next(
                asset
                for asset in receipt["assets"]
                if asset["role"] == "vegetation.trees[0]"
            )
            self.assertEqual(
                validated_tree["lod_level_count"],
                3,
            )

            tree["lod"]["levels"] = ["LOD0", "LOD1", "LOD2"]
            tree["lod"]["level_count"] = 3
            tree["materials"]["bound_material_prim_count"] = 0
            metadata = {
                key: tree[key]
                for key in (
                    "native_dimensions_m",
                    "ground_anchor_m",
                    "anchor_validation",
                    "lod",
                    "materials",
                    "placement",
                )
            }
            tree["metadata_validation_sha256"] = _canonical_digest(metadata)
            _write_json(manifest, payload)

            with self.assertRaisesRegex(
                ValueError,
                "no passed bound-material validation",
            ):
                omniverse_pod.validate_materialized_assets(
                    manifest_path=manifest,
                    volume_root=volume,
                )

    def test_photoreal_manifest_rejects_non_uniform_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, payload = _write_materialized_asset_manifest(volume)
            building = payload["environment"]["buildings"]["habitat"][0]
            building["placement"]["non_uniform_scale_allowed"] = True
            metadata = {
                key: building[key]
                for key in (
                    "native_dimensions_m",
                    "ground_anchor_m",
                    "anchor_validation",
                    "lod",
                    "materials",
                    "placement",
                )
            }
            building["metadata_validation_sha256"] = _canonical_digest(
                metadata
            )
            _write_json(manifest, payload)

            with self.assertRaisesRegex(
                ValueError,
                "preserve native proportions",
            ):
                omniverse_pod.validate_materialized_assets(
                    manifest_path=manifest,
                    volume_root=volume,
                )

    def test_materialized_asset_manifest_rejects_remote_usd_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, payload = _write_materialized_asset_manifest(volume)
            environment = payload["environment"]
            self.assertIsInstance(environment, dict)
            vegetation = environment["vegetation"]
            self.assertIsInstance(vegetation, dict)
            remote_entry = vegetation["trees"][0]
            self.assertIsInstance(remote_entry, dict)
            wrapper = manifest.parent / str(remote_entry["path"])
            wrapper.write_text(
                "#usda 1.0\n"
                "(\n"
                "    prepend references = "
                "@https://assets.example.test/tree.usd@\n"
                ")\n",
                encoding="utf-8",
            )
            remote_entry["sha256"] = _digest(wrapper)
            _write_json(manifest, payload)

            with self.assertRaisesRegex(
                RuntimeError,
                "still references a remote asset",
            ):
                omniverse_pod.validate_materialized_assets(
                    manifest_path=manifest,
                    volume_root=volume,
                )

    def test_native_quality_gate_validates_every_tree_and_building_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory) / "volume"
            manifest, _payload = _write_materialized_asset_manifest(volume)
            locked = omniverse_pod.validate_materialized_assets(
                manifest_path=manifest,
                volume_root=volume,
            )
            with (
                patch(
                    "fireviewer_sdg.ign_catalog._validate_usd_assets",
                    return_value={"count": 26, "quality": {}},
                ) as native,
                patch(
                    "fireviewer_sdg.ign_catalog._assert_asset_quality",
                    return_value={"mesh_point_count": 1000},
                ) as quality,
            ):
                receipt = omniverse_pod.validate_native_asset_quality(
                    materialized_receipt=locked,
                    volume_root=volume,
                )
        self.assertEqual(receipt["validated_assets"], 26)
        native.assert_called_once()
        self.assertEqual(quality.call_count, 26)
        families = [call.kwargs["family"] for call in quality.call_args_list]
        self.assertEqual(families.count("vegetation"), 16)
        self.assertEqual(families.count("rural_building"), 10)

    def test_campaign_index_has_twenty_unique_blocked_simulation_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = root / "volume"
            catalog = root / "catalog"
            catalog.mkdir()
            manifest, _payload = _write_materialized_asset_manifest(volume)
            zone_manifests: dict[str, Path] = {}
            for zone_id in omniverse_pod.ZONE_ORDER:
                path = catalog / "manifests" / f"{zone_id}.json"
                _write_json(path, {"zone_id": zone_id})
                zone_manifests[zone_id] = path

            def fake_zone_manifest(
                _catalog: Path,
                zone_id: str,
            ) -> tuple[Path, dict[str, str]]:
                return zone_manifests[zone_id], {"zone_id": zone_id}

            def fake_zone_rows(
                _catalog: Path,
                zone_id: str,
            ) -> list[dict[str, object]]:
                zone_offset = omniverse_pod.ZONE_ORDER.index(zone_id) * 100_000
                return [
                    {
                        "tile_ref": f"{zone_id}-{index:03d}",
                        "xmin": zone_offset + (index % 20) * 1_000,
                        "ymin": (index // 20) * 1_000,
                        "xmax": zone_offset + (index % 20 + 1) * 1_000,
                        "ymax": (index // 20 + 1) * 1_000,
                    }
                    for index in range(400)
                ]

            output = volume / "campaign" / "index.json"
            with (
                patch.object(
                    omniverse_pod,
                    "validate_catalog",
                    return_value={"state": "CATALOG_VALIDATED"},
                ),
                patch.object(
                    omniverse_pod,
                    "_zone_manifest",
                    side_effect=fake_zone_manifest,
                ),
                patch.object(
                    omniverse_pod,
                    "_zone_rows",
                    side_effect=fake_zone_rows,
                ),
            ):
                index = omniverse_pod.build_campaign_index(
                    catalog_root=catalog,
                    asset_manifest=manifest,
                    volume_root=volume,
                    output_path=output,
                    base_zones=tuple(
                        reversed(omniverse_pod.DEFAULT_BASE_ZONES)
                    ),
                )

        simulations = index["simulations"]
        self.assertEqual(len(simulations), 20)
        self.assertEqual(
            len({slot["simulation_id"] for slot in simulations}),
            20,
        )
        self.assertEqual(len({slot["seed"] for slot in simulations}), 20)
        self.assertEqual(
            {slot["base_zone_id"] for slot in simulations},
            set(omniverse_pod.DEFAULT_BASE_ZONES),
        )
        self.assertEqual(
            {
                slot["base_zone_id"]: sum(
                    candidate["base_zone_id"] == slot["base_zone_id"]
                    for candidate in simulations
                )
                for slot in simulations
            },
            {
                zone_id: omniverse_pod.VARIANTS_PER_BASE
                for zone_id in omniverse_pod.DEFAULT_BASE_ZONES
            },
        )
        self.assertEqual(
            [scene["zone_id"] for scene in index["source_scenes"]],
            sorted(omniverse_pod.DEFAULT_BASE_ZONES),
        )
        self.assertEqual(
            {
                slot["scene_binding"]["root_usd"]
                for slot in simulations
            },
            {
                f"variant-scenes/SIM-{slot_index:02d}/build/root.usdc"
                for slot_index in range(1, 21)
            },
        )
        self.assertEqual(
            {slot["state"] for slot in simulations},
            {"blocked_pending_editor_review"},
        )
        self.assertEqual(
            index["fire_simulation_status"],
            "blocked_pending_editor_review",
        )

    def test_campaign_never_selects_four_base_scenes_automatically(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "must be supplied explicitly",
        ):
            omniverse_pod.build_campaign_index(
                catalog_root=Path("unused-catalog"),
                asset_manifest=Path("unused-assets.json"),
                volume_root=Path("unused-volume"),
                output_path=Path("unused-output.json"),
            )

    def test_pending_review_receipt_has_no_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                pending,
            ) = _write_gate_inputs(Path(directory))
            receipt = json.loads(pending.read_text(encoding="utf-8"))
            asset_receipt = omniverse_pod.validate_materialized_assets(
                manifest_path=assets,
                volume_root=volume,
            )
            build_payload = json.loads(build.read_text(encoding="utf-8"))
            build_inventory = omniverse_pod._verify_build_artifacts(
                build_payload=build_payload,
                build_receipt=build,
                root_usd=root_usd,
                allowed_root=volume,
            )
            scene_payload = json.loads(scene_validation.read_text(encoding="utf-8"))
            scene_inventory = omniverse_pod._verify_scene_validation_layers(
                scene_payload=scene_payload,
                volume_root=volume,
            )
            expected_bindings = {
                "runtime_preflight_sha256": _digest(runtime),
                "campaign_index_sha256": _digest(campaign),
                "asset_manifest_sha256": _digest(assets),
                "asset_content_sha256": asset_receipt["asset_content_sha256"],
                "root_usd_sha256": _digest(root_usd),
                "build_receipt_sha256": _digest(build),
                "scene_auto_validation_sha256": _digest(scene_validation),
                "build_artifact_content_sha256": build_inventory[
                    "artifact_content_sha256"
                ],
                "scene_layer_content_sha256": scene_inventory[
                    "layer_content_sha256"
                ],
            }

        self.assertEqual(receipt["human_review"], "pending")
        self.assertEqual(
            receipt["fire_simulation_status"],
            "blocked_pending_editor_review",
        )
        self.assertNotIn("decision", receipt)
        self.assertEqual(receipt["bindings"], expected_bindings)

    def test_sim01_pending_review_is_bound_to_its_canonical_base_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                _base_pending,
            ) = _write_gate_inputs(Path(directory))
            scene_root = build.parent.parent
            build_payload = json.loads(build.read_text(encoding="utf-8"))
            extra_artifacts: dict[str, Path] = {}
            for key in (
                "payloads",
                "detail_payloads",
                "detail_mid_payloads",
                "detail_far_payloads",
                "water_payloads",
            ):
                records = []
                count = 1 if key == "water_payloads" else 400
                for artifact_index in range(count):
                    artifact = (
                        build.parent
                        / key
                        / f"tile-{artifact_index:03d}.usdc"
                    )
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text(
                        f'#usda 1.0\ndef Xform "{key}" {{}}\n',
                        encoding="utf-8",
                    )
                    records.append(
                        {
                            "path": artifact.relative_to(
                                scene_root
                            ).as_posix(),
                            "sha256": _digest(artifact),
                        }
                    )
                build_payload[key] = records
                extra_artifacts[key] = artifact
            ground_root = build.parent / "materials"
            ground_root.mkdir(parents=True, exist_ok=True)
            ground_index = ground_root / "index.usda"
            ground_index.write_text(
                '#usda 1.0\ndef Scope "GroundIndex" {}\n',
                encoding="utf-8",
            )
            ground_tiles = []
            for index in range(400):
                ground = ground_root / f"ground-{index:03d}.usda"
                ground.write_text(
                    '#usda 1.0\ndef Material "Ground" {}\n',
                    encoding="utf-8",
                )
                x = float(index % 20) * 1_000.0
                y = float(index // 20) * 1_000.0
                ground_tiles.append(
                    {
                        "tile_id": f"T{index:03d}",
                        "tile_bounds_m": [x, y, x + 1_000.0, y + 1_000.0],
                        "path": ground.relative_to(scene_root).as_posix(),
                        "sha256": _digest(ground),
                        "prim_path": "/GroundMaterial",
                    }
                )
            identity_sha = "a" * 64
            build_payload.update(
                {
                    "scene_kind": "fictive_variant",
                    "zone_id": "SIM-01",
                    "base_scene_id": "Z16",
                    "variant_index": 1,
                    "ground_material": {
                        "topology": (
                            "payload_tiled_materials_shared_pbr_library"
                        ),
                        "index": {
                            "path": ground_index.relative_to(
                                scene_root
                            ).as_posix(),
                            "sha256": _digest(ground_index),
                            "prim_path": "/GroundIndex",
                        },
                        "tile_material_payloads": ground_tiles,
                        "binding_scope": (
                            "per_terrain_tile_stronger_than_descendants"
                        ),
                    },
                    "layers": {
                        **build_payload["layers"],
                        "terrain": {
                            "ground_material_topology": (
                                "payload_tiled_materials_shared_pbr_library"
                            ),
                            "ground_material_payload_count": 400,
                            "global_ground_material_binding": False,
                        },
                    },
                    "identity_contract": {
                        "numeric_ids_preserved": True,
                        "stable_ids_preserved": True,
                        "source_namespace_may_differ_from_destination_tile": True,
                        "source_identity_sha256": identity_sha,
                        "authored_identity_sha256": identity_sha,
                    },
                }
            )
            _write_json(build, build_payload)
            validation_payload = json.loads(
                scene_validation.read_text(encoding="utf-8")
            )
            validation_payload["build_receipt_sha256"] = _digest(build)
            _write_json(scene_validation, validation_payload)
            plan_metadata = volume / "variant-plan" / "SIM-01" / "variant.json"
            _write_json(
                plan_metadata,
                {
                    "simulation_id": "SIM-01",
                    "base_scene_id": "Z16",
                    "variant_index": 1,
                    "fire_simulation_status": (
                        "blocked_pending_editor_review"
                    ),
                },
            )
            authoring_receipt = (
                volume / "variant-scenes" / "authoring-receipt.json"
            )
            authored_variants = [
                {
                    "simulation_id": f"SIM-{index:02d}",
                    "base_scene_id": "Z16",
                    "variant_index": ((index - 1) % 5) + 1,
                    "artifacts": (
                        {
                            "composer_build_receipt": {
                                "path": build.relative_to(
                                    scene_root
                                ).as_posix(),
                                "sha256": _digest(build),
                            }
                        }
                        if index == 1
                        else {}
                    ),
                    "fire_simulation_status": (
                        "blocked_pending_editor_review"
                    ),
                }
                for index in range(1, 21)
            ]
            _write_json(
                authoring_receipt,
                {
                    "state": "VARIANT_USD_AUTHORED",
                    "simulation_count": 20,
                    "variants": authored_variants,
                    "fire_simulation_status": (
                        "blocked_pending_editor_review"
                    ),
                },
            )
            campaign_payload = json.loads(campaign.read_text(encoding="utf-8"))
            campaign_payload["simulations"] = [
                {
                    "simulation_id": "SIM-01",
                    "base_zone_id": "Z16",
                    "variant_index": 1,
                    "state": "blocked_pending_editor_review",
                    "scene_binding": {
                        "root_usd": root_usd.relative_to(volume).as_posix(),
                        "build_receipt": build.relative_to(volume).as_posix(),
                        "composition_plan": plan_metadata.relative_to(
                            volume
                        ).as_posix(),
                        "portfolio_authoring_receipt": (
                            authoring_receipt.relative_to(volume).as_posix()
                        ),
                    },
                }
            ]
            _write_json(campaign, campaign_payload)
            pending = scene_root / "sim01-review-pending.json"
            internal_qa = scene_root / "sim01-internal-qa.json"

            with self.assertRaisesRegex(
                RuntimeError,
                "requires the current internal QA receipt",
            ):
                omniverse_pod.create_review_pending(
                    scene_id="SIM-01",
                    root_usd=root_usd,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                    output_path=pending,
                )
            _write_json(
                internal_qa,
                {
                    "state": "SIM01_INTERNAL_QA_PASSED",
                    "simulation_id": "SIM-01",
                    "review_handoff_ready": True,
                    "fire_simulation_status": (
                        "blocked_pending_editor_review"
                    ),
                    "bindings": {
                        "root_usd_sha256": _digest(root_usd),
                        "build_receipt_sha256": _digest(build),
                        "asset_manifest_sha256": _digest(assets),
                        "inputs": {
                            "scene_auto_validation": {
                                "sha256": _digest(scene_validation),
                            }
                        },
                    },
                },
            )

            receipt = omniverse_pod.create_review_pending(
                scene_id="SIM-01",
                root_usd=root_usd,
                runtime_preflight=runtime,
                campaign_index=campaign,
                asset_manifest=assets,
                volume_root=volume,
                build_receipt=build,
                scene_auto_validation=scene_validation,
                internal_qa_receipt=internal_qa,
                output_path=pending,
            )
            self.assertEqual(receipt["scene_id"], "SIM-01")
            self.assertIsNone(receipt["zone_id"])
            self.assertEqual(
                receipt["bindings"]["internal_qa_receipt_sha256"],
                _digest(internal_qa),
            )
            opened = scene_root / "sim01-editor-opened.json"
            _write_json(
                opened,
                {
                    "state": "opened_for_human_review",
                    "human_review": "pending",
                    "root_usd_sha256": receipt["bindings"][
                        "root_usd_sha256"
                    ],
                    "pending_review_sha256": _digest(pending),
                },
            )
            acceptance = scene_root / "sim01-review-accepted.json"
            omniverse_pod.accept_review(
                pending_path=pending,
                opened_path=opened,
                output_path=acceptance,
                reviewer="test-reviewer",
                acknowledgement=omniverse_pod.REVIEW_ACKNOWLEDGEMENT,
            )
            allowed = omniverse_pod.assert_simulation_allowed(
                acceptance_path=acceptance,
                runtime_preflight=runtime,
                campaign_index=campaign,
                asset_manifest=assets,
                volume_root=volume,
                root_usd=root_usd,
                build_receipt=build,
                scene_auto_validation=scene_validation,
                internal_qa_receipt=internal_qa,
            )
            self.assertEqual(allowed["state"], "FIRE_SIMULATION_ALLOWED")
            self.assertIn(
                "composition_plan_sha256",
                allowed["bindings"],
            )

            build_payload["base_scene_id"] = "Z17"
            _write_json(build, build_payload)
            validation_payload["build_receipt_sha256"] = _digest(build)
            _write_json(scene_validation, validation_payload)
            authored_variants[0]["artifacts"]["composer_build_receipt"][
                "sha256"
            ] = _digest(build)
            _write_json(
                authoring_receipt,
                {
                    "state": "VARIANT_USD_AUTHORED",
                    "simulation_count": 20,
                    "variants": authored_variants,
                    "fire_simulation_status": (
                        "blocked_pending_editor_review"
                    ),
                },
            )
            qa_payload = json.loads(
                internal_qa.read_text(encoding="utf-8")
            )
            qa_payload["bindings"]["build_receipt_sha256"] = _digest(build)
            qa_payload["bindings"]["inputs"]["scene_auto_validation"][
                "sha256"
            ] = _digest(scene_validation)
            _write_json(internal_qa, qa_payload)
            with self.assertRaisesRegex(
                RuntimeError,
                "differs from its campaign simulation slot",
            ):
                omniverse_pod.create_review_pending(
                    scene_id="SIM-01",
                    root_usd=root_usd,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                    internal_qa_receipt=internal_qa,
                    output_path=pending,
                )

    def test_pending_review_rejects_a_smaller_container_ram_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                pending,
            ) = _write_gate_inputs(Path(directory))
            runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
            runtime_payload["system_memory"]["effective_mib"] = 96_000
            _write_json(runtime, runtime_payload)

            with self.assertRaisesRegex(
                RuntimeError,
                "finite container cgroup RAM limit",
            ):
                omniverse_pod.create_review_pending(
                    zone_id="Z16",
                    root_usd=root_usd,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                    output_path=pending,
                )

    def test_pending_review_rejects_a_different_rtx_6000_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                pending,
            ) = _write_gate_inputs(Path(directory))
            runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
            runtime_payload["gpu"]["name"] = "NVIDIA RTX 6000 Ada Generation"
            _write_json(runtime, runtime_payload)

            with self.assertRaisesRegex(
                RuntimeError,
                "RTX PRO 6000 Blackwell Server Edition 96 GB profile",
            ):
                omniverse_pod.create_review_pending(
                    zone_id="Z16",
                    root_usd=root_usd,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                    output_path=pending,
                )

    def test_simulation_gate_rejects_pending_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                pending,
            ) = _write_gate_inputs(Path(directory))
            with self.assertRaisesRegex(
                RuntimeError,
                "blocked pending Editor acceptance",
            ):
                omniverse_pod.assert_simulation_allowed(
                    acceptance_path=pending,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    root_usd=root_usd,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                )

    def test_simulation_gate_requires_current_accepted_hash_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                pending,
            ) = _write_gate_inputs(root)
            pending_receipt = json.loads(pending.read_text(encoding="utf-8"))
            opened = root / "editor-opened.json"
            _write_json(
                opened,
                {
                    "state": "opened_for_human_review",
                    "human_review": "pending",
                    "root_usd_sha256": pending_receipt["bindings"][
                        "root_usd_sha256"
                    ],
                    "pending_review_sha256": _digest(pending),
                },
            )
            acceptance = root / "review-accepted.json"
            accepted = omniverse_pod.accept_review(
                pending_path=pending,
                opened_path=opened,
                output_path=acceptance,
                reviewer="test-reviewer",
                acknowledgement=omniverse_pod.REVIEW_ACKNOWLEDGEMENT,
            )

            allowed = omniverse_pod.assert_simulation_allowed(
                acceptance_path=acceptance,
                runtime_preflight=runtime,
                campaign_index=campaign,
                asset_manifest=assets,
                volume_root=volume,
                root_usd=root_usd,
                build_receipt=build,
                scene_auto_validation=scene_validation,
            )
            self.assertEqual(accepted["decision"], "accepted")
            self.assertEqual(allowed["state"], "FIRE_SIMULATION_ALLOWED")

            build_payload = json.loads(build.read_text(encoding="utf-8"))
            payload_path = (
                build.parent.parent / build_payload["payloads"][0]["path"]
            ).resolve()
            original_payload = payload_path.read_bytes()
            payload_path.write_text(
                '#usda 1.0\ndef Xform "TamperedTile" {}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "payloads\\[0\\] SHA-256 mismatch"):
                omniverse_pod.assert_simulation_allowed(
                    acceptance_path=acceptance,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    root_usd=root_usd,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                )
            payload_path.write_bytes(original_payload)

            root_usd.write_text(
                '#usda 1.0\ndef Xform "ChangedWorld" {}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "build root USD SHA-256 mismatch",
            ):
                omniverse_pod.assert_simulation_allowed(
                    acceptance_path=acceptance,
                    runtime_preflight=runtime,
                    campaign_index=campaign,
                    asset_manifest=assets,
                    volume_root=volume,
                    root_usd=root_usd,
                    build_receipt=build,
                    scene_auto_validation=scene_validation,
                )

    def test_simulation_gate_rehashes_materialized_asset_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                volume,
                runtime,
                campaign,
                assets,
                root_usd,
                build,
                scene_validation,
                pending,
            ) = _write_gate_inputs(root)
            pending_receipt = json.loads(pending.read_text(encoding="utf-8"))
            opened = root / "editor-opened.json"
            _write_json(
                opened,
                {
                    "state": "opened_for_human_review",
                    "human_review": "pending",
                    "root_usd_sha256": pending_receipt["bindings"][
                        "root_usd_sha256"
                    ],
                    "pending_review_sha256": _digest(pending),
                },
            )
            acceptance = root / "review-accepted.json"
            omniverse_pod.accept_review(
                pending_path=pending,
                opened_path=opened,
                output_path=acceptance,
                reviewer="test-reviewer",
                acknowledgement=omniverse_pod.REVIEW_ACKNOWLEDGEMENT,
            )
            manifest_payload = json.loads(assets.read_text(encoding="utf-8"))
            for kind, family_minimums in (
                omniverse_pod.PHOTOREAL_FAMILY_MINIMUMS.items()
            ):
                for family in family_minimums:
                    source_relative = manifest_payload["environment"][kind][
                        family
                    ][0]["materialized_files"][0]["path"]
                    source = volume / source_relative
                    original = source.read_bytes()
                    source.write_text(
                        '#usda 1.0\ndef Xform "Tampered" {}\n',
                        encoding="utf-8",
                    )
                    with self.subTest(family=f"{kind}.{family}"):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "dependency SHA-256 mismatch",
                        ):
                            omniverse_pod.assert_simulation_allowed(
                                acceptance_path=acceptance,
                                runtime_preflight=runtime,
                                campaign_index=campaign,
                                asset_manifest=assets,
                                volume_root=volume,
                                root_usd=root_usd,
                                build_receipt=build,
                                scene_auto_validation=scene_validation,
                            )
                    source.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
