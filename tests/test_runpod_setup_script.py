from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "tools" / "runpod" / "setup-omniverse-pod.sh"
ENVIRONMENT = ROOT / "config" / "runpod-geospatial-env.yml"
EXAMPLE = ROOT / "config" / "runpod-omniverse-editor.env.example"
REVIEW = ROOT / "tools" / "runpod" / "start-omniverse-review.sh"


class RunPodSetupScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = SETUP.read_text(encoding="utf-8")
        cls.environment = ENVIRONMENT.read_text(encoding="utf-8")
        cls.example = EXAMPLE.read_text(encoding="utf-8")
        cls.review = REVIEW.read_text(encoding="utf-8")

    def test_geospatial_toolchain_is_exact_and_separate_from_kit(self) -> None:
        self.assertIn('MICROMAMBA_RELEASE="2.6.2-1"', self.setup)
        self.assertIn(
            'MICROMAMBA_SHA256="'
            "e9683b483df06dbd3fdd8a37f1b6826d7e5caf4e85bf15a0af4fbad3d4ad1a58"
            '"',
            self.setup,
        )
        self.assertIn(
            "https://github.com/mamba-org/micromamba-releases/releases/download/",
            self.setup,
        )
        self.assertIn("pdal=2.10.2", self.environment)
        self.assertRegex(self.environment, r"(?m)^  - gdal$")
        self.assertIn('GEOSPATIAL_ENV="${RUNTIME_ROOT}/geospatial-pdal-2.10.2"', self.setup)
        self.assertNotIn("${ISAAC_ROOT}/bin/pdal", self.setup)
        self.assertNotIn("${KIT_ROOT}/bin/pdal", self.setup)

    def test_first_solve_produces_and_reuse_verifies_cep23_lock(self) -> None:
        self.assertIn("--explicit", self.setup)
        self.assertIn("--sha256", self.setup)
        self.assertIn("@EXPLICIT", self.setup)
        self.assertIn("# platform: linux-64", self.setup)
        self.assertIn("printf '# platform: linux-64\\n'", self.setup)
        self.assertIn("validate_cep23_lock", self.setup)
        self.assertIn("compare_environment_to_lock", self.setup)
        self.assertIn("write_geospatial_solve_marker", self.setup)
        self.assertIn("validate_geospatial_solve_marker", self.setup)
        self.assertIn("--file \"${GEOSPATIAL_LOCK}\"", self.setup)
        self.assertIn("--channel-priority strict", self.setup)
        self.assertRegex(
            self.setup,
            r"GEOSPATIAL_LOCK=.*runpod-geospatial-linux-64\.cep23\.txt",
        )

    def test_driver_and_version_evidence_is_persisted(self) -> None:
        for expected in (
            "pdal-drivers.txt",
            "gdal-raster-drivers.txt",
            "ogr-vector-drivers.txt",
            "gpu-driver.txt",
            "GEOSPATIAL_RUNTIME_LOCKED",
            "readers.copc",
            "readers.las",
            "filters.stats",
            "filters.hag_dem",
            "filters.expression",
            "filters.reprojection",
            "writers.gdal",
            "writers.las",
            "GTiff",
            "COG",
            "GeoJSON",
            "GPKG",
        ):
            self.assertIn(expected, self.setup)

    def test_lidar_is_probed_before_the_native_scene_build(self) -> None:
        self.assertIn("ensure_zone_lidar_evidence", self.setup)
        self.assertIn("-m fireviewer_sdg.lidar_evidence create", self.setup)
        self.assertIn("-m fireviewer_sdg.lidar_evidence verify", self.setup)
        for classification in (2, 5, 6):
            self.assertIn(f"--require-class {classification}", self.setup)
        evidence = self.setup.index(
            'ensure_zone_lidar_evidence "${zone}"',
            self.setup.index("build_base_scene()"),
        )
        build = self.setup.index(
            'run_zone_phase "${ISAAC_PYTHON}" "${zone}" build',
            self.setup.index("build_base_scene()"),
        )
        self.assertLess(evidence, build)

    def test_required_hardware_and_production_budget_are_not_downgraded(self) -> None:
        self.assertIn("iproute2", self.setup)
        self.assertIn('BLACKWELL_MIN_DRIVER_VERSION="570.158.01"', self.setup)
        self.assertIn('FW_OMNI_MIN_VRAM_MIB:-90000', self.setup)
        self.assertIn(
            "FW_OMNI_REQUIRED_GPU_NAME:-RTX PRO 6000 Blackwell Server Edition",
            self.setup,
        )
        self.assertIn('FW_OMNI_MIN_SYSTEM_RAM_MIB:-138000', self.setup)
        self.assertIn("--minimum-system-ram-mib", self.setup)
        self.assertIn("/sys/fs/cgroup/memory.max", self.setup)
        self.assertIn(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            self.setup,
        )
        self.assertIn("cgroup_limit < system_ram_bytes", self.setup)
        self.assertIn("no finite container cgroup memory limit", self.setup)
        self.assertIn(
            "/proc/meminfo describes the RunPod host and is never accepted",
            self.setup,
        )
        self.assertNotIn("awk '/^MemTotal:/", self.setup)
        self.assertIn('FW_OMNI_STORAGE_MODE:-ephemeral-nvme', self.setup)
        self.assertIn('FW_OMNI_MIN_STORAGE_GB:-1500', self.setup)
        self.assertIn("--minimum-storage-gb", self.setup)
        self.assertIn("FW_OMNI_STORAGE_MODE=ephemeral-nvme", self.example)
        self.assertIn("FW_OMNI_MIN_STORAGE_GB=1500", self.example)
        self.assertIn('FW_SDG_FOREST_INSTANCE_BUDGET:-2500000', self.setup)
        self.assertIn("FW_OMNI_MIN_VRAM_MIB=90000", self.example)
        self.assertIn(
            "FW_OMNI_REQUIRED_GPU_NAME=RTX PRO 6000 Blackwell Server Edition",
            self.example,
        )
        self.assertIn("FW_OMNI_MIN_SYSTEM_RAM_MIB=138000", self.example)
        self.assertIn("FW_SDG_FOREST_INSTANCE_BUDGET=2500000", self.example)
        self.assertIn("FW_OMNI_EDITOR_TARGET_FPS=60", self.example)
        self.assertIn(
            'touch "${KIT_ROOT}/.omniverse_eula_accepted.txt"',
            self.setup,
        )
        self.assertIn(
            'FW_OMNI_EDITOR_TARGET_FPS="${FW_OMNI_EDITOR_TARGET_FPS:-60}"',
            self.review,
        )
        self.assertNotIn("FW_SDG_FOREST_INSTANCE_BUDGET=180000", self.example)

    def test_review_launcher_opens_only_the_final_sim01_variant(self) -> None:
        self.assertNotIn("FW_OMNI_PILOT_ZONE", self.review)
        self.assertIn('BASE_ZONES_CSV="${FW_OMNI_BASE_ZONES:-}"', self.review)
        self.assertIn('REVIEW_SCENE="${FW_OMNI_REVIEW_SCENE:-SIM-01}"', self.review)
        self.assertIn(
            'VARIANT_SCENES_ROOT="${FW_OMNI_VARIANT_SCENES_ROOT:-${VOLUME_ROOT}/variant-scenes}"',
            self.review,
        )
        self.assertIn('ROOT_USD="${SCENE_ROOT}/build/root.usdc"', self.review)
        self.assertIn('OPENED_RECEIPT="${SCENE_ROOT}/review-opened.json"', self.review)
        self.assertIn(
            "FW_OMNI_BASE_ZONES must contain exactly four scene identifiers",
            self.review,
        )
        self.assertIn(
            "FW_OMNI_BASE_ZONES contains a duplicate scene",
            self.review,
        )
        self.assertIn(
            "the pre-simulation manual review target must be SIM-01",
            self.review,
        )
        self.assertIn('EDITOR_BINDING="${STATE_ROOT}/editor-root.sha256"', self.review)
        self.assertIn(
            'FW_SDG_REVIEW_BUILD_RECEIPT="${BUILD_RECEIPT}"',
            self.review,
        )
        self.assertNotIn("zone_scenes --phase review", self.review)

    def test_campaign_requires_four_explicit_base_scene_ids(self) -> None:
        self.assertIn('BASE_ZONES_CSV="${FW_OMNI_BASE_ZONES:-}"', self.setup)
        self.assertIn(
            "FW_OMNI_BASE_ZONES must contain exactly four comma-separated zones",
            self.setup,
        )
        self.assertIn('base_args+=(--base-zone "${zone}")', self.setup)
        self.assertIn("build_all_base_scenes()", self.setup)
        self.assertIn('build_base_scene "${zone}"', self.setup)
        self.assertIn("ALL_BASE_SCENES_READY count=4", self.setup)
        self.assertIn("FW_OMNI_BASE_ZONES=", self.example)
        self.assertNotIn("FW_OMNI_BASE_ZONES=Z", self.example)

    def test_source_and_final_campaign_bundles_are_safely_installed(self) -> None:
        for expected in (
            "FW_OMNI_ASSET_BUNDLE_URL",
            "FW_OMNI_ASSET_BUNDLE_SHA256",
            "FW_OMNI_ASSET_BUNDLE_ALLOWED_HOSTS",
            "FW_OMNI_ASSET_BUNDLE_MANIFEST_RELATIVE",
            "FW_OMNI_ASSET_BUNDLE_MAX_FILES",
            "FW_OMNI_ASSET_BUNDLE_MAX_UNPACKED_GIB",
            "FW_OMNI_ASSET_BUNDLE_MAX_ARCHIVE_GIB",
            "FW_OMNI_ASSET_BUNDLE_MIN_FREE_AFTER_INSTALL_GIB",
            "FW_OMNI_ASSET_BUNDLE_CONNECT_TIMEOUT_SECONDS",
            "FW_OMNI_ASSET_BUNDLE_DOWNLOAD_TIMEOUT_SECONDS",
        ):
            self.assertIn(expected, self.setup)
            self.assertIn(expected, self.example)
        self.assertIn("--proto '=https'", self.setup)
        self.assertIn("validate_asset_bundle_source", self.setup)
        self.assertIn("-m fireviewer_sdg.asset_bundle", self.setup)
        self.assertIn("--sha256 \"${ASSET_BUNDLE_SHA256}\"", self.setup)
        self.assertIn("--destination-root \"${ASSET_BUNDLE_ROOT}\"", self.setup)
        self.assertNotIn("--native-lod-receipt", self.setup)
        self.assertNotIn("--native-pbr-receipt", self.setup)
        self.assertIn("asset-bundle-native-lods.json", self.setup)
        self.assertIn("asset-bundle-native-pbr.json", self.setup)
        self.assertIn("--connect-timeout", self.setup)
        self.assertIn("--max-time", self.setup)
        self.assertIn("--max-filesize", self.setup)
        self.assertIn("--minimum-free-after-install-gib", self.setup)
        self.assertIn("shutil.disk_usage(partial.parent).free", self.setup)
        self.assertIn(
            "asset bundle download would consume the workspace reserve",
            self.setup,
        )
        self.assertIn(
            "-m fireviewer_sdg.campaign_asset_bundle",
            self.setup,
        )
        self.assertIn("validate_native_lod_quality", self.setup)
        self.assertIn("validate_native_pbr_quality", self.setup)
        self.assertIn("verify_campaign_asset_bundle", self.setup)
        self.assertIn("verify_native_quality_receipts", self.setup)
        campaign = self.setup.index("write_campaign_contract()")
        verify = self.setup.index(
            "bind_campaign_asset_bundle",
            campaign,
        )
        campaign_command = self.setup.index(
            "-m fireviewer_sdg.omniverse_pod campaign-index",
            campaign,
        )
        self.assertLess(verify, campaign_command)
        self.assertIn("--native-usd-quality", self.setup)
        for role in (
            "forest_floor",
            "grass",
            "soil",
            "rock",
            "asphalt",
            "gravel",
            "water",
        ):
            self.assertIn(role, self.example)
        self.assertIn("HERO/MID/FAR", self.example)
        self.assertNotIn("FW_OMNI_ASSET_BUNDLE_URL=https://", self.example)
        self.assertNotIn("FW_OMNI_ASSET_BUNDLE_SHA256=0", self.example)

    def test_configured_bundle_merges_corrected_nvidia_before_community(self) -> None:
        match = re.search(
            r"(?ms)^materialize_assets\(\) \{\n(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        branch = body.index('if [[ "${ASSET_BUNDLE_ENABLED}" -eq 1 ]]')
        community_resume = body.index(
            "community_source_merge_resume_ready",
            branch,
        )
        source_resume = body.index(
            "source_manifest_merge_resume_ready",
            community_resume,
        )
        curated = body.index("install_curated_asset_bundle", branch)
        official = body.index(
            "provision_official_nvidia_source_manifest",
            curated,
        )
        merge = body.index(
            "merge_curated_and_official_sources",
            official,
        )
        community = body.index("install_community_building_sources", merge)
        assemble = body.index("assemble_campaign_asset_bundle", community)
        select = body.index("select_campaign_asset_bundle", assemble)
        native = body.index("--native-usd-quality", select)
        bind = body.index("bind_campaign_asset_bundle", native)
        self.assertLess(branch, community_resume)
        self.assertLess(community_resume, source_resume)
        self.assertLess(source_resume, curated)
        self.assertLess(curated, official)
        self.assertLess(official, merge)
        self.assertLess(merge, community)
        self.assertLess(community, assemble)
        self.assertLess(assemble, select)
        self.assertLess(select, native)
        self.assertLess(native, bind)
        self.assertIn(
            "corrected official NVIDIA inventory must leave exactly the three",
            self.setup,
        )

    def test_materialize_assets_executes_fresh_and_resume_branches(self) -> None:
        bash = shutil.which("bash")
        git_bash = Path(r"D:\Programs\Git\bin\bash.exe")
        if git_bash.is_file():
            bash = str(git_bash)
        if not bash:
            self.skipTest("Bash is unavailable")
        probe = subprocess.run(
            [bash, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            self.skipTest("Bash cannot execute in this environment")
        function = re.search(
            r"(?ms)^materialize_assets\(\) \{\n.*?^\}$",
            self.setup,
        )
        self.assertIsNotNone(function)
        harness = (
            "set -Eeuo pipefail\n"
            "CASE=\"$1\"\n"
            "EVENTS=()\n"
            "COMMUNITY_CALLS=0\n"
            "ASSET_BUNDLE_ENABLED=1\n"
            "CAMPAIGN_ASSET_MANIFEST=/definitely-absent/campaign.json\n"
            "CAMPAIGN_ASSET_RECEIPT=/definitely-absent/campaign-receipt.json\n"
            "ASSET_BUNDLE_NATIVE_LOD_RECEIPT=/definitely-absent/lod.json\n"
            "ASSET_BUNDLE_NATIVE_PBR_RECEIPT=/definitely-absent/pbr.json\n"
            "ASSET_RECEIPT=/definitely-absent/assets.json\n"
            "GROUND_BUNDLE_ROOT=/fixture/ground\n"
            "GROUND_MATERIAL_MANIFEST=/fixture/ground/manifest.json\n"
            "VOLUME_ROOT=/fixture/volume\n"
            "OFFICIAL_ASSET_MANIFEST=/fixture/official.json\n"
            "ASSET_MANIFEST=/fixture/source.json\n"
            "ISAAC_PYTHON=fake_isaac\n"
            "record(){ EVENTS+=(\"$1\"); }\n"
            "configure_asset_bundle_contract(){ record configure; }\n"
            "ensure_isaac_runtime(){ record ensure_isaac; }\n"
            "community_source_merge_resume_ready(){\n"
            "  COMMUNITY_CALLS=$((COMMUNITY_CALLS+1))\n"
            "  record \"community_gate:${COMMUNITY_CALLS}\"\n"
            "  if [[ \"${CASE}\" == community ]]; then return 0; fi\n"
            "  [[ \"${COMMUNITY_CALLS}\" -ge 2 ]]\n"
            "}\n"
            "source_manifest_merge_resume_ready(){\n"
            "  record source_gate\n"
            "  [[ \"${CASE}\" == base ]]\n"
            "}\n"
            "select_merged_source_manifest(){ record select_merged; }\n"
            "install_curated_asset_bundle(){ record install_curated; }\n"
            "provision_official_nvidia_source_manifest(){ record provision_official; }\n"
            "merge_curated_and_official_sources(){ record merge_sources; }\n"
            "install_community_building_sources(){ record install_community; }\n"
            "python3.12(){ record \"python:$*\"; }\n"
            "require_file(){ record \"require:$1\"; }\n"
            "assemble_campaign_asset_bundle(){ record assemble_campaign; }\n"
            "select_campaign_asset_bundle(){ record select_campaign; }\n"
            "fake_isaac(){ record \"isaac:$*\"; }\n"
            "bind_campaign_asset_bundle(){ record bind_campaign; }\n"
            "fail(){ record \"fail:$*\"; return 1; }\n"
            + function.group(0)
            + "\nmaterialize_assets\n"
            "printf 'EVENT:%s\\n' \"${EVENTS[@]}\"\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "materialize-harness.sh"
            script.write_text(harness, encoding="utf-8", newline="\n")
            observed: dict[str, list[str]] = {}
            for case in ("fresh", "base", "community"):
                result = subprocess.run(
                    [bash, str(script), case],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                observed[case] = [
                    line.removeprefix("EVENT:")
                    for line in result.stdout.splitlines()
                    if line.startswith("EVENT:")
                ]

        fresh = observed["fresh"]
        ordered = (
            "community_gate:1",
            "source_gate",
            "install_curated",
            "provision_official",
            "merge_sources",
            "install_community",
            "community_gate:2",
            "assemble_campaign",
        )
        positions = [fresh.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("fail:post-install community source merge verification failed", fresh)

        base = observed["base"]
        self.assertIn("source_gate", base)
        self.assertIn("select_merged", base)
        self.assertIn("install_community", base)
        self.assertIn("community_gate:2", base)
        self.assertNotIn("install_curated", base)
        self.assertNotIn("provision_official", base)
        self.assertNotIn("merge_sources", base)

        community = observed["community"]
        self.assertEqual(community.count("community_gate:1"), 1)
        self.assertIn("select_merged", community)
        self.assertNotIn("source_gate", community)
        self.assertNotIn("install_curated", community)
        self.assertNotIn("install_community", community)

    def test_uncurated_source_fails_before_community_when_actors_are_missing(
        self,
    ) -> None:
        verifier = re.search(
            r"(?ms)^require_exact_source_actor_classes\(\) \{\n"
            r".*?python3\.12 -c \\\n"
            r"        '(?P<program>.*?)' \\\n"
            r'        "\$\{ASSET_MANIFEST\}"',
            self.setup,
        )
        self.assertIsNotNone(verifier)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "official.json"
            manifest.write_text(
                json.dumps({"actors": {}}),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    verifier.group("program"),
                    str(manifest),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact seven reviewed actor classes", result.stderr)

        materialize = re.search(
            r"(?ms)^materialize_assets\(\) \{\n(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(materialize)
        body = materialize.group("body")
        uncurated = body.index(
            "provision_official_nvidia_source_manifest",
            body.index("else", body.index("ASSET_BUNDLE_ENABLED")),
        )
        actor_gate = body.index(
            "require_exact_source_actor_classes",
            uncurated,
        )
        community = body.index(
            "install_community_building_sources",
            actor_gate,
        )
        self.assertLess(uncurated, actor_gate)
        self.assertLess(actor_gate, community)

    def test_locked_objaverse_sources_are_downloaded_before_kit_install(self) -> None:
        for expected in (
            'OBJAVERSE_VERSION="0.1.7"',
            (
                'OBJAVERSE_CLIENT_ROOT="${RUNTIME_ROOT}/'
                'objaverse-client-${OBJAVERSE_VERSION}"'
            ),
            'OBJAVERSE_CLIENT_PYTHON="${OBJAVERSE_CLIENT_ROOT}/bin/python"',
            "download-community-building-assets.py",
            "objaverse-0.1.7-py3-none-any.whl",
            (
                'OBJAVERSE_WHEEL_SHA256="'
                "7396d119efde5794d0e87d3ca03047d0b0585b2a83ea381a8cc2ddc219d6f1a3"
                '"'
            ),
            "tqdm-4.67.1-py3-none-any.whl",
            (
                'OBJAVERSE_TQDM_WHEEL_SHA256="'
                "26445eca388f82e72884e0d580d5464cd801a3ea01e63e5601bdff9ba6a48de2"
                '"'
            ),
            "--no-deps",
            "--destination-root \"${COMMUNITY_BUILDING_SOURCE_ROOT}\"",
            "--cache-root \"${OBJAVERSE_CACHE_ROOT}\"",
            "--workers 4",
            "FW_OMNI_OBJAVERSE_DOWNLOAD_TIMEOUT_SECONDS",
            "--kill-after=60s",
            '"${ISAAC_PYTHON}" -m fireviewer_sdg.community_building_assets',
            "--source-root \"${COMMUNITY_BUILDING_SOURCE_ROOT}\"",
            "--metadata \"${COMMUNITY_BUILDING_METADATA}\"",
        ):
            self.assertIn(expected, self.setup)
        materialize = re.search(
            r"(?ms)^materialize_assets\(\) \{\n(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(materialize)
        body = materialize.group("body")
        community = body.index("install_community_building_sources")
        campaign_bundle = body.index(
            "assemble_campaign_asset_bundle",
            community,
        )
        selection = body.index(
            "select_campaign_asset_bundle",
            campaign_bundle,
        )
        native_validation = body.index(
            "-m fireviewer_sdg.omniverse_pod validate-assets",
            selection,
        )
        final_binding = body.index(
            "bind_campaign_asset_bundle",
            native_validation,
        )
        self.assertLess(community, campaign_bundle)
        self.assertLess(campaign_bundle, selection)
        self.assertLess(selection, native_validation)
        self.assertLess(native_validation, final_binding)
        installer = re.search(
            r"(?ms)^install_community_building_sources\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(installer)
        installer_body = installer.group("body")
        download = installer_body.index('"${OBJAVERSE_CLIENT_PYTHON}"')
        kit = installer_body.index(
            '"${ISAAC_PYTHON}" -m fireviewer_sdg.community_building_assets'
        )
        self.assertLess(download, kit)
        self.assertNotIn("GPUtil", self.setup)

    def test_campaign_index_requires_current_final_native_asset_receipt(
        self,
    ) -> None:
        verifier = re.search(
            r"(?ms)^verify_materialized_asset_receipt\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(verifier)
        verifier_body = verifier.group("body")
        for expected in (
            'require_file "${ASSET_RECEIPT}"',
            "validate_materialized_assets",
            "observed != expected",
            "fireviewer_native_usd_photoreal_quality_v2",
            "observed_identities != expected_identities",
            "MATERIALIZED_ASSET_RECEIPT_CURRENT",
        ):
            self.assertIn(expected, verifier_body)

        binding = re.search(
            r"(?ms)^bind_campaign_asset_bundle\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(binding)
        binding_body = binding.group("body")
        selection = binding_body.index("select_campaign_asset_bundle")
        receipt = binding_body.index(
            "verify_materialized_asset_receipt",
            selection,
        )
        self.assertLess(selection, receipt)

        campaign = re.search(
            r"(?ms)^write_campaign_contract\(\) \{\n(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(campaign)
        campaign_body = campaign.group("body")
        binding_call = campaign_body.index("bind_campaign_asset_bundle")
        index_call = campaign_body.index(
            "-m fireviewer_sdg.omniverse_pod campaign-index",
            binding_call,
        )
        self.assertLess(binding_call, index_call)
        self.assertIn(
            "FW_OMNI_OBJAVERSE_DOWNLOAD_TIMEOUT_SECONDS=21600",
            self.example,
        )

    def test_asset_resume_never_treats_the_mutated_source_marker_as_final(
        self,
    ) -> None:
        base_resume = re.search(
            r"(?ms)^source_manifest_merge_resume_ready\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(base_resume)
        base_body = base_resume.group("body")
        source_resume = re.search(
            r"(?ms)^community_source_merge_resume_ready\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(source_resume)
        source_body = source_resume.group("body")
        for expected in (
            "-m fireviewer_sdg.source_manifest_merge",
            '--curated-manifest "${CURATED_ASSET_MANIFEST}"',
            '--official-manifest "${OFFICIAL_ASSET_MANIFEST}"',
            '--output-manifest "${MERGED_SOURCE_MANIFEST}"',
            '--receipt "${SOURCE_MERGE_RECEIPT}"',
            '--curated-bundle-root "${ASSET_BUNDLE_ROOT}"',
            '--curated-bundle-sha256 "${ASSET_BUNDLE_SHA256}"',
            "--verify-only",
        ):
            self.assertIn(expected, base_body)
            self.assertIn(expected, source_body)
        self.assertNotIn("--require-community", base_body)
        self.assertIn("--require-community", source_body)

        materialize = re.search(
            r"(?ms)^materialize_assets\(\) \{\n(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(materialize)
        body = materialize.group("body")
        final_manifest = body.index(
            '[[ -f "${CAMPAIGN_ASSET_MANIFEST}" ]]'
        )
        final_lod = body.index(
            '[[ -f "${ASSET_BUNDLE_NATIVE_LOD_RECEIPT}" ]]',
            final_manifest,
        )
        final_pbr = body.index(
            '[[ -f "${ASSET_BUNDLE_NATIVE_PBR_RECEIPT}" ]]',
            final_lod,
        )
        final_materialized = body.index(
            '[[ -f "${ASSET_RECEIPT}" ]]',
            final_pbr,
        )
        verified_binding = body.index(
            "(bind_campaign_asset_bundle)",
            final_materialized,
        )
        accepted_binding = body.index(
            "bind_campaign_asset_bundle",
            verified_binding + 1,
        )
        reuse_return = body.index("return 0", accepted_binding)
        source_resume_call = body.index(
            "community_source_merge_resume_ready",
            reuse_return,
        )
        base_resume_call = body.index(
            "source_manifest_merge_resume_ready",
            source_resume_call,
        )
        curated_install = body.index(
            "install_curated_asset_bundle",
            base_resume_call,
        )
        official = body.index(
            "provision_official_nvidia_source_manifest",
            curated_install,
        )
        merge = body.index(
            "merge_curated_and_official_sources",
            official,
        )
        community_revalidation = body.index(
            "install_community_building_sources",
            merge,
        )
        post_install_gate = body.index(
            "community_source_merge_resume_ready",
            community_revalidation,
        )
        ground_materials = body.index(
            "ground_material_bundle",
            post_install_gate,
        )
        self.assertLess(final_manifest, final_lod)
        self.assertLess(final_lod, final_pbr)
        self.assertLess(final_pbr, final_materialized)
        self.assertLess(final_materialized, verified_binding)
        self.assertLess(verified_binding, accepted_binding)
        self.assertLess(accepted_binding, reuse_return)
        self.assertLess(source_resume_call, base_resume_call)
        self.assertLess(base_resume_call, curated_install)
        self.assertLess(curated_install, official)
        self.assertLess(official, merge)
        self.assertLess(curated_install, community_revalidation)
        self.assertLess(community_revalidation, post_install_gate)
        self.assertLess(post_install_gate, ground_materials)
        self.assertIn(
            "post-install community source merge verification failed",
            body[community_revalidation:ground_materials],
        )
        self.assertNotIn(
            "community_source_merge_resume_ready",
            body[final_manifest:reuse_return],
        )

    def test_source_resume_predicate_executes_and_rejects_archive_drift(
        self,
    ) -> None:
        self.assertIn(
            'SOURCE_MERGE_RECEIPT="${CONTRACT_ROOT}/'
            'source-assets-merged-${ASSET_BUNDLE_SHA256}.json"',
            self.setup,
        )
        configure = re.search(
            r"(?ms)^configure_asset_bundle_contract\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(configure)
        configure_body = configure.group("body")
        self.assertIn(
            'CURATED_ASSET_MANIFEST="${ASSET_BUNDLE_ROOT}/${manifest_relative}"',
            configure_body,
        )
        self.assertIn(
            'MERGED_SOURCE_MANIFEST="$(dirname -- '
            '"${CURATED_ASSET_MANIFEST}")/merged-source-v3.json"',
            configure_body,
        )
        self.assertIn(
            'SOURCE_MERGE_RECEIPT="${CONTRACT_ROOT}/'
            'source-assets-merged-${ASSET_BUNDLE_SHA256}.json"',
            configure_body,
        )
        self.assertIn(
            'ASSET_MANIFEST="${CURATED_ASSET_MANIFEST}"',
            configure_body,
        )
        selector = re.search(
            r"(?ms)^select_merged_source_manifest\(\) \{\n"
            r"(?P<body>.*?)^\}$",
            self.setup,
        )
        self.assertIsNotNone(selector)
        self.assertIn(
            'ASSET_MANIFEST="${MERGED_SOURCE_MANIFEST}"',
            selector.group("body"),
        )

    def test_materialized_receipt_verifier_executes_and_rejects_stale_data(
        self,
    ) -> None:
        match = re.search(
            r"(?ms)^verify_materialized_asset_receipt\(\) \{\n"
            r".*?\"\$\{ISAAC_PYTHON\}\" -c \\\n"
            r"        '(?P<program>.*?)' \\\n"
            r'        "\$\{ASSET_RECEIPT\}"',
            self.setup,
        )
        self.assertIsNotNone(match)
        program = match.group("program")
        current = {
            "schema_version": 3,
            "validated_at": "recalculated",
            "state": "ASSETS_LOCKED",
            "profile": "photoreal",
            "manifest": "input/campaign-assets/manifest-v3.json",
            "manifest_sha256": "1" * 64,
            "materialization_mode": "materialized",
            "asset_content_sha256": "2" * 64,
            "family_counts": {"buildings": {"annex": 1}},
            "vegetation_assets": 0,
            "building_assets": 1,
            "asset_count": 1,
            "library_policy": {"mode": "locked"},
            "assets": [
                {
                    "role": "buildings.annex[0]",
                    "asset_id": "asset-1",
                    "family": "buildings.annex",
                }
            ],
        }
        stored = {
            **current,
            "validated_at": "original",
            "usd_quality": {
                "validator": "fireviewer_native_usd_photoreal_quality_v2",
                "validated_assets": 1,
                "family_counts": current["family_counts"],
                "assets": [
                    {
                        "role": "buildings.annex[0]",
                        "asset_id": "asset-1",
                        "family": "buildings.annex",
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_package = root / "python" / "fireviewer_sdg"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text("", encoding="utf-8")
            (fake_package / "omniverse_pod.py").write_text(
                "CURRENT = "
                + repr(current)
                + "\n"
                + "def validate_materialized_assets(*, manifest_path, volume_root):\n"
                + "    return CURRENT\n",
                encoding="utf-8",
            )
            receipt = root / "assets-materialized.json"
            manifest = root / "manifest-v3.json"
            manifest.write_text("{}", encoding="utf-8")
            receipt.write_text(json.dumps(stored), encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root / "python")
            command = [
                sys.executable,
                "-c",
                program,
                str(receipt),
                str(manifest),
                str(root),
            ]
            accepted = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn(
                "MATERIALIZED_ASSET_RECEIPT_CURRENT",
                accepted.stdout,
            )

            stored["manifest_sha256"] = "f" * 64
            receipt.write_text(json.dumps(stored), encoding="utf-8")
            rejected = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_geospatial_phase_never_launches_editor_or_simulation(self) -> None:
        match = re.search(
            r"(?ms)^    geospatial\)\n(?P<body>.*?)^        ;;$",
            self.setup,
        )
        self.assertIsNotNone(match)
        body = match.group("body")
        self.assertIn("hardware_preflight", body)
        self.assertIn("ensure_geospatial_runtime", body)
        self.assertNotIn("build_editor", body)
        self.assertNotIn("build_pilot_scene", body)
        self.assertNotIn("simulation", body.casefold())


if __name__ == "__main__":
    unittest.main()
