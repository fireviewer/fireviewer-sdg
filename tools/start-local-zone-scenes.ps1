[CmdletBinding()]
param(
    [string]$CatalogRoot = "D:\Dev\project\fireviewer-repositories\fireviewer-sdg\livrable_20_zones_france_omniverse\livrable_20_zones_france_omniverse",
    [string]$WorkspaceRoot = "D:\FVS\workspace\fireviewer-sdg",
    [ValidateSet("Z16", "Z10", "Z08", "Z19", "Z17", "Z18", "Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07", "Z09", "Z11", "Z12", "Z13", "Z14", "Z15", "Z20")]
    [string]$Zone = "Z16",
    [ValidateSet("preflight", "resolve", "acquire", "build", "review", "render", "qa", "archive", "cleanup")]
    [string]$Phase = "preflight",
    [string[]]$Lod0Tile = @(),
    [double]$MinimumFreeGiB = 20,
    [ValidateRange(1, 4)]
    [int]$DownloadWorkers = 3,
    [ValidateSet("full", "light")]
    [string]$SourceProfile = "light",
    [string]$Receipt = "",
    [switch]$HumanReviewed,
    [string]$ConfirmCleanup = "",
    [string]$IsaacSimRoot = "",
    [ValidateRange(1, 720)]
    [int]$BuildTimeoutMinutes = 240,
    [ValidateRange(1, 720)]
    [int]$RenderTimeoutMinutes = 240
)

$ErrorActionPreference = "Stop"
$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$catalog = (Resolve-Path -LiteralPath $CatalogRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $catalog "package_manifest.json") -PathType Leaf)) {
    throw "CatalogRoot is not the supplied 20-zone catalog: $catalog"
}
$workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot)
New-Item -ItemType Directory -Force -Path $workspace | Out-Null

function Resolve-NativeIsaacSimRoot {
    $candidates = @()
    if ($IsaacSimRoot) { $candidates += $IsaacSimRoot }
    if ($env:FW_SDG_ISAAC_SIM_ROOT) { $candidates += $env:FW_SDG_ISAAC_SIM_ROOT }
    $candidates += @(
        "C:\isaacsim",
        (Join-Path $env:LOCALAPPDATA "ov\pkg\isaac-sim-6.0.1"),
        (Join-Path $env:LOCALAPPDATA "ov\pkg\isaac-sim")
    )
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Container)) { continue }
        $root = (Resolve-Path -LiteralPath $candidate).Path
        if (
            (Test-Path -LiteralPath (Join-Path $root "python.bat") -PathType Leaf) -and
            (Test-Path -LiteralPath (Join-Path $root "isaac-sim.compatibility_check.bat") -PathType Leaf)
        ) {
            return $root
        }
    }
    throw "Isaac Sim Workstation native is required. Install Isaac Sim 6.0.1 or pass -IsaacSimRoot with python.bat and isaac-sim.compatibility_check.bat."
}

function Resolve-FireViewerUsdComposer {
    $candidates = @()
    if ($env:FW_SDG_REVIEW_EDITOR) { $candidates += $env:FW_SDG_REVIEW_EDITOR }
    $candidates += "D:\Programs\NVIDIA omni\kit-app-template\_build\windows-x86_64\release\fireviewer_usd_composer.kit.bat"
    foreach ($candidate in $candidates | Select-Object -Unique) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "FireViewer USD Composer registered through Omniverse Hub is required for the manual review launch. Set FW_SDG_REVIEW_EDITOR to its .kit.bat launcher."
}

function Invoke-NativeZoneScenePhase {
    $nativeRoot = Resolve-NativeIsaacSimRoot
    $python = Join-Path $nativeRoot "python.bat"
    $existingPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = "$serviceRoot\src"
    if ($existingPythonPath) { $env:PYTHONPATH += ";$existingPythonPath" }
    $env:FW_SDG_RUN_MODE = "zone_scenes"
    $env:FW_SDG_VOLUME_ROOT = $workspace
    $env:FW_SDG_PROVISION_MANIFEST = Join-Path $serviceRoot "provision-manifest.json"
    $env:FW_SDG_CAMPAIGN = Join-Path $serviceRoot "campaigns\fireviewer-new-synthetic-cases-v1.json"
    $env:FW_SDG_ZONE_CATALOG_ROOT = $catalog
    $env:FW_SDG_ZONE_WORKSPACE_ROOT = $workspace
    $env:FW_SDG_ZONE_ID = $Zone
    $env:FW_SDG_ZONE_PHASE = $Phase
    $env:FW_SDG_ZONE_MINIMUM_FREE_GIB = "$MinimumFreeGiB"
    $env:FW_SDG_ZONE_DOWNLOAD_WORKERS = "$DownloadWorkers"
    $env:FW_SDG_ZONE_SOURCE_PROFILE = $SourceProfile
    $env:FW_SDG_ZONE_LOD0_TILES = $Lod0Tile -join ','
    $env:FW_SDG_ZONE_BUILD_TIMEOUT = "$($BuildTimeoutMinutes * 60)"
    $env:FW_SDG_ZONE_RENDER_TIMEOUT = "$($RenderTimeoutMinutes * 60)"
    $env:FW_SDG_PREPARE_IGN_CATALOG = "0"
    $env:FW_SDG_ISAAC_COMPATIBILITY_CHECKER = Join-Path $nativeRoot "isaac-sim.compatibility_check.bat"
    $env:FW_SDG_GPU_PREFLIGHT_RECEIPT = Join-Path $workspace "zone-scenes\$Zone\runtime-preflight.json"
    $env:FW_SDG_REVIEW_EDITOR = Resolve-FireViewerUsdComposer
    if ($Receipt) {
        $receiptPath = (Resolve-Path -LiteralPath $Receipt).Path
        if (-not $receiptPath.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) { throw "Receipt must remain inside WorkspaceRoot." }
        $env:FW_SDG_ZONE_RECEIPT = $receiptPath
    }
    if ($HumanReviewed) { $env:FW_SDG_ZONE_HUMAN_REVIEWED = "1" }
    if ($ConfirmCleanup) { $env:FW_SDG_ZONE_CONFIRM_CLEANUP = $ConfirmCleanup }
    & $python -m fireviewer_sdg.bootstrap
    if ($LASTEXITCODE -ne 0) { throw "Native zone scene phase $Phase failed for $Zone." }
}
Invoke-NativeZoneScenePhase
