[CmdletBinding()]
param(
    [string]$RuntimeRoot = "D:\FVS",
    [int]$Port = 8000,
    [int]$ReadyTimeoutSeconds = 2100
)

$ErrorActionPreference = "Stop"

$serviceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $RuntimeRoot ".venv\Scripts\python.exe"
$volumeRoot = Join-Path $RuntimeRoot "workspace\fireviewer-sdg"
$configRoot = Join-Path $RuntimeRoot "config"
$logRoot = Join-Path $RuntimeRoot "logs"
$tokenPath = Join-Path $configRoot "console-token.txt"
$campaign = Join-Path $serviceRoot "campaigns\fireviewer-new-synthetic-cases-local-720p-v1.json"
$manifest = Join-Path $serviceRoot "provision-manifest.json"

foreach ($required in @($python, $campaign, $manifest)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required local runtime file is absent: $required"
    }
}
foreach ($directory in @($configRoot, $logRoot, $volumeRoot)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

if (-not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) {
    $tokenBytes = [byte[]]::new(48)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($tokenBytes)
    $token = [Convert]::ToBase64String($tokenBytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
    Set-Content -LiteralPath $tokenPath -Value $token -Encoding ascii -NoNewline
}
$token = (Get-Content -LiteralPath $tokenPath -Raw).Trim()
if ($token.Length -lt 32) {
    throw "The local console token is invalid; remove $tokenPath and rerun setup."
}

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port $Port already has a listener; refusing to start a duplicate service."
}

$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:ACCEPT_EULA = "Y"
$env:PRIVACY_CONSENT = "Y"
$env:OMNI_CONFIG_PATH = $configRoot
$env:PYTHONPATH = Join-Path $serviceRoot "src"
$env:FW_SDG_VOLUME_ROOT = $volumeRoot
$env:FW_SDG_PROVISION_MANIFEST = $manifest
$env:FW_SDG_CAMPAIGN = $campaign
$env:FW_SDG_PREPARE_IGN_CATALOG = "1"
$env:FW_SDG_RUN_MODE = "service"
$env:FW_SDG_PORT = [string]$Port
$env:FW_SDG_AUTH_TOKEN = $token

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $logRoot "service-$timestamp.stdout.log"
$stderrPath = Join-Path $logRoot "service-$timestamp.stderr.log"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList @("-m", "fireviewer_sdg.bootstrap") `
    -WorkingDirectory $RuntimeRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$metadata = @{
    pid = $process.Id
    port = $Port
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    campaign = $campaign
    resolution = @(1280, 720)
    stdout = $stdoutPath
    stderr = $stderrPath
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RuntimeRoot "service.json") -Encoding utf8

$deadline = (Get-Date).AddSeconds($ReadyTimeoutSeconds)
$healthUri = "http://127.0.0.1:$Port/healthz"
do {
    if ($process.HasExited) {
        throw "FireViewer SDG exited with code $($process.ExitCode). Inspect $stderrPath"
    }
    try {
        $health = Invoke-RestMethod -Uri $healthUri -TimeoutSec 3
        if ($health.status -eq "alive") {
            [pscustomobject]@{
                pid = $process.Id
                state = "ready"
                console = "http://127.0.0.1:$Port/console"
                token_file = $tokenPath
                stdout = $stdoutPath
                stderr = $stderrPath
            }
            exit 0
        }
    }
    catch {
        Start-Sleep -Seconds 2
    }
} while ((Get-Date) -lt $deadline)

throw "FireViewer SDG did not become healthy within $ReadyTimeoutSeconds seconds. Inspect $stderrPath"
