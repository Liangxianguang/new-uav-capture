[CmdletBinding()]
param(
    [string]$OutputRoot = "",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "results\reproduce_v5_seed661606_validation"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Output already exists: $OutputRoot. Choose a new -OutputRoot."
}

$Checkpoint = "models\v5_development_exact_reactive_seed661606.pt"
$EnvironmentConfig = "configs\capture_radius_pursuit_central_v4_flee.yaml"
$Protocol = "configs\central_random_mixed_obstacle_s3_v5_protocol.yaml"
$ReplayRoot = Join-Path $OutputRoot "episode0"
$ThreeDRoot = Join-Path $ReplayRoot "three_d"

function Invoke-UavPython {
    param([string[]]$Arguments)
    & conda run --no-capture-output --name uav-encirclement-gpu python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Invoke-UavPython @(
    "scripts\evaluate_random_central_mixed_obstacles.py",
    "--checkpoint", $Checkpoint,
    "--environment-config", $EnvironmentConfig,
    "--protocol", $Protocol,
    "--split", "validation",
    "--output-dir", $OutputRoot,
    "--use-cbf",
    "--recurrent-reset-interval", "1",
    "--device", $Device
)

Invoke-UavPython @(
    "scripts\render_random_capture_episode.py",
    "--checkpoint", $Checkpoint,
    "--scenes", (Join-Path $OutputRoot "scenes.jsonl"),
    "--episode-index", "0",
    "--use-cbf",
    "--recurrent-reset-interval", "1",
    "--output-dir", $ReplayRoot,
    "--device", $Device
)

Invoke-UavPython @(
    "scripts\render_3d_capture_animation.py",
    "--trajectory", (Join-Path $ReplayRoot "trajectory.npz"),
    "--result", (Join-Path $ReplayRoot "episode.json"),
    "--output-dir", $ThreeDRoot
)

Write-Output "V5 development validation and episode-0 media were written to: $OutputRoot"
