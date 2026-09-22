[CmdletBinding()]
param(
    [ValidateSet("mappo", "ippo")]
    [string]$Algorithm = "mappo",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$Python = "python",
    [int]$Seed = 101,
    [int]$Updates = 5000,
    [int]$EpisodesPerUpdate = 8,
    [int]$PpoEpochs = 4,
    [int]$MinibatchSize = 512,
    [int]$HiddenDim = 128,
    [double]$LearningRate = 3e-4,
    [double]$Gamma = 0.99,
    [double]$GaeLambda = 0.95,
    [double]$ClipRange = 0.2,
    [double]$EntropyCoef = 0.01,
    [int]$MaxSteps = 250,
    [int]$TorchThreads = 1,
    [int]$TrainingEpisodes = 300,
    [int]$BcEpochs = 10,
    [int]$BcBatchSize = 1024,
    [int]$CheckpointIntervalUpdates = 500,
    [int]$LogInterval = 1,
    [string]$Config = "",
    [string]$EnvironmentConfig = "",
    [string]$TrainingScenes = "",
    [string]$ExpertDataset = "",
    [string]$Output = "",
    [string]$Resume = "",
    [switch]$UseCbfTrain
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Resolve-ProjectPath {
    param([string]$Value, [string]$DefaultRelative)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return (Join-Path $ProjectRoot $DefaultRelative)
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    return (Join-Path $ProjectRoot $Value)
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

$ConfigPath = Resolve-ProjectPath $Config "configs\phase86_pure_mappo_repro.yaml"
$EnvironmentConfigPath = Resolve-ProjectPath $EnvironmentConfig "configs\phase85_target_contract_repaired_environment.yaml"
$ScenesPath = Resolve-ProjectPath $TrainingScenes "results\phase86_nominal_repaired_development_calibration\scenes.jsonl"
$ExpertDatasetPath = Resolve-ProjectPath $ExpertDataset "models\phase86_fix_route_teacher_formal_contract_20260921\expert_dataset.npz"
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = "models\phase86_pure_${Algorithm}_route_nocbf_seed${Seed}_repro"
}
$OutputPath = Resolve-ProjectPath $Output $Output

foreach ($RequiredPath in @($ConfigPath, $EnvironmentConfigPath, $ScenesPath)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Required input is missing: $RequiredPath"
    }
}

if ($Device -eq "auto") {
    & $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    $Device = if ($LASTEXITCODE -eq 0) { "cuda" } else { "cpu" }
}
if ($Device -eq "cuda") {
    & $Python -c "import torch,sys; print('torch=',torch.__version__,'cuda=',torch.version.cuda,'available=',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA was requested but PyTorch cannot see a CUDA device. Install a CUDA-enabled PyTorch build and the NVIDIA driver; refusing silent CPU fallback."
    }
}

$TrainingArgs = @(
    "scripts\train_mappo_ippo_baseline.py",
    "--config", $ConfigPath,
    "--output", $OutputPath,
    "--algorithm", $Algorithm,
    "--seed", $Seed,
    "--device", $Device,
    "--updates", $Updates,
    "--episodes-per-update", $EpisodesPerUpdate,
    "--ppo-epochs", $PpoEpochs,
    "--minibatch-size", $MinibatchSize,
    "--hidden-dim", $HiddenDim,
    "--learning-rate", $LearningRate,
    "--gamma", $Gamma,
    "--gae-lambda", $GaeLambda,
    "--clip-range", $ClipRange,
    "--entropy-coef", $EntropyCoef,
    "--max-steps", $MaxSteps,
    "--torch-threads", $TorchThreads,
    "--checkpoint-interval-updates", $CheckpointIntervalUpdates,
    "--log-interval", $LogInterval,
    "--tensorboard"
)

if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $ResumePath = Resolve-ProjectPath $Resume $Resume
    if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) {
        throw "Resume checkpoint is missing: $ResumePath"
    }
    $TrainingArgs += @("--resume", $ResumePath)
} else {
    if (-not (Test-Path -LiteralPath $ExpertDatasetPath -PathType Leaf)) {
        throw "Expert dataset is missing: $ExpertDatasetPath. Pass -ExpertDataset or prepare the reproducibility asset."
    }
    $TrainingArgs += @(
        "--training-scenes", $ScenesPath,
        "--training-episodes", $TrainingEpisodes,
        "--expert-dataset", $ExpertDatasetPath,
        "--bc-epochs", $BcEpochs,
        "--bc-batch-size", $BcBatchSize
    )
}

if ($UseCbfTrain) {
    $TrainingArgs += "--use-cbf-train"
}

Write-Output ("Starting Phase86 {0}: device={1}, seed={2}, updates={3}, output={4}" -f $Algorithm, $Device, $Seed, $Updates, $OutputPath)
Write-Output ("Checkpoint interval: every {0} updates" -f $CheckpointIntervalUpdates)
Invoke-CheckedPython $TrainingArgs
