# Synthetic operational-header/negative checks, NOT native audit or model training.
$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts/run_cwm36_after_audit.ps1')
$script:CwmQueueTestCount = 0

function Test-CwmReject([scriptblock]$Action) {
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    if (-not $rejected) { throw 'Expected fail-closed refusal' }
    $script:CwmQueueTestCount++
}

function New-CwmHeaderFixture {
    $data = [pscustomobject]@{
        status = 'complete_fresh_geometry_prefix_collection_pending_independent_audit'
        episodes = @(1..512); data_gate_passed = $true; calls = 12000
    }
    $audit = [pscustomobject]@{
        status = 'complete_independent_fresh_geometry_prefix_public_branch_cost_audit_passed'
        data_summary_sha256 = ('0' * 64); data_gate_passed = $true
        training_authorized_by_this_report = $true; enhanced_control_enabled = $false
        new_model_trained = $false; holdout_used = $false
        supported_calls = 12000; checked_calls = 12000
    }
    return @($data, $audit)
}

function New-CwmTrainingFixture {
    $rows = foreach ($model in @('cv_motion_only', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2')) {
        foreach ($seed in @(1034101, 1034102, 1034103)) {
            [pscustomobject]@{ configuration = $model; seed = $seed }
        }
    }
    return [pscustomobject]@{
        status = 'complete_fixed_streamed_nine_model_training_pending_repeat_reload'
        models = @($rows); enhanced_control_enabled = $false; holdout_used = $false
        prior_gate_overridden = $false; common_motion_in_response_optimizer = $false
        two_complete_runs_verified = $false; independent_full_reload_verified = $false
        primary_research_eligible = $false
    }
}

$fixture = New-CwmHeaderFixture
Assert-CwmAuditHeaders $fixture[0] $fixture[1] ('0' * 64)
$script:CwmQueueTestCount++
foreach ($mutation in @('partial', 'data_status', 'audit_status', 'binding', 'permission',
    'data_gate', 'audit_gate', 'control', 'trained', 'holdout', 'support', 'checked',
    'bool_count', 'float_count', 'string_permission', 'null_permission', 'zero_calls')) {
    $fixture = New-CwmHeaderFixture
    $data, $audit = $fixture
    switch ($mutation) {
        partial { $data.episodes = @(1..511) }
        data_status { $data.status = 'partial' }
        audit_status { $audit.status = 'partial' }
        binding { $audit.data_summary_sha256 = ('1' * 64) }
        permission { $audit.training_authorized_by_this_report = $false }
        data_gate { $data.data_gate_passed = $false }
        audit_gate { $audit.data_gate_passed = $false }
        control { $audit.enhanced_control_enabled = $true }
        trained { $audit.new_model_trained = $true }
        holdout { $audit.holdout_used = $true }
        support { $audit.supported_calls = 11999 }
        checked { $audit.checked_calls = 11999 }
        bool_count { $data.calls = $true }
        float_count { $data.calls = 12000.0 }
        string_permission { $audit.training_authorized_by_this_report = 'true' }
        null_permission { $audit.training_authorized_by_this_report = $null }
        zero_calls { $data.calls = 0; $audit.supported_calls = 0 }
    }
    Test-CwmReject { Assert-CwmAuditHeaders $data $audit ('0' * 64) }
}

foreach ($eligible in @($true, $false)) {
    $report = New-CwmTrainingFixture
    $report.primary_research_eligible = $eligible
    Assert-CwmTrainingHeader $report
    $script:CwmQueueTestCount++
}
foreach ($mutation in @('status', 'missing', 'duplicate', 'seed', 'bool_seed', 'model',
    'control', 'holdout', 'override', 'common', 'premature_repeat', 'premature_reload', 'string_eligibility')) {
    $report = New-CwmTrainingFixture
    switch ($mutation) {
        status { $report.status = 'partial' }
        missing { $report.models = @($report.models[0..7]) }
        duplicate { $report.models[8] = $report.models[0] }
        seed { $report.models[0].seed = 999999 }
        bool_seed { $report.models[0].seed = $true }
        model { $report.models[0].configuration = 'best_model' }
        control { $report.enhanced_control_enabled = $true }
        holdout { $report.holdout_used = $true }
        override { $report.prior_gate_overridden = $true }
        common { $report.common_motion_in_response_optimizer = $true }
        premature_repeat { $report.two_complete_runs_verified = $true }
        premature_reload { $report.independent_full_reload_verified = $true }
        string_eligibility { $report.primary_research_eligible = 'false' }
    }
    Test-CwmReject { Assert-CwmTrainingHeader $report }
}

# Actual command/runtime refusal before queue/output creation; never starts Python.
Test-CwmReject { Invoke-Cwm36Queue 2147483647 $false }
$savedData = $script:CwmData
try {
    $script:CwmData = Join-Path ([System.IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString() + '-missing-cwm-data')
    Test-CwmReject { Invoke-Cwm36Queue 2147483647 $true }
} finally { $script:CwmData = $savedData }

# Actual neutral Windows handle/exit status checks. No training, credentials or UAVs.
$shellPath = (Get-Process -Id $PID).Path
foreach ($code in @(0, 7)) {
    $child = Start-Process -FilePath $shellPath -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-Command', "Start-Sleep -Milliseconds 1500; exit $code")
    $attached = Get-Process -Id $child.Id
    $heldHandle = $attached.Handle
    try {
        if ($code -eq 0) { Wait-CwmDriverCompletion $attached; $script:CwmQueueTestCount++ }
        else { Test-CwmReject { Wait-CwmDriverCompletion $attached } }
    } finally { $attached.Dispose(); $child.Dispose() }
}

$savedPython = $script:CwmPython
try {
    $script:CwmPython = $shellPath
    Invoke-CwmChild @('-NoProfile', '-Command', 'exit 0')
    $script:CwmQueueTestCount++
    Test-CwmReject { Invoke-CwmChild @('-NoProfile', '-Command', 'exit 7') }
} finally { $script:CwmPython = $savedPython }

$parseTokens = $null; $parseErrors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $script:CwmRoot 'scripts/run_cwm36_after_audit.ps1'),
    [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Operational script syntax invalid' }
$commands = $tree.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.CommandAst] }, $true)
foreach ($forbidden in @('Start-Sleep', 'Stop-Process', 'Remove-Item', 'Restart-Computer')) {
    if (@($commands | Where-Object { $_.GetCommandName() -eq $forbidden }).Count) {
        throw 'No poll/sleep/stop/delete command allowed in this queue'
    }
}
$script:CwmQueueTestCount++
Write-Output (@{ status = 'synthetic_operational_queue_checks_passed_not_native_training';
    passed = $script:CwmQueueTestCount; scientific_training_executed = $false } | ConvertTo-Json)
exit 0
