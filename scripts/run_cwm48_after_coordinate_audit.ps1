param(
    [ValidateRange(1, 2147483647)][int]$GateDriverId = 33492,
    [switch]$EvidenceCheckOnly
)

# Operational queue only. Native Python entries remain the scientific authority.
# No polling loop, budget overrides, checkpoint resume, holdout or deployment.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$script:CwmRoot = Split-Path -Parent $PSScriptRoot
$script:CwmPython = 'D:\miniconda3\envs\uav-encirclement-gpu\python.exe'
$script:CwmData = 'D:\uav-capture\cwm34-data-20261011'
$script:CwmAudit = 'D:\uav-capture\cwm34-audit-20261011'
$script:CwmPrimary = 'D:\uav-capture\cwm48-p-20261012'
$script:CwmRepeated = 'D:\uav-capture\cwm48-r-20261012'
$script:CwmQueue = 'D:\uav-capture\cwm48-queue-20261012'

function Assert-CwmBool($Value, [bool]$Expected, [string]$Label) {
    if ($Value -isnot [bool] -or $Value -ne $Expected) {
        throw "Exact boolean required: $Label"
    }
}

function Assert-CwmInteger($Value, [string]$Label) {
    if ($Value -isnot [int] -and $Value -isnot [long]) {
        throw "Exact integer required: $Label"
    }
}

function Assert-CwmAuditHeaders($DataReport, $AuditReport, [string]$DataSummarySHA) {
    if ($DataReport.status -ne 'complete_fresh_geometry_prefix_collection_pending_independent_audit' -or
        @($DataReport.episodes).Count -ne 512 -or
        $AuditReport.status -ne 'complete_independent_fresh_geometry_prefix_public_branch_cost_audit_passed' -or
        $AuditReport.data_summary_sha256 -cne $DataSummarySHA) {
        throw 'Complete collection and independently bound native audit required'
    }
    Assert-CwmBool $DataReport.data_gate_passed $true 'collection data gate'
    Assert-CwmBool $AuditReport.data_gate_passed $true 'audit data gate'
    Assert-CwmBool $AuditReport.training_authorized_by_this_report $true 'native training permission'
    foreach ($key in @('enhanced_control_enabled', 'new_model_trained', 'holdout_used')) {
        Assert-CwmBool $AuditReport.$key $false $key
    }
    foreach ($value in @($DataReport.calls, $AuditReport.supported_calls, $AuditReport.checked_calls)) {
        Assert-CwmInteger $value 'native call count'
        if ($value -lt 1) { throw 'Positive complete native call support required' }
    }
    if ($AuditReport.supported_calls -ne $DataReport.calls -or
        $AuditReport.checked_calls -lt $AuditReport.supported_calls) {
        throw 'Native audited support does not cover the collection'
    }
}

function Assert-CwmTrainingHeader($Report) {
    if ($Report.status -ne 'complete_fixed_streamed_nine_model_training_pending_repeat_reload' -or
        @($Report.models).Count -ne 9) {
        throw 'Complete nine-model final80 result required before the next run'
    }
    foreach ($key in @('enhanced_control_enabled', 'holdout_used', 'prior_gate_overridden',
        'common_motion_in_response_optimizer', 'two_complete_runs_verified', 'independent_full_reload_verified')) {
        Assert-CwmBool $Report.$key $false $key
    }
    Assert-CwmBool $Report.primary_research_eligible $Report.primary_research_eligible 'eligibility must remain boolean'
    $expected = @{}
    foreach ($model in @('cv_motion_only', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2')) {
        foreach ($seed in @(1034101, 1034102, 1034103)) { $expected["$model/$seed"] = $true }
    }
    foreach ($row in $Report.models) {
        Assert-CwmInteger $row.seed 'model seed'
        $identity = "$($row.configuration)/$($row.seed)"
        if (-not $expected.ContainsKey($identity)) { throw 'Missing, duplicate or switched fixed model identity' }
        $expected.Remove($identity)
    }
    if ($expected.Count -ne 0) { throw 'All fixed model/seed combinations required' }
}

function Get-CwmAuditedHeaders {
    $dataSummary = Join-Path $script:CwmData 'summary.json'
    $auditSummary = Join-Path $script:CwmAudit 'summary.json'
    $dataSHA = (Get-FileHash -LiteralPath $dataSummary -Algorithm SHA256).Hash.ToLowerInvariant()
    $dataReport = Get-Content -LiteralPath $dataSummary -Raw | ConvertFrom-Json
    $auditReport = Get-Content -LiteralPath $auditSummary -Raw | ConvertFrom-Json
    Assert-CwmAuditHeaders $dataReport $auditReport $dataSHA
    foreach ($pair in @(
        @('checked_calls.json', 'checked_calls_sha256'),
        @('full_label_cost_checks.json', 'full_label_cost_checks_sha256'),
        @('audited_data_manifest.json', 'audited_data_manifest_sha256'))) {
        $fileSHA = (Get-FileHash -LiteralPath (Join-Path $script:CwmAudit $pair[0]) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($fileSHA -cne $auditReport.($pair[1])) { throw 'Native audit inventory identity differs' }
    }
    # This is only a header/manifest gate; V35/V36 still check ALL native evidence.
    return @{ data_summary_sha256 = $dataSHA;
        audit_summary_sha256 = (Get-FileHash -LiteralPath $auditSummary -Algorithm SHA256).Hash.ToLowerInvariant() }
}

function Assert-CwmResources {
    $memory = Get-CimInstance Win32_OperatingSystem
    $disk = Get-PSDrive -Name D
    if ($memory.FreePhysicalMemory -lt 1048576 -or $disk.Free -lt 25GB) {
        throw 'Resource deferral: require at least1GiB free RAM and25GiB disk before a new stage; do not stop other jobs'
    }
}

function Invoke-CwmChild([string[]]$ChildArguments) {
    & $script:CwmPython @ChildArguments
    if ($LASTEXITCODE -ne 0) { throw 'Native child failed; preserve all output and do not resume/restart' }
}

function Wait-CwmDriverCompletion([System.Diagnostics.Process]$NativeProcess) {
    $NativeProcess.WaitForExit()  # OS process event; no periodic poll or busy loop.
    if ($NativeProcess.ExitCode -ne 0) { throw 'Original collection/audit driver failed; no training permitted' }
}

function Set-CwmQueuePhase([string]$Phase, [hashtable]$Details = @{}) {
    $record = @{ phase = $Phase; time = (Get-Date -Format o); queue_pid = $PID;
        holdout_used = $false; enhanced_control_enabled = $false; details = $Details }
    $json = $record | ConvertTo-Json -Depth 6
    # Generated operational state in this queue's exclusively owned directory.
    $json | Set-Content -LiteralPath (Join-Path $script:CwmQueue 'state.json') -Encoding utf8NoBOM
    Write-Output $json
}

function Invoke-Cwm48Queue([int]$DriverId, [bool]$CheckOnly) {
    if ($CheckOnly) {
        $binding = Get-CwmAuditedHeaders
        Write-Output (@{ status = 'complete_headers_only_not_native_training_or_full_evidence_validation';
            binding = $binding; training_entry_started = $false } | ConvertTo-Json)
        return
    }
    if ((Get-Item -LiteralPath $script:CwmRoot).FullName -ne 'D:\uav-capture\causal-world-model-20261009') {
        throw 'Use the isolated research worktree, never the original checkout'
    }
    foreach ($path in @($script:CwmQueue, $script:CwmPrimary, $script:CwmRepeated)) {
        if (Test-Path -LiteralPath $path) { throw "Exclusive new operational/scientific output required: $path" }
    }
    $driver = Get-CimInstance Win32_Process -Filter "ProcessId = $DriverId"
    if ($null -eq $driver -or $driver.Name -notin @('python.exe')) {
        throw 'Existing native collection/audit DRIVER must be live before attaching the queue'
    }
    $command = $driver.CommandLine.Replace('/', '\')
    foreach ($fragment in @('experiments\cwm_v47\audit_native.py', $script:CwmData, $script:CwmAudit,
        'D:\uav-capture\cwm47-native-audit-20261012')) {
        if (-not $command.Contains($fragment)) { throw 'Refuse an unrelated/reused driver PID' }
    }
    $gateProcess = Get-Process -Id $DriverId
    # Hold the actual handle while the PID is known to identify the native driver.
    $gateHandle = $gateProcess.Handle
    if ([Math]::Abs(($gateProcess.StartTime.ToUniversalTime() - $driver.CreationDate.ToUniversalTime()).TotalSeconds) -gt 1) {
        $gateProcess.Dispose()
        throw 'Driver identity changed before handle attachment'
    }
    $transcribing = $false
    try {
        # New-Item without Force atomically rejects another queue with the same outputs.
        New-Item -ItemType Directory -Path $script:CwmQueue | Out-Null
        Start-Transcript -LiteralPath (Join-Path $script:CwmQueue 'queue.log') -NoClobber | Out-Null
        $transcribing = $true
        Set-CwmQueuePhase 'waiting_existing_native_audit_driver_not_training' @{ driver_id = $DriverId;
            driver_start_time = $gateProcess.StartTime.ToString('o'); driver_command = $driver.CommandLine }
        Wait-CwmDriverCompletion $gateProcess
        $diagnosticRoot = 'D:\uav-capture\cwm47-native-audit-20261012'
        $diagnostic = Get-Content -LiteralPath (Join-Path $diagnosticRoot 'summary.json') -Raw | ConvertFrom-Json
        if ($diagnostic.status -ne 'complete_native_coordinate_diagnostic_not_training_qualification' -or $diagnostic.optimizer_run -ne $false) { throw 'Complete new audit proposal evidence required' }
        $diagnosticBinding = Get-Content -LiteralPath (Join-Path $diagnosticRoot 'diagnostic_contract.json') -Raw | ConvertFrom-Json
        foreach ($source in $diagnosticBinding.sources.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath (Join-Path $script:CwmRoot $source.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $source.Value) { throw 'Diagnostic source changed' }
        }
        if (@($diagnostic.certificates.PSObject.Properties).Count -ne 2) { throw 'Both full reference audits required' }
        foreach ($origin in @('gru','cv')) {
            $certificate = $diagnostic.certificates.$origin
            if ($certificate.status -ne 'all_native_BASE_original_score_derivatives_checked' -or $certificate.groups -ne 64) { throw 'Incomplete original cost/derivative support' }
            $digest = (Get-FileHash -LiteralPath (Join-Path $diagnosticRoot $certificate.audit_path) -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($digest -cne $certificate.audit_sha256) { throw 'Diagnostic audit evidence changed' }
        }
        $binding = Get-CwmAuditedHeaders
        if ($diagnosticBinding.data_summary_sha256 -cne $binding.data_summary_sha256 -or $diagnosticBinding.audit_summary_sha256 -cne $binding.audit_summary_sha256) { throw 'Diagnostic input binding changed' }
        Set-Location -LiteralPath $script:CwmRoot
        foreach ($entry in @(@{ name = 'primary'; path = $script:CwmPrimary },
            @{ name = 'repeated'; path = $script:CwmRepeated })) {
            Assert-CwmResources
            if (Test-Path -LiteralPath $entry.path) { throw 'Another worker created this scientific output; no duplicate run' }
            $currentBinding = Get-CwmAuditedHeaders
            if ($binding.data_summary_sha256 -cne $currentBinding.data_summary_sha256 -or
                $binding.audit_summary_sha256 -cne $currentBinding.audit_summary_sha256) {
                throw 'Original audited inputs changed between stages'
            }
            $head = (& git rev-parse HEAD).Trim()
            if ($LASTEXITCODE -ne 0) { throw 'Git HEAD unavailable' }
            Invoke-CwmChild @('scripts/publish_exact_github_api.py', '--expected-remote-head', $head, '--authenticated-read')
            Set-CwmQueuePhase ($entry.name + '_training_entry_running_not_optimizer_evidence') $binding
            Invoke-CwmChild @('experiments/cwm_v48/train_streamed.py', '--data', $script:CwmData,
                '--audit', $script:CwmAudit, '--output', $entry.path)
            $summary = Get-Content -LiteralPath (Join-Path $entry.path 'summary.json') -Raw | ConvertFrom-Json
            Assert-CwmTrainingHeader $summary
            Set-CwmQueuePhase ($entry.name + '_nine_models_complete_pending_independent_reload') @{
                primary_research_eligible = $summary.primary_research_eligible; output = $entry.path }
            # A failed efficacy gate is retained; it never selects a different model,
            # cancels the preregistered repeat, or permits active control.
        }
        Set-CwmQueuePhase 'both_training_runs_complete_pending_full_independent_reload' $binding
    } catch {
        if ($transcribing) { Set-CwmQueuePhase 'queue_stopped_preserve_evidence_no_restart' @{
            reason = $_.Exception.Message; scientific_qualification_claimed = $false } }
        throw
    } finally {
        if ($transcribing) { Stop-Transcript | Out-Null }
        $gateProcess.Dispose()
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Invoke-Cwm48Queue $GateDriverId $EvidenceCheckOnly.IsPresent
}
