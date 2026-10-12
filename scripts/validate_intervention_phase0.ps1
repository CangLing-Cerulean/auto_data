$ErrorActionPreference = 'Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskStage=Join-Path $taskRoot 'research_runs/agent_research/intervention_diagnosis/20261012_phase0'
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
$taskChecks=@()
function Assert-Phase($Condition, $Label) {
    if (-not $Condition) { throw "Phase-0 validation failed: $Label" }
    $script:taskChecks += $Label
}
$taskRecords=(Get-Content -LiteralPath "$taskStage/historical_response_collection.json" -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable).records
Assert-Phase ($taskRecords.Count -eq 14) '14 historical records'
Assert-Phase (($taskRecords.record_id | Sort-Object -Unique).Count -eq 14) 'unique record identities'
foreach ($taskRecord in $taskRecords) {
    $taskHash=(Get-FileHash -LiteralPath (Join-Path $taskRoot $taskRecord.source_ref) -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-Phase ($taskHash -eq $taskRecord.source_sha256) "source receipt $($taskRecord.record_id)"
    foreach ($taskRegion in @('global','target')) {
        foreach ($taskMetric in @('NMSE','NMAE')) {
            $taskValue=$taskRecord.response[$taskRegion]
            Assert-Phase ([double]::IsFinite($taskValue.delta[$taskMetric]) -and
                [Math]::Abs($taskValue.delta[$taskMetric]-($taskValue.parent[$taskMetric]-$taskValue.child[$taskMetric])) -lt 1e-14) "delta $($taskRecord.record_id) $taskRegion $taskMetric"
        }
    }
}
$taskR2=@($taskRecords | Where-Object action -eq 'source_sign_flip')
Assert-Phase ($taskR2.Count -eq 3 -and @($taskR2 | Where-Object {$_.response.global.delta.NMSE -ge 0}).Count -eq 0) 'all three R2 increments harmful in overall NMSE'
Assert-Phase (@($taskR2 | Where-Object {$null -ne $_.prestate -or $null -ne $_.prospective_prediction}).Count -eq 0) 'R2 missing prestate remains null'
$taskTasks=(Get-Content -LiteralPath "$taskStage/tasks.json" -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable).tasks
Assert-Phase ($taskTasks.Count -eq 3) 'three separate task contracts'
foreach ($taskTask in $taskTasks) {
    Assert-Phase ((Get-FileHash -LiteralPath "$taskRoot/data/$($taskTask.dataset_id)/manifest.json" -Algorithm SHA256).Hash.ToLowerInvariant() -eq $taskTask.manifest_sha256) "manifest $($taskTask.dataset_id)"
    Assert-Phase ($taskTask.parameters -eq 528*$taskTask.channels -and $taskTask.fixed_training.steps -eq 3000) "fixed architecture $($taskTask.dataset_id)"
    Assert-Phase ($taskTask.test_access -eq 'forbidden_except_integrity') "test isolation $($taskTask.dataset_id)"
}
foreach ($taskFile in Get-ChildItem -LiteralPath $taskStage -File -Filter '*.json') {
    $null=Get-Content -LiteralPath $taskFile.FullName -Raw -Encoding utf8 | ConvertFrom-Json
}
foreach ($taskScript in @('scripts/prepare_intervention_phase0.ps1','scripts/validate_intervention_phase0.ps1')) {
    $taskTokens=$null;$taskErrors=$null
    $null=[Management.Automation.Language.Parser]::ParseFile((Join-Path $taskRoot $taskScript),[ref]$taskTokens,[ref]$taskErrors)
    Assert-Phase ($taskErrors.Count -eq 0) "PowerShell syntax $taskScript"
}
$taskVerify=Get-Content -LiteralPath "$taskStage/dataset_verification.json" -Raw -Encoding utf8 | ConvertFrom-Json
Assert-Phase ($taskVerify.status -eq 'PASS' -and $taskVerify.metropt3_source_reconstruction -eq 'PASS') 'all base datasets integrity verified'
Assert-Phase (-not (Test-Path -LiteralPath "$taskStage/paired_evaluation/receipt.json")) 'no unexecuted tensor result represented as complete'
$taskWatch.Stop()
$taskReceipt=[ordered]@{status='passed_local_static_and_historical_aggregate_checks';checks=$taskChecks;count=$taskChecks.Count;
    python_runtime_tests='NOT_RUN';python_syntax_compilation='NOT_RUN';paired_checkpoint_evaluation='NOT_RUN';
    no_new_sensor_records=$true;new_trainings=0;local_process_seconds=$taskWatch.Elapsed.TotalSeconds;
    limitations='This validates JSON, evidence identities, arithmetic and PowerShell syntax; it does not execute Python prototypes.'}
[IO.File]::WriteAllText((Join-Path $taskStage 'local_validation.json'),(($taskReceipt | ConvertTo-Json -Depth 8).Replace("`r`n","`n")+"`n"),[Text.UTF8Encoding]::new($false))
Write-Output "PASS: $($taskChecks.Count) local static/evidence checks; Python runtime and paired evaluation NOT_RUN."
