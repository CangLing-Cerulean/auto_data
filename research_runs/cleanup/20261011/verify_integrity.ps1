$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
function Read-Json($relative) { Get-Content -LiteralPath (Join-Path $root $relative) -Raw | ConvertFrom-Json }
function Assert-Hash($relative, $expected) {
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required asset: $relative" }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Frozen asset mismatch: $relative" }
}
Assert-Hash 'data/metropt3_v1/manifest.json' '672c0ecf55bc94a52abc6b256c0bfe7a4da02140e57dd0527375bf859a22c087'
Assert-Hash 'research_runs/protocol.json' '573ee4e87ef442b7140afb26f4befe80a8de1a43fec3b5c1fe6b878e527f8afe'
Assert-Hash 'scripts/run_research.py' '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052'
$manifest = Read-Json 'data/metropt3_v1/manifest.json'
foreach ($split in $manifest.splits) { Assert-Hash ('data/metropt3_v1/'+$split.file) $split.sha256 }
$freeze = Read-Json 'research_runs/model_exploration/freeze.json'
foreach ($model in $freeze.data_and_models.models) {
    Assert-Hash ('research_runs/results/'+$model.run_id+'/best.pt') $model.checkpoint_sha256
}
$receipts = Read-Json 'research_runs/model_exploration/download_receipt.json'
foreach ($entry in $receipts.PSObject.Properties) { Assert-Hash $entry.Name $entry.Value }
$v1 = Read-Json 'research_runs/agent_research/experience_library.json'
foreach ($record in $v1.records) { Assert-Hash $record.source.path $record.source.sha256 }
$memory = Read-Json 'research_runs/agent_research/memory/v2/library.json'
foreach ($record in $memory.records) {
    if ($record.record_type -eq 'measured_run') {
        Assert-Hash ('research_runs/agent_research/memory/v2/'+$record.source.storage_path) $record.source.sha256
    } else {
        # Conditional summaries use the original v1 report, not a memory-directory file.
        Assert-Hash $record.source.path $record.source.sha256
    }
}
$started = Read-Json 'research_runs/agent_research/phase4/20261010_signal_v2/batch_started.json'
$codeCount = 0
foreach ($entry in $started.code.PSObject.Properties) {
    $prefix = '/tmp/auto_data_signal_project_20261010_v2/'
    if (-not $entry.Name.StartsWith($prefix)) { throw "Unexpected frozen code root: $($entry.Name)" }
    Assert-Hash $entry.Name.Substring($prefix.Length) $entry.Value
    $codeCount++
}
Assert-Hash 'research_runs/agent_research/phase4/20261010_signal_v2/protocol.json' $started.protocol_sha256
Assert-Hash 'research_runs/agent_research/phase4/20261010_signal_v2/PROTOCOL.md' $started.document_sha256
$plan = Read-Json 'research_runs/cleanup/20261011/private_plan.json'
foreach ($relative in $plan.mutable_files) {
    $saved = Join-Path $plan.archive ('metadata_before_cleanup/'+$relative)
    $expected = ($plan.files | Where-Object path -eq $relative).sha256
    if ((Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
        throw "Original metadata backup mismatch: $relative"
    }
}
# Check only rewritten entry-point links; historical frozen reports keep their original text.
$linkCount=0
foreach ($relative in @('README.md','research_runs/HANDOFF.md','research_runs/OVERVIEW.md',
    'research_runs/NEXT_STEPS.md','research_runs/agent_research/README.md','research_runs/cleanup/20261011/REPORT.md')) {
    $path=Join-Path $root $relative
    $content=Get-Content -LiteralPath $path -Raw
    foreach ($match in [regex]::Matches($content,'\]\(([^)]+)\)')) {
        $target=$match.Groups[1].Value
        if ($target -match '^(https?://|#)') { continue }
        $resolved=Join-Path (Split-Path $path) $target
        if (-not (Test-Path -LiteralPath $resolved)) { throw "Entry link missing: $relative -> $target" }
        $linkCount++
    }
}
$state=Read-Json 'research_runs/state.json'; $progress=Read-Json 'research_runs/agent_research/progress.json'
if ($state.historical_completed_run_ids.Count -ne 134 -or $memory.records.Count -ne 144) { throw 'Historical coverage changed' }
if ($state.total_training_runs_completed -ne 200 -or $progress.total_completed_trainings -ne 200 -or $progress.api_calls -ne 1079) { throw 'Cost snapshot changed' }
$result=[ordered]@{
    status='passed';verified_at=(Get-Date).ToString('o');
    base_manifest_protocol_and_core='matched_frozen_hashes';base_CSV_files_verified=$manifest.splits.Count;
    model_and_baseline_checkpoints_verified=$freeze.data_and_models.models.Count;
    model_study_receipt_files_verified=@($receipts.PSObject.Properties).Count;
    experience_original_source_records_verified=$v1.records.Count;memory_source_records_verified=$memory.records.Count;
    signal_v2_frozen_code_files_verified=$codeCount;signal_v2_frozen_protocol_and_document='matched';
    metadata_originals_verified=$plan.mutable_files.Count;current_entry_links_verified=$linkCount;
    historical_training_IDs_preserved=134;cost_snapshot_unchanged=$true;
    new_API_calls=0;new_trainings=0;server_runtime_tests='not_run_local_integrity_and_dependencies_only'
}
$json=($result | ConvertTo-Json -Depth 8).Replace("`r`n","`n")+"`n"
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'integrity_verification.json'),$json,[Text.UTF8Encoding]::new($false))
$result | ConvertTo-Json -Depth 8
exit 0
