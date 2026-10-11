param([ValidateSet('Plan','Apply','Verify')][string]$Mode = 'Plan')
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..')).TrimEnd('\')
$receiptDir = $PSScriptRoot
$planPath = Join-Path $receiptDir 'private_plan.json'
$historyParent = 'C:\Users\31227\Desktop\auto_data_history'
$mutable = @('README.md','research_runs/HANDOFF.md','research_runs/OVERVIEW.md',
    'research_runs/NEXT_STEPS.md','research_runs/state.json',
    'research_runs/agent_research/README.md','research_runs/agent_research/progress.json')
function Write-Json($path, $value) {
    $json = ($value | ConvertTo-Json -Depth 40).Replace("`r`n", "`n") + "`n"
    [IO.File]::WriteAllText($path, $json, [Text.UTF8Encoding]::new($false))
}
function Check-Path($path, $parent) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($parent.TrimEnd('\')+'\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes allowed directory: $full"
    }
    return $full
}
function Hash($path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
if ($Mode -eq 'Plan') {
    if (Test-Path -LiteralPath $planPath) { throw 'Existing cleanup plan will not be overwritten' }
    $keepExact = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    @('.gitattributes','.gitignore','AGENTS.md','README.md',
      'research_runs/protocol.json','research_runs/PROTOCOL.md','research_runs/GITHUB_SYNC.md',
      'research_runs/selection/protocol.json',
      'source/metropt3/LICENSE_AND_CITATION.md','source/metropt3/dataset_manifest.yaml',
      'source/metropt3/sensor_dictionary.csv') + $mutable | ForEach-Object { [void]$keepExact.Add($_) }
    $library = Get-Content -LiteralPath (Join-Path $root 'research_runs/agent_research/experience_library.json') -Raw | ConvertFrom-Json
    foreach ($source in $library.records.source.path) { [void]$keepExact.Add($source) }
    foreach ($name in @('.gitignore','EXECUTION_CONTRACT.md','experience_library.json','experience_specs.json','ROADMAP.md')) {
        [void]$keepExact.Add('research_runs/agent_research/'+$name)
    }
    foreach ($name in @('REPORT.md','bank_review.json','bank_summary.json','closure_summary.json','cost_audit.json','download_verification.json')) {
        [void]$keepExact.Add('research_runs/agent_research/capability_acceptance/20261010_v3/'+$name)
    }
    $prefixes = @('data/','online_ts/','tests/','research_runs/cleanup/',
        'research_runs/direct_diagnostics/','research_runs/direct_processing/',
        'research_runs/model_exploration/','research_runs/github_sync/',
        'research_runs/agent_research/memory/v2/',
        'research_runs/agent_research/capability_acceptance/20261010_v4/',
        'research_runs/agent_research/phase4/20261010_plan/',
        'research_runs/agent_research/phase4/20261010_signal/',
        'research_runs/agent_research/phase4/20261010_signal_v2/',
        'archives/capability_acceptance_20261010_v4/')
    $modelProgress = Get-Content -LiteralPath (Join-Path $root 'research_runs/model_exploration/progress.json') -Raw | ConvertFrom-Json
    $runs = @($modelProgress.rows.run_id) + @('S000_uniform_s2026','S005_uniform_s2027','S010_uniform_s2028','A009_leading_sources','A015_model_export')
    foreach ($id in $runs) { $prefixes += 'research_runs/results/'+$id+'/' }
    $scriptNames = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    @('verify_dataset.py','run_research.py','run_capability.py','run_capability_checks.py',
      'run_signal_study.py','run_signal_checks.py','check_signal_readiness.py','launch_signal_study.py',
      'prepare_signal_project.py','read_signal_status.py','read_signal_training_failures.py',
      'setup_deepseek_key.ps1','check_deepseek_api.py','publish_stage_release.ps1',
      'run_direct_processing.py','run_direct_batch.py','run_direct_mechanism_batch.py',
      'run_direct_own_ablation.py','confirm_direct_wave1.py','check_direct_processing.py',
      'audit_direct_mechanism.py','package_direct_study.py','verify_direct_download.ps1',
      'audit_feature_diagnostics.py','diagnose_feature_errors.py','inspect_constant_response.py',
      'forecast_models.py','run_model_exploration.py','run_model_study.py','export_model_study.py',
      'evaluate_frozen_models.py','package_model_study.py','summarize_model_study.py','leading_sources.py') | ForEach-Object { [void]$scriptNames.Add($_) }
    # Retain local script imports and literal script references recursively.
    $more = $true
    while ($more) {
        $more = $false
        $scan = @($scriptNames | ForEach-Object { Join-Path $root ('scripts/'+$_) } | Where-Object { Test-Path -LiteralPath $_ })
        $scan += @(Get-ChildItem -LiteralPath (Join-Path $root 'online_ts') -Recurse -File -Filter '*.py' | ForEach-Object FullName)
        foreach ($path in $scan) {
            $code = Get-Content -LiteralPath $path -Raw
            $matches = [regex]::Matches($code, '(?m)^\s*(?:from|import)\s+([A-Za-z_]\w*)')
            foreach ($match in $matches) {
                $name = $match.Groups[1].Value+'.py'
                if ((Test-Path -LiteralPath (Join-Path $root ('scripts/'+$name))) -and $scriptNames.Add($name)) { $more = $true }
            }
            foreach ($match in [regex]::Matches($code, 'scripts/([A-Za-z_]\w*\.py)')) {
                $name = $match.Groups[1].Value
                if ((Test-Path -LiteralPath (Join-Path $root ('scripts/'+$name))) -and $scriptNames.Add($name)) { $more = $true }
            }
        }
    }
    foreach ($name in $scriptNames) { [void]$keepExact.Add('scripts/'+$name) }
    $files = @(Get-ChildItem -LiteralPath $root -Force | Where-Object { $_.Name -notin @('.git','.agents','.codex') } | ForEach-Object {
        if ($_.PSIsContainer) { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force } else { $_ }
    } | Where-Object { $_.FullName -notlike "$receiptDir\*" })
    $rows = [Collections.Generic.List[object]]::new()
    $count = 0
    foreach ($file in $files) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point requires separate review: $($file.FullName)" }
        $relative = $file.FullName.Substring($root.Length+1).Replace('\','/')
        $keep = $keepExact.Contains($relative)
        foreach ($prefix in $prefixes) { if ($relative.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)) { $keep=$true; break } }
        if ($relative -match '(^|/)__pycache__/|\.pyc$') { $keep=$false }
        if ($relative -match '\.tar(\.gz)?$|\.zip$') { $keep=$false }
        $rows.Add([pscustomobject]@{path=$relative;bytes=$file.Length;sha256=(Hash $file.FullName);keep=$keep})
        $count++
        if ($count % 1000 -eq 0) { Write-Output "Hashed $count/$($files.Count) files" }
    }
    $head = & git -c "safe.directory=$($root.Replace('\','/'))" rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot identify Git HEAD' }
    $archive = Join-Path $historyParent (Get-Date -Format 'yyyyMMdd_HHmmss')
    [void](Check-Path $archive $historyParent)
    $plan = [ordered]@{version=1;created_at=(Get-Date).ToString('o');workspace=$root;archive=$archive;git_head=$head;
        mutable_files=$mutable;script_dependencies=@($scriptNames | Sort-Object);files=@($rows)}
    Write-Json $planPath $plan
    $moved = @($rows | Where-Object { -not $_.keep }); $kept=@($rows | Where-Object keep)
    Write-Json (Join-Path $receiptDir 'plan_summary.json') ([ordered]@{
        status='planned';archive=$archive;files_before=$rows.Count;bytes_before=($rows | Measure-Object bytes -Sum).Sum;
        files_to_archive=$moved.Count;bytes_to_archive=($moved | Measure-Object bytes -Sum).Sum;
        files_retained=$kept.Count;bytes_retained=($kept | Measure-Object bytes -Sum).Sum;
        retained_scripts=@($scriptNames | Sort-Object);new_trainings=0;new_API_calls=0})
    Get-Content -LiteralPath (Join-Path $receiptDir 'plan_summary.json') -Raw
    exit
}
$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ($plan.workspace -ne $root) { throw 'Workspace differs from plan' }
$archive = Check-Path $plan.archive $historyParent
if ($Mode -eq 'Apply') {
    [void](New-Item -ItemType Directory -Path $archive -Force)
    Copy-Item -LiteralPath $planPath -Destination (Join-Path $archive 'archive_manifest.json')
    foreach ($relative in $plan.mutable_files) {
        $source=Check-Path (Join-Path $root $relative) $root
        $dest=Check-Path (Join-Path $archive ('metadata_before_cleanup/'+$relative)) $archive
        [void](New-Item -ItemType Directory -Path (Split-Path $dest) -Force)
        if (-not (Test-Path -LiteralPath $dest)) { Copy-Item -LiteralPath $source -Destination $dest }
        $expected=($plan.files | Where-Object path -eq $relative).sha256
        if ((Hash $dest) -ne $expected) { throw "Metadata backup mismatch: $relative" }
    }
    $count=0; $moved=@($plan.files | Where-Object { -not $_.keep })
    foreach ($row in $moved) {
        $source=Check-Path (Join-Path $root $row.path) $root
        $dest=Check-Path (Join-Path $archive ('files/'+$row.path)) $archive
        if ($row.path -match '^(\.git/|data/)') { throw "Protected path in move list: $($row.path)" }
        if (Test-Path -LiteralPath $source) {
            if ((Hash $source) -ne $row.sha256) { throw "Source changed after planning: $($row.path)" }
            if (Test-Path -LiteralPath $dest) { throw "Archive destination already exists: $dest" }
            [void](New-Item -ItemType Directory -Path (Split-Path $dest) -Force)
            Move-Item -LiteralPath $source -Destination $dest
        }
        if (-not (Test-Path -LiteralPath $dest) -or (Hash $dest) -ne $row.sha256) { throw "Archive mismatch: $($row.path)" }
        $count++
        if ($count % 1000 -eq 0) { Write-Output "Archived and verified $count/$($moved.Count) files" }
    }
    # Remove empty directories only; never recurse through .git or delete any files.
    $dirs=@(Get-ChildItem -LiteralPath $root -Force -Directory | Where-Object { $_.Name -notin @('.git','.agents','.codex') } | ForEach-Object {
        Get-ChildItem -LiteralPath $_.FullName -Recurse -Directory -Force; $_
    } | Sort-Object { $_.FullName.Length } -Descending)
    foreach ($dir in $dirs) {
        [void](Check-Path $dir.FullName $root)
        if (@(Get-ChildItem -LiteralPath $dir.FullName -Force).Count -eq 0) { Remove-Item -LiteralPath $dir.FullName }
    }
}
$retained=@($plan.files | Where-Object keep); $checked=0
foreach ($row in $retained) {
    $path=Check-Path (Join-Path $root $row.path) $root
    if (-not (Test-Path -LiteralPath $path)) { throw "Retained file missing: $($row.path)" }
    if ($row.path -notin $plan.mutable_files -and (Hash $path) -ne $row.sha256) { throw "Retained file changed: $($row.path)" }
    $checked++
}
$archived=@($plan.files | Where-Object { -not $_.keep })
foreach ($row in $archived) {
    $path=Check-Path (Join-Path $archive ('files/'+$row.path)) $archive
    if ((Hash $path) -ne $row.sha256) { throw "Archived file changed: $($row.path)" }
    if (Test-Path -LiteralPath (Join-Path $root $row.path)) { throw "Archived file still in workspace: $($row.path)" }
}
Write-Json (Join-Path $receiptDir 'verification.json') ([ordered]@{
    status='passed';verified_at=(Get-Date).ToString('o');archive=$archive;
    retained_files_verified=$checked;archived_files_verified=$archived.Count;
    immutable_retained_hashes_unchanged=$true;metadata_originals_backed_up=$true;
    deleted_scientific_assets=0;new_trainings=0;new_API_calls=0;
    server_runtime_tests='not_run_local_file_cleanup_only';git_sync='pending'})
Get-Content -LiteralPath (Join-Path $receiptDir 'verification.json') -Raw
