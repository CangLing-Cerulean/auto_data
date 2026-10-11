param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskStage = Join-Path $taskRoot 'research_runs/dataset_build/20261011'
$taskRaw = Join-Path $taskRoot 'data/raw_sources/20261011'
$taskArchive = Join-Path $taskRoot 'data/share_20260727.zip'
$taskInnerName = 'industrial_timeseries_eval_120_v0.7.0.zip'
$taskInnerHash = 'b0c2489d1103f224c55e070da956f00e7905d4389850f31f5d2a34f42227a940'
$taskStarted = [DateTime]::UtcNow
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
Add-Type -Path (Join-Path $PSScriptRoot 'dataset_tables.cs')
function Write-TaskJson($Path, $Value) {
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Path -Encoding utf8
}
function Assert-TaskEqual($Actual, $Expected, $Label) {
    # JSON object property order has no semantic meaning. Normalize CLR/PS objects first.
    $taskActualJson=ConvertTo-Json -InputObject $Actual -Depth 30 -Compress
    $taskExpectedJson=ConvertTo-Json -InputObject $Expected -Depth 30 -Compress
    Assert-TaskNode ($taskActualJson | ConvertFrom-Json -AsHashtable) ($taskExpectedJson | ConvertFrom-Json -AsHashtable) $Label
}
function Assert-TaskNode($Actual, $Expected, $Label) {
    if ($Expected -is [Collections.IDictionary]) {
        if ($Actual -isnot [Collections.IDictionary] -or $Actual.Count -ne $Expected.Count) { throw "Verification mismatch: $Label keys" }
        foreach ($taskKey in $Expected.Keys) {
            if (-not $Actual.Contains($taskKey)) { throw "Verification missing: $Label.$taskKey" }
            Assert-TaskNode $Actual[$taskKey] $Expected[$taskKey] "$Label.$taskKey"
        }
    } elseif ($Expected -is [array]) {
        if ($Actual -isnot [array] -or $Actual.Count -ne $Expected.Count) { throw "Verification mismatch: $Label array" }
        for ($taskIndex=0; $taskIndex -lt $Expected.Count; $taskIndex++) {
            Assert-TaskNode $Actual[$taskIndex] $Expected[$taskIndex] "$Label[$taskIndex]"
        }
    } elseif ((ConvertTo-Json -InputObject $Actual -Compress) -cne (ConvertTo-Json -InputObject $Expected -Compress)) {
        throw "Verification mismatch: $Label"
    }
}
$taskMetroRoot = Join-Path $taskRoot 'data/metropt3_v1'
$taskMetroManifest = Get-Content -LiteralPath (Join-Path $taskMetroRoot 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskMetroIdentity = @{}
foreach ($taskName in @('manifest.json','train_profile.json','train.csv','validation.csv','test.csv')) {
    $taskMetroIdentity[$taskName] = [DatasetTables]::HashFile((Join-Path $taskMetroRoot $taskName))
}
foreach ($taskSplit in $taskMetroManifest.splits) {
    if ($taskMetroIdentity[$taskSplit.file] -ne $taskSplit.sha256) { throw 'Original MetroPT3 checksum changed' }
}
if ([DatasetTables]::ReconstructMetro($taskMetroRoot) -ne $taskMetroManifest.source_sha256) { throw 'MetroPT3 source reconstruction failed' }
$taskSkabSensors = @('Accelerometer1RMS','Accelerometer2RMS','Current','Pressure','Temperature','Thermocouple','Voltage','Volume Flow RateRMS')
$taskHydraulicSensors = @('CE','CP','SE','TS1','TS2','TS3','TS4','VS1')
$taskDefinitions = @(
    @{ id='skab_v1'; raw='skab'; sensors=$taskSkabSensors; skab=$true; expectedFiles=35; expectedRows=46860 },
    @{ id='uci_hydraulic_v1'; raw='hydraulic'; sensors=$taskHydraulicSensors; skab=$false; expectedFiles=8; expectedRows=132300 }
)
if (-not $VerifyOnly) {
    foreach ($taskDef in $taskDefinitions) {
        if (Test-Path -LiteralPath (Join-Path $taskRoot "data/$($taskDef.id)")) { throw "Refusing to overwrite frozen dataset: $($taskDef.id)" }
    }
    New-Item -ItemType Directory -Path $taskStage -Force | Out-Null
    New-Item -ItemType Directory -Path $taskRaw -Force | Out-Null
    $taskOuter = [IO.Compression.ZipFile]::OpenRead($taskArchive)
    $taskMemory = [IO.MemoryStream]::new()
    try {
        $taskStream = $taskOuter.GetEntry("share_20260727/$taskInnerName").Open()
        try { $taskStream.CopyTo($taskMemory) } finally { $taskStream.Dispose() }
        $taskMemory.Position = 0
        $taskActualHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($taskMemory)).ToLowerInvariant()
        if ($taskActualHash -ne $taskInnerHash) { throw 'Inner archive identity differs from original manifest' }
        $taskMemory.Position = 0
        $taskInner = [IO.Compression.ZipArchive]::new($taskMemory,[IO.Compression.ZipArchiveMode]::Read)
        try {
            $taskReader = [IO.StreamReader]::new($taskInner.GetEntry('version2_v0.7.0/SHA256SUMS').Open())
            try { $taskChecksums = @{}; foreach ($taskLine in $taskReader.ReadToEnd().Split("`n")) {
                if ($taskLine -match '^([a-f0-9]{64})  (.+?)\r?$') { $taskChecksums[$Matches[2]]=$Matches[1] }
            } } finally { $taskReader.Dispose() }
            $taskSources = @()
            foreach ($taskEntry in $taskInner.Entries) {
                $taskRelative = $taskEntry.FullName.Replace('version2_v0.7.0/','')
                $taskFamily=$null; $taskLocal=$null
                if ($taskRelative -match '^raw/skab_v0_9/SKAB/(anomaly-free|other|valve1|valve2)/[^/]+\.csv$') {
                    $taskFamily='skab'; $taskLocal=$taskRelative.Replace('raw/skab_v0_9/SKAB/','')
                } elseif ($taskRelative -match '^raw/uci_hydraulic/(CE|CP|SE|TS[1-4]|VS1)\.txt$') {
                    $taskFamily='hydraulic'; $taskLocal=[IO.Path]::GetFileName($taskRelative)
                }
                if (-not $taskFamily) { continue }
                $taskPath=Join-Path $taskRaw "$taskFamily/$taskLocal"
                New-Item -ItemType Directory -Path (Split-Path -Parent $taskPath) -Force | Out-Null
                # Never overwrite a different source, including a partial previous run.
                if (Test-Path -LiteralPath $taskPath) {
                    if ([DatasetTables]::HashFile($taskPath) -ne $taskChecksums[$taskRelative]) { throw 'Existing source differs' }
                } else {
                    $taskInput=$taskEntry.Open(); $taskOutput=[IO.File]::Open($taskPath,[IO.FileMode]::CreateNew)
                    try { $taskInput.CopyTo($taskOutput) } finally { $taskInput.Dispose();$taskOutput.Dispose() }
                }
                $taskHash=[DatasetTables]::HashFile($taskPath)
                if ($taskHash -ne $taskChecksums[$taskRelative]) { throw "Source checksum mismatch: $taskRelative" }
                $taskSources += [ordered]@{family=$taskFamily; archive_entry=$taskEntry.FullName;
                    local_file="data/raw_sources/20261011/$taskFamily/$taskLocal"; sha256=$taskHash; bytes=$taskEntry.Length}
            }
            foreach ($taskMetadata in @('LICENSE_AND_CITATION.md','dataset_manifest.yaml')) {
                $taskEntry=$taskInner.GetEntry("version2_v0.7.0/$taskMetadata")
                $taskInput=$taskEntry.Open();$taskOutput=[IO.File]::Create((Join-Path $taskRaw $taskMetadata))
                try {$taskInput.CopyTo($taskOutput)} finally {$taskInput.Dispose();$taskOutput.Dispose()}
            }
        } finally { $taskInner.Dispose() }
    } finally { $taskOuter.Dispose();$taskMemory.Dispose() }
    Write-TaskJson (Join-Path $taskStage 'sources.json') ([ordered]@{outer_archive='data/share_20260727.zip';
        outer_sha256=[DatasetTables]::HashFile($taskArchive); inner_archive=$taskInnerName;inner_sha256=$taskInnerHash;sources=$taskSources})
} else {
    $taskSourceIndex=Get-Content -LiteralPath (Join-Path $taskStage 'sources.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $taskSources=$taskSourceIndex.sources
    if ([DatasetTables]::HashFile($taskArchive) -ne $taskSourceIndex.outer_sha256) { throw 'Outer archive changed' }
}
$taskAudits=@()
foreach ($taskDef in $taskDefinitions) {
    $taskSourcesForDataset=@($taskSources | Where-Object family -eq $taskDef.raw)
    if ($taskSourcesForDataset.Count -ne $taskDef.expectedFiles) { throw 'Incomplete source scope' }
    foreach ($taskSource in $taskSourcesForDataset) {
        $taskPath=Join-Path $taskRoot $taskSource.local_file
        if ([DatasetTables]::HashFile($taskPath) -ne $taskSource.sha256 -or (Get-Item -LiteralPath $taskPath).Length -ne $taskSource.bytes) { throw 'Raw source identity failed' }
    }
    $taskOutputDir=Join-Path $taskRoot "data/$($taskDef.id)"
    if (-not $VerifyOnly) { New-Item -ItemType Directory -Path $taskOutputDir | Out-Null }
    $taskData=if ($taskDef.skab) { [DatasetTables]::Skab((Join-Path $taskRaw $taskDef.raw),$taskDef.sensors) } else {
        [DatasetTables]::Hydraulic((Join-Path $taskRaw $taskDef.raw),$taskDef.sensors)
    }
    if ($taskData.Count -ne $taskDef.expectedRows) { throw 'Unexpected row count' }
    $taskPackage=[DatasetTables]::Package($taskData,$taskDef.sensors,$taskOutputDir,$taskDef.skab,[bool]$VerifyOnly)
    $taskProfile=$taskPackage['train_profile'];$taskPackage.Remove('train_profile') | Out-Null
    $taskExpectedManifest=[ordered]@{dataset_id=$taskDef.id;schema_version='multiseries-raw-v1';created_date='2026-10-11';
            source_archive=$taskInnerName;source_archive_sha256=$taskInnerHash;sources=$taskSourcesForDataset;
            sensor_columns=$taskDef.sensors;split_rule='chronological; floor(N*0.7), floor(N*0.2), remainder';
            ordering=$(if($taskDef.skab){'datetime, source_file (ordinal), source_row'}else{'cycle_id, sample_offset_seconds'});
            transforms=@('lossless table assembly; original numeric strings retained; provenance added');
            window_protocol=$null;boundary_contract=$(if($taskDef.skab){
                'Group by source_file within each partition; retain source_row order; never cross files, partitions, or timestamp gaps >30s. Equal timestamps from overlapping experiments remain separate observations in one partition.'
            }else{
                'Group by cycle_id within each partition; never cross cycles or partitions. A cycle lasts 60s with 1Hz samples. No wall-clock timestamps or continuous inter-cycle time inferred.'
            });excluded_input_fields=$(if($taskDef.skab){@('source_file','source_row','datetime','anomaly','changepoint')}else{@('cycle_id','sample_offset_seconds','profile.txt')});
            quality=@{raw_sources_byte_identical=$true;table_raw_bytes_preserved=$false;sensor_numeric_tokens_preserved=$true;
                missing_or_nonfinite_sensor_values=0;all_source_samples_retained=$true;strictly_increasing_time_within_series=$true};
            builder=@{file='scripts/build_additional_datasets.ps1';sha256=[DatasetTables]::HashFile($PSCommandPath);
                table_code='scripts/dataset_tables.cs';table_code_sha256=[DatasetTables]::HashFile((Join-Path $PSScriptRoot 'dataset_tables.cs'))}}
    foreach ($taskKey in $taskPackage.Keys) { $taskExpectedManifest[$taskKey]=$taskPackage[$taskKey] }
    if ($VerifyOnly) {
        $taskManifest=Get-Content -LiteralPath (Join-Path $taskOutputDir 'manifest.json') -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
        Assert-TaskEqual $taskManifest $taskExpectedManifest 'complete manifest, including input, boundary, source and builder contracts'
        $taskStoredProfile=Get-Content -LiteralPath (Join-Path $taskOutputDir 'train_profile.json') -Raw -Encoding utf8 | ConvertFrom-Json -AsHashtable
        Assert-TaskEqual $taskStoredProfile $taskProfile 'train-only profile'
    } else {
        Write-TaskJson (Join-Path $taskOutputDir 'manifest.json') $taskExpectedManifest
        Write-TaskJson (Join-Path $taskOutputDir 'train_profile.json') $taskProfile
        Copy-Item -LiteralPath (Join-Path $taskRaw 'LICENSE_AND_CITATION.md') -Destination $taskOutputDir
    }
    $taskAudits += [ordered]@{dataset_id=$taskDef.id;status='PASS';rows=$taskData.Count;source_files=$taskSourcesForDataset.Count;
        split_rows=@($taskPackage['splits'] | ForEach-Object { $_['rows'] });
        manifest_sha256=[DatasetTables]::HashFile((Join-Path $taskOutputDir 'manifest.json'));
        checks=@('raw SHA256 and bytes','complete source scope','all original numeric tokens and provenance','exact 7:2:1 counts',
            'strictly disjoint split order boundaries','series chronology','finite sensors (integrity only)','train-only profile','all split CSV SHA256')}
    Write-Output "PASS: $($taskDef.id), $($taskData.Count) rows"
}
foreach ($taskName in $taskMetroIdentity.Keys) {
    if ([DatasetTables]::HashFile((Join-Path $taskMetroRoot $taskName)) -ne $taskMetroIdentity[$taskName]) { throw 'Original dataset changed during packaging' }
}
$taskWatch.Stop()
Write-TaskJson (Join-Path $taskStage $(if($VerifyOnly){'verification.json'}else{'build_receipt.json'})) ([ordered]@{
    status='PASS';started_utc=$taskStarted.ToString('o');finished_utc=[DateTime]::UtcNow.ToString('o');
    local_wall_seconds=$taskWatch.Elapsed.TotalSeconds;datasets=$taskAudits;metropt3_file_sha256=$taskMetroIdentity;
    metropt3_source_reconstruction='PASS';new_trainings=0;model_api_calls=0;gpu_seconds=0;
    test_access='integrity and deterministic packaging only; no distribution summaries, model scores or selection feedback';
    ground_truth_access=$false})
