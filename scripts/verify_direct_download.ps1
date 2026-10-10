$ErrorActionPreference = 'Stop'

$taskWorkspace = 'C:\Users\31227\Desktop\auto_data'
$taskStudy = Join-Path $taskWorkspace 'research_runs\direct_processing\20261009'
$taskName = 'direct_processing_complete_20261009'
$taskArchive = Join-Path $taskWorkspace "research_runs\direct_processing\$taskName.tar"
$taskReceipt = Get-Content -LiteralPath (Join-Path $taskStudy "$taskName`_archive_receipt.json") -Raw | ConvertFrom-Json
$taskManifest = Get-Content -LiteralPath (Join-Path $taskStudy "$taskName`_asset_manifest.json") -Raw | ConvertFrom-Json
$taskExtraction = Join-Path $taskWorkspace 'archives\direct_processing_20261009_complete'

if ((Get-Item -LiteralPath $taskArchive).Length -ne $taskReceipt.bytes) { throw 'Archive length differs' }
if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskReceipt.sha256) { throw 'Archive hash differs' }
$taskMembers = @(& tar -tf $taskArchive)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect archive members' }
$taskDetailed = @(& tar -tvf $taskArchive)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect archive member types' }
if (@($taskDetailed | Where-Object { -not $_.StartsWith('-') }).Count -ne 0) { throw 'Archive contains non-regular files' }
if ($taskMembers.Count -ne $taskReceipt.members -or @($taskMembers | Select-Object -Unique).Count -ne $taskMembers.Count) { throw 'Archive member count or uniqueness differs' }
$taskManifestRelative = "research_runs/direct_processing/20261009/$taskName`_asset_manifest.json"
$taskExpected = @($taskManifest.PSObject.Properties.Name) + @($taskManifestRelative)
if (@(Compare-Object $taskMembers $taskExpected).Count -ne 0) { throw 'Archive and manifest membership differs' }
foreach ($taskMember in $taskMembers) {
    if ([IO.Path]::IsPathRooted($taskMember) -or $taskMember.Contains(':') -or @($taskMember -split '[/\\]' | Where-Object { $_ -eq '..' }).Count -ne 0) { throw "Unsafe member: $taskMember" }
    $taskResolved = [IO.Path]::GetFullPath((Join-Path $taskExtraction $taskMember))
    if (-not $taskResolved.StartsWith($taskExtraction + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Member escapes extraction directory' }
}
if (Test-Path -LiteralPath $taskExtraction) { throw 'Extraction directory already exists; preserve it' }
New-Item -ItemType Directory -Path $taskExtraction | Out-Null
& tar -xf $taskArchive -C $taskExtraction
if ($LASTEXITCODE -ne 0) { throw 'Archive extraction failed' }

$taskDivergences = @()
foreach ($taskProperty in $taskManifest.PSObject.Properties) {
    $taskPath = Join-Path $taskExtraction $taskProperty.Name
    if ((Get-Item -LiteralPath $taskPath).Length -ne $taskProperty.Value.bytes) { throw "Member length differs: $($taskProperty.Name)" }
    if ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value.sha256) { throw "Member hash differs: $($taskProperty.Name)" }
    $taskLive = Join-Path $taskWorkspace $taskProperty.Name
    if (Test-Path -LiteralPath $taskLive) {
        if ((Get-FileHash -LiteralPath $taskLive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value.sha256) { $taskDivergences += $taskProperty.Name }
    } else {
        New-Item -ItemType Directory -Path (Split-Path -Parent $taskLive) -Force | Out-Null
        Copy-Item -LiteralPath $taskPath -Destination $taskLive
    }
}
$taskReference = Join-Path $taskStudy 'runs\DP03_full_source_half_s2026\best.pt'
if ((Get-FileHash -LiteralPath $taskReference -Algorithm SHA256).Hash.ToLowerInvariant() -ne '73a34d209a630a20cebc183254349ed1884b8546d6ec9265989e4792970079e9') { throw 'Reference checkpoint differs' }
$taskVerification = [ordered]@{
    status = 'passed'
    archive_sha256 = $taskReceipt.sha256
    archive_bytes = $taskReceipt.bytes
    archive_members = $taskMembers.Count
    individually_hashed_members = @($taskManifest.PSObject.Properties).Count
    extraction_directory = $taskExtraction
    completed_training_runs = $taskReceipt.training_runs
    live_files_with_different_snapshot_hash_preserved = @($taskDivergences)
    reference_checkpoint_verified = $true
    additional_training = 0
    API_calls = 0
    test_access = $false
}
$taskVerification | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $taskStudy 'download_verification.json') -Encoding utf8
$taskVerification | ConvertTo-Json -Depth 10
