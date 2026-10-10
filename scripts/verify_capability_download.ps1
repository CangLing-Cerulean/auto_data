param([Parameter(Mandatory=$true)][ValidateSet('20261010_v3','20261010_v4')][string]$StudyVersion)
$ErrorActionPreference = 'Stop'
$taskWorkspace = 'C:\Users\31227\Desktop\auto_data'
$taskStudy = Join-Path $taskWorkspace "research_runs/agent_research/capability_acceptance/$StudyVersion"
$taskReceipt = Get-Content -LiteralPath (Join-Path $taskStudy 'closure_archive_receipt.json') -Raw | ConvertFrom-Json
$taskArchive = Join-Path $taskStudy ([IO.Path]::GetFileName($taskReceipt.archive))
if ((Get-Item -LiteralPath $taskArchive).Length -ne $taskReceipt.bytes) { throw 'Archive size differs' }
if ((Get-FileHash -LiteralPath $taskArchive).Hash.ToLowerInvariant() -ne $taskReceipt.sha256) { throw 'Archive hash differs' }
$taskMembers = @(& tar -tf $taskArchive)
if ($LASTEXITCODE -ne 0) { throw 'Member listing failed' }
$taskTypes = @(& tar -tvf $taskArchive)
if ($LASTEXITCODE -ne 0 -or @($taskTypes | Where-Object { -not $_.StartsWith('-') }).Count -ne 0) { throw 'Unexpected member type' }
if ($taskMembers.Count -ne $taskReceipt.members -or @($taskMembers | Select-Object -Unique).Count -ne $taskMembers.Count) { throw 'Member count or uniqueness differs' }
$taskExtraction = [IO.Path]::GetFullPath((Join-Path $taskWorkspace "archives/capability_acceptance_$StudyVersion"))
foreach ($taskMember in $taskMembers) {
    if ([IO.Path]::IsPathRooted($taskMember) -or $taskMember.Contains(':') -or @($taskMember -split '[/\\]' | Where-Object { $_ -eq '..' }).Count -ne 0) { throw 'Unsafe member' }
    $taskResolved = [IO.Path]::GetFullPath((Join-Path $taskExtraction $taskMember))
    if (-not $taskResolved.StartsWith($taskExtraction + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Member escapes extraction' }
}
if (Test-Path -LiteralPath $taskExtraction) { throw 'Extraction exists; preserve it' }
New-Item -ItemType Directory -Path $taskExtraction | Out-Null
& tar -xf $taskArchive -C $taskExtraction
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
$taskManifest = Get-Content -LiteralPath (Join-Path $taskExtraction 'closure_asset_manifest.json') -Raw | ConvertFrom-Json
$taskExpected = @($taskManifest.PSObject.Properties.Name) + @('closure_asset_manifest.json')
if (@(Compare-Object $taskMembers $taskExpected).Count -ne 0) { throw 'Manifest membership differs' }
foreach ($taskProperty in $taskManifest.PSObject.Properties) {
    $taskPath = Join-Path $taskExtraction $taskProperty.Name
    if ((Get-Item -LiteralPath $taskPath).Length -ne $taskProperty.Value.bytes) { throw 'Member length differs' }
    if ((Get-FileHash -LiteralPath $taskPath).Hash.ToLowerInvariant() -ne $taskProperty.Value.sha256) { throw "Member hash differs: $($taskProperty.Name)" }
}
$taskVerification = [ordered]@{status='passed';archive_sha256=$taskReceipt.sha256;archive_bytes=$taskReceipt.bytes;archive_members=$taskMembers.Count;individually_hashed_members=@($taskManifest.PSObject.Properties).Count;extraction_directory=$taskExtraction;additional_training=0;API_calls=0;test_access=$false}
$taskVerification | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $taskStudy 'download_verification.json') -Encoding utf8
foreach ($taskName in @('cost_audit.json','closure_summary.json')) { Copy-Item -LiteralPath (Join-Path $taskExtraction $taskName) -Destination $taskStudy }
$taskVerification | ConvertTo-Json -Depth 10
