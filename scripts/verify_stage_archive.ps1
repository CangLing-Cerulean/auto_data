param([Parameter(Mandatory)][string]$Stage)
$ErrorActionPreference = 'Stop'
$taskStage = (Resolve-Path -LiteralPath $Stage).Path
$taskReceipt = Get-Content -LiteralPath (Join-Path $taskStage 'archive_receipt.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskArchive = Join-Path $taskStage ('archives/' + $taskReceipt.file)
$taskDestination = Join-Path $taskStage 'archives/verified'
if ((Get-Item -LiteralPath $taskArchive).Length -ne $taskReceipt.bytes) { throw 'Archive length mismatch' }
if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskReceipt.sha256) { throw 'Archive hash mismatch' }
$taskMembers = @(& tar -tf $taskArchive)
if ($LASTEXITCODE -ne 0) { throw 'Archive listing failed' }
$taskTypes = @(& tar -tvf $taskArchive)
if ($LASTEXITCODE -ne 0 -or @($taskTypes | Where-Object { -not $_.StartsWith('-') }).Count) { throw 'Non-regular archive member' }
$taskExpected = @($taskReceipt.members_sha256.PSObject.Properties.Name)
if (@($taskMembers | Select-Object -Unique).Count -ne $taskMembers.Count -or @(Compare-Object $taskMembers $taskExpected).Count) { throw 'Member identity mismatch' }
foreach ($taskMember in $taskMembers) {
    if ([IO.Path]::IsPathRooted($taskMember) -or $taskMember.Contains(':') -or $taskMember.Contains('\') -or @($taskMember.Split('/') | Where-Object { $_ -in @('..', '.', '') }).Count) { throw "Unsafe member $taskMember" }
    $taskResolved = [IO.Path]::GetFullPath((Join-Path $taskDestination $taskMember))
    if (-not $taskResolved.StartsWith($taskDestination + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Member escapes destination' }
}
if (Test-Path -LiteralPath $taskDestination) { throw 'Preserve existing extraction' }
New-Item -ItemType Directory -Path $taskDestination | Out-Null
& tar -xf $taskArchive -C $taskDestination
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
foreach ($taskProperty in $taskReceipt.members_sha256.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath (Join-Path $taskDestination $taskProperty.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProperty.Value) { throw "Member hash mismatch $($taskProperty.Name)" }
}
[ordered]@{status='PASS';verified_utc=[DateTime]::UtcNow.ToString('o');archive_sha256=$taskReceipt.sha256;archive_bytes=$taskReceipt.bytes;individually_hashed_members=$taskMembers.Count;safe_regular_members=$true;new_training=0;test_access=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskStage 'download_verification.json') -Encoding utf8
Write-Output "Verified $($taskMembers.Count) archive members"
