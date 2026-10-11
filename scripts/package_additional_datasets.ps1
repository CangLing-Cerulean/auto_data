$ErrorActionPreference = 'Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskStage=Join-Path $taskRoot 'research_runs/dataset_build/20261011'
$taskArchiveDir=Join-Path $taskRoot 'archives/20261011_datasets'
$taskArchivePath=Join-Path $taskArchiveDir 'datasets_skab_hydraulic_v1.zip'
if (Test-Path -LiteralPath $taskArchivePath) { throw 'Refusing to overwrite an existing data release asset' }
$taskSources=Get-Content -LiteralPath (Join-Path $taskStage 'sources.json') -Raw -Encoding utf8 | ConvertFrom-Json
$taskPaths=@($taskSources.sources | ForEach-Object { $_.local_file })
foreach ($taskId in @('skab_v1','uci_hydraulic_v1')) {
    $taskPaths+=Get-ChildItem -LiteralPath (Join-Path $taskRoot "data/$taskId") -File | ForEach-Object {
        [IO.Path]::GetRelativePath($taskRoot,$_.FullName).Replace('\','/')
    }
}
$taskPaths+=@('scripts/build_additional_datasets.ps1','scripts/dataset_tables.cs','scripts/package_additional_datasets.ps1')
foreach ($taskName in @('sources.json','PROTOCOL.md','REPORT.md','build_receipt.json','verification.json',
    'costs.json','failures.json','license_receipt.json','independent_review.json')) {
    $taskPaths+="research_runs/dataset_build/20261011/$taskName"
}
$taskPaths=@($taskPaths | Sort-Object -Unique)
$taskFiles=foreach ($taskRelative in $taskPaths) {
    $taskPath=Join-Path $taskRoot $taskRelative
    [ordered]@{path=$taskRelative;bytes=(Get-Item -LiteralPath $taskPath).Length;
        sha256=(Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$taskCatalog=Join-Path $taskStage 'package_files.json'
$taskFiles | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $taskCatalog -Encoding utf8
New-Item -ItemType Directory -Path $taskArchiveDir -Force | Out-Null
$taskZip=[IO.Compression.ZipFile]::Open($taskArchivePath,[IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($taskRelative in $taskPaths) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskZip,(Join-Path $taskRoot $taskRelative),$taskRelative,
            [IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($taskZip,$taskCatalog,
        'research_runs/dataset_build/20261011/package_files.json',[IO.Compression.CompressionLevel]::Optimal) | Out-Null
} finally { $taskZip.Dispose() }
# Reopen each asset entry and verify its uncompressed bytes against the publication catalog.
$taskZip=[IO.Compression.ZipFile]::OpenRead($taskArchivePath)
try {
    if ($taskZip.Entries.Count -ne $taskFiles.Count+1) { throw 'Archive entry count differs' }
    foreach ($taskFile in $taskFiles) {
        $taskEntry=$taskZip.GetEntry($taskFile.path)
        if (-not $taskEntry -or $taskEntry.Length -ne $taskFile.bytes) { throw 'Archive scope/size mismatch' }
        $taskStream=$taskEntry.Open()
        try { $taskHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($taskStream)).ToLowerInvariant() } finally { $taskStream.Dispose() }
        if ($taskHash -ne $taskFile.sha256) { throw "Archive checksum mismatch: $($taskFile.path)" }
    }
    $taskEntry=$taskZip.GetEntry('research_runs/dataset_build/20261011/package_files.json')
    $taskStream=$taskEntry.Open()
    try { $taskHash=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($taskStream)).ToLowerInvariant() } finally { $taskStream.Dispose() }
    if ($taskHash -ne (Get-FileHash $taskCatalog -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Archive catalog checksum mismatch' }
} finally { $taskZip.Dispose() }
$taskAsset=[ordered]@{name=[IO.Path]::GetFileName($taskArchivePath);path='archives/20261011_datasets/datasets_skab_hydraulic_v1.zip';
    bytes=(Get-Item -LiteralPath $taskArchivePath).Length;sha256=(Get-FileHash -LiteralPath $taskArchivePath -Algorithm SHA256).Hash.ToLowerInvariant();
    entry_count=$taskFiles.Count+1;scope='Selected 43 raw sensor files, prepared tables, licenses, builder and verification metadata only'}
[ordered]@{repository='CangLing-Cerulean/auto_data';tag='datasets-20261011-v1';title='SKAB and UCI Hydraulic frozen base datasets v1';assets=@($taskAsset)} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $taskStage 'assets.json') -Encoding utf8
$taskAsset | ConvertTo-Json -Depth 4
