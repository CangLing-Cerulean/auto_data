param(
    [Parameter(Mandatory)][string]$Tag,
    [Parameter(Mandatory)][string]$TargetCommit,
    [Parameter(Mandatory)][string]$NotesPath,
    [Parameter(Mandatory)][string]$ManifestPath,
    [Parameter(Mandatory)][string]$ReceiptPath
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskRepository = 'CangLing-Cerulean/auto_data'
$taskApiRoot = "https://api.github.com/repos/$taskRepository"
$taskManifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskManifest.tag -ne $Tag -or $taskManifest.repository -ne $taskRepository) {
    throw 'Release identity differs from the registered manifest'
}
if ($TargetCommit -notmatch '^[a-f0-9]{40}$') { throw 'Expected a full commit SHA' }
$taskFiles = @()
foreach ($taskAsset in $taskManifest.assets) {
    $taskPath = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskAsset.path))
    if (-not $taskPath.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Asset path is outside the project'
    }
    $taskItem = Get-Item -LiteralPath $taskPath
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskItem.Length -ne $taskAsset.bytes -or $taskHash -ne $taskAsset.sha256) {
        throw "Local asset identity mismatch: $($taskAsset.name)"
    }
    if ($taskItem.Length -ge 2GB) { throw 'Split and register assets larger than the GitHub per-file limit before uploading' }
    $taskFiles += [pscustomobject]@{ Path=$taskPath; Asset=$taskAsset }
}

$env:GIT_TERMINAL_PROMPT = '0'
$taskCredential = "protocol=https`nhost=github.com`n`n" | git -c "safe.directory=$taskRoot" credential fill 2>$null
if ($LASTEXITCODE -ne 0) { throw 'GitHub credential lookup failed' }
$taskPasswordLine = $taskCredential | Where-Object { $_.StartsWith('password=') } | Select-Object -First 1
if (-not $taskPasswordLine) { throw 'GitHub credential missing' }
$taskHttp = [Net.Http.HttpClient]::new()
$taskHttp.Timeout = [TimeSpan]::FromMinutes(60)
$taskHttp.DefaultRequestHeaders.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $taskPasswordLine.Substring(9))
$taskHttp.DefaultRequestHeaders.UserAgent.ParseAdd('auto-data-stage-sync/1.0')
$taskHttp.DefaultRequestHeaders.Accept.ParseAdd('application/vnd.github+json')
$taskHttp.DefaultRequestHeaders.Add('X-GitHub-Api-Version', '2022-11-28')
Remove-Variable taskCredential,taskPasswordLine

function Invoke-StageApi {
    param([string]$Method,[string]$Url,[object]$Body,[switch]$AllowNotFound)
    $taskRequest = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Method), $Url)
    try {
        if ($null -ne $Body) {
            $taskRequest.Content = [Net.Http.StringContent]::new(($Body | ConvertTo-Json -Depth 12 -Compress), [Text.Encoding]::UTF8, 'application/json')
        }
        $taskResponse = $taskHttp.SendAsync($taskRequest).GetAwaiter().GetResult()
        try {
            if ($AllowNotFound -and [int]$taskResponse.StatusCode -eq 404) { return $null }
            if (-not $taskResponse.IsSuccessStatusCode) { throw "GitHub API failed: HTTP $([int]$taskResponse.StatusCode)" }
            return $taskResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
        } finally { $taskResponse.Dispose() }
    } finally { $taskRequest.Dispose() }
}

try {
    $taskRelease = Invoke-StageApi -Method GET -Url "$taskApiRoot/releases/tags/$Tag" -AllowNotFound
    if (-not $taskRelease) {
        $taskRelease = Invoke-StageApi -Method POST -Url "$taskApiRoot/releases" -Body @{
            tag_name=$Tag; target_commitish=$TargetCommit; name=$taskManifest.title;
            body=(Get-Content -LiteralPath $NotesPath -Raw -Encoding utf8); draft=$true; prerelease=$false
        }
    }
    # Reuse only a release bound to exactly the same source snapshot.
    if ($taskRelease.target_commitish -ne $TargetCommit) { throw 'Existing release targets a different source commit' }
    $taskAssets = @(Invoke-StageApi -Method GET -Url "$taskApiRoot/releases/$($taskRelease.id)/assets?per_page=100")
    foreach ($taskFile in $taskFiles) {
        $taskAsset = $taskFile.Asset
        $taskExisting = $taskAssets | Where-Object { $_.name -eq $taskAsset.name } | Select-Object -First 1
        if ($taskExisting) {
            if ($taskExisting.size -ne $taskAsset.bytes -or $taskExisting.digest -ne "sha256:$($taskAsset.sha256)") {
                throw "Existing release asset differs; it will not be overwritten: $($taskAsset.name)"
            }
            continue
        }
        Write-Output "Uploading $($taskAsset.name) ($($taskAsset.bytes) bytes)"
        $taskStream = [IO.File]::OpenRead($taskFile.Path)
        $taskContent = [Net.Http.StreamContent]::new($taskStream)
        $taskContent.Headers.ContentType = [Net.Http.Headers.MediaTypeHeaderValue]::new('application/octet-stream')
        $taskContent.Headers.ContentLength = $taskAsset.bytes
        try {
            $taskUploadBase = $taskRelease.upload_url.Split('{')[0]
            $taskUploadResponse = $taskHttp.PostAsync("${taskUploadBase}?name=$([Uri]::EscapeDataString($taskAsset.name))", $taskContent).GetAwaiter().GetResult()
            try {
                if (-not $taskUploadResponse.IsSuccessStatusCode) { throw "Release upload failed: HTTP $([int]$taskUploadResponse.StatusCode)" }
                $taskUploaded = $taskUploadResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
                if ($taskUploaded.size -ne $taskAsset.bytes -or $taskUploaded.digest -ne "sha256:$($taskAsset.sha256)") {
                    throw "Uploaded asset identity mismatch: $($taskAsset.name)"
                }
                Write-Output "Verified $($taskAsset.name)"
            } finally { $taskUploadResponse.Dispose() }
        } finally { $taskContent.Dispose(); $taskStream.Dispose() }
    }
    $taskAssets = @(Invoke-StageApi -Method GET -Url "$taskApiRoot/releases/$($taskRelease.id)/assets?per_page=100")
    $taskVerified = @()
    foreach ($taskExpected in $taskManifest.assets) {
        $taskActual = $taskAssets | Where-Object { $_.name -eq $taskExpected.name } | Select-Object -First 1
        if (-not $taskActual -or $taskActual.size -ne $taskExpected.bytes -or $taskActual.digest -ne "sha256:$($taskExpected.sha256)") {
            throw "Release verification failed: $($taskExpected.name)"
        }
        $taskVerified += [ordered]@{name=$taskActual.name;bytes=$taskActual.size;sha256=$taskExpected.sha256;download_url=$taskActual.browser_download_url;asset_id=$taskActual.id}
    }
    $taskRelease = Invoke-StageApi -Method PATCH -Url "$taskApiRoot/releases/$($taskRelease.id)" -Body @{draft=$false}
    # Publishing replaces draft-only untagged download URLs with the final tag.
    $taskPublishedAssets = @(Invoke-StageApi -Method GET -Url "$taskApiRoot/releases/$($taskRelease.id)/assets?per_page=100")
    foreach ($taskVerifiedAsset in $taskVerified) {
        $taskPublishedAsset = $taskPublishedAssets | Where-Object { $_.id -eq $taskVerifiedAsset.asset_id } | Select-Object -First 1
        if (-not $taskPublishedAsset -or $taskPublishedAsset.size -ne $taskVerifiedAsset.bytes -or $taskPublishedAsset.digest -ne "sha256:$($taskVerifiedAsset.sha256)") {
            throw 'Published release asset identity changed'
        }
        $taskVerifiedAsset.download_url = $taskPublishedAsset.browser_download_url
    }
    $taskReceipt = [ordered]@{
        status='release_assets_verified'; verified_utc=[DateTime]::UtcNow.ToString('o');
        repository=$taskRepository;tag=$Tag;source_commit=$TargetCommit;release_url=$taskRelease.html_url;
        assets=$taskVerified;additional_training=0;model_API_calls=0;test_access=$false
    }
    $taskReceipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $ReceiptPath -Encoding utf8
    Write-Output "Release verified: $($taskRelease.html_url)"
} finally { $taskHttp.Dispose() }
