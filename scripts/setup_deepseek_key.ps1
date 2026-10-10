$ErrorActionPreference = 'Stop'
$Host.UI.RawUI.WindowTitle = 'DeepSeek API key setup'
Write-Host 'DeepSeek API key setup'
Write-Host 'Paste your API key here and press Enter. Input is hidden.'
Write-Host 'The key is saved to your Windows user environment, not to project files.'
$taskApiSecret = Read-Host 'DEEPSEEK_API_KEY' -AsSecureString
$taskApiPlain = $null
try {
    $taskApiPlain = [System.Net.NetworkCredential]::new('', $taskApiSecret).Password
    if ([string]::IsNullOrWhiteSpace($taskApiPlain)) {
        throw 'The key was empty. Nothing was saved.'
    }
    [Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', $taskApiPlain.Trim(), 'User')
    Write-Host 'Key saved. Return to Codex and reply: configured.'
}
finally {
    $taskApiPlain = $null
    $taskApiSecret.Dispose()
    Remove-Variable taskApiPlain, taskApiSecret -ErrorAction SilentlyContinue
}
