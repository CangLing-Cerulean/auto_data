param(
    [Parameter(Mandatory)][ValidateSet('2','4','8')][string]$Node,
    [Parameter(Mandatory)][string]$Endpoint,
    [Parameter(Mandatory)][string]$IdentityFile,
    [Parameter(Mandatory)][int]$Port
)
$ErrorActionPreference = 'Stop'
$stage = Join-Path (Split-Path -Parent $PSScriptRoot) 'research_runs/agent_research/intervention_diagnosis/20261012_skab_pilot'
$serverRoot = '/tmp/auto_data_intervention_skab_20261012_r1'
$lock = Get-Content -LiteralPath "$stage/prediction_lock.json" -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($entry in $lock.files) {
    if ((Get-FileHash -LiteralPath "$stage/$($entry.name)" -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) { throw 'Prediction lock changed' }
}
$actions = switch ($Node) { '2' { @('W_UP','W_DOWN') }; '4' { @('S_UP','S_DOWN') }; '8' { @('CW_UP','CW_DOWN','CS_UP','CS_DOWN') } }
if ($Node -ne '2') {
    $prior = if ($Node -eq '4') { '2' } else { '4' }
    foreach ($variant in @('full','strong_prompt')) { if (-not (Test-Path -LiteralPath "$stage/${variant}_node${prior}.json")) { throw 'Prior real decision node missing' } }
}
foreach ($action in $actions) {
    & ssh -i $IdentityFile -p $Port -o BatchMode=yes -o ConnectTimeout=15 $Endpoint "env CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/miniconda3/bin/python -B $serverRoot/run_intervention_skab_pilot.py train --root $serverRoot --action $action --seed 2026101301"
    if ($LASTEXITCODE -ne 0) { throw "Run failed: $action (do not repeat ID)" }
    & scp -i $IdentityFile -P $Port -o BatchMode=yes "${Endpoint}:$serverRoot/runs/${action}_s2026101301/result.json" "$stage/${action}_result.json"
    if ($LASTEXITCODE -ne 0) { throw "Result download failed: $action" }
}
$revealed = switch ($Node) { '2' { @('W_UP','W_DOWN') }; '4' { @('W_UP','W_DOWN','S_UP','S_DOWN') }; '8' { @('W_UP','W_DOWN','S_UP','S_DOWN','CW_UP','CW_DOWN','CS_UP','CS_DOWN') } }
$baseline = Get-Content -LiteralPath "$stage/baseline_result.json" -Raw -Encoding utf8 | ConvertFrom-Json
$responses = foreach ($action in $revealed) {
    $result = Get-Content -LiteralPath "$stage/${action}_result.json" -Raw -Encoding utf8 | ConvertFrom-Json
    $gain = [ordered]@{}
    foreach ($scope in @('global','target','complement')) {
        $gain[$scope] = @{NMSE_gain_pct=100*(1-$result.metrics.$scope.NMSE/$baseline.metrics.$scope.NMSE);NMAE_gain_pct=100*(1-$result.metrics.$scope.NMAE/$baseline.metrics.$scope.NMAE);signed_bias_change=$result.metrics.$scope.signed_bias-$baseline.metrics.$scope.signed_bias}
    }
    @{action=$action;result=$result;paired_response=$gain}
}
[ordered]@{node=[int]$Node;created_utc=[DateTime]::UtcNow.ToString('o');baseline=$baseline;responses=@($responses);remaining_actions_unrevealed=8-[int]$Node;test_access=$false} | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath "$stage/common_feedback_node${Node}.json" -Encoding utf8
