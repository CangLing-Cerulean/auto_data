param([string]$DecisionFile)
$ErrorActionPreference = 'Stop'
$stageLocal = 'C:/Users/31227/Desktop/auto_data/research_runs/agent_research/autonomous_memory/20261012_sol_high_r2'
$stageRemote = '/tmp/auto_data_autonomous_memory_20261012_r2/research_runs/agent_research/autonomous_memory/20261012_sol_high_r2'
$serverTarget = 'root@instance-lcpdyopw.zju.smartml.cn'
$sshKey = 'C:/Users/31227/.ssh/smartml_codex'
if ($DecisionFile) {
    $decisionName = Split-Path $DecisionFile -Leaf
    if ($decisionName -notmatch '^decision_[0-9]{4}\.json$') { throw 'Unexpected decision filename' }
    scp -P 20000 -i $sshKey $DecisionFile "${serverTarget}:${stageRemote}/$decisionName"
    if ($LASTEXITCODE -ne 0) { throw 'Decision upload failed' }
    ssh -p 20000 -i $sshKey $serverTarget "/opt/miniconda3/bin/python -B /tmp/auto_data_autonomous_memory_20261012_r2/scripts/run_autonomous_memory.py --folder $stageRemote --decision $stageRemote/$decisionName"
    if ($LASTEXITCODE -ne 0) { throw 'Research action failed' }
}
foreach ($name in @('context.json','progress.json','preregistration.json')) {
    scp -P 20000 -i $sshKey "${serverTarget}:${stageRemote}/$name" "$stageLocal/$name"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $name" }
}
if ($DecisionFile) {
    $number = (Get-Content "$stageLocal/progress.json" -Raw -Encoding utf8 | ConvertFrom-Json).decision_version
    $resultName = 'result_{0:D4}.json' -f [int]$number
    scp -P 20000 -i $sshKey "${serverTarget}:${stageRemote}/$resultName" "$stageLocal/$resultName"
    if ($LASTEXITCODE -ne 0) { throw 'Result download failed' }
}
Write-Output 'Current context and execution records downloaded.'
