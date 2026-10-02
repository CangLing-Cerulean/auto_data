param(
    [ValidateSet('start', 'status', 'fetch')][string]$Action = 'status',
    [string]$RunId = 'E000_baseline_s2026',
    [string]$Recipe = 'baseline',
    [int]$Seed = 2026
)
$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[A-Za-z0-9_]+$' -or $Recipe -notmatch '^[a-z_]+$') { throw 'Invalid run ID or recipe' }
$projectRoot = Split-Path $PSScriptRoot -Parent
$remoteRoot = '/root/data/auto_data_research/metropt3_v1/session_20260926'
$sshArgs = @('-p', '20000', '-i', 'C:/Users/31227/.ssh/smartml_codex', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', 'root@instance-lcpdyopw.zju.smartml.cn')
if ($Action -eq 'fetch') {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'research_runs/results') | Out-Null
    & scp -P 20000 -i C:/Users/31227/.ssh/smartml_codex -o IdentitiesOnly=yes -o BatchMode=yes -r "root@instance-lcpdyopw.zju.smartml.cn:${remoteRoot}/research_runs/results/$RunId" (Join-Path $projectRoot 'research_runs/results/')
    if ($LASTEXITCODE -ne 0) { throw 'Result fetch failed' }
    return
}
$remoteScript = @'
import json, os, subprocess, time
from pathlib import Path
root = Path('/root/data/auto_data_research/metropt3_v1/session_20260926')
run_id = '__RUN_ID__'
run = root / 'research_runs/results' / run_id
action = '__ACTION__'
if action == 'start':
 run.mkdir(parents=True, exist_ok=True)
 if (run/'result.json').exists():
  raise RuntimeError('Run completed; cannot restart')
 if (run/'pid.json').exists():
  old = json.loads((run/'pid.json').read_text())
  proc = Path('/proc') / str(old['pid']) / 'cmdline'
  if proc.exists() and run_id.encode() in proc.read_bytes():
   raise RuntimeError('Run already active')
 consumed = sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (root/'research_runs/results').glob('*/status.json'))
 protocol = json.loads((root/'research_runs/protocol.json').read_text())
 remaining = int(protocol['session_gpu_wall_budget_seconds']-consumed)
 if remaining <= 0:
  raise RuntimeError('Session budget exhausted')
 command = ['timeout','--signal=TERM',str(remaining),'/opt/miniconda3/bin/python','-u',str(root/'scripts/run_research.py'),'--run-id',run_id,'--recipe','__RECIPE__','--seed','__SEED__']
 with (run/'process.log').open('ab') as log:
  child = subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (run/'pid.json').write_text(json.dumps({'pid':child.pid,'started_unix':time.time(),'timeout_seconds':remaining}))
 print('Started',run_id,'PID',child.pid,'time limit',remaining)
else:
 for name in ['pid.json','status.json']:
  if (run/name).exists(): print(name,(run/name).read_text())
 if (run/'result.json').exists():
  result=json.loads((run/'result.json').read_text())
  print('RESULT',json.dumps({'best_step':result['best_step'],'groups':result['groups']}))
 if (run/'process.log').exists():
  print('LOG TAIL', '\n'.join((run/'process.log').read_text()[-6000:].splitlines()[-8:]))
'@
$remoteScript = $remoteScript.Replace('__RUN_ID__', $RunId).Replace('__ACTION__', $Action).Replace('__RECIPE__', $Recipe).Replace('__SEED__', [string]$Seed)
$remoteScript | & ssh @sshArgs /opt/miniconda3/bin/python -
if ($LASTEXITCODE -ne 0) { throw "Remote $Action failed" }
