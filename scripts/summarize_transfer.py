"""Temporal diagnostic ledger, separate from external validation scores."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle,'sha256').hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    folder=ROOT/'research_runs/transfer'; records=[]
    for batch in sorted(folder.glob('batch*.json')): records.extend(json.loads(batch.read_text()))
    completed=[]; active=[]
    for item in records:
        run=ROOT/'research_runs/results'/item['id']
        if not (run/'result.json').exists():
            if (run/'status.json').exists(): active.append({'id':item['id'],**json.loads((run/'status.json').read_text())})
            continue
        result=json.loads((run/'result.json').read_text()); identity=json.loads((run/'identity.json').read_text())
        assert identity==result['identity']
        assert identity['code_sha256']==digest(run/'code_snapshot.py')
        assert identity['core_sha256']==digest(ROOT/'scripts/run_research.py')==digest(run/'core_snapshot.py')
        assert identity['manifest_sha256']==digest(ROOT/'data/metropt3_v1/manifest.json')
        assert identity['protocol_sha256']==digest(ROOT/'research_runs/selection/protocol.json')
        assert identity['diagnostic_protocol_sha256']==digest(run/'diagnostic_protocol.md')
        if identity.get('thin','none')!='none':
            assert identity['thinning_protocol_sha256']==digest(run/'thinning_protocol.md')
        if identity.get('state_balance'):
            assert identity['state_protocol_sha256']==digest(run/'state_protocol.md')
            assert identity['state_helper_sha256']==digest(run/'state_helper.py')
        assert result['windows_sha256']==digest(run/'windows.npz')
        assert result['steps_completed']==3000 and result['samples_presented']==768000
        assert json.loads((run/'status.json').read_text())['phase']=='completed'
        completed.append({**item,'future':result['future']['groups']['all'],
            'fitted':result['prefix']['groups']['fitted_members'],'all_prefix':result['prefix']['groups']['all'],
            'future_per_channel_NMSE':result['future']['per_channel_NMSE'],
            'data_audit':json.loads((run/'data_audit.json').read_text())})
    baselines={r['origin']:r for r in completed if r['drop_block']==-1 and r.get('thin','none')=='none' and not r.get('state_balance')}
    for r in completed:
        base=baselines[r['origin']]
        for key in ('mean','scale','train_raw_rows','probe_raw_rows','source_blocks'):
            assert r['data_audit'][key]==base['data_audit'][key]
        r['future_NMSE_reduction_pct']=100*(1-r['future']['NMSE']/base['future']['NMSE'])
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
    report={'registered':len(records),'completed':completed,'active':active,'budget_used_seconds':used,'budget_remaining_seconds':7200-used}
    write_json(folder/'progress.json',report)
    lines=['# 训练内部时间迁移诊断','','不同原点使用各自前缀尺度，只在同一原点配对比较；不与正式验证得分混排。','',
        '| ID | origin | removed block | common prefix NMSE | fitted NMSE | future NMSE | future reduction |','|---|---:|---:|---:|---:|---:|---:|']
    for r in completed: lines.append(f'| {r["id"]} | {r["origin"]} | {r["drop_block"]} | {r["all_prefix"]["NMSE"]:.8f} | {r["fitted"]["NMSE"]:.8f} | {r["future"]["NMSE"]:.8f} | {r["future_NMSE_reduction_pct"]:.3f}% |')
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'completed':[{k:r[k] for k in ('id','future','all_prefix','fitted','future_NMSE_reduction_pct')} for r in completed],
        'active':active,'remaining_seconds':7200-used},indent=2))


if __name__=='__main__': main()
