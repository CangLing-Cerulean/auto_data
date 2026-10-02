"""Portable paired results for representation, separate from selection claims."""
import hashlib
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    folder=ROOT/'research_runs/representation'; records=[]
    for path in sorted(folder.glob('batch*.json')): records.extend(json.loads(path.read_text()))
    rows=[]; pending=[]
    for item in records:
        run=ROOT/'research_runs/results'/item['id']
        if not (run/'result.json').exists(): pending.append(item['id']); continue
        result=json.loads((run/'result.json').read_text()); identity=json.loads((run/'identity.json').read_text())
        assert result['identity']==identity
        assert result['steps_completed']==3000 and result['samples_presented']==768000
        assert identity['protocol_sha256']==digest(ROOT/'research_runs/selection/protocol.json')
        assert identity['core_sha256']==digest(ROOT/'scripts/run_research.py')
        for key,name in [('code_sha256','code_snapshot.py'),('representation_helper_sha256','representation_helper.py'),
            ('representation_protocol_sha256','representation_protocol.md')]: assert identity[key]==digest(run/name)
        audit=json.loads((run/'rotation_audit.json').read_text()); assert audit['asset_sha256']==digest(run/'rotation.npz')
        baseline={2026:'S000_uniform_s2026',2027:'S005_uniform_s2027',2028:'S010_uniform_s2028'}[item['seed']]
        base=json.loads((run.parent/baseline/'result.json').read_text())
        row={**item,'baseline':baseline,'metrics':result['groups']['all'],'baseline_metrics':base['groups']['all'],
            'best_step':result['best_step'],'elapsed_seconds':json.loads((run/'status.json').read_text())['elapsed_seconds'],
            'group_contrasts':{name:{'NMSE':v['NMSE'],'baseline_NMSE':base['groups'][name]['NMSE'],
                'NMSE_reduction_pct':100*(1-v['NMSE']/base['groups'][name]['NMSE'])} for name,v in result['groups'].items()},
            'channel_contrasts':[{'channel':name,'NMSE':v,'baseline_NMSE':b,'NMSE_reduction_pct':100*(1-v/b) if b else None}
                for name,v,b in zip(result['sensor_columns'],result['per_channel_NMSE'],base['per_channel_NMSE'])]}
        for metric in ['NMSE','NMAE']: row[metric+'_reduction_pct']=100*(1-row['metrics'][metric]/row['baseline_metrics'][metric])
        rows.append(row)
    summary={}
    if rows:
        summary={'seeds':[r['seed'] for r in rows],'all_seeds_NMSE_improve':all(r['NMSE_reduction_pct']>0 for r in rows)}
        for metric in ['NMSE','NMAE']:
            values=[r['metrics'][metric] for r in rows]; base=[r['baseline_metrics'][metric] for r in rows]
            summary[metric]={'mean':statistics.mean(values),'baseline_mean':statistics.mean(base),
                'relative_reduction_pct':100*(1-statistics.mean(values)/statistics.mean(base)),
                'sample_std':statistics.stdev(values) if len(values)>1 else None}
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
    output={'scope':'reversible input-output representation, NOT data-selection improvement','runs':rows,
        'summary':summary,'pending':pending,'budget_used_seconds':used,'budget_remaining_seconds':7200-used}
    (folder/'summary.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    lines=['# 可逆表示实验结果','','相同DLinear骨干、相同训练步数和采样顺序；有效物理通道函数类因数据表示而变化。','',
        '| ID | 完整验证NMSE | 配对改善 | NMAE | 最佳步 |','|---|---:|---:|---:|---:|']
    for r in rows: lines.append(f'| {r["id"]} | {r["metrics"]["NMSE"]:.8f} | {r["NMSE_reduction_pct"]:.4f}% | {r["metrics"]["NMAE"]:.8f} | {r["best_step"]} |')
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'summary':summary,'pending':pending,'remaining_seconds':7200-used},indent=2))


if __name__=='__main__': main()
