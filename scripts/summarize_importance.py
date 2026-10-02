"""Portable paired summaries; importance scores never stand in for NMSE."""
import json
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    folder=ROOT/'research_runs/importance'; rows=[]; pending=[]
    baseline_ids={2026:'S000_uniform_s2026',2027:'S005_uniform_s2027',2028:'S010_uniform_s2028'}
    for batch in sorted(folder.glob('batch*.json')):
        for item in json.loads(batch.read_text()):
            run=ROOT/'research_runs/results'/item['id']
            if not (run/'result.json').exists(): pending.append(item['id']); continue
            result=json.loads((run/'result.json').read_text()); identity=json.loads((run/'identity.json').read_text())
            assert result['identity']==identity and result['steps_completed']==3000 and result['samples_presented']==768000
            for key,name in [('code_sha256','code_snapshot.py'),('core_sha256','core_snapshot.py'),
                ('helper_sha256','helper_snapshot.py'),('sampling_protocol_sha256','sampling_protocol.md'),('protocol_sha256','protocol.json')]:
                assert identity[key]==digest(run/name)
            assert identity['core_sha256']==digest(ROOT/'scripts/run_research.py')
            assert identity['manifest_sha256']==digest(ROOT/'data/metropt3_v1/manifest.json')
            assert identity['protocol_sha256']==digest(ROOT/'research_runs/selection/protocol.json')
            baseline=json.loads((run.parent/baseline_ids[item['seed']]/'result.json').read_text())
            base=baseline['groups']['all']; value=result['groups']['all']
            audit=json.loads((run/'sampling_audit.json').read_text())
            assert audit['window_draws_sha256']==digest(run/'window_draws.npz')
            for phase in audit['phases']: assert phase['asset_sha256']==digest(run/f'phase_{phase["start_step"]:04d}.npz')
            row={**item,'baseline':baseline_ids[item['seed']],'metrics':value,'baseline_metrics':base,
                'NMSE_reduction_pct':100*(1-value['NMSE']/base['NMSE']),
                'NMAE_reduction_pct':100*(1-value['NMAE']/base['NMAE']),
                'best_step':result['best_step'],'scoring_forward_presentations':audit['scoring_forward_presentations'],
                'scoring_seconds':audit['scoring_seconds'],'elapsed_seconds':json.loads((run/'status.json').read_text())['elapsed_seconds'],
                'group_contrasts':{g:{'NMSE':v['NMSE'],'baseline_NMSE':baseline['groups'][g]['NMSE'],
                    'NMSE_reduction_pct':100*(1-v['NMSE']/baseline['groups'][g]['NMSE'])} for g,v in result['groups'].items()},
                'channel_contrasts':[{'channel':name,'NMSE':v,'baseline_NMSE':b,'NMSE_reduction_pct':100*(1-v/b) if b else None}
                    for name,v,b in zip(result['sensor_columns'],result['per_channel_NMSE'],baseline['per_channel_NMSE'])]}
            rows.append(row)
    policies={}
    superseded={r['supersedes'] for r in rows if r.get('supersedes')}
    for row in rows: row['superseded']=row['id'] in superseded
    for name in sorted(set(row['recipe'] for row in rows)):
        selected=[r for r in rows if r['recipe']==name and not r['superseded']]; n=len(selected)
        mean=sum(r['metrics']['NMSE'] for r in selected)/n; base=sum(r['baseline_metrics']['NMSE'] for r in selected)/n
        policies[name]={'seeds':[r['seed'] for r in selected],'mean_NMSE':mean,'paired_baseline_mean_NMSE':base,
            'NMSE_reduction_pct':100*(1-mean/base),'all_seeds_improve':all(r['NMSE_reduction_pct']>0 for r in selected),
            'mean_NMAE':sum(r['metrics']['NMAE'] for r in selected)/n,
            'mean_elapsed_seconds':sum(r['elapsed_seconds'] for r in selected)/n}
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
    output={'runs':rows,'policies':policies,'pending':pending,'budget_used_seconds':used,'budget_remaining_seconds':7200-used}
    write(folder/'summary.json',output)
    lines=['# 动态重要性采样真实结果','','每项与同种子全量均匀基线配对；完整原验证集，NMSE主指标。','',
        '| ID | NMSE | 相对配对基线改善 | NMAE | 最佳步 | 总秒数（含评分） | 状态 |', '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        label='被修正替代，非独立种子' if row['superseded'] else '当前单种子筛选，未确认'
        lines.append(f'| {row["id"]} | {row["metrics"]["NMSE"]:.8f} | {row["NMSE_reduction_pct"]:.4f}% | {row["metrics"]["NMAE"]:.8f} | {row["best_step"]} | {row["elapsed_seconds"]:.2f} | {label} |')
    (folder/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'policies':policies,'pending':pending,'remaining':7200-used},indent=2))


if __name__=='__main__': main()
