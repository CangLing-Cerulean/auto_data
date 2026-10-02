"""Audit completed selection experiments and compare against matched uniform runs."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/selection'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    records={}
    for path in sorted(OUT.glob('batch*.json')):
        for item in json.loads(path.read_text()):
            records[item['id']]=item
    runs=[]
    reference=None
    for item in records.values():
        run=ROOT/'research_runs/results'/item['id']
        if not (run/'result.json').exists():
            continue
        result=json.loads((run/'result.json').read_text())
        identity=json.loads((run/'identity.json').read_text())
        sample=json.loads((run/'sampling_audit.json').read_text())
        audit=json.loads((run/'data_audit.json').read_text())
        if reference is None:
            reference=audit
        if audit!=reference:
            raise ValueError('Fixed data audit changed')
        if identity!=result['identity'] or identity['protocol_sha256']!=digest(OUT/'protocol.json'):
            raise ValueError('Identity mismatch')
        if identity['manifest_sha256']!=digest(ROOT/'data/metropt3_v1/manifest.json'):
            raise ValueError('Dataset mismatch')
        if identity['core_sha256']!=digest(ROOT/'scripts/run_research.py'):
            raise ValueError('Fixed model/data core changed')
        for name,key in [('code_snapshot.py','code_sha256'),('core_snapshot.py','core_sha256')]:
            if digest(run/name)!=identity[key]:
                raise ValueError('Code snapshot mismatch')
        if digest(run/'sampling_weights.npz')!=sample['sampling_weights_sha256'] or digest(OUT/'window_table.npz')!=sample['window_table_sha256']:
            raise ValueError('Sampling artifact mismatch')
        if sample['total_draws']!=768000 or result['steps_completed']!=3000:
            raise ValueError('Training budget differs')
        if json.loads((run/'status.json').read_text())['phase']!='completed':
            raise ValueError('Incomplete status')
        runs.append({**item,'metrics':result['groups']['all'],'groups':result['groups'],
                     'sampling':sample,'best_step':result['best_step'],'elapsed_seconds':result['elapsed_seconds']})
    baseline={r['seed']:r for r in runs if r['recipe']=='uniform'}
    for run in runs:
        if run['seed'] in baseline:
            base=baseline[run['seed']]
            run['relative_reduction_pct']={g:{m:100*(1-v[m]/base['groups'][g][m]) for m in ('NMSE','NMAE')} for g,v in run['groups'].items()}
    aggregates={}
    for recipe in sorted({r['recipe'] for r in runs}):
        selected=[r for r in runs if r['recipe']==recipe]
        a={'seeds':[r['seed'] for r in selected],'metrics':{},'paired_relative_reduction_pct':{}}
        for metric in ('NMSE','NMAE'):
            values=[r['metrics'][metric] for r in selected]
            a['metrics'][metric]={'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None}
            paired=[r for r in selected if r['seed'] in baseline]
            a['paired_relative_reduction_pct'][metric]=[r['relative_reduction_pct']['all'][metric] for r in paired]
            if paired:
                a.setdefault('relative_reduction_of_paired_means_pct',{})[metric]=100*(1-statistics.mean(r['metrics'][metric] for r in paired)/statistics.mean(baseline[r['seed']]['metrics'][metric] for r in paired))
        aggregates[recipe]=a
    report={'completed':runs,'aggregates':aggregates,'completed_training_seconds':sum(r['elapsed_seconds'] for r in runs)}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# 真实训练数据筛选实验汇总','','固定 baseline 预处理；所有指标来自同一完整验证集。','','| ID | policy | seed | retained | quiet draw share | NMSE | NMAE | quiet NMSE | high-change NMSE |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        a=r['sampling']
        lines.append(f'| {r["id"]} | {r["recipe"]} | {r["seed"]} | {a["retained_windows"]} | {a["actual_quiet_draws"]/a["total_draws"]:.3%} | {r["metrics"]["NMSE"]:.8f} | {r["metrics"]["NMAE"]:.8f} | {r["groups"]["low_quartile"]["NMSE"]:.8f} | {r["groups"]["high_input_variation"]["NMSE"]:.8f} |')
    lines+=['','负的相对降低率表示退化；逐月份、逐种子与其他分组详见 summary.json。',f'已完成筛选训练用时 {report["completed_training_seconds"]:.2f} 秒；共享预算还包括此前变换研究及诊断。']
    lines+=['','| policy | seeds | mean NMSE | mean NMAE | paired NMSE reductions |','|---|---|---:|---:|---|']
    for recipe,a in aggregates.items():
        effects=', '.join(f'{v:.3f}%' for v in a['paired_relative_reduction_pct']['NMSE'])
        lines.append(f'| {recipe} | {a["seeds"]} | {a["metrics"]["NMSE"]["mean"]:.8f} | {a["metrics"]["NMAE"]["mean"]:.8f} | {effects} |')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
