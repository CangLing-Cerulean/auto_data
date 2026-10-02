"""Audit local experiment assets and summarize actual completed validation runs."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'research_runs'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    manifest_hash = digest(ROOT / 'data/metropt3_v1/manifest.json')
    protocol_hash = digest(RUNS / 'protocol.json')
    registry = json.loads((RUNS / 'experiments.json').read_text(encoding='utf-8'))
    completed = []
    reference_audit = None
    for item in registry:
        run = RUNS / 'results' / item['id']
        if not (run / 'result.json').exists():
            continue
        result = json.loads((run / 'result.json').read_text())
        identity = json.loads((run / 'identity.json').read_text())
        audit = json.loads((run / 'data_audit.json').read_text())
        if reference_audit is None:
            reference_audit = audit
        if audit != reference_audit:
            raise ValueError(f'{item["id"]}: normalization/window audit changed')
        if identity != result['identity'] or identity['manifest_sha256'] != manifest_hash or identity['protocol_sha256'] != protocol_hash:
            raise ValueError('Experiment identity mismatch')
        if digest(run / 'code_snapshot.py') != identity['code_sha256'] or json.loads((run / 'protocol.json').read_text()) != json.loads((RUNS / 'protocol.json').read_text()):
            raise ValueError('Snapshot checksum mismatch')
        if identity['seed'] != item['seed'] or identity['recipe'] != item['recipe']:
            raise ValueError('Registry mismatch')
        if not all((run / f).exists() for f in ('best.pt', 'latest.pt', 'environment.json', 'history.json')):
            raise ValueError('Missing continuation assets')
        status = json.loads((run / 'status.json').read_text())
        if status['phase'] != 'completed':
            raise ValueError('Result/status inconsistent')
        completed.append({**item, 'metrics': result['groups']['all'], 'groups':result['groups'],
                          'best_step': result['best_step'], 'elapsed_seconds':status['elapsed_seconds']})
    lines = ['# 自动生成的验证实验汇总', '', '所有结果均来自固定验证集；未评估测试集。', '',
             '| ID | recipe | seed | NMSE | NMAE | best step | seconds |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in completed:
        lines.append(f'| {r["id"]} | {r["recipe"]} | {r["seed"]} | {r["metrics"]["NMSE"]:.8f} | {r["metrics"]["NMAE"]:.8f} | {r["best_step"]} | {r["elapsed_seconds"]:.2f} |')
    aggregates = {}
    for recipe in sorted({r['recipe'] for r in completed}):
        selected = [r for r in completed if r['recipe'] == recipe]
        aggregates[recipe] = {'seeds':[r['seed'] for r in selected]}
        for metric in ('NMSE','NMAE'):
            values = [r['metrics'][metric] for r in selected]
            aggregates[recipe][metric] = {'mean':statistics.mean(values), 'sample_std':statistics.stdev(values) if len(values)>1 else None}
    lines.extend(['', '## 按方案汇总', '', '只有配对种子覆盖一致的方案才适合直接比较均值。标准差是种子间样本标准差，不是时间相关样本的置信区间。', '', '| recipe | seeds | NMSE mean ± sd | NMAE mean ± sd |', '|---|---|---|---|'])
    for recipe,a in aggregates.items():
        formatted = []
        for metric in ('NMSE','NMAE'):
            v=a[metric]
            formatted.append(f'{v["mean"]:.8f} ± {v["sample_std"]:.8f}' if v['sample_std'] is not None else f'{v["mean"]:.8f} (single seed)')
        lines.append(f'| {recipe} | {a["seeds"]} | {formatted[0]} | {formatted[1]} |')
    elapsed = sum(r['elapsed_seconds'] for r in completed)
    lines.extend(['', f'已完成训练进程累计用时：{elapsed:.2f} 秒；预算 7200 秒。未完成/失败进程用时需另查远程 status.json。', '', '审计通过：数据与协议指纹一致、固定窗口和标准化审计一致、源码快照指纹一致、恢复资产齐全。'])
    (RUNS / 'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (RUNS / 'summary.json').write_text(json.dumps({'completed':completed,'aggregates':aggregates,'completed_elapsed_seconds':elapsed},indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(completed)} completed experiments audited; {elapsed:.2f} seconds.')


if __name__ == '__main__':
    main()
