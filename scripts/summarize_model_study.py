"""Build the final report from saved validation and frozen test results only."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/model_exploration'


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    progress = read(FOLDER / 'progress.json'); test = read(FOLDER / 'test_result.json')
    validation = read(FOLDER / 'validation_cpu.json'); freeze = read(FOLDER / 'freeze.json')
    assert digest(FOLDER / 'freeze.json') == test['freeze_sha256']
    assert digest(ROOT / 'scripts/evaluate_frozen_models.py') == freeze['evaluator_sha256']
    receipts = read(FOLDER / 'download_receipt.json')
    for relative, expected in receipts.items(): assert digest(ROOT / relative) == expected, relative
    used = sum(read(p).get('elapsed_seconds', 0) for p in (ROOT / 'research_runs/results').glob('*/status.json'))
    rows = []
    for r in progress['rows']:
        folder = ROOT / 'research_runs/results' / r['run_id']
        result = read(folder / 'result.json'); audit = read(folder / 'data_audit.json')
        with np.load(folder / 'windows.npz') as z, np.load(ROOT / 'research_runs/results' / r['baseline'] / 'sampling_weights.npz') as baseline:
            assert np.array_equal(z['draw_count'], baseline['draw_count'])
            assert z['draw_count'].sum() == 768000
            assert len(z['train_start']) == 1053433 and len(z['validation_start']) == 300535
        rows.append({**r, 'best_step': result['best_step'], 'parameter_count': result['parameter_count'],
            'budget_seconds': read(folder / 'status.json')['elapsed_seconds'],
            'raw_unit_MSE': result['raw_unit_MSE'], 'source_windows_and_sampling_verified': True})
    summary = {**progress, 'rows': rows, 'budget_used_seconds': used, 'remaining_seconds': 7200 - used,
        'round_used_seconds': used - progress['round_start_seconds'], 'test': test,
        'training_seconds': sum(r['budget_seconds'] for r in rows),
        'export_diagnostic_seconds': read(ROOT / 'research_runs/results/A015_model_export/status.json')['elapsed_seconds'],
        'validation_cpu_seconds': validation['cpu_wall_seconds'], 'test_cpu_seconds': test['cpu_wall_seconds'],
        'failed_validation_attempt_cpu_wall_upper_bound_seconds': 5.2614011,
        'cpu_budget_note': 'Local NumPy evaluation consumes no GPU; times recorded separately. Initial validation attempt used an overly strict absolute-only probe check; full metric tolerance remained 1e-5 throughout.',
        'test_accessed': True, 'model_adopted_for_MSE': True, 'sampling_policy': 'uniform',
        'reference_seed': 2026, 'reference_checkpoint': 'research_runs/results/M001_residual_mlp_s2026/best.pt',
        'reference_seed_reason': 'Original preset seed 2026; not the best test seed.',
        'downloaded_assets_verified': len(receipts),
        'remote_metadata_note': 'New training and exports exist on research server. Final freeze, CPU validation, final test, and updated handoff/state are local only. Previously rejected metadata upload was not retried.'}
    baseline_channel = np.mean([r['per_channel_NMSE'] for r in test['rows'] if r['kind'] == 'dlinear'], axis=0)
    candidate_channel = np.mean([r['per_channel_NMSE'] for r in test['rows'] if r['kind'] != 'dlinear'], axis=0)
    summary['test_per_channel_means'] = [{'sensor': sensor, 'baseline_NMSE': float(a), 'candidate_NMSE': float(b),
        'NMSE_delta': float(b - a)} for sensor, a, b in zip(test['sensor_columns'], baseline_channel, candidate_channel)]
    save(FOLDER / 'summary.json', summary)
    lines = ['# 跨传感器模型：冻结后的最终测试结果', '',
        '残差MLP的三种子平均测试NMSE从0.26860836降至0.18998724，降低29.2698%；三个种子均改善。测试NMAE从0.19875750升至0.20028621，退化0.7691%。结论基于一次冻结后的最终测试，不是把验证收益代称为测试收益。', '',
        '本轮用户明确转向其他探索，研究变量扩为模型。全部合法训练窗口仍保留并均匀采样；没有证明数据筛选有效。协议、模型选择与最终冻结见[PROTOCOL.md](PROTOCOL.md)、[progress.json](progress.json)、[freeze.json](freeze.json)。', '',
        '## 验证与选择', '',
        '| 模型 / 种子 | 参数量 | 最佳步数 | 验证NMSE | 较配对DLinear降低 | 验证NMAE降低 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['kind']} / {r['seed']} | {r['parameter_count']} | {r['best_step']} | {r['NMSE']:.8f} | {r['NMSE_reduction_pct']:.4f}% | {r['NMAE_reduction_pct']:.4f}% |")
    lines += ['', f"赢家三种子平均验证NMSE={progress['mean_NMSE']:.8f}，降低{progress['mean_NMSE_reduction_pct']:.4f}%；NMAE降低{progress['mean_NMAE_reduction_pct']:.4f}%。两个固定架构先比较2026，只有预登记赢家MLP运行2027/2028；没有学习率/宽度/种子扫描。", '',
        '## 最终测试', '',
        '| 种子 | DLinear NMSE | MLP NMSE | NMSE降低 | NMAE降低 |',
        '|---|---:|---:|---:|---:|']
    for pair in test['paired_comparisons']:
        b = next(r for r in test['rows'] if r['run_id'] == pair['baseline'])
        c = next(r for r in test['rows'] if r['run_id'] == pair['candidate'])
        lines.append(f"| {pair['seed']} | {b['NMSE']:.8f} | {c['NMSE']:.8f} | {pair['NMSE_reduction_pct']:.4f}% | {pair['NMAE_reduction_pct']:.4f}% |")
    means = test['means']; reductions = test['mean_reduction_pct']
    lines += [f"| 三种子均值 | {means['baseline']['NMSE']:.8f} | {means['candidate']['NMSE']:.8f} | {reductions['NMSE']:.4f}% | {reductions['NMAE']:.4f}% |", '',
        '降低为负代表退化。主指标NMSE是以训练总体标准差固定标准化后的MSE，平均全部合法窗口、8个预测位置及15通道。三种子均值是三个独立训练模型的指标平均，没有做预测集成或按测试选择最好种子。', '',
        f"补充原单位MSE：{means['baseline']['raw_unit_MSE']:.8f}→{means['candidate']['raw_unit_MSE']:.8f}，降低{reductions['raw_unit_MSE']:.4f}%。它按每通道NMSE×训练scale²还原后平均；不同传感器量纲混合，故只作补充，主指标定义未改变。", '',
        f"最终测试合法窗口数：{test['valid_windows']}。冻结时间：{freeze['created_utc']}；评估完成：{test['completed_utc']}。冻结文件SHA256：`{test['freeze_sha256']}`。测试评估在本地CPU进行，测试CSV未上传研究服务器。", '',
        '## 变化、解释与限制', '',
        'DLinear原模型逐通道独立，7920参数；MLP展平32×15历史，480→128→120，ReLU，并加最后观测作为残差跳接，共77048参数。训练损失、尺度、3000更新、batch256与768000次呈现保持一致。实际抽样次数逐窗口与各自S000/S005/S010精确一致。', '',
        '跨通道信息、非线性、残差路径和更大容量共同构成本次模型变化。线性跨通道模型也有明显验证收益，但MAE退化；不能据这两个候选单独量化每个结构要素的因果贡献，也不是等参数量或等FLOPs对照。', '',
        '三个种子的MSE收益同向，证据仍限于本数据集和这一个冻结时间划分。MAE小幅平均退化，2026/2028分别退化4.0919%/0.5307%，2027改善2.2273%；不声称所有误差指标均提升。以下通道表仅作结果描述，禁止据测试误差再选通道或调参。', '',
        '| 通道 | DLinear三种子平均NMSE | MLP三种子平均NMSE | MLP−基线 |',
        '|---|---:|---:|---:|']
    for r in summary['test_per_channel_means']:
        lines.append(f"| {r['sensor']} | {r['baseline_NMSE']:.8f} | {r['candidate_NMSE']:.8f} | {r['NMSE_delta']:+.8f} |")
    lines += ['', '## 复现与资产', '',
        '`scripts/forecast_models.py`定义模型；`run_model_exploration.py`与`run_model_study.py`保存训练流程；`export_model_study.py`导出部署权重；`evaluate_frozen_models.py`执行验证核对、不可覆盖冻结和最终测试。最后脚本刻意拒绝覆盖已执行测试。常规接续直接读取test_result.json，不能把重新跑测试当作下一轮开发入口。', '',
        '本地NumPy复算六个完整验证集指标，与原PyTorch绝对误差均小于1e-5；抽样预测使用atol=rtol=1e-5。初次核对在DLinear的绝对预测误差1.335e-5处因仅设绝对容差而中断，随后改为常规绝对/相对联合浮点比较；完整验证指标门槛始终不变，所有修正发生在冻结及测试前，没有修改模型。', '',
        f"新增4次主模型训练用时{summary['training_seconds']:.6f}秒，导出诊断计费{summary['export_diagnostic_seconds']:.6f}秒；本轮共{summary['round_used_seconds']:.6f}秒，原7200秒累计{used:.6f}秒，剩{7200-used:.6f}秒。训练累计87次（84主预测+3辅助）。CPU验证{validation['cpu_wall_seconds']:.4f}秒、最终测试{test['cpu_wall_seconds']:.4f}秒另记，首次中断CPU核对约5.26秒；均无GPU使用，没有新增预算。", '',
        f"服务器下载的{len(receipts)}项新训练/导出资产已核验；最佳及最终模型、优化器检查点、原始窗口映射、实际抽样次数、配置和源码快照保留。固定seed2026入口为`results/M001_residual_mlp_s2026/best.pt`，另两种子结果全部保留；没有因2028测试更好而改选默认种子。", '',
        '当前测试集已产生最终结果，后续不得据此调参并继续宣称它是未见数据。没有待运行训练队列。之前被拒绝的元数据回写未重试：服务器有新模型及导出原件，本地保存最终freeze/CPU验证/test/report和最新接续状态；远程总state仍可能为65次旧快照，GitHub Release仍为55次快照。', '',
        '方法依据：[Google Research TiDE研究](https://research.google/pubs/long-horizon-forecasting-with-tide-time-series-dense-encoder/)及本项目R001旧跨传感器证据；本模型不是TiDE论文完整复现。']
    (FOLDER / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(FOLDER / 'asset_verification.json', {'download_receipt_sha256': digest(FOLDER / 'download_receipt.json'),
        'downloaded_assets_verified': len(receipts), 'run_ids': [r['run_id'] for r in rows],
        'source_sampling_verified': True, 'freeze_sha256': test['freeze_sha256'],
        'test_result_sha256': digest(FOLDER / 'test_result.json'), 'summary_sha256': digest(FOLDER / 'summary.json'),
        'report_sha256': digest(FOLDER / 'REPORT.md'), 'budget_used_seconds': used,
        'local_only_final_evaluation': True})
    print(json.dumps({'test_mean_reduction_pct': reductions, 'budget_used_seconds': used,
                      'round_used_seconds': summary['round_used_seconds'], 'test_windows': test['valid_windows']}))


if __name__ == '__main__': main()
