"""Report the complete audited FTR1 forecasts without reading full child labels."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261014_file_transfer_r1'
METHODS = ('M1', 'RAW', 'SOURCE', 'SHORT')
POLICIES = ('PARENT', 'SIZE_DROP', 'LOSS_DROP')


def read(name):
    return json.loads((STAGE / name).read_text(encoding='utf-8-sig'))


def digest(name):
    return hashlib.sha256((STAGE / name).read_bytes()).hexdigest()


def save_csv(name, rows):
    with (STAGE / name).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    plan, pred, audit = read('plan.json'), read('predictions.json'), read('forecast_audit.json')
    assert pred['status'] == 'completed' and audit['status'] == 'PASS'
    assert pred['plan_sha256'] == audit['plan_sha256'] == digest('plan.json')
    assert audit['predictions_sha256'] == digest('predictions.json')
    assert pred['eligible_to_label'] == audit['eligible_to_label'] and not pred['full_candidate_responses_observed']
    assert pred['cost_gate'] == audit['cost_gate']
    assert pred['identical_recommendation_baselines'] == audit['identical_recommendation_baselines']
    assert len(pred['records']) == 69 and len(pred['costs']) == 9
    records = {(r['task'], r['action'], r['seed']): r for r in pred['records']}
    assert len(records) == 69
    seed_rows, cells = [], []
    for task in plan['tasks']:
        for action in task['actions']:
            cell = dict(task=task['id'], action=action['id'], removed_source_file=action['removed_source_file'])
            for seed in plan['seeds']:
                record = records[(task['id'], action['id'], seed)]
                row = dict(cell, seed=seed)
                for method in METHODS:
                    row[method + '_predicted_gain_percent'] = record['predictions'][method]['gain_percent']
                    row[method + '_selected_step'] = record['predictions'][method]['selected_step']
                seed_rows.append(row)
            for method in METHODS:
                values = [records[(task['id'], action['id'], seed)]['predictions'][method]['gain_percent'] for seed in plan['seeds']]
                mean = pred['recommendations'][task['id']][method]['means'][action['id']]
                assert math.isclose(mean, statistics.mean(values), rel_tol=1e-12, abs_tol=1e-12)
                cell[method + '_mean_predicted_gain_percent'] = mean
                cell[method + '_min_seed_predicted_gain_percent'] = min(values)
                cell[method + '_max_seed_predicted_gain_percent'] = max(values)
            cells.append(cell)
    assert len(cells) == 23 and len(seed_rows) == 69
    save_csv('forecast_seed_records.csv', seed_rows)
    save_csv('forecast_cell_records.csv', cells)
    comparison = []
    for method in METHODS + POLICIES:
        comparison.append(dict(method=method,
            warm_seconds=sum(c['warm'][method] for c in pred['costs']),
            cold_required_computation_seconds=sum(c['cold_required_computation'][method] for c in pred['costs']),
            cold_registered_execution_seconds=sum(c['cold_registered_execution'][method] for c in pred['costs'])))
    save_csv('forecast_method_costs.csv', comparison)
    eligible = pred['eligible_to_label']
    agreement = pred['identical_recommendation_baselines']
    lines = ['# 完整候选预测与标签前验收', '',
        '全部69个代理子、69个SHORT及SOURCE/便宜策略预测完成，独立审计通过。新增完整实验训练0次，累计418；完整子标签仍未启动。', '',
        ('标签前门槛通过；完成预测与归档同步后，才可按原协议运行全部69个完整子。' if eligible else
         '标签前门槛未通过，本批停止完整子标签。响应预测精度和完整重训的实际收益仍未知。'), '',
        f'- M1实际选择成本≤SHORT：{pred["cost_gate"]}。',
        f'- 与M1在三个任务上推荐完全相同的基线：{", ".join(agreement) if agreement else "无"}。',
        '- 若出现全任务同推荐，二者在任何真实标签下的平均推荐收益均相等，原+0.10pp决策增量门槛不能成立；这不证明响应数值精度相同。', '',
        '## 已锁定推荐', '', '| 任务 | M1 | RAW | SOURCE | SHORT | PARENT | SIZE_DROP | LOSS_DROP |',
        '|---|---|---|---|---|---|---|---|']
    for task in plan['tasks']:
        lines.append('| ' + task['id'] + ' | ' + ' | '.join(pred['recommendations'][task['id']][m]['action'] for m in METHODS + POLICIES) + ' |')
    lines += ['', '## 全部动作的三种子均值预测', '',
        '下列数值是相对各自真实父模型的预测改善百分比，全部23格均保留；不是真实完整子改善。逐种子、范围、选点和文件映射见同目录CSV。', '',
        '| 任务 | 动作 | M1 | RAW | SOURCE | SHORT |', '|---|---|---:|---:|---:|---:|']
    for cell in cells:
        lines.append('| ' + cell['task'] + ' | ' + cell['action'] + ' | ' + ' | '.join(f'{cell[m + "_mean_predicted_gain_percent"]:.6f}' for m in METHODS) + ' |')
    lines += ['', '## 选择成本', '',
        '下表汇总全部9个任务/种子组。方法间共用资产按各方法单独支付口径计入，不能将各行相加当作本次实际总消耗。冷启动值包含分摊，不是单独启动各方法的实测。', '',
        '| 方法 | 增量秒数 | 必需计算冷启动秒数 | 登记执行冷启动秒数 |', '|---|---:|---:|---:|']
    for row in comparison:
        lines.append(f'| {row["method"]} | {row["warm_seconds"]:.6f} | {row["cold_required_computation_seconds"]:.6f} | {row["cold_registered_execution_seconds"]:.6f} |')
    lines += ['', f'本次预测阶段实际耗时{pred["phase_seconds"]:.6f}秒，独立审计{audit["seconds"]:.6f}秒，另有前序父阶段成本。审计重放单独记账；人工管理与传输时间未独立计量。', '',
        '## 解释边界', '',
        '这是已接触SKAB基础train内部的文件隔离开发问题。三任务共享文件，固定父标准化仍含被删文件的统计信息；不能解释为独立跨源验证、完整数据遗忘或最终测试。测试、基础validation和旧D保持隔离。', '',
        '本报告读取已验收的全部预测和成本，不读取完整子标签。后续是否推进严格使用原冻结门槛，不追加文件组、种子、动作或预算步数。', '']
    with (STAGE / 'FORECAST_REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    summary = dict(status='audited_forecast_report', predictions_sha256=digest('predictions.json'),
        forecast_audit_sha256=digest('forecast_audit.json'), cells=23, seed_cells=69,
        cost_gate=pred['cost_gate'], identical_recommendation_baselines=agreement, eligible_to_label=eligible,
        new_complete_experimental_trainings=0, cumulative_complete_experimental_trainings=418,
        true_child_response_accuracy=None, true_decision_gain=None)
    with (STAGE / 'forecast_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2); stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
