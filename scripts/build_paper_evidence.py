"""Rebuild manuscript evidence tables and a descriptive figure; never run training."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research_runs/agent_research/intervention_diagnosis'
OUT = BASE / '20261015_paper_evidence_r1'


def read(stage, name):
    return json.loads((BASE / stage / name).read_text(encoding='utf-8-sig'))


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    m1 = read('20261014_proxy_selection_recovery_r1', 'final_audit.json')
    weighted = read('20261014_weighted_response_r1', 'final_audit.json')
    fd = read('20261015_n1_path_engineering_r1', 'fd_summary.json')
    diag = read('20261015_n1_error_decomposition_r1', 'summary.json')
    assert m1['status'] == weighted['status'] == diag['status'] == 'PASS'
    assert fd['numerical_gate_pass'] is False
    advantage = m1['mean_recommendation_gain']['M1'] - m1['mean_recommendation_gain']['SHORT']
    assert abs(advantage - m1['recommendation_gains']['M1'][0] / 3) < 1e-12
    assert fd['derivative_passed'] == 64 and fd['derivative_total'] == 96
    assert fd['two_h_passed'] == 32 and fd['two_h_total'] == 48
    diagnosed = {}
    for kind in ('derivative', 'two_h'):
        rows = [r for r in diag['summaries'] if r['kind'] == kind]
        failures = sum(r['original_failures'] for r in rows)
        bound_failures = sum(r['failures_evaluation_alone_cannot_explain'] for r in rows)
        assert failures == bound_failures
        diagnosed[kind] = {'original_failures': failures, 'lower_bound_failures': bound_failures}
    metrics = {
        'main_MAE_pp': m1['prediction_MAE_pp'],
        'main_mean_selected_gain_percent': m1['mean_recommendation_gain'],
        'main_task_selected_gain_percent': m1['recommendation_gains'],
        'main_cold_accounting_seconds': m1['batch_cold_total_seconds'],
        'M1_minus_SHORT_pp': advantage,
        'weighted_MAE_pp': {k: weighted[k] for k in ('M1_S_MAE_pp', 'M1_W_MAE_pp', 'M1_joint_MAE_pp', 'mean_only_joint_MAE_lower_bound_pp')},
        'N1_gate_pass': fd['numerical_gate_pass'],
        'endpoint_diagnosis': diagnosed,
        'new_training': 0, 'cumulative_full_trainings': 434,
        'submission_ready': False,
    }
    write('metrics.json', metrics)
    claims = [
        ('C01', 'M1 has a local decision advantage over SHORT in the frozen Hydraulic pool', 'supported_local',
         '20261014_proxy_selection_recovery_r1/final_audit.json;20261013_proxy_selection_r1/PROTOCOL.md',
         'Three tasks; three seeds; 12 seed-mean cells; advantage 0.777702 pp comes from G1 only',
         'No independent-source or general method superiority claim'),
        ('C02', 'The SOURCE formula adaptation performed poorly on this revealed pool', 'supported_retrospective',
         '20261014_source_hydraulic_r1/final_audit.json;20261014_source_hydraulic_r1/REPORT.md',
         'Fixed local adaptation and retrospective comparison',
         'Not an author Adam implementation or prospective equal-budget victory'),
        ('C03', 'Same expected gradients can miss finite-training S/W response differences', 'supported_local',
         '20261014_weighted_response_r1/final_audit.json;20261014_weighted_response_r1/REPORT.md',
         '36 matched W trainings; joint cell-mean lower bound 2.506055 pp',
         'No proof of decision need, variance-only causality, or a new theorem'),
        ('C04', 'Separate-pool zero regret does not imply joint-pool zero regret', 'supported_retrospective',
         '20261014_noise_necessity_review_r1/decision_value.json;20261014_noise_necessity_review_r1/REPORT.md',
         'Fixed S/W priority rules evaluated post hoc',
         'W-priority is not a prospectively validated strategy'),
        ('C05', 'MetroPT3/SKAB established cross-source M1 accuracy and benefit', 'not_established',
         '20261014_existing_sources_r1/STOP_REPORT.md',
         'Budget stop; 24 complete child labels absent; candidate numerical audit incomplete',
         'Accuracy and benefit unknown, not zero'),
        ('C06', 'Four-group predictions established decision improvement', 'not_established',
         '20261015_fourgroup_m1_r1/FORECAST_REPORT.md;20261015_fourgroup_m1_r1/forecast_summary.json',
         'M1/RAW/SHORT recommendations identical; 72 complete child labels not started',
         'Predicted improvements and replay checks are not realized labels'),
        ('C07', 'N1 approximates the complete float32 program at the registered tolerance', 'contradicted_in_registered_test',
         '20261015_n1_path_engineering_r1/FD_REPORT.md;20261015_n1_path_engineering_r1/fd_summary.json',
         '64/96 derivative and 32/48 two-step-size passes; every comparison was required',
         'Exact replay and smooth-map differentiation do not reverse failed gate'),
        ('C08', 'Measured endpoint evaluation error alone explains N1 gate failures', 'ruled_out_by_bound',
         '20261015_n1_error_decomposition_r1/summary.json;20261015_n1_error_decomposition_r1/local_audit.json;20261015_n1_error_decomposition_r1/REPORT.md',
         'All 32 derivative and 16 two-step-size failures retain lower bound above 0.02',
         'Does not identify training-rounding/nonlinearity/implementation causality'),
        ('C09', 'The current combination is a distinct methodological contribution', 'not_established',
         '20261015_n1_path_engineering_r1/RELATED_WORK.md;20261015_n1_path_engineering_r1/NEIGHBOR_AUDIT.md',
         'Existing unrolling, path sensitivity, and meta-reweighting require explicit comparison',
         'Current literature review is not exhaustive novelty certification'),
        ('C10', 'The complete requested method paper is ready for submission', 'not_established',
         '20261015_n1_path_engineering_r1/FD_REPORT.md;20261015_n1_error_decomposition_r1/REPORT.md',
         'Credible method, strong reweighting comparison, prospective decision gain, cross-source confirmation, and ablations remain open',
         'A compiled draft is not scientific completion'),
    ]
    with (OUT / 'claim_evidence.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('id', 'claim', 'status', 'evidence_paths_relative_to_stage_base', 'scope', 'boundary'))
        writer.writerows(claims)
    # Include the separate source-file study even though it is not an independent-source validation.
    sources = sorted({p for claim in claims for p in claim[3].split(';')} | {
        '20261014_file_transfer_r1/FORECAST_REPORT.md',
        '20261014_file_transfer_r1/CLAIM_CLOSURE.md',
    })
    inventory = []
    for relative in sources:
        path = BASE / relative
        payload = path.read_bytes()
        inventory.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    write('evidence_inventory.json', {'status': 'all_sources_present', 'sources': inventory})
    # Check reported rounded values against authoritative machine-readable audits.
    tex = (OUT / 'manuscript.tex').read_text(encoding='utf-8')
    checks = []
    for group in ('main_MAE_pp', 'main_mean_selected_gain_percent', 'main_cold_accounting_seconds', 'weighted_MAE_pp'):
        for key, value in metrics[group].items():
            formatted = f'{value:.6f}'
            assert formatted in tex, (group, key, formatted)
            checks.append({'group': group, 'metric': key, 'rounded_value': formatted})
    assert f'{advantage:.6f}' in tex
    write('artifact_audit.json', {
        'status': 'PASS', 'source_files_hashed': len(inventory), 'claim_rows': len(claims),
        'checked_numeric_literals': checks, 'decision_advantage_identity_verified': True,
        'all_original_failed_comparisons_fail_endpoint_bound': True,
        'scope': 'Local artifact consistency only; does not rerun scientific audits or certify novelty/layout',
        'new_training': 0, 'D_access': False, 'test_access': False,
    })
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={'width_ratios': [1.2, 1]})
    x = np.arange(3)
    for offset, method, color in [(-.18, 'M1', '#245b78'), (.18, 'SHORT', '#b1bbc5')]:
        values = m1['recommendation_gains'][method]
        bars = axes[0].bar(x + offset, values, width=.34, label=method, color=color)
        axes[0].bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=9)
    axes[0].set(xticks=x, xticklabels=['G1', 'G2', 'G3'], ylabel='Selected relative NMSE improvement (%)', ylim=(-2, 61), title='A  Same-source development decisions')
    axes[0].legend(frameon=False)
    axes[0].axhline(0, color='#888888', linewidth=.6)
    axes[1].barh(['Derivative', 'Two step sizes'], [64/96*100, 32/48*100], color='#245b78', label='Passed')
    axes[1].barh(['Derivative', 'Two step sizes'], [32/96*100, 16/48*100], left=[64/96*100,32/48*100], color='#bd5b45', label='Failed')
    axes[1].set(xlim=(0, 100), xlabel='Registered comparisons (%)', title='B  N1 numerical gate failed')
    for i, label in enumerate(['64 / 96 passed', '32 / 48 passed']):
        axes[1].text(3, i, label, color='white', va='center', fontsize=10)
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2)
    fig.text(.01, .015, 'A: Three seeds per task; tasks share one source. B: Every comparison had to pass. No confidence intervals inferred.', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(OUT / 'evidence_overview.png', dpi=180)
    fig.savefig(OUT / 'evidence_overview.pdf')
    plt.close(fig)
    print(json.dumps({'status': 'PASS', 'sources': len(inventory), 'claims': len(claims), 'new_training': 0}))


if __name__ == '__main__':
    main()
