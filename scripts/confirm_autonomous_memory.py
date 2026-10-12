"""Confirm only a frozen R2 selection; never return this output as search feedback."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_ts.agent_research.runtime import ResearchRepository
from online_ts.agent_research.signal_gateway import SignalGateway, scope_summary
from online_ts.agent_research.signal_scoring import build_control, feedback_scores, outcome_gates
from online_ts.agent_research.task import TaskAssets
from online_ts.agent_research.memory import MEMORY
from online_ts.agent_research.processing import DEFAULTS
from online_ts.config import GateError, digest
from online_ts.selection.common import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()
    folder = Path(args.folder)
    identity = read_json(folder/'preregistration.json')
    protocol = read_json(folder/'protocol.json')
    repo = ResearchRepository('solhigh_memory_r2', identity, folder=folder/'run')
    if not repo.state().incident_finished:
        raise GateError('SEARCH_MUST_FINISH_BEFORE_CONFIRMATION')
    if (folder/'confirmation_result.json').exists():
        raise GateError('CONFIRMATION_ALREADY_COMPLETED')
    gateway = SignalGateway(repo, TaskAssets(ROOT/'research_runs/agent_research/tasks/metropt3_dev'),
        read_json(MEMORY/'library.json'), folder)
    gateway.guards.update(identity['code'])
    gateway.guards[str(Path(__file__))] = file_hash(__file__)
    for row in read_json(folder/'memory_manifest.json')['sources']:
        gateway.guards[str(folder/row['file'])] = row['sha256']
    for path, expected in gateway.guards.items():
        if file_hash(path) != expected:
            raise GateError('CONFIRMATION_PROTECTED_FILE_CHANGED')
    base = gateway.baseline()
    final = read_json(folder/'run/final_policy.json')
    result = dict(selected_candidate=final['selected_candidate'], independent_task=False,
        feedback_checks=feedback_scores(gateway),
        information_violations=repo.count_records('information_violation'),
        registered_facts=[{'signal_id': s['signal_id'], **repo.get('signal-verification:'+s['signal_id'])}
            for s in gateway.signals()])
    selected = final['selected_candidate']
    if selected == 'B0':
        write_json(folder/'confirmation_result.json', result | dict(status='baseline_selected_no_confirmation', E=False))
        return
    candidate = gateway.candidate(selected)
    measured = candidate['result']['metrics']
    gain = 100*(1-measured['NMSE']/base['metrics']['NMSE'])
    mae = 100*(measured['NMAE']/base['metrics']['NMAE']-1)
    result.update(exploration_NMSE_gain_pct=gain, exploration_NMAE_deterioration_pct=mae)
    if gain < 1 or mae > 1:
        write_json(folder/'confirmation_result.json', result | dict(status='exploration_gate_not_met', E=False))
        return
    signal = repo.get('signal:'+candidate['signal_id']) if candidate.get('signal_id') else None
    frozen = dict(final_policy_sha256=file_hash(folder/'run/final_policy.json'),
        asset_sha256=candidate['asset_sha256'], control_code_sha256=digest(candidate['control_code']),
        signal_sha256=digest(signal), confirmation_seeds=protocol['confirmation_seeds'],
        confirmation_runner_sha256=file_hash(__file__), identity_sha256=identity['sha256'])
    freeze_path = folder/'confirmation_freeze.json'
    if freeze_path.exists() and read_json(freeze_path) != frozen:
        raise GateError('CONFIRMATION_FREEZE_CHANGED')
    write_json(freeze_path, frozen)
    control_path = folder/'confirmation_control.json'
    if control_path.exists():
        control = read_json(control_path)
        if control and file_hash(control['asset']) != control['asset_sha256']:
            raise GateError('CONFIRMATION_CONTROL_CHANGED')
    else:
        control = build_control(gateway, candidate) if candidate['control_code'] else None
        if not candidate['control_code'] and not candidate.get('input_transform_path'):
            control = dict(asset=candidate['asset'], transform=None, kind='group_shuffle', asset_sha256=candidate['asset_sha256'])
        if control:
            control = {k: str(v) if isinstance(v, Path) else v for k, v in control.items()}
        write_json(control_path, control)
    outcomes = []
    for seed in protocol['confirmation_seeds']:
        rows = {}
        for name, asset, transform in [('baseline', None, None),
                ('candidate', candidate['asset'], candidate.get('input_transform_path')),
                *([('control', control['asset'], control['transform'])] if control else [])]:
            if shutil.disk_usage(folder).free < 2*1024**3:
                raise GateError('DISK_RESERVE_REQUIRES_ARCHIVE_BEFORE_NEXT_JOB')
            rows[name] = gateway.adapter.evaluate('confirmation_'+name, asset, DEFAULTS, seed=seed,
                transform=transform, control='group_shuffle' if name == 'control' and control['kind'] == 'group_shuffle' else 'aligned',
                control_seed=20261012)
            repo.export()
        local = lambda r: scope_summary(read_json(folder/'run/training'/r['job_id']/'scope_metrics.json'),
            signal['group'], signal['channels'], gateway.task.task['sensor_columns']) if signal else None
        outcomes.append(dict(seed=seed, **{name: r['metrics'] for name, r in rows.items()},
            **{'local_'+name: local(r) for name, r in rows.items()},
            jobs={name: r['job_id'] for name, r in rows.items()}))
        write_json(folder/'confirmation_progress.json', dict(results=outcomes, completed_seeds=len(outcomes)))
        print('Confirmed seed', seed, flush=True)
    gates = outcome_gates(outcomes, signal['local_gain_pct'] if signal else 3)
    valid = result['information_violations'] == 0 and gateway.journal.summary()['autonomy_eligible']
    write_json(folder/'confirmation_result.json', result | dict(status='completed_pending_semantic_review',
        results=outcomes, gates=gates, E=bool(gates['E_numeric'] and valid),
        information_conditions_valid=valid, control_kind=control['kind'] if control else 'unavailable'))
    repo.export()


if __name__ == '__main__':
    main()
