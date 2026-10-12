"""Offline Sol-high decisions through the existing audited research gateway; no API."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_ts.agent_research.runtime import CONFIG, ResearchRepository, code_identity
from online_ts.agent_research.signal_gateway import SignalGateway
from online_ts.agent_research.signal_study import study_context
from online_ts.agent_research.task import TaskAssets
from online_ts.agent_research.memory import MEMORY, verify_memory
from online_ts.agent_research.processing import DEFAULTS
from online_ts.agent_runtime.controller import AgentController
from online_ts.agent_runtime.state import Event
from online_ts.config import digest, GateError
from online_ts.selection.common import file_hash, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--decision')
    args = parser.parse_args()
    if os.name != 'posix':
        raise GateError('RESEARCH_SERVER_REQUIRED')
    folder = Path(args.folder)
    protocol = read_json(folder / 'protocol.json')
    library = read_json(MEMORY / 'library.json')
    verify_memory(library)
    task = TaskAssets(ROOT / 'research_runs/agent_research/tasks/metropt3_dev')
    config = CONFIG | dict(version='autonomous-full-memory-r2', model='internal-sol-high',
        scope='development_only', seed=protocol['exploration_seed'], group='full_memory',
        max_candidates=None, max_diagnostics=None, max_calls=None, max_llm_tokens=None,
        max_training_jobs=None, execution_profile='research_open', experience_mode='full',
        feedback_mode='visible', input_transforms_allowed=True, formal_research_protocol=True,
        parameter_ranges={k: [v] for k, v in DEFAULTS.items()})
    supplemental = read_json(folder / 'memory_manifest.json')
    for row in supplemental['sources']:
        if file_hash(folder / row['file']) != row['sha256']:
            raise GateError('SUPPLEMENTAL_MEMORY_CHANGED')
    identity = dict(config=config, code=code_identity() | {str(Path(__file__)): file_hash(__file__)},
        protocol_sha256=file_hash(folder / 'protocol.json'),
        protocol_document_sha256=file_hash(folder / 'PROTOCOL.md'),
        agent_instructions_sha256=file_hash(folder / 'AGENT_INSTRUCTIONS.md'),
        task_sha256=file_hash(task.folder / 'task.json'), library_sha256=library['sha256'],
        supplemental_sha256=file_hash(folder / 'memory_manifest.json'))
    identity['sha256'] = digest(identity)
    frozen = folder / 'preregistration.json'
    if frozen.exists() and read_json(frozen) != identity:
        raise GateError('AUTONOMOUS_IDENTITY_CHANGED')
    write_json(frozen, identity)
    repo = ResearchRepository('solhigh_memory_r2', identity, folder=folder / 'run')
    gateway = SignalGateway(repo, task, library, folder)
    gateway.guards.update(identity['code'])
    for row in supplemental['sources']:
        gateway.guards[str(folder / row['file'])] = row['sha256']
        gateway.journal.evidence(row['evidence_id'], dataset_id=task.task['dataset_id'],
            role='historical_development', artifact_sha256=row['sha256'],
            summary=json.dumps(row, ensure_ascii=False))

    def context(repository, active_gateway, event):
        value = study_context(repository, active_gateway, event)
        value['supplemental_memory'] = supplemental
        value['visible_evidence_refs'] = list(dict.fromkeys(value['visible_evidence_refs'] +
            [r['evidence_id'] for r in supplemental['sources']]))
        value['context_version'] = digest({k: v for k, v in value.items() if k != 'context_version'})
        return value

    gateway.baseline()
    event = Event('solhigh-' + str(repo.state().decision_version), 'offline_model_decision', time.time_ns())
    if args.decision:
        decision = read_json(args.decision)
        expected = read_json(folder / 'context.json')['context_version']
        def model(value):
            if value['context_version'] != expected:
                raise GateError('STALE_OFFLINE_DECISION')
            return decision
        result = AgentController(repo, gateway, model, context_builder=context).step(event)
        write_json(folder / ('result_%04d.json' % repo.state().decision_version), result)
        print(json.dumps({'tool': result['selected_action'], 'result': result['actual_result']}, ensure_ascii=False), flush=True)
    repo.export()
    current = context(repo, gateway, event)
    write_json(folder / 'context.json', current)
    write_json(folder / 'progress.json', dict(status=repo.state().status,
        decision_version=repo.state().decision_version, completed=repo.state().incident_finished,
        trainings=list(repo.iter_records('research_training')), external_API_calls=0,
        internal_model_cost='not exposed by the platform; not zero', updated_ns=time.time_ns()))
    print(json.dumps({'context_version': current['context_version'], 'status': repo.state().status,
        'decision_version': repo.state().decision_version}), flush=True)


if __name__ == '__main__':
    main()
