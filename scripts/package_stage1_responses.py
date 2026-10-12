"""Archive one completed response phase, without changing earlier archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--phase', choices=['initial', 'confirmation'], required=True)
    args = parser.parse_args()
    root = Path(args.root)
    target = root/f'stage1-construction-{args.phase}-20261012-r1.tar.gz'
    assert not target.exists()
    protocol = json.loads((root/'candidate_protocol.json').read_text(encoding='utf-8-sig'))
    seeds = [protocol['exploration_seed']] if args.phase == 'initial' else protocol['confirmation_seeds']
    actions = ['P1', 'CP1'] if args.phase == 'initial' else ['PARENT', 'P1', 'CP1']
    runs = [f'candidate_runs/{task}_{action}_s{seed}' for task in ('C1', 'C2') for seed in seeds for action in actions]
    names = ['baseline_protocol.json', 'candidate_protocol.json', 'candidate_predictions.json', 'proposal_median3.py',
             'prepare_stage1_candidate.py', 'run_stage1_candidate.py', 'run_intervention_stage1_baseline.py',
             'audit_stage1_candidates.py', 'package_stage1_responses.py', 'candidate_preparation_receipt.json',
             'candidate_prepared', 'frozen_core_snapshot.py', 'online_ts/agent_research/stage1_scoring.py',
             'construction_state.json', 'candidate_first_seed_gate.json', 'construction_feedback.json']
    if args.phase == 'confirmation':
        names += ['confirmation_activation.json', 'audit_stage1_confirmation.py', 'construction_confirmation.json']
    files = []
    for name in names+runs:
        path = root/name
        assert path.exists()
        if name in runs:
            assert json.loads((path/'status.json').read_text())['status'] == 'completed'
        candidates = sorted(path.rglob('*')) if path.is_dir() else [path]
        for source in candidates:
            if source.is_file():
                assert not source.is_symlink() and root.resolve() in source.resolve().parents
                relative = source.relative_to(root).as_posix()
                assert not any(token in relative.lower() for token in ('private', 'test.csv', 'verification/', '.ssh'))
                files.append(dict(name=relative, bytes=source.stat().st_size, sha256=sha(source)))
    with tarfile.open(target, 'w:gz', compresslevel=6) as archive:
        for item in files:
            archive.add(root/item['name'], arcname=item['name'], recursive=False)
    receipt = dict(name=target.name, bytes=target.stat().st_size, sha256=sha(target), members=files,
                   phase=args.phase, new_phase_trainings=len(runs), test_data_included=False, raw_CSV_included=False,
                   base_task_arrays_dependency='stage1-fixed-references-20261012-r1 Release asset; baseline_assets.json',
                   reference_archive_sha256='ed511b9524de8420ee80826ab555041be57eeb989b0179231fba76cd8afc4734')
    (root/f'{args.phase}_archive_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in receipt.items() if key != 'members'}))


if __name__ == '__main__':
    main()
