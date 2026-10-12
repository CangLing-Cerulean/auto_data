"""Package the new nine runs; preserve the earlier interrupted archive as dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    plan = read(root/'confirmation_recovery_protocol.json')
    report = read(root/'construction_confirmation_recovered.json')
    assert report['status'] == 'PASS_WITH_PRESERVED_FAILED_DELIVERY'
    assert report['new_recovery_trainings'] == 9
    assert report['cumulative_successful_completed_trainings'] == 264
    assert report['recovery_protocol_sha256'] == sha(root/'confirmation_recovery_protocol.json')
    assert read(root/'confirmation_recovery_control/status.json')['status'] == 'completed'
    names = ['baseline_protocol.json','candidate_protocol.json','candidate_predictions.json','proposal_median3.py',
             'prepare_stage1_candidate.py','run_stage1_candidate.py','run_intervention_stage1_baseline.py',
             'audit_stage1_confirmation.py','audit_stage1_confirmation_recovered.py','audit_stage1_interruption.py',
             'supervise_stage1_recovery.py','check_stage1_recovery.py','package_stage1_recovery.py',
             'candidate_preparation_receipt.json','candidate_prepared','frozen_core_snapshot.py',
             'online_ts/agent_research/stage1_scoring.py','construction_state.json',
             'candidate_first_seed_gate.json','construction_feedback.json','confirmation_activation.json',
             'confirmation_recovery_protocol.json','confirmation_recovery_preflight.json',
             'confirmation_recovery_acceptance.json','confirmation_recovery_source_sync_receipt.json',
             'construction_confirmation_recovered.json','confirmation_recovery_control',
             'confirmation_interruption_audit.json',
             'candidate_runs/C1_CP1_s2026101404/status.json','candidate_runs/C1_CP1_s2026101404/result.json']
    for job in plan['new_jobs']:
        directory = root/'candidate_runs'/job['id']
        assert read(directory/'status.json')['status'] == 'completed'
        result = read(directory/'result.json')
        for name,digest in result['assets'].items():
            assert sha(directory/name) == digest
        names.append('candidate_runs/'+job['id'])
    files = []
    for name in names:
        path = root/name
        assert path.exists()
        for source in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if source.is_file():
                assert not source.is_symlink() and root.resolve() in source.resolve().parents
                relative = source.relative_to(root).as_posix()
                assert not any(token in relative.lower() for token in ('private','test.csv','verification/','.ssh'))
                files.append(dict(name=relative,bytes=source.stat().st_size,sha256=sha(source)))
    target = root/'stage1-confirmation-recovered-20261012-r1.tar.gz'
    assert not target.exists()
    with tarfile.open(target,'w:gz',compresslevel=6) as archive:
        for item in files:
            archive.add(root/item['name'],arcname=item['name'],recursive=False)
    receipt = dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),members=files,
        new_successful_trainings=9,preserved_failed_delivery_count=1,raw_CSV_included=False,test_data_included=False,
        dependencies=[dict(tag='stage1-fixed-references-20261012-r1',sha256='ed511b9524de8420ee80826ab555041be57eeb989b0179231fba76cd8afc4734'),
                      dict(tag='stage1-construction-initial-20261012-r1',sha256='f960ebf0d29b62b08d3ba091dfc2bc2793ae23e72ae8de1183f310b6bf5abf5d'),
                      dict(tag='stage1-confirmation-interrupted-20261012-r1',sha256='0438212952ed2f89eb1e1407bc8ec710c9b4780b9c85b26696b70709e4868c81')])
    (root/'confirmation_recovery_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'members'}))


if __name__ == '__main__':
    main()
