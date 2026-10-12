"""Archive the full paired S confirmation without rewriting initial predictions."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    plan=read(root/'selection_confirmation_protocol.json')
    report=read(root/'selection_confirmation_feedback.json')
    assert report['status']=='PASS' and report['new_trainings']==18
    assert report['protocol_sha256']==sha(root/'selection_confirmation_protocol.json')
    names=set(plan['code_sha256'])|set(plan['prerequisite_sha256'])|set(plan['required_prediction_files'])
    names.update(['selection_confirmation_protocol.json','selection_prediction_lock.json','selection_confirmation_preflight.json',
        'selection_confirmation_activation.json','selection_confirmation_source_sync_receipt.json',
        'selection_confirmation_feedback.json','selection_confirmation_control','candidate_prepared','package_stage1_selection_confirmation.py'])
    for name in plan['run_ids']:
        folder=root/'selection_confirmation_runs'/name
        assert read(folder/'status.json')['status']=='completed'
        assert sha(folder/'result.json')==report['result_sha256'][name]
        for asset,digest in read(folder/'result.json')['assets'].items():
            assert sha(folder/asset)==digest
        names.add('selection_confirmation_runs/'+name)
    files={}
    for name in sorted(names):
        path=root/name
        assert path.exists(),name
        for source in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if source.is_file():
                assert not source.is_symlink() and root.resolve() in source.resolve().parents
                relative=source.relative_to(root).as_posix()
                assert not any(token in relative.lower() for token in ('private','test.csv','verification/','.ssh','__pycache__'))
                files[relative]=dict(name=relative,bytes=source.stat().st_size,sha256=sha(source))
    target=root/'stage1-selection-confirmation-20261013-r1.tar.gz'
    assert not target.exists()
    with tarfile.open(target,'w:gz',compresslevel=6) as archive:
        for item in files.values():
            archive.add(root/item['name'],arcname=item['name'],recursive=False)
    receipt=dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),members=list(files.values()),
        new_successful_trainings=18,cumulative_successful_trainings=286,raw_CSV_included=False,test_data_included=False,
        dependencies=[dict(tag='stage1-fixed-references-20261012-r1',sha256='ed511b9524de8420ee80826ab555041be57eeb989b0179231fba76cd8afc4734'),
                      dict(tag='stage1-selection-initial-20261012-r1',sha256='d6fbdeaebf7a436e2f50a4cf772116df30bd49bdc9dafc13509f54d7203c150e')])
    (root/'selection_confirmation_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in receipt.items() if key!='members'}))


if __name__=='__main__':
    main()
