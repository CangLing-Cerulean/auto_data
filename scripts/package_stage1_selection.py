"""Archive frozen prospective predictions and four audited shared S responses."""
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
    plan=read(root/'selection_protocol.json')
    report=read(root/'selection_node4.json')
    assert report['status']=='PASS' and report['node']==4
    assert report['selection_protocol_sha256']==sha(root/'selection_protocol.json')
    names=set(plan['code_sha256'])|set(plan['prerequisite_sha256'])|set(plan['required_prediction_files'])
    names.update(['selection_protocol.json','selection_prediction_lock.json','selection_prediction_preflight.json',
        'selection_execution_activation.json','selection_prediction_source_sync_receipt.json','selection_code_preflight.json',
        'selection_node2.json','selection_node2_lock.json','selection_node4.json','selection_initial_feedback.json',
        'selection_control','candidate_prepared','package_stage1_selection.py'])
    for name in plan['initial_run_ids']:
        folder=root/'selection_runs'/name
        assert read(folder/'status.json')['status']=='completed'
        assert sha(folder/'result.json')==report['result_sha256'][name]
        for asset,digest in read(folder/'result.json')['assets'].items():
            assert sha(folder/asset)==digest
        names.add('selection_runs/'+name)
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
    target=root/'stage1-selection-initial-20261012-r1.tar.gz'
    assert not target.exists()
    with tarfile.open(target,'w:gz',compresslevel=6) as archive:
        for item in files.values():
            archive.add(root/item['name'],arcname=item['name'],recursive=False)
    receipt=dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target),members=list(files.values()),
        new_successful_trainings=4,cumulative_successful_trainings=268,raw_CSV_included=False,test_data_included=False,
        dependencies=[dict(tag='stage1-fixed-references-20261012-r1',sha256='ed511b9524de8420ee80826ab555041be57eeb989b0179231fba76cd8afc4734'),
                      dict(tag='stage1-confirmation-recovered-20261012-r1',sha256='6a2b34a3d447f1bad0b03a48eae65ae1788ee8835e3f97151d3702b021dd6be2')])
    (root/'selection_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in receipt.items() if key!='members'}))


if __name__=='__main__':
    main()
