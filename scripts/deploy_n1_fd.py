"""Emit an exact-commit, create-only extension after the N1 path delivery."""
import ast
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path
import register_n1_fd as registry


def main():
    stage=registry.STAGE
    publication=registry.read(stage/'fd_publication_receipt.json')
    assert publication['status']=='exact_commit_and_tree_verified'
    commit=publication['source_commit']
    assert re.fullmatch('[a-f0-9]{40}',commit)
    def git(*args):
        return subprocess.run(['git','-c',f'safe.directory={registry.ROOT.as_posix()}',*args],
            cwd=registry.ROOT,check=True,capture_output=True).stdout
    assert git('rev-parse',commit+'^{tree}').decode().strip()==publication['tree_sha']
    plan=registry.read(stage/'fd_plan.json')
    files=[]
    for name in list(plan['code_sha256'])+['fd_plan.json','paths_sync_receipt.json','fd_accounting_checks.json']:
        path=registry.ROOT/'scripts'/name if name.endswith('.py') else stage/name
        data=path.read_bytes()
        expected=hashlib.sha256(data).hexdigest()
        assert hashlib.sha256(git('show',commit+':'+path.relative_to(registry.ROOT).as_posix())).hexdigest()==expected
        if name in plan['code_sha256']:assert expected==plan['code_sha256'][name]
        files.append(dict(name=name,sha256=expected,content=base64.b64encode(data).decode()))
    source=dict(publication,fd_plan_sha256=registry.sha(stage/'fd_plan.json'))
    with (stage/'fd_source_sync_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(source,stream,indent=2);stream.write('\n')
    data=(stage/'fd_source_sync_receipt.json').read_bytes()
    files.append(dict(name='fd_source_sync_receipt.json',sha256=hashlib.sha256(data).hexdigest(),content=base64.b64encode(data).decode()))
    payload=base64.b64encode(json.dumps(files).encode()).decode()
    program="import base64,hashlib,json,os,pathlib,shutil,sys\nfiles=json.loads(base64.b64decode('"+payload+"'))\n"
    program+='''
if not __debug__ or os.name!='posix':raise RuntimeError('Unoptimized Linux deployment required')
root=pathlib.Path('/root/auto_data_experiments/20261015_n1_path_engineering_r1/experiment')
assert root.resolve()==root
sys.path.insert(0,str(root))
import n1_path_runtime as base
base.rt.configure();plan=base.verify(root)
audit=base.rt.read(root/'paths_audit.json')
assert audit['status']=='PASS' and audit['all_cost_lower_bounds_pass']
assert audit['paths_sha256']==base.rt.sha(root/'paths.json')
assert not (root/'stage_failure.json').exists()
assert shutil.disk_usage(root).free>=plan['preflight_min_free_bytes']
for item in files:
 assert pathlib.Path(item['name']).name==item['name'] and not (root/item['name']).exists()
 assert hashlib.sha256(base64.b64decode(item['content'])).hexdigest()==item['sha256']
for item in files:
 with (root/item['name']).open('xb') as stream:
  stream.write(base64.b64decode(item['content']));stream.flush();os.fsync(stream.fileno())
 assert base.rt.sha(root/item['name'])==item['sha256']
import n1_finite_difference
n1_finite_difference.verify(root)
base.rt.write(root/'fd_deployment.json',dict(status='PASS',fd_plan_sha256=base.rt.sha(root/'fd_plan.json'),
 free_bytes=shutil.disk_usage(root).free))
print(json.dumps(dict(status='FD_source_and_prerequisites_verified',fd_plan_sha256=base.rt.sha(root/'fd_plan.json'))))
'''
    ast.parse(program)
    with (stage/'private/deploy_fd_program.py').open('x',encoding='utf-8') as stream:stream.write(program)


if __name__=='__main__':
    main()
