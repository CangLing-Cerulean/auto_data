"""Observed Python I/O restrictions for study jobs; not an OS security sandbox."""
import ast
from pathlib import Path
import sys

from ..config import GateError
from ..selection.common import file_hash,read_json,write_json
from .native import run_process

def check_builder(code):
    if not isinstance(code,str) or len(code.encode())>8000:raise GateError('STUDY_CODE_SIZE')
    try:tree=ast.parse(code)
    except SyntaxError as error:raise GateError('STUDY_CODE_SYNTAX') from error
    imports={'numpy','torch','math','json','pathlib','sys','collections','statistics'}
    blocked={'eval','exec','compile','__import__','getattr','setattr','globals','locals','vars',
        'ctypes','memmap','fromfile','from_file','socket','subprocess','requests','urllib',
        'nn','optim','hub','backward','parameters','requires_grad_','load_library',
        'system','popen','fork','spawn','execve','settrace','setprofile'}
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(name.split('.')[0] not in imports for name in names):raise GateError('STUDY_CODE_IMPORT')
            if any(part in blocked or part.startswith('_') for name in names for part in name.split('.')):
                raise GateError('STUDY_CODE_EXTERNAL_OR_MODEL_API')
            if isinstance(node,ast.ImportFrom) and any(a.name in blocked or a.name.startswith('_') for a in node.names):
                raise GateError('STUDY_CODE_EXTERNAL_OR_MODEL_API')
        if isinstance(node,ast.Attribute) and (node.attr in blocked or node.attr.startswith('_')):
            raise GateError('STUDY_CODE_EXTERNAL_OR_MODEL_API')
        if isinstance(node,ast.Name) and (node.id in blocked or (node.id.startswith('__') and node.id!='__name__')):
            raise GateError('STUDY_CODE_EXTERNAL_OR_MODEL_API')
        if isinstance(node,ast.keyword) and node.arg=='requires_grad':raise GateError('STUDY_CODE_TRAINABLE')

def install_boundary(inputs,outputs,*,read_files=()):
    inputs,outputs=Path(inputs).resolve(),Path(outputs).resolve()
    explicit={Path(p).resolve() for p in read_files}
    violations=[]
    def audit(event,args):
        if event in {'socket.__new__','subprocess.Popen','os.system','os.fork','os.exec','ctypes.dlopen'}:
            violations.append({'event':event,'reason':'process_network_or_native_extension'})
            raise PermissionError('STUDY_INFORMATION_BOUNDARY')
        if event=='open':
            path=args[0]
            if isinstance(path,int):return
            resolved=Path(path).resolve()
            mode=args[1] or ''
            flags=args[2]
            writing=any(c in mode for c in 'wax+') or bool(flags & (1|2|64|512|1024))
            allowed=resolved.is_relative_to(outputs) if writing else (
                resolved.is_relative_to(inputs) or resolved.is_relative_to(outputs) or resolved in explicit)
            if not allowed:
                violations.append({'event':'open','reason':'outside_job_assets','writing':writing})
                raise PermissionError('STUDY_INFORMATION_BOUNDARY')
        if event in {'os.listdir','os.scandir'} and args and isinstance(args[0],(str,bytes)):
            p=Path(args[0]).resolve()
            if not (p.is_relative_to(inputs) or p.is_relative_to(outputs)):
                violations.append({'event':event,'reason':'outside_job_assets'})
                raise PermissionError('STUDY_INFORMATION_BOUNDARY')
    sys.addaudithook(audit)
    return violations

def execute_bounded_job(folder,guards,*,seconds=30):
    folder=Path(folder)
    job=read_json(folder/'job.json')
    from ..config import digest
    if digest({k:v for k,v in job.items() if k!='sha256'})!=job['sha256']:raise GateError('STUDY_JOB_IDENTITY')
    expected=guards|{str(folder/'code/agent.py'):job['code_sha256']}|{
        str(folder/'inputs'/a['name']):a['sha256'] for a in job['inputs']}
    for p,h in expected.items():
        if file_hash(p)!=h:raise GateError('STUDY_ASSET_CHANGED')
    if (folder/'started.json').exists():raise GateError('STUDY_JOB_NOT_REPLAYED')
    write_json(folder/'started.json',{'job_sha256':job['sha256'],'information_boundary':'python_audit_not_os_sandbox'})
    result=run_process([sys.executable,'-B','-m','online_ts.agent_research.boundary_worker','--folder',str(folder)],
        cwd=Path(__file__).resolve().parents[2],outputs=folder/'outputs',seconds=seconds,memory_bytes=8*1024**3)
    result.update(job_sha256=job['sha256'],protected_assets_changed=[p for p,h in expected.items() if file_hash(p)!=h])
    boundary=folder/'outputs/information_boundary.json'
    if not boundary.exists():
        result.update(status='failed',error='STUDY_BOUNDARY_RECEIPT_MISSING')
    else:
        result['information_boundary']=read_json(boundary)
        if result['information_boundary']['violations']:result.update(status='failed',error='STUDY_INFORMATION_BOUNDARY')
    result['output_sha256']={name:file_hash(folder/'outputs'/name) for name in result['files']}
    write_json(folder/'execution.json',result)
    if result['protected_assets_changed']:raise GateError('STUDY_ASSET_CHANGED')
    return result
