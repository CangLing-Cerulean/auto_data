"""Re-evaluate immutable R2 checkpoints, write only new phase-0 artifacts."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from online_ts.agent_research.intervention import paired_response, dual_update


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--old-project',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    old=Path(args.old_project);out=Path(args.output)
    if out.exists(): raise RuntimeError('Refusing to overwrite existing evaluation')
    out.mkdir(parents=True)
    started=time.monotonic()
    sys.path.insert(0,str(old/'runtime_deps'))
    from online_ts.agent_research.transforms import load_processor,apply_processor
    stage=old/'research_runs/agent_research/autonomous_memory/20261012_sol_high_r2'
    audit=read(stage/'audit_summary.json')
    rows=[r for r in audit['trainings'] if r['status']=='completed' and r['seed'] in (2026101202,2026101203,2026101204)]
    assert len(rows)==9
    first=read(stage/'run/training'/rows[0]['job_id']/'job.json')
    base=Path(first['identity']['base_root'])
    p=read(base/'research_runs/protocol.json');p.update(first['config'])
    spec=importlib.util.spec_from_file_location('paired_frozen_core',base/'scripts/run_research.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    manifest=read(base/'data/metropt3_v1/manifest.json')
    arrays,starts,groups,norm=core.prepare(p,manifest)
    # Explicit retained source-row mapping, no test data loaded by prepare.
    validation_start=next(s['row_start_inclusive'] for s in manifest['splits'] if s['name']=='validation')
    ids=starts['validation'];n=len(ids);assert n==300535
    np.savez_compressed(out/'window_identity.npz',start_row=ids,source_row=ids+validation_start,
                       target_start_row=ids+32,high_input_variation=groups['high_input_variation'],
                       sensor_columns=np.asarray(manifest['sensor_columns']),horizons=np.arange(1,9))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    data=torch.tensor(arrays['validation'],device='cuda');offsets=torch.arange(40,device='cuda')
    target=np.zeros((n,8,15),dtype=bool)
    # Original R2 registered target; do not narrow to the two changed signs.
    channel_names=['TP2','H1','DV_eletric']
    channel_ids=[manifest['sensor_columns'].index(k) for k in channel_names]
    target[:,:,channel_ids]=groups['high_input_variation'][:,None,None]
    responses=[];identities=[]
    for seed in (2026101202,2026101203,2026101204):
        errors={}
        for row in [r for r in rows if r['seed']==seed]:
            folder=stage/'run/training'/row['job_id'];job=read(folder/'job.json');ident=job['identity']
            for path,expected in ident['protected_assets'].items():
                if sha(path)!=expected:raise RuntimeError('protected asset changed: '+path)
            result=read(folder/'result.json')
            for name,expected in result['assets'].items():
                if sha(folder/name)!=expected:raise RuntimeError('result asset changed')
            assert result['steps']==3000 and result['parameters']==7920
            assert job['config']==first['config'] and ident['base_root']==first['identity']['base_root']
            label=('B0' if row['candidate_id']=='confirmation_baseline' else
                   'parent' if 'control' in row['candidate_id'] else 'child')
            if label in errors:raise RuntimeError('ambiguous recipe mapping')
            model=core.DLinear(p).cuda();model.load_state_dict(torch.load(folder/'best.pt',weights_only=True));model.eval()
            program=load_processor(ident['input_transform_path'],ident['input_state_path']) if ident['input_transform_path'] else None
            error=np.empty((n,8,15),dtype=np.float32)
            with torch.no_grad():
                for lo in range(0,n,p['evaluation_batch_size']):
                    ix=torch.tensor(ids[lo:lo+p['evaluation_batch_size']],device='cuda')
                    batch=data[ix[:,None]+offsets]
                    prediction=model(apply_processor(batch[:,:32],program,identity_control=ident['control']=='input_identity'))
                    error[lo:lo+len(ix)]=(prediction-batch[:,32:]).cpu().numpy()
            metric=float(np.mean(error.astype(np.float64)**2))
            if not np.isclose(metric,result['metrics']['NMSE'],atol=1e-9,rtol=1e-7):raise RuntimeError('metric replay mismatch')
            errors[label]=error
            # Full elemental evidence for all three paired seeds, not window means.
            name=f'errors_{seed}_{label}.npz';np.savez_compressed(out/name,error=error)
            identities.append(dict(seed=seed,recipe=label,job_id=row['job_id'],checkpoint_sha256=sha(folder/'best.pt'),
                result_sha256=sha(folder/'result.json'),error_file=name,error_sha256=sha(out/name),NMSE=metric))
            print(json.dumps(dict(seed=seed,recipe=label,NMSE=metric)),flush=True)
        assert set(errors)=={'B0','parent','child'}
        for parent,child,action in [('B0','parent','historical_source_half'),('parent','child','source_sign_flip')]:
            response=paired_response(errors[parent],errors[child],target)
            responses.append(dict(seed=seed,parent=parent,child=child,action=action,response=response,
                eligibility='retrospective_mechanism_regression_only',prestate=None,prospective_prediction=None,
                update=dual_update(response,preregistered=False,matched_control_passed=False,independent_confirmation=False)))
    with (out/'response_long.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','parent','child','action','region','elements','metric','parent_loss','child_loss','delta'])
        writer.writeheader()
        for r in responses:
            for region,values in r['response'].items():
                for metric,delta in values['delta'].items():
                    writer.writerow(dict(seed=r['seed'],parent=r['parent'],child=r['child'],action=r['action'],region=region,
                        elements=values['elements'],metric=metric,parent_loss=values['parent'][metric],child_loss=values['child'][metric],delta=delta))
    write(out/'response_collection.json',responses)
    write(out/'receipt.json',dict(status='passed',source_audit_sha256=sha(stage/'audit_summary.json'),
        manifest_sha256=sha(base/'data/metropt3_v1/manifest.json'),protocol_sha256=sha(base/'research_runs/protocol.json'),
        normalization=norm,window_identity_sha256=sha(out/'window_identity.npz'),identities=identities,
        target=dict(group='high_input_variation',channels=channel_names,horizons=list(range(1,9))),
        complement='all evaluation elements outside target, including other channels in target windows',
        training_count=0,test_access=False,external_API_calls=0,process_seconds=time.monotonic()-started,
        independent_sources=1,independent_contexts=1,training_seeds=3,
        uncertainty='No confidence intervals treating overlapping windows as independent',
        missing=['prospective state for newly proposed signals','intermediate per-window training dynamics'],
        source_sha256=sha(__file__)))


if __name__=='__main__':main()
