"""Local r6 interfaces only: no prepare/run, full matrix, LLM, or raw arrays."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import feedback_forks as f

CHECKS, ERRORS = [], []
def require(value, message):
    if not value: raise AssertionError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(name, function):
    try: CHECKS.append(dict(check=name, status='PASS', details=function()))
    except Exception as exc:
        ERRORS.append(dict(check=name, error=repr(exc)))
        CHECKS.append(dict(check=name, status='FAIL', error=repr(exc)))
def close(a, b):
    require(np.allclose(a, b, rtol=1e-12, atol=1e-12), 'numeric disagreement')

def full_compatibility():
    comparisons = 0
    for family in ('ETT', 'Hydraulic'):
        source = [f.public_record(r, 'full') for r in f.records_for(family) if r['family'] != family]
        for names, quadratic in ((f.BASE_FEATURES, False), (f.original_names(source), True), *(((*f.BASE_FEATURES, h), False) for h in f.SIGNALS)):
            actual, old = f.fit_partial(source, names, quadratic).snapshot(), f.fit(source, names, quadratic).snapshot()
            require(actual == old, 'full fit differs from stable r4 fit')
            comparisons += 2
    return dict(model_components=comparisons, exact_snapshots=True)

def missing_heads():
    records = f.records_for('ETT')
    source = [f.public_record(r, 'full') for r in records if r['family'] != 'ETT']
    target = [r for r in records if (r['family'], r['task']) == ('ETT', 'ETTh1')]
    mixed = source + [f.public_record(target[0], 'global_only'), f.public_record(target[1], 'hidden_current'), f.public_record(target[2], 'full')]
    model = f.fit_partial(mixed, f.BASE_FEATURES).snapshot()
    for head, required in (('marginal', {'Y'}), ('conditional', {'Y', 'Z'})):
        expected = [r['id'] for r in mixed if required <= r['response'].keys()]
        require(model[head]['train_record_ids'] == expected, 'incorrect training subset '+head)
    for mode, keys in (('full', {'Y','Z','NMAE_gain'}), ('global_only', {'Y','NMAE_gain'}), ('hidden_current', set())):
        public = f.public_record(target[0], mode)
        require(set(public) == {'id','family','task','action','implementation','features','response'}, 'public schema')
        require(set(public['response']) == keys, 'response visibility '+mode)
    return dict(Y_rows=len(model['marginal']['train_record_ids']), Z_rows=len(model['conditional']['train_record_ids']))

def perturbation_and_prefix():
    details = []
    for family, task in f.TASKS:
        original = f.records_for(family)
        altered = copy.deepcopy(original)
        for r in altered:
            if r['family'] == family:
                r['response'] = dict(Y=999., Z=-888., NMAE_gain=-777.)
                r['seed_Y'] = [-9., -9., -9.]
        prefix = f.make_prefix(original, family, task)
        require(prefix == f.make_prefix(altered, family, task), 'prefix target-label leakage')
        first = next(r for r in original if r['id'] == prefix['selected'])
        source = [f.public_record(r, 'full') for r in original if r['family'] != family]
        pending = [f.view(r) for r in original if (r['family'],r['task']) == (family,task) and r['id'] != first['id']]
        for mode in ('global_only', 'hidden_current'):
            changed = copy.deepcopy(first)
            changed['response']['Z'] += 1234.
            if mode == 'hidden_current': changed['response'].update(Y=-4321., NMAE_gain=5432.)
            require(f.public_record(first, mode) == f.public_record(changed, mode), 'masked report changes')
            a = f.prediction_state(source+[f.public_record(first,mode)], pending, 'mismatch_conditional', f.original_names(original), {h:0. for h in f.SIGNALS})
            b = f.prediction_state(source+[f.public_record(changed,mode)], pending, 'mismatch_conditional', f.original_names(original), {h:0. for h in f.SIGNALS})
            require(a[0] == b[0] and a[4] == b[4], 'hidden perturbation changes continuation')
        details.append(dict(family=family, task=task, selected=prefix['selected']))
    return dict(prefixes=details, masked_continuation_pairs=10)

def scores_and_selection():
    records = f.records_for('ETT')
    source = [f.public_record(r, 'full') for r in records if r['family'] != 'ETT']
    target = next(r for r in records if r['family']=='ETT')
    xb = f.feature_vector(target, f.BASE_FEATURES)
    xh = {h:f.feature_vector(target, (*f.BASE_FEATURES,h)) for h in f.SIGNALS}
    ref, models, _, _ = f.model_set(source, 'mismatch_conditional', f.original_names(records))
    checked = 0
    for method in ('mismatch_conditional','mismatch_joint'):
        for mode in f.BRANCHES:
            response = f.public_record(target,mode)['response']; rho={h:0. for h in f.SIGNALS}
            event=f.update_evidence(method,ref,models,xb,xh,response,rho)
            for h,m in models.items():
                if mode=='full':
                    base,extra=ref.log_scores(xb,response['Y'],response['Z']),m.log_scores(xh[h],response['Y'],response['Z'])
                    expected={k:extra[k]-base[k] for k in base}
                    for k,v in expected.items(): close(event['scores'][h][k],v)
                    close(rho[h],expected['conditional' if method.endswith('conditional') else 'joint'])
                elif mode=='global_only':
                    marginal=m.predict_gain(xh[h]).logpdf(response['Y'])-ref.predict_gain(xb).logpdf(response['Y'])
                    close(event['scores'][h]['marginal'],marginal)
                    close(rho[h],marginal if method.endswith('joint') else 0.)
                    require(event['scores'][h]['conditional'] is None, 'missing Z must not be scored')
                else:
                    close(rho[h],0.)
                    require(all(v is None for v in event['scores'][h].values()), 'hidden values scored')
                checked+=1
    observed=[dict(id='a',query_step=1,response=dict(Y=.5,NMAE_gain=.1)),dict(id='b',query_step=2,response=dict(Y=.1,NMAE_gain=0.)),dict(id='z',query_step=2,response=dict(Y=.1,NMAE_gain=0.))]
    require(f.choose_observed(observed)=='a' and f.choose_observed(observed,True)=='z', 'actual/continuation or tie adoption rule')
    require(f.choose_observed([dict(id='x',query_step=1,response={})])=='PARENT', 'hidden first result adopted')
    return dict(score_cases=checked, adoption_and_continuation_separate=True)

def order_and_independence():
    tree=ast.parse((ROOT/'scripts/feedback_forks.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_trajectory')
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    def line(fragment):
        matches=[n.lineno for n in ast.walk(loop) if isinstance(n,(ast.Assign,ast.Expr,ast.If)) and fragment in ast.unparse(n)]
        return min(matches)
    # Locate direct body statements; enclosing loop/if cannot obscure ordering.
    direct=[ast.unparse(n) for n in loop.body]
    registration=next(i for i,s in enumerate(direct) if s.startswith('registration ='))
    save=next(i for i,s in enumerate(direct) if s.startswith('if folder:') and 'registration.json' in s)
    reveal=next(i for i,s in enumerate(direct) if s.startswith('revealed ='))
    update=next(i for i,s in enumerate(direct) if s.startswith('event ='))
    append=next(i for i,s in enumerate(direct) if s.startswith('observed.append'))
    require(registration<save<reveal<update<append,'register/reveal/refit order')
    evaluation=next(n for n in loop.body if isinstance(n,ast.If) and ast.unparse(n.test)=='step in (2, 4)')
    names={n.id for n in ast.walk(evaluation) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    require(not names & {'visible','rho','candidates','observed','reference','models'},'evaluation state feedback')
    for n in ast.walk(evaluation):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name):
            require(n.func.value.id not in {'visible','rho','candidates','observed'},'evaluation mutates policy state')
    local_assignments={n.targets[0].id for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    require({'visible','rho','observed'} <= local_assignments,'branch mutable state not initialized locally')
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_trajectory']
    require(len(calls)==1 and [ast.unparse(v) for v in calls[0].args[:6]]==['records','family','task','method','branch','prefix'],'independent branch call inputs')
    return dict(registration_line=loop.body[save].lineno,reveal_line=loop.body[reveal].lineno,evaluation_line=evaluation.lineno, branch_state='local per trajectory', limitation='Static dataflow plus local interface perturbations; no full trajectory executed at gate.')

def main():
    start=time.perf_counter()
    for name,fn in [('full_fit_matches_r4',full_compatibility),('partial_head_training_sets',missing_heads),('prefix_and_hidden_response_perturbations',perturbation_and_prefix),('scores_and_adoption',scores_and_selection),('register_reveal_evaluation_and_branch_flow',order_and_independence)]: check(name,fn)
    path=STAGE/'gate_audit.json'
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    report=dict(status='PASS' if not ERRORS else 'FAIL',checks=CHECKS,failures=ERRORS,audit_seconds=time.perf_counter()-start,code_sha256=sha(ROOT/'scripts/feedback_forks.py'),auditor_sha256=sha(__file__),new_full_trainings=0,new_LLM_calls=0,full_trajectories_executed=0,scope='Locked-feature pools and local interfaces only; no raw arrays, D/test, server, prepare or run.',previous_attempt=previous)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],checks=len(CHECKS),failures=ERRORS)))
    if ERRORS: raise SystemExit(1)
if __name__=='__main__': main()
