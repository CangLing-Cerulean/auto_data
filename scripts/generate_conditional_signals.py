"""One frozen, source-isolated LLM call per signal-generation condition."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from conditional_diagnosis import LinearGaussian
from diagnosis_pool_features import read,sha,write
from replay_conditional_diagnosis import BASE_FEATURES,feature_vector

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'
OUT=STAGE/'signal_generation'
SANDBOX=Path('C:/Users/31227/.codex/visualizations/2026/10/11/01a12b12-9990-74c3-92f2-32f6f65c592f/conditional_signal_calls')
MODES=('ordinary','mismatch')
FUNCTIONS={'abs':abs,'tanh':np.tanh,'log1p':np.log1p,'safe_ratio':lambda a,b:a/(1+abs(b))}


def evaluate(expression,features):
    tree=ast.parse(expression,mode='eval')
    if len(list(ast.walk(tree)))>31:raise ValueError('Expression exceeds 31 AST nodes')
    def visit(n):
        if isinstance(n,ast.Expression):return visit(n.body)
        if isinstance(n,ast.Name) and n.id in features:return features[n.id]
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and n.value in (-1,0,.5,1,2):return n.value
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -visit(n.operand)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)):
            a,b=visit(n.left),visit(n.right)
            return a+b if isinstance(n.op,ast.Add) else a-b if isinstance(n.op,ast.Sub) else a*b
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUNCTIONS and not n.keywords:
            if len(n.args)!=(2 if n.func.id=='safe_ratio' else 1):raise ValueError('Invalid arity')
            if n.func.id=='log1p' and not (isinstance(n.args[0],ast.Call) and isinstance(n.args[0].func,ast.Name) and n.args[0].func.id=='abs'):
                raise ValueError('log1p argument must be abs')
            return FUNCTIONS[n.func.id](*[visit(arg) for arg in n.args])
        raise ValueError('Unapproved expression node')
    result=float(visit(tree))
    if not np.isfinite(result):raise ValueError('Nonfinite expression value')
    return result


def cross_task_predictions(records,quadratic):
    names=tuple(sorted(records[0]['features'])) if quadratic else BASE_FEATURES
    out={}
    for held in sorted({r['task'] for r in records}):
        train=[r for r in records if r['task']!=held]
        x=np.array([feature_vector(r,names,quadratic) for r in train])
        y=np.array([r['response']['Y'] for r in train]);group=np.array([r['task'] for r in train])
        c,s,b=LinearGaussian._fit_mean(x,y,group,1.)
        for r in records:
            if r['task']==held:out[r['id']]=float(np.r_[1.,(feature_vector(r,names,quadratic)-c)/s]@b)
    return out


def prepare():
    OUT.mkdir();SANDBOX.mkdir()
    records=read(STAGE/'development_pool.json')['records']
    item=dict(type='object',additionalProperties=False,properties={k:{'type':'string'} for k in ('name','expression','measurement_meaning','limitation','falsifying_response')},required=['name','expression','measurement_meaning','limitation','falsifying_response'])
    schema=dict(type='object',additionalProperties=False,properties={'signals':dict(type='array',items=item,minItems=0,maxItems=2),'reasoning':{'type':'string'}},required=['signals','reasoning'])
    write(OUT/'schema.json',schema)
    calls=[]
    for family in ('ETT','Hydraulic'):
        rows=[r for r in records if r['family']==family]
        predictions=[cross_task_predictions(rows,q) for q in (False,True)]
        tasks={name:f'T{i+1}' for i,name in enumerate(sorted({r['task'] for r in rows}))}
        facts=[]
        for i,r in enumerate(rows):
            pred=[p[r['id']] for p in predictions]
            qualified=[abs(p-r['response']['Y'])>.01 and (all(p-y>.0025 for y in r['seed_Y']) or all(p-y<-.0025 for y in r['seed_Y'])) for p in pred]
            facts.append(dict(record=f'R{i+1}',task=tasks[r['task']],features=r['features'],response=r['response'],seed_Y=r['seed_Y'],seed_Z=r['seed_Z'],source_task_out_predictions=dict(linear=pred[0],quadratic=pred[1]),qualified_mismatch=dict(linear=qualified[0],quadratic=qualified[1])))
        for mode in MODES:
            name=f'{family}_{mode}';directory=OUT/name;directory.mkdir()
            packet=dict(allowed_features=sorted(rows[0]['features']),records=facts)
            write(directory/'input.json',packet)
            focus=('Construct up to two reusable scalar feature combinations to reduce out-of-task Y prediction errors. Use ordinary error-driven feature engineering. You have the complete Y/Z feedback.' if mode=='ordinary' else 'Construct up to two reusable diagnostic scalar signals from qualified response mismatches. Distinguish observed recipe Y from explanations: predict local contrast Z conditional on Y, state a falsifying future response, and target signals useful for unseen candidate decisions. A mismatch is not proof of missing information.')
            prompt='''You are an experimental signal generator. Return only the requested JSON. Do not call tools, read files, search the web, or use prior conversations. All permitted evidence is below; records and tasks are anonymized. Hidden target tasks are unavailable. No source or task ID may be an expression input. This is retrospective development, not causal confirmation.
Y is (parent global MSE-child global MSE)/parent global MSE. Z is target-minus-complement improvement using the SAME parent-global denominator. Target is the latter half of horizon, p=0.5. Positive Y means lower loss. Features use parent B errors, A group/B input statistics and action exposure. S samples nonuniformly with ordinary loss; W samples uniformly with corresponding loss weights. first/second are training groups. No hidden candidate results may be inferred as facts.
The next fixed predictor is task-balanced ridge with coefficient 1, task-held-out uncertainty and a 0.01 response scale floor. Strong baseline already has all 26 features and all quadratic terms. New features must add plausible reusable interactions, not memorize rows. At most two features; zero is allowed. Expressions allow only feature identifiers; constants -1,0,0.5,1,2; +,-,*; abs(x), tanh(x), log1p(abs(x)), safe_ratio(a,b)=a/(1+abs(b)). Each expression at most 31 Python AST nodes (operators and load nodes count). No arbitrary thresholds, division, powers, indexing or code. State measurement meaning, limitation, and a future falsifying response. Do not claim discovery of information beyond these measurements.
'''+focus+'\nFACTS\n'+json.dumps(packet,ensure_ascii=False,separators=(',',':'))
            (directory/'prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
            calls.append(dict(name=name,family=family,mode=mode,prompt_sha256=sha(directory/'prompt.txt'),input_sha256=sha(directory/'input.json'),qualified_linear=sum(f['qualified_mismatch']['linear'] for f in facts),qualified_quadratic=sum(f['qualified_mismatch']['quadratic'] for f in facts)))
    write(OUT/'generation_lock.json',dict(calls=calls,model='gpt-6.1-sol',reasoning='high',max_calls=4,no_retries=True,pool_sha256=sha(STAGE/'development_pool.json'),protocol_sha256=sha(STAGE/'SIGNAL_GENERATION_PROTOCOL.md'),code_sha256=sha(__file__)))
    print(json.dumps(dict(status='PREPARED',calls=len(calls))))


def run(name):
    lock=read(OUT/'generation_lock.json');entry=next(x for x in lock['calls'] if x['name']==name)
    directory=OUT/name
    assert sha(directory/'prompt.txt')==entry['prompt_sha256'] and sha(__file__)==lock['code_sha256']
    write(directory/'started.json',dict(started=time.time(),model=lock['model'],reasoning=lock['reasoning']))
    command=[shutil.which('codex'),'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-s','read-only','-C',str(SANDBOX),'-m',lock['model'],'-c','model_reasoning_effort="high"','--json','--output-schema',str(OUT/'schema.json'),'-o',str(directory/'response.json'),'-']
    begin=time.perf_counter()
    with (directory/'events.jsonl').open('xb') as output,(directory/'stderr.log').open('xb') as errors:
        result=subprocess.run(command,input=(directory/'prompt.txt').read_bytes(),stdout=output,stderr=errors)
    status=dict(exit_code=result.returncode,seconds=time.perf_counter()-begin,model_requested=lock['model'],reasoning_requested='high',tool_items=[],usage=None)
    for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line)
        if event.get('type')=='turn.completed':status['usage']=event.get('usage')
        item=event.get('item',{})
        if item.get('type') not in (None,'agent_message','reasoning'):status['tool_items'].append(item.get('type'))
    try:
        assert result.returncode==0 and not status['tool_items']
        response=read(directory/'response.json');signals=response['signals'];assert len(signals)<=2
        assert len({s['name'] for s in signals})==len(signals)
        for signal in signals:
            for record in read(directory/'input.json')['records']:evaluate(signal['expression'],record['features'])
        status['status']='PASS';status['signals']=len(signals);status['response_sha256']=sha(directory/'response.json')
    except Exception as exc:
        status['status']='FAILED';status['error']=str(exc)
    write(directory/'receipt.json',status)
    print(json.dumps(dict(name=name,**status)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--name')
    args=parser.parse_args();prepare() if args.action=='prepare' else run(args.name)
