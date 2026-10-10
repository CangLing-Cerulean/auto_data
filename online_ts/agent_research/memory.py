"""Versioned evidence memory: measured receipts and revisable interpretations.

No search result is an action gate. Only approved development receipts enter
this store; source files and generated views have independent frozen hashes.
"""
from copy import deepcopy
from pathlib import Path
import re

from ..config import GateError, digest
from ..selection.common import file_hash, read_json, write_json
from .experience import ROOT, STUDY, verify_library as verify_v1

MEMORY = STUDY / 'memory/v2'
POLICY = {
    'evidence_is_advisory': True, 'method_family_bans': False,
    'revisiting_and_replication_allowed': True, 'automatic_strategy_selection': False,
    'interpretations_are_correctness_labels': False,
    'human_curation': 'prepared_before_runs; no online human strategy additions',
    'independent_search_memory': 'freeze each run; do not pool peer run discoveries',
    'new_task_history': 'exclude target dataset family',
    'test_metrics': 'excluded', 'exploration_effect': 'unassessed_until_controlled_comparison',
    'research_goal': 'diagnose measured performance deficits; test processing hypotheses; assess signal and feedback independently',
    'distribution_phenomena_are_performance_causes': False,
}
# Search aliases only; these do not prescribe recipes, thresholds or actions.
CONCEPTS = {
 'variation': ['variation','variance','quiet','static','变异','变化','静态','静默','低变化'],
 'weighting': ['weight','weighting','downweight','降权','权重','重加权','bilevel','双层'],
 'sampling': ['sampling','sample','采样','抽样','importance','重要性'],
 'deletion': ['drop','deletion','delete','删除','剔除'],
 'coverage': ['coverage','group','cluster','覆盖','分组','簇','稀有'],
 'time': ['time','temporal','recent','cadence','时间','时序','近期','间隔'],
 'geometry': ['leverage','geometry','杠杆','几何'],
 'centering': ['center','centering','中心化'],
 'feedback': ['feedback','learning','progress','反馈','学习进展'],
 'quality': ['quality','constant','measurement','质量','恒定','测量','故障'],
 'control': ['control','shuffle','random','对照','打乱','随机'],
}


def _source_path(value):
    # Input catalogue accepts exact result receipts, never recursive discovery.
    if not isinstance(value,str) or not re.fullmatch(
        r'research_runs/results/[A-Za-z0-9_-]+/result\.json|'
        r'research_runs/agent_research/acceptance/20261009/approved_engineering/[A-Za-z0-9_-]+\.json',value):
        raise GateError('MEMORY_SOURCE_NOT_APPROVED')
    return value


def _frozen(library):
    if library.get('version')!='evidence-memory-v2' or library.get('policy')!=POLICY:
        raise GateError('MEMORY_POLICY')
    if digest({k:v for k,v in library.items() if k!='sha256'})!=library.get('sha256'):
        raise GateError('MEMORY_IDENTITY')


def _outcome(raw):
    if isinstance(raw.get('groups'),dict) and 'all' in raw['groups']:
        return 'development_validation', raw['groups'], raw
    future=raw.get('future')
    if isinstance(future,dict) and 'all' in future.get('groups',{}):
        return 'train_internal_forward', future['groups'], future
    if isinstance(raw.get('metrics'),dict) and 'NMSE' in raw['metrics']:
        return 'engineering_development', {'all':raw['metrics']}, raw
    return 'training_auxiliary_no_forecast_evaluation', {}, {}


def _run_record(entry,raw):
    identity=raw.get('identity',raw.get('config',{}))
    scope,groups,metrics_raw=_outcome(raw)
    if entry['kind']=='engineering':scope='engineering_development'
    model=identity.get('model','original_DLinear' if entry['kind']!='auxiliary' else 'auxiliary')
    recipe=identity.get('recipe',identity.get('policy',entry['run_id']))
    permitted_identity={k:identity[k] for k in ('model','recipe','policy','seed','origin','preprocessing',
        'drop_block','protocol_sha256','manifest_sha256','code_sha256','core_sha256','policy_sha256') if k in identity}
    observed={
        'run_id':entry['run_id'],'evaluation_scope':scope,'model':model,'recipe':recipe,
        'identity':permitted_identity,'groups':groups,
        'per_channel_NMSE':metrics_raw.get('per_channel_NMSE',[]),
        'per_channel_NMAE':metrics_raw.get('per_channel_NMAE',[]),
        'steps_completed':raw.get('steps_completed'), 'best_step':raw.get('best_step'),
        'elapsed_seconds':raw.get('elapsed_seconds'),
    }
    if scope=='training_auxiliary_no_forecast_evaluation':
        observed['auxiliary_metrics']={k:raw[k] for k in ('in_sample_domain_accuracy',
            'in_sample_domain_BCE','eligible_windows','fitted_rows_per_domain') if k in raw}
    metrics=groups.get('all',{})
    fact=f"{entry['run_id']}：{scope}；recipe={recipe}；"
    fact+=f"NMSE={metrics['NMSE']:.8g}, NMAE={metrics['NMAE']:.8g}。" if 'NMSE' in metrics else '未提供预测泛化指标。'
    limits='单次记录不证明机制或整个方法族有效/无效；不同评价范围与模型不可直接混排。'
    if entry['kind']=='engineering':limits+='工程验收不计正式能力证据。'
    return {'record_id':'experience:run:'+entry['run_id'],'dataset_id':entry['dataset_id'],
        'dataset_family':entry['dataset_family'],'record_type':'measured_run',
        'evaluation_scope':scope,'historical_intervention':'unknown',
        'provenance':'historical_receipt' if entry['kind']!='engineering' else 'engineering_receipt',
        'tags':entry['run_id'].lower().split('_'), 'lesson':fact,'status':'observed_not_causal',
        'limits':limits,'observation':observed,
        'interpretation':{'text':'此记录未自动生成因果结论。','verification':'unassessed'},
        'applicability':{'dataset':entry['dataset_id'],'model':model,'scope':scope},
        'unknowns':['跨种子稳定性须结合其他记录','机制及方法族外推未由该单次结果确认'],
        'related_runs':[], 'source':{'path':entry['source'],'sha256':entry['sha256'],
            'storage_path':'sources/'+entry['source'],'excerpt_sha256':digest(observed)}}


def _conditional_record(old,catalog,entries):
    item=deepcopy(old);item.update(record_type='conditional_interpretation',dataset_family=catalog['dataset_family'])
    item['observation']={'report_excerpt':old['source']['excerpt']}
    item['interpretation']={'text':old['lesson'],'verification':'unassessed_human_curated_summary'}
    item['applicability']={'dataset':old['dataset_id'],'scope':'historical_development_only'}
    item['unknowns']=[old['limits'],'独立机制检验及跨任务外推未由摘要确认']
    ids=re.findall(r'\b[A-Z]\d{3}(?:_[A-Za-z0-9]+)*\b',old['source']['excerpt'])
    item['related_runs']=sorted({e['run_id'] for e in entries if any(e['run_id']==i or e['run_id'].split('_')[0]==i for i in ids)})
    item['source']['storage_path']='v1_source:'+old['source']['path']
    return item


def build_memory(folder=MEMORY,*,root=ROOT):
    folder=Path(folder);catalog=read_json(folder/'input_catalog.json')
    if catalog['version']!='approved-development-memory-inputs-v1':raise GateError('MEMORY_CATALOG')
    v1=read_json(Path(root)/'research_runs/agent_research/experience_library.json')
    verify_v1(v1,root=root)
    records=[];entries=catalog['entries'];seen=set()
    for entry in entries:
        source=_source_path(entry['source'])
        if entry['run_id'] in seen:raise GateError('MEMORY_DUPLICATE_RUN')
        seen.add(entry['run_id'])
        if entry['kind'] not in {'main','auxiliary','engineering'}:raise GateError('MEMORY_RUN_KIND')
        if entry['dataset_id']!=catalog['dataset_id'] or entry['dataset_family']!=catalog['dataset_family']:
            raise GateError('MEMORY_DATASET_FAMILY')
        path=folder/'sources'/source
        if file_hash(path)!=entry['sha256']:raise GateError('MEMORY_SOURCE_CHANGED')
        raw=read_json(path)
        if raw['run_id']!=entry['run_id']:raise GateError('MEMORY_RUN_ID')
        records.append(_run_record(entry,raw))
    historical={e['run_id'] for e in entries if e['kind']!='engineering'}
    if historical!=set(catalog['completed_run_ids']) or len(historical)!=catalog['declared_historical_count']:
        raise GateError('MEMORY_LEDGER_COVERAGE')
    for old in v1['records']:
        records.append(_conditional_record(old,catalog,entries))
    library={'version':'evidence-memory-v2','policy':POLICY,'scope':'development_only',
        'catalog_sha256':file_hash(folder/'input_catalog.json'),'v1_library_sha256':v1['sha256'],
        'coverage':{'historical_completed':len(historical),
            'historical_main':sum(e['kind']=='main' for e in entries),
            'historical_auxiliary':sum(e['kind']=='auxiliary' for e in entries),
            'engineering_receipts':sum(e['kind']=='engineering' for e in entries),
            'conditional_summaries':len(v1['records'])},'records':records}
    library['sha256']=digest(library)
    destination=folder/'library.json'
    if destination.exists() and read_json(destination)!=library:raise GateError('MEMORY_FROZEN_USE_NEW_VERSION')
    write_json(destination,library);verify_memory(library,folder=folder,root=root)
    return library


def verify_memory(library,*,folder=MEMORY,root=ROOT):
    _frozen(library);folder=Path(folder)
    if file_hash(folder/'input_catalog.json')!=library['catalog_sha256']:raise GateError('MEMORY_CATALOG_CHANGED')
    catalog=read_json(folder/'input_catalog.json')
    entries={e['run_id']:e for e in catalog['entries']};v1=read_json(Path(root)/'research_runs/agent_research/experience_library.json')
    verify_v1(v1,root=root)
    if v1['sha256']!=library['v1_library_sha256']:raise GateError('MEMORY_V1_CHANGED')
    old={r['record_id']:r for r in v1['records']};seen=set()
    for record in library['records']:
        if record['record_id'] in seen:raise GateError('MEMORY_DUPLICATE_RECORD')
        seen.add(record['record_id'])
        if record['record_type']=='measured_run':
            entry=entries[record['observation']['run_id']];_source_path(entry['source'])
            path=folder/'sources'/entry['source']
            if file_hash(path)!=entry['sha256']:raise GateError('MEMORY_SOURCE_CHANGED')
            if record!=_run_record(entry,read_json(path)):raise GateError('MEMORY_FACT_CHANGED')
        elif record['record_type']=='conditional_interpretation':
            previous=old[record['record_id']]
            if record!=_conditional_record(previous,catalog,list(entries.values())):
                raise GateError('MEMORY_INTERPRETATION_CHANGED')
        else:raise GateError('MEMORY_RECORD_TYPE')
    if len(seen)!=len(entries)+len(old):raise GateError('MEMORY_RECORD_COVERAGE')
    return len(seen)


class EvidenceMemory:
    def __init__(self,library,*,target_dataset,target_family,track,mode='full',folder=MEMORY,root=ROOT):
        if track not in {'experience_combination','new_task_discovery'} or mode not in {'full','none','other_tasks'}:
            raise GateError('MEMORY_INFORMATION_SCOPE')
        if not target_dataset or not target_family:raise GateError('MEMORY_TARGET_ID')
        self.library,self.folder,self.root=library,Path(folder),root
        self.track,self.mode,self.target_dataset,self.target_family=track,mode,target_dataset,target_family
        verify_memory(library,folder=folder,root=root)

    def records(self):
        _frozen(self.library)
        if self.mode=='none':return []
        exclude=self.track=='new_task_discovery' or self.mode=='other_tasks'
        return [r for r in self.library['records'] if not exclude or
            (r['dataset_id']!=self.target_dataset and r['dataset_family']!=self.target_family)]

    @staticmethod
    def brief(record):
        return {k:deepcopy(record[k]) for k in ('record_id','dataset_id','dataset_family','record_type',
            'evaluation_scope','lesson','limits','status','applicability','unknowns','related_runs','source')
            if k!='source'} | {'source':{k:record['source'][k] for k in ('path','sha256','excerpt_sha256')}}

    def result(self,records,**extra):
        return {'library_sha256':self.library['sha256'],'track':self.track,'mode':self.mode,
            'policy':POLICY,'records':[self.brief(r) for r in records],**extra}

    def list(self,offset=0,limit=10):
        if type(offset)is not int or offset<0 or type(limit)is not int or not 1<=limit<=20:
            raise GateError('MEMORY_PAGE')
        records=sorted(self.records(),key=lambda r:r['record_id'])
        return self.result(records[offset:offset+limit],total=len(records),offset=offset,
            next_offset=offset+limit if offset+limit<len(records) else None)

    def retrieve(self,query='',limit=5):
        if not isinstance(query,str) or len(query)>1000 or type(limit)is not int or not 1<=limit<=10:
            raise GateError('MEMORY_QUERY')
        query=query.lower().strip();words=set(re.findall(r'[a-z0-9_]+',query))
        concepts={c for c,aliases in CONCEPTS.items() if any(a in query for a in aliases)}
        def score(r):
            text=' '.join(r['tags']+[r['record_id'],r['lesson'],r['limits']]).lower()
            return sum(w in text for w in words)+sum(any(a in text for a in CONCEPTS[c]) for c in concepts)
        records=self.records()
        if query:records=[r for r in records if score(r)]
        records=sorted(records,key=lambda r:(-score(r),r['record_id']))
        return self.result(records[:limit],matched=len(records),query=query,
            no_match=bool(query and not records))

    def read(self,record_id):
        record=next((r for r in self.records() if r['record_id']==record_id),None)
        if record is None:raise GateError('MEMORY_RECORD_NOT_VISIBLE')
        if record['record_type']=='measured_run':
            path=self.folder/record['source']['storage_path']
            if file_hash(path)!=record['source']['sha256']:raise GateError('MEMORY_SOURCE_CHANGED')
        else:verify_v1(read_json(Path(self.root)/'research_runs/agent_research/experience_library.json'),root=self.root)
        return deepcopy(record)
