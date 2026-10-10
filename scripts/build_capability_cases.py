"""Build public cases and withheld mechanical answer keys from exact historical receipts."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.agent_research.capability import PILOT
from online_ts.selection.common import read_json,write_json,file_hash

def source(name):
    path=PILOT/'sources'/name
    return {'source_id':name.removesuffix('.json'),'sha256':file_hash(path),'artifact':'sources/'+name}

direct=read_json(PILOT/'sources/C01_confirmation.json')
means=direct['confirmation_seed_means'];baseline=read_json(PILOT/'sources/shared_baseline.json')['metrics']
negative=read_json(PILOT/'sources/C02_candidate.json')['metrics']
conflict=read_json(PILOT/'sources/C03_candidate.json')['metrics']
failure=read_json(PILOT/'sources/C04_execution.json')
cases=[{'case_id':'C01','sources':[source('C01_confirmation.json')],
    'measurement':{'execution_status':'completed','baseline':means['baseline'],'candidate':means['source_half'],
        'matched_control':means['source_half_scramble'],'confirmation_seeds':[3026,3027,3028],
        'seed_measurements':{key:{role:direct['seeds'][key][role] for role in ('baseline','candidate','control')}
            for key in ('3026','3027','3028')},'evaluation_dataset':'previously_explored_metropt3_v1_development',
        'model':'fixed_original_DLinear','dose':'3000_steps_per_training','independent_dataset_used':False}},
    {'case_id':'C02','sources':[source('shared_baseline.json'),source('C02_candidate.json')],
     'measurement':{'execution_status':'completed','baseline':baseline,'candidate':negative,'training_seeds':[20261101],
        'evaluation_dataset':'metropt3_v1_development','model':'fixed_original_DLinear','matched_control':None,'independent_dataset_used':False}},
    {'case_id':'C03','sources':[source('shared_baseline.json'),source('C03_candidate.json')],
     'measurement':{'execution_status':'completed','baseline':baseline,'candidate':conflict,'training_seeds':[20261101],
        'evaluation_dataset':'metropt3_v1_development','model':'fixed_original_DLinear','matched_control':None,'independent_dataset_used':False}},
    {'case_id':'C04','sources':[source('C04_execution.json')],
     'measurement':{'execution_status':failure['status'],'error':failure['error'],'stage':failure['stage'],
        'baseline':None,'candidate':None,'evaluation_dataset':'metropt3_v1_development','independent_dataset_used':False}}]
answers={}
for case in cases:
    data=case['measurement'];done=data['execution_status']=='completed'
    changes={key:(data['candidate'][key]/data['baseline'][key]-1)*100 if done else None for key in ('NMSE','NMAE')}
    supported=done and changes['NMSE']<=-1 and changes['NMAE']<=1
    answers[case['case_id']]={'execution_status':data['execution_status'],'nmse_change_pct':changes['NMSE'],
        'nmae_change_pct':changes['NMAE'],'meets_processing_criterion':supported if done else None,
        'assessment':'supported' if supported else ('contradicted' if done else 'inconclusive'),
        'claim_scope':'confirmed_development' if case['case_id']=='C01' else ('single_seed_development' if done else 'execution_only'),
        'physical_causality_proven':False,'independent_task_proven':False,
        'action':'reuse_as_known_example' if supported else ('retain_baseline' if done else 'repair_execution_and_retest')}
if (PILOT/'cases.json').exists():raise RuntimeError('Frozen case bank already exists')
write_json(PILOT/'cases.json',{'version':'historical_capability_cases_v1','cases':cases,'test_data_used':False})
write_json(PILOT/'private_expected.json',answers)
write_json(PILOT/'sources_manifest.json',{name:{'sha256':file_hash(path),'bytes':path.stat().st_size}
    for path in sorted((PILOT/'sources').glob('*.json')) for name in [path.name]})
print('Four historical cases frozen; expected answers withheld from model tools',flush=True)
