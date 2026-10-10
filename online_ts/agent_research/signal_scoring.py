"""Post-submission confirmation; calculated gates are distinct from mechanism review."""
from pathlib import Path
import time

import numpy as np

from ..config import GateError,digest
from ..selection.common import file_hash,read_json,write_json
from .experience import STUDY
from .jobs import prepare_job
from .memory import MEMORY
from .native import run_process
from .processing import DEFAULTS,validate_processing
from .runtime import ResearchRepository
from .signal_gateway import SignalGateway,scope_summary
from .task import TaskAssets
from .transforms import processor_metadata

def outcome_gates(outcomes,local_target):
    avg=lambda kind,metric:float(np.mean([r[kind][metric] for r in outcomes]))
    gain=100*(1-avg('candidate','NMSE')/avg('baseline','NMSE'))
    mae=100*(avg('candidate','NMAE')/avg('baseline','NMAE')-1)
    each=all(r['candidate']['NMSE']<r['baseline']['NMSE'] for r in outcomes)
    local=100*(1-float(np.mean([r['local_candidate']['NMSE'] for r in outcomes]))/
        float(np.mean([r['local_baseline']['NMSE'] for r in outcomes]))) if all(r.get('local_candidate') for r in outcomes) else None
    controls=all(r.get('control') for r in outcomes)
    advantage=100*(1-avg('candidate','NMSE')/avg('control','NMSE')) if controls else None
    local_advantage=100*(1-float(np.mean([r['local_candidate']['NMSE'] for r in outcomes]))/
        float(np.mean([r['local_control']['NMSE'] for r in outcomes]))) if controls and local is not None else None
    return {'mean_NMSE_gain_pct':gain,'mean_NMAE_deterioration_pct':mae,
        'each_seed_NMSE_improves':each,'E_numeric':bool(each and gain>=1 and mae<=1),
        'mean_local_NMSE_gain_pct':local,'L_numeric':None if local is None else bool(local>=local_target),
        'control_overall_advantage_pct':advantage,'control_local_advantage_pct':local_advantage,
        'C_numeric':None if advantage is None or local_advantage is None else bool(advantage>=.25 and local_advantage>0),
        'C_semantics':'unassessed_pending_primary_review','physical_causality':False}

def feedback_scores(gateway):
    checks=[];base=gateway.baseline()['metrics'];hidden=gateway.repo.config['feedback_mode']=='withheld'
    for r in gateway.candidates():
        update=r.get('post_result_update');expected=None
        if r['status']=='failed' or hidden:expected='inconclusive'
        elif r['status']=='trained':
            m=r['result']['metrics'];expected='supported' if m['NMSE']<=base['NMSE']*.99 and m['NMAE']<=base['NMAE']*1.01 else 'contradicted'
        actual=update.get('assessment') if update else None
        checks.append({'candidate_id':r['candidate_id'],'execution_status':r['status'],'assessment':actual,
            'visible_criterion_supported':None if hidden or r['status']!='trained' else expected=='supported',
            'classification_appropriate':bool(update and (actual==expected or (not hidden and r['status']=='trained' and actual=='inconclusive'))),
            'receipt_traceable':bool(update and update['receipt_sha256']==r.get('receipt_sha256')),
            'text_explanation':'requires_primary_review','withheld_outcome':hidden})
    return checks

def build_control(gateway,candidate):
    folder=gateway.repo.folder/'external_control'/candidate['candidate_id']
    if folder.exists():raise GateError('SIGNAL_CONTROL_NOT_REPLAYED')
    prepare_job(folder,candidate['control_code'],gateway.assets())
    execution=gateway.execute_generated(folder,seconds=45)
    if execution['status']!='completed':
        write_json(folder/'control_failure.json',{'status':'failed','error':execution['error'],'science_refuted':False})
        return None
    try:
        asset=folder/'outputs/processing.npz';validate_processing(asset,gateway.task.data)
        transform=folder/'outputs/processor.py';state=folder/'outputs/processor_state.json'
        if transform.exists() or state.exists():
            processor_metadata(transform,state)
            checked=run_process([__import__('sys').executable,'-B','-m','online_ts.agent_research.processor_check','--folder',str(folder)],
                cwd=Path(__file__).resolve().parents[2],outputs=folder/'outputs',seconds=45,gpu=True)
            write_json(folder/'processor_check_execution.json',checked)
            if checked['status']!='completed':raise GateError('SIGNAL_CONTROL_PROCESSOR')
        return {'asset':asset,'transform':transform if transform.exists() else None,'kind':'agent_registered_code',
            'asset_sha256':file_hash(asset),'control_code_sha256':digest(candidate['control_code'])}
    except (GateError,OSError,ValueError) as error:
        write_json(folder/'control_failure.json',{'status':'failed','error':str(error),'science_refuted':False});return None

def confirm_and_score(folder,protocol):
    folder=Path(folder)
    if (folder/'external_score.json').exists():raise GateError('SIGNAL_SCORE_NOT_REPLAYED')
    identity=read_json(folder/'preregistration.json');group=identity['config']['group']
    basic={'run_id':folder.name,'group':group,'search_completed':(folder/'search_result.json').exists(),
        'denominator_retained':True,'independent_task':False,'F':None,'P':None,'L':None,'C':None,'E':False,
        'operational_signal':False,'semantic_review':'pending_primary_review'}
    if not basic['search_completed']:
        basic.update(status='search_failed');write_json(folder/'external_score.json',basic);return basic
    from .signal_study import PILOT
    repo=ResearchRepository(folder.name,identity,folder=folder)
    gateway=SignalGateway(repo,TaskAssets(STUDY/'tasks/metropt3_dev'),read_json(MEMORY/'library.json'),PILOT)
    gateway.guards.update(identity['code']);base=gateway.baseline()
    basic.update(feedback_checks=feedback_scores(gateway),
        information_violations=repo.count_records('information_violation'),
        registered_facts=[{'signal_id':s['signal_id'],**repo.get('signal-verification:'+s['signal_id'])} for s in gateway.signals()])
    selected=read_json(folder/'final_policy.json')['selected_candidate']
    if selected=='B0':
        basic.update(status='baseline_retained_no_confirmed_intervention',selected_candidate='B0')
        write_json(folder/'external_score.json',basic);repo.export();return basic
    candidate=gateway.candidate(selected);signal=repo.get('signal:'+candidate['signal_id']) if candidate.get('signal_id') else None
    proof=repo.get('signal-verification:'+candidate['signal_id']) if signal else None
    frozen={'final_policy_sha256':file_hash(folder/'final_policy.json'),'processing_sha256':candidate['asset_sha256'],
        'control_code_sha256':digest(candidate['control_code']),'signal_sha256':digest(signal) if signal else None,
        'confirmation_seeds':protocol['confirmation_seeds'],'identity_sha256':identity['sha256']}
    write_json(folder/'confirmation_freeze.json',frozen)
    control=build_control(gateway,candidate) if candidate['control_code'] else None
    if not candidate['control_code'] and not candidate.get('input_transform_path'):
        control={'asset':Path(candidate['asset']),'transform':None,'kind':'group_shuffle'}
    outcomes=[]
    try:
        for seed in protocol['confirmation_seeds']:
            b=gateway.adapter.evaluate('external_B0',None,DEFAULTS,seed=seed)
            c=gateway.adapter.evaluate('external_selected',Path(candidate['asset']),DEFAULTS,seed=seed,transform=candidate.get('input_transform_path'))
            m=gateway.adapter.evaluate('external_matched',control['asset'],DEFAULTS,seed=seed,transform=control['transform'],
                control='group_shuffle' if control['kind']=='group_shuffle' else 'aligned',control_seed=protocol['control_seed']) if control else None
            local=lambda result:scope_summary(read_json(folder/'training'/result['job_id']/'scope_metrics.json'),
                signal['group'],signal['channels'],gateway.task.task['sensor_columns']) if signal else None
            row={'seed':seed,'baseline':b['metrics'],'candidate':c['metrics'],'control':m['metrics'] if m else None,
                'local_baseline':local(b),'local_candidate':local(c),'local_control':local(m) if m else None,
                'baseline_job':b['job_id'],'candidate_job':c['job_id'],'control_job':m['job_id'] if m else None}
            outcomes.append(row);write_json(folder/'confirmation_progress.json',{'results':outcomes,'completed_seeds':len(outcomes)})
        gates=outcome_gates(outcomes,signal['local_gain_pct'] if signal else 3)
        valid=basic['information_violations']==0 and gateway.journal.summary()['autonomy_eligible']
        basic.update(status='confirmed_pending_semantic_review',selected_candidate=selected,results=outcomes,gates=gates,
            F=proof['F'] if proof else None,P=proof['P'] if proof else None,L=gates['L_numeric'],
            C=None,E=bool(gates['E_numeric'] and valid),information_conditions_valid=valid,
            control_kind=control['kind'] if control else 'unavailable',
            operational_signal=None if signal and valid else False)
    except Exception as error:
        basic.update(status='external_confirmation_failed',error=str(error),completed_confirmation_seeds=len(outcomes))
    write_json(folder/'external_score.json',basic);repo.export();return basic
