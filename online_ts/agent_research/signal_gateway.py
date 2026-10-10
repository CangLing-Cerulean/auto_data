"""Study tools: preregistered deficits, executable controls and feedback visibility."""
from copy import deepcopy
from pathlib import Path
import time
import zipfile

import numpy as np

from ..config import GateError,canonical,digest
from ..selection.common import file_hash,read_json,write_json
from .code_boundary import check_builder,execute_bounded_job
from .experience import ROOT
from .jobs import prepare_job
from .processing import DEFAULTS
from .runtime import ResearchGateway,TOOLS
from .signal_training import SignalTrainingAdapter

NEW_TOOLS=deepcopy(TOOLS)
NEW_TOOLS['propose_processing'].update(signal_id='registered signal ID, or empty if unassessed',
    control_code='independent executable control asset builder, or empty if unavailable')
NEW_TOOLS['read_baseline_error']={'group':'one baseline_error_groups name','channels':'array of exact sensor names'}
NEW_TOOLS['register_signal']={
    'signal_id':'unique ID <=40 characters','group':'baseline_error_groups name','channels':'array of exact sensor names',
    'reference':'persistence | overall','minimum_gap_pct':'finite >=0',
    'observed_gap_pct':'finite measured relative excess percentage',
    'training_claim':{'array':'values or a named T window feature','channel':'sensor name for values; empty otherwise',
        'statistic':'mean | std | min | max | fraction_zero | quantile','quantile':'0..1, used only for quantile',
        'value':'finite measured value'},
    'hypothesis':'testable processing mechanism','alternative':'competing explanation',
    'falsification':'what refutes it','local_gain_pct':'finite >=3','evidence_refs':'visible evidence IDs'}

def scope_summary(audit,group,channels,names):
    if group not in audit['groups']['best.pt']:raise GateError('SIGNAL_GROUP')
    if not isinstance(channels,list) or not channels or len(channels)!=len(set(channels)) or any(c not in names for c in channels):
        raise GateError('SIGNAL_CHANNELS')
    ids=[names.index(c) for c in channels];r=audit['groups']['best.pt'][group]
    return {'group':group,'channels':channels,'windows':r['windows'],
        'NMSE':float(np.mean(np.asarray(r['per_channel_NMSE'])[ids])),
        'NMAE':float(np.mean(np.asarray(r['per_channel_NMAE'])[ids])),
        'persistence_NMSE':float(np.mean(np.asarray(r['persistence_per_channel_NMSE'])[ids])),
        'persistence_NMAE':float(np.mean(np.asarray(r['persistence_per_channel_NMAE'])[ids]))}

def verify_training_claim(claim,inputs,names):
    required={'array','channel','statistic','quantile','value'}
    if not isinstance(claim,dict) or set(claim)!=required:raise GateError('SIGNAL_T_CLAIM_SCHEMA')
    with np.load(inputs,allow_pickle=False) as saved:
        if claim['array']=='values':
            if claim['channel'] not in names:raise GateError('SIGNAL_T_CHANNEL')
            x=saved['values'][:,names.index(claim['channel'])].astype(np.float64)
        else:
            if claim['array'] not in saved.files or claim['channel']!='':raise GateError('SIGNAL_T_ARRAY')
            x=saved[claim['array']]
            if x.ndim!=1 or x.dtype.kind not in 'fiu' or len(x)!=len(saved['start_row']):raise GateError('SIGNAL_T_WINDOW_FEATURE')
            x=x.astype(np.float64)
    stats={'mean':lambda:np.mean(x),'std':lambda:np.std(x),'min':lambda:np.min(x),'max':lambda:np.max(x),
        'fraction_zero':lambda:np.mean(x==0),'quantile':lambda:np.quantile(x,claim['quantile'])}
    if claim['statistic'] not in stats or type(claim['quantile']) not in (int,float) or not 0<=claim['quantile']<=1:
        raise GateError('SIGNAL_T_STATISTIC')
    observed=claim['value']
    if type(observed) not in (int,float) or not np.isfinite(observed):raise GateError('SIGNAL_T_VALUE')
    actual=float(stats[claim['statistic']]())
    return {'claimed':observed,'reference_value':actual,'matches':bool(np.isclose(observed,actual,atol=1e-6,rtol=1e-4)),
        'array':claim['array'],'channel':claim['channel'],'statistic':claim['statistic'],
        'scope':'T_normalized_rows' if claim['array']=='values' else 'T_legal_windows','independent_reference_calculation':True}

class SignalGateway(ResearchGateway):
    environment_kind='native_signal_development_study'
    def __init__(self,repo,task,library,protocol_folder):
        super().__init__(repo,task,library)
        self.protocol_folder=Path(protocol_folder)
        self.guards.update({str(self.protocol_folder/n):file_hash(self.protocol_folder/n) for n in ('protocol.json','PROTOCOL.md')})
        self.adapter=SignalTrainingAdapter(repo,task,self.guards)

    @staticmethod
    def check_code(code):check_builder(code)

    def execute_generated(self,folder,*,seconds=30):
        result=execute_bounded_job(folder,self.guards,seconds=seconds)
        if result.get('information_boundary',{}).get('violations'):
            self.repo.put('boundary:'+str(folder),'information_violation',{'job':str(folder),
                'violations':result['information_boundary']['violations']},time.time_ns())
        path=Path(folder)/'outputs/processing.npz'
        if result['status']=='completed' and path.exists():
            try:
                with zipfile.ZipFile(path) as zipped:
                    shapes={}
                    for member in zipped.infolist():
                        with zipped.open(member) as stream:
                            version=np.lib.format.read_magic(stream)
                            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                            shape,order,dtype=reader(stream);shapes[member.filename]={'shape':list(shape),'dtype':str(dtype)}
                    expected=len(self.task.data['start_row'])
                    if any(v['shape']!=[expected] for v in shapes.values()):
                        result.update(status='failed',error='PROCESSING_ARRAY_SHAPE: '+canonical({'expected':[expected],'actual':shapes}),stdout='')
                        write_json(Path(folder)/'execution.json',result)
            except (ValueError,OSError,zipfile.BadZipFile):pass
        return result

    def catalog(self):return NEW_TOOLS
    def baseline_audit(self):
        base=self.baseline()
        return read_json(self.repo.folder/'training'/base['job_id']/'scope_metrics.json')
    def signals(self):return list(self.repo.iter_records('registered_signal'))

    def results(self):
        value=super().results()
        pending=[r['candidate_id'] for r in self.candidates() if r['status'] in ('trained','failed') and not r.get('hypothesis_updated')]
        value.update(pending_update=pending[:10],pending_update_count=len(pending),
            registered_signals=[{'signal_id':r['signal_id'],'group':r['group'],'channels':r['channels']} for r in self.signals()[-4:]])
        return value

    def execute(self,record):
        tool,args=record['selected_action'],record['arguments']
        if tool not in NEW_TOOLS or set(args)!=set(NEW_TOOLS[tool]):raise GateError('SIGNAL_TOOL_ARGUMENTS')
        self.repo.check_binding(record['binding'])
        if tool=='describe_task':
            return {'config':self.repo.config,'tools':self.catalog(),'parameter_ranges':{k:[v] for k,v in DEFAULTS.items()},
                'baseline_error_groups':list(self.baseline_audit()['groups']['best.pt']),
                'sensor_columns':self.task.task['sensor_columns'],'scope_schema':'channels and existing input-defined groups; additional free diagnoses remain unassessed'}
        if tool=='read_baseline_error':
            summary=scope_summary(self.baseline_audit(),args['group'],args['channels'],self.task.task['sensor_columns'])
            summary['overall_NMSE']=self.baseline()['metrics']['NMSE']
            for name in ('persistence','overall'):
                comparator=summary['persistence_NMSE'] if name=='persistence' else summary['overall_NMSE']
                summary[name+'_gap_pct']=100*(summary['NMSE']/max(comparator,1e-12)-1)
            ref='baseline_scope:'+digest(args)[:24]
            self.journal.evidence(ref,dataset_id=self.task.task['dataset_id'],role='development_metrics',
                artifact_sha256=digest(summary),summary=canonical(summary))
            return summary|{'evidence_id':ref,'baseline_only_not_candidate_feedback':True}
        if tool=='register_signal':
            name=args['signal_id']
            import re
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',name):raise GateError('SIGNAL_ID')
            if any(r.get('signal_id')==name for r in self.candidates()):raise GateError('SIGNAL_AFTER_PROPOSAL')
            self.journal._refs(args['evidence_refs'])
            if not any(ref.startswith('diagnostic:') for ref in args['evidence_refs']):raise GateError('SIGNAL_CODED_DIAGNOSTIC_REFERENCE_REQUIRED')
            for k in ('hypothesis','alternative','falsification'):
                if not isinstance(args[k],str) or not 1<=len(args[k])<=2000:raise GateError('SIGNAL_HYPOTHESIS_FIELD')
            for k in ('minimum_gap_pct','observed_gap_pct','local_gain_pct'):
                if type(args[k]) not in (int,float) or not np.isfinite(args[k]):raise GateError('SIGNAL_FINITE_NUMERIC')
            if args['minimum_gap_pct']<0 or args['local_gain_pct']<3 or args['reference'] not in ('persistence','overall'):
                raise GateError('SIGNAL_THRESHOLD')
            summary=scope_summary(self.baseline_audit(),args['group'],args['channels'],self.task.task['sensor_columns'])
            comparator=summary['persistence_NMSE'] if args['reference']=='persistence' else self.baseline()['metrics']['NMSE']
            gap=100*(summary['NMSE']/max(comparator,1e-12)-1)
            t_claim=verify_training_claim(args['training_claim'],self.inputs,self.task.task['sensor_columns'])
            proof={'baseline_gap_pct':gap,'reported_gap_matches':bool(np.isclose(gap,args['observed_gap_pct'],atol=1e-4,rtol=1e-4)),
                'prediction_deficit_present':gap>args['minimum_gap_pct'],'training_claim':t_claim,
                'F':bool(t_claim['matches'] and np.isclose(gap,args['observed_gap_pct'],atol=1e-4,rtol=1e-4)),
                'P':bool(gap>args['minimum_gap_pct']),'interpretation':'registered comparative prediction weakness, not a physical fault'}
            # Verification is private until the final frozen choice; never repair an Agent's scientific claim.
            self.repo.put('signal:'+name,'registered_signal',deepcopy(args)|{'registered_ns':record['created_at']},time.time_ns())
            self.repo.put('signal-verification:'+name,'signal_verification',proof,time.time_ns())
            return {'signal_id':name,'registered_before_execution':True,'external_fact_scoring':'withheld_until_frozen_submission'}
        if tool=='propose_processing':
            if args['parameters']!={}:raise GateError('SIGNAL_FIXED_PARAMETERS')
            if args['signal_id'] and not any(r['signal_id']==args['signal_id'] for r in self.signals()):raise GateError('SIGNAL_NOT_REGISTERED')
            if not isinstance(args['control_code'],str):raise GateError('SIGNAL_CONTROL_CODE')
            if args['control_code']:self.check_code(args['control_code'])
            mapped=deepcopy(record);mapped['arguments']={k:v for k,v in args.items() if k not in ('signal_id','control_code')}
            result=super().execute(mapped)
            if result.get('signal_id',args['signal_id'])!=args['signal_id'] or result.get('control_code',args['control_code'])!=args['control_code']:
                raise GateError('SIGNAL_CANDIDATE_IMMUTABLE')
            result.update(signal_id=args['signal_id'],control_code=args['control_code']);self.save(result)
        elif tool=='update_hypothesis':
            old=self.candidate(args['candidate_id'])
            if old.get('hypothesis_updated'):return {'candidate_id':old['candidate_id'],'recorded_update':old['post_result_update'],'already_frozen':True}
            if self.repo.config['feedback_mode']=='withheld' and args['update']['assessment']!='inconclusive':raise GateError('SIGNAL_OUTCOME_WITHHELD')
            result=super().execute(record)
        elif tool=='read_evidence' and self.repo.config['feedback_mode']=='withheld' and args['evidence_id'].startswith('baseline_scope:'):
            if type(args['offset'])is not int or args['offset']<0 or type(args['limit'])is not int or not 100<=args['limit']<=1000:
                raise GateError('RESEARCH_RECORD_PAGE')
            row=self.journal.repo.get('evidence:'+args['evidence_id']);text=canonical(row);end=args['offset']+args['limit']
            return {'record_id':args['evidence_id'],'text':text[args['offset']:end],'total_chars':len(text),'next_offset':end if end<len(text) else None}
        else:result=super().execute(record)
        if tool in ('propose_processing','run_processing','evaluate_candidate'):
            public={k:result[k] for k in ('candidate_id','index','status','receipt_sha256','error','signal_id') if k in result}
            if self.repo.config['feedback_mode']=='visible' and result.get('result'):
                public['metrics']=result['result']['metrics']
                if result.get('signal_id'):
                    s=self.repo.get('signal:'+result['signal_id'])
                    audit=read_json(self.repo.folder/'training'/result['result']['job_id']/'scope_metrics.json')
                    public['local_metrics']=scope_summary(audit,s['group'],s['channels'],self.task.task['sensor_columns'])
            return public
        return result

    def submit(self,args):
        pending=[r['candidate_id'] for r in self.candidates() if r['status'] not in ('trained','failed') or not r.get('hypothesis_updated')]
        if pending:raise GateError('POST_RESULT_UPDATE_REQUIRED: '+canonical({'count':len(pending),'first':pending[:10]}))
        rows=self.candidates();trained=[r for r in rows if r['status']=='trained']
        if not isinstance(args['reason'],str) or not 1<=len(args['reason'])<=2000:raise GateError('SUBMISSION_REASON')
        if self.repo.config['feedback_mode']=='visible':
            base=self.baseline()['metrics'];choices=[('B0',base)]+[(r['candidate_id'],r['result']['metrics']) for r in trained if r['result']['metrics']['NMAE']<=base['NMAE']*1.01]
            if args['candidate_id']!=min(choices,key=lambda v:v[1]['NMSE'])[0]:raise GateError('SELECT_MEASURED_ELIGIBLE_BEST')
        elif args['candidate_id'] not in {'B0'}|{r['candidate_id'] for r in trained}:raise GateError('SELECT_FROZEN_TRAINED_POLICY')
        final={'run_id':self.repo.run_id,'group':self.repo.config['group'],'selected_candidate':args['candidate_id'],
            'reason':args['reason'],'scope':'development_only','independent_task':False,'candidate_count':len(rows),
            'diagnostic_count':self.repo.count_records('diagnostic_job'),'registered_signals':len(self.signals()),
            'information_violations':self.repo.count_records('information_violation'),'journal':self.journal.summary(),
            'diagnosis_correctness':'pending_external_scoring','empty_search':not rows}
        write_json(self.repo.folder/'final_policy.json',final);self.repo.put('final_policy','final_policy',final,time.time_ns())
        return {'agent_finalized':True,'final_policy':final}
