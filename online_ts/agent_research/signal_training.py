"""New study worker and durable grouped evaluation, isolated from old run snapshots."""
from pathlib import Path
import sys
import time

from ..config import GateError
from ..selection.common import file_hash,read_json,write_json
from .native import run_process
from .training import TrainingAdapter

class SignalTrainingAdapter(TrainingAdapter):
    worker_module='online_ts.agent_research.signal_trainer'
    worker_path=Path(__file__).with_name('signal_trainer.py')

    def evaluate(self,*args,**kwargs):
        result=super().evaluate(*args,**kwargs)
        folder=self.repo.folder/'training'/result['job_id']
        if 'scope_metrics.json' in result['assets']:return result
        if (folder/'scope_audit_started.json').exists():raise GateError('SIGNAL_UNFINISHED_AUDIT_NOT_REPLAYED')
        write_json(folder/'scope_audit_started.json',{'job_id':result['job_id']})
        audit=run_process([sys.executable,'-B','-m','online_ts.agent_research.signal_evaluation','--folder',str(folder)],
            cwd=Path(__file__).resolve().parents[2],outputs=folder,seconds=120,gpu=True)
        write_json(folder/'scope_audit_execution.json',audit)
        self.repo.put('scope-audit:'+result['job_id'],'evaluation_audit',{
            'job_id':result['job_id'],'status':audit['status'],'elapsed_seconds':audit['elapsed_seconds'],'training_calls':0},time.time_ns())
        if audit['status']!='completed' or not (folder/'scope_metrics.json').exists():
            raise GateError('SIGNAL_SCOPE_EVALUATION_FAILED')
        result['assets']['scope_metrics.json']=file_hash(folder/'scope_metrics.json')
        write_json(folder/'result.json',result)
        record=self.repo.get('training:'+result['job_id']);record['result_sha256']=file_hash(folder/'result.json')
        self.repo.put(record['record_id'],'research_training',record,time.time_ns(),replace=True)
        return result
