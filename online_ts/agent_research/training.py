"""Durable scientific identity and budget accounting around the trusted GPU worker."""
from pathlib import Path
import sys
import time

from ..config import GateError, digest
from ..selection.common import file_hash, read_json, write_json
from .native import run_process
from .processing import parameters


class TrainingAdapter:
    def __init__(self,repo,task,guards):
        self.repo,self.task,self.guards=repo,task,guards

    def evaluate(self,candidate_id,asset,config,*,transform=None,seed=None,control='aligned',control_seed=20261009):
        config=parameters(config)
        identity={'base_root':str(self.task.base_root),'features_path':str(self.task.feature_path),
            'protected_assets':dict(self.guards),'processing_path':str(asset) if asset else None,
            'processing_sha256':file_hash(asset) if asset else None,'parameters':config,
            'seed':self.repo.config['seed'] if seed is None else seed,
            'input_transform_path':str(transform) if transform else None,
            'input_transform_sha256':file_hash(transform) if transform else None,
            'input_state_path':str(Path(transform).with_name('processor_state.json')) if transform else None,
            'input_state_sha256':file_hash(Path(transform).with_name('processor_state.json')) if transform else None,
            'control':control,'control_seed':control_seed,
            'scope':self.repo.config['scope'],
            'worker_sha256':file_hash(Path(__file__).with_name('trainer.py')),
            'torch_runtime':'2.14.0+cu130','numpy_runtime':'2.4.4'}
        job_id=digest(identity); folder=self.repo.folder/'training'/job_id
        if (folder/'result.json').exists():
            result=read_json(folder/'result.json')
            if result['identity']!=identity or any(file_hash(folder/name)!=value for name,value in result['assets'].items()):
                raise GateError('RESEARCH_CACHED_TRAINING_CHANGED')
            return result
        if folder.exists():
            raise GateError('RESEARCH_UNFINISHED_TRAINING_NOT_REPLAYED')
        used=list(self.repo.iter_records('research_training'))
        limit=self.repo.config.get('max_training_jobs')
        if 'max_training_jobs' not in self.repo.config and self.repo.config['max_candidates'] is not None:
            limit=self.repo.config['max_candidates']+1
        if limit is not None and len(used)>=limit:
            raise GateError('RESEARCH_TRAINING_BUDGET')
        folder.mkdir(parents=True)
        job={'identity':identity,'job_id':job_id,'config':config,'seconds':300}
        write_json(folder/'job.json',job)
        record={'record_id':'training:'+job_id,'candidate_id':candidate_id,'job_id':job_id,
                'status':'started','identity':identity}
        self.repo.put(record['record_id'],'research_training',record,time.time_ns())
        execution=run_process([sys.executable,'-B','-m','online_ts.agent_research.trainer','--job',str(folder/'job.json')],
            cwd=Path(__file__).resolve().parents[2],outputs=folder,seconds=310,gpu=True)
        write_json(folder/'execution.json',execution)
        if execution['status']!='completed' or not (folder/'result.json').exists():
            record.update(status='failed',elapsed_seconds=execution['elapsed_seconds'],error=execution['error'] or 'GPU_WORKER_FAILED')
            self.repo.put(record['record_id'],'research_training',record,time.time_ns(),replace=True)
            raise GateError('RESEARCH_GPU_EXECUTION_FAILED')
        result=read_json(folder/'result.json')
        if result['identity']!=identity:
            raise GateError('RESEARCH_GPU_RECEIPT_IDENTITY')
        record.update(status='completed',elapsed_seconds=execution['elapsed_seconds'],
                      result_sha256=file_hash(folder/'result.json'),metrics=result['metrics'])
        self.repo.put(record['record_id'],'research_training',record,time.time_ns(),replace=True)
        return result
