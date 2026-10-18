"""Descriptive aggregation of audited conditional scores; no additional fitting."""
from pathlib import Path
import numpy as np
from diagnosis_pool_features import read,sha,write
ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_measurement_generation_r4'
rows=[];summary=[]
for mode in ('ordinary','mismatch'):
    means=[]
    for path in sorted((STAGE/'replay').glob(f'*_{mode}_conditional.json')):
        result=read(path);scores=[s['conditional'] for t in result['trace'] for s in t['event']['scores'].values()]
        mean=float(np.mean(scores));means.append(mean)
        rows.append(dict(mode=mode,family=result['family'],task=result['task'],mean_conditional_log_score_gain=mean,min_score=min(scores),max_score=max(scores),queries=len(result['trace']),experts=2,trace_sha256=sha(path)))
    summary.append(dict(mode=mode,task_equal_mean_conditional_log_score_gain=float(np.mean(means)),tasks=len(means)))
write(STAGE/'conditional_score_summary.json',dict(kind='posthoc_descriptive_audited_score_aggregation_not_a_new_success_criterion',rows=rows,summary=summary,no_new_fitting=True,script_sha256=sha(__file__)))
print(summary)
