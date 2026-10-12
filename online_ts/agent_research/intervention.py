"""Phase-0 contracts: paired responses and two offline Agent adapters.

No model calls, training or source-data writes. The caller supplies a frozen LLM
decision; both variants receive exactly the same facts and executor permissions.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def paired_response(parent, child, target):
    """Errors have shape [window, horizon, channel]; complement is elemental."""
    parent, child = np.asarray(parent, dtype=np.float64), np.asarray(child, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    if parent.shape != child.shape or parent.shape != target.shape or parent.ndim != 3:
        raise ValueError('unaligned error tensors / slice')
    if not np.isfinite(parent).all() or not np.isfinite(child).all():
        raise ValueError('nonfinite errors')
    if not target.any() or target.all():
        raise ValueError('target and complement must both be nonempty')
    def summarize(mask):
        p, c = parent[mask], child[mask]
        def scores(x):
            return dict(NMSE=float(np.mean(x*x)), NMAE=float(np.mean(abs(x))),
                        signed_bias=float(x.mean()))
        a, b = scores(p), scores(c)
        return dict(elements=len(p), parent=a, child=b,
                    delta={key:a[key]-b[key] for key in a},
                    gain_pct={key:100*(a[key]-b[key])/a[key] if a[key] else None
                              for key in ('NMSE', 'NMAE')})
    return {name:summarize(mask) for name,mask in
            [('global', np.ones_like(target)), ('target', target), ('complement', ~target)]}


def equivalent_processing(left, right, *, tolerance=1e-8):
    """Exact effective sampling/loss equivalence; input code hash is conservative.

    Near equality is reported for review, never silently deduplicated. Different
    programs with identical outputs on a finite probe are not proven equivalent.
    """
    if left['source_identity'] != right['source_identity']:
        return 'different_source'
    if left['input_program_sha256'] != right['input_program_sha256']:
        return 'unknown_program_equivalence'
    exact, near = True, True
    for key in ('start_row', 'probability', 'loss_weight'):
        a,b = np.asarray(left[key]),np.asarray(right[key])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return 'different_or_invalid'
        exact &= np.array_equal(a,b)
        near &= np.allclose(a,b,atol=tolerance,rtol=0)
    return 'exact' if exact else 'near_review' if near else 'different'


COMMON_PROMPT = """Research data interventions with the frozen backbone and executor.
All facts, parent recipes, failed experiments and costs are shared across variants.
Predict only from evidence available before the intervention. Do not equate gain
over B0 with gain over the parent, or recipe effectiveness with mechanism support.
Keep sample selection, weighting, sampling and representation results separate.
Return concise JSON: parent_id, action, evidence_ids, uncertainty, next_action.
You may keep the parent or stop. Never invent outcomes for unexecuted candidates.
"""
FULL_PROMPT = """Additionally preregister slice, expected global/target/complement
directions and falsifiers before execution. After paired feedback, update recipe
acceptance separately from hypothesis support. Counterexamples may motivate a
new measurable signal; it remains a proposal until it improves held-out response
prediction. Missing pre-state cannot be backfilled from an observed response.
Return prediction and hypothesis_update as separate fields; propose signals only
with explicit inputs, formula, time scope, complexity and falsification criteria.
"""


def context(facts, variant):
    if variant not in ('strong_prompt', 'full'):
        raise ValueError('unknown variant')
    return dict(variant=variant, facts=facts, facts_sha256=fingerprint(facts),
                prompt=COMMON_PROMPT + (FULL_PROMPT if variant == 'full' else ''),
                tools=['diagnose', 'propose_processing', 'train', 'paired_feedback', 'stop'],
                transport='offline_json; external API disabled')


def register_prediction(path, facts, decision):
    """Create-only preregistration. Evaluation must reference this exact digest."""
    required={'parent_id','action','slice','directions','falsifiers','evidence_ids'}
    if not required <= decision.keys() or not decision['falsifiers']:
        raise ValueError('incomplete preregistration')
    if set(decision['directions']) != {'global','target','complement'}:
        raise ValueError('all three response regions required')
    if any(v not in ('improve','unchanged','harm','uncertain') for v in decision['directions'].values()):
        raise ValueError('invalid direction')
    if facts.get('response_visible', False):
        raise ValueError('response already visible; retrospective record only')
    if 'parent_id' in facts and facts['parent_id']!=decision['parent_id']:
        raise ValueError('parent does not match prestate')
    record=dict(facts_sha256=fingerprint(facts), decision=decision,
                registered_utc=datetime.now(timezone.utc).isoformat())
    record['sha256']=fingerprint(record)
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False)
    return record


def dual_update(response, *, preregistered, matched_control_passed,
                independent_confirmation, directions=None,
                additional_falsifiers_passed=None,
                min_gain_pct=1., mae_guardrail_pct=1., unchanged_band_pct=.25):
    """Deterministic evidence gate, not proof of a unique physical mechanism."""
    global_gain=response['global']['gain_pct']['NMSE']
    mae_gain=response['global']['gain_pct']['NMAE']
    local_gain=response['target']['gain_pct']['NMSE']
    if global_gain is None or mae_gain is None or local_gain is None:
        return dict(recipe='insufficient', hypothesis='insufficient',next_action='measure')
    admissible=global_gain >= min_gain_pct and mae_gain >= -mae_guardrail_pct
    recipe=('confirmed' if independent_confirmation else 'confirmation_candidate') if admissible else 'retain_parent'
    if not preregistered:
        hypothesis='retrospective_only'
    elif directions is None or set(directions) != {'global','target','complement'}:
        hypothesis='insufficient_prediction'
    elif local_gain <= 0:
        hypothesis='not_supported'
    else:
        supported=[]
        for region,direction in directions.items():
            gain=response[region]['gain_pct']['NMSE']
            if direction not in ('improve','unchanged','harm','uncertain'):
                raise ValueError('invalid direction')
            if direction=='uncertain':continue
            if gain is None:
                supported.append(False)
            else:
                supported.append(gain>unchanged_band_pct if direction=='improve' else
                                 gain < -unchanged_band_pct if direction=='harm' else
                                 abs(gain)<=unchanged_band_pct)
        if not supported:
            hypothesis='insufficient_prediction'
        elif not all(supported):
            hypothesis='response_prediction_contradicted'
        elif additional_falsifiers_passed is False:
            hypothesis='falsifier_triggered'
        elif additional_falsifiers_passed is not True:
            hypothesis='pending_falsifier_review'
        elif matched_control_passed and independent_confirmation:
            hypothesis='supported_not_unique_cause'
        else:
            hypothesis='local_support_pending_control_or_confirmation'
    return dict(recipe=recipe,hypothesis=hypothesis,
                next_action='confirm' if admissible and not independent_confirmation else 'stop_or_revise')


def signal_value(prediction_before, prediction_after, observed, *,
                 construction_sources, selection_sources):
    """Direction accuracy on disjoint sources; labels -1/0/+1, 0 = uncertain."""
    if not construction_sources or not selection_sources or set(construction_sources)&set(selection_sources):
        raise ValueError('source isolation required')
    arrays=[np.asarray(x) for x in (prediction_before,prediction_after,observed)]
    if not arrays[0].size or len({x.shape for x in arrays}) != 1 or any(x.ndim!=1 for x in arrays):
        raise ValueError('aligned nonempty labels required')
    if any(not np.isin(x,[-1,0,1]).all() for x in arrays):
        raise ValueError('invalid direction labels')
    before,after,y=arrays
    return dict(before_accuracy=float(np.mean(before==y)),after_accuracy=float(np.mean(after==y)),
                before_decisive_coverage=float(np.mean(before!=0)),after_decisive_coverage=float(np.mean(after!=0)),
                before_harm_missed=int(np.sum((y==-1)&(before==1))),
                after_harm_missed=int(np.sum((y==-1)&(after==1))),
                gain=float(np.mean(after==y)-np.mean(before==y)),
                status='screening_only; freeze and independently confirm before promotion')


def counterexamples(records):
    """Only eligible, preregistered, already-observed development records."""
    result=[]
    for row in records:
        if row.get('scope')!='development' or not row.get('prestate_locked'):
            continue
        prediction=row.get('predicted_direction'); observed=row.get('observed_direction')
        if prediction not in (-1,0,1) or observed not in (-1,0,1):
            continue
        if prediction and observed and prediction!=observed:
            result.append({key:row[key] for key in
                           ('record_id','parent_id','action','prestate','predicted_direction','observed_direction')})
    return result


def measure_signal(prestate, rule, allowed_features):
    """Minimal transparent condition rule, no arbitrary eval or identity features.

    This first prototype measures one or two numeric pre-action features. Agent
    code for new features continues through the existing isolated diagnostic tool.
    """
    if set(rule) != {'operator','conditions'} or rule['operator'] not in ('all','any'):
        raise ValueError('invalid rule')
    conditions=rule['conditions']
    if not 1<=len(conditions)<=2:raise ValueError('condition complexity exceeded')
    values=[]
    for condition in conditions:
        if set(condition)!={'feature','comparison','threshold'}:raise ValueError('invalid condition')
        key=condition['feature'];threshold=condition['threshold']
        if key not in allowed_features:raise ValueError('undeclared or identity feature')
        if key not in prestate or prestate[key] is None:return None
        value=prestate[key]
        if not np.isfinite(value) or not np.isfinite(threshold):raise ValueError('nonfinite signal')
        if condition['comparison']=='gt':values.append(value>threshold)
        elif condition['comparison']=='le':values.append(value<=threshold)
        else:raise ValueError('unsupported comparison')
    return bool(all(values) if rule['operator']=='all' else any(values))
