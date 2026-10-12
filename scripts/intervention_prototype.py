"""Two comparable offline JSON adapters; no LLM API or training is called.

prepare writes a context for the same frozen LLM/manual decision transport used
by R1/R2. register locks a full-method prediction before response visibility.
review keeps raw feedback shared and adds deterministic dual updates only in full.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.agent_research.intervention import (
    context, counterexamples, dual_update, fingerprint, register_prediction,
    measure_signal, signal_value)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def create(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','register','review','screen-signal'])
    parser.add_argument('--variant',choices=['strong_prompt','full'],required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();payload=read(args.input)
    if args.command=='prepare':
        result=context(payload['facts'],args.variant)
        if args.variant=='full':result['counterexamples']=counterexamples(payload.get('history',[]))
        # Baseline receives the exact same raw history, including failed predictions.
        result['history']=payload.get('history',[])
        result['shared_information_sha256']=fingerprint(dict(facts=payload['facts'],history=result['history']))
    elif args.command=='register':
        if args.variant!='full':raise ValueError('baseline has no mandatory prediction gate')
        register_prediction(args.output,payload['facts'],payload['decision']);return
    elif args.command=='review':
        result=dict(raw_response=payload['response'],facts=payload['facts'],variant=args.variant)
        if args.variant=='full':
            registration=payload.get('registration')
            if registration:
                check={k:v for k,v in registration.items() if k!='sha256'}
                if fingerprint(check)!=registration['sha256']:raise ValueError('registration hash mismatch')
                if registration['sha256']!=payload['evaluation_registration_sha256']:
                    raise ValueError('evaluation not bound to registration')
                if registration['decision']['parent_id']!=payload['facts']['parent_id']:
                    raise ValueError('parent identity differs')
                if fingerprint(registration['decision']['slice'])!=payload['slice_sha256']:
                    raise ValueError('slice differs from registration')
                if fingerprint(registration['decision']['action'])!=payload['action_sha256']:
                    raise ValueError('action differs from registration')
                if registration['facts_sha256']!=payload['prestate_sha256']:
                    raise ValueError('pre-action state differs from registration')
            result['dual_update']=dual_update(payload['response'],preregistered=bool(registration),
                directions=registration['decision']['directions'] if registration else None,
                additional_falsifiers_passed=payload.get('additional_falsifiers_passed'),
                matched_control_passed=payload.get('matched_control_passed',False),
                independent_confirmation=payload.get('independent_confirmation',False))
            result['counterexamples']=counterexamples(payload.get('history',[]))
    else:
        if args.variant!='full':raise ValueError('signal learning disabled for baseline')
        # Predict labels without reading response labels; score only after predictions are assembled.
        rows=payload['selection_records'];predictions=[]
        if any(not r.get('prestate_locked') for r in rows):raise ValueError('missing prospective prestate')
        for row in rows:
            condition=measure_signal(row['prestate'],payload['rule'],payload['allowed_features'])
            predictions.append(0 if condition is None else payload['if_true'] if condition else payload['if_false'])
        result=signal_value([r['baseline_prediction'] for r in rows],predictions,
            [r['observed_direction'] for r in rows],construction_sources=payload['construction_sources'],
            selection_sources=sorted({r['source_family'] for r in rows}))
        result.update(predictions=predictions,rule_sha256=fingerprint(payload['rule']),
            promotion='pending independent confirmation; no automatic scientific claim')
    create(args.output,result)


if __name__=='__main__':main()
