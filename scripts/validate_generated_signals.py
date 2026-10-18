"""Supplementary validation of completed outputs; no LLM call or regeneration."""
import json
from generate_conditional_signals import OUT,evaluate
from diagnosis_pool_features import read,sha,write

lock=read(OUT/'generation_lock.json')
for entry in lock['calls']:
    directory=OUT/entry['name'];initial=read(directory/'receipt.json')
    assert initial['exit_code']==0
    assert sha(directory/'prompt.txt')==entry['prompt_sha256']
    assert sha(directory/'input.json')==entry['input_sha256']
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    items=[e['item'] for e in events if 'item' in e]
    assert all(item['type'] in ('error','reasoning','agent_message') for item in items)
    errors=[item['message'] for item in items if item['type']=='error']
    assert all('WebSockets to HTTPS transport' in message for message in errors)
    assert any(e['type']=='turn.completed' for e in events)
    response=read(directory/'response.json');signals=response['signals']
    assert 0<=len(signals)<=2 and len({s['name'] for s in signals})==len(signals)
    assert set(response)=={'signals','reasoning'}
    for signal in signals:
        assert set(signal)=={'name','expression','measurement_meaning','limitation','falsifying_response'}
        assert all(isinstance(value,str) and value for value in signal.values())
        for record in read(directory/'input.json')['records']:evaluate(signal['expression'],record['features'])
    receipt=dict(status='PASS',original_receipt_status=initial['status'],original_receipt_sha256=sha(directory/'receipt.json'),
        correction='transport error item was incorrectly classified as a tool call; original failure retained',
        no_tool_items=True,transport_error_messages=errors,signals=len(signals),response_sha256=sha(directory/'response.json'),
        events_sha256=sha(directory/'events.jsonl'),new_LLM_calls=0,new_full_trainings=0)
    write(directory/'validation_receipt.json',receipt)
    print(entry['name'],receipt['status'],len(signals))
