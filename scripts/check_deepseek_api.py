"""One real JSON-response acceptance call; never records credentials or reasoning."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'research_runs/agent_research/acceptance/20261008/deepseek_api.json'


def main():
    if DESTINATION.exists():
        raise RuntimeError('Existing acceptance receipt must not be overwritten')
    key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not key:
        raise RuntimeError('DEEPSEEK_API_KEY is missing')
    body = {'model': 'deepseek-flash', 'reasoning_effort': 'low', 'max_tokens': 512,
            'response_format': {'type': 'json_object'}, 'messages': [
                {'role': 'system', 'content': 'Return JSON only. This is an engineering interface check, not a research experiment. Return exactly {"action":{"tool":"report_readiness","arguments":{"ready":true}},"reason":"接口检查","hypotheses":[],"evidence_refs":[]}.'},
                {'role': 'user', 'content': 'Check that the JSON decision interface is working. No data, training, or research findings are provided.'}]}
    encoded = json.dumps(body, ensure_ascii=False).encode('utf-8')
    receipt = {'scope': 'engineering_only', 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'model_requested': body['model'], 'request_sha256': hashlib.sha256(encoded).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'network_requests': 1, 'research_experiment': False}
    started = time.monotonic()
    request = urllib.request.Request('https://api.deepseek.com/chat/completions', data=encoded,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            envelope = json.load(response)
        receipt.update(model_returned=envelope.get('model'), usage=envelope.get('usage'))
        choice = envelope['choices'][0]
        value = json.loads(choice['message']['content'])
        expected = {'action': {'tool': 'report_readiness', 'arguments': {'ready': True}},
                    'reason': '接口检查', 'hypotheses': [], 'evidence_refs': []}
        if choice.get('finish_reason') != 'stop' or value != expected:
            raise ValueError('JSON acceptance response did not match the requested contract')
        if not isinstance(receipt['usage'], dict) or type(receipt['usage'].get('total_tokens')) is not int:
            raise ValueError('API usage accounting is missing')
        receipt.update(status='passed', decision=value)
    except urllib.error.HTTPError as error:
        receipt.update(status='failed', error_type='HTTPError', http_status=error.code)
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__)
    receipt['elapsed_seconds'] = time.monotonic() - started
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open('x', encoding='utf-8') as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
