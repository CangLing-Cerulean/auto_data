"""Read completed acceptance ledgers without model requests or score changes."""
from collections import Counter
import json
from pathlib import Path
import sqlite3

root = Path('/tmp/auto_data_capability_acceptance_20261010')
outcomes = []
for path in sorted((root / 'bank_v2/runs').glob('*/runtime.sqlite')):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        rows = [(kind, json.loads(payload)) for kind, payload in db.execute(
            'SELECT kind,payload FROM agent_runtime_records ORDER BY created_ns')]
        calls = db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall()
    formats = [value for kind, value in rows if kind == 'format_failure']
    errors = Counter(json.dumps(value.get('validation_errors'), ensure_ascii=False, sort_keys=True)
                     for value in formats)
    decisions = [value for kind, value in rows if kind == 'decision']
    outcomes.append({'run_id': path.parent.name, 'records': dict(Counter(kind for kind, _ in rows)),
        'API_attempts': len(calls), 'actual_known_tokens': sum(row[1] or 0 for row in calls),
        'format_failure_count': len(formats), 'format_validation_counts': dict(errors),
        'tool_counts': dict(Counter(value.get('selected_action') for value in decisions)),
        'decision_outcomes': dict(Counter(value.get('decision_outcome') for value in decisions))})
value = {'source': 'completed_bank_v2_read_only_SQLite_ledgers', 'runs': outcomes,
    'API_attempts': sum(row['API_attempts'] for row in outcomes),
    'actual_known_tokens': sum(row['actual_known_tokens'] for row in outcomes),
    'format_failure_count': sum(row['format_failure_count'] for row in outcomes),
    'new_training_calls': 0}
target = root / 'cost_audit.json'
if target.exists():
    if json.loads(target.read_text()) != value:
        raise RuntimeError('Existing audit differs; preserve it')
else:
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(value, ensure_ascii=False))
