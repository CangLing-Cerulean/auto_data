"""Verify identical sampling state and untouched channel parameters at final step."""
import json
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
PAIRS=[('E000_baseline_s2026','E003_selective_center_s2026'),
       ('E004_baseline_s2027','E005_selective_center_s2027'),
       ('E006_baseline_s2028','E007_selective_center_s2028')]


def main():
    outcomes=[]
    for left,right in PAIRS:
        a=torch.load(ROOT/'research_runs/results'/left/'latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(ROOT/'research_runs/results'/right/'latest.pt',map_location='cpu',weights_only=False)
        assert a['step']==b['step']==3000
        assert torch.equal(a['sampling_rng'],b['sampling_rng'])
        untouched=[]
        changed=[]
        for key in a['model']:
            channel=int(key.split('.')[1])
            equal=torch.equal(a['model'][key],b['model'][key])
            if channel not in (5,13):
                assert equal, (left,right,key)
                untouched.append(key)
            elif not equal:
                changed.append(key)
        assert changed
        outcomes.append({'baseline':left,'candidate':right,'same_sampling_rng':True,
                         'identical_unmodified_parameter_tensors':len(untouched),
                         'changed_selected_parameter_tensors':len(changed),'step':3000})
    (ROOT/'research_runs/pair_checkpoint_audit.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    print('PASS: all pairs have identical sampling RNG and unchanged 13-channel final weights.')


if __name__=='__main__':
    main()
