"""Frozen train-only selection of real historical input channels."""
import json
import torch
from run_research import ROOT, digest


def load_mapping():
    path=ROOT/'research_runs/results/A009_leading_sources/diagnostic.json'
    result=json.loads(path.read_text())
    assert result['gate_passed']
    mapping=result['folds'][0]['mapping']
    assert mapping==result['folds'][1]['mapping']
    return mapping,digest(path)


class SourceDLinear(torch.nn.Module):
    def __init__(self,core,mapping):
        super().__init__(); self.core=core
        self.register_buffer('source_index',torch.tensor(mapping,dtype=torch.long))
        self.register_buffer('unchanged',torch.tensor([i==j for i,j in enumerate(mapping)]))

    def forward(self,x):
        history=x[:,:,self.source_index]
        transformed=history-history[:,-1:]+x[:,-1:]
        transformed=torch.where(self.unchanged[None,None,:],x,transformed)
        return self.core(transformed)
