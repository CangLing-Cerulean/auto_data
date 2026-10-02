"""Check window boundaries, DLinear, metrics and checkpoint replay on actual data."""
import copy
import csv
import io
import json

import numpy as np
import torch

from run_research import ROOT, DLinear, SELECTIVE_CENTER_COLUMNS, evaluate, predict, valid_starts


def main():
    p = json.loads((ROOT / 'research_runs/protocol.json').read_text())
    path = ROOT / 'data/metropt3_v1/train.csv'
    x = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(2,17), max_rows=10000)
    with path.open(newline='') as f:
        reader = csv.reader(f)
        next(reader)
        times = np.array([next(reader)[1] for _ in range(10000)], dtype='datetime64[s]')
    starts = valid_starts(times, 40, 30)
    brute = np.array([i for i in range(len(times)-39) if np.all(np.diff(times[i:i+40]) <= np.timedelta64(30,'s'))])
    assert np.array_equal(starts, brute)
    assert starts[0] == 0 and starts[-1] + 39 < len(times)
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    model = DLinear(p)
    batch = torch.from_numpy(x[starts[:8,None] + np.arange(40)]).float()
    inp, target = batch[:,:32], batch[:,32:]
    output = model(inp)
    assert output.shape == target.shape
    padded = torch.cat([inp[:,0:1,:].repeat(1,12,1), inp, inp[:,-1:,:].repeat(1,12,1)], dim=1)
    trend = torch.nn.AvgPool1d(25, stride=1)(padded.permute(0,2,1)).permute(0,2,1)
    reference = torch.stack([model.seasonal[i]((inp-trend)[:,:,i]) + model.trend[i](trend[:,:,i]) for i in range(15)], dim=-1)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    anchor = inp[:, -1:, :]
    centered = predict(model, inp, 'last_center')
    torch.testing.assert_close(centered, model(inp-anchor)+anchor, rtol=0, atol=0)
    torch.testing.assert_close((centered-target).square().mean(), (model(inp-anchor)-(target-anchor)).square().mean())
    anchor_mean = inp.mean(dim=1, keepdim=True)
    torch.testing.assert_close(predict(model, inp, 'mean_center'), model(inp-anchor_mean)+anchor_mean, rtol=0, atol=0)
    selective = predict(model, inp, 'selective_center')
    for i in range(15):
        expected = centered[:,:,i] if i in SELECTIVE_CENTER_COLUMNS.values() else output[:,:,i]
        torch.testing.assert_close(selective[:,:,i], expected, rtol=0, atol=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    def update(m, opt):
        opt.zero_grad(set_to_none=True)
        loss = (m(inp)-target).square().mean()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
    update(model, optimizer)
    saved = io.BytesIO()
    torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, saved)
    saved.seek(0)
    state = torch.load(saved, weights_only=False)
    resumed = DLinear(p)
    resumed.load_state_dict(state['model'])
    opt2 = torch.optim.Adam(resumed.parameters(), lr=.001)
    opt2.load_state_dict(state['optimizer'])
    update(model, optimizer)
    update(resumed, opt2)
    for a,b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    subset = torch.from_numpy(starts[:8])
    metrics = evaluate(model, torch.from_numpy(x).float(), subset, {'all':np.ones(8,dtype=bool)}, p, 'baseline', True)
    err = (inp[:,-1:,:].expand_as(target)-target).double()
    assert abs(metrics['groups']['all']['NMSE'] - err.square().mean().item()) < 1e-9
    assert abs(metrics['groups']['all']['NMAE'] - err.abs().mean().item()) < 1e-9
    print('PASS: real-data window boundary enumeration, reference DLinear equivalence, Adam checkpoint replay, direct metric calculation.')


if __name__ == '__main__':
    main()
