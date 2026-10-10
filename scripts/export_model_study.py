"""Export frozen candidates and references for local NumPy evaluation."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from forecast_models import CrossChannelForecast
from run_research import ROOT, DLinear, digest, write_json, prepare
from reducible_sampling import used_seconds
from run_model_exploration import FOLDER, ROUND_LIMIT, BASELINES


def main():
    progress = json.loads((FOLDER / 'progress.json').read_text())
    assert progress['phase'] == 'completed_confirmation' and progress['test_eligible']
    folder = FOLDER / 'exports'; assert not folder.exists(); folder.mkdir()
    audit_run = ROOT / 'research_runs/results/A015_model_export'
    assert not audit_run.exists(); audit_run.mkdir()
    begun = time.monotonic(); allowance = ROUND_LIMIT - used_seconds()
    def status(phase):
        write_json(audit_run / 'status.json', {'phase': phase, 'elapsed_seconds': time.monotonic() - begun})
    status('exporting')
    try:
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        p = json.loads((ROOT / 'research_runs/selection/protocol.json').read_text())
        manifest = json.loads((ROOT / 'data/metropt3_v1/manifest.json').read_text())
        arrays, starts, _, audit = prepare(p, manifest)
        probes = starts['validation'][np.linspace(0, len(starts['validation']) - 1, 512, dtype=np.int64)]
        x = arrays['validation'][probes[:, None] + np.arange(32)]
        meta = {'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'study_protocol_sha256': digest(FOLDER / 'PROTOCOL.md'), 'export_script_sha256': digest(Path(__file__)),
            'mean': audit['mean'], 'scale': audit['scale'], 'models': []}
        for name in progress['selected_runs'] + list(BASELINES.values()):
            assert time.monotonic() - begun < allowance
            run = ROOT / 'research_runs/results' / name
            result = json.loads((run / 'result.json').read_text())
            kind = progress['winner'] if name in progress['selected_runs'] else 'dlinear'
            model = DLinear(p) if kind == 'dlinear' else CrossChannelForecast(kind)
            model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True))
            model.eval()
            if kind == 'dlinear':
                # Exact affine function in float64, then float32 deployment coefficients.
                model.double()
                with torch.no_grad():
                    bias = model(torch.zeros(1, 32, 15, dtype=torch.float64)).reshape(120)
                    weight = (model(torch.eye(480, dtype=torch.float64).reshape(480, 32, 15)).reshape(480, 120) - bias).T
                values = {'w': weight.float().numpy(), 'b': bias.float().numpy()}
                model.float()
            elif kind == 'cross_linear':
                values = {'w': model.net.weight.detach().numpy(), 'b': model.net.bias.detach().numpy()}
            else:
                values = {'w1': model.net[0].weight.detach().numpy(), 'b1': model.net[0].bias.detach().numpy(),
                          'w2': model.net[2].weight.detach().numpy(), 'b2': model.net[2].bias.detach().numpy()}
            model.cuda()
            with torch.no_grad(): expected = model(torch.from_numpy(x).cuda()).cpu().numpy()
            artifact = folder / (name + '.npz')
            np.savez_compressed(artifact, **values, probe_x=x, probe_expected=expected)
            meta['models'].append({'run_id': name, 'kind': kind, 'seed': result['identity']['seed'],
                'best_step': result['best_step'], 'checkpoint_sha256': digest(run / 'best.pt'),
                'export_file': artifact.name, 'export_sha256': digest(artifact),
                'reference_validation': result['groups']['all'],
                'code_sources_sha256': result['identity'].get('sources_sha256', result['identity'])})
            del model
        write_json(folder / 'export_manifest.json', meta)
        (folder / 'export_script.py').write_bytes(Path(__file__).read_bytes())
        write_json(audit_run / 'result.json', {'operation': 'model_weight_export_and_validation_probes',
            'models': [m['run_id'] for m in meta['models']], 'test_accessed': False,
            'export_manifest_sha256': digest(folder / 'export_manifest.json')})
        status('completed')
        print(json.dumps({'exported': len(meta['models']), 'budget_used_seconds': used_seconds()}))
    except BaseException:
        status('failed'); raise


if __name__ == '__main__': main()
