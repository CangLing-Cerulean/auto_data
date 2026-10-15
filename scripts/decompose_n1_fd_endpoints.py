"""Read-only endpoint decomposition of every previously audited N1 FD row.

Run through stdin on the original CUDA host. No server output files or updates.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/root/auto_data_experiments/20261015_n1_path_engineering_r1/experiment')
RECEIPT_SHA = '41ad23ab48ae503484c58bd31798f6b714275b5ff612bea4da8ff559f7d74c93'
SEED = 2026101701
GRID = (500, 1000, 1500, 2000, 2500, 3000)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.perf_counter()
    assert __debug__ and os.name == 'posix' and ROOT.resolve() == ROOT
    receipt_path = ROOT/'delivery/n1_fd_20261015/receipt.json'
    assert sha(receipt_path) == RECEIPT_SHA
    archive = json.loads(receipt_path.read_text())
    assert archive['status'] == 'PASS' and archive['counts']['files'] == 975
    members = {r['path']: r for r in archive['members']}
    consumed = {}

    def verified(name):
        path = ROOT/name
        assert path.resolve() == path and path.is_file() and path.is_relative_to(ROOT)
        assert path.stat().st_size == members[name]['bytes'] and sha(path) == members[name]['sha256']
        consumed[name] = members[name]['sha256']
        return path

    def read(name):
        return json.loads(verified(name).read_text())

    for name in ('fourgroup_runtime.py','audit_fourgroup_m1.py','response_proxy.py','run_research.py'):
        verified(name)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import fourgroup_runtime as rt
    import audit_fourgroup_m1 as prior
    rt.configure()
    for phase in ('preflight','paths','audit_paths','fd','audit_fd'):
        assert read(f'control/{phase}/exit.json')['exit_code'] == 0
    audit = read('fd_audit.json')
    assert audit['status']=='PASS' and not audit['numerical_gate_pass']
    assert audit['fd_sha256'] == sha(verified('fd.json'))
    assert audit['fd_plan_sha256'] == sha(verified('fd_plan.json'))
    plan = read('plan.json')
    original = {(r['step'],r['split'],r['direction'],r['h_denominator']):r for r in audit['derivative_comparisons']}
    original_cross = {(r['step'],r['split'],r['direction']):r for r in audit['two_h_comparisons']}
    assert len(original)==96 and len(original_cross)==48
    with np.load(verified('inputs/prepared/G1_ABC.npz'), allow_pickle=False) as abc:
        batches = {}
        for split in ('B','C'):
            values = torch.tensor(abc[f'{split}_values'], device='cuda')
            starts = torch.tensor(abc[f'{split}_starts'], device='cuda')
            assert len(starts)==1617
            batches[split] = values[starts[:,None]+torch.arange(40,device='cuda')]
    phi = {s:prior.features(batch[:,:32]).double().cpu().numpy() for s,batch in batches.items()}
    model = prior.new_model(plan, SEED)
    basis = np.array([[1.,-1.,0.,0.],[1.,1.,-2.,0.],[1.,1.,1.,-3.],[1.,1.,1.,1.]]).T
    basis /= np.sqrt(np.array([2.,6.,12.,4.]))
    norm = lambda value: float(np.linalg.norm(value))
    rows, cross_rows = [], []
    residual_checks = 0
    max_identity = 0.

    def close(left,right):
        np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-12)

    def endpoint(directory,step,metrics):
        nonlocal residual_checks
        saved = torch.load(verified(f'{directory}/step_{step}.pt'),map_location='cpu',weights_only=True)
        assert saved['step']==step
        theta = prior.coordinates(saved['model']).double().cpu().numpy()
        close(theta,saved['original_coordinates'].double().cpu().numpy())
        model.load_state_dict(saved['model']);model.eval()
        predictions, errors = {}, {}
        with torch.no_grad():
            for split,batch in batches.items():
                prediction = model(batch[:,:32])
                error = (prediction-batch[:,32:]).cpu().numpy()
                assert prior.metric(error)==metrics[split]
                residual_checks += 1
                predictions[split] = prediction.double().cpu().numpy()
                errors[split] = error.astype(np.float64)
        return theta,predictions,errors

    def verify_original(left,right,parent,row):
        denominator = max(norm(left),norm(right),1e-6*norm(parent))
        values = {'left_norm':norm(left),'right_norm':norm(right),'parent_norm':norm(parent),
                  'absolute_error':norm(left-right),'denominator':denominator,
                  'normalized_error':norm(left-right)/denominator}
        for key,value in values.items():close(value,row[key])
        assert row['passed']==(values['normalized_error']<=.02)
        return denominator

    def record_parts(total,evaluation,parameter,denominator,original_row):
        nonlocal max_identity
        identity=norm(total-evaluation-parameter)
        max_identity=max(max_identity,identity)
        close(total,evaluation+parameter)
        te,ee,pe=norm(total)**2,norm(evaluation)**2,norm(parameter)**2
        cross=float(2*np.sum(evaluation*parameter))
        close(te,ee+pe+cross)
        bound=max(0.,norm(total)-norm(evaluation))/denominator
        return dict(original_passed=original_row['passed'],original_normalized_error=original_row['normalized_error'],
            original_denominator=denominator,total_error_norm=norm(total),evaluation_error_norm=norm(evaluation),
            parameter_response_error_norm=norm(parameter),evaluation_to_total_norm_ratio=norm(evaluation)/norm(total),
            parameter_to_total_norm_ratio=norm(parameter)/norm(total),squared_cross_term=cross,
            decomposition_residual_norm=identity,parameter_normalized_lower_bound=bound,
            evaluation_alone_cannot_explain_original_failure=(not original_row['passed'] and bound>.02),
            channel_total_squared_error=np.sum(total**2,axis=(0,1)).tolist(),
            channel_evaluation_squared_error=np.sum(evaluation**2,axis=(0,1)).tolist(),
            channel_parameter_squared_error=np.sum(parameter**2,axis=(0,1)).tolist())

    for step in GRID:
        parent_dir=f'paths/N1_G1_s{SEED}'
        parent_metrics={s:read(f'{parent_dir}/step_{step}_{s}.json')['residual'] for s in ('B','C')}
        _,_,parent_errors=endpoint(parent_dir,step,parent_metrics)
        with np.load(verified(f'{parent_dir}/step_{step}_tangent.npz'),allow_pickle=False) as data:
            tangent=data['S'].reshape(8,66,8,4)
        jacobians={s:np.einsum('bcd,cdhj->bhcj',x,tangent) for s,x in phi.items()}
        for direction in range(4):
            per_h={}
            for den in (64,128):
                endpoints,weights={},{}
                for side in ('MINUS','PLUS'):
                    directory=f'fd_runs/N1FD_G1_d{direction}_h{den}_{side}_s{SEED}'
                    with np.load(verified(f'{directory}/weights.npz'),allow_pickle=False) as data:
                        weights[side]=data['group_weights'].astype(np.float64)
                    endpoints[side]=endpoint(directory,step,read(f'{directory}/step_{step}_metrics.json'))
                effective=basis.T@(weights['PLUS']-weights['MINUS'])/(2./den)
                theta=(endpoints['PLUS'][0]-endpoints['MINUS'][0])/(2./den)
                predicted_theta=tangent@effective
                for split in ('B','C'):
                    old=original[(step,split,direction,den)]
                    a=(endpoints['PLUS'][2][split]-endpoints['MINUS'][2][split])/(2./den)
                    p=(endpoints['PLUS'][1][split]-endpoints['MINUS'][1][split])/(2./den)
                    b=np.einsum('bcd,cdh->bhc',phi[split],theta)
                    c=jacobians[split]@effective
                    denominator=verify_original(a,c,parent_errors[split],old)
                    details=record_parts(a-c,a-b,b-c,denominator,old)
                    close(a-b,(a-p)+(p-b))
                    rows.append(dict(step=step,split=split,direction=direction,h_denominator=den,
                        residual_subtraction_error_norm=norm(a-p),forward_evaluation_error_norm=norm(p-b),
                        parameter_FD_norm=norm(theta),parameter_tangent_norm=norm(predicted_theta),
                        parameter_FD_tangent_error_norm=norm(theta-predicted_theta),**details))
                    per_h[(den,split)]=(a,b)
            for split in ('B','C'):
                a64,b64=per_h[(64,split)];a128,b128=per_h[(128,split)]
                old=original_cross[(step,split,direction)]
                denominator=verify_original(a64,a128,parent_errors[split],old)
                details=record_parts(a64-a128,(a64-b64)-(a128-b128),b64-b128,denominator,old)
                cross_rows.append(dict(step=step,split=split,direction=direction,**details))
    assert len(rows)==96 and len(cross_rows)==48 and residual_checks==204
    # Recheck every consumed asset after all inference; no in-place source edits.
    for name,digest in consumed.items():assert sha(ROOT/name)==digest
    result=dict(status='PASS',analysis='posthoc_read_only_endpoint_decomposition',
        source_archive_receipt_sha256=RECEIPT_SHA,inputs_sha256=consumed,
        derivative_rows=rows,two_h_rows=cross_rows,residual_sets_verified=residual_checks,
        maximum_decomposition_residual_norm=max_identity,seconds=time.perf_counter()-started,
        new_full_trainings=0,cumulative_full_trainings=434,original_numerical_gate_pass=False,
        old_scores_changed=False,N1_proposals_enabled=False,N2_scientific_enabled=False,old_M1_labels_enabled=False,
        A_values_read=False,D_access=False,test_access=False,server_output_files_written=False,
        environment=rt.environment())
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    main()
