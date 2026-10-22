"""Small intervention-diagnosis learner; consumes only explicitly supplied folds."""
import numpy as np
import torch

METHODS = ('OVERALL', 'JOINT', 'SEPARATE')
UPDATES = 1000
LEARNING_RATE = 0.03
REGULARIZATION = 0.01


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def response(alpha, polynomial, parent_c, known_prefix=None, known_mask=None):
    # polynomial: candidate,step,split,channel,(constant,linear,quadratic).
    curves = (polynomial[..., 0] + polynomial[..., 1] * alpha[:, None, None, :] +
              polynomial[..., 2] * alpha[:, None, None, :].square()).mean(-1)
    if known_prefix is not None:
        first = torch.where(known_mask[:, None], known_prefix, curves[:, 0])
        curves = torch.cat((first[:, None], curves[:, 1:]), dim=1)
    steps = curves[:, :, 0].argmin(1)
    selected_c = curves[torch.arange(len(curves)), steps, 1]
    return 100. * (parent_c - selected_c) / parent_c, steps, curves


def fit(fold, method):
    assert method in METHODS
    history, probes = fold['history'], fold['probes']
    observed = history + probes
    features = tensor([r['x'] for r in history])
    center = features.mean(0)
    scale = features.std(0, unbiased=False)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x = (tensor([r['x'] for r in observed]) - center) / scale
    design = torch.cat((torch.ones((*x.shape[:2], 1), dtype=torch.float64), x), -1)
    poly = tensor([r['polynomial'] for r in observed])
    p = tensor([r['prefix_p'] for r in observed])
    p_scale = tensor([r['parent_prefix_p'] for r in observed])
    parent_c = tensor([r['parent_c'] for r in history])
    y = tensor([r['y'] for r in history])
    y_scale = max(float(y.std(unbiased=False)), 1.)
    if method != 'OVERALL':
        z = tensor([r['z'] for r in observed])
        energy = tensor([r['z_energy'] for r in observed])
        average = energy.mean(1, keepdim=True)
        z_weights = energy / torch.where(average == 0, torch.ones_like(average), average)
    w = torch.nn.Parameter(torch.zeros((8, 4), dtype=torch.float64))
    beta = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam((w, beta), lr=LEARNING_RATE)
    trace = []
    for iteration in range(UPDATES):
        optimizer.zero_grad(set_to_none=True)
        alpha = torch.sigmoid((design * w).sum(-1))
        _, _, unobserved_curves = response(alpha, poly, tensor([r['parent_c'] for r in observed]))
        prefix_loss = ((unobserved_curves[:, 0] - p) / p_scale).square().mean()
        alpha_for_value = alpha[:len(history)].detach() if method == 'SEPARATE' else alpha[:len(history)]
        raw_gain, _, _ = response(alpha_for_value, poly[:len(history)], parent_c,
                                  p[:len(history)], torch.ones(len(history), dtype=torch.bool))
        predicted = raw_gain + beta[0] * y_scale + beta[1] * raw_gain
        value_loss = ((predicted - y) / y_scale).square().mean()
        diagnostic_loss = ((alpha - z).square() * z_weights).mean() if method != 'OVERALL' else alpha.sum() * 0.
        penalty = REGULARIZATION * torch.cat((w.flatten(), beta)).square().mean()
        objective = prefix_loss + diagnostic_loss + value_loss + penalty
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite auxiliary objective')
        objective.backward()
        if not all(torch.isfinite(v.grad).all() for v in (w, beta)):
            raise ValueError('Nonfinite auxiliary gradient')
        optimizer.step()
        if iteration in (0, UPDATES - 1):
            trace.append(dict(iteration=iteration, prefix=float(prefix_loss.detach()),
                diagnostic=float(diagnostic_loss.detach()), value=float(value_loss.detach()),
                total=float(objective.detach())))
    return dict(method=method, weights=w.detach().tolist(), beta=beta.detach().tolist(),
                center=center.tolist(), scale=scale.tolist(), y_scale=y_scale, updates=UPDATES, trace=trace)


@torch.no_grad()
def predict(state, rows):
    x = (tensor([r['x'] for r in rows]) - tensor(state['center'])) / tensor(state['scale'])
    design = torch.cat((torch.ones((*x.shape[:2], 1), dtype=torch.float64), x), -1)
    alpha = torch.sigmoid((design * tensor(state['weights'])).sum(-1))
    known = torch.tensor(['prefix_p' in r for r in rows], dtype=torch.bool)
    prefix = tensor([r.get('prefix_p', [0., 0.]) for r in rows])
    gain, steps, curves = response(alpha, tensor([r['polynomial'] for r in rows]),
        tensor([r['parent_c'] for r in rows]), prefix, known)
    beta = state['beta']
    predicted = gain + beta[0] * state['y_scale'] + beta[1] * gain
    return [dict(id=r['id'], gain_percent=float(predicted[i]), uncalibrated_gain=float(gain[i]),
        selected_step=(int(steps[i]) + 1) * 500, alpha=alpha[i].tolist(), curves=curves[i].tolist())
        for i, r in enumerate(rows)]


def ridge_features(rows):
    poly = tensor([r['polynomial'] for r in rows])
    pc = tensor([r['parent_c'] for r in rows])
    known = torch.tensor(['prefix_p' in r for r in rows], dtype=torch.bool)
    prefix = tensor([r.get('prefix_p', [0., 0.]) for r in rows])
    gains = [response(torch.full((len(rows), 8), a, dtype=torch.float64), poly, pc, prefix, known)[0].numpy()
             for a in (0., .5, 1.)]
    return np.column_stack((np.asarray([r['x'] for r in rows]).reshape(len(rows), -1), *gains))


def ridge_fit(fold):
    rows = fold['history']
    x = ridge_features(rows)
    center, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1.
    design = np.column_stack((np.ones(len(x)), (x - center) / scale))
    y = np.asarray([r['y'] for r in rows])
    y_scale = max(float(y.std()), 1.)
    # M1 baseline is cheap and independent of observed candidate prefixes.
    baseline = np.asarray([r['m1_gain'] for r in rows])
    target = (y - baseline) / y_scale
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.
    weights = np.linalg.solve(design.T @ design / len(x) + penalty, design.T @ target / len(x))
    return dict(method='RIDGE_Y', weights=weights.tolist(), center=center.tolist(), scale=scale.tolist(), y_scale=y_scale)


def ridge_predict(state, rows):
    x = ridge_features(rows)
    design = np.column_stack((np.ones(len(x)), (x - state['center']) / state['scale']))
    gain = np.asarray([r['m1_gain'] for r in rows]) + state['y_scale'] * (design @ state['weights'])
    return [dict(id=r['id'], gain_percent=float(g)) for r, g in zip(rows, gain)]
