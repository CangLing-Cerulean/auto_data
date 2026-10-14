"""Read original DLinear/Adam coordinates without advancing an optimizer."""
import math
import numpy as np
import torch
import source_quadratic as source


def parent_coordinates(model, snapshot):
    """Validate the saved optimizer mapping and extract theta, m, v and P.

    The model must be the original DLinear class, with the original parameter
    registration order. Loading its optimizer is read-only with respect to the
    saved snapshot and does not perform an optimizer step.
    """
    model.load_state_dict(snapshot['model'],strict=True)
    named=list(model.named_parameters());groups=snapshot['optimizer']['param_groups']
    channels=len(model.seasonal);width=model.seasonal[0].in_features;horizon=model.seasonal[0].out_features
    expected=[f'{kind}.{c}.{part}' for kind in ('seasonal','trend') for c in range(channels) for part in ('weight','bias')]
    if [n for n,_ in named]!=expected:raise ValueError('Unexpected original parameter registration order')
    if list(snapshot['model'])!=expected:raise ValueError('Snapshot model mapping differs')
    if len(groups)!=1 or len(groups[0]['params'])!=len(named) or len(set(groups[0]['params']))!=len(named):
        raise ValueError('Unexpected optimizer parameter groups')
    group=groups[0]
    if group['params']!=list(range(len(named))):raise ValueError('Saved optimizer registration IDs differ')
    if group['lr']!=.001 or tuple(group['betas'])!=(.9,.999) or group['eps']!=1e-8 or group['weight_decay']!=0.:
        raise ValueError('Optimizer differs from frozen training protocol')
    if group.get('amsgrad',False) or group.get('maximize',False):raise ValueError('Unsupported Adam variant')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    optimizer.load_state_dict(snapshot['optimizer'])
    step=int(snapshot['step'])
    if step<=0 or step!=snapshot['step']:raise ValueError('Parent checkpoint must have positive integral step')
    packed={key:{} for key in ('theta','exp_avg','exp_avg_sq','P')}
    for name,param in named:
        state=optimizer.state[param]
        state_step=float(state['step'].item())
        if not math.isfinite(state_step) or state_step!=step:raise ValueError('Adam state steps differ')
        arrays={'theta':param.detach().cpu().numpy(),**{k:state[k].detach().cpu().numpy() for k in ('exp_avg','exp_avg_sq')}}
        if any(a.shape!=tuple(param.shape) or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Invalid saved optimizer coordinates')
        for key,value in arrays.items():packed[key][name]=value.astype(np.float64)
        packed['P'][name]=source.adam_diagonal(arrays['exp_avg_sq'],step,beta2=group['betas'][1],epsilon=group['eps'])
    def coordinates(values):
        return np.stack([np.concatenate((values[f'seasonal.{c}.weight'].T,values[f'seasonal.{c}.bias'][None],
            values[f'trend.{c}.weight'].T,values[f'trend.{c}.bias'][None])) for c in range(channels)])
    result={key:coordinates(value) for key,value in packed.items()}
    expected_shape=(channels,2*width+2,horizon)
    if any(a.shape!=expected_shape for a in result.values()):raise ValueError('Original coordinate shapes differ')
    saved=snapshot['original_coordinates'].detach().cpu().numpy().astype(np.float64)
    if not np.array_equal(result['theta'],saved):raise ValueError('Snapshot original coordinates mismatch')
    return dict(step=step,**result)


def quadratic_segment(gram_parent,cross_parent,gram_action,cross_action,parent_state,previous_delta,*,steps=500):
    """Right-endpoint segment statistics, fixed P and exact quadratic Hessian.

    No labels or selection losses are arguments. The action's normalized
    probabilities must already be represented in its moments. This is a
    first-order response even for a finite action, not an exact retraining map.
    """
    theta=parent_state['theta'];channels,width,horizon=theta.shape
    parent_gradient=source.quadratic_gradient(gram_parent,cross_parent,theta)
    action_gradient=source.quadratic_gradient(gram_action,cross_action,theta)
    forcing=action_gradient-parent_gradient
    previous=np.asarray(previous_delta,dtype=np.float64)
    if previous.shape!=theta.shape or not np.isfinite(previous).all():raise ValueError('Invalid incoming response')
    gram=np.asarray(gram_parent,dtype=np.float64)
    p=np.asarray(parent_state['P'],dtype=np.float64)
    if gram.shape!=(channels,width,width) or p.shape!=theta.shape:raise ValueError('Invalid curvature/preconditioner shapes')
    delta=np.empty_like(theta);diagnostics=[]
    for channel in range(channels):
        hessian=2./(channels*horizon)*gram[channel]
        for output in range(horizon):
            segment=source.segment_response(hessian,p[channel,:,output],forcing[channel,:,output],previous[channel,:,output],
                learning_rate=.001,steps=steps)
            delta[channel,:,output]=segment.delta
            diagnostics.append(dict(channel=channel,output=output,min_eigenvalue=float(segment.eigenvalues.min()),
                max_eigenvalue=float(segment.eigenvalues.max()),clipped_negative_eigenvalues=segment.clipped_negative_eigenvalues,
                roundoff_tolerance=segment.roundoff_tolerance))
    return delta,diagnostics
