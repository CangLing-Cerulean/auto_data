"""Shared, parent-only features and fixed long-horizon response slice."""
import hashlib
import json
from pathlib import Path
import numpy as np
from conditional_diagnosis import paired_response


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')


def region_response(parent, child):
    if parent.ndim != 3 or parent.shape[1] % 2:
        raise ValueError('Expected even forecast horizon and [window,horizon,channel] errors')
    mask=np.zeros(parent.shape,dtype=bool)
    mask[:,parent.shape[1]//2:,:]=True
    return paired_response(parent,child,mask)


def input_statistics(values, starts, first, input_length):
    """No rows after these explicit starts/input positions are inspected."""
    offsets=np.arange(input_length)
    accum={key:[] for key in ('level','variance','roughness')}
    # Work in chunks; each row here is a window-level summary, not a new observation.
    for lo in range(0,len(starts),256):
        x=values[starts[lo:lo+256,None]+offsets].astype(np.float64)
        accum['level'].append(np.mean(x,axis=1))
        accum['variance'].append(np.var(x,axis=1).mean(axis=1))
        accum['roughness'].append(np.mean(np.diff(x,axis=1)**2,axis=(1,2)))
    measured={key:np.concatenate(chunks) for key,chunks in accum.items()}
    if first is None:
        return {key:np.mean(value,axis=0).tolist() for key,value in measured.items()}
    if first.shape != (len(starts),) or not first.any() or first.all():
        raise ValueError('Invalid training exposure groups')
    return {name:{key:np.mean(value[mask],axis=0).tolist() for key,value in measured.items()}
            for name,mask in [('first',first),('second',~first)]}


def parent_features(a_statistics, b_statistics, b_error, curve, selected_step):
    b=b_error.astype(np.float64); horizon=b.shape[1]; mid=horizon//2
    loss=float(np.mean(b*b)); target=float(np.mean(b[:,mid:]**2)); complement=float(np.mean(b[:,:mid]**2))
    if loss<=0: raise ValueError('Parent B risk must be positive')
    curve=np.asarray(curve,dtype=np.float64)
    if curve.shape!=(6,) or not np.isfinite(curve).all():raise ValueError('Six actual B checkpoints required')
    result=dict(parent_B_log_loss=float(np.log(loss)),parent_B_target_ratio=target/loss,
                parent_B_local_contrast=(target-complement)/loss,
                parent_B_bias=float(np.mean(b)/np.sqrt(loss)),
                parent_B_target_bias=float(np.mean(b[:,mid:])/np.sqrt(loss)),
                parent_B_channel_dispersion=float(np.std(np.mean(b*b,axis=(0,1)))/loss),
                parent_B_learning_gain=float((curve[0]-curve[-1])/curve[0]),
                parent_B_best_fraction=float(selected_step/3000),
                parent_B_curve_reversal=float((curve[-1]-curve.min())/curve[0]),
                B_input_variance=float(b_statistics['variance']),B_input_roughness=float(b_statistics['roughness']))
    blevel=np.asarray(b_statistics['level'])
    for name in ('first','second'):
        group=a_statistics[name]
        result[name+'_input_variance']=float(group['variance'])
        result[name+'_input_roughness']=float(group['roughness'])
        result[name+'_B_level_distance']=float(np.mean((np.asarray(group['level'])-blevel)**2))
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite parent-only feature')
    return result


def action_features(action, implementation, first_mass):
    first=action.endswith('FIRST') or action.endswith('EARLY')
    keep=action.startswith('KEEP')
    favored_mass=first_mass if first else 1-first_mass
    if keep:
        resulting=1. if first else 0.
        retained=favored_mass
        second_moment=1/favored_mass
    else:
        denominator=3*favored_mass+1-favored_mass
        resulting=3*first_mass/denominator if first else first_mass/denominator
        retained=1.
        second_moment=(9*favored_mass+1-favored_mass)/denominator**2
    return dict(exposure_shift_first=resulting-first_mass,removed_fraction=1-retained,
                weighted_implementation=float(implementation=='W'),
                target_weight_second_moment=second_moment,first_training_mass=first_mass)


def combine_features(parent, action):
    out=dict(parent,**action)
    shift=out['exposure_shift_first']
    # Fixed diagnostic primitives, available to all comparison methods.
    out['aligned_level_shift']=shift*(out['second_B_level_distance']-out['first_B_level_distance'])
    out['aligned_variance_shift']=shift*(out['first_input_variance']-out['second_input_variance'])
    out['aligned_roughness_shift']=shift*(out['first_input_roughness']-out['second_input_roughness'])
    out['optimizer_exposure_interaction']=out['weighted_implementation']*(out['target_weight_second_moment']-1)
    return out

