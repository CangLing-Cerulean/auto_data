"""Frozen general-statistic, case-match and learned-signal selection comparators."""
import math
import numpy as np

FIELDS = ('weighted_edit_ratio','edit_ratio','roughness_change','edited_fraction','parent_nmse_ratio','parent_nmae_ratio')
CATEGORIES = ('improve','unchanged','harm','uncertain')


def ratio(dose, error):
    if not math.isfinite(dose) or not math.isfinite(error) or dose < 0 or error < 0:
        raise ValueError('Finite nonnegative dose/error required')
    if dose == 0:
        return 0.
    if error == 0:
        return 1.
    return dose/(dose+error)


def generic_features(parent, changed, parent_metrics):
    if parent.shape != changed.shape or parent.ndim != 3 or parent.shape[1] != 32:
        raise ValueError('Paired 32-row inputs required')
    x, z = parent.astype(np.float64), changed.astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Nonfinite input')
    channel_error = np.asarray(parent_metrics['channel_NMSE'],dtype=np.float64)
    if channel_error.shape != (x.shape[2],) or not np.isfinite(channel_error).all() or (channel_error < 0).any():
        raise ValueError('Invalid parent channel error')
    weights = channel_error/channel_error.sum() if channel_error.sum() > 0 else np.full(len(channel_error),1/len(channel_error))
    channel_dose = np.mean((z-x)**2,axis=(0,1))
    old_variation, new_variation = float(np.mean(abs(np.diff(x,axis=1)))), float(np.mean(abs(np.diff(z,axis=1))))
    denominator = old_variation+new_variation
    result = dict(weighted_edit_ratio=ratio(float(np.dot(weights,channel_dose)),parent_metrics['NMSE']),
                  edit_ratio=ratio(float(channel_dose.mean()),parent_metrics['NMSE']),
                  roughness_change=(old_variation-new_variation)/denominator if denominator else 0.,
                  edited_fraction=float(np.mean(changed != parent)),
                  parent_nmse_ratio=ratio(parent_metrics['NMSE'],1.),
                  parent_nmae_ratio=ratio(parent_metrics['NMAE'],1.))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite generic feature')
    return result


def nearest(cases, query, multivariate):
    train = np.asarray([[case['features'][key] for key in FIELDS] for case in cases],dtype=np.float64)
    x = np.asarray([query['features'][key] for key in FIELDS],dtype=np.float64)
    if multivariate:
        mean, scale = train.mean(axis=0),train.std(axis=0)
        scale[scale == 0] = 1.
        distances = np.sum(((train-mean)/scale-(x-mean)/scale)**2,axis=1)
    else:
        distances = abs(train[:,0]-x[0])
    closest = np.flatnonzero(distances == distances.min())
    labels = {cases[index]['category'] for index in closest}
    prediction = next(iter(labels)) if len(labels) == 1 else 'uncertain'
    return prediction,float(np.mean([cases[index]['mean_gain_percent'] for index in closest])),[cases[index]['id'] for index in closest]


def signal_category(value, thresholds):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Signal must be finite in [0,1]')
    if len(thresholds) != 2 or not 0 < thresholds[0] < thresholds[1] < 1:
        raise ValueError('Two ordered frozen thresholds required')
    if value <= thresholds[0]:
        return 'unchanged'
    if value <= thresholds[1]:
        return 'improve'
    return 'harm'


def ranking(first, second):
    return 'P1>CP1' if first > second else 'CP1>P1' if second > first else 'tie'


def deterministic_predictions(construction, selection, signals, thresholds):
    result = {}
    for method in ('generic_stats','case_match','new_signal'):
        predictions,scores,matched = {},{},{}
        for case in selection:
            key = case['id']
            if method == 'new_signal':
                label = signal_category(signals[key],thresholds)
                members = [item for item in construction if item['category'] == label]
                score = float(np.mean([item['mean_gain_percent'] for item in members]))
                matches = [item['id'] for item in members]
            else:
                label,score,matches = nearest(construction,case,method=='case_match')
            predictions[key],scores[key],matched[key] = label,score,matches
        result[method] = dict(predictions=predictions,ranking_score=scores,matched_construction=matched,
            ranking={task:ranking(scores[task+'_P1'],scores[task+'_CP1']) for task in ('S1','S2')})
    return result


def recommend(observed):
    result = {}
    for task in ('S1','S2'):
        eligible = [item for item in observed if item['task']==task and item['gain_percent']['global']['NMSE'] >= 1.
                    and item['gain_percent']['global']['NMAE'] >= -1.]
        eligible.sort(key=lambda item:(-item['gain_percent']['global']['NMSE'],-item['gain_percent']['global']['NMAE'],item['action']))
        result[task] = eligible[0]['action'] if eligible else 'PARENT'
    return result
