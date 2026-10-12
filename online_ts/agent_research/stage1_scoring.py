"""Stage1 four-class forecasts; unchanged outcomes and abstentions are distinct."""
import math

UNCHANGED_BAND_PERCENT = 0.25
MIN_ADOPTION_GAIN_PERCENT = 1.0
MAX_NMAE_DETERIORATION_PERCENT = 1.0
PREDICTIONS = ('improve', 'unchanged', 'harm', 'uncertain')

def observed_direction(gain_percent):
    if not math.isfinite(gain_percent):
        raise ValueError('A finite paired response is required')
    if gain_percent > UNCHANGED_BAND_PERCENT:
        return 'improve'
    if gain_percent < -UNCHANGED_BAND_PERCENT:
        return 'harm'
    return 'unchanged'

def score_predictions(predictions, gains):
    if not predictions or len(predictions) != len(gains):
        raise ValueError('Nonempty aligned predictions are required')
    if any(value not in PREDICTIONS for value in predictions):
        raise ValueError('Unknown prediction category')
    observed = [observed_direction(value) for value in gains]
    decided = [i for i, value in enumerate(predictions) if value != 'uncertain']
    correct = sum(predictions[i] == observed[i] for i in decided)
    harmful = [i for i, value in enumerate(observed) if value == 'harm']
    return dict(total=len(predictions), decided=len(decided), correct=correct,
                abstained=len(predictions)-len(decided), coverage=len(decided)/len(predictions),
                accuracy=correct/len(decided) if decided else None,
                harm_total=len(harmful), harm_warned=sum(predictions[i]=='harm' for i in harmful),
                harm_false_benefit=sum(predictions[i]=='improve' for i in harmful),
                harm_false_unchanged=sum(predictions[i]=='unchanged' for i in harmful),
                harm_abstained=sum(predictions[i]=='uncertain' for i in harmful),observed=observed)

def repeatability(gains):
    if len(gains) != 3:
        raise ValueError('Three preregistered confirmation seeds are required')
    labels = [observed_direction(value) for value in gains]
    return dict(labels=labels, stable_category=labels[0] if len(set(labels)) == 1 else 'uncertain',
                all_raw_signs_positive=all(value > 0 for value in gains))
