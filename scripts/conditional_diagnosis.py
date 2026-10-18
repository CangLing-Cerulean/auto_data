"""Small numerical core for prequential conditional diagnostic evidence.

Scores measure prediction, never causal truth. No dataset or response files are
opened here: the caller supplies only the currently permitted records.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paired_response(parent, child, target_mask, weights=None):
    """Summarize squared errors; region means share the global parent denominator."""
    parent, child = np.asarray(parent, dtype=np.float64), np.asarray(child, dtype=np.float64)
    mask = np.asarray(target_mask)
    if parent.shape != child.shape or parent.shape != mask.shape or mask.dtype != bool:
        raise ValueError("Aligned parent/child arrays and a same-shape boolean mask are required")
    if not np.isfinite(parent).all() or not np.isfinite(child).all():
        raise ValueError("Nonfinite residuals")
    w = np.ones(parent.shape) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != parent.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid fixed evaluation weights")
    mass = float(w[mask].sum() / w.sum())
    if not 0 < mass < 1:
        raise ValueError("Both target and complement must have positive evaluation mass")
    mean = lambda a, m: float(np.sum(a[m] * w[m]) / np.sum(w[m]))
    all_mask = np.ones(parent.shape, dtype=bool)
    p2, c2 = parent * parent, child * child
    lp, lc = mean(p2, all_mask), mean(c2, all_mask)
    if lp <= 0:
        raise ValueError("Relative response is undefined when parent loss is zero")
    ps, pc = mean(p2, mask), mean(p2, ~mask)
    cs, cc = mean(c2, mask), mean(c2, ~mask)
    ds, dc = (ps - cs) / lp, (pc - cc) / lp
    y, z = (lp - lc) / lp, ds - dc
    if not np.allclose([y, ds, dc], [mass * ds + (1-mass) * dc,
                                     y + (1-mass) * z, y - mass * z], rtol=1e-12, atol=1e-12):
        raise ArithmeticError("Response decomposition failed")
    lpa, lca = mean(np.abs(parent), all_mask), mean(np.abs(child), all_mask)
    return dict(Y=y, Z=z, p=mass, d_target=ds, d_complement=dc,
                parent_loss=lp, child_loss=lc, parent_target_loss=ps,
                parent_complement_loss=pc, child_target_loss=cs, child_complement_loss=cc,
                NMAE_gain=(lpa-lca)/lpa, element_count=parent.size,
                target_element_count=int(mask.sum()))


def normal_logpdf(observed, mean, std):
    values = np.asarray([observed, mean, std], dtype=np.float64)
    if not np.isfinite(values).all() or std <= 0:
        raise ValueError("A finite observation, mean and positive standard deviation are required")
    return float(-.5 * math.log(2*math.pi) - math.log(std) - .5*((observed-mean)/std)**2)


@dataclass(frozen=True)
class GaussianPrediction:
    mean: float
    std: float

    def logpdf(self, observed):
        return normal_logpdf(observed, self.mean, self.std)


@dataclass(frozen=True)
class LinearGaussian:
    """Ridge mean with leave-group-out predictive-residual scale.

    Groups are task identities, not seeds/windows. Scale is a predictive model
    estimate; it does not turn the evidence score into a calibrated test.
    """
    center: tuple
    scale: tuple
    beta: tuple
    std: float
    feature_names: tuple
    train_record_ids: tuple
    ridge: float
    scale_floor: float

    @staticmethod
    def _fit_mean(x, y, group, ridge):
        names, counts = np.unique(group, return_counts=True)
        lookup = dict(zip(names, counts))
        w = np.asarray([1/lookup[g] for g in group])
        # Every task has equal mass; ridge has a fixed interpretation per task.
        center = np.average(x, axis=0, weights=w)
        variance = np.average((x-center)**2, axis=0, weights=w)
        scale = np.sqrt(variance)
        scale = np.where(scale > 1e-12, scale, 1.)
        design = np.column_stack([np.ones(len(x)), (x-center)/scale])
        penalty = np.eye(design.shape[1]) * ridge
        penalty[0, 0] = 0.
        beta = np.linalg.solve(design.T @ (w[:,None]*design) + penalty,
                               design.T @ (w*y))
        return center, scale, beta

    @classmethod
    def fit(cls, x, y, groups, record_ids, feature_names, ridge=1., scale_floor=.01):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        group = np.asarray(groups, dtype=str)
        if x.ndim != 2 or y.shape != (len(x),) or group.shape != y.shape:
            raise ValueError("Incompatible training shapes")
        if len(feature_names) != x.shape[1] or len(record_ids) != len(x):
            raise ValueError("Feature/record identities do not match")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("Duplicate training records")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or ridge <= 0 or scale_floor <= 0:
            raise ValueError("Invalid finite training values or regularization")
        unique = np.unique(group)
        if len(unique) < 2:
            raise ValueError("At least two task groups are needed to estimate predictive scale")
        residual = np.empty(len(x))
        for held in unique:
            train = group != held
            center, scale, beta = cls._fit_mean(x[train], y[train], group[train], ridge)
            design = np.column_stack([np.ones((~train).sum()), (x[~train]-center)/scale])
            residual[~train] = y[~train] - design @ beta
        task_mse = [np.mean(residual[group == name]**2) for name in unique]
        std = max(float(np.sqrt(np.mean(task_mse))), float(scale_floor))
        center, scale, beta = cls._fit_mean(x, y, group, ridge)
        return cls(tuple(center), tuple(scale), tuple(beta), std, tuple(feature_names),
                   tuple(record_ids), float(ridge), float(scale_floor))

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (len(self.feature_names),) or not np.isfinite(x).all():
            raise ValueError("Prediction feature shape or values invalid")
        design = np.r_[1., (x-np.asarray(self.center))/np.asarray(self.scale)]
        return GaussianPrediction(float(design @ self.beta), self.std)

    def snapshot(self):
        return dict(center=list(self.center), scale=list(self.scale), beta=list(self.beta),
                    std=self.std, feature_names=list(self.feature_names),
                    train_record_ids=list(self.train_record_ids), ridge=self.ridge,
                    scale_floor=self.scale_floor)


@dataclass(frozen=True)
class DiagnosticPredictor:
    """Frozen p(Y|x) and p(Z|Y,x); the Y feature is scoring-only."""
    marginal: LinearGaussian
    conditional: LinearGaussian

    @classmethod
    def fit(cls, x, y, z, groups, ids, names, ridge=1., scale_floor=.01):
        x = np.asarray(x, dtype=np.float64)
        return cls(LinearGaussian.fit(x,y,groups,ids,names,ridge,scale_floor),
                   LinearGaussian.fit(np.column_stack([x,y]),z,groups,ids,
                                      [*names,"observed_Y_scoring_only"],ridge,scale_floor))

    def predict_gain(self, x):
        return self.marginal.predict(x)

    def log_scores(self, x, y, z):
        ey = self.marginal.predict(x).logpdf(y)
        ez = self.conditional.predict(np.r_[x,y]).logpdf(z)
        return dict(marginal=ey, conditional=ez, joint=ey+ez)

    def snapshot(self):
        return dict(marginal=self.marginal.snapshot(), conditional=self.conditional.snapshot())


def diagnostic_evidence(reference, diagnostic, x_reference, x_diagnostic, y, z):
    base = reference.log_scores(x_reference, y, z)
    proposed = diagnostic.log_scores(x_diagnostic, y, z)
    result = {key:proposed[key]-base[key] for key in base}
    if not math.isclose(result['conditional'], result['joint']-result['marginal'],
                        rel_tol=1e-10, abs_tol=1e-10):
        raise ArithmeticError("Joint/conditional/marginal factorization mismatch")
    return result


class PrequentialLedger:
    """Small explicit state machine: register -> score -> record -> refit externally."""
    def __init__(self, mode='conditional'):
        if mode not in ('conditional','joint','marginal','uniform'):
            raise ValueError("Unknown update rule")
        self.mode = mode
        self.registered = {}
        self.observed = {}
        self.reliability = {}
        self.events = []

    def register(self, event_id, reference, diagnostics, x_reference, x_diagnostics):
        if event_id in self.registered or event_id in self.observed:
            raise ValueError("Experimental response already registered or observed")
        if set(diagnostics) != set(x_diagnostics):
            raise ValueError("Diagnostic versions and feature versions differ")
        models = dict(reference=reference.snapshot(), diagnostics={h:m.snapshot() for h,m in diagnostics.items()})
        # Prediction objects are immutable; arrays are stored as immutable tuples.
        entry = dict(reference=reference, diagnostics=dict(diagnostics), x_reference=tuple(x_reference),
                     x_diagnostics={h:tuple(x) for h,x in x_diagnostics.items()},
                     snapshot_hash=digest(models), model_snapshots=models)
        self.registered[event_id] = entry
        for h in diagnostics:
            self.reliability.setdefault(h, 0.)
        return dict(event_id=event_id, model_hash=entry['snapshot_hash'],
                    gain_predictions={h:m.predict_gain(x_diagnostics[h]).mean for h,m in diagnostics.items()},
                    reference_gain=reference.predict_gain(x_reference).mean)

    def observe(self, event_id, y, z, nmae_gain, source_response_id):
        if event_id in self.observed or any(e['source_response_id']==source_response_id for e in self.events):
            raise ValueError("Duplicate experimental evidence")
        if event_id not in self.registered:
            raise ValueError("Response has no pre-observation prediction registration")
        if not np.isfinite([y,z,nmae_gain]).all():
            raise ValueError("Nonfinite response")
        old = self.registered[event_id]
        scores = {h:diagnostic_evidence(old['reference'],m,old['x_reference'],old['x_diagnostics'][h],y,z)
                  for h,m in old['diagnostics'].items()}
        before = dict(self.reliability)
        for h,score in scores.items():
            if self.mode != 'uniform':
                self.reliability[h] += score[self.mode]
        # Observed benefit is not multiplied by diagnostic credibility.
        self.observed[event_id] = dict(Y=float(y), Z=float(z), NMAE_gain=float(nmae_gain))
        event = dict(event_id=event_id, source_response_id=source_response_id,
                     model_hash=old['snapshot_hash'], scores=scores, before=before,
                     after=dict(self.reliability), response=dict(self.observed[event_id]),
                     order='registered_prediction_scored_before_any_refit')
        self.events.append(event)
        return event

    def weights(self, versions):
        # Unit total prior mass for diagnostics, unit mass for reference.
        if not versions:
            return {'reference':1.}
        logits = np.r_[0., [self.reliability.get(h,0.)-math.log(len(versions)) for h in versions]]
        logits -= logits.max()
        weight = np.exp(logits); weight /= weight.sum()
        return dict(zip(['reference',*versions], map(float,weight)))

    def predict_gain(self, reference, diagnostics, x_reference, x_diagnostics):
        weights = self.weights(list(diagnostics))
        return weights['reference'] * reference.predict_gain(x_reference).mean + sum(
            weights[h] * model.predict_gain(x_diagnostics[h]).mean for h,model in diagnostics.items())

    def best_observed(self, min_gain=.01, nmae_floor=-.01):
        eligible = [(v['Y'],key) for key,v in self.observed.items()
                    if v['Y']>=min_gain and v['NMAE_gain']>=nmae_floor]
        return max(eligible, key=lambda value:(value[0],value[1]))[1] if eligible else 'PARENT'

