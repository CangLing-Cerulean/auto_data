"""Operational evidence projected from the original TrialObservation ledger."""
import time
from dataclasses import asdict

from ..config import GateError, digest
from ..memory import query_observations, outbound_observation
from ..models import Candidate


def observations(repo, now, limit=6):
    values = query_observations(repo.store, repo.key, decision_ns=time.time_ns(), limit=limit * 3)
    result = []
    for envelope in values:
        value = envelope["observation"]
        if value["event_cutoff_ns"] > now:
            continue
        result.append({"record_id": envelope["observation_id"],
                       **outbound_observation(envelope, envelope["observation_id"]),
                       "scope": {"task_key": repo.key, "snapshot_sha": value["snapshot_sha"],
                                 "protocol": value["evaluation_protocol"]}})
        if len(result) == limit:
            break
    return result


def verify_contrasts(repo, experiment, reports, now):
    if experiment["purpose"] != "diagnosis":
        return []
    results = []
    for i, left in enumerate(reports):
        for right in reports[i + 1:]:
            if not all("score" in r and r.get("validation_artifact") and len(r.get("folds", [])) >= 2 for r in (left, right)):
                continue
            a, b = asdict(Candidate.from_dict(left["candidate"])), asdict(Candidate.from_dict(right["candidate"]))
            changed = [key for key in a if a[key] != b[key]]
            if (len(changed) != 1 or left["snapshot_sha"] != right["snapshot_sha"]
                    or left["normalization"] != right["normalization"]
                    or [f["issue_times"] for f in left["folds"]] != [f["issue_times"] for f in right["folds"]]):
                continue
            evidence = [left["validation_artifact"], right["validation_artifact"]]
            record_id = "experience-" + digest(evidence)[:24]
            value = {"record_id": record_id, "scope": {"task_key": repo.key, "snapshot_sha": left["snapshot_sha"],
                     "evaluation_protocol": "rolling-origin-v1"}, "evidence": evidence,
                     "conditions": {"baseline": a, "variant": b, "changed_factor": changed[0]},
                     "measured_score_delta": right["score"] - left["score"],
                     "counter_evidence": ["Cross-device and out-of-window generalization untested"],
                     "version": 1, "status": "verified_within_matched_development_contrast",
                     "created_at": now}
            repo.put(record_id, "verified_experience", value, now)
            results.append(record_id)
    return results
