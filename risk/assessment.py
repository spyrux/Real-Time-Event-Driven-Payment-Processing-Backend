import asyncio
import logging
import math
import time

from risk.features import FEATURE_NAMES, FEATURE_VERSION


def assess_event(event, model=None, jev=None):
    start = time.perf_counter()
    result = {"mode": "shadow", "policy_version": "shadow-v1"}
    features = event.get("risk_features")
    valid = event.get("risk_feature_version") == FEATURE_VERSION and isinstance(features, dict)
    if valid:
        valid = set(features) == set(FEATURE_NAMES)
    if valid:
        numeric = [features[name] for name in FEATURE_NAMES[:-2]]
        valid = all(isinstance(v, (float, int)) and not isinstance(v, bool)
                    and math.isfinite(v) and v >= 0 for v in numeric)
        valid = valid and features["amount"] > 0 and features["hour_utc"] < 24
        valid = valid and features["weekday_utc"] < 7
        valid = valid and all(isinstance(features[name], str) for name in FEATURE_NAMES[-2:])
    if not valid:
        result.update({
            "baseline": {"status": "unavailable", "error": "missing_or_invalid_feature_snapshot"},
            "jev": {"status": "skipped", "reason": "missing_or_invalid_feature_snapshot"},
        })
    else:
        baseline_start = time.perf_counter()
        if model is None:
            result["baseline"] = {"status": "unavailable", "error": "model_not_configured"}
        else:
            try:
                result["baseline"] = model.assess(features)
            except Exception:
                # Preserve a durable failure, without raw exception text or an allow decision.
                logging.getLogger(__name__).exception("Baseline inference failed")
                result["baseline"] = {"status": "unavailable", "error": "inference_failed"}
        result["baseline"]["latency_ms"] = round((time.perf_counter() - baseline_start) * 1000, 3)
        context = event.get("classification_context", {})
        result["jev"] = asyncio.run(jev.assess(features, context)) if jev and isinstance(context, dict) else {
            "status": "disabled" if jev is None else "unavailable",
        }
    result["assessment_ms"] = round((time.perf_counter() - start) * 1000, 3)
    timestamp = event.get("timestamp")
    result["event_age_ms"] = (
        round(max(0, time.time() - timestamp) * 1000, 3)
        if isinstance(timestamp, (int, float)) and math.isfinite(timestamp) else None
    )
    return result
