import asyncio
import json

import httpx
import numpy as np
import pytest

from risk.compare import (
    choose_threshold,
    evaluate,
    hybrid_inputs,
    stratified_sample,
    weighted_metrics,
)
from risk.features import FEATURE_NAMES, make_features
from risk.jev_fraud import JevFraudClient, fraud_payload, request_hash


def test_fraud_request_excludes_labels_and_other_model_scores():
    features = {**make_features(100, 0, "USD", "shopping_net"),
                "is_fraud": 1, "user_id": "secret", "catboost_score": 0.99}
    payload = fraud_payload(features, "jev-1.13.0")
    assert set(payload["state"]) == set(FEATURE_NAMES)
    assert payload["questions"]["fraud"]["type"] == "noul"
    assert request_hash(features, "jev-1.13.0") == request_hash(
        {**features, "is_fraud": 0}, "jev-1.13.0",
    )
    assert request_hash(features, "jev-1.13.0") != request_hash(
        {**features, "amount": 200}, "jev-1.13.0",
    )


def test_stratified_weights_restore_natural_prevalence():
    rows = [{"is_fraud": int(i >= 990)} for i in range(1000)]
    selected, counts = stratified_sample(rows, list(range(1000)), 20, 10, 1)
    assert counts["1"] == {"sample": 10, "population": 10}
    labels = [rows[i]["is_fraud"] for i, _ in selected]
    weights = [weight for _, weight in selected]
    # Flagging everyone has precision equal to original prevalence, not 50%.
    result = weighted_metrics(labels, [1] * 20, weights, 0.5)
    assert result["precision"] == pytest.approx(0.01)
    assert result["sample_counts"]["fp"] == 10
    assert result["estimated_population_counts"]["fp"] == 990


def test_threshold_respects_ties_and_false_positive_budget():
    threshold = choose_threshold([0, 0, 1, 1], [0.1, 0.8, 0.8, 0.9], [1] * 4, 0)
    assert threshold == 0.9
    # The top-scoring negative prevents any automatic flags at zero FPR.
    assert choose_threshold([0, 1], [0.9, 0.1], [1, 1], 0) is None


def sample_records():
    records = []
    for phase_number, phase in enumerate(("hybrid_fit", "calibration", "test")):
        for i in range(20):
            label = i % 2
            records.append({
                "payment_id": f"{phase}-{i}", "timestamp": phase_number * 100 + i,
                "phase": phase, "label": label, "weight": 1 if label else 100,
                "catboost_score": 0.85 if label else 0.1, "catboost_ms": 0.1,
                "jev": {"status": "ok", "fraud_score": 0.7 if label else 0.2, "latency_ms": 200},
            })
    return records


def test_test_labels_cannot_change_hybrid_or_thresholds():
    records = sample_records()
    first = evaluate(records, 0.001)
    for record in records:
        if record["phase"] == "test":
            record["label"] = 1 - record["label"]
    second = evaluate(records, 0.001)
    assert first["hybrid_model"] == second["hybrid_model"]
    assert first["methods"]["catboost"]["test"]["recall"] == 1
    assert second["methods"]["catboost"]["test"]["recall"] == 0


def test_missing_jev_response_is_review_not_silently_dropped():
    records = sample_records()
    records[-2]["jev"] = {"status": "unavailable", "error": "timeout", "latency_ms": 800}
    report = evaluate(records, 0.001)
    assert report["response_coverage"]["test"]["successful"] == 19
    assert {value["test"]["rows"] for value in report["methods"].values()} == {19}
    assert report["methods"]["jev"]["all_test_review_routing"]["sample_counts"]["fp"] == 1
    assert report["methods"]["catboost"]["all_test_review_routing"]["sample_counts"]["fp"] == 0


def test_comparison_rejects_temporal_overlap_and_duplicate_rows():
    records = sample_records()
    records[0]["timestamp"] = 200
    with pytest.raises(ValueError, match="chronological"):
        evaluate(records, 0.001)
    records = sample_records()
    records[-1]["payment_id"] = records[0]["payment_id"]
    with pytest.raises(ValueError, match="unique"):
        evaluate(records, 0.001)


def test_logit_inputs_are_finite_at_extreme_scores():
    values = hybrid_inputs([{"catboost_score": 0, "jev": {"fraud_score": 1}}])
    assert np.isfinite(values).all()


def test_jev_fraud_response_and_token_usage():
    def handler(request):
        assert "is_fraud" not in json.loads(request.content)["state"]
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "fraud": {"type": "noul", "noul": 0.25}}, "usage": {"input_tokens": 100}})

    async def check():
        async with JevFraudClient("test-key", transport=httpx.MockTransport(handler)) as client:
            result = await client.assess(make_features(100, 0, "USD", "home"))
        assert result["status"] == "ok"
        assert result["fraud_score"] == 0.25
        assert result["input_tokens"] == 100
    asyncio.run(check())


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1, True, "0.9"])
def test_jev_fraud_rejects_invalid_scores(score):
    body = json.dumps({"model": "jev-1.13.0", "answers": {"fraud": {"type": "noul", "noul": score}}})

    async def check():
        async with JevFraudClient("test-key", transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=body),
        )) as client:
            result = await client.assess(make_features(100, 0, "USD", "home"))
        assert result["status"] == "unavailable"
    asyncio.run(check())


@pytest.mark.parametrize("body", [
    None, [], {"model": "jev-1.13.0", "answers": []},
    {"model": "jev-1.13.0", "answers": {"fraud": None}},
])
def test_jev_fraud_rejects_malformed_response_shapes(body):
    async def check():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(body)))
        async with JevFraudClient("fake", transport=transport) as client:
            result = await client.assess(make_features(100, 0, "USD", "home"))
        assert result["status"] == "unavailable"
    asyncio.run(check())


@pytest.mark.parametrize("usage", [None, [], {"input_tokens": True}, {"input_tokens": "100"}])
def test_jev_fraud_keeps_valid_score_when_usage_is_malformed(usage):
    body = {"model": "jev-1.13.0", "answers": {
        "fraud": {"type": "noul", "noul": 0.25}}, "usage": usage}

    async def check():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        async with JevFraudClient("fake", transport=transport) as client:
            result = await client.assess(make_features(100, 0, "USD", "home"))
        assert result["status"] == "ok"
        assert result["fraud_score"] == 0.25
        assert result["input_tokens"] is None
    asyncio.run(check())
