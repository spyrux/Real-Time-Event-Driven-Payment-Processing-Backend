import asyncio
import json
from copy import deepcopy

import httpx
import numpy as np
import pytest

from risk.text_compare import (
    JevTextClient,
    add_coverage_metrics,
    fit_combiner_and_thresholds,
    payload,
    template,
    validate_groups,
)


def test_template_groups_variable_numbers_and_links():
    assert template("Claim 123 at https://a.example/x") == template("CLAIM 999 at https://b.example/y")


def test_payload_contains_only_message_and_no_labels():
    body = payload("An ordinary message")
    assert body["state"] == {"message": "An ordinary message"}
    assert set(body["questions"]) == {"fraud", "credentials", "deceptive_payment"}


def test_template_group_cannot_cross_train_test_boundary():
    with pytest.raises(ValueError, match="crosses"):
        validate_groups([{"id": "a", "group": 1, "phase": "train"},
                         {"id": "b", "group": 1, "phase": "test"}])


def test_external_and_test_labels_cannot_change_fit_or_thresholds():
    rows = []
    for phase in ("hybrid_fit", "calibration", "test", "external"):
        for i in range(40):
            label = i % 2
            score = 0.1 + 0.6 * label + 0.001 * i
            rows.append({"phase": phase, "label": label, "catboost": score,
                         "tfidf_logistic": score, "jev": {"status": "ok", "scores": [score] * 3}})
    modified = deepcopy(rows)
    for row in modified:
        if row["phase"] in {"test", "external"}:
            row["label"] = 1 - row["label"]
    original_model, original_thresholds = fit_combiner_and_thresholds(rows, 0.01)
    changed_model, changed_thresholds = fit_combiner_and_thresholds(modified, 0.01)
    np.testing.assert_array_equal(original_model.coef_, changed_model.coef_)
    assert original_thresholds == changed_thresholds


def test_invalid_jev_text_answer_is_unavailable():
    def handler(request):
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
            "fraud": {"type": "noul", "noul": True}}})

    async def run():
        async with JevTextClient("test", transport=httpx.MockTransport(handler)) as client:
            result = await client.assess("message")
            assert result["status"] == "unavailable"
    asyncio.run(run())


def test_unavailable_request_counts_as_review_in_full_population():
    rows = [
        {"phase": "test", "label": 0, "catboost": 0.1, "tfidf_logistic": 0.1,
         "jev": {"status": "unavailable", "error": "timeout"}},
        {"phase": "test", "label": 1, "catboost": 0.9, "tfidf_logistic": 0.9,
         "jev": {"status": "ok"}, "jev_only": 0.9, "hybrid": 0.9},
    ]
    report = {"thresholds": dict.fromkeys(["catboost", "tfidf_logistic", "jev_only", "hybrid"], 0.5),
              "evaluations": {"test": {}}}
    add_coverage_metrics(report, rows)
    assert report["jev_errors"] == {"timeout": 1}
    full = report["evaluations"]["test"]["including_unavailable_routed_to_review"]["hybrid"]
    assert full["rows"] == 2
    assert full["sample_counts"] == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}


@pytest.mark.parametrize("body", [
    None, {"model": "jev-1.13.0", "answers": []},
    {"model": "jev-1.13.0", "answers": {"fraud": None}},
])
def test_jev_text_rejects_malformed_response_shapes(body):
    async def check():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(body)))
        async with JevTextClient("fake", transport=transport) as client:
            result = await client.assess("message")
        assert result["status"] == "unavailable"
    asyncio.run(check())


def test_jev_text_keeps_valid_scores_when_usage_is_null():
    body = {"model": "jev-1.13.0", "usage": None, "answers": {
        name: {"type": "noul", "noul": 0.25}
        for name in ("fraud", "credentials", "deceptive_payment")}}

    async def check():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        async with JevTextClient("fake", transport=transport) as client:
            result = await client.assess("message")
        assert result["status"] == "ok"
        assert result["scores"] == [0.25] * 3
        assert result["input_tokens"] is None
    asyncio.run(check())
