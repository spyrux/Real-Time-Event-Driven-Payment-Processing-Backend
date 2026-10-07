import asyncio
import csv
import json
from functools import partial
from itertools import pairwise
from unittest.mock import MagicMock

import httpx
import pytest

from consumer import risk_consumer
from risk.assessment import assess_event
from risk.features import FEATURE_VERSION, historical_features, make_features
from risk.jev import CATEGORIES, JevClient
from risk.model import FraudModel
from risk.train import (
    chronological_split,
    load_history,
    metrics,
    select_threshold,
    train,
)


def test_history_is_past_only_and_separated_by_user_currency():
    def row(timestamp, amount, user="a", currency="USD", label=0):
        return {"timestamp": timestamp, "amount": amount, "user_id": user,
                "currency": currency, "is_fraud": label}

    rows = [row(0, 100, label=1), row(0, 200), row(1, 300), row(1, 900, user="b"),
            row(2, 400, currency="EUR"), row(3600, 500), row(3601, 600)]
    features = historical_features(rows)
    assert [f["prior_amount_1h"] for f in features] == [0, 0, 300, 0, 0, 600, 800]
    assert features[2]["prior_count_1h"] == 2
    assert all("is_fraud" not in f for f in features)


def test_temporal_split_keeps_equal_timestamps_together():
    rows = [{"timestamp": i, "is_fraud": label} for i in range(20) for label in (0, 1)]
    groups = chronological_split(rows)
    for previous, following in pairwise(groups):
        assert rows[previous[-1]]["timestamp"] < rows[following[0]]["timestamp"]


def test_threshold_and_metrics_on_imbalanced_data():
    labels = [0] * 99 + [1]
    scores = [0.1] * 98 + [0.6, 0.9]
    threshold = select_threshold(labels, scores, 0.9)
    report = metrics(labels, scores, threshold)
    assert threshold == 0.9
    assert report["precision"] == report["recall"] == 1
    assert report["false_positive_rate"] == 0
    with pytest.raises(ValueError, match="No validation threshold"):
        select_threshold([0, 1], [0.9, 0.1], 0.9)


def test_training_bundle_round_trip_and_integrity(tmp_path):
    history = tmp_path / "history.csv"
    with history.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "payment_id", "user_id", "timestamp", "amount", "currency", "is_fraud",
        ])
        writer.writeheader()
        for i in range(100):
            writer.writerow({"payment_id": str(i), "user_id": f"user_{i % 4}",
                             "timestamp": i * 100, "amount": 100 if i % 2 else 90000,
                             "currency": "USD", "is_fraud": 0 if i % 2 else 1})
    bundle = tmp_path / "model"
    report = train(history, bundle, min_precision=0.8, iterations=20)
    model = FraudModel(bundle)
    assert report["test"]["recall"] > 0.8
    features = historical_features(load_history(history))[-2]
    assert model.assess(features)["suggested_action"] == "review"
    assert model.assess({**features, "currency": "EUR"})["status"] == "unavailable"
    with pytest.raises(ValueError, match="new output directory"):
        train(history, bundle)
    with (bundle / "model.cbm").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        FraudModel(bundle)


def test_csv_rejects_duplicate_ids(tmp_path):
    file = tmp_path / "invalid.csv"
    file.write_text("payment_id,user_id,timestamp,amount,is_fraud\np1,u1,1,100,0\np1,u1,2,100,1\n")
    with pytest.raises(ValueError, match="unique"):
        load_history(file)


def jev_body():
    return {"model": "jev-1.13.0", "answers": {
        "category": {"type": "choice", "choice": "unknown", "confidence": 0.9,
                     "probabilities": {key: int(key == "unknown") for key in CATEGORIES}},
        "suspicious_text": {"type": "noul", "noul": 0.1},
    }}


def test_jev_sends_only_selected_context_and_validates_response():
    def handle(request):
        body = json.loads(request.content)
        assert "user_id" not in body["state"]
        assert "merchant_category" not in body["state"]["transaction"]
        assert body["state"]["description"] == "Example purchase"
        return httpx.Response(200, json=jev_body())

    client = JevClient("fake-test-key", transport=httpx.MockTransport(handle))
    result = asyncio.run(client.assess(make_features(100, 1000, "USD", "home"), {
        "user_id": "private", "description": "Example purchase",
    }))
    assert result["status"] == "ok"
    assert result["category"] == "unknown"


@pytest.mark.parametrize("response", [
    httpx.Response(429), httpx.Response(200, json={}),
    httpx.Response(200, json={**jev_body(), "answers": {}}),
])
def test_jev_errors_are_not_allow_decisions(response):
    client = JevClient("fake-test-key", transport=httpx.MockTransport(lambda _: response))
    result = asyncio.run(client.assess({}, {}))
    assert result["status"] == "unavailable"
    assert "suggested_action" not in result


def test_jev_total_deadline():
    async def slow(_):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json=jev_body())

    client = JevClient("fake-test-key", deadline_seconds=0.01, transport=httpx.MockTransport(slow))
    result = asyncio.run(client.assess({}, {}))
    assert result["error"] == "timeout"


def test_legacy_event_does_not_invent_features():
    model = MagicMock()
    result = assess_event({"amount": 100, "user_id": "user_123"}, model)
    assert result["baseline"]["status"] == "unavailable"
    model.assess.assert_not_called()


def test_baseline_failure_stays_shadow_and_unavailable():
    model = MagicMock()
    model.assess.side_effect = ValueError("failure")
    result = assess_event({"risk_feature_version": FEATURE_VERSION,
                           "risk_features": make_features(100, 0, "USD", "home")}, model)
    assert result["mode"] == "shadow"
    assert result["baseline"]["status"] == "unavailable"
    assert "suggested_action" not in result["baseline"]


def test_duplicate_assessment_does_not_call_models():
    message = MagicMock()
    message.value.return_value = b'{"event_id":"e1","payment_id":"p1"}'
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (1,)
    model = MagicMock()
    risk_consumer.process_message(message, model, connection_factory=lambda: conn)
    model.assess.assert_not_called()
    conn.close.assert_called_once()


def test_persistence_failure_prevents_offset_commit(monkeypatch):
    consumer = MagicMock()
    process = MagicMock(side_effect=RuntimeError("database unavailable"))
    monkeypatch.setattr(risk_consumer, "process_message", process)
    with pytest.raises(RuntimeError):
        risk_consumer.handle_message(consumer, MagicMock())
    consumer.commit.assert_not_called()


@pytest.mark.parametrize("part, value", [
    ("body", None), ("answers", []), ("category", None),
    ("suspicious_text", []), ("probabilities", list(CATEGORIES)),
    ("model", "unexpected-version"),
])
def test_jev_rejects_malformed_response_shapes(part, value):
    body = jev_body()
    if part == "body":
        body = value
    elif part in {"answers", "model"}:
        body[part] = value
    elif part == "probabilities":
        body["answers"]["category"][part] = value
    else:
        body["answers"][part] = value
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(body)))
    result = asyncio.run(JevClient("fake", transport=transport).assess({}, {}))
    assert result["status"] == "unavailable"


def test_malformed_jev_assessment_is_persisted_before_offset_commit(monkeypatch):
    body = jev_body()
    body["answers"]["category"]["probabilities"] = list(CATEGORIES)
    jev = JevClient("fake", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json=body),
    ))
    message = MagicMock()
    message.value.return_value = json.dumps({
        "event_id": "event-test", "payment_id": "payment-test",
        "risk_feature_version": FEATURE_VERSION,
        "risk_features": make_features(100, 0, "USD", "home"),
    }).encode()
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = None
    consumer = MagicMock()
    order = []
    conn.commit.side_effect = lambda: order.append("database")
    consumer.commit.side_effect = lambda **_: order.append("offset")
    monkeypatch.setattr(risk_consumer, "process_message", partial(
        risk_consumer.process_message, connection_factory=lambda: conn,
    ))
    risk_consumer.handle_message(consumer, message, jev=jev)
    assessment = cursor.execute.call_args.args[1][-1].adapted
    assert assessment["jev"]["status"] == "unavailable"
    assert order[-2:] == ["database", "offset"]
    consumer.commit.assert_called_once_with(message=message, asynchronous=False)
