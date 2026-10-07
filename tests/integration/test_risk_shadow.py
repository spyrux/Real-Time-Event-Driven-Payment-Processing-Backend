import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import MagicMock

from api.main import create_payment, get_payment_risk
from api.models import CreatePaymentRequest
from consumer.payment_consumer import process_event
from consumer.risk_consumer import process_message
from db.connection import get_connection
from risk.features import snapshot_features


def test_shadow_assessment_never_changes_payment_and_is_idempotent():
    user = "user_risk_" + uuid.uuid4().hex
    conn = get_connection()
    payment_ids = []
    try:
        for amount in (100, 200):
            payment = create_payment(CreatePaymentRequest(
                user_id=user, amount=amount, currency="USD", merchant_category="shopping_net",
            ))
            payment_ids.append(payment["payment_id"])
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM outbox WHERE aggregate_id = %s", (payment_ids[-1],))
            event = cur.fetchone()[0]
        conn.commit()
        assert event["currency"] == "USD"
        assert event["risk_features"]["prior_count_1h"] == 1
        assert event["risk_features"]["prior_amount_1h"] == 100
        assert get_payment_risk(payment_ids[-1])["status"] == "pending"
        message = MagicMock()
        message.value.return_value = json.dumps(event).encode()
        message.topic.return_value = "payment-events"
        message.partition.return_value = 0
        message.offset.return_value = 1
        model = MagicMock()
        model.assess.return_value = {"status": "ok", "fraud_score": 0.99, "suggested_action": "review"}
        process_message(message, model)
        process_message(message, model)
        model.assess.assert_called_once()
        assert get_payment_risk(payment_ids[-1])["assessment"]["mode"] == "shadow"
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM payments WHERE payment_id = %s", (payment_ids[-1],))
            assert cur.fetchone()[0] == "pending"
        conn.commit()
        process_event(event, time.time())
        with conn.cursor() as cur:
            cur.execute("SELECT balance FROM users WHERE user_id = %s", (user,))
            assert cur.fetchone()[0] == -200
        conn.commit()
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM processed_events WHERE event_id IN (SELECT event_id FROM outbox WHERE aggregate_id = ANY(%s))", (payment_ids,))
            cur.execute("DELETE FROM risk_assessments WHERE payment_id = ANY(%s)", (payment_ids,))
            cur.execute("DELETE FROM outbox WHERE aggregate_id = ANY(%s)", (payment_ids,))
            cur.execute("DELETE FROM payments WHERE payment_id = ANY(%s)", (payment_ids,))
            cur.execute("DELETE FROM users WHERE user_id = %s", (user,))
        conn.commit()
        conn.close()


def test_live_history_excludes_future_and_other_currencies():
    user = "user_history_" + uuid.uuid4().hex
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for seconds, amount, currency in ((-60, 100, "USD"), (-60, 900, "EUR"), (3600, 800, "USD"), (-7200, 700, "USD")):
                cur.execute("""
                    INSERT INTO payments(payment_id, user_id, amount, status, currency, created_at)
                    VALUES (%s, %s, %s, 'pending', %s, %s)
                """, (str(uuid.uuid4()), user, amount, currency,
                      datetime.fromtimestamp(time.time() + seconds, timezone.utc).replace(tzinfo=None)))
            _, features = snapshot_features(cur, user, 200, "USD", "home")
        assert features["prior_count_1h"] == 1
        assert features["prior_amount_1h"] == 100
    finally:
        conn.rollback()
        conn.close()


def test_concurrent_payment_snapshots_see_the_preceding_commit():
    user = "user_parallel_" + uuid.uuid4().hex
    conn = get_connection()
    try:
        def submit(amount):
            return create_payment(CreatePaymentRequest(user_id=user, amount=amount, currency="USD"))

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(submit, (100, 200)))
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM outbox WHERE payload->>'user_id' = %s", (user,))
            features = [row[0]["risk_features"] for row in cur.fetchall()]
        assert sorted(f["prior_count_1h"] for f in features) == [0, 1]
        later = next(f for f in features if f["prior_count_1h"] == 1)
        assert later["amount"] + later["prior_amount_1h"] == 300
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM outbox WHERE payload->>'user_id' = %s", (user,))
            cur.execute("DELETE FROM payments WHERE user_id = %s", (user,))
        conn.commit()
        conn.close()
