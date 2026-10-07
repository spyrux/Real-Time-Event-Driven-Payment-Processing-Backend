from unittest.mock import MagicMock

import pytest

from consumer import payment_consumer


@pytest.mark.parametrize("currency", ["EUR", "JPY"])
def test_consumer_rejects_non_usd_before_changing_balances(monkeypatch, currency):
    connection = MagicMock()
    monkeypatch.setattr(payment_consumer, "get_db_connection", connection)
    with pytest.raises(ValueError, match="only supports USD"):
        payment_consumer.process_event({"currency": currency}, 0)
    connection.assert_not_called()


@pytest.mark.parametrize("currency", [None, "unknown", "USD"])
def test_consumer_processes_usd_and_legacy_events(monkeypatch, currency):
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.fetchone.side_effect = [("payment-test",), (1,)]
    monkeypatch.setattr(payment_consumer, "get_db_connection", lambda: connection)
    event = {"event_id": "event-test", "payment_id": "payment-test",
             "user_id": "user_123", "amount": 100}
    if currency is not None:
        event["currency"] = currency
    payment_consumer.process_event(event, 0)
    assert any("UPDATE users SET balance = balance -" in call.args[0]
               and call.args[1] == (100, "user_123") for call in cursor.execute.call_args_list)
    connection.commit.assert_called_once()
