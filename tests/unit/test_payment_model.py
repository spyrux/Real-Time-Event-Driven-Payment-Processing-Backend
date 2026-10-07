from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import main
from api.models import CreatePaymentRequest


def test_valid_payment_request():
    payment = CreatePaymentRequest(
        user_id="user_123",
        amount=100
    )

    assert payment.user_id == "user_123"
    assert payment.amount == 100

def test_payment_request_rejects_zero_amount():
    with pytest.raises(ValidationError):
        CreatePaymentRequest(
            user_id="user_123",
            amount=0
        )

def test_payment_request_rejects_negative_amount():
    with pytest.raises(ValidationError):
        CreatePaymentRequest(
            user_id="user_123",
            amount=-50
        )

def test_payment_request_rejects_short_user_id():
    with pytest.raises(ValidationError):
        CreatePaymentRequest(
            user_id="usr",
            amount=100
        )


def test_payment_request_accepts_usd_and_legacy_currency():
    assert CreatePaymentRequest(user_id="user_123", amount=100, currency="USD").currency == "USD"
    assert CreatePaymentRequest(user_id="user_123", amount=100).currency is None
    assert CreatePaymentRequest(user_id="user_123", amount=100, currency=None).currency is None


@pytest.mark.parametrize("currency", ["EUR", "JPY", "ZZZ", "usd"])
def test_payment_rejects_other_currencies_before_database_access(monkeypatch, currency):
    connection = MagicMock(side_effect=AssertionError("Invalid request reached the database"))
    monkeypatch.setattr(main, "get_connection", connection)
    with TestClient(main.app) as client:
        response = client.post("/payments", json={
            "user_id": "user_123", "amount": 100, "currency": currency,
        })
    assert response.status_code == 422
    connection.assert_not_called()
