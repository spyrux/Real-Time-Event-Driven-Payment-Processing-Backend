import logging
import os
import uuid
from datetime import timezone

from fastapi import FastAPI, HTTPException
from psycopg2.extras import Json

from api.models import CreatePaymentRequest
from db.connection import get_connection
from risk.features import FEATURE_VERSION, snapshot_features
from utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/")
def health_check():
    return {"message": "API is running"}


@app.post("/payments")
def create_payment(request: CreatePaymentRequest):
    user_id = request.user_id
    amount = request.amount

    if not user_id.startswith("user_"):
        raise HTTPException(
            status_code=400,
            detail="Invalid user_id format. Must start with 'user_'."
        )

    conn = get_connection()
    cur = conn.cursor()

    try:
        payment_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())
        currency = request.currency or "unknown"
        created_at, features = snapshot_features(
            cur, user_id, amount, currency, request.merchant_category,
        )

        event = {
            "event_id": event_id,
            "payment_id": payment_id,
            "user_id": user_id,
            "amount": amount,
            "currency": currency,
            "event_type": "payment_created",
            "timestamp": created_at.timestamp(),
            "risk_feature_version": FEATURE_VERSION,
            "risk_features": features,
            "classification_context": {
                "merchant": request.merchant, "description": request.description,
            },
        }

        cur.execute("""
            INSERT INTO payments (payment_id, user_id, amount, status, currency, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)""", (
                payment_id, user_id, amount, "pending", currency,
                created_at.astimezone(timezone.utc).replace(tzinfo=None),
            ))

        cur.execute("""
            INSERT INTO outbox (event_id, aggregate_id, event_type, payload)
            VALUES (%s, %s, %s, %s)""", (event_id, payment_id, event["event_type"], Json(event),))

        conn.commit()

        if os.getenv("FAULT_INJECT_CRASH_AFTER_PAYMENT_COMMIT") == "true":
            os._exit(1)

        return {
            "payment_id": payment_id,
            "status": "pending"
        }

    except Exception as e:
        conn.rollback()
        logger.exception("Failed to create payment for user %s", user_id)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()


@app.get("/payments/{payment_id}/risk")
def get_payment_risk(payment_id: str):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM payments WHERE payment_id = %s", (payment_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Payment not found")
            cur.execute("""
                SELECT assessment, created_at FROM risk_assessments
                WHERE payment_id = %s ORDER BY created_at DESC LIMIT 1
            """, (payment_id,))
            row = cur.fetchone()
        return {
            "payment_id": payment_id, "mode": "shadow",
            "status": "assessed" if row else "pending",
            "assessment": row[0] if row else None,
            "assessed_at": row[1] if row else None,
        }
    finally:
        conn.close()


@app.get("/payments/{payment_id}")
def get_payment_status(payment_id: str):
    logger.info("Fetching payment status for %s", payment_id)

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT payment_id, user_id, amount, status, created_at, currency
            FROM payments
            WHERE payment_id = %s
        """, (payment_id,))

        result = cur.fetchone()

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Payment not found"
            )

        payment = {
            "payment_id": result[0],
            "user_id": result[1],
            "amount": result[2],
            "status": result[3],
            "created_at": result[4],
            "currency": result[5],
        }

        return payment

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch payment status for %s", payment_id)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cur.close()
        conn.close()
