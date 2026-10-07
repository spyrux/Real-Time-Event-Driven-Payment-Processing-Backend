"""Shadow worker. No payment-status or balance updates are permitted here."""

import argparse
import json
import logging
import os

from confluent_kafka import Consumer
from psycopg2.extras import Json

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_CONSUMER_GROUP_ID, KAFKA_PAYMENT_TOPIC
from db.connection import get_connection
from risk.assessment import assess_event
from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def process_message(message, model=None, jev=None, connection_factory=get_connection):
    source = f"{message.topic()}:{message.partition()}:{message.offset()}"
    try:
        event = json.loads(message.value())
        if not isinstance(event, dict) or not all(
            isinstance(event.get(key), str) and event[key] for key in ("event_id", "payment_id")
        ):
            raise ValueError("Invalid event identifiers")
        event_id, payment_id = event["event_id"], event["payment_id"]
    except (ValueError, TypeError, UnicodeDecodeError):
        event = None
        event_id, payment_id = f"invalid:{source}", None

    conn = connection_factory()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM risk_assessments WHERE event_id = %s", (event_id,))
            if cursor.fetchone():
                conn.commit()
                return
        conn.commit()  # No database transaction held across model/network work.
        assessment = assess_event(event, model, jev) if event is not None else {
            "mode": "shadow", "policy_version": "shadow-v1", "error": "invalid_event",
        }
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO risk_assessments (event_id, payment_id, source, assessment)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
            """, (event_id, payment_id, source, Json(assessment)))
        conn.commit()
        logger.info("Shadow assessment persisted event_id=%s", event_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def handle_message(consumer, message, model=None, jev=None):
    process_message(message, model, jev)
    consumer.commit(message=message, asynchronous=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=os.getenv("RISK_MODEL_DIR"))
    args = parser.parse_args()
    configure_logging()
    group = os.getenv("KAFKA_RISK_CONSUMER_GROUP_ID", "payment-risk-shadow")
    if group == KAFKA_CONSUMER_GROUP_ID:
        raise ValueError("Risk worker must have its own Kafka consumer group")
    model = None
    if args.model_dir:
        from risk.model import FraudModel
        model = FraudModel(args.model_dir)
    jev = None
    if os.getenv("JEV_ENABLED", "false").lower() == "true":
        from risk.jev import JevClient
        jev = JevClient(
            os.getenv("TYPESAFE_API_KEY"), os.getenv("JEV_MODEL", "jev-1.13.0"),
            float(os.getenv("JEV_DEADLINE_SECONDS", "0.8")),
        )
    if model is None and jev is None:
        raise ValueError("Configure RISK_MODEL_DIR or enable Jev before starting the worker")
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": group, "auto.offset.reset": "earliest", "enable.auto.commit": False,
    })
    consumer.subscribe([KAFKA_PAYMENT_TOPIC])
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                raise RuntimeError(str(message.error()))
            # Persistence failure terminates the worker. Do not process/commit a
            # later offset and accidentally skip a failed assessment.
            handle_message(consumer, message, model, jev)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
