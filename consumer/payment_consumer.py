import json
import logging
import os
import time

import psycopg2
from confluent_kafka import Consumer, Producer

from config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP_ID,
    KAFKA_DLQ_TOPIC,
    KAFKA_PAYMENT_TOPIC,
)
from utils.logging_config import configure_logging

PROCESS_ID = os.getpid()
DLQ_TOPIC = KAFKA_DLQ_TOPIC
BOOTSTRAP_SERVERS = KAFKA_BOOTSTRAP_SERVERS
TOPIC_NAME = KAFKA_PAYMENT_TOPIC
GROUP_ID = KAFKA_CONSUMER_GROUP_ID

logger = logging.getLogger(__name__)

dlq_producer = Producer({
    "bootstrap.servers": BOOTSTRAP_SERVERS
})

DB_CONFIG = {
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "host": DB_HOST,
    "port": DB_PORT
}


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def process_event(event, start_time):
    # Legacy events omit currency or use "unknown". Explicit currencies must
    # match this demo's single USD ledger, including events from other producers.
    if event.get("currency") not in (None, "unknown", "USD"):
        raise ValueError("Payment ledger only supports USD")

    logger.info("Received event %s for payment %s", event["event_id"], event["payment_id"])

    conn = get_db_connection()
    conn.autocommit = False
    cursor = conn.cursor()

    try:
        event_id = event["event_id"]
        payment_id = event["payment_id"]
        user_id = event["user_id"]
        amount = event["amount"]

        cursor.execute("""
            INSERT INTO processed_events (event_id)
            VALUES (%s)
            ON CONFLICT (event_id) DO NOTHING
        """, (event_id,))

        if cursor.rowcount == 0:
            conn.rollback()
            logger.info("Duplicate event detected for event_id=%s", event_id)
            return

        cursor.execute(
            "INSERT INTO users (user_id, balance) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO NOTHING",
            (user_id, 0)
        )

        cursor.execute(
            "UPDATE users SET balance = balance - %s WHERE user_id = %s",
            (amount, user_id)
        )

        cursor.execute("""
            UPDATE payments
            SET status = 'processed'
            WHERE payment_id = %s AND status = 'pending'
            RETURNING payment_id
        """, (payment_id,))

        result = cursor.fetchone()

        if result is None:
            logger.warning("Payment %s already processed or in an invalid state", payment_id)
            conn.rollback()
            return

        cursor.execute(
            "UPDATE metrics SET total_processed = total_processed + 1 WHERE id = 1"
        )
        cursor.execute(
            "SELECT total_processed FROM metrics WHERE id = 1"
        )
        total = cursor.fetchone()[0]

        conn.commit()

        logger.info("Processed payment %s for user %s", payment_id, user_id)

        if total == 1000:
            logger.info("All events processed in %.2fs", time.time() - start_time)

    except psycopg2.Error:
        conn.rollback()

        cursor.execute("""
            UPDATE payments
            SET status = 'failed'
            WHERE payment_id = %s AND status = 'pending'
            RETURNING payment_id
        """, (payment_id,))

        result = cursor.fetchone()

        if result is None:
            logger.warning(
                "Payment %s already updated, skipping failure status change",
                payment_id,
            )
        else:
            conn.commit()

        logger.exception("Database error while processing payment %s", payment_id)
        raise

    finally:
        cursor.close()
        conn.close()


def main():
    configure_logging()

    start_time = None

    logger.info("Starting consumer process %s", PROCESS_ID)

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False
    })

    consumer.subscribe([TOPIC_NAME])
    logger.info("Subscribed to topic %s", TOPIC_NAME)

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                logger.debug("No message received in this poll interval")
                continue

            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue

            logger.info(
                "Processor %s handling partition=%s offset=%s",
                PROCESS_ID,
                msg.partition(),
                msg.offset(),
            )

            try:
                event = json.loads(msg.value().decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON message")
                continue

            required_fields = ["event_id", "payment_id", "user_id", "amount"]
            if not all(field in event for field in required_fields):
                logger.warning("Skipping invalid event payload: %s", event)
                continue

            try:
                if start_time is None:
                    start_time = time.time()

                process_event(event, start_time)
                consumer.commit(message=msg)
                logger.info("Committed offset %s", msg.offset())
            except Exception as exc:
                logger.exception("Processing failed for event_id=%s", event.get("event_id"))

                dlq_event = {
                    "original_event": event,
                    "error": str(exc),
                    "failed_at": time.time(),
                    "partition": msg.partition(),
                    "offset": msg.offset()
                }

                dlq_producer.produce(
                    DLQ_TOPIC,
                    key=msg.key(),
                    value=json.dumps(dlq_event)
                )
                dlq_producer.flush()

                logger.warning("Sent event_id=%s to DLQ", event["event_id"])
                consumer.commit(message=msg)
                logger.info("Committed failed message offset %s after DLQ handoff", msg.offset())
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
