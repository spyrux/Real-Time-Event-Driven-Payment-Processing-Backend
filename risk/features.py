"""One feature contract shared by historical replay and live requests."""

import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import groupby

FEATURE_VERSION = "payment-risk-v1"
FEATURE_NAMES = [
    "amount", "prior_count_1h", "prior_amount_1h", "hour_utc", "weekday_utc",
    "currency", "merchant_category",
]
CATEGORICAL_FEATURES = ["currency", "merchant_category"]


def parse_timestamp(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        result = date.timestamp()
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


def make_features(amount, timestamp, currency, merchant_category, count=0, total=0):
    date = datetime.fromtimestamp(parse_timestamp(timestamp), tz=timezone.utc)
    amount = float(amount)
    if not math.isfinite(amount) or amount <= 0 or not amount.is_integer():
        raise ValueError("amount must be a positive integer in minor units")
    return {
        "amount": amount,
        "prior_count_1h": int(count),
        "prior_amount_1h": float(total),
        "hour_utc": date.hour,
        "weekday_utc": date.weekday(),
        "currency": currency or "unknown",
        "merchant_category": merchant_category or "unknown",
    }


def historical_features(rows):
    """Rows must be time ordered; same-time records cannot see each other.

    Count all earlier attempts (including fraud), within the same user/currency.
    Labels never enter features. Call on the entire timeline before splitting so
    test records can use earlier activity, just as live requests do.
    """
    history = defaultdict(deque)
    totals = defaultdict(float)
    result = []
    for timestamp, group in groupby(rows, key=lambda row: row["timestamp"]):
        batch = list(group)
        for row in batch:
            key = (row["user_id"], row.get("currency") or "unknown")
            queue = history[key]
            while queue and queue[0][0] < timestamp - 3600:
                totals[key] -= queue.popleft()[1]
            result.append(make_features(
                row["amount"], timestamp, key[1], row.get("merchant_category"),
                len(queue), totals[key],
            ))
        for row in batch:
            key = (row["user_id"], row.get("currency") or "unknown")
            amount = float(row["amount"])
            history[key].append((timestamp, amount))
            totals[key] += amount
    return result


def snapshot_features(cursor, user_id, amount, currency, merchant_category):
    # Serialize API submissions for one user before taking the timestamp. The
    # lock lasts through the payment/outbox commit, including concurrent requests.
    cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (user_id,))
    cursor.execute("SELECT clock_timestamp()")
    created_at = cursor.fetchone()[0]
    cursor.execute("""
        SELECT COUNT(*), COALESCE(SUM(amount), 0)
        FROM payments
        WHERE user_id = %s AND currency = %s
          AND created_at >= %s AND created_at < %s
    """, (
        user_id, currency,
        datetime.fromtimestamp(created_at.timestamp() - 3600, timezone.utc).replace(tzinfo=None),
        created_at.astimezone(timezone.utc).replace(tzinfo=None),
    ))
    count, total = cursor.fetchone()
    return created_at, make_features(
        amount, created_at.timestamp(), currency, merchant_category, count, total,
    )
