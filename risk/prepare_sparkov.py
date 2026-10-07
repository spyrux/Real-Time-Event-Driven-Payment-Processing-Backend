"""Normalize the Kaggle Sparkov archive without retaining card/customer details."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

SOURCE = "https://www.kaggle.com/datasets/kartik2112/fraud-detection"


def prepare(archive, output):
    output = Path(output)
    counts = {"rows": 0, "fraud": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as data, output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "payment_id", "user_id", "timestamp", "amount", "currency",
            "merchant_category", "is_fraud",
        ])
        writer.writeheader()
        for name in ("fraudTrain.csv", "fraudTest.csv"):
            with data.open(name) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw)):
                    # The source has naive simulated calendar times. Interpret
                    # those consistently as UTC; do not use its nonstandard epoch.
                    date = datetime.strptime(
                        row["trans_date_trans_time"], "%Y-%m-%d %H:%M:%S",
                    ).replace(tzinfo=timezone.utc)
                    minor_units = Decimal(row["amt"]) * 100
                    if minor_units != minor_units.to_integral_value() or minor_units <= 0:
                        raise ValueError("Source amount must be positive with at most two decimals")
                    writer.writerow({
                        "payment_id": row["trans_num"],
                        "user_id": "user_" + hashlib.sha256(row["cc_num"].encode()).hexdigest(),
                        "timestamp": int(date.timestamp()),
                        "amount": int(minor_units), "currency": "USD",
                        "merchant_category": row["category"], "is_fraud": row["is_fraud"],
                    })
                    counts["rows"] += 1
                    counts["fraud"] += int(row["is_fraud"])
    provenance = {
        "source": SOURCE, "data_kind": "synthetic", **counts,
        "mapping": "Naive simulated times treated as UTC; amounts converted to USD cents.",
        "excluded": "Card number, demographics, names, addresses, locations, and post-outcome data.",
        "evaluation": "Combined archive is re-split chronologically by risk.train.",
    }
    output.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
