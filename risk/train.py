"""Train/evaluate: python -m risk.train --csv history.csv --output artifacts/risk-v1."""

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from risk.features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_VERSION,
    historical_features,
    parse_timestamp,
)


def load_history(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"payment_id", "user_id", "timestamp", "amount", "is_fraud"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {', '.join(sorted(required))}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if not row["payment_id"] or row["payment_id"] in seen or not row["user_id"]:
            raise ValueError("Payment IDs must be unique and user IDs must be present")
        seen.add(row["payment_id"])
        if row["is_fraud"] not in {"0", "1"}:
            raise ValueError("is_fraud must be 0 or 1 with a confirmed outcome")
        row["is_fraud"] = int(row["is_fraud"])
        row["timestamp"] = parse_timestamp(row["timestamp"])
    return sorted(rows, key=lambda row: row["timestamp"])


def chronological_split(rows):
    times = sorted({row["timestamp"] for row in rows})
    if len(times) < 10:
        raise ValueError("Need at least 10 distinct timestamps for chronological splits")
    validation_start = times[int(len(times) * 0.6)]
    test_start = times[int(len(times) * 0.8)]
    groups = [[], [], []]
    for index, row in enumerate(rows):
        group = 0 if row["timestamp"] < validation_start else (
            1 if row["timestamp"] < test_start else 2
        )
        groups[group].append(index)
    for name, indices in zip(("train", "validation", "test"), groups):
        if {rows[i]["is_fraud"] for i in indices} != {0, 1}:
            raise ValueError(f"{name} split must contain fraud and legitimate payments")
    return groups


def select_threshold(labels, scores, min_precision):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    candidates = [i for i in range(len(thresholds)) if precision[i] >= min_precision]
    if not candidates:
        raise ValueError("No validation threshold meets --min-precision; no model exported")
    best = max(candidates, key=lambda i: (recall[i], precision[i], thresholds[i]))
    return float(thresholds[best])


def metrics(labels, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, np.asarray(scores) >= threshold, labels=[0, 1]).ravel()
    return {
        "rows": len(labels), "fraud_count": int(tp + fn),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
    }


def train(csv_path, output, min_precision=0.9, iterations=300):
    if not math.isfinite(min_precision) or not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory to preserve existing model bundles")
    rows = load_history(csv_path)
    groups = chronological_split(rows)
    features = historical_features(rows)

    def pool(indices):
        return Pool(
            [[features[i][name] for name in FEATURE_NAMES] for i in indices],
            label=[rows[i]["is_fraud"] for i in indices],
            feature_names=FEATURE_NAMES, cat_features=CATEGORICAL_FEATURES,
        )

    training, validation, test = [pool(indices) for indices in groups]
    model = CatBoostClassifier(
        iterations=iterations, depth=6, learning_rate=0.05, loss_function="Logloss",
        random_seed=42, thread_count=2, verbose=False, allow_writing_files=False,
        has_time=True,
    )
    model.fit(training)
    validation_scores = model.predict_proba(validation)[:, 1]
    threshold = select_threshold(validation.get_label(), validation_scores, min_precision)
    report = {
        "mode": "shadow", "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES, "review_threshold": threshold,
        "minimum_validation_precision": min_precision,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "validation": metrics(validation.get_label(), validation_scores, threshold),
        "test": metrics(test.get_label(), model.predict_proba(test)[:, 1], threshold),
        "splits": {
            name: {"rows": len(indices), "start": rows[indices[0]]["timestamp"],
                   "end": rows[indices[-1]]["timestamp"]}
            for name, indices in zip(("train", "validation", "test"), groups)
        },
        "note": "Uncalibrated model scores; this report does not enable payment enforcement.",
        "supported_currencies": sorted({rows[i].get("currency") or "unknown" for i in groups[0]}),
        "merchant_categories": sorted({rows[i].get("merchant_category") or "unknown" for i in groups[0]}),
    }
    # Warm, local single-record inference only: excludes Kafka, DB and Jev.
    inference_times = []
    for index in groups[2][:1001]:
        before = time.perf_counter()
        model.predict_proba([[features[index][name] for name in FEATURE_NAMES]], thread_count=1)
        inference_times.append((time.perf_counter() - before) * 1000)
    report["local_inference_ms"] = {
        f"p{percentile}": float(np.percentile(inference_times[1:], percentile))
        for percentile in (50, 95, 99)
    }
    source_file = Path(csv_path).with_suffix(".source.json")
    if source_file.exists():
        report["dataset"] = json.loads(source_file.read_text())
    output.mkdir(parents=True)
    model.save_model(str(output / "model.cbm"))
    report["model_sha256"] = hashlib.sha256((output / "model.cbm").read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(train(args.csv, args.output, args.min_precision, args.iterations), indent=2))


if __name__ == "__main__":
    main()
