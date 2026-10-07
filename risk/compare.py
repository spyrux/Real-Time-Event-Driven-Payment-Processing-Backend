"""Reproducible Jev/CatBoost/hybrid pilot: prepare a plan, then run paid requests."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from risk.features import FEATURE_VERSION, historical_features
from risk.jev_fraud import (
    FRAUD_INSTRUCTIONS,
    PROMPT_VERSION,
    JevFraudClient,
    request_hash,
)
from risk.model import FraudModel
from risk.train import load_history

SEED = 20260930
PHASE_SIZES = {"hybrid_fit": 2000, "calibration": 3000, "test": 5000}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stratified_sample(rows, indices, size, fraud_size, seed):
    rng = random.Random(seed)
    selected, counts = [], {}
    for label, target in ((0, size - fraud_size), (1, fraud_size)):
        population = [i for i in indices if rows[i]["is_fraud"] == label]
        if len(population) < target or target <= 0:
            raise ValueError("Insufficient records for requested stratified sample")
        counts[str(label)] = {"population": len(population), "sample": target}
        selected.extend((i, len(population) / target) for i in rng.sample(population, target))
    return sorted(selected), counts


def prepare(csv_path, model_dir, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new plan path")
    model = FraudModel(model_dir)
    dataset_hash = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    if dataset_hash != model.metadata["dataset_sha256"]:
        raise ValueError("Comparison dataset must match the CatBoost training dataset")
    rows = load_history(csv_path)
    features = historical_features(rows)
    bounds = model.metadata["splits"]
    validation = [i for i, row in enumerate(rows)
                  if bounds["validation"]["start"] <= row["timestamp"] <= bounds["validation"]["end"]]
    middle = rows[validation[len(validation) // 2]]["timestamp"]
    groups = {
        "hybrid_fit": [i for i in validation if rows[i]["timestamp"] < middle],
        "calibration": [i for i in validation if rows[i]["timestamp"] >= middle],
        "test": [i for i, row in enumerate(rows)
                 if bounds["test"]["start"] <= row["timestamp"] <= bounds["test"]["end"]],
    }
    plan = {
        "version": "comparison-v1", "feature_version": FEATURE_VERSION, "seed": SEED,
        "dataset_sha256": dataset_hash, "catboost_sha256": model.metadata["model_sha256"],
        "dataset": model.metadata.get("dataset"), "phases": {}, "records": [],
    }
    # Warm CatBoost before recording per-row inference latency.
    model.assess(features[groups["hybrid_fit"][0]])
    for number, (phase, indices) in enumerate(groups.items()):
        sample, counts = stratified_sample(rows, indices, PHASE_SIZES[phase], 200, SEED + number)
        plan["phases"][phase] = {
            "classes": counts, "start": rows[indices[0]]["timestamp"],
            "end": rows[indices[-1]]["timestamp"],
        }
        for index, weight in sample:
            before = time.perf_counter()
            result = model.assess(features[index])
            elapsed = (time.perf_counter() - before) * 1000
            if result["status"] != "ok":
                raise ValueError("CatBoost cannot score a sampled record")
            plan["records"].append({
                "payment_id": rows[index]["payment_id"], "phase": phase,
                "timestamp": rows[index]["timestamp"], "label": rows[index]["is_fraud"],
                "weight": weight, "features": features[index],
                "catboost_score": result["fraud_score"], "catboost_ms": elapsed,
            })
    plan["plan_sha256"] = digest(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan) + "\n")
    return {"path": str(output), "rows": len(plan["records"]), "phases": plan["phases"]}


def logits(scores):
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def hybrid_inputs(records):
    return logits([[r["catboost_score"], r["jev"]["fraud_score"]] for r in records])


def choose_threshold(labels, scores, weights, max_fpr):
    fpr, recall, thresholds = roc_curve(labels, scores, sample_weight=weights, drop_intermediate=False)
    allowed = [i for i in range(len(thresholds)) if fpr[i] <= max_fpr]
    best = max(allowed, key=lambda i: (recall[i], -fpr[i], thresholds[i]))
    return float(thresholds[best]) if math.isfinite(thresholds[best]) else None


def weighted_metrics(labels, scores, weights, threshold):
    labels, scores, weights = np.asarray(labels), np.asarray(scores), np.asarray(weights)
    flags = np.zeros(len(labels), dtype=bool) if threshold is None else scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, flags, sample_weight=weights, labels=[0, 1]).ravel()
    raw = confusion_matrix(labels, flags, labels=[0, 1]).ravel()
    return {
        "rows": len(labels), "fraud_cases": int(sum(labels)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)), "false_positive_rate": float(fp / (fp + tn)),
        "average_precision": float(average_precision_score(labels, scores, sample_weight=weights)),
        "roc_auc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "sample_counts": dict(zip(("tn", "fp", "fn", "tp"), map(int, raw))),
        "estimated_population_counts": dict(zip(("tn", "fp", "fn", "tp"), map(float, (tn, fp, fn, tp)))),
    }


def percentiles(values):
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)} if values else {}


def paired_bootstrap(records, scores_by_name, thresholds, repeats=300):
    """Stratify within the sampled test labels, preserving population weights."""
    rng = np.random.default_rng(SEED)
    labels = np.array([r["label"] for r in records])
    weights = np.array([r["weight"] for r in records])
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    deltas = []
    for _ in range(repeats):
        selected = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values = []
        for name in ("catboost", "hybrid"):
            metric = weighted_metrics(labels[selected], np.asarray(scores_by_name[name])[selected],
                                      weights[selected], thresholds[name])
            values.append([metric["average_precision"], metric["recall"], metric["precision"]])
        deltas.append(np.asarray(values[1]) - values[0])
    return {name: {"lower": float(np.percentile(np.asarray(deltas)[:, i], 2.5)),
                   "upper": float(np.percentile(np.asarray(deltas)[:, i], 97.5))}
            for i, name in enumerate(("average_precision", "recall", "precision"))}


def evaluate(records, max_fpr):
    if len({r["payment_id"] for r in records}) != len(records):
        raise ValueError("Comparison records must have unique payment IDs")
    times = {phase: [r["timestamp"] for r in records if r["phase"] == phase] for phase in PHASE_SIZES}
    if (not all(times.values()) or max(times["hybrid_fit"]) >= min(times["calibration"])
            or max(times["calibration"]) >= min(times["test"])):
        raise ValueError("Hybrid fit, calibration and test must be strictly chronological")
    available = {phase: [r for r in records if r["phase"] == phase and r["jev"]["status"] == "ok"]
                 for phase in PHASE_SIZES}
    for phase, group in available.items():
        if {r["label"] for r in group} != {0, 1}:
            raise ValueError(f"{phase} needs successful Jev calls for both classes")
    fit = available["hybrid_fit"]
    weights = np.array([r["weight"] for r in fit])
    stacker = LogisticRegression(C=10, max_iter=1000, random_state=SEED)
    stacker.fit(hybrid_inputs(fit), [r["label"] for r in fit], sample_weight=weights / weights.mean())
    artifact = {
        "type": "logistic_score_stacker", "features": ["logit_catboost", "logit_jev"],
        "clip": [1e-5, 1 - 1e-5], "C": 10,
        "coefficients": stacker.coef_[0].tolist(), "intercept": float(stacker.intercept_[0]),
    }
    thresholds, methods = {}, {}
    calibration, test = available["calibration"], available["test"]

    def scores(group):
        return {"catboost": [r["catboost_score"] for r in group],
                "jev": [r["jev"]["fraud_score"] for r in group],
                "hybrid": stacker.predict_proba(hybrid_inputs(group))[:, 1]}

    calibration_scores, test_scores = scores(calibration), scores(test)
    for name in ("catboost", "jev", "hybrid"):
        thresholds[name] = choose_threshold(
            [r["label"] for r in calibration], calibration_scores[name],
            [r["weight"] for r in calibration], max_fpr,
        )
        methods[name] = {"threshold": thresholds[name]}
        for phase, group, predictions in (("calibration", calibration, calibration_scores[name]),
                                          ("test", test, test_scores[name])):
            methods[name][phase] = weighted_metrics(
                [r["label"] for r in group], predictions, [r["weight"] for r in group], thresholds[name],
            )
    # Every method's main metrics use exactly the same successful rows. Report
    # missing rows explicitly, and operational routing separately on all test rows.
    all_test = [r for r in records if r["phase"] == "test"]
    for name, method in methods.items():
        flags, labels, weights = [], [], []
        successful_scores = iter(test_scores[name])
        for row in all_test:
            if row["jev"]["status"] == "ok":
                score = next(successful_scores)
                flag = thresholds[name] is not None and score >= thresholds[name]
            elif name == "catboost":
                flag = thresholds[name] is not None and row["catboost_score"] >= thresholds[name]
            else:
                flag = True  # Explicit human-review route on unavailable model response.
            flags.append(flag)
            labels.append(row["label"])
            weights.append(row["weight"])
        method["all_test_review_routing"] = weighted_metrics(labels, flags, weights, 0.5)
        method["all_test_review_routing"].pop("average_precision")
        method["all_test_review_routing"].pop("roc_auc")
    latency = {
        "catboost": percentiles([r["catboost_ms"] for r in all_test]),
        "jev": percentiles([r["jev"]["latency_ms"] for r in all_test]),
        "hybrid_sequential_components": percentiles([
            r["catboost_ms"] + r["jev"]["latency_ms"] for r in all_test
        ]),
    }
    before = time.perf_counter()
    for row in test[:1000]:
        stacker.predict_proba(hybrid_inputs([row]))
    latency["hybrid_combiner_mean_ms"] = (time.perf_counter() - before) * 1000 / min(1000, len(test))
    return {
        "methods": methods, "hybrid_model": {**artifact, "thresholds": thresholds},
        "latency_ms": latency,
        "hybrid_minus_catboost_paired_bootstrap_95pct": paired_bootstrap(test, test_scores, thresholds),
        "response_coverage": {
            phase: {"total": sum(r["phase"] == phase for r in records), "successful": len(group),
                    "failures": dict(Counter(r["jev"].get("error", "unknown") for r in records
                                             if r["phase"] == phase and r["jev"]["status"] != "ok"))}
            for phase, group in available.items()
        },
    }


async def collect(plan, cache_path, model, deadline, concurrency, requests_per_second):
    database = sqlite3.connect(cache_path)
    database.execute("CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, response TEXT NOT NULL)")
    semaphore = asyncio.Semaphore(concurrency)
    rate_lock = asyncio.Lock()
    next_slot = 0.0
    complete, fresh, cached = 0, 0, 0
    started = time.monotonic()
    config = {"model": model, "deadline": deadline, "prompt": PROMPT_VERSION}
    try:
        async with JevFraudClient(os.getenv("TYPESAFE_API_KEY"), model, deadline) as client:
            async def one(row):
                nonlocal complete, fresh, cached, next_slot
                key = digest({"payment_id": row["payment_id"], "config": config,
                              "request": request_hash(row["features"], model)})
                existing = database.execute("SELECT response FROM responses WHERE cache_key = ?", (key,)).fetchone()
                if existing:
                    result = json.loads(existing[0])
                    cached += 1
                else:
                    async with semaphore:
                        async with rate_lock:
                            wait = max(0, next_slot - time.monotonic())
                            if wait:
                                await asyncio.sleep(wait)
                            next_slot = time.monotonic() + 1 / requests_per_second
                        result = await client.assess(row["features"])
                        database.execute("INSERT INTO responses VALUES (?, ?)", (key, json.dumps(result)))
                        database.commit()
                        fresh += 1
                row["jev"] = result
                if result.get("error") in {"http_401", "http_403"}:
                    raise RuntimeError("Jev authentication failed; stopped comparison")
                complete += 1
                if complete % 250 == 0 or complete == len(plan["records"]):
                    errors = sum(r.get("jev", {}).get("status") == "unavailable" for r in plan["records"])
                    print(json.dumps({"completed": complete, "total": len(plan["records"]),
                                      "errors": errors, "elapsed_seconds": round(time.monotonic() - started)}), flush=True)

            tasks = [asyncio.create_task(one(row)) for row in plan["records"]]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    finally:
        database.close()
    return {"fresh_requests": fresh, "cached_responses": cached,
            "wall_seconds": time.monotonic() - started}


def run(args):
    if not 0 < args.max_fpr < 1:
        raise ValueError("max-fpr must be in (0, 1)")
    if not 1 <= args.concurrency <= 20 or not 0 < args.rps <= 30:
        raise ValueError("Use concurrency 1-20 and rps in (0, 30]")
    plan = json.loads(Path(args.plan).read_text())
    recorded_hash = plan.pop("plan_sha256")
    if digest(plan) != recorded_hash:
        raise ValueError("Plan checksum mismatch")
    if len(plan["records"]) > args.max_requests:
        raise ValueError("Plan exceeds --max-requests")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"plan_sha256": recorded_hash, "model": args.model, "deadline": args.deadline,
              "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(FRAUD_INSTRUCTIONS),
              "max_fpr": args.max_fpr, "concurrency": args.concurrency, "rps": args.rps}
    manifest = output / "run-config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Output belongs to another comparison configuration; use a new directory")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    execution = asyncio.run(collect(plan, output / "jev-cache.sqlite", args.model, args.deadline,
                                   args.concurrency, args.rps))
    report = evaluate(plan["records"], args.max_fpr)
    known_tokens = sum(r["jev"].get("input_tokens") or 0 for r in plan["records"])
    report.update({
        "created_at": datetime.now(timezone.utc).isoformat(), "configuration": config,
        "dataset": plan["dataset"], "dataset_sha256": plan["dataset_sha256"],
        "catboost_sha256": plan["catboost_sha256"], "phases": plan["phases"],
        "execution": execution, "known_input_tokens": known_tokens,
        "estimated_known_token_cost_usd": known_tokens / 1_000_000 * 0.042,
        "pricing_reference": "https://docs.typesafe.ai/models",
        "caveats": [
            "Synthetic-data pilot. No production enforcement enabled.",
            "Fraud-enriched sample; inverse inclusion weights restore each period's natural prevalence.",
            "Common-success metrics use identical rows for all methods; failures separately route to review.",
            "Jev is zero-shot on the same seven features; no evaluated labels or CatBoost scores are sent.",
            "Hybrid fits the earlier half of original validation; thresholds use the later half only.",
            "Test uses the previous benchmark's held-out period; no prompt or threshold tuning on test outcomes.",
            "All methods select maximum calibration recall under the same false-positive-rate budget.",
            "Precision/FPR estimates are uncertain with only 4,800 sampled legitimate test transactions.",
            "Bootstrap intervals resample transactions, not customers; correlation may make them optimistic.",
            "Latencies exclude queue/DB work. Hybrid combines separately measured sequential component times.",
            "Known-token cost excludes timed-out requests with unknown usage; dashboard billing is authoritative.",
        ],
    })
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "hybrid.json").write_text(json.dumps(report["hybrid_model"], indent=2) + "\n")
    (output / "predictions.json").write_text(json.dumps(plan["records"]) + "\n")
    print(json.dumps({"report": str(output / "report.json"), "methods": report["methods"]}, indent=2))


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("--csv", default="data/credit-card-history.csv")
    setup.add_argument("--model-dir", default="artifacts/risk-v1")
    setup.add_argument("--output", default="artifacts/comparison-plan.json")
    execute = commands.add_parser("run")
    execute.add_argument("--plan", default="artifacts/comparison-plan.json")
    execute.add_argument("--output", default="artifacts/comparison-v1")
    execute.add_argument("--model", default="jev-1.13.0")
    execute.add_argument("--deadline", type=float, default=0.8)
    execute.add_argument("--max-fpr", type=float, default=0.001)
    execute.add_argument("--concurrency", type=int, default=6)
    execute.add_argument("--rps", type=float, default=20)
    execute.add_argument("--max-requests", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.csv, args.model_dir, args.output), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
