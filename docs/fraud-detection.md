# Fraud assessment

The risk worker subscribes to `payment-events` using a separate consumer group.
It persists assessments in `risk_assessments` and never changes balances or
payment status. There is no enforcement switch: payment blocking requires a
separate, evaluated decision gate before debiting.

```text
POST /payments → payment + feature snapshot in outbox → Kafka
                                                       ├─ payment processor → debit
                                                       └─ risk shadow worker → assessment
```

## Install and migrate

Use Python 3.12, the existing Kafka/PostgreSQL services, and the repository root:

```sh
uv venv --python 3.12
uv pip install -r requirements-risk.txt -r requirements-dev.txt
source .venv/bin/activate
psql -h localhost -U admin -d payments -f db/migrations/002_risk_shadow.sql
```

For a fresh database, `db/schema.sql` already includes the risk tables. Apply the
standalone migration to an existing database before starting the updated API;
do not re-run the entire existing schema script, which resets metrics.

## Dataset and training

This implementation uses the public [Kaggle credit-card transactions dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
generated with [Sparkov](https://github.com/namebrandon/Sparkov_Data_Generation).
These are synthetic transactions, not real cardholder outcomes. Results establish
a reproducible demo baseline, not production fraud effectiveness. The credit-card
default dataset is not used: repayment default and transaction fraud are different targets.

The downloaded ZIP and trained model stay in ignored `data/` and `artifacts/`
directories. Retain the dataset's attribution and license when redistributing it.

```sh
mkdir -p data/raw
curl -fL https://www.kaggle.com/api/v1/datasets/download/kartik2112/fraud-detection \
  -o data/raw/credit-card-fraud.zip
python -m risk.prepare_sparkov --archive data/raw/credit-card-fraud.zip \
  --output data/credit-card-history.csv
python -m risk.train --csv data/credit-card-history.csv --output artifacts/risk-v1
```

The preparer combines the archive's two CSVs, hashes synthetic card identifiers
into user identifiers, converts amounts to cents, and treats simulated calendar
times consistently as UTC. It excludes names, card numbers, demographic fields,
addresses and locations from the normalized dataset. User IDs only group history;
they are never model inputs. The model uses amount, hour, weekday, currency,
merchant category, and the count/sum of earlier attempts in the last hour.
The category is merchant metadata available before payment; it is not the fraud label.

Training creates new chronological splits at 60% and 80% of distinct timestamps,
keeping identical timestamps together. It does not reuse the archive's original
train/test split. The validation set chooses the review threshold for maximum
recall while meeting `--min-precision` (default 0.9). The untouched later test set
reports average precision, precision, recall, false-positive rate and confusion
counts. There is no accuracy-based success claim or automatic model promotion.
Training fails rather than exporting a bundle if no validation threshold meets the
requested precision. Model scores have not been probability-calibrated.

The bundle contains `model.cbm` and `metadata.json`, with checksums, feature
version/order, decision threshold, supported categories/currencies and evaluation
results. Use a new output directory for each run. Replaying a duplicate Kafka
event preserves its first assessment; use a separate evaluation workflow when
comparing a new model rather than overwriting that audit record.

For your own CSV, required columns are `payment_id`, `user_id`, `timestamp`,
`amount`, `is_fraud`; optional columns are `currency`, `merchant_category`.
`timestamp` is Unix seconds or timezone-aware ISO 8601, `amount` is a positive
integer in minor units, and `is_fraud` is a confirmed 0/1 outcome. Include both
legitimate and fraudulent transactions with their natural prevalence. Only use
labels known by the training cutoff, allow fraud outcomes time to mature, and
include complete prior transaction history. Missing/selected history changes the
velocity features. The public demo has no label-arrival timestamps, so it cannot
validate delayed-label operational behavior.

## Run the shadow worker

```sh
python -m consumer.risk_consumer --model-dir artifacts/risk-v1
```

Keep the API, outbox relay and existing payment consumer running as usual. The
default risk group is `payment-risk-shadow`; it must not equal the payment
processor's group. The worker reads from the beginning for a new group. Legacy
events without feature snapshots receive an explicit unavailable assessment.

An enriched payment request looks like:

```json
{
  "user_id": "user_123",
  "amount": 4999,
  "currency": "USD",
  "merchant_category": "shopping_net",
  "merchant": "Example Store",
  "description": "Online purchase of household supplies"
}
```

The payment ledger supports USD only, with amounts in cents. Explicit currencies
other than USD are rejected before payment creation; the payment consumer also
rejects events declaring other currencies before changing balances. Supporting
multiple currencies requires separate balances per currency.

Existing requests with only user/amount still work under the same USD-cent
convention. They retain unknown currency metadata; the USD demo model will mark
those assessments unavailable. Historical model replay can contain other
currencies, but the USD demo model will not score them. Unfamiliar merchant
categories are flagged and suggest review. Inputs
currently come from the client in this demo. Before enforcement, obtain merchant
metadata from a trusted payment source rather than letting a payer choose it.

The API computes velocity from prior attempts for the same user and currency,
serializing concurrent submissions per user with a PostgreSQL transaction lock.
The feature snapshot and UTC timestamp are committed atomically with the payment
and outbox record, so scoring later cannot accidentally use future transactions.
Amounts and counts include failed attempts, and exactly one-hour-old records are
included. New payment timestamps are stored as UTC. Older rows retain unknown
currency and do not enter the USD demo model's history. Normalize any historical
local timestamps before assigning a currency to old records.

Read an assessment through `GET /payments/{payment_id}/risk`. `pending` means no
assessment has been stored; `assessed` means a result was persisted, including
unavailable/error results. Inspect each component's `status` before using scores.
Payment status and assessment status are independent.

Assessment persistence precedes the Kafka offset commit. Duplicate event IDs do
not call the models again after a result is stored. A database failure stops the
worker before any later offset is committed; restarting retries that delivery.
A crash between inference and persistence can repeat inference. Model failures
and malformed events are stored explicitly rather than converted into allow
decisions. No raw exception text or event body is saved for malformed messages.

## Optional Jev

CatBoost alone is the default transaction workflow. Jev is not required to train
the baseline or run its shadow worker. The recorded numerical transaction pilot
found no improvement from adding Jev, while the external call added latency and
an API dependency. Keep `JEV_ENABLED=false` for that workflow. Jev remains an
opt-in experiment for merchant classification and scam language; the separate
[text comparison](text-fraud-comparison.md) supports further testing on messages,
without establishing improvement on card transactions.

Set `JEV_ENABLED=true` and `TYPESAFE_API_KEY` in `.env`, then restart the worker.
Without explicit enablement, no requests are sent to TypeSafe. The version is
pinned with `JEV_MODEL=jev-1.13.0` by default. Jev receives the numerical snapshot,
merchant and description; user/card identifiers, labels and supplied category
are excluded. Avoid putting personal data in free-text fields.

Jev independently classifies the purchase and returns a suspicious-text signal.
That signal is not a calibrated fraud probability and does not alter the trained
baseline's decision. Calls have a total asynchronous deadline (default 0.8s), no
automatic retries, and validated response shapes/probabilities. Timeouts, rate
limits and malformed responses produce unavailable results. This worker records
text signals independently of the offline fraud comparison described below.

[TypeSafe customization](https://docs.typesafe.ai/models#customizing-jev) does not
support fine-tuning Jev on this dataset. Supervised training updates CatBoost and
the comparison's local score-combining model, not Jev's weights.

## Compare Jev, CatBoost and a hybrid

The offline comparison uses an actual fraud question, separate from the shadow
worker's suspicious-text question. Each Jev request sees exactly the same seven
features as CatBoost. No customer identifiers, transaction identifiers, outcome
labels, or CatBoost predictions are sent. Jev uses zero-shot inference with a
pinned prompt and model; the CatBoost model was trained on the earlier history.

```sh
source .venv/bin/activate
python -m risk.compare prepare
python -m risk.compare run
```

`prepare` verifies the dataset checksum against the trained model, constructs
features using complete preceding history, and creates the fixed seeded plan at
`artifacts/comparison-plan.json`. `run` makes up to 10,000 paid Jev requests using
`TYPESAFE_API_KEY` from `.env`. It defaults to 20 requests/second, six concurrent
requests and an 800 ms total deadline, with no retries. It reuses HTTP connections
and writes each response, including failures, to a resumable SQLite cache. A
configuration or plan mismatch prevents accidental reuse of incompatible results.

The comparison has three strictly chronological phases, all after CatBoost's
training period:

| Phase | Sample | Purpose |
| --- | ---: | --- |
| Earlier half of validation | 2,000, including 200 fraud | Fit the hybrid |
| Later half of validation | 3,000, including 200 fraud | Select thresholds for all three methods |
| Existing held-out test period | 5,000, including 200 fraud | Evaluate once without tuning |

The hybrid is regularized logistic regression over the clipped log-odds of the
CatBoost and Jev scores. It learns coefficients on the first phase only. Every
method independently chooses its threshold to maximize calibration recall while
keeping calibration false-positive rate at or below 0.1%. If no finite threshold
qualifies, that method makes no automatic fraud flags. The false-positive budget
is a comparison policy, not a production decision recommendation. It differs from
the original CatBoost benchmark's 90% minimum validation precision policy.

Fraud is intentionally oversampled to obtain enough positive cases for a small
pilot. Inverse inclusion weights restore each period's original prevalence when
fitting the hybrid and reporting precision, average precision and estimated
population confusion counts. Raw sample confusion counts are also reported. Do
not report the enriched sample's unweighted precision as a deployment estimate.

All three primary metrics use the identical subset with valid Jev responses.
Coverage and error counts are reported separately. The additional all-test routing
metrics count unavailable Jev/hybrid decisions as review cases; CatBoost can still
score those rows locally. This distinguishes model discrimination from timeout
effects rather than treating missing scores as legitimate payments.

Outputs under `artifacts/comparison-v1/` include `report.json`, the hybrid's
`hybrid.json` coefficients and thresholds, `predictions.json`, `run-config.json`,
and `jev-cache.sqlite`. Together these identify the input plan, CatBoost checksum,
Jev model and prompt hash. Re-running the same command reuses cached responses
without paying for them again. To change prompt/model/deadline, create a new output
directory. Do not tune on the reported test outcomes; use a new held-out period
for a subsequent experiment.

The report includes precision, recall, false-positive rate, average precision,
ROC AUC, p50/p95/p99 component latency, known API token usage and a price estimate.
Timed-out requests may incur charges without returning usage. Dashboard billing
is authoritative. Hybrid latency is the sum of separately measured sequential
components, with combiner overhead reported separately; none of these times
includes Kafka or database latency.

This is a synthetic-data pilot with only 4,800 sampled legitimate test records.
Very small false-positive rates and precision therefore have substantial sampling
uncertainty. Paired bootstrap intervals describe hybrid-minus-CatBoost differences
conditional on the fixed fitted models, but do not account for customer correlation
or retraining uncertainty. The comparison does not enable any payment blocking.

### Recorded pilot: September 30, 2026

All 10,000 Jev requests succeeded, consuming 4,908,970 reported input tokens
(approximately $0.206 at the published input-token rate). The run took 510 seconds
with the configured request-rate cap. [Full machine-readable results](model-comparison.json)
include the model/prompt fingerprints, coefficients, thresholds, phase populations,
raw confusion counts, coverage, and bootstrap intervals.

The test sample contains 200 fraudulent and 4,800 legitimate transactions.
Precision and average precision below are weighted to the original held-out
period's fraud prevalence (1,368 of 371,552 transactions).

| Method | Weighted precision | Fraud recall | False-positive rate | Average precision | p95 component latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 71.87% | 72.0% | 0.1042% | 0.8020 | 0.090 ms |
| Jev, zero-shot | 0% | 0% | 0.0208% | 0.0293 | 219.19 ms |
| Hybrid | 71.72% | 71.5% | 0.1042% | 0.8021 | 219.23 ms plus combiner |

CatBoost detected 144 of 200 fraud cases with 5 false positives. The hybrid
detected 143 with the same 5 false positives. Jev flagged one legitimate payment
and none of the fraud cases at its calibration-selected threshold. This is a
result for this zero-shot prompt and limited numerical context, not a claim that
Jev can never detect fraud. Jev's threshold-independent average precision was also
substantially below CatBoost's on this sample.

The hybrid-minus-CatBoost average-precision difference was 0.00014, with a paired
bootstrap 95% interval of [-0.00203, 0.00234]. That does not establish an improvement.
The hybrid's local combining step averaged 0.038 ms per record; the external call
dominates its latency. CatBoost remains the preferred baseline for this experiment.

The selected thresholds are CatBoost 0.33977, Jev 0.29 and hybrid 0.56529. The 0.1%
calibration false-positive budget is not guaranteed on later test data. These
results differ from the original 86.2% CatBoost precision because this pilot uses
a smaller sampled test set and a different threshold-selection policy. The
original model artifact and its original threshold were not changed.

## Latency and verification

Each stored assessment records baseline/Jev latency, total assessment time, and
event age through assessment completion. Event age includes queue delay, but does
not include the final database commit or any payment-network latency. The training
report's p50/p95/p99 measurements cover warm single-row local inference only.
The relay still has its original one-second idle polling interval. Shadow work
does not delay debiting; an enforced decision path would have a different budget.

```sh
python -m pytest tests/unit -q
# Against an isolated database initialized with schema.sql:
python -m pytest tests/integration/test_risk_shadow.py \
  tests/integration/test_consumer_idempotency.py -q
ruff check .
```

The tests cover temporal feature boundaries, chronology, model bundle integrity,
threshold selection, Jev deadline/error handling, duplicate assessments, offset
commit failure behavior, and PostgreSQL payment isolation. Comparison tests also
verify that test-label changes cannot alter hybrid coefficients or thresholds,
that sampling weights restore prevalence, and that API failures route to review.
The recorded pilot measures live Jev requests on synthetic data; Kafka end-to-end
latency and performance on real transactions remain unmeasured.
