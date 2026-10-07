# Text fraud comparison

This experiment asks whether Jev adds value when the input contains actual scam
language. It is a **message classification experiment**, not a credit-card
authorization benchmark. It does not change the payment service, its trained
transaction model, or its shadow-only decision policy.

## Dataset selection and attribution

- Sandhya Mishra and Devpriya Soni, [SMS PHISHING DATASET FOR MACHINE LEARNING AND
  PATTERN RECOGNITION, version 1](https://data.mendeley.com/datasets/f45bkkt8pr/1),
  DOI 10.17632/f45bkkt8pr.1, CC BY 4.0. The source contains 5,971 messages labeled
  ham, spam, or smishing. We predict smishing; both ordinary spam and ham are
  negative. Only raw message text is used, not the supplied extracted flags.
- Al Rafi Ahmed and Gazi Faizul Islam, [Financial scams detection dataset,
  version 1](https://data.mendeley.com/datasets/znsk27yk3h/1),
  DOI 10.17632/znsk27yk3h.1, CC BY 4.0. The source contains 523 manually labeled
  English/Bangla scam and ham messages. The authors describe collection from
  real communications with consent and anonymization. This entire corpus is
  reserved for external evaluation, subject to duplicate removal.

The label definitions differ: financial scams are broader than SMS phishing,
and the external corpus has a much higher positive fraction. Its results are a
transfer check, not a second estimate of the same deployment distribution.
Neither corpus supplies usable transaction timestamps, payment outcomes, or
paired transaction features. Neither supports a chronological card-fraud test.
Public text may also have appeared in Jev's pretraining; that cannot be ruled out.

Two transaction-plus-text candidates were rejected for this experiment:

- [Phoenix21/mock_fraud-detection-dataset](https://huggingface.co/datasets/Phoenix21/mock_fraud-detection-dataset):
  the downloaded 34,767-row synthetic CSV has bracketed risk annotations on 374
  fraud records and zero legitimate records. These annotations may encode the
  intended label rather than information independently available at checkout.
- [AI4Risk/MS-FFSD](https://github.com/AI4Risk/MS-FFSD): generated entity context
  needs temporal and label-isolation auditing. Its published
  [merchant semantic adjustment code](https://github.com/AI4Risk/MS-FFSD/blob/main/consistency_optimization/code/apply_merchant_fraud_semantics.py)
  explicitly uses transaction fraud labels to alter merchant categories. Using
  the final enriched dataset directly would not establish leakage-free,
  real-time improvement.

## Frozen protocol

Code: [risk/text_compare.py](../risk/text_compare.py). Local plan and model files
are in `artifacts/text-comparison-v1/`; raw downloads in `data/text-candidates/`
are ignored by Git. Source CSV SHA-256 hashes are included in the report.

1. Normalize case, Unicode, URLs, email addresses, numbers and punctuation for
   duplicate detection only. Deduplicate matching templates within each source
   and exclude templates with contradictory labels. Models still see raw text.
2. Group near duplicates by connected components of character TF-IDF cosine
   similarity at least 0.90. This grouping uses text only, across both sources;
   its vocabulary is not passed to a classifier. Exclude six external messages
   whose groups overlap the SMS source. Similarity grouping reduces leakage but
   does not guarantee that all paraphrased templates are found.
3. Use seed 20260930 and ten stratified group folds for the SMS corpus: six folds
   train the local classifiers, one fits the hybrid, one calibrates thresholds,
   and two are untouched test folds. Every template group stays in one phase.
   There is no chronological claim and no class resampling or weighting.
4. Train native-text CatBoost (400 iterations, depth 6, learning rate 0.05) on
   3,396 messages. Add a character TF-IDF logistic regression control trained on
   exactly the same messages. This is a new text model; the existing transaction
   CatBoost model cannot consume this dataset.
5. Query Jev `jev-1.13.0` with a fixed prompt and three Noul questions: overall
   scam likelihood, suspicious credential requests, and deceptive payment
   requests. The API receives only the message and fixed instructions, never
   labels, split identifiers or local-model scores. Jev-only uses the first
   answer. This does not fine-tune Jev's weights.
6. Fit an L2 logistic hybrid on the logits of CatBoost and all three Jev scores,
   using only the 567 hybrid-fit messages. Select each method's threshold on
   567 different calibration messages to maximize recall subject to at most 1%
   false positives. Freeze thresholds before both test evaluations. A calibration
   false-positive budget is not a guarantee on later data.
7. Evaluate on 1,132 internal test messages (82 smishing) and 510 external
   messages (306 scams). Report precision, recall, false positives, average
   precision and raw confusion counts. Paired 500-replicate bootstrap intervals
   resample message groups, conditional on the fitted models; they do not account
   for retraining uncertainty or unknown data collection bias.

All models share the same available evaluation records. API failures are counted
and reserved for review; common-available metrics exclude failures. The client
has an 800 ms total deadline, no retries, six concurrent requests and a 20 request
per second scheduling cap. Cached requests are not repeated. Jev timing covers
the external request; local CatBoost timing covers warmed single-message
inference. These are not end-to-end payment latency measurements.

To reproduce after downloading and extracting the two source CSV files:

```sh
python -m risk.text_compare prepare
python -m risk.text_compare run
python -m pytest tests/unit/test_text_comparison.py -q
```

`prepare` refuses to overwrite the frozen plan. `run` uses `TYPESAFE_API_KEY` from
`.env` and caches every API result. Preserve the original artifacts for audit.

## Recorded results: September 30, 2026

[Full machine-readable report](text-model-comparison.json).

The tables compare the same messages successfully scored by Jev. One message
per test corpus was unavailable: one smishing message internally and one ham
message externally. The report also includes all-record local-model metrics and
the operational counts obtained by routing unavailable Jev decisions to review.

### Grouped SMS phishing test: 1,131 available messages, 81 positive

| Method | Precision | Recall | False-positive rate | Average precision | True positives / false positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost text | 80.82% | 72.84% | 1.33% | 0.8653 | 59 / 14 |
| Jev only | 88.41% | 75.31% | 0.76% | 0.8828 | 61 / 8 |
| CatBoost + Jev hybrid | 88.46% | 85.19% | 0.86% | 0.9310 | 69 / 9 |
| Character TF-IDF logistic control | 80.00% | 88.89% | 1.71% | 0.9149 | 72 / 18 |

The hybrid detected ten more phishing messages than CatBoost while producing
five fewer false positives. Its average-precision increase was 0.0657, with a
paired group bootstrap 95% interval of [0.0139, 0.1318]. The TF-IDF control caught
three more phishing messages than the hybrid but produced twice as many false
positives at its independently calibrated threshold. No claim is made that the
hybrid outperforms every possible local text model or tuning configuration.

### External financial-scam test: 509 available messages, 306 positive

All thresholds and model weights are unchanged from the SMS experiment.

| Method | Precision | Recall | False-positive rate | Average precision | True positives / false positives |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost text | 84.96% | 36.93% | 9.85% | 0.7665 | 113 / 20 |
| Jev only | 97.65% | 67.97% | 2.46% | 0.9771 | 208 / 5 |
| CatBoost + Jev hybrid | 97.71% | 69.61% | 2.46% | 0.9689 | 213 / 5 |
| Character TF-IDF logistic control | 83.43% | 47.71% | 14.29% | 0.7886 | 146 / 29 |

The hybrid detected 100 more scams than CatBoost with 15 fewer false positives.
Its average-precision increase was 0.2025, with a paired group bootstrap 95%
interval of [0.1611, 0.2503]. Jev alone had the highest external average precision;
the hybrid was not uniformly superior to Jev. High external precision partly
reflects the high scam prevalence and cannot be transferred to payment traffic.
Every method exceeded its 1% calibration false-positive budget on this corpus,
illustrating the effect of distribution shift.

### Availability and latency

Of 2,776 live Jev requests, 2,772 succeeded and four were unavailable. Successful
responses reported 1,269,955 input tokens. Jev latency was p50 161.9 ms, p95
267.2 ms and p99 428.0 ms, including unavailable attempts in the timing population.
Local CatBoost text inference was p95 0.159 ms. Hybrid use incurs the external
call plus local computation; end-to-end hybrid/payment latency was not measured.

The result supports testing Jev as a **textual scam signal** when payment-related
messages or other relevant text actually exist before a decision. It does not
reverse the earlier numerical transaction benchmark or establish safe automatic
payment rejection. Real paired transaction/text outcomes and a prospective
shadow evaluation are still needed before making that inference.
