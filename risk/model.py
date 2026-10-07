"""Load a versioned CatBoost bundle without pickle deserialization."""

import hashlib
import json
import math
from pathlib import Path

from risk.features import FEATURE_NAMES, FEATURE_VERSION


class FraudModel:
    def __init__(self, directory):
        from catboost import CatBoostClassifier

        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        if self.metadata["feature_version"] != FEATURE_VERSION:
            raise ValueError("Model feature version does not match this worker")
        if self.metadata["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model feature order does not match this worker")
        model_file = directory / "model.cbm"
        checksum = hashlib.sha256(model_file.read_bytes()).hexdigest()
        if checksum != self.metadata["model_sha256"]:
            raise ValueError("Model checksum does not match metadata")
        self.threshold = float(self.metadata["review_threshold"])
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Invalid review threshold")
        self.model = CatBoostClassifier()
        self.model.load_model(str(model_file))
        if self.model.feature_names_ != FEATURE_NAMES:
            raise ValueError("Saved model has unexpected features")

    def assess(self, features):
        if features["currency"] not in self.metadata["supported_currencies"]:
            return {"status": "unavailable", "error": "unsupported_currency"}
        unfamiliar_category = features["merchant_category"] not in self.metadata["merchant_categories"]
        score = float(self.model.predict_proba(
            [[features[name] for name in FEATURE_NAMES]], thread_count=1,
        )[0][1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Model returned an invalid fraud score")
        return {
            "status": "ok", "model_version": self.metadata["model_sha256"],
            "fraud_score": score, "review_threshold": self.threshold,
            "suggested_action": "review" if unfamiliar_category or score >= self.threshold else "allow",
            "unfamiliar_category": unfamiliar_category,
        }
