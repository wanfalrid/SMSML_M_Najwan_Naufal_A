"""
Model utility functions for loading and inference.
"""

import os
import json
import logging
import numpy as np
import joblib

log = logging.getLogger(__name__)


class WineQualityModel:
    """Wrapper for the wine quality prediction model."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        self.load()

    def load(self):
        """Load all model artifacts."""
        self.model = joblib.load(os.path.join(self.model_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))

        with open(os.path.join(self.model_dir, "feature_cols.json")) as f:
            self.feature_cols = json.load(f)

        meta_path = os.path.join(self.model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        log.info(f"Model loaded from {self.model_dir}")

    def predict(self, input_data: dict) -> dict:
        """Make prediction from a dictionary of features."""
        features = np.array([float(input_data[f]) for f in self.feature_cols]).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        prediction = int(self.model.predict(features_scaled)[0])
        probability = float(self.model.predict_proba(features_scaled)[0][1])

        return {
            "prediction": prediction,
            "label": "High Quality" if prediction == 1 else "Low Quality",
            "probability_high_quality": round(probability, 4),
        }

    def predict_batch(self, batch: list) -> list:
        """Make predictions for a list of input dictionaries."""
        return [self.predict(item) for item in batch]
