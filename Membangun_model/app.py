"""
Wine Quality Prediction — Flask Serving API
=============================================
Endpoints:
  POST /predict          → Predict wine quality (JSON input)
  GET  /health           → Health check
  GET  /metrics          → Prometheus metrics
  GET  /info             → Model metadata

Integrates:
  - Prometheus metrics (request count, latency, prediction distribution)
  - Structured logging
"""

import os
import json
import time
import logging
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, request, jsonify

# ── Prometheus metrics ────────────────────────────────────────────────
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY,
)
from flask import Response

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("wine-quality-api")

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "model"))

# ── Flask app ─────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Prometheus metrics definition ─────────────────────────────────────
REQUEST_COUNT = Counter(
    "wine_prediction_requests_total",
    "Total number of prediction requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "wine_prediction_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_GAUGE = Gauge(
    "wine_prediction_result",
    "Latest prediction result (0=low, 1=high)",
)
PREDICTION_DISTRIBUTION = Counter(
    "wine_prediction_class_total",
    "Prediction class distribution",
    ["predicted_class"],
)
MODEL_LOAD_TIME = Gauge(
    "wine_model_load_timestamp",
    "Timestamp when the model was loaded",
)
PREDICTION_PROBABILITY = Histogram(
    "wine_prediction_probability",
    "Prediction probability distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Load model artifacts ──────────────────────────────────────────────
model        = None
scaler       = None
feature_cols = None
metadata     = None


def load_model_artifacts():
    """Load model, scaler, feature columns, and metadata."""
    global model, scaler, feature_cols, metadata

    model_path   = os.path.join(MODEL_DIR, "model.joblib")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_cols.json")
    meta_path    = os.path.join(MODEL_DIR, "metadata.json")

    if not os.path.exists(model_path):
        log.error(f"Model not found at {model_path}")
        raise FileNotFoundError(f"Model not found at {model_path}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"model_name": "unknown", "timestamp": "unknown"}

    MODEL_LOAD_TIME.set(time.time())
    log.info(f"Model loaded: {metadata.get('model_name', 'unknown')}")
    log.info(f"Features: {feature_cols}")


# Load on startup
try:
    load_model_artifacts()
except FileNotFoundError:
    log.warning("Model artifacts not found. Endpoints will return errors.")


# ── Helper ────────────────────────────────────────────────────────────
def validate_input(data: dict) -> tuple:
    """Validate and extract feature values from input JSON."""
    missing = [f for f in feature_cols if f not in data]
    if missing:
        return None, f"Missing features: {missing}"

    try:
        values = [float(data[f]) for f in feature_cols]
    except (ValueError, TypeError) as e:
        return None, f"Invalid feature value: {e}"

    return np.array(values).reshape(1, -1), None


# ══════════════════════════════════════════════════════════════════════ #
#  ROUTES                                                                #
# ══════════════════════════════════════════════════════════════════════ #

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    status = "healthy" if model is not None else "unhealthy"
    code   = 200 if model is not None else 503
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }), code


@app.route("/info", methods=["GET"])
def info():
    """Model metadata endpoint."""
    return jsonify({
        "model_name": metadata.get("model_name", "unknown") if metadata else "unknown",
        "features": feature_cols,
        "metadata": metadata,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict wine quality.

    Input JSON:
    {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "wine_type": 0
    }
    """
    start_time = time.time()

    if model is None:
        REQUEST_COUNT.labels("POST", "/predict", "503").inc()
        return jsonify({"error": "Model not loaded"}), 503

    # Parse input
    data = request.get_json(force=True)
    if not data:
        REQUEST_COUNT.labels("POST", "/predict", "400").inc()
        return jsonify({"error": "Empty request body"}), 400

    # Support both single and batch prediction
    if isinstance(data, list):
        results = []
        for item in data:
            result = _predict_single(item)
            if "error" in result:
                REQUEST_COUNT.labels("POST", "/predict", "400").inc()
                return jsonify(result), 400
            results.append(result)
        response = {"predictions": results}
    else:
        result = _predict_single(data)
        if "error" in result:
            REQUEST_COUNT.labels("POST", "/predict", "400").inc()
            return jsonify(result), 400
        response = result

    duration = time.time() - start_time
    REQUEST_COUNT.labels("POST", "/predict", "200").inc()
    REQUEST_LATENCY.labels("POST", "/predict").observe(duration)

    log.info(f"Prediction served in {duration:.4f}s")
    return jsonify(response)


def _predict_single(data: dict) -> dict:
    """Process a single prediction request."""
    features, error = validate_input(data)
    if error:
        return {"error": error}

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    # Update Prometheus metrics
    PREDICTION_GAUGE.set(prediction)
    PREDICTION_DISTRIBUTION.labels(str(prediction)).inc()
    PREDICTION_PROBABILITY.observe(probability)

    label = "High Quality" if prediction == 1 else "Low Quality"

    return {
        "prediction": prediction,
        "label": label,
        "probability_high_quality": round(probability, 4),
        "timestamp": datetime.now().isoformat(),
    }


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


# ── Error handlers ────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    log.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500


# ══════════════════════════════════════════════════════════════════════ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting Wine Quality Prediction API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
