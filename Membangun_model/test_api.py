"""
Pytest tests for the Wine Quality Prediction API.
"""

import os
import json
import pytest
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── Setup: create a mock model for testing ────────────────────────────
TEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "test_model")

FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]

SAMPLE_INPUT = {
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
    "wine_type": 0,
}


@pytest.fixture(scope="session", autouse=True)
def setup_mock_model():
    """Create a mock model for testing purposes."""
    os.makedirs(TEST_MODEL_DIR, exist_ok=True)

    # Create a simple, fitted model
    np.random.seed(42)
    X_mock = np.random.randn(100, len(FEATURE_COLS))
    y_mock = (np.random.rand(100) > 0.7).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_mock)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y_mock)

    joblib.dump(model, os.path.join(TEST_MODEL_DIR, "model.joblib"))
    joblib.dump(scaler, os.path.join(TEST_MODEL_DIR, "scaler.joblib"))

    with open(os.path.join(TEST_MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f)

    with open(os.path.join(TEST_MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"model_name": "test_model", "timestamp": "2024-01-01"}, f)

    # Set environment variable so app picks up the test model
    os.environ["MODEL_DIR"] = TEST_MODEL_DIR

    yield

    # Cleanup
    import shutil
    if os.path.exists(TEST_MODEL_DIR):
        shutil.rmtree(TEST_MODEL_DIR)


@pytest.fixture
def client():
    """Flask test client."""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ══════════════════════════════════════════════════════════════════════ #
#  TESTS                                                                 #
# ══════════════════════════════════════════════════════════════════════ #

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "timestamp" in data

    def test_health_json_format(self, client):
        response = client.get("/health")
        assert response.content_type == "application/json"


class TestInfoEndpoint:
    def test_info_returns_200(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.get_json()
        assert "model_name" in data
        assert "features" in data

    def test_info_has_features_list(self, client):
        response = client.get("/info")
        data = response.get_json()
        assert isinstance(data["features"], list)
        assert len(data["features"]) == len(FEATURE_COLS)


class TestPredictEndpoint:
    def test_predict_single_success(self, client):
        response = client.post(
            "/predict",
            data=json.dumps(SAMPLE_INPUT),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert "label" in data
        assert "probability_high_quality" in data
        assert 0 <= data["probability_high_quality"] <= 1

    def test_predict_batch_success(self, client):
        batch = [SAMPLE_INPUT, SAMPLE_INPUT]
        response = client.post(
            "/predict",
            data=json.dumps(batch),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_missing_features(self, client):
        incomplete = {"fixed acidity": 7.4}
        response = client.post(
            "/predict",
            data=json.dumps(incomplete),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_predict_invalid_value(self, client):
        invalid = {**SAMPLE_INPUT, "alcohol": "not_a_number"}
        response = client.post(
            "/predict",
            data=json.dumps(invalid),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_predict_empty_body(self, client):
        response = client.post(
            "/predict",
            data="",
            content_type="application/json",
        )
        # Flask will either return 400 or 500 depending on parsing
        assert response.status_code in [400, 500]


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_custom_metrics(self, client):
        # Make a prediction first to generate metrics
        client.post(
            "/predict",
            data=json.dumps(SAMPLE_INPUT),
            content_type="application/json",
        )
        response = client.get("/metrics")
        text = response.data.decode()
        assert "wine_prediction_requests_total" in text
        assert "wine_prediction_request_duration_seconds" in text


class TestNotFoundEndpoint:
    def test_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404
