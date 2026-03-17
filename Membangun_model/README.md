# Membangun Model — Wine Quality Prediction API

**Student:** M_Najwan_Naufal_A  
**Course:** Membangun Sistem Machine Learning — Dicoding

## Overview

REST API untuk prediksi kualitas wine menggunakan Flask, dikemas dalam Docker container.

### Endpoints

| Method | Endpoint   | Deskripsi                        |
|--------|-----------|----------------------------------|
| POST   | `/predict` | Prediksi kualitas wine (JSON)    |
| GET    | `/health`  | Health check                     |
| GET    | `/info`    | Metadata model                   |
| GET    | `/metrics` | Prometheus metrics               |

### Contoh Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Contoh Response

```json
{
  "prediction": 0,
  "label": "Low Quality",
  "probability_high_quality": 0.1234,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Setup

### 1. Siapkan Model Artifacts

Jalankan experiment pipeline di `Eksperimen_SML_M_Najwan_Naufal_A/` terlebih dulu, lalu salin best model:

```bash
# Dari root project
cp -r Eksperimen_SML_M_Najwan_Naufal_A/output/best_model/* Membangun_model/model/
```

### 2. Jalankan Lokal (tanpa Docker)

```bash
pip install -r requirements.txt
python app.py
```

### 3. Jalankan dengan Docker

```bash
docker compose up --build
```

### 4. Jalankan Tests

```bash
pip install pytest
pytest test_api.py -v
```

## Struktur

```
Membangun_model/
├── app.py                   # Flask API
├── model_utils.py           # Model loader utility
├── test_api.py              # Pytest test suite
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose config
├── requirements.txt         # Python dependencies
├── model/                   # Model artifacts (dari K1)
│   ├── model.joblib
│   ├── scaler.joblib
│   ├── feature_cols.json
│   └── metadata.json
└── README.md
```
