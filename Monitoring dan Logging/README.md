# Monitoring dan Logging — Wine Quality MLOps

**Student:** M_Najwan_Naufal_A  
**Course:** Membangun Sistem Machine Learning — Dicoding

## Overview

Stack monitoring lengkap menggunakan **Prometheus + Grafana** untuk memantau Wine Quality Prediction API.

### Komponen
| Service           | Port  | URL                          |
|-------------------|-------|------------------------------|
| Wine Quality API  | 5000  | http://localhost:5000        |
| Prometheus        | 9090  | http://localhost:9090        |
| Grafana           | 3000  | http://localhost:3000        |

## Quick Start

### 1. Siapkan Model

Pastikan model artifacts sudah ada di `Membangun_model/model/`:
```bash
# Dari root project, setelah menjalankan experiment pipeline
cp -r Eksperimen_SML_M_Najwan_Naufal_A/output/best_model/* Membangun_model/model/
```

### 2. Jalankan Monitoring Stack

```bash
cd "Monitoring dan Logging"
docker compose -f docker-compose.monitoring.yml up --build -d
```

### 3. Akses Dashboard

1. Buka **Grafana**: http://localhost:3000
2. Login: `admin` / `admin`
3. Navigasi ke **Dashboards → Wine Quality MLOps → Wine Quality ML API**

### 4. Generate Traffic (untuk melihat metrics)

```bash
# Kirim beberapa prediction requests
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0,"residual sugar":1.9,"chlorides":0.076,"free sulfur dioxide":11,"total sulfur dioxide":34,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4,"wine_type":0}'
```

Atau gunakan load test script:
```bash
python load_test.py --url http://localhost:5000 --requests 100 --concurrency 5
```

## Dashboard Panels

Dashboard Grafana berisi panel-panel berikut:

### Overview Row
- **API Status** — UP/DOWN indicator
- **Total Predictions** — Counter total predictions
- **Error Rate** — Persentase error (5 menit)
- **P95 Latency** — 95th percentile response time
- **Request Rate** — Requests per second

### Request Metrics
- **Request Rate Over Time** — Success vs Error rate timeline
- **Request Latency Percentiles** — p50, p90, p95, p99 latency

### Prediction Metrics
- **Prediction Distribution** — Stacked bar chart (Low vs High quality over time)
- **Prediction Class Distribution** — Donut chart (total proportions)
- **Probability Distribution** — Histogram of prediction probabilities

## Alert Rules

| Alert              | Condition                              | Severity |
|--------------------|----------------------------------------|----------|
| APIDown            | Service unreachable > 1 min            | Critical |
| HighErrorRate      | Error rate > 10% for 5 min             | Warning  |
| HighLatency        | p95 latency > 1s for 5 min             | Warning  |
| NoTraffic          | Zero requests for 30 min               | Info     |
| PredictionSkew     | >80% "High Quality" predictions (1 hr) | Warning  |

## Menghentikan Stack

```bash
docker compose -f docker-compose.monitoring.yml down
# Dengan volume data:
docker compose -f docker-compose.monitoring.yml down -v
```

## Struktur Folder

```
Monitoring dan Logging/
├── docker-compose.monitoring.yml
├── prometheus/
│   ├── prometheus.yml
│   └── alert_rules.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── datasource.yml
│       └── dashboards/
│           ├── dashboard.yml
│           └── wine_quality_dashboard.json
├── load_test.py
└── README.md
```
