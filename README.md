# Wine Quality MLOps — Proyek Akhir Dicoding

**Student:** M_Najwan_Naufal_A  
**Username:** najwanopal  
**Course:** Membangun Sistem Machine Learning — Dicoding

## Deskripsi Proyek

Sistem Machine Learning end-to-end untuk **klasifikasi biner kualitas wine** menggunakan Wine Quality Dataset dari UCI Repository. Wine dengan quality score ≥ 7 diklasifikasikan sebagai **High Quality**, sisanya **Low Quality**.

### Tech Stack
- **Python 3.10+** — Bahasa utama
- **Scikit-Learn** — Model ML (Logistic Regression, Random Forest, Gradient Boosting)
- **MLflow** — Experiment tracking & model registry
- **DagsHub** — Remote MLflow tracking
- **Flask** — Model serving REST API
- **Docker** — Containerization
- **GitHub Actions** — CI/CD pipeline
- **Prometheus + Grafana** — Monitoring & alerting

## Struktur Submission

```
SMSML_M_Najwan_Naufal_A.zip
├── Eksperimen_SML_M_Najwan_Naufal_A/     ← (K1) Eksperimen ML
│   ├── modular_pipeline.py                  Pipeline utama
│   ├── eda_analysis.py                      EDA & visualisasi
│   ├── download_data.py                     Download dataset
│   ├── requirements.txt                     Dependencies
│   └── README.md
│
├── Membangun_model/                        ← (K2) Model Serving
│   ├── app.py                               Flask REST API
│   ├── model_utils.py                       Model loader
│   ├── test_api.py                          API tests
│   ├── Dockerfile                           Docker image
│   ├── docker-compose.yml                   Docker Compose
│   ├── requirements.txt                     Dependencies
│   ├── model/                               Model artifacts
│   └── README.md
│
├── Workflow-CI.txt                         ← (K3) Link repo GitHub
│
├── .github/workflows/ci-cd.yml            ← CI/CD Pipeline
│
├── Monitoring dan Logging/                 ← (K4) Monitoring
│   ├── docker-compose.monitoring.yml        Full monitoring stack
│   ├── prometheus/
│   │   ├── prometheus.yml                   Prometheus config
│   │   └── alert_rules.yml                  Alert rules
│   ├── grafana/provisioning/
│   │   ├── datasources/datasource.yml       Datasource config
│   │   └── dashboards/
│   │       ├── dashboard.yml                Dashboard provisioning
│   │       └── wine_quality_dashboard.json  Dashboard JSON
│   ├── load_test.py                         Load testing script
│   └── README.md
│
├── .gitignore
└── README.md                               ← File ini
```

## Cara Menjalankan

### Step 1: Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r Eksperimen_SML_M_Najwan_Naufal_A/requirements.txt
```

### Step 2: Jalankan Experiment Pipeline (K1)
```bash
cd Eksperimen_SML_M_Najwan_Naufal_A
python download_data.py
python modular_pipeline.py
# Dengan DagsHub:
# python modular_pipeline.py --dagshub-repo najwanopal/wine-quality-mlops
```

### Step 3: Setup Model Serving (K2)
```bash
# Copy best model ke serving directory
cp -r Eksperimen_SML_M_Najwan_Naufal_A/output/best_model/* Membangun_model/model/

# Jalankan API
cd Membangun_model
docker compose up --build
```

### Step 4: Start Monitoring (K4)
```bash
cd "Monitoring dan Logging"
docker compose -f docker-compose.monitoring.yml up --build -d

# Akses:
# - API:        http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)
```

## Kriteria Penilaian Advanced

| Kriteria | Poin Advanced | Implementasi |
|----------|--------------|-------------|
| K1 - Eksperimen | 4 pts | 3 model + GridSearchCV + SMOTE + MLflow + EDA |
| K2 - Model | 4 pts | Flask API + Docker + DagsHub remote tracking |
| K3 - CI/CD | 4 pts | GitHub Actions (lint → test → Docker → train) |
| K4 - Monitoring | 4 pts | Prometheus + Grafana + Alert Rules + Dashboard |
