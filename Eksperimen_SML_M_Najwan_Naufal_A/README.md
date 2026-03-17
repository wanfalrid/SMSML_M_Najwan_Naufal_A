# Eksperimen SML — Wine Quality Binary Classification

**Student:** M_Najwan_Naufal_A  
**Course:** Membangun Sistem Machine Learning — Dicoding

## Overview

Pipeline eksperimen ML lengkap untuk klasifikasi biner kualitas wine (quality ≥ 7 = **High Quality**).

### Features
- **Dataset**: Wine Quality (red + white) dari UCI Repository
- **Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Hyperparameter Tuning**: GridSearchCV dengan 5-fold Stratified CV
- **Class Imbalance**: SMOTE (Synthetic Minority Oversampling)
- **Tracking**: MLflow (lokal atau DagsHub remote)
- **EDA**: Visualisasi lengkap (distribusi, korelasi, boxplots)

## Setup

```bash
# 1. Buat virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python download_data.py
```

## Menjalankan Eksperimen

### Lokal (tanpa DagsHub)
```bash
python modular_pipeline.py
```

### Dengan DagsHub Remote Tracking
```bash
# Set credential DagsHub
set MLFLOW_TRACKING_USERNAME=najwanopal
set MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>

# Jalankan pipeline
python modular_pipeline.py --dagshub-repo najwanopal/wine-quality-mlops
```

### Melihat MLflow UI (Lokal)
```bash
mlflow ui --port 5000
# Buka http://localhost:5000
```

## Struktur Output

```
output/
├── plots/
│   ├── quality_distribution.png
│   ├── binary_target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_boxplots.png
│   ├── quality_by_wine_type.png
│   ├── model_comparison.png
│   ├── cm_*.png               # Confusion matrices
│   └── roc_*.png              # ROC curves
├── best_model/
│   ├── model.joblib
│   ├── scaler.joblib
│   ├── feature_cols.json
│   └── metadata.json
├── eda_summary.json
├── experiment_summary.json
└── scaler.joblib
```

## Hasil Eksperimen

Setelah menjalankan pipeline, lihat:
- **MLflow UI** → semua eksperimen, params, metrics
- **output/experiment_summary.json** → ringkasan model comparison
- **output/plots/** → visualisasi EDA & evaluasi model
