# Eksperimen SML — Wine Quality Binary Classification

![Preprocessing Pipeline](https://github.com/najwanopal/wine-quality-mlops/actions/workflows/preprocessing.yml/badge.svg)
![CI/CD Pipeline](https://github.com/najwanopal/wine-quality-mlops/actions/workflows/ci-cd.yml/badge.svg)

**Student:** M_Najwan_Naufal_A  
**Username Dicoding:** najwanopal  
**Course:** Membangun Sistem Machine Learning — Dicoding

---

## 📋 Deskripsi Proyek

Pipeline eksperimen Machine Learning lengkap untuk **klasifikasi biner kualitas wine**.  
Menentukan apakah wine berkualitas **tinggi** (quality ≥ 7) atau **rendah** (quality < 7) berdasarkan 11 fitur fisikokimia.

### Dataset
- **Sumber:** [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **File:**
  - `winequality-red.csv` — 1.599 sampel red wine
  - `winequality-white.csv` — 4.898 sampel white wine
  - `winequality_combined.csv` — 6.497 sampel gabungan (+ kolom `wine_type`)
- **Fitur (12):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, wine_type
- **Target:** `quality_label` — 1 (High Quality ≥ 7), 0 (Low Quality < 7)
- **Class Imbalance:** ~80% Low vs ~20% High

### Tech Stack
| Tool | Fungsi |
|------|--------|
| Python 3.10+ | Bahasa utama |
| Pandas & NumPy | Data manipulation |
| Scikit-Learn | Preprocessing & modeling |
| Matplotlib & Seaborn | Visualisasi |
| GitHub Actions | Automasi preprocessing |

---

## 📁 Struktur Folder

```
Eksperimen_SML_M_Najwan_Naufal_A/
├── download_data.py                           # Download & gabungkan dataset
├── modular_pipeline.py                        # Full ML pipeline (training)
├── eda_analysis.py                            # EDA visualisasi standalone
├── requirements.txt                           # Dependencies
├── README.md                                  # Dokumentasi ini
│
├── winequality_raw/                           # Dataset (auto-download)
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality_combined.csv
│
└── preprocessing/                             # Preprocessing
    ├── Eksperimen_M_Najwan_Naufal_A.ipynb      # Notebook EDA & preprocessing
    ├── automate_M_Najwan_Naufal_A.py           # Script otomatis preprocessing
    └── winequality_preprocessing/              # Output hasil preprocessing
        ├── X_train.csv
        ├── X_test.csv
        ├── y_train.csv
        ├── y_test.csv
        ├── scaler.pkl
        ├── label_encoder.pkl
        └── preprocessing_report.json
```

---

## 🚀 Cara Menjalankan

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/najwanopal/wine-quality-mlops.git
cd wine-quality-mlops

# Buat virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r Eksperimen_SML_M_Najwan_Naufal_A/requirements.txt
```

### 2. Download Dataset

```bash
cd Eksperimen_SML_M_Najwan_Naufal_A
python download_data.py
```

Output: `winequality_raw/winequality_combined.csv` (6.497 sampel)

### 3. Menjalankan Notebook Eksperimen

```bash
pip install jupyter
cd preprocessing
jupyter notebook Eksperimen_M_Najwan_Naufal_A.ipynb
```

Jalankan semua cell secara berurutan. Notebook berisi:
- Section 1: Import Libraries
- Section 2: Data Loading
- Section 3: Comprehensive EDA (10 subsection)
- Section 4: Data Preprocessing (7 step)
- Section 5: Kesimpulan & ringkasan

### 4. Menjalankan Automate Script

```bash
cd preprocessing

# Default (test_size=0.2, random_state=42)
python automate_M_Najwan_Naufal_A.py

# Custom parameters
python automate_M_Najwan_Naufal_A.py --test-size 0.3 --random-state 123

# Custom input/output
python automate_M_Najwan_Naufal_A.py \
  --input ../winequality_raw/winequality_combined.csv \
  --output ./winequality_preprocessing/
```

Output: `preprocessing/winequality_preprocessing/` (6 file + report)

### 5. Trigger GitHub Actions Workflow

**Otomatis:**
- Push perubahan ke `winequality_raw/` → workflow jalan otomatis
- Setiap Senin 00:00 UTC → scheduled run

**Manual:**
1. Buka tab **Actions** di GitHub repository
2. Pilih workflow **Preprocessing Pipeline**
3. Klik **Run workflow**
4. (Opsional) Isi `test_size` dan `random_state`
5. Klik **Run workflow**

---

## 📊 Preprocessing Pipeline

| Step | Deskripsi |
|------|-----------|
| 1 | Load `winequality_combined.csv` |
| 2 | Cek kualitas data (missing, duplikat, dtypes) |
| 3 | Drop baris duplikat (~1.177 baris) |
| 4 | Handle missing values (median imputation) |
| 5 | Feature engineering: binary target + encode wine_type |
| 6 | Handle outlier dengan IQR capping |
| 7 | Stratified train-test split (80:20) |
| 8 | StandardScaler (fit on train, transform on both) |
| 9 | Simpan semua file + report JSON |

---

## 📈 Output Files

| File | Deskripsi |
|------|-----------|
| `X_train.csv` | Fitur training set (scaled) |
| `X_test.csv` | Fitur test set (scaled) |
| `y_train.csv` | Label training set |
| `y_test.csv` | Label test set |
| `scaler.pkl` | StandardScaler (fitted) — untuk inference |
| `label_encoder.pkl` | LabelEncoder (fitted) — untuk inference |
| `preprocessing_report.json` | Statistik lengkap preprocessing |
