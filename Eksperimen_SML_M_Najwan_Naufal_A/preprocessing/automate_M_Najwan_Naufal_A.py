"""
=============================================================================
  Automated Preprocessing Pipeline — Wine Quality Binary Classification
  Student : M_Najwan_Naufal_A
  Course  : Membangun Sistem Machine Learning — Dicoding
=============================================================================

Script ini adalah versi otomatis dari notebook eksperimen.
Menjalankan seluruh preprocessing pipeline dari command line.

Usage:
    python automate_M_Najwan_Naufal_A.py
    python automate_M_Najwan_Naufal_A.py --input ../winequality_raw/winequality_combined.csv
    python automate_M_Najwan_Naufal_A.py --test-size 0.2 --random-state 42

Output:
    preprocessing/winequality_preprocessing/
    ├── X_train.csv
    ├── X_test.csv
    ├── y_train.csv
    ├── y_test.csv
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── preprocessing_report.json
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Logging Configuration ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("automate_preprocessing")


# ===================================================================== #
#  1. LOAD DATA                                                          #
# ===================================================================== #
def load_data(raw_path: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV.

    Parameters
    ----------
    raw_path : str
        Path ke file CSV dataset (winequality_combined.csv).

    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah di-load.

    Raises
    ------
    FileNotFoundError
        Jika file CSV tidak ditemukan.
    ValueError
        Jika file CSV kosong atau tidak valid.
    """
    log.info(f"Loading data dari: {raw_path}")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File tidak ditemukan: {raw_path}")

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        raise ValueError(f"Gagal membaca CSV: {e}")

    if df.empty:
        raise ValueError("Dataset kosong — tidak ada baris data")

    log.info(f"Data loaded: {df.shape[0]} baris × {df.shape[1]} kolom")
    log.info(f"Kolom: {list(df.columns)}")
    return df


# ===================================================================== #
#  2. CHECK DATA QUALITY                                                 #
# ===================================================================== #
def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Cek kualitas data: missing values, duplikasi, dan data types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    dict
        Report berisi ringkasan kualitas data:
        - shape, dtypes, missing_values, duplicates, class_distribution, dll.
    """
    log.info("Mengecek kualitas data ...")

    # Missing values per kolom
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Duplikasi
    n_duplicates = int(df.duplicated().sum())
    pct_duplicates = round(n_duplicates / len(df) * 100, 2)

    # Distribusi quality
    quality_dist = df["quality"].value_counts().sort_index().to_dict()

    # Binary target distribution
    high_quality = int((df["quality"] >= 7).sum())
    low_quality = int((df["quality"] < 7).sum())
    imbalance_ratio = round(low_quality / max(high_quality, 1), 2)

    report = {
        "shape": list(df.shape),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "total_missing_values": total_missing,
        "missing_per_column": {k: int(v) for k, v in missing_per_col.items()},
        "n_duplicates": n_duplicates,
        "pct_duplicates": pct_duplicates,
        "quality_distribution": {str(k): int(v) for k, v in quality_dist.items()},
        "binary_target": {
            "high_quality_gte7": high_quality,
            "low_quality_lt7": low_quality,
            "imbalance_ratio": imbalance_ratio,
        },
    }

    log.info(f"  Missing values  : {total_missing}")
    log.info(f"  Duplikat        : {n_duplicates} ({pct_duplicates}%)")
    log.info(f"  High Quality    : {high_quality} ({high_quality/len(df)*100:.1f}%)")
    log.info(f"  Low Quality     : {low_quality} ({low_quality/len(df)*100:.1f}%)")
    log.info(f"  Imbalance Ratio : {imbalance_ratio}:1")

    return report


# ===================================================================== #
#  3. HANDLE DUPLICATES                                                  #
# ===================================================================== #
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus baris duplikat dari dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa baris duplikat.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    n_removed = n_before - n_after

    if n_removed > 0:
        log.info(f"Duplikat dihapus: {n_removed} baris ({n_before} → {n_after})")
    else:
        log.info("Tidak ada duplikat ditemukan")

    return df


# ===================================================================== #
#  4. HANDLE MISSING VALUES                                              #
# ===================================================================== #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values menggunakan median imputation.
    Jika tidak ada missing values, tidak ada perubahan.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa missing values.
    """
    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        log.info("Tidak ada missing values — tidak perlu handling")
        return df

    log.info(f"Handling {total_missing} missing values dengan median imputation ...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.info(f"  {col}: {n_missing} missing → filled with median={median_val:.4f}")

    remaining = df.isnull().sum().sum()
    if remaining > 0:
        log.warning(f"Masih ada {remaining} missing values (non-numeric columns)")
    else:
        log.info("Semua missing values berhasil di-handle")

    return df


# ===================================================================== #
#  5. FEATURE ENGINEERING                                                #
# ===================================================================== #
def feature_engineering(df: pd.DataFrame) -> tuple:
    """
    Lakukan feature engineering:
    - Buat kolom target binary 'quality_label' (1 jika quality >= 7, 0 jika tidak)
    - Encode wine_type menggunakan LabelEncoder

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    tuple
        (df, label_encoder) — DataFrame yang sudah di-engineer dan LabelEncoder.
    """
    log.info("Feature engineering ...")

    # Buat kolom target binary
    df["quality_label"] = (df["quality"] >= 7).astype(int)
    high = int((df["quality_label"] == 1).sum())
    low = int((df["quality_label"] == 0).sum())
    log.info(f"  Target binary 'quality_label' dibuat: High={high}, Low={low}")

    # Encode wine_type menggunakan LabelEncoder
    le = LabelEncoder()
    df["wine_type"] = le.fit_transform(df["wine_type"])
    log.info(f"  wine_type di-encode: classes={list(le.classes_)}")

    return df, le


# ===================================================================== #
#  6. HANDLE OUTLIERS                                                    #
# ===================================================================== #
def handle_outliers(df: pd.DataFrame, columns: list) -> tuple:
    """
    Handle outlier menggunakan IQR method (capping/winsorizing).
    Nilai di bawah Q1 - 1.5*IQR di-cap ke lower bound.
    Nilai di atas Q3 + 1.5*IQR di-cap ke upper bound.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.
    columns : list
        List nama kolom numerik yang perlu di-handle outliernya.

    Returns
    -------
    tuple
        (df, outlier_report) — DataFrame setelah capping dan dict report outlier.
    """
    log.info("Handle outlier dengan IQR capping ...")

    outlier_report = {}
    total_capped = 0

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Hitung outlier sebelum capping
        n_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())

        # Capping
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        if n_outliers > 0:
            log.info(f"  {col:<26s} capped={n_outliers}  bounds=[{lower_bound:.3f}, {upper_bound:.3f}]")

        outlier_report[col] = {
            "n_outliers": n_outliers,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
        }
        total_capped += n_outliers

    log.info(f"Total nilai di-cap: {total_capped}")
    return df, outlier_report


# ===================================================================== #
#  7. SCALE FEATURES                                                     #
# ===================================================================== #
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Normalisasi fitur menggunakan StandardScaler.
    Fit pada train set, transform pada train dan test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Fitur training set.
    X_test : pd.DataFrame
        Fitur test set.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler) — DataFrames yang sudah di-scale
        dan objek StandardScaler.
    """
    log.info("Scaling fitur dengan StandardScaler ...")

    scaler = StandardScaler()

    feature_columns = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_columns,
        index=X_test.index,
    )

    log.info(f"  X_train mean ≈ {X_train_scaled.mean().mean():.6f} (target: 0)")
    log.info(f"  X_train std  ≈ {X_train_scaled.std().mean():.6f} (target: 1)")

    return X_train_scaled, X_test_scaled, scaler


# ===================================================================== #
#  8. SPLIT DATA                                                         #
# ===================================================================== #
def split_data(
    df: pd.DataFrame,
    target_col: str = "quality_label",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """
    Split dataset menjadi train dan test set dengan stratified sampling.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input yang sudah dipreprocessing.
    target_col : str
        Nama kolom target. Default: 'quality_label'.
    test_size : float
        Proporsi test set. Default: 0.20.
    random_state : int
        Random seed. Default: 42.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) — split dataset.
    """
    log.info(f"Splitting data (test_size={test_size}, random_state={random_state}) ...")

    # Pisahkan fitur dan target
    feature_columns = [col for col in df.columns if col not in ("quality", target_col)]
    X = df[feature_columns].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    log.info(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    log.info(
        f"  y_train — Low: {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%), "
        f"High: {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)"
    )
    log.info(
        f"  y_test  — Low: {(y_test==0).sum()} ({(y_test==0).mean()*100:.1f}%), "
        f"High: {(y_test==1).sum()} ({(y_test==1).mean()*100:.1f}%)"
    )

    return X_train, X_test, y_train, y_test


# ===================================================================== #
#  9. SAVE PREPROCESSED DATA                                             #
# ===================================================================== #
def save_preprocessed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    encoder: LabelEncoder,
    output_dir: str,
) -> dict:
    """
    Simpan semua hasil preprocessing ke folder output.

    Parameters
    ----------
    X_train : pd.DataFrame
        Fitur training set (scaled).
    X_test : pd.DataFrame
        Fitur test set (scaled).
    y_train : pd.Series
        Label training set.
    y_test : pd.Series
        Label test set.
    scaler : StandardScaler
        Fitted StandardScaler.
    encoder : LabelEncoder
        Fitted LabelEncoder.
    output_dir : str
        Path folder output.

    Returns
    -------
    dict
        Report berisi informasi file yang disimpan.
    """
    log.info(f"Menyimpan hasil preprocessing ke: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Simpan datasets
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    # Simpan scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Simpan label encoder
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    # Verifikasi dan buat report
    saved_files = {}
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size_kb = round(os.path.getsize(fpath) / 1024, 1)
        saved_files[fname] = f"{size_kb} KB"
        log.info(f"  📄 {fname:<30s} ({size_kb} KB)")

    log.info("Semua file berhasil disimpan ✓")
    return saved_files


# ===================================================================== #
#  10. MAIN — ORCHESTRATOR                                               #
# ===================================================================== #
def main():
    """
    Main function — menjalankan seluruh preprocessing pipeline.

    Pipeline steps:
        1. Load data
        2. Check data quality
        3. Handle duplicates
        4. Handle missing values
        5. Feature engineering
        6. Handle outliers (IQR capping)
        7. Split data (stratified)
        8. Scale features (StandardScaler)
        9. Save preprocessed data
        10. Generate preprocessing report
    """
    # ── Parse arguments ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Automated Preprocessing Pipeline — Wine Quality Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automate_M_Najwan_Naufal_A.py
  python automate_M_Najwan_Naufal_A.py --input ../winequality_raw/winequality_combined.csv
  python automate_M_Najwan_Naufal_A.py --test-size 0.3 --random-state 123
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("..", "winequality_raw", "winequality_combined.csv"),
        help="Path ke raw dataset CSV (default: ../winequality_raw/winequality_combined.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(".", "winequality_preprocessing"),
        help="Path folder output preprocessing (default: ./winequality_preprocessing/)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # ── Banner ───────────────────────────────────────────────────────
    print()
    log.info("=" * 65)
    log.info("  AUTOMATED PREPROCESSING PIPELINE")
    log.info("  Wine Quality Binary Classification")
    log.info("  Student: M_Najwan_Naufal_A")
    log.info("=" * 65)
    log.info(f"  Input  : {os.path.abspath(args.input)}")
    log.info(f"  Output : {os.path.abspath(args.output)}")
    log.info(f"  Test % : {args.test_size}")
    log.info(f"  Seed   : {args.random_state}")
    log.info("=" * 65)
    print()

    # Inisialisasi report
    report = {
        "pipeline": "Wine Quality Preprocessing",
        "student": "M_Najwan_Naufal_A",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "input_path": os.path.abspath(args.input),
            "output_dir": os.path.abspath(args.output),
            "test_size": args.test_size,
            "random_state": args.random_state,
        },
    }

    try:
        # ── Step 1 — Load data ───────────────────────────────────────
        log.info("[Step 1/9] Loading data ...")
        df = load_data(args.input)
        report["original_shape"] = list(df.shape)

        # ── Step 2 — Check data quality ──────────────────────────────
        log.info("[Step 2/9] Checking data quality ...")
        quality_report = check_data_quality(df)
        report["data_quality"] = quality_report

        # ── Step 3 — Handle duplicates ───────────────────────────────
        log.info("[Step 3/9] Handling duplicates ...")
        df = handle_duplicates(df)
        report["shape_after_dedup"] = list(df.shape)
        report["rows_removed_duplicates"] = (
            report["original_shape"][0] - df.shape[0]
        )

        # ── Step 4 — Handle missing values ───────────────────────────
        log.info("[Step 4/9] Handling missing values ...")
        df = handle_missing_values(df)

        # ── Step 5 — Feature engineering ─────────────────────────────
        log.info("[Step 5/9] Feature engineering ...")
        df, label_encoder = feature_engineering(df)

        # ── Step 6 — Handle outliers ─────────────────────────────────
        log.info("[Step 6/9] Handling outliers ...")
        outlier_columns = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col not in ("quality", "quality_label", "wine_type")
        ]
        df, outlier_report = handle_outliers(df, outlier_columns)
        report["outlier_handling"] = outlier_report

        # ── Step 7 — Split data ──────────────────────────────────────
        log.info("[Step 7/9] Splitting data ...")
        X_train, X_test, y_train, y_test = split_data(
            df,
            target_col="quality_label",
            test_size=args.test_size,
            random_state=args.random_state,
        )

        report["split"] = {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_distribution": {
                "low_quality_0": int((y_train == 0).sum()),
                "high_quality_1": int((y_train == 1).sum()),
            },
            "y_test_distribution": {
                "low_quality_0": int((y_test == 0).sum()),
                "high_quality_1": int((y_test == 1).sum()),
            },
            "feature_columns": list(X_train.columns),
        }

        # ── Step 8 — Scale features ─────────────────────────────────
        log.info("[Step 8/9] Scaling features ...")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        report["scaling"] = {
            "method": "StandardScaler",
            "train_mean": round(float(X_train_scaled.mean().mean()), 6),
            "train_std": round(float(X_train_scaled.std().mean()), 6),
        }

        # ── Step 9 — Save preprocessed data ──────────────────────────
        log.info("[Step 9/9] Saving preprocessed data ...")
        saved_files = save_preprocessed_data(
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            scaler, label_encoder,
            args.output,
        )
        report["saved_files"] = saved_files

        # ── Save preprocessing report ────────────────────────────────
        report["status"] = "SUCCESS"
        report["final_shape"] = {
            "total_samples": int(X_train_scaled.shape[0] + X_test_scaled.shape[0]),
            "n_features": int(X_train_scaled.shape[1]),
            "train_samples": int(X_train_scaled.shape[0]),
            "test_samples": int(X_test_scaled.shape[0]),
        }

        report_path = os.path.join(args.output, "preprocessing_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"Report disimpan ke: {report_path}")

        # ── Final Summary ────────────────────────────────────────────
        print()
        log.info("=" * 65)
        log.info("  PREPROCESSING SELESAI ✓")
        log.info("=" * 65)
        log.info(f"  Shape awal     : {report['original_shape']}")
        log.info(f"  Duplikat hapus : {report['rows_removed_duplicates']}")
        log.info(f"  Shape akhir    : {report['final_shape']['total_samples']} samples × {report['final_shape']['n_features']} features")
        log.info(f"  Train samples  : {report['final_shape']['train_samples']}")
        log.info(f"  Test samples   : {report['final_shape']['test_samples']}")
        log.info(f"  Output folder  : {os.path.abspath(args.output)}")
        log.info("=" * 65)
        print()

    except FileNotFoundError as e:
        log.error(f"File tidak ditemukan: {e}")
        report["status"] = "FAILED"
        report["error"] = str(e)
        sys.exit(1)

    except ValueError as e:
        log.error(f"Data error: {e}")
        report["status"] = "FAILED"
        report["error"] = str(e)
        sys.exit(1)

    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        report["status"] = "FAILED"
        report["error"] = str(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
