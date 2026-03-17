"""
=============================================================================
  Modelling — Wine Quality Binary Classification (Baseline)
  Student : M_Najwan_Naufal_A
  Course  : Membangun Sistem Machine Learning — Dicoding
=============================================================================

Script ini melatih model RandomForestClassifier sebagai baseline,
dengan MLflow autolog tracking ke DagsHub remote.

Usage:
    # Dengan DagsHub (production)
    set MLFLOW_TRACKING_URI=https://dagshub.com/najwanopal/wine-quality-mlops.mlflow
    set MLFLOW_TRACKING_USERNAME=najwanopal
    set MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
    python modelling.py

    # Lokal (tanpa DagsHub)
    python modelling.py --local
"""

import os
import sys
import json
import pickle
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

import mlflow
import mlflow.sklearn

# ── Optional: DagsHub integration ────────────────────────────────────
try:
    import dagshub
    DAGSHUB_AVAILABLE = True
except ImportError:
    DAGSHUB_AVAILABLE = False

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("modelling")

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSING_DIR = os.path.join(
    BASE_DIR, "..", "Eksperimen_SML_M_Najwan_Naufal_A",
    "preprocessing", "winequality_preprocessing"
)
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


# ===================================================================== #
#  1. LOAD PREPROCESSED DATA                                             #
# ===================================================================== #
def load_preprocessed_data(data_dir: str) -> tuple:
    """
    Load data hasil preprocessing.

    Parameters
    ----------
    data_dir : str
        Path ke folder winequality_preprocessing/.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    log.info(f"Loading preprocessed data dari: {data_dir}")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    log.info(f"  X_train: {X_train.shape}")
    log.info(f"  X_test:  {X_test.shape}")
    log.info(f"  y_train: {y_train.shape} — 0:{(y_train==0).sum()}, 1:{(y_train==1).sum()}")
    log.info(f"  y_test:  {y_test.shape} — 0:{(y_test==0).sum()}, 1:{(y_test==1).sum()}")

    return X_train, X_test, y_train, y_test


# ===================================================================== #
#  2. SETUP MLFLOW                                                       #
# ===================================================================== #
def setup_mlflow(local: bool = False, dagshub_repo: str = None):
    """
    Setup MLflow tracking.
    - Jika local=True, tracking disimpan di folder lokal mlruns/
    - Jika dagshub_repo diberikan, tracking ke DagsHub remote

    Parameters
    ----------
    local : bool
        Gunakan local tracking (default: False).
    dagshub_repo : str
        DagsHub repo format 'username/repo_name'. Default: None.
    """
    if local:
        # PENTING: gunakan file:/// prefix agar MLflow tidak error
        mlruns_path = os.path.abspath(os.path.join(BASE_DIR, "mlruns"))
        tracking_uri = "file:///" + mlruns_path.replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)
        log.info(f"MLflow tracking: LOCAL → {tracking_uri}")
    elif dagshub_repo and DAGSHUB_AVAILABLE:
        owner, repo = dagshub_repo.split("/")
        dagshub.init(repo_owner=owner, repo_name=repo, mlflow=True)
        log.info(f"MLflow tracking: DAGSHUB → {mlflow.get_tracking_uri()}")
    else:
        # Fallback: cek environment variables
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            log.info(f"MLflow tracking: ENV → {tracking_uri}")
        else:
            mlruns_path = os.path.abspath(os.path.join(BASE_DIR, "mlruns"))
            tracking_uri = "file:///" + mlruns_path.replace("\\", "/")
            mlflow.set_tracking_uri(tracking_uri)
            log.info(f"MLflow tracking: LOCAL (fallback) → {tracking_uri}")


# ===================================================================== #
#  3. PLOT CONFUSION MATRIX                                              #
# ===================================================================== #
def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """
    Buat dan simpan confusion matrix plot.

    Parameters
    ----------
    y_true : array-like
        Label ground truth.
    y_pred : array-like
        Label prediksi.
    save_path : str
        Path untuk menyimpan plot.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Low Quality", "High Quality"],
        cmap="Blues", ax=ax,
        colorbar=True,
    )
    ax.set_title("Confusion Matrix — Baseline Random Forest",
                 fontsize=14, fontweight="bold", pad=15)

    # Tambahkan subtitle dengan metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ax.text(0.5, -0.12, f"Accuracy: {acc:.4f}  |  F1-Score: {f1:.4f}",
            transform=ax.transAxes, ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Confusion matrix saved: {save_path}")


# ===================================================================== #
#  4. PLOT FEATURE IMPORTANCE                                            #
# ===================================================================== #
def plot_feature_importance(model, feature_names, save_path: str):
    """
    Buat dan simpan feature importance plot.

    Parameters
    ----------
    model : RandomForestClassifier
        Model yang sudah di-train.
    feature_names : list
        Nama-nama fitur.
    save_path : str
        Path untuk menyimpan plot.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    sorted_colors = [colors[i] for i in indices]

    bars = ax.barh(
        range(len(feature_names)),
        importances[indices],
        color=sorted_colors,
        edgecolor="black", linewidth=0.5,
    )

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Feature Importance — Baseline Random Forest",
                 fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Tambahkan nilai di ujung bar
    for bar, imp in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{imp:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Feature importance saved: {save_path}")


# ===================================================================== #
#  5. PLOT ROC CURVE                                                     #
# ===================================================================== #
def plot_roc_curve(model, X_test, y_test, save_path: str):
    """
    Buat dan simpan ROC curve plot.

    Parameters
    ----------
    model : trained model
        Model yang sudah di-train.
    X_test : pd.DataFrame
        Fitur test set.
    y_test : pd.Series
        Label test set.
    save_path : str
        Path untuk menyimpan plot.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(
        model, X_test, y_test,
        ax=ax, name="Random Forest",
        color="#2ecc71", linewidth=2,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_title("ROC Curve — Baseline Random Forest",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  ROC curve saved: {save_path}")


# ===================================================================== #
#  6. TRAIN AND LOG MODEL                                                #
# ===================================================================== #
def train_model(X_train, X_test, y_train, y_test):
    """
    Train RandomForestClassifier dan log semua ke MLflow.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Fitur train dan test set.
    y_train, y_test : pd.Series
        Label train dan test set.

    Returns
    -------
    dict
        Hasil training berisi model, metrics, run_id, dll.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Set experiment ───────────────────────────────────────────────
    experiment_name = "wine-quality-classification"
    mlflow.set_experiment(experiment_name)
    log.info(f"MLflow experiment: {experiment_name}")

    # ── Start run ────────────────────────────────────────────────────
    with mlflow.start_run(run_name="baseline-random-forest") as run:
        run_id = run.info.run_id
        log.info(f"MLflow Run ID: {run_id}")

        # ── Train model ─────────────────────────────────────────────
        log.info("Training RandomForestClassifier (baseline) ...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        log.info("Training selesai ✓")

        # ── Log parameters manually ─────────────────────────────────
        params = model.get_params()
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass
        mlflow.log_param("target_threshold", 7)
        mlflow.log_param("task_type", "binary_classification")
        mlflow.log_param("dataset", "wine-quality-combined")

        # ── Log model artifact (tanpa registry) ─────────────────────
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
            log.info("Model logged to MLflow ✓")
        except Exception as e:
            log.warning(f"Gagal log model ke MLflow (wajar untuk local): {e}")

        # ── Predictions ─────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # ── Calculate metrics ────────────────────────────────────────
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # ── Generate artifacts ───────────────────────────────────────
        log.info("Generating artifacts ...")

        # 1. Confusion matrix
        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path, "plots")

        # 2. Feature importance
        fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plot_feature_importance(model, list(X_train.columns), fi_path)
        mlflow.log_artifact(fi_path, "plots")

        # 3. ROC curve
        roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
        plot_roc_curve(model, X_test, y_test, roc_path)
        mlflow.log_artifact(roc_path, "plots")

        # 4. Classification report
        report_text = classification_report(
            y_test, y_pred,
            target_names=["Low Quality", "High Quality"],
        )
        report_path = os.path.join(PLOTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("Classification Report — Baseline Random Forest\n")
            f.write("=" * 55 + "\n\n")
            f.write(report_text)
            f.write(f"\nROC AUC: {metrics['test_roc_auc']:.4f}\n")
        mlflow.log_artifact(report_path, "reports")
        log.info(f"  Classification report saved: {report_path}")

        # ── Save model locally ───────────────────────────────────────
        model_path = os.path.join(MODELS_DIR, "baseline_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Model saved locally: {model_path}")

        # Save metadata
        metadata = {
            "model_name": "RandomForestClassifier",
            "run_name": "baseline-random-forest",
            "run_id": run_id,
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "params": model.get_params(),
            "feature_columns": list(X_train.columns),
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = os.path.join(MODELS_DIR, "baseline_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        mlflow.log_artifact(meta_path, "metadata")
        log.info(f"  Metadata saved: {meta_path}")

    return {
        "model": model,
        "metrics": metrics,
        "run_id": run_id,
        "report": report_text,
    }


# ===================================================================== #
#  7. MAIN                                                               #
# ===================================================================== #
def main():
    """Main function — jalankan seluruh pipeline modelling."""
    parser = argparse.ArgumentParser(
        description="Wine Quality Baseline Modelling with MLflow",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Gunakan MLflow tracking lokal (tanpa DagsHub)",
    )
    parser.add_argument(
        "--dagshub-repo", type=str, default=None,
        help="DagsHub repo format 'username/repo_name'",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path ke folder preprocessing output",
    )
    args = parser.parse_args()

    # Banner
    print()
    log.info("=" * 65)
    log.info("  WINE QUALITY MODELLING — BASELINE")
    log.info("  Student: M_Najwan_Naufal_A")
    log.info("=" * 65)

    # ── Setup MLflow ─────────────────────────────────────────────────
    setup_mlflow(local=args.local, dagshub_repo=args.dagshub_repo)

    # ── Load data ────────────────────────────────────────────────────
    data_dir = args.data_dir or PREPROCESSING_DIR
    if not os.path.exists(data_dir):
        log.error(f"Preprocessing folder tidak ditemukan: {data_dir}")
        log.error("Jalankan automate_M_Najwan_Naufal_A.py terlebih dahulu!")
        sys.exit(1)

    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)

    # ── Train model ──────────────────────────────────────────────────
    result = train_model(X_train, X_test, y_train, y_test)

    # ── Print summary ────────────────────────────────────────────────
    print()
    log.info("=" * 65)
    log.info("  TRAINING SELESAI ✓")
    log.info("=" * 65)
    log.info(f"  Model     : RandomForestClassifier (baseline)")
    log.info(f"  Run ID    : {result['run_id']}")
    log.info(f"  Tracking  : {mlflow.get_tracking_uri()}")
    print()
    log.info("  Test Metrics:")
    for name, value in result["metrics"].items():
        bar = "█" * int(value * 30)
        log.info(f"    {name:<20s} : {value:.4f}  {bar}")
    print()
    log.info(f"  Classification Report:")
    for line in result["report"].split("\n"):
        log.info(f"    {line}")
    print()
    log.info("  Files saved:")
    log.info(f"    📄 models/baseline_model.pkl")
    log.info(f"    📄 models/baseline_metadata.json")
    log.info(f"    📊 plots/confusion_matrix.png")
    log.info(f"    📊 plots/feature_importance.png")
    log.info(f"    📊 plots/roc_curve.png")
    log.info(f"    📋 plots/classification_report.txt")
    log.info("=" * 65)
    print()


if __name__ == "__main__":
    main()
