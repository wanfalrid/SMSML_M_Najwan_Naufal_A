"""
=============================================================================
  Modular ML Experiment Pipeline — Wine Quality Binary Classification
  Student : M_Najwan_Naufal_A
  Course  : Membangun Sistem Machine Learning — Dicoding
=============================================================================

Pipeline steps
──────────────
1. Load & merge red/white CSV  →  add `wine_type` feature
2. Feature engineering & EDA summary
3. Binary target  (quality >= 7 → 1, else → 0)
4. Train / Validation / Test split  (70 / 15 / 15)
5. Preprocessing  (StandardScaler + SMOTE on train only)
6. Train 3+ models with GridSearchCV
7. Evaluate on val & test  →  log everything to MLflow
8. Register best model
"""

import os, json, warnings, argparse, logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import mlflow
import mlflow.sklearn

# ── Optional: DagsHub integration ──────────────────────────────────────────
try:
    import dagshub
    DAGSHUB_AVAILABLE = True
except ImportError:
    DAGSHUB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "winequality_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")

# ===================================================================== #
#  1. DATA LOADING                                                       #
# ===================================================================== #
def load_data() -> pd.DataFrame:
    """Load wine quality dataset.
    Prefers combined CSV if available, otherwise loads & merges raw CSVs.
    """
    combined_path = os.path.join(DATA_DIR, "winequality_combined.csv")
    red_path      = os.path.join(DATA_DIR, "winequality-red.csv")
    white_path    = os.path.join(DATA_DIR, "winequality-white.csv")

    # Option 1: load from combined CSV (already has wine_type)
    if os.path.exists(combined_path):
        df = pd.read_csv(combined_path)
        log.info(f"Loaded combined dataset: {len(df)} samples from {combined_path}")
        return df

    # Option 2: load & merge raw CSVs
    if not os.path.exists(red_path) or not os.path.exists(white_path):
        log.info("Data not found — running download_data.py …")
        from download_data import download_data, combine_data
        download_data()
        combined = combine_data()
        return combined

    red   = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    red["wine_type"]   = 0          # 0 = red
    white["wine_type"] = 1          # 1 = white

    df = pd.concat([red, white], ignore_index=True)
    log.info(f"Loaded {len(red)} red + {len(white)} white = {len(df)} total samples")
    return df


# ===================================================================== #
#  2. EDA (Exploratory Data Analysis)                                    #
# ===================================================================== #
def run_eda(df: pd.DataFrame) -> dict:
    """Generate EDA plots and return summary statistics."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    summary = {}

    # -- Basic stats -------------------------------------------------------
    summary["shape"]            = list(df.shape)
    summary["missing_values"]   = int(df.isnull().sum().sum())
    summary["class_distribution"] = df["quality"].value_counts().to_dict()
    summary["describe"]         = df.describe().to_dict()

    # -- 2a. Quality distribution ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="quality", data=df, hue="quality", palette="viridis", ax=ax, legend=False)
    ax.set_title("Wine Quality Score Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "quality_distribution.png"), dpi=150)
    plt.close(fig)

    # -- 2b. Binary target distribution ------------------------------------
    df_temp = df.copy()
    df_temp["is_high_quality"] = (df_temp["quality"] >= 7).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df_temp["is_high_quality"].value_counts()
    ax.bar(["Low (< 7)", "High (≥ 7)"], [counts[0], counts[1]],
           color=["#e74c3c", "#2ecc71"])
    ax.set_title("Binary Target Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v + 30, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "binary_target_distribution.png"), dpi=150)
    plt.close(fig)

    # -- 2c. Correlation heatmap -------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # -- 2d. Feature distributions by quality class -----------------------
    features = [c for c in numeric_cols if c not in ("quality", "wine_type")]
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        sns.boxplot(x="is_high_quality", y=feat, data=df_temp,
                    hue="is_high_quality", palette=["#e74c3c", "#2ecc71"],
                    ax=axes[i], legend=False)
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("High Quality")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions: Low vs High Quality", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_boxplots.png"), dpi=150)
    plt.close(fig)

    # -- 2e. Wine type vs quality -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ct = pd.crosstab(df["quality"], df["wine_type"])
    ct.columns = ["Red", "White"]
    ct.plot(kind="bar", stacked=True, color=["#c0392b", "#f1c40f"], ax=ax)
    ax.set_title("Quality by Wine Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "quality_by_wine_type.png"), dpi=150)
    plt.close(fig)

    log.info(f"EDA plots saved to {PLOTS_DIR}")
    return summary


# ===================================================================== #
#  3. PREPROCESSING                                                      #
# ===================================================================== #
def preprocess(df: pd.DataFrame):
    """
    • Create binary target  (quality >= 7)
    • 70/15/15 train-validation-test split  (stratified)
    • StandardScaler on features
    • SMOTE on training set only
    """
    df = df.copy()
    df["is_high_quality"] = (df["quality"] >= 7).astype(int)

    feature_cols = [c for c in df.columns if c not in ("quality", "is_high_quality")]
    X = df[feature_cols].values
    y = df["is_high_quality"].values

    log.info(f"Class ratio → 0: {(y == 0).sum()}  |  1: {(y == 1).sum()}")

    # First split: 70 % train, 30 % remaining
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # Second split: 50/50 of remaining → 15 % val, 15 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    log.info(f"Split sizes → train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info(
        f"After SMOTE → 0: {(y_train_res == 0).sum()}  |  "
        f"1: {(y_train_res == 1).sum()}"
    )

    return (X_train_res, y_train_res, X_val, y_val, X_test, y_test,
            scaler, feature_cols)


# ===================================================================== #
#  4. MODEL DEFINITIONS AND HYPERPARAMETER GRIDS                         #
# ===================================================================== #
def get_model_configs():
    """Return list of (name, estimator, param_grid) tuples."""
    return [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
        ),
    ]


# ===================================================================== #
#  5. TRAINING + EVALUATION                                              #
# ===================================================================== #
def evaluate_model(model, X, y, dataset_name="val"):
    """Return dict of metrics for a given dataset split."""
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        f"{dataset_name}_accuracy":  accuracy_score(y, y_pred),
        f"{dataset_name}_precision": precision_score(y, y_pred, zero_division=0),
        f"{dataset_name}_recall":    recall_score(y, y_pred, zero_division=0),
        f"{dataset_name}_f1":        f1_score(y, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics[f"{dataset_name}_roc_auc"] = roc_auc_score(y, y_proba)

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, name, dataset_name):
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Low", "High"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"{name} — Confusion Matrix ({dataset_name})", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"cm_{name}_{dataset_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(model, X, y, name, dataset_name):
    """Save ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)
    ax.set_title(f"{name} — ROC Curve ({dataset_name})", fontweight="bold")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_{name}_{dataset_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_and_evaluate(
    name, estimator, param_grid,
    X_train, y_train, X_val, y_val, X_test, y_test,
):
    """Run GridSearchCV, evaluate on val+test, log to MLflow."""

    log.info(f"\n{'='*60}\n  Training: {name}\n{'='*60}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator, param_grid,
        cv=cv, scoring="f1", n_jobs=-1, verbose=0, refit=True,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    log.info(f"Best params: {grid.best_params_}")
    log.info(f"Best CV F1: {grid.best_score_:.4f}")

    # Evaluate on val & test
    val_metrics,  val_pred,  val_proba  = evaluate_model(best, X_val, y_val, "val")
    test_metrics, test_pred, test_proba = evaluate_model(best, X_test, y_test, "test")

    all_metrics = {**val_metrics, **test_metrics, "cv_best_f1": grid.best_score_}

    for k, v in all_metrics.items():
        log.info(f"  {k}: {v:.4f}")

    # Plots
    cm_val_path  = plot_confusion_matrix(y_val, val_pred, name, "val")
    cm_test_path = plot_confusion_matrix(y_test, test_pred, name, "test")
    roc_val_path  = plot_roc_curve(best, X_val, y_val, name, "val")
    roc_test_path = plot_roc_curve(best, X_test, y_test, name, "test")

    # MLflow logging
    with mlflow.start_run(run_name=name):
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("model_type", name)
        mlflow.log_metrics(all_metrics)

        # Log plots as artifacts
        mlflow.log_artifact(cm_val_path,  "plots")
        mlflow.log_artifact(cm_test_path, "plots")
        mlflow.log_artifact(roc_val_path,  "plots")
        mlflow.log_artifact(roc_test_path, "plots")

        # Log model
        mlflow.sklearn.log_model(
            best, artifact_path="model",
            registered_model_name=f"wine_quality_{name}",
        )

        # Classification report as text artifact
        report = classification_report(
            y_test, test_pred, target_names=["Low Quality", "High Quality"]
        )
        report_path = os.path.join(OUTPUT_DIR, f"report_{name}.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path, "reports")

        run_id = mlflow.active_run().info.run_id

    return {
        "name": name,
        "run_id": run_id,
        "best_params": grid.best_params_,
        "metrics": all_metrics,
        "model": best,
    }


# ===================================================================== #
#  6. MODEL COMPARISON PLOT                                              #
# ===================================================================== #
def plot_model_comparison(results: list):
    """Bar chart comparing all models across key metrics."""
    metrics_list = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    model_names  = [r["name"] for r in results]
    data = {m: [r["metrics"].get(m, 0) for r in results] for m in metrics_list}

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    for i, metric in enumerate(metrics_list):
        ax.bar(x + i * width, data[metric], width, label=metric.replace("test_", ""),
               color=colors[i])

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(path, "plots")
    return path


# ===================================================================== #
#  7. MAIN ORCHESTRATOR                                                  #
# ===================================================================== #
def main():
    parser = argparse.ArgumentParser(description="Wine Quality ML Pipeline")
    parser.add_argument(
        "--dagshub-repo", type=str, default=None,
        help="DagsHub repo in format 'username/repo_name'. "
             "If provided, enables remote MLflow tracking.",
    )
    parser.add_argument(
        "--experiment-name", type=str,
        default="wine-quality-binary-classification",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Configure MLflow ─────────────────────────────────────────────
    if args.dagshub_repo and DAGSHUB_AVAILABLE:
        dagshub.init(
            repo_owner=args.dagshub_repo.split("/")[0],
            repo_name=args.dagshub_repo.split("/")[1],
            mlflow=True,
        )
        log.info(f"MLflow tracking URI → {mlflow.get_tracking_uri()}")
    else:
        mlflow.set_tracking_uri(os.path.join(BASE_DIR, "mlruns"))
        log.info("Using local MLflow tracking")

    mlflow.set_experiment(args.experiment_name)

    # ── Step 1 — Load data ──────────────────────────────────────────
    df = load_data()

    # ── Step 2 — EDA ────────────────────────────────────────────────
    eda_summary = run_eda(df)
    eda_path = os.path.join(OUTPUT_DIR, "eda_summary.json")
    with open(eda_path, "w") as f:
        json.dump(eda_summary, f, indent=2, default=str)
    log.info(f"EDA summary saved to {eda_path}")

    # ── Step 3 — Preprocess ─────────────────────────────────────────
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, feature_cols) = preprocess(df)

    # Save scaler & feature columns for serving
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    features_path = os.path.join(OUTPUT_DIR, "feature_cols.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f)
    log.info(f"Scaler saved to {scaler_path}")

    # ── Step 4 — Train & evaluate each model ────────────────────────
    model_configs = get_model_configs()
    results = []
    for name, estimator, param_grid in model_configs:
        result = train_and_evaluate(
            name, estimator, param_grid,
            X_train, y_train, X_val, y_val, X_test, y_test,
        )
        results.append(result)

    # ── Step 5 — Compare & select best model ────────────────────────
    with mlflow.start_run(run_name="model_comparison"):
        comparison_path = plot_model_comparison(results)

    best_result = max(results, key=lambda r: r["metrics"]["test_f1"])
    log.info(f"\n{'='*60}")
    log.info(f"  BEST MODEL: {best_result['name']}")
    log.info(f"  Test F1:    {best_result['metrics']['test_f1']:.4f}")
    log.info(f"  Run ID:     {best_result['run_id']}")
    log.info(f"{'='*60}\n")

    # Save best model locally for Docker serving
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    joblib.dump(best_result["model"], os.path.join(best_model_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(best_model_dir, "scaler.joblib"))
    with open(os.path.join(best_model_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(best_model_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_name": best_result["name"],
            "run_id": best_result["run_id"],
            "best_params": best_result["best_params"],
            "test_metrics": {k: round(v, 4)
                             for k, v in best_result["metrics"].items()
                             if k.startswith("test_")},
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    log.info(f"Best model saved to {best_model_dir}")

    # Summary
    summary = {
        "best_model": best_result["name"],
        "best_run_id": best_result["run_id"],
        "results": [
            {"name": r["name"], "test_f1": r["metrics"]["test_f1"],
             "test_roc_auc": r["metrics"].get("test_roc_auc", 0)}
            for r in results
        ],
    }
    summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Pipeline complete ✓")
    return results


if __name__ == "__main__":
    main()
