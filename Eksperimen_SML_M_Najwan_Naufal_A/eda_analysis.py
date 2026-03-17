"""
EDA Script — Wine Quality Dataset
Generates comprehensive visualizations and statistics.
Can be run standalone or called from modular_pipeline.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "output", "plots")


def load_and_merge():
    red   = pd.read_csv(os.path.join(DATA_DIR, "winequality-red.csv"), sep=";")
    white = pd.read_csv(os.path.join(DATA_DIR, "winequality-white.csv"), sep=";")
    red["wine_type"]   = 0
    white["wine_type"] = 1
    return pd.concat([red, white], ignore_index=True)


def generate_eda():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_and_merge()
    df["is_high_quality"] = (df["quality"] >= 7).astype(int)

    print("=" * 60)
    print("  WINE QUALITY DATASET — EDA REPORT")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nClass Distribution (quality):")
    print(df["quality"].value_counts().sort_index().to_string())
    print(f"\nBinary Target Distribution:")
    print(df["is_high_quality"].value_counts().to_string())
    print(f"\nDescriptive Statistics:")
    print(df.describe().T.to_string())

    # --- Plot 1: Quality distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.countplot(x="quality", data=df, hue="quality", palette="viridis",
                  ax=axes[0], legend=False)
    axes[0].set_title("Wine Quality Score Distribution", fontsize=13, fontweight="bold")

    counts = df["is_high_quality"].value_counts()
    colors = ["#e74c3c", "#2ecc71"]
    axes[1].bar(["Low (< 7)", "High (≥ 7)"], [counts[0], counts[1]], color=colors)
    axes[1].set_title("Binary Target Distribution", fontsize=13, fontweight="bold")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[1].text(i, v + 50, str(v), ha="center", fontweight="bold")
    pct = counts[1] / len(df) * 100
    axes[1].set_ylabel("Count")
    axes[1].annotate(f"High quality: {pct:.1f}%", xy=(0.5, 0.85),
                     xycoords="axes fraction", ha="center", fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0"))

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_target_distributions.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Feature distributions by quality class ---
    features = [c for c in numeric_cols if c not in ("quality", "wine_type", "is_high_quality")]
    n = len(features)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten()
    for i, feat in enumerate(features):
        sns.violinplot(x="is_high_quality", y=feat, data=df,
                       hue="is_high_quality", palette=colors,
                       ax=axes_flat[i], legend=False, inner="box")
        axes_flat[i].set_title(feat, fontsize=11, fontweight="bold")
        axes_flat[i].set_xlabel("High Quality")
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Feature Distributions: Low vs High Quality",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "03_feature_violinplots.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: Pairplot (top features) ---
    top_features = ["alcohol", "volatile acidity", "sulphates", "citric acid",
                    "is_high_quality"]
    fig = sns.pairplot(df[top_features], hue="is_high_quality",
                       palette=colors, diag_kind="kde",
                       plot_kws={"alpha": 0.4, "s": 20})
    fig.figure.suptitle("Pairplot of Top Features", y=1.02, fontsize=14,
                        fontweight="bold")
    fig.savefig(os.path.join(PLOTS_DIR, "04_pairplot_top_features.png"), dpi=150)
    plt.close(fig.figure)

    # --- Plot 5: Wine type comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    type_quality = pd.crosstab(df["quality"], df["wine_type"])
    type_quality.columns = ["Red", "White"]
    type_quality.plot(kind="bar", stacked=True, color=["#c0392b", "#f1c40f"],
                      ax=axes[0])
    axes[0].set_title("Quality by Wine Type", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Quality Score")

    type_binary = pd.crosstab(df["wine_type"].map({0: "Red", 1: "White"}),
                              df["is_high_quality"].map({0: "Low", 1: "High"}),
                              normalize="index") * 100
    type_binary.plot(kind="bar", color=colors, ax=axes[1])
    axes[1].set_title("% High Quality by Wine Type", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Percentage")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "05_wine_type_analysis.png"), dpi=150)
    plt.close(fig)

    print(f"\n✅  EDA plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    generate_eda()
