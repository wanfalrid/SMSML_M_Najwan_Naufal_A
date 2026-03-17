"""
=============================================================================
  Download & Combine Wine Quality Dataset
  Source: UCI Machine Learning Repository
  Student: M_Najwan_Naufal_A
=============================================================================

Steps:
  1. Download winequality-red.csv   (1599 samples)
  2. Download winequality-white.csv (4898 samples)
  3. Combine with 'wine_type' column → winequality_combined.csv
"""

import os
import sys
import urllib.request
import pandas as pd

# ── URLs ─────────────────────────────────────────────────────────────
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
FILES = {
    "winequality-red.csv":   f"{BASE_URL}/winequality-red.csv",
    "winequality-white.csv": f"{BASE_URL}/winequality-white.csv",
}

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "winequality_raw")


def download_data():
    """Download wine quality datasets if they don't already exist."""
    os.makedirs(RAW_DIR, exist_ok=True)

    for filename, url in FILES.items():
        filepath = os.path.join(RAW_DIR, filename)
        if os.path.exists(filepath):
            print(f"[SKIP]     {filename} already exists")
            continue
        print(f"[DOWNLOAD] {filename} from {url} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"[OK]       Saved to {filepath}")
        except Exception as e:
            print(f"[ERROR]    Failed to download {filename}: {e}")
            print(f"[INFO]     Please download manually from:\n           {url}")
            print(f"[INFO]     and place it in:\n           {RAW_DIR}")
            sys.exit(1)

    print(f"\n✅  All files downloaded to {RAW_DIR}")


def combine_data():
    """Combine red and white wine CSVs into a single dataset with wine_type."""
    red_path   = os.path.join(RAW_DIR, "winequality-red.csv")
    white_path = os.path.join(RAW_DIR, "winequality-white.csv")

    if not os.path.exists(red_path) or not os.path.exists(white_path):
        print("[ERROR] CSV files not found — run download first")
        sys.exit(1)

    # Load CSVs (UCI uses semicolon delimiter)
    red   = pd.read_csv(red_path, sep=";")
    white = pd.read_csv(white_path, sep=";")

    print(f"[LOAD]     Red wine:   {red.shape[0]} rows, {red.shape[1]} columns")
    print(f"[LOAD]     White wine: {white.shape[0]} rows, {white.shape[1]} columns")

    # Add wine_type column: 0 = red, 1 = white
    red["wine_type"]   = 0
    white["wine_type"] = 1

    # Combine
    combined = pd.concat([red, white], ignore_index=True)

    # Save
    output_path = os.path.join(RAW_DIR, "winequality_combined.csv")
    combined.to_csv(output_path, index=False)

    print(f"\n[COMBINE]  Combined dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")
    print(f"[SAVE]     Saved to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples:     {len(combined)}")
    print(f"  Red wine:          {len(red)}")
    print(f"  White wine:        {len(white)}")
    print(f"  Features:          {combined.shape[1] - 1}")
    print(f"  Missing values:    {combined.isnull().sum().sum()}")
    print(f"\n  Quality distribution:")
    for q in sorted(combined["quality"].unique()):
        count = (combined["quality"] == q).sum()
        pct = count / len(combined) * 100
        bar = "█" * int(pct)
        print(f"    Score {q}: {count:>5}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Binary target (quality >= 7):")
    high = (combined["quality"] >= 7).sum()
    low  = (combined["quality"] < 7).sum()
    print(f"    High quality (1): {high:>5}  ({high/len(combined)*100:.1f}%)")
    print(f"    Low quality  (0): {low:>5}  ({low/len(combined)*100:.1f}%)")
    print(f"{'='*60}\n")

    return combined


if __name__ == "__main__":
    print("=" * 60)
    print("  Wine Quality Dataset — Download & Combine")
    print("=" * 60)
    print()

    download_data()
    print()
    combine_data()

    print("✅  Dataset ready!\n")
