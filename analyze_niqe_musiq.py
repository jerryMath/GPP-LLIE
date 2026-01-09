#!/usr/bin/env python3
"""
Combine 01–10.JPG_comparison.csv and compute average NIQE/MUSIQ improvements.

Expected columns (from your CSVs):
- niqe_improvement
- musiq_improvement
Optionally:
- without_bo_niqe, with_bo_niqe, without_bo_musiq, with_bo_musiq
"""

from __future__ import annotations

import glob
import os
from typing import Tuple
import pandas as pd


def load_all_csvs(pattern: str = "/mnt/data/*JPG_comparison.csv") -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    # Basic sanity checks
    required = ["niqe_improvement", "musiq_improvement"]
    missing = [c for c in required if c not in all_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(all_df.columns)}")

    return all_df


def mean_std(series: pd.Series) -> Tuple[float, float, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()), float(s.std(ddof=1)), int(s.shape[0])


def main(input_path, output_path):
    all_df = load_all_csvs(input_path)
    print(f"=== all_df: \n{all_df}")

    # Keep only rows where NIQE is improved by BO
    # (strictly lower NIQE after BO)
    good = all_df[all_df["with_bo_niqe"] < all_df["without_bo_niqe"]].copy()
    print(f"{len(good)} NIQE-improved samples found.")

    if len(good) == 0:
        print("No NIQE-improved samples found. Nothing to average.")
        return

    # Averages on filtered subset
    avg_niqe_imp = good["niqe_improvement"].mean()
    std_niqe_imp = good["niqe_improvement"].std()

    avg_musiq_imp = good["musiq_improvement"].mean()
    std_musiq_imp = good["musiq_improvement"].std()

    print("\n=== Averages over NIQE-improved subset only ===")
    print(f"Avg NIQE improvement (mean ± std):  {avg_niqe_imp:.6f} ± {std_niqe_imp:.6f}")
    print(f"Avg MUSIQ improvement (mean ± std): {avg_musiq_imp:.6f} ± {std_musiq_imp:.6f}")

    all_df['avg_niqe_imp'] = avg_niqe_imp
    all_df['std_niqe_imp'] = std_niqe_imp
    all_df['avg_musiq_imp'] = avg_musiq_imp
    all_df['std_musiq_imp'] = std_musiq_imp

    # --- Optional: save combined file ---
    all_df.to_csv(output_path, index=False)
    print(f"\nSaved combined CSV to: {output_path}")


if __name__ == "__main__":
    # input_path = "./benchmark-BO/dataset/DICM/outputs_bo_v1_weights_3/*_comparison.csv"
    # output_path = "./benchmark-BO/dataset/DICM/outputs_bo_v1_weights_3/combined_comparisons.csv"

    input_path = "./dataset/DICM/outputs_bo_v2_weights_12/*_comparison.csv"
    output_path = "./dataset/DICM/outputs_bo_v2_weights_12/combined_comparisons.csv"
    main(input_path, output_path)
