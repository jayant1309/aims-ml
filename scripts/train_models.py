#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from aims_ml.ml import TARGET_COLUMNS, train_all_targets, train_and_evaluate
from aims_ml.visualization import plot_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train surrogate ML models")
    parser.add_argument("--dataset", type=str, default="data/sim_runs.csv", help="Input CSV path")
    parser.add_argument("--target", type=str, default="all", help=f"Target column or 'all' ({TARGET_COLUMNS})")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    if args.target == "all":
        metrics = train_all_targets(df, artifacts_dir=args.artifacts_dir)
        plot_model_comparison(metrics, out_path=f"{args.artifacts_dir}/model_comparison_rmse.png")
    else:
        metrics = train_and_evaluate(df, target_col=args.target, artifacts_dir=args.artifacts_dir)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
