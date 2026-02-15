#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from aims_ml.ml import (
    TARGET_COLUMNS,
    compute_permutation_feature_importance,
    evaluate_ood_by_material,
    run_full_ml_diagnostics,
    train_and_evaluate,
)
from aims_ml.visualization import plot_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train surrogate ML models")
    parser.add_argument("--dataset", type=str, default="data/sim_runs.csv", help="Input CSV path")
    parser.add_argument("--target", type=str, default="all", help=f"Target column or 'all' ({TARGET_COLUMNS})")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    if args.target == "all":
        outputs = run_full_ml_diagnostics(df, artifacts_dir=args.artifacts_dir)
        metrics = outputs["metrics"]
        ood_metrics = outputs["ood_metrics"]
        feature_importance = outputs["feature_importance"]
        plot_model_comparison(metrics, out_path=f"{args.artifacts_dir}/model_comparison_rmse.png")

        print("\nIn-distribution metrics:")
        print(metrics.to_string(index=False))
        print("\nOOD by material metrics:")
        print(ood_metrics.to_string(index=False))
        print("\nTop feature importances:")
        print(feature_importance.groupby("target").head(5).to_string(index=False))
    else:
        metrics = train_and_evaluate(df, target_col=args.target, artifacts_dir=args.artifacts_dir)
        ood_metrics = evaluate_ood_by_material(df, target_col=args.target, artifacts_dir=args.artifacts_dir)
        feature_importance = compute_permutation_feature_importance(
            df,
            target_col=args.target,
            artifacts_dir=args.artifacts_dir,
            model_name="random_forest",
            n_repeats=10,
            top_k=20,
        )

        print("\nIn-distribution metrics:")
        print(metrics.to_string(index=False))
        print("\nOOD by material metrics:")
        print(ood_metrics.to_string(index=False))
        print("\nTop feature importances:")
        print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
