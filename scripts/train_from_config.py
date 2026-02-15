#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from aims_ml.config import load_yaml
from aims_ml.ml import (
    compute_permutation_feature_importance,
    evaluate_ood_by_material,
    run_full_ml_diagnostics,
    train_and_evaluate,
)
from aims_ml.visualization import plot_model_comparison


def main() -> None:
    cfg = load_yaml("configs/training.yaml")
    dataset_csv = str(cfg.get("dataset_csv", "data/sim_runs.csv"))
    artifacts_dir = str(cfg.get("artifacts_dir", "artifacts"))
    target = str(cfg.get("target", "all"))

    df = pd.read_csv(dataset_csv)

    if target == "all":
        outputs = run_full_ml_diagnostics(df, artifacts_dir=artifacts_dir)
        metrics = outputs["metrics"]
        plot_model_comparison(metrics, out_path=f"{artifacts_dir}/model_comparison_rmse.png")
        print("In-distribution metrics:\n", metrics.to_string(index=False))
        print("\nOOD by material metrics:\n", outputs["ood_metrics"].to_string(index=False))
        print("\nTop feature importances:\n", outputs["feature_importance"].groupby("target").head(5).to_string(index=False))
    else:
        metrics = train_and_evaluate(df, target_col=target, artifacts_dir=artifacts_dir)
        ood_metrics = evaluate_ood_by_material(df, target_col=target, artifacts_dir=artifacts_dir)
        fi = compute_permutation_feature_importance(
            df,
            target_col=target,
            artifacts_dir=artifacts_dir,
            model_name="random_forest",
            n_repeats=10,
            top_k=20,
        )
        print("In-distribution metrics:\n", metrics.to_string(index=False))
        print("\nOOD by material metrics:\n", ood_metrics.to_string(index=False))
        print("\nTop feature importances:\n", fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
