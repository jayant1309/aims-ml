#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from aims_ml.config import load_yaml
from aims_ml.ml import train_all_targets, train_and_evaluate
from aims_ml.visualization import plot_model_comparison


def main() -> None:
    cfg = load_yaml("configs/training.yaml")
    dataset_csv = str(cfg.get("dataset_csv", "data/sim_runs.csv"))
    artifacts_dir = str(cfg.get("artifacts_dir", "artifacts"))
    target = str(cfg.get("target", "all"))

    df = pd.read_csv(dataset_csv)

    if target == "all":
        metrics = train_all_targets(df, artifacts_dir=artifacts_dir)
        plot_model_comparison(metrics, out_path=f"{artifacts_dir}/model_comparison_rmse.png")
    else:
        metrics = train_and_evaluate(df, target_col=target, artifacts_dir=artifacts_dir)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
