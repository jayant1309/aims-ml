#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation feature importance for one target")
    parser.add_argument("--dataset", type=str, default="data/sim_runs.csv")
    parser.add_argument("--target", type=str, default="target_total_displacements")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    feature_cols = [
        c for c in df.columns if c.startswith("cfg_") or c.startswith("mat_")
    ]
    feature_cols = [c for c in feature_cols if c not in {"cfg_random_seed", "cfg_material", "mat_name", "mat_crystal_type"}]

    x = df[feature_cols].copy()
    y = pd.to_numeric(df[args.target], errors="coerce")

    valid = y.notna()
    x = x.loc[valid]
    y = y.loc[valid]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    perm = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42)
    order = np.argsort(perm.importances_mean)[::-1]

    top_k = min(args.top_k, len(order))
    print(f"Top {top_k} features for {args.target}:")
    for idx in order[:top_k]:
        print(f"{x.columns[idx]}: {perm.importances_mean[idx]:.6f}")


if __name__ == "__main__":
    main()
