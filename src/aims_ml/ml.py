from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_COLUMNS = [
    "target_total_displacements",
    "target_mean_displacement_depth",
    "target_max_displacement_depth",
]


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_cols = [
        c
        for c in df.columns
        if c.startswith("cfg_") or c.startswith("mat_")
    ]
    feature_cols = [c for c in feature_cols if c not in {"cfg_random_seed", "mat_name", "mat_crystal_type"}]

    numeric_cols = [c for c in feature_cols if df[c].dtype != "object"]
    categorical_cols = [c for c in ["cfg_material", "mat_crystal_type"] if c in df.columns]

    return numeric_cols, categorical_cols


def _make_models() -> dict[str, object]:
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=42),
    }


def _build_pipeline(model: object, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    artifacts_dir: str = "artifacts",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    numeric_cols, categorical_cols = _feature_columns(df)
    x_cols = numeric_cols + categorical_cols

    clean_df = df[x_cols + [target_col]].copy()
    clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[target_col])

    x = clean_df[x_cols]
    y = clean_df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_seed
    )

    models = _make_models()
    metrics_rows = []

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        pipeline = _build_pipeline(model, numeric_cols, categorical_cols)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        metrics_rows.append(
            {
                "target": target_col,
                "model": name,
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(root_mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }
        )

        model_path = out_dir / f"{target_col}_{name}.joblib"
        joblib.dump(pipeline, model_path)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="rmse", ascending=True)
    metrics_df.to_csv(out_dir / f"metrics_{target_col}.csv", index=False)
    return metrics_df


def train_all_targets(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> pd.DataFrame:
    parts = [train_and_evaluate(df, target_col=t, artifacts_dir=artifacts_dir) for t in TARGET_COLUMNS]
    all_metrics = pd.concat(parts, ignore_index=True)
    all_metrics.to_csv(Path(artifacts_dir) / "metrics_all_targets.csv", index=False)
    return all_metrics
