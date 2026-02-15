from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
    feature_cols = [c for c in df.columns if c.startswith("cfg_") or c.startswith("mat_")]
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


def _prepare_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    numeric_cols, categorical_cols = _feature_columns(df)
    x_cols = numeric_cols + categorical_cols

    clean_df = df[x_cols + [target_col]].copy()
    clean_df[target_col] = pd.to_numeric(clean_df[target_col], errors="coerce")
    clean_df = clean_df.dropna(subset=[target_col])

    x = clean_df[x_cols]
    y = clean_df[target_col]
    return x, y, x_cols, numeric_cols, categorical_cols


def _rf_prediction_std(pipeline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    model = pipeline.named_steps["model"]
    if not isinstance(model, RandomForestRegressor):
        return np.full(len(x), np.nan)

    transformed = pipeline.named_steps["preprocessor"].transform(x)
    tree_preds = np.vstack([tree.predict(transformed) for tree in model.estimators_])
    return tree_preds.std(axis=0)


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    artifacts_dir: str = "artifacts",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    x, y, _, numeric_cols, categorical_cols = _prepare_xy(df, target_col)

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

        pred_std = _rf_prediction_std(pipeline, x_test)

        metrics_rows.append(
            {
                "target": target_col,
                "model": name,
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(root_mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "pred_uncertainty_mean": float(np.nanmean(pred_std)),
                "pred_uncertainty_p90": float(np.nanpercentile(pred_std, 90)),
            }
        )

        model_path = out_dir / f"{target_col}_{name}.joblib"
        joblib.dump(pipeline, model_path)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="rmse", ascending=True)
    metrics_df.to_csv(out_dir / f"metrics_{target_col}.csv", index=False)
    return metrics_df


def evaluate_ood_by_material(
    df: pd.DataFrame,
    target_col: str,
    artifacts_dir: str = "artifacts",
) -> pd.DataFrame:
    if "cfg_material" not in df.columns:
        raise ValueError("`cfg_material` column is required for OOD material evaluation")

    x, y, _, numeric_cols, categorical_cols = _prepare_xy(df, target_col)
    material = df.loc[x.index, "cfg_material"].astype(str)

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    models = _make_models()

    for holdout in sorted(material.unique()):
        test_mask = material == holdout
        train_mask = ~test_mask

        x_train = x.loc[train_mask]
        y_train = y.loc[train_mask]
        x_test = x.loc[test_mask]
        y_test = y.loc[test_mask]

        if len(x_train) < 20 or len(x_test) < 5:
            continue

        for name, model in models.items():
            pipeline = _build_pipeline(model, numeric_cols, categorical_cols)
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            pred_std = _rf_prediction_std(pipeline, x_test)

            rows.append(
                {
                    "target": target_col,
                    "holdout_material": holdout,
                    "model": name,
                    "n_train": int(len(x_train)),
                    "n_test": int(len(x_test)),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "rmse": float(root_mean_squared_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred)),
                    "pred_uncertainty_mean": float(np.nanmean(pred_std)),
                    "pred_uncertainty_p90": float(np.nanpercentile(pred_std, 90)),
                }
            )

    ood_df = pd.DataFrame(rows).sort_values(by=["holdout_material", "rmse"], ascending=True)
    ood_df.to_csv(out_dir / f"metrics_ood_{target_col}.csv", index=False)
    return ood_df


def compute_permutation_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    artifacts_dir: str = "artifacts",
    model_name: str = "random_forest",
    n_repeats: int = 10,
    random_seed: int = 42,
    top_k: int = 20,
) -> pd.DataFrame:
    x, y, x_cols, numeric_cols, categorical_cols = _prepare_xy(df, target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_seed
    )

    models = _make_models()
    if model_name not in models:
        raise ValueError(f"Unsupported model_name '{model_name}'. Choose from {list(models)}")

    pipeline = _build_pipeline(models[model_name], numeric_cols, categorical_cols)
    pipeline.fit(x_train, y_train)

    perm = permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "target": target_col,
            "feature": x_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)

    if top_k > 0:
        importance_df = importance_df.head(top_k)

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_dir / f"feature_importance_{target_col}_{model_name}.csv", index=False)
    return importance_df


def train_all_targets(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> pd.DataFrame:
    parts = [train_and_evaluate(df, target_col=t, artifacts_dir=artifacts_dir) for t in TARGET_COLUMNS]
    all_metrics = pd.concat(parts, ignore_index=True)
    all_metrics.to_csv(Path(artifacts_dir) / "metrics_all_targets.csv", index=False)
    return all_metrics


def run_full_ml_diagnostics(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> dict[str, pd.DataFrame]:
    metrics = train_all_targets(df, artifacts_dir=artifacts_dir)

    ood_parts = [evaluate_ood_by_material(df, target_col=t, artifacts_dir=artifacts_dir) for t in TARGET_COLUMNS]
    ood_metrics = pd.concat(ood_parts, ignore_index=True)
    ood_metrics.to_csv(Path(artifacts_dir) / "metrics_ood_all_targets.csv", index=False)

    fi_parts = [
        compute_permutation_feature_importance(
            df,
            target_col=t,
            artifacts_dir=artifacts_dir,
            model_name="random_forest",
            n_repeats=10,
            top_k=20,
        )
        for t in TARGET_COLUMNS
    ]
    feature_importance = pd.concat(fi_parts, ignore_index=True)
    feature_importance.to_csv(Path(artifacts_dir) / "feature_importance_all_targets.csv", index=False)

    return {
        "metrics": metrics,
        "ood_metrics": ood_metrics,
        "feature_importance": feature_importance,
    }
