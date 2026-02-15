from aims_ml.dataset import generate_dataset
from aims_ml.ml import run_full_ml_diagnostics, train_and_evaluate


def test_dataset_and_training_smoke(tmp_path) -> None:
    out_csv = tmp_path / "sim.csv"
    df = generate_dataset(n_runs=12, output_csv=str(out_csv), seed=11, materials=("Al", "Fe"))

    assert len(df) == 12
    assert out_csv.exists()
    assert "target_total_displacements" in df.columns

    metrics = train_and_evaluate(
        df,
        target_col="target_total_displacements",
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    assert len(metrics) >= 1
    assert {"model", "mae", "rmse", "r2"}.issubset(metrics.columns)


def test_full_diagnostics_smoke(tmp_path) -> None:
    df = generate_dataset(n_runs=30, seed=12, materials=("Al", "Fe", "Cu"))
    outputs = run_full_ml_diagnostics(df, artifacts_dir=str(tmp_path / "artifacts"))

    assert {"metrics", "ood_metrics", "feature_importance"}.issubset(outputs.keys())
    assert not outputs["metrics"].empty
    assert not outputs["feature_importance"].empty
