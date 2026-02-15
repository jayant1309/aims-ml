# AIMS-ML: Radiation Damage Surrogate Modeling

This project turns a radiation cascade simulation into an ML-focused materials science workflow.

## What This Project Does

- Simulates atomic displacement cascades in crystalline materials (Al, Ti, Fe, Cu).
- Builds a dataset by sweeping irradiation and simulation parameters.
- Trains surrogate ML models to predict damage metrics faster than full simulations.
- Includes interactive 3D cascade visualization for notebook/Colab demos.

## ML Relevance

The simulator is used as a data generator. The ML task is supervised regression:

- Inputs (`X`): material + beam + simulation parameters.
- Targets (`y`):
  - `target_total_displacements`
  - `target_mean_displacement_depth`
  - `target_max_displacement_depth`
  - `target_damage_bin_*` (depth-profile bins)

This matches common materials informatics practice: train a surrogate for expensive simulations.

## Project Structure

- `src/aims_ml/materials.py`: material properties and lattice generation
- `src/aims_ml/simulation.py`: cascade simulation engine
- `src/aims_ml/dataset.py`: parameter sampling and dataset generation
- `src/aims_ml/ml.py`: model training and evaluation
- `src/aims_ml/visualization.py`: plotting helpers
- `scripts/`: executable pipeline scripts
- `configs/`: YAML configs for reproducible runs
- `tests/`: unit and smoke tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart

1. Generate a dataset:

```bash
python scripts/generate_dataset.py --n-runs 500 --out data/sim_runs.csv
```

2. Train surrogate models for all primary targets:

```bash
python scripts/train_models.py --dataset data/sim_runs.csv --target all --artifacts-dir artifacts
```

3. Run one simulation and save plots:

```bash
python scripts/run_single_simulation.py --material Al --initial-energy-ev 40000 --seed 42
```

4. Interactive 3D visualization in notebook/Colab:

```python
from aims_ml.simulation import SimConfig, simulate_cascade
from aims_ml.visualization import plot_cascade_3d_interactive

result = simulate_cascade(SimConfig(material="Al", random_seed=42))
fig = plot_cascade_3d_interactive(result, sample_atoms=6000)
fig.show()
```

## Reproducible Config-Driven Runs

```bash
python scripts/generate_dataset_from_config.py
python scripts/train_from_config.py
```

Edit `configs/dataset.yaml` and `configs/training.yaml` to control experiments.

## Recommended Course Deliverables

- Baseline model comparison table (`artifacts/metrics_all_targets.csv`)
- Error metrics (MAE/RMSE/R2) per target
- Feature importance analysis for the best model
- Discussion of limitations in simulator physics assumptions

## Notes and Limitations

- This is a first-order cascade approximation, not full molecular dynamics.
- Ti is represented with a simple HCP approximation in lattice creation.
- Physics assumptions are intentionally simplified for ML coursework scale.
