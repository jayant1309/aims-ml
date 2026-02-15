#!/usr/bin/env python3
from __future__ import annotations

from aims_ml.config import load_yaml
from aims_ml.dataset import generate_dataset


def main() -> None:
    cfg = load_yaml("configs/dataset.yaml")
    df = generate_dataset(
        n_runs=int(cfg.get("n_runs", 500)),
        output_csv=str(cfg.get("output_csv", "data/sim_runs.csv")),
        seed=int(cfg.get("seed", 42)),
        materials=tuple(cfg.get("materials", ["Al", "Ti", "Fe", "Cu"])),
    )
    print(f"Wrote {len(df)} rows")


if __name__ == "__main__":
    main()
