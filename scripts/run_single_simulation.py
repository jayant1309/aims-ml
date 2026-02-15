#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from aims_ml.simulation import run_single_simulation
from aims_ml.visualization import plot_damage_histogram, plot_energy_depth


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one cascade simulation and save summary plots")
    parser.add_argument("--material", type=str, default="Al")
    parser.add_argument("--initial-energy-ev", type=float, default=40000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots-dir", type=str, default="artifacts")
    args = parser.parse_args()

    result = run_single_simulation(
        material=args.material,
        initial_energy_ev=args.initial_energy_ev,
        random_seed=args.seed,
    )

    stats = result["stats"]
    print(json.dumps(stats, indent=2))

    plot_damage_histogram(result, out_path=f"{args.plots_dir}/damage_hist.png")
    plot_energy_depth(result, out_path=f"{args.plots_dir}/energy_depth.png")


if __name__ == "__main__":
    main()
