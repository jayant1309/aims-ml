#!/usr/bin/env python3
from __future__ import annotations

import argparse

from aims_ml.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate radiation cascade dataset")
    parser.add_argument("--n-runs", type=int, default=500, help="Number of simulation runs")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--out", type=str, default="data/sim_runs.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(n_runs=args.n_runs, output_csv=args.out, seed=args.seed)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
