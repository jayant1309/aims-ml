from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .simulation import SimConfig, simulate_cascade


def sample_configs(
    n_runs: int,
    seed: int = 42,
    materials: tuple[str, ...] = ("Al", "Ti", "Fe", "Cu"),
) -> list[SimConfig]:
    rng = np.random.default_rng(seed)
    configs: list[SimConfig] = []

    for idx in range(n_runs):
        material = str(rng.choice(materials))
        config = SimConfig(
            material=material,
            supercell_x=int(rng.integers(8, 13)),
            supercell_y=int(rng.integers(8, 13)),
            supercell_z=int(rng.integers(15, 26)),
            initial_energy_ev=float(rng.uniform(10_000, 100_000)),
            collision_probability=float(rng.uniform(0.15, 0.6)),
            interaction_radius_a=float(rng.uniform(1.2, 3.5)),
            step_size_a=float(rng.uniform(0.2, 1.0)),
            stopping_factor=float(rng.uniform(0.90, 0.99)),
            max_total_particles=int(rng.integers(100, 501)),
            max_steps_per_particle=int(rng.integers(250, 701)),
            incidence_theta_deg=float(rng.uniform(0.0, 40.0)),
            incidence_phi_deg=float(rng.uniform(0.0, 360.0)),
            projectile_mass_amu=float(rng.uniform(1.0, 4.0)),
            random_seed=int(seed * 10_000 + idx),
        )
        configs.append(config)

    return configs


def _flatten_result(result: dict) -> dict:
    row = {}
    config = result["config"]
    material = result["material"]
    stats = result["stats"]

    row.update({f"cfg_{k}": v for k, v in config.items()})
    row.update({f"mat_{k}": v for k, v in material.items()})
    row.update({f"target_{k}": v for k, v in stats.items()})

    hist = result["damage_hist"]
    for i, count in enumerate(hist):
        row[f"target_damage_bin_{i}"] = int(count)

    return row


def generate_dataset(
    n_runs: int,
    output_csv: str | None = None,
    seed: int = 42,
    materials: tuple[str, ...] = ("Al", "Ti", "Fe", "Cu"),
) -> pd.DataFrame:
    rows = []
    for config in sample_configs(n_runs=n_runs, seed=seed, materials=materials):
        result = simulate_cascade(config)
        rows.append(_flatten_result(result))

    df = pd.DataFrame(rows)
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return df
