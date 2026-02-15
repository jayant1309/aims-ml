from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_damage_histogram(result: dict, out_path: str | None = None) -> None:
    displaced = result["displaced_atoms"]
    if len(displaced) == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.hist(displaced[:, 2], bins=30)
    plt.xlabel("Depth (A)")
    plt.ylabel("Displacements")
    plt.title("Damage Distribution vs Depth")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_energy_depth(result: dict, out_path: str | None = None) -> None:
    energy_history = result["energy_history"]
    if len(energy_history) == 0:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(energy_history[:, 0], energy_history[:, 1], linewidth=1)
    plt.xlabel("Depth (A)")
    plt.ylabel("Energy (eV)")
    plt.title("Projectile Energy Decay")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_model_comparison(metrics_df: pd.DataFrame, out_path: str | None = None) -> None:
    if metrics_df.empty:
        return

    pivot = metrics_df.pivot(index="model", columns="target", values="rmse")
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("RMSE")
    ax.set_title("Model Comparison by Target")
    ax.legend(title="Target")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()
