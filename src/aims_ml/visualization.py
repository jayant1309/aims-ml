from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


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


def plot_cascade_3d_interactive(
    result: dict,
    sample_atoms: int = 8000,
    show_lattice: bool = True,
    out_html: str | None = None,
) -> go.Figure:
    atomic_coords = result["atomic_coords"]
    displaced = result["displaced_atoms"]
    paths = result["projectile_paths"]

    fig = go.Figure()

    if show_lattice and len(atomic_coords) > 0:
        if len(atomic_coords) > sample_atoms:
            idx = np.random.default_rng(42).choice(len(atomic_coords), size=sample_atoms, replace=False)
            lattice_points = atomic_coords[idx]
        else:
            lattice_points = atomic_coords

        fig.add_trace(
            go.Scatter3d(
                x=lattice_points[:, 0],
                y=lattice_points[:, 1],
                z=lattice_points[:, 2],
                mode="markers",
                name="Lattice (sampled)",
                marker={"size": 2, "color": "#7a7a7a", "opacity": 0.12},
                hoverinfo="skip",
            )
        )

    if len(displaced) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=displaced[:, 0],
                y=displaced[:, 1],
                z=displaced[:, 2],
                mode="markers",
                name="Displaced atoms",
                marker={"size": 4, "color": "#e63946", "opacity": 0.9},
            )
        )

    for i, path in enumerate(paths):
        if len(path) == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode="lines",
                name=f"Particle path {i + 1}",
                line={"width": 3, "color": "#1d3557"},
                opacity=0.8,
            )
        )

    fig.update_layout(
        title="3D Radiation Cascade (Interactive)",
        scene={
            "xaxis_title": "X (A)",
            "yaxis_title": "Y (A)",
            "zaxis_title": "Z (A)",
        },
        template="plotly_white",
        width=950,
        height=700,
        legend={"itemsizing": "constant"},
    )

    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")

    return fig
