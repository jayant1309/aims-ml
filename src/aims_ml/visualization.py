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


def plot_cascade_3d_animated(
    result: dict,
    sample_atoms: int = 4000,
    stride: int = 1,
    frame_duration_ms: int = 80,
    show_projectile_paths: bool = False,
    out_html: str | None = None,
) -> go.Figure:
    atomic_coords = result["atomic_coords"]
    events = result.get("displacement_events", [])
    paths = [path for path in result["projectile_paths"] if len(path) > 0]

    if stride < 1:
        raise ValueError("stride must be >= 1")

    if len(atomic_coords) > sample_atoms:
        idx = np.random.default_rng(42).choice(len(atomic_coords), size=sample_atoms, replace=False)
        lattice_points = atomic_coords[idx]
    else:
        lattice_points = atomic_coords

    if not events:
        fig = go.Figure()
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
        for i, path in enumerate(paths):
            fig.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines",
                    name=f"Projectile path {i + 1}",
                    line={"width": 2, "color": "#457b9d"},
                    opacity=0.55,
                )
            )
        fig.update_layout(
            title="3D Radiation Damage Animation (No displacement events in this run)",
            scene={
                "xaxis_title": "X (A)",
                "yaxis_title": "Y (A)",
                "zaxis_title": "Z (A)",
            },
            template="plotly_white",
            width=950,
            height=700,
        )
        if out_html:
            Path(out_html).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(out_html, include_plotlyjs="cdn")
        return fig

    base_traces = [
        go.Scatter3d(
            x=lattice_points[:, 0],
            y=lattice_points[:, 1],
            z=lattice_points[:, 2],
            mode="markers",
            name="Lattice (sampled)",
            marker={"size": 2, "color": "#7a7a7a", "opacity": 0.12},
            hoverinfo="skip",
        )
    ]

    max_step = max(int(e["event_step"]) for e in events)
    steps = list(range(1, max_step + 1, stride))
    if steps[-1] != max_step:
        steps.append(max_step)

    def _event_arrays(frame_step: int) -> tuple[np.ndarray, np.ndarray]:
        vacancy_list = []
        displaced_list = []
        for event in events:
            event_step = int(event["event_step"])
            if event_step <= frame_step:
                vacancy_list.append(event["vacancy_pos"])
                frac = min(1.0, max(0.0, (frame_step - event_step + 1) / max(1, stride)))
                start = np.array(event["vacancy_pos"])
                end = np.array(event["interstitial_pos"])
                displaced_list.append(start + frac * (end - start))

        vacancies = np.array(vacancy_list) if vacancy_list else np.empty((0, 3))
        displaced = np.array(displaced_list) if displaced_list else np.empty((0, 3))
        return vacancies, displaced

    init_vacancies, init_displaced = _event_arrays(steps[0])
    init_data = list(base_traces)
    init_data.append(
        go.Scatter3d(
            x=init_vacancies[:, 0],
            y=init_vacancies[:, 1],
            z=init_vacancies[:, 2],
            mode="markers",
            name="Vacancies",
            marker={"size": 4, "color": "#111111", "opacity": 0.95, "symbol": "circle-open"},
        )
    )
    init_data.append(
        go.Scatter3d(
            x=init_displaced[:, 0],
            y=init_displaced[:, 1],
            z=init_displaced[:, 2],
            mode="markers",
            name="Displaced atoms (moving)",
            marker={"size": 5, "color": "#e63946", "opacity": 0.9},
        )
    )

    if show_projectile_paths:
        for i, path in enumerate(paths):
            segment = path[:1]
            init_data.append(
                go.Scatter3d(
                    x=segment[:, 0],
                    y=segment[:, 1],
                    z=segment[:, 2],
                    mode="lines",
                    name=f"Projectile path {i + 1}",
                    line={"width": 2, "color": "#457b9d"},
                    opacity=0.55,
                )
            )

    frames = []
    for step in steps:
        frame_data = list(base_traces)
        frame_vacancies, frame_displaced = _event_arrays(step)
        frame_data.append(
            go.Scatter3d(
                x=frame_vacancies[:, 0],
                y=frame_vacancies[:, 1],
                z=frame_vacancies[:, 2],
                mode="markers",
                name="Vacancies",
                marker={"size": 4, "color": "#111111", "opacity": 0.95, "symbol": "circle-open"},
            )
        )
        frame_data.append(
            go.Scatter3d(
                x=frame_displaced[:, 0],
                y=frame_displaced[:, 1],
                z=frame_displaced[:, 2],
                mode="markers",
                name="Displaced atoms (moving)",
                marker={"size": 5, "color": "#e63946", "opacity": 0.9},
            )
        )

        if show_projectile_paths:
            for i, path in enumerate(paths):
                segment = path[: min(step, len(path))]
                frame_data.append(
                    go.Scatter3d(
                        x=segment[:, 0],
                        y=segment[:, 1],
                        z=segment[:, 2],
                        mode="lines",
                        name=f"Projectile path {i + 1}",
                        line={"width": 2, "color": "#457b9d"},
                        opacity=0.55,
                    )
                )
        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        title="3D Radiation Damage Animation (Vacancies + Displaced Atoms)",
        scene={
            "xaxis_title": "X (A)",
            "yaxis_title": "Y (A)",
            "zaxis_title": "Z (A)",
        },
        template="plotly_white",
        width=950,
        height=700,
        legend={"itemsizing": "constant"},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )

    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")

    return fig


def plot_cascade_3d_explainer(
    result: dict,
    sample_atoms: int = 2500,
    stride: int = 1,
    roi_padding_a: float = 4.0,
    show_paths: bool = False,
    out_html: str | None = None,
) -> go.Figure:
    atomic_coords = result["atomic_coords"]
    events = result.get("displacement_events", [])
    paths = [path for path in result.get("projectile_paths", []) if len(path) > 0]

    if stride < 1:
        raise ValueError("stride must be >= 1")

    if len(atomic_coords) > sample_atoms:
        idx = np.random.default_rng(42).choice(len(atomic_coords), size=sample_atoms, replace=False)
        lattice_points = atomic_coords[idx]
    else:
        lattice_points = atomic_coords

    if not events:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=lattice_points[:, 0],
                y=lattice_points[:, 1],
                z=lattice_points[:, 2],
                mode="markers",
                name="Lattice",
                marker={"size": 2, "color": "#8a8a8a", "opacity": 0.12},
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title="Cascade Explainer (No displacement events in this run)",
            template="plotly_white",
            scene={"xaxis_title": "X (A)", "yaxis_title": "Y (A)", "zaxis_title": "Z (A)"},
            width=1100,
            height=720,
        )
        if out_html:
            Path(out_html).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(out_html, include_plotlyjs="cdn")
        return fig

    max_step = max(int(e["event_step"]) for e in events)
    steps = list(range(1, max_step + 1, stride))
    if steps[-1] != max_step:
        steps.append(max_step)

    def _event_arrays(frame_step: int) -> tuple[np.ndarray, np.ndarray]:
        vacancy_list = []
        displaced_list = []
        for event in events:
            event_step = int(event["event_step"])
            if event_step <= frame_step:
                vacancy_list.append(event["vacancy_pos"])
                frac = min(1.0, max(0.0, (frame_step - event_step + 1) / max(1, stride)))
                start = np.array(event["vacancy_pos"])
                end = np.array(event["interstitial_pos"])
                displaced_list.append(start + frac * (end - start))
        vacancies = np.array(vacancy_list) if vacancy_list else np.empty((0, 3))
        displaced = np.array(displaced_list) if displaced_list else np.empty((0, 3))
        return vacancies, displaced

    def _roi_ranges(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
        if len(points) == 0:
            pmin = lattice_points.min(axis=0)
            pmax = lattice_points.max(axis=0)
        else:
            pmin = points.min(axis=0)
            pmax = points.max(axis=0)
        return (
            [float(pmin[0] - roi_padding_a), float(pmax[0] + roi_padding_a)],
            [float(pmin[1] - roi_padding_a), float(pmax[1] + roi_padding_a)],
            [float(pmin[2] - roi_padding_a), float(pmax[2] + roi_padding_a)],
        )

    def _frame_data(frame_step: int) -> tuple[list, int, tuple[list[float], list[float], list[float]]]:
        vacancies, displaced = _event_arrays(frame_step)
        damage_cloud = displaced if len(displaced) > 0 else vacancies
        xr, yr, zr = _roi_ranges(damage_cloud)

        data = [
            go.Scatter3d(
                x=lattice_points[:, 0],
                y=lattice_points[:, 1],
                z=lattice_points[:, 2],
                mode="markers",
                name="Lattice",
                marker={"size": 2, "color": "#8a8a8a", "opacity": 0.10},
                visible=True,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=vacancies[:, 0],
                y=vacancies[:, 1],
                z=vacancies[:, 2],
                mode="markers",
                name="Vacancies",
                marker={"size": 4, "color": "#111111", "opacity": 0.95, "symbol": "circle-open"},
                visible=True,
            ),
            go.Scatter3d(
                x=displaced[:, 0],
                y=displaced[:, 1],
                z=displaced[:, 2],
                mode="markers",
                name="Displaced atoms",
                marker={"size": 5, "color": "#e63946", "opacity": 0.9},
                visible=True,
            ),
        ]

        if show_paths:
            for i, path in enumerate(paths):
                seg = path[: min(frame_step, len(path))]
                data.append(
                    go.Scatter3d(
                        x=seg[:, 0],
                        y=seg[:, 1],
                        z=seg[:, 2],
                        mode="lines",
                        name=f"Path {i + 1}",
                        line={"width": 2, "color": "#457b9d"},
                        opacity=0.45,
                    )
                )

        return data, len(displaced), (xr, yr, zr)

    init_data, init_count, (init_xr, init_yr, init_zr) = _frame_data(steps[0])

    frames = []
    for step in steps:
        data, count, _ = _frame_data(step)
        frames.append(go.Frame(data=data, name=str(step), layout={"annotations": [dict(
            text=f"Step: {step} | Displacements so far: {count}",
            x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False, bgcolor="white"
        )]}))

    fig = go.Figure(data=init_data, frames=frames)
    fig.update_layout(
        title="3D Cascade Explainer",
        template="plotly_white",
        width=1150,
        height=760,
        scene={
            "xaxis_title": "X (A)",
            "yaxis_title": "Y (A)",
            "zaxis_title": "Z (A)",
            "xaxis": {"range": init_xr},
            "yaxis": {"range": init_yr},
            "zaxis": {"range": init_zr},
        },
        annotations=[
            dict(
                text=f"Step: {steps[0]} | Displacements so far: {init_count}",
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                showarrow=False,
                bgcolor="white",
            )
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 1.12,
                "showactive": True,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.45,
                "y": 1.12,
                "showactive": True,
                "buttons": [
                    {
                        "label": "Isometric",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2}}],
                    },
                    {
                        "label": "Top",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": {"x": 0.01, "y": 0.01, "z": 2.4}}],
                    },
                    {
                        "label": "Side",
                        "method": "relayout",
                        "args": [{"scene.camera.eye": {"x": 2.2, "y": 0.01, "z": 0.2}}],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.76,
                "y": 1.12,
                "showactive": True,
                "buttons": [
                    {
                        "label": "All Layers",
                        "method": "restyle",
                        "args": [{"visible": True}, [0, 1, 2]],
                    },
                    {
                        "label": "Damage Only",
                        "method": "restyle",
                        "args": [{"visible": [False, True, True]}],
                    },
                ],
            },
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.02,
                "y": 0.02,
                "len": 0.96,
                "pad": {"b": 10, "t": 50},
                "steps": [
                    {
                        "label": str(step),
                        "method": "animate",
                        "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for step in steps
                ],
            }
        ],
    )

    if out_html:
        Path(out_html).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn")

    return fig
