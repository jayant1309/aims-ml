import numpy as np

from aims_ml.visualization import plot_cascade_3d_animated


def test_animated_plot_fallback_when_no_displacements() -> None:
    result = {
        "atomic_coords": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "projectile_paths": [np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])],
        "displacement_events": [],
    }

    fig = plot_cascade_3d_animated(result, sample_atoms=10)
    assert "No displacement events" in fig.layout.title.text
