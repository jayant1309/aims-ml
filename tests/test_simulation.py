import numpy as np

from aims_ml.simulation import SimConfig, simulate_cascade


def test_simulation_returns_expected_keys() -> None:
    cfg = SimConfig(material="Al", supercell_x=4, supercell_y=4, supercell_z=6, random_seed=123)
    result = simulate_cascade(cfg)

    for key in [
        "config",
        "material",
        "stats",
        "atomic_coords",
        "displaced_atoms",
        "projectile_paths",
        "energy_history",
        "damage_hist",
        "damage_bin_edges",
    ]:
        assert key in result

    assert result["atomic_coords"].shape[1] == 3
    assert result["damage_hist"].shape[0] == 10


def test_simulation_reproducible_with_seed() -> None:
    cfg = SimConfig(material="Fe", supercell_x=4, supercell_y=4, supercell_z=6, random_seed=77)
    r1 = simulate_cascade(cfg)
    r2 = simulate_cascade(cfg)

    assert r1["stats"]["total_displacements"] == r2["stats"]["total_displacements"]
    assert np.array_equal(r1["damage_hist"], r2["damage_hist"])
