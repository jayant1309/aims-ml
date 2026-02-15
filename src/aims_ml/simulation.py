from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import cKDTree

from .materials import generate_lattice


@dataclass
class SimConfig:
    material: str = "Al"
    supercell_x: int = 10
    supercell_y: int = 10
    supercell_z: int = 20
    initial_energy_ev: float = 40000.0
    collision_probability: float = 0.3
    interaction_radius_a: float = 2.0
    step_size_a: float = 0.5
    stopping_factor: float = 0.95
    max_total_particles: int = 300
    max_steps_per_particle: int = 500
    incidence_theta_deg: float = 0.0
    incidence_phi_deg: float = 0.0
    projectile_mass_amu: float = 1.0
    displacement_jump_scale: float = 1.5
    random_seed: int | None = None


def _direction_from_angles(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = -np.cos(theta)
    vec = np.array([x, y, z], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.array([0.0, 0.0, -1.0])


def simulate_cascade(config: SimConfig) -> dict:
    rng = np.random.default_rng(config.random_seed)
    dims = (config.supercell_x, config.supercell_y, config.supercell_z)
    atomic_coords, box_min, box_max, material = generate_lattice(config.material, dims)
    tree = cKDTree(atomic_coords)

    displaced_mask = np.zeros(len(atomic_coords), dtype=bool)
    displaced_atoms: list[np.ndarray] = []
    displaced_final_positions: list[np.ndarray] = []
    vacancy_positions: list[np.ndarray] = []
    displacement_events: list[dict] = []
    projectile_paths: list[np.ndarray] = []
    energy_history: list[tuple[float, float]] = []
    global_step = 0

    center_x = (box_min[0] + box_max[0]) / 2
    center_y = (box_min[1] + box_max[1]) / 2
    start_z = box_max[2] - 1.0

    primary_direction = _direction_from_angles(config.incidence_theta_deg, config.incidence_phi_deg)

    active_particles = [
        {
            "pos": np.array([center_x, center_y, start_z], dtype=float),
            "dir": primary_direction,
            "energy": float(config.initial_energy_ev),
            "mass": float(config.projectile_mass_amu),
        }
    ]

    while active_particles:
        particle = active_particles.pop()
        pos = particle["pos"]
        direction = particle["dir"]
        energy = float(particle["energy"])
        mass = float(particle["mass"])

        path = []
        steps = 0

        while energy > material.displacement_threshold_ev and steps < config.max_steps_per_particle:
            steps += 1
            global_step += 1
            pos = pos + direction * config.step_size_a
            path.append(pos.copy())
            energy_history.append((float(pos[2]), energy))

            if np.any(pos < box_min) or np.any(pos > box_max):
                break

            nearby = tree.query_ball_point(pos, config.interaction_radius_a)
            if nearby and rng.random() < config.collision_probability:
                idx = int(rng.choice(nearby))
                target_pos = atomic_coords[idx]
                target_mass = material.atomic_mass_amu
                e_transfer = energy * (4 * mass * target_mass) / ((mass + target_mass) ** 2)

                if e_transfer > material.displacement_threshold_ev and not displaced_mask[idx]:
                    displaced_mask[idx] = True
                    displaced_atoms.append(target_pos)
                    vacancy_positions.append(target_pos.copy())

                    if len(active_particles) < config.max_total_particles:
                        new_dir = rng.normal(size=3)
                        new_dir = new_dir / np.linalg.norm(new_dir)
                        displaced_end = target_pos + (
                            new_dir * config.interaction_radius_a * config.displacement_jump_scale
                        )
                        displaced_end = np.clip(displaced_end, box_min, box_max)
                        displaced_final_positions.append(displaced_end)
                        displacement_events.append(
                            {
                                "event_step": int(global_step),
                                "vacancy_pos": target_pos.copy(),
                                "interstitial_pos": displaced_end.copy(),
                            }
                        )
                        active_particles.append(
                            {
                                "pos": target_pos.copy(),
                                "dir": new_dir,
                                "energy": float(e_transfer),
                                "mass": target_mass,
                            }
                        )
                    else:
                        displaced_final_positions.append(target_pos.copy())
                        displacement_events.append(
                            {
                                "event_step": int(global_step),
                                "vacancy_pos": target_pos.copy(),
                                "interstitial_pos": target_pos.copy(),
                            }
                        )
                    energy -= e_transfer

            energy *= config.stopping_factor
            if energy <= 0:
                break

        projectile_paths.append(np.array(path))

    displaced_arr = np.array(displaced_atoms) if displaced_atoms else np.empty((0, 3))
    displaced_final_arr = (
        np.array(displaced_final_positions) if displaced_final_positions else np.empty((0, 3))
    )
    vacancy_arr = np.array(vacancy_positions) if vacancy_positions else np.empty((0, 3))
    energy_arr = np.array(energy_history) if energy_history else np.empty((0, 2))

    if len(displaced_arr) > 0:
        mean_depth = float(displaced_arr[:, 2].mean())
        std_depth = float(displaced_arr[:, 2].std())
        max_depth = float(displaced_arr[:, 2].max())
    else:
        mean_depth = np.nan
        std_depth = np.nan
        max_depth = np.nan

    damage_hist, bin_edges = np.histogram(
        displaced_arr[:, 2] if len(displaced_arr) > 0 else np.array([]),
        bins=10,
        range=(float(box_min[2]), float(box_max[2])),
    )

    stats = {
        "total_displacements": int(len(displaced_arr)),
        "total_projectile_tracks": int(len(projectile_paths)),
        "total_energy_samples": int(len(energy_arr)),
        "mean_displacement_depth": mean_depth,
        "std_displacement_depth": std_depth,
        "max_displacement_depth": max_depth,
        "box_min_z": float(box_min[2]),
        "box_max_z": float(box_max[2]),
    }

    return {
        "config": asdict(config),
        "material": asdict(material),
        "stats": stats,
        "atomic_coords": atomic_coords,
        "displaced_atoms": displaced_arr,
        "displaced_final_positions": displaced_final_arr,
        "vacancy_positions": vacancy_arr,
        "displacement_events": displacement_events,
        "projectile_paths": projectile_paths,
        "energy_history": energy_arr,
        "damage_hist": damage_hist,
        "damage_bin_edges": bin_edges,
    }


def run_single_simulation(**kwargs) -> dict:
    config = SimConfig(**kwargs)
    return simulate_cascade(config)
