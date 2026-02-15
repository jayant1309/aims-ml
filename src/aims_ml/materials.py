from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.core import Lattice, Structure


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    crystal_type: str
    lattice_constant: float
    displacement_threshold_ev: float
    atomic_mass_amu: float


MATERIAL_DB: dict[str, MaterialSpec] = {
    "Al": MaterialSpec(
        name="Al",
        crystal_type="fcc",
        lattice_constant=4.046,
        displacement_threshold_ev=25.0,
        atomic_mass_amu=26.98,
    ),
    "Ti": MaterialSpec(
        name="Ti",
        crystal_type="hcp_approx",
        lattice_constant=2.95,
        displacement_threshold_ev=30.0,
        atomic_mass_amu=47.87,
    ),
    "Fe": MaterialSpec(
        name="Fe",
        crystal_type="bcc",
        lattice_constant=2.866,
        displacement_threshold_ev=40.0,
        atomic_mass_amu=55.85,
    ),
    "Cu": MaterialSpec(
        name="Cu",
        crystal_type="fcc",
        lattice_constant=3.615,
        displacement_threshold_ev=30.0,
        atomic_mass_amu=63.55,
    ),
}


def _build_unit_cell(material: MaterialSpec) -> Structure:
    a = material.lattice_constant
    if material.name in {"Al", "Cu"}:
        lattice = Lattice.cubic(a=a)
        coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        return Structure(lattice, [material.name] * 4, coords)
    if material.name == "Fe":
        lattice = Lattice.cubic(a=a)
        coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
        return Structure(lattice, [material.name] * 2, coords)
    # Approximate Ti using hexagonal lattice with two-atom basis.
    c = 1.587 * a
    lattice = Lattice.hexagonal(a=a, c=c)
    coords = [[0, 0, 0], [2 / 3, 1 / 3, 0.5]]
    return Structure(lattice, [material.name] * 2, coords)


def generate_lattice(
    element: str,
    supercell_dims: tuple[int, int, int] = (10, 10, 20),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MaterialSpec]:
    if element not in MATERIAL_DB:
        raise ValueError(f"Unsupported material '{element}'. Supported: {list(MATERIAL_DB)}")

    material = MATERIAL_DB[element]
    structure = _build_unit_cell(material)
    structure.make_supercell(supercell_dims)

    atomic_coords = np.array(structure.cart_coords)
    box_min = atomic_coords.min(axis=0)
    box_max = atomic_coords.max(axis=0)

    return atomic_coords, box_min, box_max, material
