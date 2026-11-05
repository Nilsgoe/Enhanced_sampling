#!/usr/bin/env python
"""Run ff19SB/OPC trialanine-in-water WT-MetaD with biASE + OpenMM forces.

OPC water uses massless extra points. ASE must not integrate these particles.
This driver therefore exposes only real atoms to ASE/biASE and lets OpenMM
reconstruct virtual sites internally before each force evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from openmm import LangevinMiddleIntegrator, Platform, Vec3, unit
from openmm.app import AmberInpcrdFile, AmberPrmtopFile, PME, Simulation

from Metadynamics import WT_Metadynamics


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad/amber_ff19sb_opc")
KJMOL_TO_EV = 0.010364269656262174
KJMOL_NM_TO_EV_A = KJMOL_TO_EV / 10.0
CV_INDICES = {
    "phi": (14, 16, 18, 24),
    "psi": (16, 18, 24, 26),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2_000_000, help="2,000,000 at 0.5 fs = 1 ns.")
    parser.add_argument("--temperature", type=float, default=293.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--bias-height-ev", type=float, default=0.01036426966, help="1 kJ/mol in eV.")
    parser.add_argument("--sigma-deg", type=float, default=15.0)
    parser.add_argument("--deposition-interval", type=int, default=100)
    parser.add_argument("--trajectory-loginterval", type=int, default=100)
    parser.add_argument("--run-label", default="ff19sb_opc_biase_1ns")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--platform", default="CPU")
    parser.add_argument("--prmtop", type=Path, default=ROOT / "structures" / "trialanine_solution_ff19sb_opc.prmtop")
    parser.add_argument("--restart", type=Path, default=ROOT / "structures" / "trialanine_solution_ff19sb_opc_equil_npt.rst7")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "work")
    parser.add_argument("--restart-traj", type=Path, default=None)
    parser.add_argument("--input-bias", type=Path, default=None)
    parser.add_argument("--append-trajectory", action="store_true")
    return parser.parse_args()


def compute_dihedral(positions, indices):
    p1, p2, p3, p4 = positions[jnp.array(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    b2_unit = b2 / jnp.linalg.norm(b2)
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2_unit)
    return jnp.degrees(jnp.arctan2(sin_phi, cos_phi))


def build_cv_function():
    def phi_psi(positions):
        return jnp.array(
            [
                compute_dihedral(positions, CV_INDICES["phi"]),
                compute_dihedral(positions, CV_INDICES["psi"]),
            ]
        )

    return phi_psi


def get_platform(name: str) -> Platform:
    try:
        return Platform.getPlatformByName(name)
    except Exception:
        if name != "CPU":
            return Platform.getPlatformByName("CPU")
        raise


def positions_quantity_to_nm_array(positions) -> np.ndarray:
    return np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)


def box_to_ase_cell(box_vectors) -> np.ndarray:
    vectors_nm = np.asarray([[v.x, v.y, v.z] for v in box_vectors], dtype=float)
    return vectors_nm * 10.0


def real_atom_metadata(prmtop: AmberPrmtopFile, system):
    symbols: list[str] = []
    real_to_omm: list[int] = []
    for atom in prmtop.topology.atoms():
        idx = atom.index
        mass = system.getParticleMass(idx).value_in_unit(unit.dalton)
        if mass == 0.0:
            continue
        if atom.element is None:
            raise ValueError(f"Real particle {idx} has no chemical element; check topology.")
        symbols.append(atom.element.symbol)
        real_to_omm.append(idx)
    return symbols, np.asarray(real_to_omm, dtype=int)


def build_start_atoms(prmtop_path: Path, restart_path: Path, platform_name: str) -> tuple[Atoms, object, np.ndarray]:
    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(restart_path))
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=0.8 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        ewaldErrorTolerance=5.0e-4,
    )
    symbols, real_to_omm = real_atom_metadata(prmtop, system)
    positions_nm = positions_quantity_to_nm_array(inpcrd.positions)
    atoms = Atoms(symbols=symbols, positions=positions_nm[real_to_omm] * 10.0)
    if inpcrd.boxVectors is not None:
        atoms.set_cell(box_to_ase_cell(inpcrd.boxVectors), scale_atoms=False)
        atoms.pbc = True

    # Build the context once here so virtual-site support is validated before MD.
    integrator = LangevinMiddleIntegrator(293.0 * unit.kelvin, 1.0 / unit.picosecond, 0.5 * unit.femtosecond)
    simulation = Simulation(prmtop.topology, system, integrator, get_platform(platform_name))
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    simulation.context.computeVirtualSites()
    return atoms, simulation, real_to_omm


class OpenMMVirtualSiteCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, simulation: Simulation, real_to_omm: np.ndarray):
        super().__init__()
        self.simulation = simulation
        self.real_to_omm = real_to_omm
        self.n_particles = simulation.system.getNumParticles()
        self._positions_nm = np.zeros((self.n_particles, 3), dtype=float)
        state = simulation.context.getState(getPositions=True)
        self._positions_nm[:] = positions_quantity_to_nm_array(state.getPositions())

    def _openmm_positions(self, real_positions_a: np.ndarray):
        self._positions_nm[self.real_to_omm] = real_positions_a / 10.0
        return [Vec3(*xyz) for xyz in self._positions_nm] * unit.nanometer

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        context = self.simulation.context
        context.setPositions(self._openmm_positions(np.asarray(self.atoms.get_positions(), dtype=float)))
        if self.atoms.pbc.any():
            vectors = np.asarray(self.atoms.cell.array, dtype=float) / 10.0
            context.setPeriodicBoxVectors(*(Vec3(*row) for row in vectors))
        context.computeVirtualSites()
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        self._positions_nm[:] = positions_quantity_to_nm_array(state.getPositions())
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        self.results["energy"] = float(energy) * KJMOL_TO_EV
        self.results["forces"] = forces[self.real_to_omm] * KJMOL_NM_TO_EV_A


def main() -> None:
    args = parse_args()
    run_dir = args.work_dir / args.run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    atoms, simulation, real_to_omm = build_start_atoms(args.prmtop, args.restart, args.platform)
    if args.restart_traj is not None:
        restart_atoms = read(args.restart_traj, index=-1)
        atoms.set_positions(restart_atoms.get_positions())
        atoms.set_cell(restart_atoms.get_cell(), scale_atoms=False)
        atoms.pbc = restart_atoms.pbc

    atoms.calc = OpenMMVirtualSiteCalculator(simulation, real_to_omm)

    if atoms.get_momenta().shape != (len(atoms), 3) or not np.any(atoms.get_momenta()):
        rng = np.random.default_rng(args.random_seed) if args.random_seed is not None else None
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_dir / f"solution_{args.run_label}"

    dyn = WT_Metadynamics(
        atoms=atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(output_prefix.with_suffix(".traj")),
        fixcm=False,
        cvs=build_cv_function(),
        bias_height=args.bias_height_ev,
        interval_size=args.deposition_interval,
        output_file=str(output_prefix.with_suffix(".bias")),
        well_temp=True,
        bias_factor=10,
        append_trajectory=args.append_trajectory,
        input_file=str(args.input_bias) if args.input_bias is not None else None,
        max_bias=int(1e6),
        std_dev=[args.sigma_deg, args.sigma_deg],
        wrapping=[True, True],
        bounds=((-180.0, 180.0), (-180.0, 180.0)),
        loginterval=args.trajectory_loginterval,
    )
    dyn.run(args.steps)


if __name__ == "__main__":
    main()
