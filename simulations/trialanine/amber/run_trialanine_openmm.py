#!/usr/bin/env python
"""Run Amber ff14SB/TIP3P trialanine WT-MetaD with native OpenMM metadynamics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from openmm import (
    CustomTorsionForce,
    LangevinMiddleIntegrator,
    Platform,
    unit,
)
from openmm.app import (
    AmberInpcrdFile,
    AmberPrmtopFile,
    BiasVariable,
    CheckpointReporter,
    DCDReporter,
    HBonds,
    Metadynamics,
    PDBFile,
    PME,
    PDBReporter,
    Simulation,
    StateDataReporter,
)


PROJECT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad")
CV_INDICES = {
    "phi": (14, 16, 18, 24),
    "psi": (16, 18, 24, 26),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("solution",), default="solution")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--temperature-k", type=float, default=293.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--friction-ps", type=float, default=1.0)
    parser.add_argument("--height-kjmol", type=float, default=1.0)
    parser.add_argument("--sigma-deg", type=float, default=15.0)
    parser.add_argument("--bias-factor", type=float, default=10.0)
    parser.add_argument("--deposition-interval", type=int, default=100)
    parser.add_argument("--grid-width", type=int, default=181)
    parser.add_argument("--report-interval", type=int, default=1000)
    parser.add_argument("--dcd-interval", type=int, default=10)
    parser.add_argument("--bias-save-interval", type=int, default=100)
    parser.add_argument("--no-state-log", action="store_true")
    parser.add_argument("--no-dcd", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-free-energy", action="store_true")
    parser.add_argument("--no-final-pdb", action="store_true")
    parser.add_argument("--platform", default="CPU", help="CPU, CUDA, or Reference.")
    parser.add_argument("--run-name", default="solution_openmm_wtmetad_1kjmol_15deg_0p5fs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "openmm_amber_metad" / "outputs",
    )
    parser.add_argument(
        "--bias-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def phase_inputs(phase: str) -> tuple[Path, Path]:
    if phase != "solution":
        raise ValueError("Only solution is wired for this OpenMM reference run.")
    return (
        PROJECT / "structures" / "trialanine_solution.prmtop",
        PROJECT / "structures" / "trialanine_solution_equil_npt.rst7",
    )


def torsion_cv(name: str, indices: tuple[int, int, int, int]) -> CustomTorsionForce:
    # OpenMM torsion theta is in radians and naturally spans [-pi, pi].
    force = CustomTorsionForce("theta")
    force.setName(name)
    force.addTorsion(*indices, [])
    return force


def torsion_degrees(positions_nm, indices: tuple[int, int, int, int]) -> float:
    coords = np.asarray(positions_nm.value_in_unit(unit.nanometer), dtype=float)
    p1, p2, p3, p4 = coords[list(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2 /= np.linalg.norm(b2)
    angle = np.arctan2(np.dot(np.cross(n1, n2), b2), np.dot(n1, n2))
    return float(np.degrees(angle))


def get_platform(name: str) -> Platform:
    try:
        return Platform.getPlatformByName(name)
    except Exception:
        if name != "CPU":
            return Platform.getPlatformByName("CPU")
        raise


def write_cv_header(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "time_fs", "phi_deg", "psi_deg", "potential_kjmol"])


def append_cv_row(path: Path, simulation: Simulation, step: int, timestep_fs: float) -> None:
    context = simulation.context
    state = context.getState(getEnergy=True, getPositions=True)
    phi = torsion_degrees(state.getPositions(), CV_INDICES["phi"])
    psi = torsion_degrees(state.getPositions(), CV_INDICES["psi"])
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([step, step * timestep_fs, phi, psi, energy])


def main() -> None:
    args = parse_args()
    if args.bias_dir is None:
        args.bias_dir = PROJECT / "openmm_amber_metad" / "bias" / args.run_name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.bias_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = args.output_dir / args.run_name

    prmtop_path, inpcrd_path = phase_inputs(args.phase)
    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=0.8 * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        ewaldErrorTolerance=5.0e-4,
    )

    variables = [
        BiasVariable(
            torsion_cv("phi_cv", CV_INDICES["phi"]),
            -np.pi,
            np.pi,
            np.radians(args.sigma_deg),
            gridWidth=args.grid_width,
            periodic=True,
        ),
        BiasVariable(
            torsion_cv("psi_cv", CV_INDICES["psi"]),
            -np.pi,
            np.pi,
            np.radians(args.sigma_deg),
            gridWidth=args.grid_width,
            periodic=True,
        ),
    ]
    metad = Metadynamics(
        system,
        variables,
        args.temperature_k * unit.kelvin,
        args.bias_factor,
        args.height_kjmol * unit.kilojoule_per_mole,
        args.deposition_interval,
        saveFrequency=args.bias_save_interval,
        biasDir=str(args.bias_dir),
    )

    integrator = LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    platform = get_platform(args.platform)
    simulation = Simulation(prmtop.topology, system, integrator, platform)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    simulation.context.setVelocitiesToTemperature(args.temperature_k * unit.kelvin)

    if not args.no_state_log:
        simulation.reporters.append(StateDataReporter(
            str(run_prefix.with_suffix(".log")),
            args.report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            separator=",",
        ))
    if not args.no_dcd:
        simulation.reporters.append(DCDReporter(str(run_prefix.with_suffix(".dcd")), args.dcd_interval))
    if not args.no_checkpoint:
        simulation.reporters.append(CheckpointReporter(str(run_prefix.with_suffix(".chk")), args.report_interval))

    cv_path = run_prefix.with_suffix(".cv.csv")
    if not args.no_cv:
        write_cv_header(cv_path)
        append_cv_row(cv_path, simulation, 0, args.timestep_fs)

    remaining = args.steps
    current_step = 0
    while remaining > 0:
        block = min(args.report_interval, remaining)
        metad.step(simulation, block)
        current_step += block
        remaining -= block
        if not args.no_cv:
            append_cv_row(cv_path, simulation, current_step, args.timestep_fs)

    if not args.no_free_energy:
        free_energy = metad.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)
        np.save(run_prefix.with_suffix(".free_energy_kjmol.npy"), free_energy)
    if not args.no_final_pdb:
        final_state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        with run_prefix.with_suffix(".final.pdb").open("w") as handle:
            PDBFile.writeFile(prmtop.topology, final_state.getPositions(), handle)


if __name__ == "__main__":
    main()
