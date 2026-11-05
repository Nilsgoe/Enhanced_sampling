#!/usr/bin/env python
"""Restartable trialanine PET-SPICE/SO3LR equilibration and MetaD for viper-gpu."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS

from Metadynamics import WT_Metadynamics


DEFAULT_PHI = (14, 16, 18, 24)  # C(ALA2)-N(ALA3)-CA(ALA3)-C(ALA3)
DEFAULT_PSI = (16, 18, 24, 26)  # N(ALA3)-CA(ALA3)-C(ALA3)-N(ALA4)

TEMPERATURE_K = 293.0
FRICTION = 0.1
TIMESTEP_FS = 0.5
BIAS_HEIGHT = 0.01036426966
BIAS_FACTOR = 10
INTERVAL_SIZE = 100
MAX_BIAS = 1_000_000
STD_DEV = [15.0, 15.0]
DIHEDRAL_BOUNDS = ((-180.0, 180.0), (-180.0, 180.0))


def parse_indices(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated 0-based atom indices.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gas_metad", "solution_equil", "solution_metad"], required=True)
    parser.add_argument("--model-kind", choices=["upet", "sol3r"], required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--steps-per-chunk", type=int, default=1_000_000)
    parser.add_argument("--nvt-steps", type=int, default=100_000)
    parser.add_argument("--npt-steps", type=int, default=500_000)
    parser.add_argument("--nvt-output", type=Path)
    parser.add_argument("--npt-output", type=Path)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=TIMESTEP_FS)
    parser.add_argument("--phi", type=parse_indices, default=DEFAULT_PHI)
    parser.add_argument("--psi", type=parse_indices, default=DEFAULT_PSI)
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--target-fs", type=float, default=1_000_000.0)
    parser.add_argument("--restart-traj", type=Path)
    parser.add_argument("--restart-bias", type=Path)
    parser.add_argument("--trajectory-loginterval", type=int, default=10)
    return parser.parse_args()


def build_calculator(model_kind: str):
    if model_kind == "upet":
        from upet.calculator import UPETCalculator

        return UPETCalculator(model="pet-spice-l", device="cuda")
    if model_kind == "sol3r":
        from so3lr import So3lrCalculator

        return So3lrCalculator(calculate_stress=False, lr_cutoff=12.0, dtype=np.float64)
    raise ValueError(f"Unsupported model kind: {model_kind}")


def compute_dihedral(positions, indices: tuple[int, int, int, int]):
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


def build_cv_function(phi: tuple[int, int, int, int], psi: tuple[int, int, int, int]):
    def cvs(positions):
        return jnp.array([compute_dihedral(positions, phi), compute_dihedral(positions, psi)])

    return cvs


def initialize_velocities(atoms, temperature: float) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)


def ensure_velocities(atoms, temperature: float) -> None:
    if atoms.get_velocities() is None:
        initialize_velocities(atoms, temperature)


def metad_paths(run_name: str):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    trajectory_path = outputs_dir / f"{run_name}.traj"
    bias_path = outputs_dir / f"{run_name}.bias"
    bfgs_log = outputs_dir / f"bfgs_{run_name}.log"
    final_xyz = outputs_dir / f"{run_name}_chunk_last.xyz"
    return trajectory_path, bias_path, bfgs_log, final_xyz


def finite_atoms(atoms) -> bool:
    return np.isfinite(atoms.get_positions()).all()


def read_last_finite_frame(trajectory_path: Path, max_back: int = 200):
    for offset in range(1, max_back + 1):
        atoms = read(f"{trajectory_path}@-{offset}")
        if finite_atoms(atoms):
            if offset > 1:
                print(f"Using trajectory restart frame -{offset}; newer frame(s) were non-finite.")
            return atoms
    raise ValueError(f"No finite restart frame found in last {max_back} frames of {trajectory_path}")


def load_atoms(args: argparse.Namespace, trajectory_path: Path):
    if trajectory_path.exists():
        return read_last_finite_frame(trajectory_path).copy(), True
    if args.restart_traj is not None:
        return read_last_finite_frame(args.restart_traj).copy(), False
    return read(args.start_file).copy(), False


def maybe_minimize(atoms, logfile: Path) -> None:
    BFGS(atoms, logfile=str(logfile)).run(fmax=0.05, steps=500)


def run_solution_equil(args: argparse.Namespace) -> None:
    if args.npt_output is None:
        raise ValueError("--npt-output is required for solution_equil.")

    atoms = read(args.start_file).copy()
    atoms.set_pbc(True)
    if atoms.cell.volume <= 0:
        raise ValueError(f"{args.start_file} has no valid periodic cell for solution_equil.")
    atoms.calc = build_calculator(args.model_kind)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    if args.minimize:
        maybe_minimize(atoms, outputs_dir / f"bfgs_{args.run_name}.log")

    initialize_velocities(atoms, args.temperature)
    nvt = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=FRICTION,
        trajectory=str(outputs_dir / f"{args.run_name}.nvt.traj"),
        logfile=str(outputs_dir / f"{args.run_name}.nvt.log"),
        loginterval=100,
    )
    nvt.run(args.nvt_steps)

    if args.nvt_output is not None:
        args.nvt_output.parent.mkdir(parents=True, exist_ok=True)
        write(args.nvt_output, atoms, format="extxyz")

    dyn = NPTBerendsen(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        pressure_au=args.pressure_bar * units.bar,
        taut=100.0 * units.fs,
        taup=1000.0 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
        trajectory=str(outputs_dir / f"{args.run_name}.npt.traj"),
        logfile=str(outputs_dir / f"{args.run_name}.npt.log"),
        loginterval=100,
    )
    dyn.run(args.npt_steps)

    args.npt_output.parent.mkdir(parents=True, exist_ok=True)
    write(args.npt_output, atoms, format="extxyz")
    write(outputs_dir / f"{args.run_name}.npt_final.xyz", atoms, format="extxyz")
    print(f"Completed {args.run_name} solution_equil")


def run_metad(args: argparse.Namespace) -> None:
    trajectory_path, bias_path, bfgs_log, final_xyz = metad_paths(args.run_name)
    if args.restart_bias is not None and not bias_path.exists():
        shutil.copy2(args.restart_bias, bias_path)
        print(f"Seeded continuation bias from {args.restart_bias} -> {bias_path}")
    atoms, append_trajectory = load_atoms(args, trajectory_path)
    continue_bias = bias_path.exists()

    if args.mode == "gas_metad":
        atoms.set_pbc(False)
    else:
        atoms.set_pbc(True)
        if atoms.cell.volume <= 0:
            raise ValueError(f"{args.start_file} has no valid periodic cell for solution_metad.")

    atoms.calc = build_calculator(args.model_kind)

    if not continue_bias:
        if args.minimize:
            maybe_minimize(atoms, bfgs_log)
        initialize_velocities(atoms, args.temperature)
    else:
        ensure_velocities(atoms, args.temperature)

    steps = args.steps_per_chunk
    if continue_bias:
        last_time_fs = 0.0
        with bias_path.open(errors="ignore") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 8 and not fields[0].startswith("!") and fields[0] != "time":
                    try:
                        last_time_fs = float(fields[0])
                    except ValueError:
                        pass
        remaining_steps = max(0, int(round((args.target_fs - last_time_fs) / args.timestep_fs)))
        steps = min(steps, remaining_steps)
        print(f"Current bias time: {last_time_fs} fs; remaining steps: {remaining_steps}")
    if steps <= 0:
        print(f"{args.run_name} already reached target {args.target_fs} fs")
        return

    dyn = WT_Metadynamics(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=FRICTION,
        trajectory=str(trajectory_path),
        fixcm=(args.mode == "gas_metad"),
        cvs=build_cv_function(args.phi, args.psi),
        std_dev=STD_DEV,
        bias_height=BIAS_HEIGHT,
        interval_size=INTERVAL_SIZE,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=BIAS_FACTOR,
        append_trajectory=append_trajectory,
        input_file=str(bias_path) if continue_bias else None,
        wrapping=[True, True],
        bounds=DIHEDRAL_BOUNDS,
        max_bias=MAX_BIAS,
        loginterval=args.trajectory_loginterval,
    )
    dyn.run(steps)
    write(final_xyz, atoms, format="extxyz")
    print(
        f"Completed {args.run_name} chunk {args.chunk_id} "
        f"for {steps} steps; continue_bias={continue_bias}; append_trajectory={append_trajectory}"
    )


def main() -> None:
    args = parse_args()
    if args.mode == "solution_equil":
        run_solution_equil(args)
    else:
        run_metad(args)


if __name__ == "__main__":
    main()
