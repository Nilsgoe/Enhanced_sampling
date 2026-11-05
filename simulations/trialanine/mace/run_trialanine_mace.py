#!/usr/bin/env python
"""Run trialanine MACE equilibration and biASE well-tempered MetaD."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import torch
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS

from Metadynamics import WT_Metadynamics


MODEL_SPECS = {
    "off": {"family": "off", "size": "large"},
    "off24": {
        "family": "off24",
        "checkpoint": "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/malonaldehyd/MACE-OFF24_medium.model",
    },
    "omol": {"family": "omol", "size": "extra_large"},
    "mh1": {"family": "mh1", "size": "mh-1"},
    "polar": {"family": "polar", "size": "m"},
}

# 0-based ASE atom indices for ACE-ALA-ALA-ALA-NME as built by Amber sequence.
# These target the central alanine residue (ALA 3 in the Amber PDB numbering).
DEFAULT_PHI = (14, 16, 18, 24)  # C(ALA2)-N(ALA3)-CA(ALA3)-C(ALA3)
DEFAULT_PSI = (16, 18, 24, 26)  # N(ALA3)-CA(ALA3)-C(ALA3)-N(ALA4)


def parse_indices(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(part) for part in text.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Expected four comma-separated 0-based atom indices.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["gas_metad", "solution_npt", "solution_npt_continue", "solution_metad"],
        required=True,
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--npt-output", type=Path, help="Required for solution_npt.")
    parser.add_argument("--nvt-output", type=Path, help="Optional fixed-cell NVT output for solution_npt.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="MetaD or plain MD steps.")
    parser.add_argument("--nvt-steps", type=int, default=40_000)
    parser.add_argument("--npt-steps", type=int, default=200_000)
    parser.add_argument("--temperature", type=float, default=293.0)
    parser.add_argument("--pressure-bar", type=float, default=1.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--bias-height-ev", type=float, default=0.1)
    parser.add_argument("--sigma-deg", type=float, default=5.0)
    parser.add_argument("--deposition-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-float32", action="store_true")
    parser.add_argument("--phi", type=parse_indices, default=DEFAULT_PHI)
    parser.add_argument("--psi", type=parse_indices, default=DEFAULT_PSI)
    parser.add_argument("--minimize", action="store_true", help="Run a short MACE minimization before dynamics.")
    parser.add_argument(
        "--continue-run",
        action="store_true",
        help="Continue MetaD from output-prefix.metad.traj and output-prefix.metad.txt.",
    )
    parser.add_argument(
        "--trajectory-loginterval",
        type=int,
        default=1,
        help="Write every Nth MD step to the MetaD trajectory.",
    )
    return parser.parse_args()


def build_calculator(model_key: str, device: str):
    spec = MODEL_SPECS[model_key]
    family = spec["family"]
    size = spec.get("size")

    if family == "off":
        from mace.calculators import mace_off

        return mace_off(model=size, dispersion=True, enable_cueq=True, device=device)
    if family == "off24":
        from mace.calculators import mace_off

        return mace_off(
            checkpoint_path=spec["checkpoint"],
            dispersion=True,
            enable_cueq=True,
            device=device,
            default_dtype="float32",
        )
    if family == "omol":
        from mace.calculators import mace_omol

        return mace_omol(model=size, enable_cueq=True, device=device)
    if family == "mh1":
        from mace.calculators import mace_mp

        return mace_mp(
            model=size,
            head="omol",
            enable_cueq=True,
            device=device,
            default_dtype="float32",
        )
    if family == "polar":
        from mace.calculators import mace_polar

        return mace_polar(model=f"polar-1-{size}", enable_cueq=True, device=device)
    raise ValueError(f"Unsupported model key: {model_key}")


def load_last_valid_traj_frame(path: str):
    traj = Trajectory(path)
    for idx in range(len(traj) - 1, -1, -1):
        try:
            return traj[idx].copy()
        except Exception:
            continue
    raise ValueError(f"No readable frame found in trajectory {path}")


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


def prepare_atoms(args: argparse.Namespace):
    input_spec = str(args.input)
    if args.continue_run:
        traj_path = args.output_prefix.with_suffix(".metad.traj")
        if not traj_path.exists():
            raise FileNotFoundError(f"Continuation requested but missing trajectory: {traj_path}")
        atoms = load_last_valid_traj_frame(str(traj_path))
    elif args.mode == "solution_npt_continue" and input_spec.endswith(".traj"):
        atoms = load_last_valid_traj_frame(input_spec)
    else:
        atoms = read(input_spec).copy()
    if args.mode == "gas_metad":
        atoms.set_pbc(False)
    else:
        atoms.set_pbc(True)
        if atoms.cell.volume <= 0:
            raise ValueError(f"{args.input} has no valid periodic cell for solution mode.")
    atoms.calc = build_calculator(args.model_key, args.device)
    return atoms


def initialize_velocities(atoms, temperature: float) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    ZeroRotation(atoms)


def maybe_minimize(atoms, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    BFGS(atoms, logfile=str(output_prefix.with_suffix(".bfgs.log"))).run(fmax=0.05, steps=500)


def run_solution_npt(atoms, args: argparse.Namespace) -> None:
    if args.npt_output is None:
        raise ValueError("--npt-output is required for solution_npt.")

    initialize_velocities(atoms, args.temperature)
    nvt = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(args.output_prefix.with_suffix(".nvt.traj")),
        logfile=str(args.output_prefix.with_suffix(".nvt.log")),
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
        taut=1000.0 * units.fs,
        taup=2000.0 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
        logfile=str(args.output_prefix.with_suffix(".npt.log")),
        trajectory=str(args.output_prefix.with_suffix(".npt.traj")),
        loginterval=100,
    )
    dyn.run(args.npt_steps)

    args.npt_output.parent.mkdir(parents=True, exist_ok=True)
    write(args.npt_output, atoms, format="extxyz")
    write(args.output_prefix.with_suffix(".npt_final.xyz"), atoms, format="extxyz")


def run_solution_npt_continue(atoms, args: argparse.Namespace) -> None:
    if args.npt_output is None:
        raise ValueError("--npt-output is required for solution_npt_continue.")

    initialize_velocities(atoms, args.temperature)
    dyn = NPTBerendsen(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        pressure_au=args.pressure_bar * units.bar,
        taut=1000.0 * units.fs,
        taup=2000.0 * units.fs,
        compressibility_au=4.57e-5 / units.bar,
        logfile=str(args.output_prefix.with_suffix(".npt.log")),
        trajectory=str(args.output_prefix.with_suffix(".npt.traj")),
        loginterval=100,
    )
    dyn.run(args.npt_steps)

    args.npt_output.parent.mkdir(parents=True, exist_ok=True)
    write(args.npt_output, atoms, format="extxyz")
    write(args.output_prefix.with_suffix(".npt_final.xyz"), atoms, format="extxyz")


def run_gas_thermalization(atoms, args: argparse.Namespace) -> None:
    initialize_velocities(atoms, args.temperature)
    dyn = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(args.output_prefix.with_suffix(".thermalize.traj")),
        logfile=str(args.output_prefix.with_suffix(".thermalize.log")),
        loginterval=100,
    )
    dyn.run(min(args.steps, 1000))


def run_metad(atoms, args: argparse.Namespace) -> None:
    bias_path = args.output_prefix.with_suffix(".metad.txt")
    traj_path = args.output_prefix.with_suffix(".metad.traj")
    if args.continue_run and not bias_path.exists():
        raise FileNotFoundError(f"Continuation requested but missing bias file: {bias_path}")
    if not args.continue_run:
        initialize_velocities(atoms, args.temperature)
    dyn = WT_Metadynamics(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.1,
        trajectory=str(traj_path),
        fixcm=False,
        cvs=build_cv_function(args.phi, args.psi),
        std_dev=[args.sigma_deg, args.sigma_deg],
        bias_height=args.bias_height_ev,
        interval_size=args.deposition_interval,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=10,
        append_trajectory=args.continue_run,
        input_file=str(bias_path) if args.continue_run else None,
        wrapping=[True, True],
        bounds=((-180.0, 180.0), (-180.0, 180.0)),
        max_bias=int(1e6),
        loginterval=args.trajectory_loginterval,
    )
    dyn.run(args.steps)


def main() -> None:
    args = parse_args()
    if args.force_float32:
        torch.set_default_dtype(torch.float32)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    atoms = prepare_atoms(args)

    if args.minimize:
        maybe_minimize(atoms, args.output_prefix)

    if args.mode == "solution_npt":
        run_solution_npt(atoms, args)
    elif args.mode == "solution_npt_continue":
        run_solution_npt_continue(atoms, args)
    elif args.mode == "solution_metad":
        run_metad(atoms, args)
    elif args.mode == "gas_metad":
        run_gas_thermalization(atoms, args)
        run_metad(atoms, args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
