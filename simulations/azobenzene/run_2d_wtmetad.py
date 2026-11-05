#!/usr/bin/env python
"""Short azobenzene MetaD tests with finite-force walls on both adjacent angles."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from pathlib import Path

import jax.experimental.layout as _jax_layout

if not hasattr(_jax_layout, "DeviceLocalLayout"):
    _jax_layout.DeviceLocalLayout = _jax_layout.Layout

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.calculators.mixing import LinearCombinationCalculator
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS

from biASE import Direct_General_Bias_Calculator
from Metadynamics import WT_Metadynamics


SYSTEMS = {
    "cis": {
        "dihedral": (1, 6, 7, 8),
        "left_angle": (1, 6, 7),
        "right_angle": (6, 7, 8),
    },
    "trans": {
        "dihedral": (2, 11, 12, 13),
        "left_angle": (2, 11, 12),
        "right_angle": (11, 12, 13),
    },
}

TEMPERATURE_K = 333.0
TIMESTEP_FS = 0.5
FRICTION = 0.1
BIAS_HEIGHT_EV = 0.1
BIAS_FACTOR = 10
HILL_INTERVAL = 100
HILL_SIGMA_DEG = [10.0, 10.0]
CV_EPS = 1.0e-12

WALL_ENERGY_EV = 5.0
LOW_ONSET_DEG = 45.0
LOW_TARGET_DEG = 35.0
HIGH_ONSET_DEG = 170.0
HIGH_TARGET_DEG = 180.0

FT_SO3LR_WORKDIR = "/ptmp/ngoen/Documents/azobenzene_so3lr_training/ft_so3lr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[
            "mace_off",
            "mace_mh1",
            "mace_polar_m",
            "mace_scratch",
            "mace_ft_mh1",
            "pet_spice",
            "sol3r",
            "ft_so3lr",
            "mace_les_off",
        ],
        required=True,
    )
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--start-file", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--trajectory-loginterval", type=int, default=10)
    return parser.parse_args()


def safe_norm(vector):
    return jnp.sqrt(jnp.dot(vector, vector) + CV_EPS**2)


def angle_degrees(positions, indices):
    pa, pb, pc = positions[jnp.asarray(indices)]
    v1 = pa - pb
    v2 = pc - pb
    return jnp.degrees(jnp.arctan2(safe_norm(jnp.cross(v1, v2)), jnp.dot(v1, v2)))


def dihedral_degrees(positions, indices):
    p1, p2, p3, p4 = positions[jnp.asarray(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 = n1 / safe_norm(n1)
    n2 = n2 / safe_norm(n2)
    b2_unit = b2 / safe_norm(b2)
    return jnp.degrees(
        jnp.arctan2(jnp.dot(jnp.cross(n1, n2), b2_unit), jnp.dot(n1, n2))
    )


def angle_cosine(positions, indices):
    pa, pb, pc = positions[jnp.asarray(indices)]
    v1 = pa - pb
    v2 = pc - pb
    return jnp.dot(v1, v2) / (safe_norm(v1) * safe_norm(v2))


def _wall_branch(normalized_penetration):
    """Quadratic cosine wall; zero value and force at the onset."""
    return WALL_ENERGY_EV * jnp.maximum(normalized_penetration, 0.0) ** 2


def single_angle_wall_from_cosine(cosine):
    cos_low_onset = math.cos(math.radians(LOW_ONSET_DEG))
    cos_low_target = math.cos(math.radians(LOW_TARGET_DEG))
    cos_high_onset = math.cos(math.radians(HIGH_ONSET_DEG))
    cos_high_target = math.cos(math.radians(HIGH_TARGET_DEG))

    low_x = (cosine - cos_low_onset) / (cos_low_target - cos_low_onset)
    high_x = (cos_high_onset - cosine) / (cos_high_onset - cos_high_target)
    return _wall_branch(low_x) + _wall_branch(high_x)


def build_angle_wall(system):
    spec = SYSTEMS[system]

    def wall_energy(positions):
        positions = jnp.asarray(positions)
        left_cos = angle_cosine(positions, spec["left_angle"])
        right_cos = angle_cosine(positions, spec["right_angle"])
        return single_angle_wall_from_cosine(left_cos) + single_angle_wall_from_cosine(
            right_cos
        )

    return Direct_General_Bias_Calculator(wall_energy, use_jit=True)


def build_metad_cvs(system):
    spec = SYSTEMS[system]

    def cvs(positions):
        positions = jnp.asarray(positions)
        return jnp.asarray(
            [
                dihedral_degrees(positions, spec["dihedral"]),
                angle_degrees(positions, spec["left_angle"]),
            ]
        )

    return cvs


def build_physical_calculator(model, model_path):
    if model in {"mace_off", "mace_mh1", "mace_polar_m"}:
        import torch
        from mace.calculators import mace_mp, mace_off, mace_polar

        torch.set_default_dtype(torch.float32)
        if model == "mace_off":
            return mace_off(
                model="large", dispersion=True, enable_cueq=False, device="cuda"
            )
        if model == "mace_mh1":
            return mace_mp(
                model="mh-1",
                head="omol",
                enable_oeq=False,
                device="cuda",
                default_dtype="float32",
            )
        return mace_polar(
            model="polar-1-m",
            enable_cueq=False,
            device="cuda",
            default_dtype="float32",
        )

    if model in {"mace_scratch", "mace_ft_mh1", "mace_les_off"}:
        import torch
        from mace.calculators import MACECalculator

        if not model_path:
            raise ValueError(f"--model-path is required for {model}")
        torch.set_default_dtype(torch.float32)
        return MACECalculator(
            model_paths=[model_path],
            device="cuda",
            default_dtype="float32" if model == "mace_les_off" else "float64",
        )

    if model == "pet_spice":
        from upet.calculator import UPETCalculator

        return UPETCalculator(model="pet-spice-l", device="cuda")

    if model == "sol3r":
        from so3lr import So3lrCalculator

        return So3lrCalculator(
            calculate_stress=False, lr_cutoff=1000, dtype=np.float64
        )

    if model == "ft_so3lr":
        from mlff.md import mlffCalculatorSparse

        return mlffCalculatorSparse.create_from_ckpt_dir(
            ckpt_dir=FT_SO3LR_WORKDIR,
            from_file=False,
            calculate_stress=False,
            lr_cutoff=1000,
            lr_neighbors_bool=True,
            dispersion_energy_cutoff_lr_damping=2.0,
        )

    raise ValueError(model)


def geometric_angle_degrees(positions, indices):
    pa, pb, pc = positions[np.asarray(indices)]
    v1 = pa - pb
    v2 = pc - pb
    return math.degrees(math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2)))


def attach_wall_diagnostics(dyn, atoms, wall_calculator, path, system, append):
    if not append:
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "step",
                    "time_fs",
                    "left_angle_deg",
                    "right_angle_deg",
                    "wall_energy_eV",
                    "max_wall_force_eV_per_A",
                ]
            )

    spec = SYSTEMS[system]

    def write_row():
        positions = atoms.get_positions()
        left = geometric_angle_degrees(positions, spec["left_angle"])
        right = geometric_angle_degrees(positions, spec["right_angle"])
        wall_energy = wall_calculator.get_potential_energy(atoms)
        wall_forces = wall_calculator.get_forces(atoms)
        max_force = float(np.linalg.norm(wall_forces, axis=1).max())
        with path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [dyn.nsteps, dyn.nsteps * TIMESTEP_FS, left, right, wall_energy, max_force]
            )

    dyn.attach(write_row, interval=HILL_INTERVAL)


def main() -> None:
    args = parse_args()
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    trajectory_path = outputs / f"{args.run_name}.traj"
    bias_path = outputs / f"{args.run_name}.bias"
    diagnostics_path = outputs / f"{args.run_name}.wall.csv"
    continue_run = args.chunk_id > 0 and trajectory_path.exists() and bias_path.exists()
    if args.chunk_id > 0 and not continue_run:
        raise FileNotFoundError(
            f"Chunk {args.chunk_id} requires {trajectory_path} and {bias_path}"
        )

    atoms = (
        read(f"{trajectory_path}@-1").copy()
        if continue_run
        else read(args.start_file).copy()
    )
    atoms.set_pbc(False)

    physical_calculator = build_physical_calculator(args.model, args.model_path)
    wall_calculator = build_angle_wall(args.system)
    atoms.calc = LinearCombinationCalculator(
        [physical_calculator, wall_calculator], [1.0, 1.0]
    )

    if not continue_run:
        BFGS(atoms, logfile=str(outputs / f"bfgs_{args.run_name}.log")).run(
            fmax=0.05, steps=500
        )
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
    elif atoms.get_velocities() is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)

    dyn = WT_Metadynamics(
        atoms=atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
        trajectory=str(trajectory_path),
        fixcm=True,
        cvs=build_metad_cvs(args.system),
        std_dev=HILL_SIGMA_DEG,
        bias_height=BIAS_HEIGHT_EV,
        interval_size=HILL_INTERVAL,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=BIAS_FACTOR,
        append_trajectory=continue_run,
        input_file=str(bias_path) if continue_run else None,
        wrapping=[True, False],
        bounds=((-180.0, 180.0), (0.0, 180.0)),
        max_bias=1_000_000,
        loginterval=args.trajectory_loginterval,
    )
    attach_wall_diagnostics(
        dyn,
        atoms,
        wall_calculator,
        diagnostics_path,
        args.system,
        append=continue_run,
    )

    try:
        dyn.run(args.steps)
    except Exception:
        job_id = os.environ.get("SLURM_JOB_ID", "nojob")
        dump_prefix = outputs / f"{args.run_name}_failure_{job_id}"
        write(dump_prefix.with_suffix(".xyz"), atoms, format="extxyz")
        if bias_path.exists():
            shutil.copy2(bias_path, dump_prefix.with_suffix(".bias"))
        raise

    write(
        outputs / f"{args.run_name}_chunk{args.chunk_id}_last.xyz",
        atoms,
        format="extxyz",
    )
    print(
        f"completed model={args.model} system={args.system} chunk={args.chunk_id} "
        f"steps={args.steps} continue={continue_run}"
    )


if __name__ == "__main__":
    main()
