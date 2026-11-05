import argparse
import csv
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from upet.calculator import UPETCalculator

from biASE import General_Bias_Calculator


def parse_args():
    parser = argparse.ArgumentParser(description="Umbrella MD with UPET PET-SPICE.")
    parser.add_argument("shift", type=int, help="Integer shift index (-5 to 35).")
    return parser.parse_args()


def compute_cv(positions):
    d_o4h8 = jnp.linalg.norm(positions[4] - positions[8])
    d_o3h8 = jnp.linalg.norm(positions[3] - positions[8])
    return d_o4h8 - d_o3h8


def umbrella_potential(x, x0, k):
    return 0.5 * k * (x - x0) ** 2


def build_calculator():
    return UPETCalculator(model="pet-spice-l", device="cuda")


def input_xyz() -> str:
    for candidate in ("optimized_malonaldehyde_initial.xyz", "optimized_fmalonaldehyde_initial.xyz"):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Missing optimized_malonaldehyde_initial.xyz / optimized_fmalonaldehyde_initial.xyz")


def main():
    args = parse_args()
    run_tag = f"viper_upet_pet_spice_shift_{args.shift}"

    upet_calc = build_calculator()
    atoms_initial = read(input_xyz())
    center = compute_cv(atoms_initial.positions) + 0.05 * args.shift
    umbrella_bias_calc = General_Bias_Calculator(
        cv_function=compute_cv,
        bias_function=partial(umbrella_potential, x0=center, k=50.0),
    )
    biased_calc = SumCalculator([upet_calc, umbrella_bias_calc])

    atoms = atoms_initial.copy()
    atoms.calc = biased_calc
    BFGS(atoms, logfile=f"outputs/bfgs_{run_tag}.log").run(fmax=0.05, steps=500)

    MaxwellBoltzmannDistribution(atoms, temperature_K=293)
    trajectory_path = f"outputs/umd_{run_tag}.traj"
    dyn = Langevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=293,
        friction=0.1,
        trajectory=trajectory_path,
        fixcm=True,
    )
    dyn.run(int(5e4))

    traj = read(f"{trajectory_path}@2000::1")
    cv_list, energy_list = [], []
    cv_energy_path = f"outputs/cv_energy_{run_tag}.csv"
    mean_energy_path = f"outputs/mean_cv_energy_{run_tag}.csv"

    with open(cv_energy_path, "w", newline="") as f:
        csv.writer(f).writerow(["cv", "energy"])

    for snapshot in traj:
        snapshot.calc = biased_calc
        energy = snapshot.get_total_energy()
        cv = compute_cv(snapshot.positions)
        with open(cv_energy_path, "a+", newline="") as f:
            csv.writer(f).writerow([cv, energy])
        cv_list.append(cv)
        energy_list.append(energy)

    with open(mean_energy_path, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                center,
                np.mean(np.array(cv_list)),
                -50 * (np.mean(np.array(cv_list)) - center),
                np.mean(np.array(energy_list)),
                compute_cv(atoms_initial.positions),
            ]
        )


if __name__ == "__main__":
    main()
