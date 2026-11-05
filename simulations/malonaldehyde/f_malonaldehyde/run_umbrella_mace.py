import argparse
import csv
from functools import partial

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from mace.calculators import mace_mp, mace_off, mace_omol, mace_polar

from biASE import General_Bias_Calculator


def parse_args():
    parser = argparse.ArgumentParser(description='Umbrella MD shifted from the CV by input * 0.05.')
    parser.add_argument('shift', type=int, help='Integer shift index.')
    parser.add_argument(
        '--model-family',
        choices=['off', 'omol', 'polar', 'mh1'],
        default='off',
        help='Model family: off, omol, polar, or mh1.',
    )
    parser.add_argument(
        '--model-size',
        default='large',
        help=(
            'Model size. off: small/medium/large; omol: extra_large; '
            'polar: s/m/l; mh1: mh-1 (used as model identifier).'
        ),
    )
    parser.add_argument(
        '--run-label',
        default='',
        help='Optional extra label to include in output filenames.',
    )
    return parser.parse_args()


def compute_cv(positions):
    d_o4h8 = jnp.linalg.norm(positions[4] - positions[8])
    d_o3h8 = jnp.linalg.norm(positions[3] - positions[8])
    return d_o4h8 - d_o3h8


def umbrella_potential(x, x0, k):
    dx = x - x0
    return 0.5 * k * dx**2


def build_calculator(model_family: str, model_size: str):
    if model_family == 'off':
        if model_size == 'extra_large':
            raise ValueError('MACE-OFF does not support model-size extra_large.')
        if model_size == 'off24':
            return mace_off(
                checkpoint_path='MACE-OFF24_medium.model',
                dispersion=True,
                enable_cueq=True,
            )
        return mace_off(model=model_size, dispersion=True, enable_cueq=True)
    if model_family == 'omol':
        if model_size != 'extra_large':
            raise ValueError('OMOL is expected to use model-size extra_large.')
        return mace_omol(model=model_size, enable_cueq=True)
    if model_family == 'polar':
        polar_key = f'polar-1-{model_size}'  # s/m/l → polar-1-s/m/l
        return mace_polar(model=polar_key, enable_cueq=False)
    if model_family == 'mh1':
        # model_size is the model identifier (e.g. "mh-1"); omol head for organics
        return mace_mp(model=model_size, head="omol", enable_cueq=False)
    raise ValueError(f'Unsupported model family: {model_family}')


def main():
    args = parse_args()
    model_tag = f'{args.model_family}_{args.model_size}'
    run_tag = f'raccoon_{model_tag}_shift_{args.shift}'
    if args.run_label:
        run_tag = f'{run_tag}_{args.run_label}'

    mace_calc = build_calculator(args.model_family, args.model_size)
    atoms_initial = read('optimized_fmalonaldehyde_initial.xyz')
    center = compute_cv(atoms_initial.positions) + 0.05 * args.shift
    umbrella_bias_calc = General_Bias_Calculator(
        cv_function=compute_cv,
        bias_function=partial(umbrella_potential, x0=center, k=50.0),
    )
    biased_calc = SumCalculator([mace_calc, umbrella_bias_calc])

    atoms = atoms_initial.copy()
    atoms.calc = biased_calc
    BFGS(atoms, logfile=f'outputs/bfgs_{run_tag}.log').run(fmax=0.05, steps=500)

    MaxwellBoltzmannDistribution(atoms, temperature_K=293)
    trajectory_path = f'outputs/umd_{run_tag}.traj'
    dyn = Langevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=293,
        friction=0.1,
        trajectory=trajectory_path,
        fixcm=True,
    )

    n_steps = int(5e4)
    dyn.run(n_steps)

    traj = read(f'{trajectory_path}@2000::1')
    cv_list = []
    energy_list = []
    cv_energy_path = f'outputs/cv_energy_{run_tag}.csv'
    mean_energy_path = f'outputs/mean_cv_energy_{run_tag}.csv'

    with open(cv_energy_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['cv', 'energy'])

    for snapshot in traj:
        snapshot.calc = biased_calc
        energy = snapshot.get_total_energy()
        cv = compute_cv(snapshot.positions)
        with open(cv_energy_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cv, energy])
        cv_list.append(cv)
        energy_list.append(energy)

    with open(mean_energy_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                center,
                np.mean(np.array(cv_list)),
                -50 * (np.mean(np.array(cv_list)) - center),
                np.mean(np.array(energy_list)),
                compute_cv(atoms_initial.positions),
            ]
        )


if __name__ == '__main__':
    main()
