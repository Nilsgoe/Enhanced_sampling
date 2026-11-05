#!/usr/bin/env python
"""Create full GPR-Umbrella-Sampling-Analysis diagnostics for explicit OFF24-M.

The source data are the completed FHI-Raccoon MACE-OFF24-M umbrella windows.
Most raw CVs are retained every 5 fs after the production driver's 10 ps
burn-in.  The central five windows are rerun at every 0.5 fs step.  This
script applies the Figure-3 1+9 ps selection in physical time to both forms.
"""

from __future__ import annotations

import csv
import argparse
import re
import sys
from pathlib import Path

import numpy as np


FIGURE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = FIGURE_ROOT / "data" / "off24_raccoon_20261011"
OUTPUT_ROOT = FIGURE_ROOT / "data" / "gpr_umbrella_evaluation_off24_m"
GPR_REPOSITORY = Path(
    "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/external/GPR-Umbrella-Sampling-Analysis"
)

KAPPA_EV_PER_A2 = 50.0
KJ_MOL_PER_EV = 96.485
DISCARD_FRAMES = 200
USE_FRAMES = 1800
SHIFT_RE = re.compile(r"_shift_(-?\d+)")

sys.path.insert(0, str(GPR_REPOSITORY))
from gpr_umbrella_1d import gpr_umbrella_integration, plot_diagnostics  # noqa: E402


def shift_from_path(path: Path) -> int:
    match = SHIFT_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not determine window shift from {path.name}")
    return int(match.group(1))


def center_from_mean_file(path: Path) -> float:
    with path.open() as handle:
        return float(next(csv.reader(handle))[0])


def add_preferred(mapping: dict[int, Path], path: Path) -> None:
    shift = shift_from_path(path)
    old = mapping.get(shift)
    if old is None or ("dense_center" in path.name and "dense_center" not in old.name):
        mapping[shift] = path


def selection_for_raw(path: Path) -> tuple[int, int, float]:
    if "dense_center" in path.name:
        return 2_000, 18_000, 0.5
    return DISCARD_FRAMES, USE_FRAMES, 5.0


def prepare_windows(system: str) -> tuple[Path, list[int]]:
    source = DATA_ROOT / system / "outputs"
    processed = OUTPUT_ROOT / system / "processed_windows"
    processed.mkdir(parents=True, exist_ok=True)

    means: dict[int, Path] = {}
    raws: dict[int, Path] = {}
    for path in source.glob("mean_cv_energy_*.csv"):
        add_preferred(means, path)
    for path in source.glob("cv_energy_*.csv"):
        add_preferred(raws, path)
    shifts = sorted(set(means) & set(raws))
    if not shifts:
        raise RuntimeError(f"No matched raw/mean window files for {system}")

    kappa_kj_mol_a2 = KAPPA_EV_PER_A2 * KJ_MOL_PER_EV
    for index, shift in enumerate(shifts):
        raw = np.genfromtxt(raws[shift], delimiter=",", names=True)
        discard, use, spacing_fs = selection_for_raw(raws[shift])
        cv = np.atleast_1d(raw["cv"])[discard : discard + use]
        if len(cv) != use:
            raise RuntimeError(f"{raws[shift]} has only {len(cv)} selected frames")
        center = center_from_mean_file(means[shift])
        # The established GPR code reads columns 1, 2, 3 as CV, centre,
        # kappa and converts kappa from kJ mol^-1 A^-2 to eV A^-2.
        data = np.column_stack(
            [np.arange(len(cv), dtype=float) * spacing_fs, cv, np.full(len(cv), center), np.full(len(cv), kappa_kj_mol_a2)]
        )
        np.savetxt(
            processed / f"window_{index:03d}.ui_dat",
            data,
            header=(f"shift={shift}; time_fs cv_A centre_A kappa_kJ_mol_A2"),
            fmt="%.10f",
        )
    return processed, shifts


def run_system(system: str) -> None:
    processed, shifts = prepare_windows(system)
    output = OUTPUT_ROOT / system
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{system}_mace_off24_m_1ps_discard_9ps"
    results = gpr_umbrella_integration(
        data_folder=str(processed),
        cv_unit="A",
        energy_unit="eV",
        output_prefix=prefix,
        output_dir=str(output),
        max_lag=1000,
        n_star=500,
        plot=True,
        save_fig=True,
        figure_dpi=300,
        show=False,
        verbose=True,
    )
    # Save the same full diagnostic figure in vector format as well.
    fig = plot_diagnostics(results, output_prefix=prefix)
    fig.savefig(output / f"{prefix}_gpr_analysis.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    with (output / "analysis_provenance.txt").open("w") as handle:
        handle.write(f"GPR repository: {GPR_REPOSITORY}\n")
        handle.write(f"windows: {len(shifts)}; shifts: {min(shifts)} through {max(shifts)}\n")
        handle.write(f"kappa: {KAPPA_EV_PER_A2} eV A^-2\n")
        handle.write("selected data: 1 ps discard + 9 ps analysis per window; 5 dense windows use 0.5 fs frames, all others 5 fs.\n")
        handle.write(f"lengthscale: {results['lengthscale']:.8f} A\n")
        handle.write(f"sigma_f: {results['sigma_f']:.8f} eV\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("malonaldehyd", "f-malonaldehyd"))
    args = parser.parse_args()
    for system in ((args.system,) if args.system else ("malonaldehyd", "f-malonaldehyd")):
        run_system(system)


if __name__ == "__main__":
    main()
