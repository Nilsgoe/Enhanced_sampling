#!/usr/bin/env python
"""Run the established GPR umbrella diagnostics for MH1 and verified OFF23."""

from __future__ import annotations

import csv
import argparse
import re
import sys
from pathlib import Path

import numpy as np


FIGURE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = FIGURE_ROOT / "data" / "legacy_mace_models"
OUTPUT_ROOT = FIGURE_ROOT / "data" / "gpr_umbrella_evaluation_legacy_mace"
GPR_REPOSITORY = Path(
    "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/external/GPR-Umbrella-Sampling-Analysis"
)
KAPPA_EV_PER_A2 = 50.0
KJ_MOL_PER_EV = 96.485
# Legacy raw CSVs were written every 0.5 fs.  These are the original
# Figure-3 1+9 ps settings (1 ps = 2,000 frames; 9 ps = 18,000 frames).
DISCARD_FRAMES = 2_000
USE_FRAMES = 18_000
SHIFT_RE = re.compile(r"_shift_(-?\d+)")

sys.path.insert(0, str(GPR_REPOSITORY))
from gpr_umbrella_1d import gpr_umbrella_integration, plot_diagnostics  # noqa: E402


def shift_from_path(path: Path) -> int:
    match = SHIFT_RE.search(path.name)
    if match is None:
        raise ValueError(f"No shift in {path.name}")
    return int(match.group(1))


def centre(path: Path) -> float:
    with path.open() as handle:
        return float(next(csv.reader(handle))[0])


def prepare_windows(system: str, model: str, stride: int, output_model: str) -> tuple[Path, list[int]]:
    source = DATA_ROOT / system / model / "outputs"
    destination = OUTPUT_ROOT / system / output_model / "processed_windows"
    destination.mkdir(parents=True, exist_ok=True)
    means = {shift_from_path(p): p for p in source.glob("mean_cv_energy_*.csv")}
    raws = {shift_from_path(p): p for p in source.glob("cv_energy_*.csv")}
    shifts = sorted(set(means) & set(raws))
    if not shifts:
        raise RuntimeError(f"No matched windows for {system}/{model}")
    kappa_kj = KAPPA_EV_PER_A2 * KJ_MOL_PER_EV
    for index, shift in enumerate(shifts):
        raw = np.genfromtxt(raws[shift], delimiter=",", names=True)
        # Decimate before the physical-time selection.  For stride=10, the
        # legacy 0.5 fs stream becomes the same 5 fs spacing as new OFF24-M.
        sampled_cv = np.atleast_1d(raw["cv"])[::stride]
        discard = DISCARD_FRAMES // stride
        use = USE_FRAMES // stride
        cv = sampled_cv[discard : discard + use]
        if len(cv) != use:
            raise RuntimeError(f"{raws[shift]}: selected {len(cv)} instead of {use} frames")
        values = np.column_stack(
            [np.arange(use) * (0.5 * stride), cv, np.full(use, centre(means[shift])), np.full(use, kappa_kj)]
        )
        np.savetxt(destination / f"window_{index:03d}.ui_dat", values,
                   header=f"shift={shift}; time_fs cv_A centre_A kappa_kJ_mol_A2", fmt="%.10f")
    return destination, shifts


def run(system: str, model: str, label: str, stride: int = 1) -> None:
    output_model = model if stride == 1 else f"{model}_every{stride}"
    windows, shifts = prepare_windows(system, model, stride, output_model)
    output = OUTPUT_ROOT / system / output_model
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{system}_{label}_1ps_discard_9ps"
    results = gpr_umbrella_integration(
        data_folder=str(windows), cv_unit="A", energy_unit="eV", output_prefix=prefix,
        output_dir=str(output), max_lag=1000, n_star=500, plot=True, save_fig=True,
        figure_dpi=300, show=False, verbose=True,
    )
    fig = plot_diagnostics(results, output_prefix=prefix)
    fig.savefig(output / f"{prefix}_gpr_analysis.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    with (output / "analysis_provenance.txt").open("w") as handle:
        handle.write(f"model: {label}\nwindows: {len(shifts)}; shifts {min(shifts)} through {max(shifts)}\n")
        handle.write("OFF23 provenance: files were formerly labelled off24, but runtime logs identify MACE-OFF23.\n" if model == "off23" else "")
        handle.write(f"kappa: {KAPPA_EV_PER_A2} eV A^-2\n")
        handle.write(f"raw stride: every {stride} frame(s); output spacing: {0.5 * stride:g} fs\n")
        handle.write(f"selection: discard {DISCARD_FRAMES // stride} saved frames (1 ps), use {USE_FRAMES // stride} saved frames (9 ps)\n")
        handle.write(f"lengthscale: {results['lengthscale']:.8f} A; sigma_f: {results['sigma_f']:.8f} eV\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("malonaldehyd", "f-malonaldehyd"))
    parser.add_argument("--model", choices=("mh1", "off23"))
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth raw frame before 1+9 ps selection.")
    args = parser.parse_args()
    if args.stride < 1 or DISCARD_FRAMES % args.stride or USE_FRAMES % args.stride:
        parser.error("--stride must be a positive divisor of 2000 and 18000")
    systems = (args.system,) if args.system else ("malonaldehyd", "f-malonaldehyd")
    models = (args.model,) if args.model else ("mh1", "off23")
    labels = {"mh1": "mace_mh1", "off23": "mace_off23"}
    for system in systems:
        for model in models:
            run(system, model, labels[model] if args.stride == 1 else f"{labels[model]}_every{args.stride}", args.stride)


if __name__ == "__main__":
    main()
