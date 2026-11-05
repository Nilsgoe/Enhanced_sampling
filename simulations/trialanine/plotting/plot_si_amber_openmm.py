#!/usr/bin/env python3
"""Plot matching 1x4 ff14SB/TIP3P bIASE and OpenMM replica FESs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad")
OUTPUT_DIR = ROOT / "analysis/trialanine_metad/amber14_biase_openmm_4rep_rows_kj_2026-10-09"
GRID_POINTS = 181
CHUNK_SIZE = 384
EV_TO_KJMOL = 96.48533212331002


def wrap_degrees(values: np.ndarray | float) -> np.ndarray | float:
    return ((values + 180.0) % 360.0) - 180.0


def reconstruct_biase_fes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 8:
                continue
            try:
                row = [float(fields[index]) for index in range(8)]
            except ValueError:
                continue
            if np.isfinite(row).all():
                rows.append(row)
    if not rows:
        raise ValueError(f"No valid hills in {path}")
    hills = np.asarray(rows, dtype=np.float64)
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.float64)
    for start in range(0, len(hills), CHUNK_SIZE):
        block = hills[start : start + CHUNK_SIZE]
        dphi = wrap_degrees(grid[None, :] - wrap_degrees(block[:, 1])[:, None])
        dpsi = wrap_degrees(grid[None, :] - wrap_degrees(block[:, 2])[:, None])
        gphi = np.exp(-0.5 * (dphi / block[:, 3, None]) ** 2)
        gpsi = np.exp(-0.5 * (dpsi / block[:, 4, None]) ** 2)
        bias += np.einsum("n,nx,ny->xy", block[:, 5], gphi, gpsi, optimize=True)
    fes = -bias.T
    fes -= np.nanmin(fes)
    return grid, fes * EV_TO_KJMOL


def load_openmm_fes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    fes = np.asarray(np.load(path), dtype=np.float64)
    if fes.shape != (GRID_POINTS, GRID_POINTS):
        raise ValueError(f"Unexpected OpenMM FES shape {fes.shape}: {path}")
    fes -= np.nanmin(fes)
    # Native OpenMM grids are already in kJ mol^-1.
    return grid, fes


def style_axis(axis: plt.Axes, column: int) -> None:
    axis.set(xlim=(-180, 180), ylim=(-180, 180))
    axis.set_xticks((-180, -90, 0, 90, 180))
    axis.set_yticks((-180, -90, 0, 90, 180))
    axis.set_xlabel(r"$\phi$ (Deg)", fontsize=16)
    if column == 0:
        axis.set_ylabel(r"$\psi$ (Deg)", fontsize=16)
    axis.tick_params(direction="in", top=True, right=True, length=5.0, width=1.5, labelsize=13)
    axis.minorticks_on()
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=2.8, width=1.1)
    for spine in axis.spines.values():
        spine.set_linewidth(2.0)


def make_figure(label: str, surfaces: list[tuple[np.ndarray, np.ndarray]], stem: str, vmax: float) -> None:
    levels = np.linspace(0.0, max(vmax, 1.0), 19)
    figure, axes = plt.subplots(1, 4, figsize=(15.8, 4.15), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.90, bottom=0.16, top=0.80, wspace=0.16)
    contour = None
    for index, (axis, (grid, fes)) in enumerate(zip(axes, surfaces, strict=True), start=1):
        contour = axis.contourf(grid, grid, np.clip(fes, 0.0, vmax), levels=levels, cmap="RdBu_r")
        axis.set_title(f"Replica {index}", pad=9)
        style_axis(axis, index - 1)
    figure.text(0.5, 0.94, label, ha="center", va="center", fontsize=18)
    colorbar = figure.colorbar(contour, ax=axes.tolist(), pad=0.018, fraction=0.035)
    colorbar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=16)
    colorbar.ax.tick_params(width=1.5, labelsize=13)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    colorbar.outline.set_linewidth(2.0)
    for extension, dpi in (("png", 600), ("pdf", 400)):
        figure.savefig(OUTPUT_DIR / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def make_combined_figure(
    biase: list[tuple[np.ndarray, np.ndarray]], openmm: list[tuple[np.ndarray, np.ndarray]], vmax: float
) -> None:
    levels = np.linspace(0.0, max(vmax, 1.0), 19)
    figure, axes = plt.subplots(2, 4, figsize=(15.8, 7.8), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.915, bottom=0.10, top=0.88, wspace=0.16, hspace=0.55)
    contour = None
    for row, surfaces in enumerate((biase, openmm)):
        for column, (grid, fes) in enumerate(surfaces):
            axis = axes[row, column]
            contour = axis.contourf(grid, grid, np.clip(fes, 0.0, vmax), levels=levels, cmap="RdBu_r")
            axis.set_title(f"Replica {column + 1}", pad=8)
            axis.set(xlim=(-180, 180), ylim=(-180, 180))
            axis.set_xticks((-180, -90, 0, 90, 180))
            axis.set_yticks((-180, -90, 0, 90, 180))
            axis.tick_params(direction="in", top=True, right=True, length=5.0, width=1.5, labelsize=13)
            axis.minorticks_on()
            axis.tick_params(which="minor", direction="in", top=True, right=True, length=2.8, width=1.1)
            for spine in axis.spines.values():
                spine.set_linewidth(2.0)
            if row == 1:
                axis.set_xlabel(r"$\phi$ (Deg)", fontsize=16)
            if column == 0:
                axis.set_ylabel(r"$\psi$ (Deg)", fontsize=16)
    header_offset = 0.065
    figure.text(0.5, axes[0, 0].get_position().y1 + header_offset, "AMBER ff14SB/TIP3P with bIASE", ha="center", va="center", fontsize=18)
    figure.text(0.5, axes[1, 0].get_position().y1 + header_offset, "AMBER ff14SB/TIP3P with OpenMM", ha="center", va="center", fontsize=18)
    colorbar = figure.colorbar(contour, ax=axes.ravel().tolist(), pad=0.018, fraction=0.035)
    colorbar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=16)
    colorbar.ax.tick_params(width=1.5, labelsize=13)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    colorbar.outline.set_linewidth(2.0)
    for extension, dpi in (("png", 600), ("pdf", 400)):
        figure.savefig(
            OUTPUT_DIR / f"trialanine_amber14_biase_openmm_4rep_combined.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    biase_dir = ROOT / "amber_reference_metad_1kjmol_0p5fs/outputs"
    openmm_dir = ROOT / "openmm_amber_metad/outputs_bias_only"
    biase = [
        reconstruct_biase_fes(biase_dir / f"solution_amber_biase_1kjmol_0p5fs_rep{rep}_indep_1ns.bias")
        for rep in range(1, 5)
    ]
    openmm = [
        load_openmm_fes(
            openmm_dir / f"solution_openmm_wtmetad_1kjmol_15deg_0p5fs_rep{rep}_cvfes_notraj_1ns.free_energy_kjmol.npy"
        )
        for rep in range(1, 5)
    ]
    vmax = float(
        np.ceil(
            max(
                np.percentile(fes[np.isfinite(fes)], 99.5)
                for _grid, fes in [*biase, *openmm]
            )
        )
    )
    make_figure("AMBER ff14SB/TIP3P with bIASE", biase, "trialanine_amber14_biase_4rep", vmax)
    make_figure("AMBER ff14SB/TIP3P with OpenMM", openmm, "trialanine_amber14_openmm_4rep", vmax)
    make_combined_figure(biase, openmm, vmax)


if __name__ == "__main__":
    main()
