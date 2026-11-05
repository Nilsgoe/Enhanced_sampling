#!/usr/bin/env python3
"""Plot unsymmetrized cis/trans wall-MetaD FESs and their two pathways."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_two_cnn_pathways import MODELS, bias_file, pathway_indices, reconstruct  # noqa: E402


def display_label(label: str) -> str:
    return "MACE-MH-1" if label == "MACE-MH1" else label


def style_axis(axis: plt.Axes, index: int) -> None:
    row, column = divmod(index, 4)
    axis.set(
        xlim=(-180, 180),
        ylim=(60, 180),
        xticks=(-180, -90, 0, 90, 180),
        yticks=(60, 90, 120, 150, 180),
    )
    axis.tick_params(direction="in", top=True, right=True, width=1.3)
    for spine in axis.spines.values():
        spine.set_linewidth(1.8)
    if row == 1:
        axis.set_xlabel(r"$\phi$ (Deg)", fontsize=14)
    if column == 0:
        axis.set_ylabel(r"$\theta$ (Deg)", fontsize=14)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    surfaces = {}
    for system in ("cis", "trans"):
        system_surfaces = []
        for model in MODELS:
            phi, theta, fes, _time_ns = reconstruct(bias_file(model, system))
            low_x, low_y = pathway_indices(phi, theta, fes, system, high=False)
            high_x, high_y = pathway_indices(phi, theta, fes, system, high=True)
            system_surfaces.append((model, phi, theta, fes, low_x, low_y, high_x, high_y))
        surfaces[system] = system_surfaces

    vmax = float(
        np.ceil(
            max(
                np.percentile(fes[np.isfinite(fes)], 99.5)
                for system_surfaces in surfaces.values()
                for _model, _phi, _theta, fes, *_paths in system_surfaces
            )
        )
    )
    levels = np.linspace(0.0, max(vmax, 1.0), 18)

    for system, system_surfaces in surfaces.items():
        figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), sharex=True, sharey=True)
        contour = None
        for index, (axis, (model, phi, theta, fes, low_x, low_y, high_x, high_y)) in enumerate(
            zip(axes.ravel(), system_surfaces, strict=True)
        ):
            contour = axis.contourf(phi, theta, np.clip(fes, 0.0, vmax), levels=levels, cmap="RdBu_r")
            axis.plot(phi[low_x], theta[low_y], color="black", linewidth=1.8, label=r"$\theta=120\pm15^\circ$")
            axis.plot(phi[high_x], theta[high_y], color="#d18b00", linewidth=1.8, label=r"$\theta=155$--$175^\circ$")
            axis.set_title(display_label(model.label))
            style_axis(axis, index)
        axes[0, 0].legend(loc="upper left", fontsize=9, frameon=True)
        colorbar = figure.colorbar(
            contour,
            ax=axes.ravel().tolist(),
            pad=0.018,
            label=r"Relative Free Energy (kJ mol$^{-1}$)",
        )
        colorbar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=14)
        colorbar.ax.tick_params(width=1.3)
        colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        colorbar.outline.set_linewidth(1.8)
        for extension, dpi in (("png", 500), ("pdf", 400)):
            figure.savefig(
                OUTPUT_DIR / f"azobenzene_{system}_initialized_wall_fes_8model_pathways.{extension}",
                dpi=dpi,
                bbox_inches="tight",
            )
        plt.close(figure)


if __name__ == "__main__":
    main()
