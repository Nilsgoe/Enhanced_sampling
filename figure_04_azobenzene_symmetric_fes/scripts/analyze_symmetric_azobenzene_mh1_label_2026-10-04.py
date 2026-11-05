#!/usr/bin/env python3
"""Symmetrize cis/trans wall-MetaD FES estimates and compare CNN pathways."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw_bias" / "azobenzene_angle_wall_test_20260812"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_two_cnn_pathways import (
    CNN_FLEX_DEG,
    LOW_CNN_DEG,
    MODELS,
    ROOT,
    bias_file,
    pathway_indices,
    profile,
    reconstruct,
)

TEMPERATURE_K = 333.0
KB_KJMOL_K = 0.00831446261815324


def display_label(label: str) -> str:
    return "MACE-MH-1" if label == "MACE-MH1" else label


def symmetric_fes(model):
    phi, cnn, cis_fes, cis_time = reconstruct(bias_file(model, "cis"))
    trans_phi, trans_cnn, trans_fes, trans_time = reconstruct(bias_file(model, "trans"))
    if not (np.array_equal(phi, trans_phi) and np.array_equal(cnn, trans_cnn)):
        raise ValueError(f"Grid mismatch for {model.label}")
    # Each history has its own arbitrary additive constant. Align before averaging.
    combined = 0.5 * ((cis_fes - cis_fes.min()) + (trans_fes - trans_fes.min()))
    combined -= combined.min()
    # phi grid is symmetric from -180 to +180, so reversal maps phi to -phi.
    symmetric = 0.5 * (combined + combined[:, ::-1])
    symmetric -= symmetric.min()
    return phi, cnn, symmetric, cis_time, trans_time


def basin_free_energy(fes: np.ndarray, mask: np.ndarray, dphi: float, dcnn: float) -> float:
    """Numerically evaluate -kT ln integral exp(-F/kT) over one CV basin."""
    values = fes[mask]
    kbt = KB_KJMOL_K * TEMPERATURE_K
    reference = values.min()
    integral = np.exp(-(values - reference) / kbt).sum() * dphi * dcnn
    return float(reference - kbt * np.log(integral))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
    surfaces, rows = [], []
    for model in MODELS:
        phi, cnn, fes, cis_time, trans_time = symmetric_fes(model)
        low_x, low_y = pathway_indices(phi, cnn, fes, "cis", high=False)
        high_x, high_y = pathway_indices(phi, cnn, fes, "cis", high=True)
        low_profile, low_barrier = profile(fes, low_x, low_y)
        high_profile, high_barrier = profile(fes, high_x, high_y)
        low_cnn = np.abs(cnn - LOW_CNN_DEG) <= CNN_FLEX_DEG
        cis_basin = (np.abs(phi) <= 30.0)[None, :] & low_cnn[:, None]
        trans_basin = (np.abs(phi) >= 150.0)[None, :] & low_cnn[:, None]
        cis_free_energy = basin_free_energy(fes, cis_basin, phi[1] - phi[0], cnn[1] - cnn[0])
        trans_free_energy = basin_free_energy(fes, trans_basin, phi[1] - phi[0], cnn[1] - cnn[0])
        relative_stability = trans_free_energy - cis_free_energy
        preferred = "CNN 120 +/- 15 deg" if low_barrier <= high_barrier else "CNN 155-175 deg"
        surfaces.append((model, phi, cnn, fes, low_x, low_y, high_x, high_y, low_profile, high_profile, low_barrier, high_barrier))
        rows.append({
            "model": model.label,
            "cis_time_ns": cis_time,
            "trans_time_ns": trans_time,
            "barrier_cnn120_kjmol": low_barrier,
            "barrier_cnn155_175_kjmol": high_barrier,
            "delta_high_minus_low_kjmol": high_barrier - low_barrier,
            "delta_g_trans_minus_cis_kjmol": relative_stability,
            "preferred_path": preferred,
        })

    vmax = max(float(np.percentile(surface[3], 99.5)) for surface in surfaces)
    levels = np.linspace(0.0, max(vmax, 1.0), 18)
    figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), sharex=True, sharey=True)
    contour = None
    for axis, (model, phi, cnn, fes, low_x, low_y, high_x, high_y, *_rest) in zip(axes.ravel(), surfaces, strict=True):
        contour = axis.contourf(phi, cnn, np.clip(fes, 0, vmax), levels=levels, cmap="RdBu_r")
        axis.plot(phi[low_x], cnn[low_y], color="black", linewidth=1.8, label="CNN 120 +/- 15 deg")
        axis.plot(phi[high_x], cnn[high_y], color="#d18b00", linewidth=1.8, label="CNN 155-175 deg")
        axis.set_title(display_label(model.label), fontsize=11)
        axis.set(xlim=(-180, 180), ylim=(60, 180), xticks=[-180, -90, 0, 90, 180], yticks=[60, 90, 120, 150, 180])
        axis.tick_params(direction="in", top=True, right=True)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\phi$ (deg)")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\theta$ (deg)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.colorbar(contour, ax=axes.ravel().tolist(), pad=0.018, label="Relative free energy (kJ mol-1)")
    figure.savefig(OUTPUT_DIR / "azobenzene_symmetric_combined_cnn_pathways_fes.png", dpi=500, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "azobenzene_symmetric_combined_cnn_pathways_fes.pdf", dpi=400, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), sharex=True, sharey=True)
    contour = None
    for axis, (model, phi, cnn, fes, *_rest) in zip(axes.ravel(), surfaces, strict=True):
        contour = axis.contourf(phi, cnn, np.clip(fes, 0, vmax), levels=levels, cmap="RdBu_r")
        axis.set_title(display_label(model.label), fontsize=13)
        axis.set(xlim=(-180, 180), ylim=(60, 180), xticks=[-180, -90, 0, 90, 180], yticks=[60, 90, 120, 150, 180])
        axis.tick_params(direction="in", top=True, right=True, width=1.2)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$\phi$ (Deg)", fontsize=14)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\theta$ (Deg)", fontsize=14)
    colorbar = figure.colorbar(
        contour,
        ax=axes.ravel().tolist(),
        pad=0.018,
        label=r"Relative Free Energy (kJ mol$^{-1}$)",
    )
    colorbar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=14)
    colorbar.ax.tick_params(width=1.2)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    colorbar.outline.set_linewidth(1.8)
    figure.savefig(OUTPUT_DIR / "azobenzene_symmetric_combined_fes.png", dpi=500, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "azobenzene_symmetric_combined_fes.pdf", dpi=400, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(15.6, 6.8), sharey=False)
    for axis, (model, _phi, _cnn, _fes, _low_x, _low_y, _high_x, _high_y, low, high, low_barrier, high_barrier) in zip(axes.ravel(), surfaces, strict=True):
        axis.plot(np.linspace(0, 1, len(low)), low, color="black", label=f"CNN 120 +/- 15: {low_barrier:.1f}")
        axis.plot(np.linspace(0, 1, len(high)), high, color="#d18b00", label=f"CNN 155-175: {high_barrier:.1f}")
        axis.set_title(display_label(model.label), fontsize=11)
        axis.set_xlabel("Path progress")
        axis.set_ylabel("Delta F (kJ mol-1)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT_DIR / "azobenzene_symmetric_combined_cnn_pathways_profiles.png", dpi=500, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "azobenzene_symmetric_combined_cnn_pathways_profiles.pdf", dpi=400, bbox_inches="tight")
    plt.close(figure)

    with (OUTPUT_DIR / "azobenzene_symmetric_combined_cnn_pathways_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Model & $\Delta F^\ddagger_{120}$ & $\Delta F^\ddagger_{155-175}$ & $\Delta G_{\mathrm{trans-cis}}$ \\",
        r" & \multicolumn{3}{c}{(kJ mol$^{-1}$)} \\",
        r"\hline",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['model']} & {row['barrier_cnn120_kjmol']:.1f} & "
            f"{row['barrier_cnn155_175_kjmol']:.1f} & {row['delta_g_trans_minus_cis_kjmol']:.1f} \\\\"
        )
    tex_lines.extend([r"\hline", r"\end{tabular}"])
    (OUTPUT_DIR / "azobenzene_symmetric_combined_barriers_stability.tex").write_text("\n".join(tex_lines) + "\n")


if __name__ == "__main__":
    main()
