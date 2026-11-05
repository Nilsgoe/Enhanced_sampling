#!/usr/bin/env python3
"""Native 8-model azobenzene FES plots from archived 3-ns grid data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


DATA = Path(__file__).resolve().parents[1] / "data" / "3ns_reconstructed"
PET_CIS = DATA / "pet_spice_cis_stable_cv_3.0ns_reconstructed_fes.npz"
OUT = Path(__file__).resolve().parents[1] / "data"
EV_TO_KJMOL = 96.485
MODELS = (
    ("off", "MACE-OFF23(M)"),
    ("mh1", "MACE-MH-1"),
    ("polar", "MACE-POLAR-1-M"),
    ("pet_spice", "PET-SPICE-L"),
    ("so3lr", "SO3LR"),
    ("scratch", "MACE from scratch"),
    ("ft_so3lr", "Fine-tuned SO3LR"),
    ("ft_mh1", "Fine-tuned MACE-MH1"),
)


def grid(model: str, system: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = PET_CIS if (model, system) == ("pet_spice", "cis") else DATA / f"azobenzene_{system}_{model}_2d_3.0ns_reconstructed_fes.npz"
    values = np.load(source)
    fes = np.asarray(values["free_energy_eV"], dtype=float) * EV_TO_KJMOL
    fes -= np.nanmin(fes)
    return np.asarray(values["cv1_deg"], dtype=float), np.asarray(values["cv2_deg"], dtype=float), fes


def paths(model: str, system: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    results = []
    for angle in ("low_angle_120", "high_angle_160"):
        file = DATA / "two_angle_paths" / f"azobenzene_{system}_{model}_3.0ns_{angle}_path.csv"
        with file.open() as handle:
            rows = list(csv.DictReader(handle))
        results.append((
            np.asarray([float(row["cv1_CNNC_deg"]) for row in rows]),
            np.asarray([float(row["cv2_CNN_deg"]) for row in rows]),
        ))
    return results[0], results[1]


def finish_axes(axes: np.ndarray) -> None:
    for index, axis in enumerate(axes.ravel()):
        row, column = divmod(index, 4)
        axis.set(xlim=(-180, 180), ylim=(60, 180), xticks=(-180, -90, 0, 90, 180), yticks=(60, 90, 120, 150, 180))
        axis.tick_params(direction="in", top=True, right=True, width=1.3)
        for spine in axis.spines.values():
            spine.set_linewidth(1.8)
        if row == 1:
            axis.set_xlabel(r"$\phi$ (Deg)", fontsize=14)
        if column == 0:
            axis.set_ylabel(r"$\theta$ (Deg)", fontsize=14)


def save(figure: plt.Figure, stem: str) -> None:
    for extension, dpi in (("png", 500), ("pdf", 400)):
        figure.savefig(OUT / f"{stem}.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_individual(surfaces: dict[str, list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]], vmax: float) -> None:
    levels = np.linspace(0.0, vmax, 18)
    for system, entries in surfaces.items():
        figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), sharex=True, sharey=True)
        for axis, (model, title, phi, theta, fes) in zip(axes.ravel(), entries, strict=True):
            contour = axis.contourf(phi, theta, np.clip(fes.T, 0.0, vmax), levels=levels, cmap="RdBu_r")
            (low_phi, low_theta), (high_phi, high_theta) = paths(model, system)
            axis.plot(low_phi, low_theta, color="black", linewidth=1.8, label=r"$\theta=120\pm15^\circ$")
            axis.plot(high_phi, high_theta, color="#d18b00", linewidth=1.8, label=r"$\theta=155$--$175^\circ$")
            axis.set_title(title, fontsize=13)
        finish_axes(axes)
        axes[0, 0].legend(loc="upper left", fontsize=9, frameon=True)
        bar = figure.colorbar(contour, ax=axes.ravel().tolist(), pad=0.018, label=r"Relative Free Energy (kJ mol$^{-1}$)")
        bar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=14)
        bar.ax.tick_params(width=1.3)
        bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        bar.outline.set_linewidth(1.8)
        save(figure, f"azobenzene_{system}_initialized_wall_fes_8model_pathways")


def plot_symmetric(surfaces: dict[str, list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]], vmax: float) -> None:
    levels = np.linspace(0.0, vmax, 18)
    figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), sharex=True, sharey=True)
    for axis, cis, trans in zip(axes.ravel(), surfaces["cis"], surfaces["trans"], strict=True):
        model, title, phi, theta, cis_fes = cis
        trans_fes = trans[-1]
        symmetric = 0.5 * (cis_fes + trans_fes)
        symmetric = 0.5 * (symmetric + symmetric[::-1, :])
        symmetric -= np.nanmin(symmetric)
        contour = axis.contourf(phi, theta, np.clip(symmetric.T, 0.0, vmax), levels=levels, cmap="RdBu_r")
        axis.set_title(title, fontsize=13)
    finish_axes(axes)
    bar = figure.colorbar(contour, ax=axes.ravel().tolist(), pad=0.018, label=r"Relative Free Energy (kJ mol$^{-1}$)")
    bar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=14)
    bar.ax.tick_params(width=1.3)
    bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    bar.outline.set_linewidth(1.8)
    save(figure, "azobenzene_symmetric_combined_fes")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    surfaces = {system: [(model, title, *grid(model, system)) for model, title in MODELS] for system in ("cis", "trans")}
    vmax = float(np.ceil(max(np.nanpercentile(entry[-1], 99.5) for entries in surfaces.values() for entry in entries)))
    plot_individual(surfaces, vmax)
    plot_symmetric(surfaces, vmax)


if __name__ == "__main__":
    main()
