#!/usr/bin/env python
"""Eight-model solvated-trialanine comparison including MACE-LES and ff19SB/OPC."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FormatStrFormatter
import numpy as np


PROJECT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad")
ROOT = PROJECT.parents[0]
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
GRID_POINTS = 181
CHUNK_SIZE = 384
PLOT_LEVELS = 18
EV_TO_KJMOL = 96.48533212331002

MINIMA_MARKERS = [
    (r"a$\beta$", -135.0, 135.0),
    ("PPII", -60.0, 135.0),
    (r"$\alpha$H(R)", -60.0, -45.0),
    (r"$\alpha$H(L)", 60.0, 40.0),
]
MINIMUM_SEARCH_HALF_WIDTH_DEG = 40.0
LOCAL_MINIMUM_STEPS = 2
MARKER_DUPLICATE_DISTANCE_DEG = 2.0
MINIMUM_CONSTRAINTS = {
    r"a$\beta$": {"phi_max": -110.0, "psi_min": 90.0},
    "PPII": {"psi_min": 110.0},
}
EXCLUDED_MARKERS_BY_MODEL = {
    "PET-SPICE-L": {r"a$\beta$"},
    "SO3LR": {r"a$\beta$"},
}


@dataclass(frozen=True)
class Record:
    group: str
    label: str
    path: Path | None
    kind: str = "biase_hills"
    paths: tuple[Path, ...] = ()


def wrap_degrees(values: np.ndarray | float) -> np.ndarray | float:
    return ((values + 180.0) % 360.0) - 180.0


def read_hills(path: Path, time_limit_fs: float | None = None) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 8:
                continue
            try:
                row = [float(fields[index]) for index in range(8)]
            except ValueError:
                continue
            if time_limit_fs is not None and row[0] > time_limit_fs:
                continue
            if np.isfinite(row[:8]).all():
                rows.append(row)
    if not rows:
        raise ValueError(f"No finite MetaD hills found in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return {
        "time_fs": data[:, 0],
        "phi": wrap_degrees(data[:, 1]),
        "psi": wrap_degrees(data[:, 2]),
        "sigma_phi": data[:, 3],
        "sigma_psi": data[:, 4],
        "height": data[:, 5],
    }


def reconstruct_biase_fes(hills: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=np.float64)
    for start in range(0, len(hills["height"]), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(hills["height"]))
        dphi = wrap_degrees(grid[None, :] - hills["phi"][start:stop, None])
        dpsi = wrap_degrees(grid[None, :] - hills["psi"][start:stop, None])
        gphi = np.exp(-0.5 * (dphi / hills["sigma_phi"][start:stop, None]) ** 2)
        gpsi = np.exp(-0.5 * (dpsi / hills["sigma_psi"][start:stop, None]) ** 2)
        bias += np.einsum("n,nx,ny->xy", hills["height"][start:stop], gphi, gpsi, optimize=True)
    fes = -bias.T
    fes -= np.nanmin(fes)
    return grid, fes * EV_TO_KJMOL


def read_openmm_fes(path: Path) -> tuple[np.ndarray, np.ndarray, int, float]:
    grid = np.linspace(-180.0, 180.0, GRID_POINTS)
    # OpenMM's saved grid already maps correctly for contourf(grid, grid, fes):
    # x = first BiasVariable (phi), y = second BiasVariable (psi).
    fes = np.asarray(np.load(path), dtype=np.float64)
    if fes.shape != (GRID_POINTS, GRID_POINTS):
        raise ValueError(f"Unexpected OpenMM FES shape {fes.shape} in {path}")
    fes -= np.nanmin(fes)
    cv_path = path.with_name(path.name.replace(".free_energy_kjmol.npy", ".cv.csv"))
    n_samples = 0
    time_fs = float("nan")
    if cv_path.exists():
        with cv_path.open(errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    time_fs = float(row["time_fs"])
                    n_samples += 1
                except (KeyError, TypeError, ValueError):
                    continue
    # Native OpenMM grids are stored in kJ mol^-1.
    return grid, fes, n_samples, time_fs


def circular_delta(values: np.ndarray, center: float) -> np.ndarray:
    return wrap_degrees(values - center)


def local_minimum_in_window(
    grid: np.ndarray,
    fes: np.ndarray,
    center_phi: float,
    center_psi: float,
    constraints: dict[str, float],
):
    phi_indices = np.flatnonzero(
        np.abs(circular_delta(grid, center_phi)) <= MINIMUM_SEARCH_HALF_WIDTH_DEG
    )
    psi_indices = np.flatnonzero(
        np.abs(circular_delta(grid, center_psi)) <= MINIMUM_SEARCH_HALF_WIDTH_DEG
    )
    candidates: list[tuple[float, int, int]] = []
    n_psi, n_phi = fes.shape
    directions = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )

    for psi_index in psi_indices:
        for phi_index in phi_indices:
            phi = grid[phi_index]
            psi = grid[psi_index]
            if "phi_min" in constraints and phi <= constraints["phi_min"]:
                continue
            if "phi_max" in constraints and phi >= constraints["phi_max"]:
                continue
            if "psi_min" in constraints and psi <= constraints["psi_min"]:
                continue
            if "psi_max" in constraints and psi >= constraints["psi_max"]:
                continue
            value = fes[psi_index, phi_index]
            if not np.isfinite(value):
                continue
            neighbours = [
                fes[(psi_index + step * dpsi) % n_psi, (phi_index + step * dphi) % n_phi]
                for step in range(1, LOCAL_MINIMUM_STEPS + 1)
                for dpsi, dphi in directions
            ]
            if all(np.isfinite(neighbour) and neighbour >= value for neighbour in neighbours):
                candidates.append((float(value), psi_index, phi_index))

    if not candidates:
        return None
    _value, psi_index, phi_index = min(
        candidates,
        key=lambda candidate: (
            candidate[0],
            float(circular_delta(grid[candidate[2]], center_phi)) ** 2
            + float(circular_delta(grid[candidate[1]], center_psi)) ** 2,
        ),
    )
    return float(grid[phi_index]), float(grid[psi_index])


def selected_minima(grid: np.ndarray, fes: np.ndarray, model_label: str) -> list[tuple[str, float, float]]:
    selected: list[tuple[str, float, float, float]] = []
    for label, center_phi, center_psi in MINIMA_MARKERS:
        if label in EXCLUDED_MARKERS_BY_MODEL.get(model_label, set()):
            continue
        minimum = local_minimum_in_window(
            grid, fes, center_phi, center_psi, MINIMUM_CONSTRAINTS.get(label, {})
        )
        if minimum is None:
            continue
        phi, psi = minimum
        distance_to_start = np.hypot(
            circular_delta(phi, center_phi), circular_delta(psi, center_psi)
        )
        for index, (_other_label, other_phi, other_psi, other_distance) in enumerate(selected):
            separation = np.hypot(
                circular_delta(phi, other_phi), circular_delta(psi, other_psi)
            )
            if separation <= MARKER_DUPLICATE_DISTANCE_DEG:
                if distance_to_start < other_distance:
                    selected[index] = (label, phi, psi, distance_to_start)
                break
        else:
            selected.append((label, phi, psi, distance_to_start))

    return [(label, phi, psi) for label, phi, psi, _distance_to_start in selected]


def annotate_minima(ax: plt.Axes, grid: np.ndarray, fes: np.ndarray, model_label: str) -> None:
    for label, phi, psi in selected_minima(grid, fes, model_label):
        ax.scatter([phi], [psi], marker="*", s=105, c="white", edgecolors="black", linewidths=0.55, zorder=8)
        text = ax.text(phi + 7.0, psi - 7.0, label, color="white", fontsize=12.0, ha="left", va="top", zorder=9)
        text.set_path_effects([path_effects.withStroke(linewidth=1.6, foreground="black")])


def style_axis(ax: plt.Axes, row: int, col: int, n_rows: int) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks((-180, -90, 0, 90, 180))
    ax.set_yticks((-180, -90, 0, 90, 180))
    ax.set_xlabel(r"$\phi$ (Deg)" if row == n_rows - 1 else "")
    ax.set_ylabel(r"$\psi$ (Deg)" if col == 0 else "")
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.2)
    x_ticklabels = ax.get_xticklabels()
    if x_ticklabels:
        first_x_ticklabel = x_ticklabels[0]
        first_x_ticklabel.set_horizontalalignment("center")
        first_x_ticklabel.set_transform(
            first_x_ticklabel.get_transform()
            + ScaledTranslation(2.8 / 72.0, 0.0, ax.figure.dpi_scale_trans)
        )
        last_x_ticklabel = x_ticklabels[-1]
        last_x_ticklabel.set_transform(
            last_x_ticklabel.get_transform()
            + ScaledTranslation(-2.8 / 72.0, 0.0, ax.figure.dpi_scale_trans)
        )
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)


def load_record(record: Record) -> tuple[Record, np.ndarray, np.ndarray, int, float]:
    if record.kind == "openmm_saved_fes":
        if record.path is None:
            raise ValueError(f"Missing FES path for {record.label}")
        grid, fes, n_samples, time_fs = read_openmm_fes(record.path)
        return record, grid, fes, n_samples, time_fs
    if record.kind == "mean_biase_fes":
        if not record.paths:
            raise ValueError(f"Missing replica paths for {record.label}")
        reconstructed = []
        for path in record.paths:
            hills = read_hills(path)
            grid, fes = reconstruct_biase_fes(hills)
            reconstructed.append((fes - np.nanmin(fes), len(hills["time_fs"]), float(hills["time_fs"][-1])))
        mean_fes = np.mean([item[0] for item in reconstructed], axis=0)
        mean_fes -= np.nanmin(mean_fes)
        return record, grid, mean_fes, sum(item[1] for item in reconstructed), max(item[2] for item in reconstructed)
    if record.path is None:
        raise ValueError(f"Missing hills path for {record.label}")
    hills = read_hills(record.path)
    grid, fes = reconstruct_biase_fes(hills)
    return record, grid, fes, len(hills["time_fs"]), float(hills["time_fs"][-1])


def solution_8model_records() -> list[Record]:
    amber = PROJECT / "amber_reference_metad_1kjmol_0p5fs" / "outputs"
    ff19opc = PROJECT / "amber_ff19sb_opc" / "outputs"
    off24 = Path(
        "/work/gpuviper_ptmp/Enhanced_sampling/MACE/"
        "nexus_moved_20260708/trialanine_mace_metad/"
        "mace_off24_m_metad_1kjmol_0p5fs/outputs"
    )
    mh1 = Path(
        "/work/gpuviper_ptmp/Enhanced_sampling/MACE/"
        "nexus_moved_20260708/trialanine_mace_metad/"
        "mace_mh1_metad_1kjmol_0p5fs/outputs"
    )
    polar = PROJECT / "mace_polar_m_metad_1kjmol_0p5fs" / "outputs"
    pet = ROOT / "viper_analysis" / "upet" / "trialanine" / "pet_spice_1kjmol_0p5fs"
    so3lr = ROOT / "viper_analysis" / "sol3r" / "trialanine" / "so3lr_1kjmol_0p5fs"
    return [
        Record(
            "solution_8model",
            "AMBER ff14SB/TIP3P",
            None,
            "mean_biase_fes",
            tuple(
                amber / f"solution_amber_biase_1kjmol_0p5fs_rep{rep}_indep_1ns.bias"
                for rep in range(1, 5)
            ),
        ),
        Record("solution_8model", "MACE-OFF23(M)", off24 / "trialanine_solution_off24_m_1kjmol_0p5fs_1ns.metad.txt"),
        Record("solution_8model", "PET-SPICE-L", pet / "solution" / "pet_spice_trialanine_solution_1kjmol_0p5fs_1ns.bias"),
        Record("solution_8model", "MACE-MH-1", mh1 / "trialanine_solution_mh1_1kjmol_0p5fs_1ns.metad.txt"),
        Record(
            "solution_8model",
            "AMBER ff19SB/OPC",
            None,
            "mean_biase_fes",
            tuple(
                ff19opc / f"solution_ff19sb_opc_biase_rep{rep}_1ns.bias"
                for rep in range(1, 5)
            ),
        ),
        Record(
            "solution_8model",
            "SO3LR",
            Path(
                "/work/gpuviper_ptmp/Enhanced_sampling/sol3r/trialanine/"
                "so3lr_1kjmol_0p5fs/solution/outputs/"
                "so3lr_trialanine_solution_1kjmol_0p5fs_1ns_to1ns_part2.bias"
            ),
        ),
        Record(
            "solution_8model",
            "MACELES-OFF",
            Path(
                "/work/gpuviper_ptmp/Enhanced_sampling/MACE/"
                "trialanine_mace_les_off_metad/outputs/"
                "trialanine_solution_mace_les_off_1kjmol_15deg_0p5fs.bias"
            ),
        ),
        Record("solution_8model", "MACE-POLAR-1-M", polar / "trialanine_solution_polar_m_1kjmol_0p5fs_1ns.metad.txt"),
    ]


def openmm_records() -> list[Record]:
    out = PROJECT / "openmm_amber_metad" / "outputs_bias_only"
    return [
        Record(
            "amber14_openmm_4rep",
            f"OpenMM rep{rep}",
            out / f"solution_openmm_wtmetad_1kjmol_15deg_0p5fs_rep{rep}_cvfes_notraj_1ns.free_energy_kjmol.npy",
            "openmm_saved_fes",
        )
        for rep in range(1, 5)
    ]


def ff19_opc_records() -> list[Record]:
    out = PROJECT / "amber_ff19sb_opc" / "outputs"
    return [
        Record("ff19sb_opc_biase_openmm_4rep", f"ff19SB/OPC rep{rep}", out / f"solution_ff19sb_opc_biase_rep{rep}_1ns.bias")
        for rep in range(1, 5)
    ]


def panel_shape(n_items: int) -> tuple[int, int]:
    if n_items <= 4:
        return 2, 2
    if n_items <= 6:
        return 2, 3
    return int(np.ceil(n_items / 4)), 4


def plot_records(
    records: list[Record],
    stem: str,
    title: str | None,
    annotate: bool,
) -> list[dict[str, str | int | float]]:
    loaded = [load_record(record) for record in records]
    finite_values = np.concatenate([fes[np.isfinite(fes)].ravel() for _record, _grid, fes, _n, _time in loaded])
    vmax = max(0.05, float(np.percentile(finite_values, 99.9)))
    levels = np.linspace(0.0, vmax, PLOT_LEVELS)
    n_rows, n_cols = panel_shape(len(loaded))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.6, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.125)
    axes_array = np.atleast_1d(axes).ravel()
    last_contour = None
    summary: list[dict[str, str | int | float]] = []

    for index, (record, grid, fes, n_samples, time_fs) in enumerate(loaded):
        ax = axes_array[index]
        last_contour = ax.contourf(grid, grid, np.clip(fes, 0.0, vmax), levels=levels, cmap="RdBu_r")
        row, col = divmod(index, n_cols)
        style_axis(ax, row, col, n_rows)
        if annotate:
            annotate_minima(ax, grid, fes, record.label)
        time_ns = time_fs / 1e6
        panel_title = record.label if time_ns >= 0.9995 else f"{record.label}\n{time_ns:.3f} ns"
        ax.set_title(panel_title)
        summary.append(
            {
                "plot": stem,
                "label": record.label,
                "kind": record.kind,
                "source": ";".join(map(str, record.paths)) if record.paths else str(record.path),
                "n_samples_or_hills": n_samples,
                "time_fs": time_fs,
                "time_ns": time_fs / 1e6,
                "vmax_kj_mol_99p9": vmax,
            }
        )

    for ax in axes_array[len(loaded) :]:
        ax.set_visible(False)

    if last_contour is not None:
        cbar = fig.colorbar(last_contour, ax=axes_array[: len(loaded)].tolist(), location="right", pad=0.018)
        cbar.set_label("Relative Free Energy (kJ mol$^{-1}$)", fontsize=14.0)
        cbar.ax.tick_params(direction="out", length=4, width=1.2)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        cbar.outline.set_linewidth(1.8)

    if title:
        fig.suptitle(title, y=0.995)
    for extension in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{extension}", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return summary


def basin_values(grid: np.ndarray, fes: np.ndarray, model_label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for label, phi, psi in selected_minima(grid, fes, model_label):
        phi_index = int(np.argmin(np.abs(circular_delta(grid, phi))))
        psi_index = int(np.argmin(np.abs(circular_delta(grid, psi))))
        values[label] = float(fes[psi_index, phi_index])
    if values:
        offset = min(values.values())
        values = {label: value - offset for label, value in values.items()}
    return values


def write_basin_table(records: list[Record]) -> None:
    labels = [entry[0] for entry in MINIMA_MARKERS]
    lines = [
        "% Relative basin free energies in kJ mol^-1.",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Model & a$\\beta$ & PPII & $\\alpha$H(R) & $\\alpha$H(L) " + chr(92) * 2,
        "\\hline",
    ]
    csv_rows: list[dict[str, str | float]] = []
    for record in records:
        replica_values: list[dict[str, float]] = []
        paths = record.paths if record.kind == "mean_biase_fes" else (record.path,)
        for path in paths:
            if path is None:
                continue
            if record.kind == "openmm_saved_fes":
                grid, fes, _n, _t = read_openmm_fes(path)
            else:
                grid, fes = reconstruct_biase_fes(read_hills(path))
            replica_values.append(basin_values(grid, fes, record.label))
        cells: list[str] = []
        row: dict[str, str | float] = {"model": record.label}
        for label in labels:
            samples = [values[label] for values in replica_values if label in values]
            if not samples:
                cells.append("--")
                row[label] = "--"
            elif len(samples) == 1:
                cells.append(f"{samples[0]:.1f}")
                row[label] = samples[0]
            else:
                mean = float(np.mean(samples))
                std = float(np.std(samples, ddof=1))
                cells.append(f"{mean:.1f} $\\pm$ {std:.1f}")
                row[label] = mean
                row[f"{label}_std"] = std
        lines.append(record.label + " & " + " & ".join(cells) + r" \\")
        csv_rows.append(row)
    lines.extend(["\\hline", "\\end{tabular}", ""])
    (OUT_DIR / "trialanine_basin_relative_free_energies_kjmol.tex").write_text("\n".join(lines))
    with (OUT_DIR / "trialanine_basin_relative_free_energies_kjmol.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in csv_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 13.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, str | int | float]] = []
    records = solution_8model_records()
    all_summary += plot_records(
        records,
        "trialanine_solution_1ns_8model_mace_les_ff19sb_opc_localmin_40deg_constrained",
        None,
        annotate=True,
    )

    with (OUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary[0]))
        writer.writeheader()
        writer.writerows(all_summary)
    write_basin_table(records)


if __name__ == "__main__":
    main()
