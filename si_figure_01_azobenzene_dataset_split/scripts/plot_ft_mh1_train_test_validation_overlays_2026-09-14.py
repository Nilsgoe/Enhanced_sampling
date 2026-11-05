from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from ase.io import iread


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = DATA_DIR

SPLIT_FILES = {
    "train": DATA_DIR / "train_azob.xyz",
    "valid": DATA_DIR / "valid_azob.xyz",
    "test": DATA_DIR / "test_azob.xyz",
}
SPLIT_STYLE = {
    "train": {"label": "training", "color": "black", "s": 13, "alpha": 0.42, "zorder": 5},
    "valid": {"label": "validation", "color": "#00a651", "s": 18, "alpha": 0.72, "zorder": 6},
    "test": {"label": "test", "color": "#ff7f00", "s": 20, "alpha": 0.78, "zorder": 7},
}
SYSTEMS = ("cis", "trans")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_two_cnn_pathways as wall_analysis

wall_analysis.ROOT = (
    PACKAGE_ROOT
    / "figure_04_azobenzene_symmetric_fes"
    / "data"
    / "raw_bias"
    / "azobenzene_angle_wall_test_20260812"
)
WALL_MODELS = wall_analysis.MODELS
bias_file = wall_analysis.bias_file
reconstruct = wall_analysis.reconstruct
CV_BY_ISOMER = {
    "cis": {
        "dihedral": (1, 6, 7, 8),
        "angle": (1, 6, 7),
    },
    "trans": {
        "dihedral": (2, 11, 12, 13),
        "angle": (2, 11, 12),
    },
}


def dihedral_deg(positions: np.ndarray, indices: tuple[int, int, int, int]) -> float:
    p1, p2, p3, p4 = positions[list(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2_unit = b2 / np.linalg.norm(b2)
    return float(np.degrees(np.arctan2(np.dot(np.cross(n1, n2), b2_unit), np.dot(n1, n2))))


def angle_deg(positions: np.ndarray, indices: tuple[int, int, int]) -> float:
    pa, pb, pc = positions[list(indices)]
    v1 = pa - pb
    v2 = pc - pb
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def wrap_deg(angle: float) -> float:
    wrapped = ((angle + 180.0) % 360.0) - 180.0
    if np.isclose(wrapped, -180.0):
        return 180.0
    return float(wrapped)


def cnnc_region(cnnc_deg: float) -> str:
    cnnc = wrap_deg(cnnc_deg)
    if -90.0 <= cnnc <= 90.0:
        return "cis"
    if 90.0 < cnnc <= 180.0 or -180.0 <= cnnc < -90.0:
        return "trans"
    return "other"


def isomer_from_config_type(config_type: str) -> str:
    if config_type.endswith("_cis"):
        return "cis"
    if config_type.endswith("_trans"):
        return "trans"
    raise ValueError(f"Cannot infer cis/trans CV indices from config_type={config_type!r}")


def split_cvs() -> dict[str, np.ndarray]:
    split_points: dict[str, np.ndarray] = {}
    rows = []
    for split, path in SPLIT_FILES.items():
        values = []
        n_total = 0
        for frame_index, atoms in enumerate(iread(path, ":")):
            n_total += 1
            if len(atoms) != 24:
                continue
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            config_type = str(atoms.info.get("config_type", ""))
            isomer = isomer_from_config_type(config_type)
            cnnc_indices = CV_BY_ISOMER[isomer]["dihedral"]
            cnn_indices = CV_BY_ISOMER[isomer]["angle"]
            if "".join(symbols[index] for index in cnnc_indices) != "CNNC":
                raise ValueError(f"{path}:{frame_index} has invalid CNNC symbols for {config_type}: {cnnc_indices}")
            if "".join(symbols[index] for index in cnn_indices) != "CNN":
                raise ValueError(f"{path}:{frame_index} has invalid CNN symbols for {config_type}: {cnn_indices}")
            cv1 = dihedral_deg(positions, cnnc_indices)
            cv2 = angle_deg(positions, cnn_indices)
            values.append((cv1, cv2))
            rows.append(
                {
                    "split": split,
                    "frame": frame_index,
                    "config_type": config_type,
                    "isomer": isomer,
                    "cnnc_indices": "-".join(str(idx) for idx in cnnc_indices),
                    "cnn_indices": "-".join(str(idx) for idx in cnn_indices),
                    "cv1_CNNC_deg": f"{cv1:.8f}",
                    "cv2_CNN_deg": f"{cv2:.8f}",
                    "cnnc_region": cnnc_region(cv1),
                }
            )
        split_points[split] = np.asarray(values, dtype=float)
        print(f"{split}: total_frames={n_total} used_azobenzene_frames={len(values)}")

    with (OUT_DIR / "azobenzene_split_points_cv.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "frame",
                "config_type",
                "isomer",
                "cnnc_indices",
                "cnn_indices",
                "cv1_CNNC_deg",
                "cv2_CNN_deg",
                "cnnc_region",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = []
    for split in SPLIT_FILES:
        split_rows = [row for row in rows if row["split"] == split]
        cis_count = sum(row["cnnc_region"] == "cis" for row in split_rows)
        trans_count = sum(row["cnnc_region"] == "trans" for row in split_rows)
        other_count = sum(row["cnnc_region"] == "other" for row in split_rows)
        counts.append(
            {
                "split": split,
                "n": len(split_rows),
                "cis_count": cis_count,
                "cis_percent": f"{100.0 * cis_count / len(split_rows):.2f}",
                "trans_count": trans_count,
                "trans_percent": f"{100.0 * trans_count / len(split_rows):.2f}",
                "other_count": other_count,
            }
        )
    with (OUT_DIR / "azobenzene_split_region_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(counts[0].keys()))
        writer.writeheader()
        writer.writerows(counts)
    with (OUT_DIR / "azobenzene_split_region_counts.md").open("w") as handle:
        handle.write("# Azobenzene Split Region Counts\n\n")
        handle.write("CVs use topological indices from `config_type`: `*_cis` uses cis indices and `*_trans` uses trans indices.\n\n")
        handle.write("| split | n | cis | trans | other |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for count in counts:
            handle.write(
                f"| {count['split']} | {count['n']} | "
                f"{count['cis_count']} ({count['cis_percent']}%) | "
                f"{count['trans_count']} ({count['trans_percent']}%) | "
                f"{count['other_count']} |\n"
            )
    return split_points


def load_wall_fes(system: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = next(model for model in WALL_MODELS if model.directory == "mace_ft_mh1")
    phi, cnn, fes, _time = reconstruct(bias_file(model, system))
    return phi, cnn, fes


def style_axes(ax) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(60, 180)
    ax.set_xticks((-180, -90, 0, 90, 180))
    ax.set_yticks((60, 90, 120, 150, 180))
    ax.tick_params(direction="in", top=True, right=True, length=4.5, width=1.7)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2.8, width=1.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(2.3)


def scatter_splits(ax, points_by_split: dict[str, np.ndarray], splits: tuple[str, ...]) -> None:
    for split in splits:
        points = points_by_split[split]
        style = SPLIT_STYLE[split]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=style["s"],
            c=style["color"],
            alpha=style["alpha"],
            linewidths=0,
            rasterized=True,
            label=f"{style['label']} ({len(points)})",
            zorder=style["zorder"],
        )


def make_plot(points_by_split: dict[str, np.ndarray]) -> None:
    surfaces = {system: load_wall_fes(system) for system in SYSTEMS}
    vmax = float(
        np.ceil(
            max(np.percentile(fes[np.isfinite(fes)], 99.5) for _phi, _cnn, fes in surfaces.values())
        )
    )
    levels = np.linspace(0.0, vmax, 18)

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12)
    contour = None
    for ax, system in zip(axes, SYSTEMS, strict=True):
        phi, cnn, free_energy = surfaces[system]
        contour = ax.contourf(
            phi,
            cnn,
            np.clip(free_energy, 0.0, vmax),
            levels=levels,
            cmap="RdBu_r",
        )
        splits = ("train",) if system == "cis" else ("valid", "test")
        scatter_splits(ax, points_by_split, splits)
        ax.set_xlabel(r"$\phi$ (Deg)")
        style_axes(ax)
        ax.legend(loc="upper right", frameon=False, fontsize=17.0, handlelength=1.0)
    axes[0].set_ylabel(r"$\theta$ (Deg)")
    cbar = fig.colorbar(contour, ax=axes, pad=0.018)
    cbar.set_label(r"Relative Free Energy (kJ mol$^{-1}$)", fontsize=19.0)
    cbar.ax.tick_params(direction="out", length=4, width=1.7)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    cbar.outline.set_linewidth(2.3)
    for extension in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"azobenzene_ft_mh1_training_cis_test_validation_trans_points_overlay.{extension}",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 15.0,
            "axes.labelsize": 19.0,
            "axes.titlesize": 16.0,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "axes.linewidth": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points_by_split = split_cvs()
    make_plot(points_by_split)
    print(f"Wrote fine-tuned MACE-MH1 split overlay plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
