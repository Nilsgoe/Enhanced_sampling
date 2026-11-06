from __future__ import annotations

import csv
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

import extend_malon_analysis as ema


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
FIGURE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = FIGURE_ROOT / "data"
OFF24_DATA_ROOT = OUT_DIR / "off24_raccoon_20261011"
KAPPA = 50.0
BURN_SAMPLES = 2_000       # 1 ps at 0.5 fs sampling.
USE_SAMPLES = 18_000       # 9 ps at 0.5 fs sampling; 10 ps total per window.
KJMOL_PER_EV = 96.485332
MEAN_RE = re.compile(r"mean_cv_energy_(.+?)_shift_(-?\d+)(?:_.+)?\.csv$")
RAW_RE = re.compile(r"cv_energy_(.+?)_shift_(-?\d+)(?:_.+)?\.csv$")


@dataclass(frozen=True)
class Curve:
    label: str
    path: Path
    color: str
    source_kind: str
    linestyle: str = "-"


@dataclass(frozen=True)
class Target:
    system: str
    model_tag: str
    label: str
    source_dir: Path
    output_dir: Path
    color: str
    burn_samples: int = BURN_SAMPLES
    use_samples: int = USE_SAMPLES


TARGETS = [
    Target("malonaldehyd", "off23", "MACE-OFF23(M)", OUT_DIR / "legacy_mace_models" / "malonaldehyd" / "off23", OUT_DIR / "malonaldehyd" / "mace_off23_m", "#009E73"),
    Target("malonaldehyd", "mh1_mh-1", "MACE-MH-1", OUT_DIR / "legacy_mace_models" / "malonaldehyd" / "mh1", OUT_DIR / "malonaldehyd" / "mace_mh1", "#D55E00"),
    Target("malonaldehyd", "polar_m", "MACE-POLAR-1-M", ROOT / "malonaldehyd", OUT_DIR / "malonaldehyd" / "mace_polar_m", "#CC79A7"),
    Target("malonaldehyd", "viper_sol3r", "SO3LR", Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/malonaldehyd"), OUT_DIR / "malonaldehyd" / "so3lr", "#0072B2"),
    Target("malonaldehyd", "viper_upet_pet_spice", "PET-SPICE-L", Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/malonaldehyd"), OUT_DIR / "malonaldehyd" / "pet_spice", "#E69F00"),
    Target("f-malonaldehyd", "off23", "MACE-OFF23(M)", OUT_DIR / "legacy_mace_models" / "f-malonaldehyd" / "off23", OUT_DIR / "f-malonaldehyd" / "mace_off23_m", "#009E73"),
    Target("f-malonaldehyd", "mh1_mh-1", "MACE-MH-1", OUT_DIR / "legacy_mace_models" / "f-malonaldehyd" / "mh1", OUT_DIR / "f-malonaldehyd" / "mace_mh1", "#D55E00"),
    Target("f-malonaldehyd", "polar_m", "MACE-POLAR-1-M", ROOT / "f-malonaldehyd", OUT_DIR / "f-malonaldehyd" / "mace_polar_m", "#CC79A7"),
    Target("f-malonaldehyd", "viper_sol3r", "SO3LR", Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/f-malonaldehyd"), OUT_DIR / "f-malonaldehyd" / "so3lr", "#0072B2"),
    Target("f-malonaldehyd", "viper_upet_pet_spice", "PET-SPICE-L", Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/f-malonaldehyd"), OUT_DIR / "f-malonaldehyd" / "pet_spice", "#E69F00"),
]


SYSTEMS = {
    "malonaldehyd": {
        "title": "Malonaldehyde",
        "output": "malonaldehyde_fes_model_comparison_10ps_all_no_petrot",
        "dft": Curve(
            r"$\omega$B97M-V",
            OUT_DIR / "umbrella_integration_orca_dft_10ps.csv",
            "black",
            "dft_10ps",
        ),
    },
    "f-malonaldehyd": {
        "title": "F-malonaldehyde",
        "output": "f_malonaldehyde_fes_model_comparison_10ps_all_no_petrot",
        "dft": Curve(
            r"$\omega$B97M-V",
            OUT_DIR / "umbrella_integration_orca_dft_10ps_mlip_cv.csv",
            "black",
            "dft_10ps",
        ),
    },
}


def normalize_model_tag(raw: str) -> str:
    if raw.startswith("raccoon_"):
        raw = raw[len("raccoon_") :]
    return raw


def raw_and_mean_files(source_dir: Path) -> tuple[dict[tuple[str, int], Path], dict[tuple[str, int], Path]]:
    mean_files: dict[tuple[str, int], Path] = {}
    raw_files: dict[tuple[str, int], Path] = {}
    outputs = source_dir / "outputs"
    def add_preferred(mapping: dict[tuple[str, int], Path], key: tuple[str, int], path: Path) -> None:
        # The five dense central OFF24-M reruns supersede their sparse
        # predecessors for identical window shifts.
        old = mapping.get(key)
        if old is None or ("dense_center" in path.name and "dense_center" not in old.name):
            mapping[key] = path

    for path in outputs.glob("mean_cv_energy_*.csv"):
        match = MEAN_RE.match(path.name)
        if match:
            add_preferred(mean_files, (normalize_model_tag(match.group(1)), int(match.group(2))), path)
    for path in outputs.glob("cv_energy_*.csv"):
        match = RAW_RE.match(path.name)
        if match:
            add_preferred(raw_files, (normalize_model_tag(match.group(1)), int(match.group(2))), path)
    return mean_files, raw_files


def load_center(mean_path: Path) -> tuple[float, float]:
    with mean_path.open() as handle:
        row = next(csv.reader(handle))
    center = float(row[0])
    initial_cv = float(row[4]) if len(row) >= 5 else center
    return center, initial_cv


def selection_for_raw(raw_path: Path, target: Target) -> tuple[int, int, float]:
    """Return 1+9 ps selection expressed in the file's own frame count."""
    return target.burn_samples, target.use_samples, 0.5


def load_10ps_series(raw_path: Path, target: Target) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(raw_path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    cv = np.asarray(data["cv"], dtype=float)
    energy = np.asarray(data["energy"], dtype=float)
    burn_samples, use_samples, _ = selection_for_raw(raw_path, target)
    start = min(burn_samples, len(cv))
    stop = min(start + use_samples, len(cv))
    if stop - start < 1000:
        raise RuntimeError(f"{raw_path} has too few post-burn samples: {stop - start}")
    return cv[start:stop], energy[start:stop]


def build_windows(target: Target) -> list[ema.WindowStats]:
    mean_files, raw_files = raw_and_mean_files(target.source_dir)
    windows: list[ema.WindowStats] = []
    for key in sorted(mean_files, key=lambda item: item[1]):
        model_tag, shift = key
        if model_tag != target.model_tag or key not in raw_files:
            continue
        mean_path = mean_files[key]
        raw_path = raw_files[key]
        center, initial_cv = load_center(mean_path)
        cv, energy = load_10ps_series(raw_path, target)
        force_samples = KAPPA * (center - cv)
        tau_int, force_stderr = ema.estimate_mean_stderr(force_samples)
        n_eff = max(len(force_samples) / (2.0 * tau_int), 1.0)
        windows.append(
            ema.WindowStats(
                model_tag=target.model_tag,
                shift=shift,
                center=center,
                mean_cv=float(np.mean(cv)),
                mean_force=float(np.mean(force_samples)),
                mean_energy=float(np.mean(energy)),
                initial_cv=initial_cv,
                cv_var=float(np.var(cv, ddof=1)),
                cv_std=float(np.std(cv, ddof=1)),
                n_samples=int(len(cv)),
                tau_int=float(tau_int),
                n_eff=float(n_eff),
                force_stderr=float(force_stderr),
                mean_path=mean_path,
                raw_path=raw_path,
            )
        )
    if len(windows) < 3:
        raise RuntimeError(f"Only {len(windows)} windows for {target.system} {target.model_tag}")
    return sorted(windows, key=lambda window: window.mean_cv)


def write_window_summary(output_dir: Path, windows: list[ema.WindowStats], target: Target) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "window_summary_10ps.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "shift",
                "window_center",
                "mean_cv",
                "mean_force",
                "mean_energy",
                "n_samples",
                "burn_samples",
                "used_samples",
                "frame_interval_fs",
                "tau_int",
                "n_eff",
                "force_stderr",
                "raw_path",
            ]
        )
        for window in windows:
            burn_samples, use_samples, frame_interval_fs = selection_for_raw(window.raw_path, target)
            writer.writerow(
                [
                    window.shift,
                    window.center,
                    window.mean_cv,
                    window.mean_force,
                    window.mean_energy,
                    window.n_samples,
                    burn_samples,
                    use_samples,
                    frame_interval_fs,
                    window.tau_int,
                    window.n_eff,
                    window.force_stderr,
                    window.raw_path,
                ]
            )
    return path


def save_fast_gpr_outputs(output_dir: Path, model_tag: str, windows: list[ema.WindowStats]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gpr = ema.run_gpr_from_windows(windows)
    cv = np.asarray(gpr["x_star"], dtype=float)
    pmf = np.asarray(gpr["pmf_mean"], dtype=float)
    pmf = pmf - np.nanmin(pmf)
    deriv = np.asarray(gpr["deriv_mean"], dtype=float)
    deriv_std = np.asarray(gpr["deriv_std"], dtype=float)

    curve_path = output_dir / f"umbrella_integration_{model_tag}_10ps_curve.csv"
    np.savetxt(
        curve_path,
        np.column_stack([cv, pmf, deriv, deriv_std]),
        delimiter=",",
        header="cv,pmf,mean_force,mean_force_std",
        comments="",
    )

    x_win = np.array([w.mean_cv for w in windows], dtype=float)
    force = np.array([w.mean_force for w in windows], dtype=float)
    fe_ui = ema.cumulative_trapezoid(x_win, force)
    fe_ui -= np.nanmin(fe_ui)
    summary_path = output_dir / f"umbrella_integration_{model_tag}.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mean_cv",
                "free_energy_ui",
                "free_energy_diff_gp",
                "mean_force",
                "mean_force_diff_gp",
                "mean_force_diff_gp_std",
                "window_center",
                "shift",
                "n_samples",
                "tau_int",
                "n_eff",
                "source_file",
            ]
        )
        for window, xi, fe in zip(windows, x_win, fe_ui):
            writer.writerow(
                [
                    xi,
                    fe,
                    float(np.interp(xi, cv, pmf)),
                    window.mean_force,
                    float(np.interp(xi, cv, deriv)),
                    float(np.interp(xi, cv, deriv_std)),
                    window.center,
                    window.shift,
                    window.n_samples,
                    window.tau_int,
                    window.n_eff,
                    window.raw_path,
                ]
            )
    return curve_path


def regenerate_10ps_outputs(off24_only: bool = False) -> dict[str, list[Curve]]:
    ema.plot_style()
    curves_by_system: dict[str, list[Curve]] = {system: [config["dft"]] for system, config in SYSTEMS.items()}
    for target in TARGETS:
        curve_path = target.output_dir / f"umbrella_integration_{target.model_tag}_10ps_curve.csv"
        if not off24_only or target.model_tag == "off23":
            print(f"Regenerating 10 ps: {target.system} {target.model_tag}", flush=True)
            windows = build_windows(target)
            target.output_dir.mkdir(parents=True, exist_ok=True)
            write_window_summary(target.output_dir, windows, target)
            curve_path = save_fast_gpr_outputs(target.output_dir, target.model_tag, windows)
        elif not curve_path.exists():
            raise FileNotFoundError(f"Missing cached curve for {target.label}: {curve_path}")
        curves_by_system[target.system].append(
            Curve(
                target.label,
                curve_path,
                target.color,
                "mlip_10ps",
            )
        )
    return curves_by_system


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.0,
            "axes.linewidth": 1.9,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_pmf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    names = data.dtype.names or ()
    x_name = "mean_cv" if "mean_cv" in names else "cv"
    y_name = "free_energy_diff_gp" if "free_energy_diff_gp" in names else "free_energy_ui"
    if y_name not in names:
        y_name = "pmf"
    x = np.asarray(data[x_name], dtype=float)
    y = np.asarray(data[y_name], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    y = y[order]
    return x[order], (y - np.nanmin(y)) * KJMOL_PER_EV


def draw(ax: plt.Axes, config: dict, curves: list[Curve], show_legend: bool = True) -> None:
    loaded = [(curve, *read_pmf(curve.path)) for curve in curves]
    mlip_loaded = loaded[1:]
    x_min = min(float(np.min(x)) for _, x, _ in mlip_loaded)
    x_max = max(float(np.max(x)) for _, x, _ in mlip_loaded)

    dft, x, y = loaded[0]
    mask = (x >= x_min) & (x <= x_max)
    if np.any(mask):
        y_plot = y[mask] - np.min(y[mask])
        ax.plot(x[mask], y_plot, color=dft.color, lw=2.6, alpha=1.0, label=dft.label, zorder=3)

    for curve, x, y in loaded[1:]:
        mask = (x >= x_min) & (x <= x_max)
        if not np.any(mask):
            continue
        y_plot = y[mask] - np.min(y[mask])
        ax.plot(
            x[mask],
            y_plot,
            color=curve.color,
            linestyle=curve.linestyle,
            lw=2.6,
            alpha=0.65,
            label=curve.label,
            zorder=2,
        )

    ax.set_xlim(x_min, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Proton-transfer Coordinate ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Free Energy (kJ mol$^{-1}$)")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.9)
    if show_legend:
        ax.legend(frameon=False, ncol=1, loc="upper right", handlelength=3.0)


def make_plots(curves_by_system: dict[str, list[Curve]]) -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for system, config in SYSTEMS.items():
        fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
        draw(ax, config, curves_by_system[system])
        for ext in ("png", "pdf", "svg"):
            path = OUT_DIR / f"{config['output']}.{ext}"
            fig.savefig(path, dpi=400)
            print(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    for ax, (system, config) in zip(axes, SYSTEMS.items()):
        draw(ax, config, curves_by_system[system], show_legend=True)
    axes[1].set_ylabel("")
    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"malonaldehyde_fes_model_comparison_combined_10ps_all_no_petrot.{ext}"
        fig.savefig(path, dpi=400)
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off24-only", action="store_true", help="Reuse cached non-OFF23 curves (legacy option name).")
    args = parser.parse_args()
    curves_by_system = regenerate_10ps_outputs(args.off24_only)
    make_plots(curves_by_system)


if __name__ == "__main__":
    main()
