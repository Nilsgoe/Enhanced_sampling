from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
SYSTEMS = ("malonaldehyd", "f-malonaldehyd")
VIPER_ANALYSIS_INPUTS = {
    "malonaldehyd": (
        (
            "viper_sol3r",
            ROOT / "viper_analysis" / "sol3r" / "malonaldehyd" / "umbrella_integration_viper_sol3r.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice",
            ROOT / "viper_analysis" / "upet" / "malonaldehyd" / "umbrella_integration_viper_upet_pet_spice.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice_rot",
            ROOT
            / "viper_analysis"
            / "upet"
            / "malonaldehyd_pet_spice_rotavg3_43w"
            / "umbrella_integration_viper_upet_pet_spice_rot.csv",
            ROOT / "viper_analysis" / "upet" / "malonaldehyd_pet_spice_rotavg3_43w" / "outputs",
        ),
    ),
    "f-malonaldehyd": (
        (
            "viper_sol3r",
            ROOT / "viper_analysis" / "sol3r" / "f-malonaldehyd" / "umbrella_integration_viper_sol3r.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/sol3r/f-malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice",
            ROOT / "viper_analysis" / "upet" / "f-malonaldehyd" / "umbrella_integration_viper_upet_pet_spice.csv",
            Path("/work/gpuviper_ptmp/Enhanced_sampling/upet/f-malonaldehyd/outputs"),
        ),
        (
            "viper_upet_pet_spice_rot",
            ROOT
            / "viper_analysis"
            / "upet"
            / "f-malonaldehyd_pet_spice_rotavg3_43w"
            / "umbrella_integration_viper_upet_pet_spice_rot.csv",
            ROOT / "viper_analysis" / "upet" / "f-malonaldehyd_pet_spice_rotavg3_43w" / "outputs",
        ),
    ),
}
KAPPA = 50.0
KJ_PER_MOL_PER_EV = 96.485

MEAN_RE = re.compile(r"mean_cv_energy_(.+?)_shift_(-?\d+)(?:_.+)?\.csv$")
RAW_RE = re.compile(r"cv_energy_(.+?)_shift_(-?\d+)(?:_.+)?\.csv$")

MODEL_LABELS = {
    "off_large": "MACE-OFF large",
    "off_medium": "MACE-OFF medium",
    "off_small": "MACE-OFF small",
    "off24_medium": "MACE-OFF24 medium",
    "omol_extra_large": "MACE-OMOL XL",
    "mh1_mh-1": "MACE-MH-1",
    "polar_l": "MACE-Polar L",
    "polar_m": "MACE-Polar M",
    "polar_s": "MACE-Polar S",
    "viper_sol3r": "SO3LR",
    "viper_upet_pet_spice": "UPET PET-SPICE-L",
    "viper_upet_pet_spice_rot": "PET-SPICE_rot",
}

MODEL_COLORS = {
    "off_large": "#1b9e77",
    "off_medium": "#66a61e",
    "off_small": "#a6d854",
    "off24_medium": "#2ca25f",
    "omol_extra_large": "#d95f02",
    "mh1_mh-1": "#7570b3",
    "polar_l": "#e7298a",
    "polar_m": "#e6ab02",
    "polar_s": "#1f78b4",
    "viper_sol3r": "#6a3d9a",
    "viper_upet_pet_spice": "#e31a1c",
    "viper_upet_pet_spice_rot": "#ff7f00",
}


@dataclass
class WindowStats:
    model_tag: str
    shift: int
    center: float
    mean_cv: float
    mean_force: float
    mean_energy: float
    initial_cv: float
    cv_var: float
    cv_std: float
    n_samples: int
    tau_int: float
    n_eff: float
    force_stderr: float
    mean_path: Path
    raw_path: Path


def plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.3,
            "lines.linewidth": 2.4,
            "savefig.bbox": "tight",
        }
    )


def normalize_model_tag(raw: str) -> str:
    if raw.startswith("raccoon_"):
        raw = raw[len("raccoon_") :]
    mapping = {
        "off_off24": "off24_medium",
        "off_large": "off_large",
        "off_medium": "off_medium",
        "off_small": "off_small",
        "omol_extra_large": "omol_extra_large",
        "mh1_mh-1": "mh1_mh-1",
        "polar_l": "polar_l",
        "polar_m": "polar_m",
        "polar_s": "polar_s",
    }
    return mapping.get(raw, raw)


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    if len(x) >= 2:
        out[1:] = np.cumsum(np.diff(x) * 0.5 * (y[1:] + y[:-1]))
    return out


def compute_tau_int(series: np.ndarray, max_lag: int | None = None) -> float:
    x = np.asarray(series, dtype=float) - np.mean(series)
    n = len(x)
    if n < 8:
        return 0.5
    if max_lag is None:
        max_lag = max(4, n // 2)
    else:
        max_lag = min(max_lag, max(4, n // 2))

    fft = np.fft.fft(x, n=2 * n)
    acov = np.fft.ifft(fft * np.conj(fft))[:n].real
    if acov[0] <= 0:
        return 0.5
    acf = (acov / acov[0])[: max_lag + 1]

    # Geyer-style initial positive sequence. This is usually more conservative
    # than a hard ACF cutoff and gives more realistic uncertainty bands.
    tau = 0.5
    for lag in range(1, len(acf), 2):
        pair_sum = acf[lag]
        if lag + 1 < len(acf):
            pair_sum += acf[lag + 1]
        if pair_sum <= 0.0:
            break
        tau += pair_sum
    return max(float(tau), 0.5)


def estimate_mean_stderr(series: np.ndarray) -> tuple[float, float]:
    values = np.asarray(series, dtype=float)
    if len(values) < 2:
        return 0.5, 0.0
    tau_int = compute_tau_int(values)
    n_eff = max(len(values) / (2.0 * tau_int), 1.0)
    stderr = float(np.std(values, ddof=1) / np.sqrt(n_eff))
    return tau_int, stderr


def load_positions(raw_path: Path) -> np.ndarray:
    data = np.genfromtxt(raw_path, delimiter=",", names=True)
    return np.atleast_1d(data["cv"]).astype(float)


def load_diff_gp(system_dir: Path):
    module_path = system_dir / "diff_gp.py"
    module_name = f"diff_gp_{system_dir.name.replace('-', '_')}"
    if module_name in sys.modules:
        return sys.modules[module_name].diff_GP
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diff_GP from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.diff_GP


def gather_windows(system_dir: Path) -> dict[str, list[WindowStats]]:
    outputs_dir = system_dir / "outputs"
    mean_files: dict[tuple[str, int], Path] = {}
    raw_files: dict[tuple[str, int], Path] = {}

    for path in outputs_dir.glob("mean_cv_energy_*.csv"):
        match = MEAN_RE.match(path.name)
        if match:
            mean_files[(normalize_model_tag(match.group(1)), int(match.group(2)))] = path

    for path in outputs_dir.glob("cv_energy_*.csv"):
        match = RAW_RE.match(path.name)
        if match:
            raw_files[(normalize_model_tag(match.group(1)), int(match.group(2)))] = path

    grouped: dict[str, list[WindowStats]] = {}
    for key, mean_path in sorted(mean_files.items()):
        if key not in raw_files:
            continue
        model_tag, shift = key
        with mean_path.open() as handle:
            row = next(csv.reader(handle))
        center, mean_cv, mean_force, mean_energy, initial_cv = [float(x) for x in row[:5]]
        positions = load_positions(raw_files[key])
        cv_var = float(np.var(positions, ddof=1))
        cv_std = float(np.sqrt(max(cv_var, 0.0)))
        force_samples = KAPPA * (center - positions)
        tau_int, force_stderr = estimate_mean_stderr(force_samples)
        n_samples = int(len(positions))
        n_eff = max(n_samples / (2.0 * tau_int), 1.0)
        grouped.setdefault(model_tag, []).append(
            WindowStats(
                model_tag=model_tag,
                shift=shift,
                center=center,
                mean_cv=mean_cv,
                mean_force=mean_force,
                mean_energy=mean_energy,
                initial_cv=initial_cv,
                cv_var=cv_var,
                cv_std=cv_std,
                n_samples=n_samples,
                tau_int=tau_int,
                n_eff=n_eff,
                force_stderr=force_stderr,
                mean_path=mean_path,
                raw_path=raw_files[key],
            )
        )
    return {model: sorted(windows, key=lambda w: w.mean_cv) for model, windows in grouped.items()}


def load_windows_from_analysis_csv(
    model_tag: str, csv_path: Path, raw_outputs_dir: Path | None = None
) -> list[WindowStats]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    data = np.atleast_1d(data)
    windows: list[WindowStats] = []
    for row in data:
        center = float(row["window_center"])
        mean_cv = float(row["mean_cv"])
        mean_force = float(row["mean_force"])
        shift = int(row["shift"])
        initial_cv = center - 0.05 * shift
        raw_path = csv_path
        if raw_outputs_dir is not None:
            candidate = raw_outputs_dir / f"cv_energy_{model_tag}_shift_{shift}.csv"
            if candidate.exists():
                raw_path = candidate
        windows.append(
            WindowStats(
                model_tag=model_tag,
                shift=shift,
                center=center,
                mean_cv=mean_cv,
                mean_force=mean_force,
                mean_energy=0.0,
                initial_cv=initial_cv,
                cv_var=0.0,
                cv_std=0.0,
                n_samples=0,
                tau_int=0.0,
                n_eff=0.0,
                force_stderr=0.0,
                mean_path=csv_path,
                raw_path=raw_path,
            )
        )
    return sorted(windows, key=lambda w: w.mean_cv)


def bootstrap_ui_band(
    x: np.ndarray,
    force: np.ndarray,
    force_stderr: np.ndarray,
    n_boot: int = 1000,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = rng.normal(loc=force[None, :], scale=force_stderr[None, :], size=(n_boot, len(force)))
    free_energy_draws = np.zeros_like(draws)
    for idx in range(n_boot):
        free_energy_draws[idx] = cumulative_trapezoid(x, draws[idx])
        free_energy_draws[idx] -= np.min(free_energy_draws[idx])
    return np.percentile(free_energy_draws, 16.0, axis=0), np.percentile(free_energy_draws, 84.0, axis=0)


def k_base(x1: np.ndarray, x2: np.ndarray, sigma_f: float, ell: float) -> np.ndarray:
    x1, x2 = np.atleast_1d(x1)[:, None], np.atleast_1d(x2)[None, :]
    return sigma_f**2 * np.exp(-0.5 * (x1 - x2) ** 2 / ell**2)


def k_f_fprime(x_star: np.ndarray, x: np.ndarray, sigma_f: float, ell: float) -> np.ndarray:
    x_star = np.atleast_1d(x_star)[:, None]
    x = np.atleast_1d(x)[None, :]
    delta = x_star - x
    base = sigma_f**2 * np.exp(-0.5 * delta**2 / ell**2)
    return (delta / ell**2) * base


def k_fprime_fprime(x1: np.ndarray, x2: np.ndarray, sigma_f: float, ell: float) -> np.ndarray:
    x1, x2 = np.atleast_1d(x1), np.atleast_1d(x2)
    sqdist = (x1[:, None] - x2[None, :]) ** 2
    base = sigma_f**2 * np.exp(-0.5 * sqdist / ell**2)
    return (1.0 / ell**2 - sqdist / ell**4) * base


def neg_log_marginal_likelihood(params: np.ndarray, x_train: np.ndarray, y: np.ndarray, errors: np.ndarray) -> float:
    sigma_f, ell = params
    if sigma_f <= 0 or ell <= 0.01:
        return 1e10
    ky = k_fprime_fprime(x_train, x_train, sigma_f, ell) + np.diag(errors**2) + 1e-8 * np.eye(len(x_train))
    try:
        chol, low = cho_factor(ky)
        alpha = cho_solve((chol, low), y)
        return float(0.5 * y.T @ alpha + np.sum(np.log(np.diag(chol))) + 0.5 * len(x_train) * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        return 1e10


def leave_one_out_cv(
    x_train: np.ndarray,
    y: np.ndarray,
    errors: np.ndarray,
    sigma_f: float,
    ell: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky = k_fprime_fprime(x_train, x_train, sigma_f, ell) + np.diag(errors**2) + 1e-8 * np.eye(len(x_train))
    chol, low = cho_factor(ky)
    ky_inv = cho_solve((chol, low), np.eye(len(x_train)))
    alpha = ky_inv @ y
    diag_inv = np.maximum(np.diag(ky_inv), 1e-15)
    loo_means = y - alpha / diag_inv
    loo_vars = 1.0 / diag_inv
    loo_stds = np.sqrt(np.maximum(loo_vars, 0.0))
    loo_z = (y - loo_means) / np.maximum(loo_stds, 1e-15)
    return loo_means, loo_stds, loo_z


def run_gpr_from_windows(windows: list[WindowStats]) -> dict[str, np.ndarray | float]:
    x_centers = np.array([w.center for w in windows], dtype=float)
    x_means = np.array([w.mean_cv for w in windows], dtype=float)
    x_vars = np.array([w.cv_var for w in windows], dtype=float)
    n_samples = np.array([w.n_samples for w in windows], dtype=float)
    tau_ints = np.array([w.tau_int for w in windows], dtype=float)
    n_eff = np.maximum(np.array([w.n_eff for w in windows], dtype=float), 1.0)
    kappa = np.full_like(x_centers, KAPPA)
    derivatives = kappa * (x_centers - x_means)
    derivative_errors = np.maximum(np.array([w.force_stderr for w in windows], dtype=float), 1e-12)

    if len(x_centers) > 1:
        window_spacing = np.diff(np.sort(x_centers)).mean()
        cv_range = x_centers.max() - x_centers.min()
    else:
        window_spacing = 0.1
        cv_range = 1.0

    ell_init = max(2.0 * window_spacing, 0.1)
    sigma_f_init = max(derivatives.std(), 1e-6) * ell_init
    bounds = [(0.001, 100.0), (0.02, max(3.0 * cv_range, 5.0))]
    starts = [
        [sigma_f_init, ell_init],
        [max(0.5 * sigma_f_init, 0.01), max(0.5 * ell_init, 0.05)],
        [max(2.0 * sigma_f_init, 0.02), max(2.0 * ell_init, 0.1)],
        [0.1, 1.0],
    ]
    best_fun = np.inf
    best_params = np.array([sigma_f_init, ell_init], dtype=float)
    for x0 in starts:
        x0 = np.array([np.clip(x0[0], *bounds[0]), np.clip(x0[1], *bounds[1])])
        result = minimize(
            neg_log_marginal_likelihood,
            x0=x0,
            args=(x_centers, derivatives, derivative_errors),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if result.success and result.fun < best_fun:
            best_fun = float(result.fun)
            best_params = result.x
    sigma_f_opt, ell_opt = [float(x) for x in best_params]

    ky = k_fprime_fprime(x_centers, x_centers, sigma_f_opt, ell_opt) + np.diag(derivative_errors**2) + 1e-8 * np.eye(
        len(x_centers)
    )
    chol, low = cho_factor(ky)
    alpha = cho_solve((chol, low), derivatives)

    x_star = np.linspace(x_centers.min(), x_centers.max(), 300)
    k_star_d = k_f_fprime(x_star, x_centers, sigma_f_opt, ell_opt)
    pmf_raw = k_star_d @ alpha
    k_ss = k_base(x_star, x_star, sigma_f_opt, ell_opt)
    cov_f = k_ss - k_star_d @ cho_solve((chol, low), k_star_d.T)
    ref_idx = int(np.argmin(pmf_raw))
    cov_ref = cov_f[:, ref_idx]
    var_ref = cov_f[ref_idx, ref_idx]
    var_f = np.diag(cov_f)
    pmf_std = np.sqrt(np.clip(var_f + var_ref - 2.0 * cov_ref, 0.0, np.inf))
    pmf_mean = pmf_raw - pmf_raw[ref_idx]

    k_dstar = k_fprime_fprime(x_star, x_centers, sigma_f_opt, ell_opt)
    deriv_mean = k_dstar @ alpha
    cov_deriv = k_fprime_fprime(x_star, x_star, sigma_f_opt, ell_opt) - k_dstar @ cho_solve((chol, low), k_dstar.T)
    deriv_std = np.sqrt(np.clip(np.diag(cov_deriv), 0.0, np.inf))

    deriv_pred_train = k_fprime_fprime(x_centers, x_centers, sigma_f_opt, ell_opt) @ alpha
    residuals = derivatives - deriv_pred_train
    std_residuals = residuals / np.maximum(derivative_errors, 1e-15)
    loo_means, loo_stds, loo_z = leave_one_out_cv(x_centers, derivatives, derivative_errors, sigma_f_opt, ell_opt)

    return {
        "x_centers": x_centers,
        "x_means": x_means,
        "x_vars": x_vars,
        "n_samples": n_samples,
        "tau_ints": tau_ints,
        "n_eff": n_eff,
        "kappa": kappa,
        "derivatives": derivatives,
        "derivative_errors": derivative_errors,
        "x_star": x_star,
        "pmf_mean": pmf_mean,
        "pmf_std": pmf_std,
        "pmf_reference_cv": float(x_star[ref_idx]),
        "deriv_mean": deriv_mean,
        "deriv_std": deriv_std,
        "sigma_f": sigma_f_opt,
        "lengthscale": ell_opt,
        "training_std_residuals": std_residuals,
        "loo_z": loo_z,
        "loo_means": loo_means,
        "loo_stds": loo_stds,
    }


def run_diff_gp_from_windows(diff_gp_source_dir: Path, windows: list[WindowStats]) -> dict[str, np.ndarray | float]:
    diff_gp_cls = load_diff_gp(diff_gp_source_dir)
    x_train = np.array([w.center for w in windows], dtype=float)
    dy_train = np.array([w.mean_force for w in windows], dtype=float)
    shifts = np.array([w.shift for w in windows], dtype=float)
    initial_cv = float(np.mean([w.initial_cv for w in windows]))
    if len(shifts) > 1:
        x_predict = initial_cv + 0.05 * np.linspace(shifts.min(), shifts.max(), len(windows))
    else:
        x_predict = x_train.copy()

    gp = diff_gp_cls(verbose=False, learning_rate=1e-5, momentum=0.5)
    gp.optimize(x_train, dy_train)
    gp.train(x_train, dy_train)
    y_predict, _ = gp.predict(x_predict)
    dy_predict, dy_std = gp.predict_diff(x_predict)

    y_predict = np.array(y_predict, dtype=float, copy=True)
    dy_predict = np.array(dy_predict, dtype=float, copy=True)
    dy_std = np.array(dy_std, dtype=float, copy=True)
    y_predict -= np.min(y_predict)

    return {
        "x_train": x_train,
        "dy_train": dy_train,
        "x_predict": x_predict,
        "pmf": y_predict,
        "mean_force": dy_predict,
        "mean_force_std": dy_std,
        "sigma": float(gp.sigma),
        "delta": float(gp.delta),
        "lengthscale": float(gp.l),
        "alpha_RQ": float(gp.alpha_RQ),
    }


def cache_paths(output_dir: Path, model_tag: str) -> tuple[Path, Path]:
    return output_dir / f"umbrella_gpr_{model_tag}_pmf.csv", output_dir / f"umbrella_gpr_{model_tag}_deriv.csv"


def load_diff_gp_cache(output_dir: Path, model_tag: str, windows: list[WindowStats]) -> dict[str, np.ndarray | float] | None:
    pmf_csv, deriv_csv = cache_paths(output_dir, model_tag)
    if not pmf_csv.exists() or not deriv_csv.exists():
        return None
    pmf_data = np.genfromtxt(pmf_csv, delimiter=",", names=True)
    deriv_data = np.genfromtxt(deriv_csv, delimiter=",", names=True)
    pmf_data = np.atleast_1d(pmf_data)
    deriv_data = np.atleast_1d(deriv_data)
    if pmf_data.size == 0 or deriv_data.size == 0:
        return None

    pmf_names = pmf_data.dtype.names or ()
    deriv_names = deriv_data.dtype.names or ()
    if "cv" not in pmf_names or "cv" not in deriv_names:
        return None
    if "pmf" in pmf_names:
        pmf = np.asarray(pmf_data["pmf"], dtype=float)
    elif "pmf_mean" in pmf_names:
        pmf = np.asarray(pmf_data["pmf_mean"], dtype=float)
    else:
        return None
    if "mean_force" in deriv_names:
        mean_force = np.asarray(deriv_data["mean_force"], dtype=float)
    elif "deriv_mean" in deriv_names:
        mean_force = np.asarray(deriv_data["deriv_mean"], dtype=float)
    else:
        return None
    if "mean_force_std" in deriv_names:
        mean_force_std = np.asarray(deriv_data["mean_force_std"], dtype=float)
    elif "deriv_std" in deriv_names:
        mean_force_std = np.asarray(deriv_data["deriv_std"], dtype=float)
    else:
        return None

    return {
        "x_train": np.array([w.center for w in windows], dtype=float),
        "dy_train": np.array([w.mean_force for w in windows], dtype=float),
        "x_predict": np.asarray(pmf_data["cv"], dtype=float),
        "pmf": pmf - np.min(pmf),
        "mean_force": mean_force,
        "mean_force_std": mean_force_std,
        "sigma": np.nan,
        "delta": np.nan,
        "lengthscale": np.nan,
        "alpha_RQ": np.nan,
    }


def get_diff_gp_results(
    output_dir: Path,
    diff_gp_source_dir: Path,
    model_tag: str,
    windows: list[WindowStats],
    refit: bool,
) -> dict[str, np.ndarray | float]:
    if not refit:
        cached = load_diff_gp_cache(output_dir, model_tag, windows)
        if cached is not None:
            return cached
    return run_diff_gp_from_windows(diff_gp_source_dir, windows)


def save_ui_outputs(
    output_dir: Path,
    diff_gp_source_dir: Path,
    model_tag: str,
    windows: list[WindowStats],
    diff_gp_results: dict[str, np.ndarray | float] | None = None,
) -> list[Path]:
    analysis_dir = output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([w.mean_cv for w in windows], dtype=float)
    force = np.array([w.mean_force for w in windows], dtype=float)
    force_stderr = np.array([w.force_stderr for w in windows], dtype=float)
    free_energy = cumulative_trapezoid(x, force)
    free_energy -= np.min(free_energy)
    if diff_gp_results is None:
        diff_gp_results = run_diff_gp_from_windows(diff_gp_source_dir, windows)
    gp_x = np.asarray(diff_gp_results["x_predict"], dtype=float)
    gp_pmf = np.asarray(diff_gp_results["pmf"], dtype=float)
    gp_force = np.asarray(diff_gp_results["mean_force"], dtype=float)
    gp_force_std = np.asarray(diff_gp_results["mean_force_std"], dtype=float)
    gp_pmf_at_windows = np.interp(x, gp_x, gp_pmf)
    gp_force_at_windows = np.interp(x, gp_x, gp_force)
    gp_force_std_at_windows = np.interp(x, gp_x, gp_force_std)

    csv_path = analysis_dir / f"umbrella_integration_{model_tag}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mean_cv",
                "free_energy_ui",
                "free_energy_diff_gp",
                "mean_force",
                "mean_force_diff_gp",
                "mean_force_diff_gp_lo",
                "mean_force_diff_gp_hi",
                "mean_force_diff_gp_std",
                "window_center",
                "shift",
                "n_samples",
                "tau_int",
                "n_eff",
                "source_file",
            ]
        )
        for window, xi, fe_ui, fe_gp, fi, fi_gp, fi_gp_std in zip(
            windows, x, free_energy, gp_pmf_at_windows, force, gp_force_at_windows, gp_force_std_at_windows
        ):
            writer.writerow(
                [
                    xi,
                    fe_ui,
                    fe_gp,
                    fi,
                    fi_gp,
                    fi_gp - 2.0 * fi_gp_std,
                    fi_gp + 2.0 * fi_gp_std,
                    fi_gp_std,
                    window.center,
                    window.shift,
                    window.n_samples,
                    window.tau_int,
                    window.n_eff,
                    window.mean_path.name,
                ]
            )

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    color = MODEL_COLORS.get(model_tag, "#1f78b4")
    label = MODEL_LABELS.get(model_tag, model_tag.replace("_", " "))
    ax1.plot(gp_x, gp_pmf, color=color, label=f"{label} diff-GP PMF")
    ax1.plot(x, free_energy, marker="o", linestyle="none", color=color, alpha=0.9, label="UI window points")
    ax1.set_xlabel("CV (A)")
    ax1.set_ylabel("Relative free energy (eV)")
    ax1.set_title(f"{output_dir.parent.name}: {label}")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.scatter(x, force, color="#444444", s=24, alpha=0.75, label="Mean force windows")
    ax2.plot(gp_x, gp_force, color="#111111", linestyle="--", linewidth=1.8, label="diff-GP mean force")
    ax2.fill_between(
        gp_x,
        gp_force - 2.0 * gp_force_std,
        gp_force + 2.0 * gp_force_std,
        color="#777777",
        alpha=0.20,
        linewidth=0,
        label="Mean force ±2σ",
    )
    ax2.set_ylabel("Mean force (eV/A)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)
    fig.tight_layout()

    outputs = [csv_path]
    for ext in ("png", "pdf"):
        out = analysis_dir / f"umbrella_integration_{model_tag}.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)
    return outputs


def plot_gpr_diagnostics(results: dict[str, np.ndarray | float], title: str) -> plt.Figure:
    x_star = results["x_star"]
    pmf_mean = results["pmf_mean"]
    pmf_std = results["pmf_std"]
    deriv_mean = results["deriv_mean"]
    deriv_std = results["deriv_std"]
    x_centers = results["x_centers"]
    derivatives = results["derivatives"]
    derivative_errors = results["derivative_errors"]
    x_means = results["x_means"]
    x_vars = results["x_vars"]
    n_samples = results["n_samples"]
    tau_ints = results["tau_ints"]
    std_residuals = results["training_std_residuals"]
    loo_z = results["loo_z"]

    fig = plt.figure(figsize=(15.5, 9.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x_star, pmf_mean, color="#1f77b4")
    ax1.fill_between(x_star, pmf_mean - 2 * pmf_std, pmf_mean + 2 * pmf_std, alpha=0.25, color="#1f77b4")
    ax1.set_xlabel("CV (A)")
    ax1.set_ylabel("PMF (eV)")
    ax1.set_title("GPR PMF")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x_star, deriv_mean, color="#ff7f0e")
    ax2.fill_between(x_star, deriv_mean - 2 * deriv_std, deriv_mean + 2 * deriv_std, alpha=0.25, color="#ff7f0e")
    ax2.errorbar(x_centers, derivatives, yerr=2 * derivative_errors, fmt="o", color="black", ms=4, alpha=0.65)
    ax2.set_xlabel("Window centre (A)")
    ax2.set_ylabel("dF/dx (eV/A)")
    ax2.set_title("GPR mean force")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.errorbar(x_centers, x_means, yerr=2 * np.sqrt(np.maximum(x_vars / np.maximum(results["n_eff"], 1.0), 0.0)), fmt="o")
    ax3.plot(x_centers, x_centers, "--", color="black", alpha=0.5)
    ax3.set_xlabel("Window centre (A)")
    ax3.set_ylabel("Mean sampled CV (A)")
    ax3.set_title("Sampling overlap")

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(np.arange(len(derivative_errors)), derivative_errors, color="#2ca02c", alpha=0.75)
    ax4.set_xlabel("Window index")
    ax4.set_ylabel("stderr dF/dx")
    ax4.set_title("Derivative uncertainty")

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(np.arange(len(std_residuals)), std_residuals, color="#d62728", alpha=0.75)
    ax5.axhline(0.0, color="black", lw=1.0)
    ax5.axhline(2.0, color="red", ls="--", alpha=0.5)
    ax5.axhline(-2.0, color="red", ls="--", alpha=0.5)
    ax5.set_xlabel("Window index")
    ax5.set_ylabel("Std residual")
    ax5.set_title("Training residuals")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(np.arange(len(loo_z)), loo_z, color="#9467bd", alpha=0.75)
    ax6.axhline(0.0, color="black", lw=1.0)
    ax6.axhline(2.0, color="red", ls="--", alpha=0.5)
    ax6.axhline(-2.0, color="red", ls="--", alpha=0.5)
    ax6.set_xlabel("Window index")
    ax6.set_ylabel("LOO z")
    ax6.set_title("LOO validation")

    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(loo_z, bins=15, density=True, alpha=0.65, color="#9467bd")
    x_norm = np.linspace(-4, 4, 100)
    ax7.plot(x_norm, 1 / np.sqrt(2 * np.pi) * np.exp(-x_norm**2 / 2), "k--", lw=2)
    ax7.set_xlabel("LOO z")
    ax7.set_ylabel("Density")
    ax7.set_title("LOO z distribution")

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.bar(np.arange(len(tau_ints)), tau_ints, color="#8c564b", alpha=0.75)
    ax8.set_xlabel("Window index")
    ax8.set_ylabel("tau_int")
    ax8.set_title("Autocorrelation time")

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    summary = (
        f"sigma_f = {results['sigma_f']:.4f} eV\n"
        f"ell = {results['lengthscale']:.4f} A\n"
        f"mean PMF stderr = {np.mean(pmf_std):.4f} eV\n"
        f"max PMF stderr = {np.max(pmf_std):.4f} eV\n"
        f"mean deriv stderr = {np.mean(derivative_errors):.4f} eV/A\n"
        f"mean tau_int = {np.mean(tau_ints):.2f}\n"
        f"std(LOO z) = {np.std(loo_z):.2f}"
    )
    ax9.text(
        0.05,
        0.95,
        summary,
        transform=ax9.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="#f4f1ea", alpha=0.8),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


def save_gpr_outputs(
    output_dir: Path,
    diff_gp_source_dir: Path,
    model_tag: str,
    windows: list[WindowStats],
    diff_gp_results: dict[str, np.ndarray | float] | None = None,
) -> list[Path]:
    analysis_dir = output_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results = diff_gp_results if diff_gp_results is not None else run_diff_gp_from_windows(diff_gp_source_dir, windows)
    label = MODEL_LABELS.get(model_tag, model_tag.replace("_", " "))
    color = MODEL_COLORS.get(model_tag, "#1f77b4")

    pmf_csv = analysis_dir / f"umbrella_gpr_{model_tag}_pmf.csv"
    with pmf_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "pmf"])
        for row in zip(results["x_predict"], results["pmf"]):
            writer.writerow(row)

    deriv_csv = analysis_dir / f"umbrella_gpr_{model_tag}_deriv.csv"
    with deriv_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "mean_force", "mean_force_std"])
        for row in zip(results["x_predict"], results["mean_force"], results["mean_force_std"]):
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(results["x_predict"], results["pmf"], color=color, label=f"{label} diff-GP PMF")
    ax.set_xlabel("CV (A)")
    ax.set_ylabel("Relative free energy (eV)")
    ax.set_title(f"{output_dir.parent.name}: {label} diff-GP umbrella analysis")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    outputs = [pmf_csv, deriv_csv]
    for ext in ("png", "pdf"):
        out = analysis_dir / f"umbrella_gpr_{model_tag}.{ext}"
        fig.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(fig)

    diag, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), constrained_layout=True)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    ax1.plot(results["x_predict"], results["pmf"], color=color)
    ax1.set_xlabel("CV (A)")
    ax1.set_ylabel("Relative free energy (eV)")
    ax1.set_title("diff-GP PMF")
    ax1.grid(alpha=0.25)

    ax2.scatter(results["x_train"], results["dy_train"], color="#444444", s=28, label="Window mean force")
    ax2.plot(results["x_predict"], results["mean_force"], color="#111111", label="diff-GP mean force")
    ax2.fill_between(
        results["x_predict"],
        results["mean_force"] - 2.0 * results["mean_force_std"],
        results["mean_force"] + 2.0 * results["mean_force_std"],
        color="#777777",
        alpha=0.25,
        linewidth=0,
        label="Mean force ±2σ",
    )
    ax2.set_xlabel("CV (A)")
    ax2.set_ylabel("Mean force (eV/A)")
    ax2.set_title("diff-GP mean force")
    ax2.legend(loc="best", frameon=False)
    ax2.grid(alpha=0.25)

    train_force_fit = np.interp(results["x_train"], results["x_predict"], results["mean_force"])
    residuals = np.asarray(results["dy_train"], dtype=float) - train_force_fit
    ax3.axhline(0.0, color="black", lw=1.0, alpha=0.65)
    ax3.scatter(results["x_train"], residuals, color="#d95f02", s=28)
    ax3.set_xlabel("Window center (A)")
    ax3.set_ylabel("Residual dF/dx (eV/A)")
    ax3.set_title("Training residuals")
    ax3.grid(alpha=0.25)

    x_centers = np.array([w.center for w in windows], dtype=float)
    x_means = np.array([w.mean_cv for w in windows], dtype=float)
    ax4.scatter(x_centers, x_means, color="#1f78b4", s=26)
    lo = min(float(np.min(x_centers)), float(np.min(x_means)))
    hi = max(float(np.max(x_centers)), float(np.max(x_means)))
    ax4.plot([lo, hi], [lo, hi], "--", color="black", alpha=0.55, lw=1.4)
    ax4.set_xlabel("Window center (A)")
    ax4.set_ylabel("Mean sampled CV (A)")
    ax4.set_title("Sampling centers")
    ax4.grid(alpha=0.25)

    ax5.plot(results["x_predict"], results["mean_force_std"], color="#7570b3")
    ax5.set_xlabel("CV (A)")
    ax5.set_ylabel("Std dF/dx (eV/A)")
    ax5.set_title("Mean-force uncertainty")
    ax5.grid(alpha=0.25)

    ax6.axis("off")
    shifts = np.array([w.shift for w in windows], dtype=float)
    summary = (
        f"windows = {len(windows)}\n"
        f"shift range = {int(np.min(shifts))} ... {int(np.max(shifts))}\n"
        f"x_predict = {len(results['x_predict'])} points\n"
        f"CV range = {np.min(results['x_predict']):.4f} ... {np.max(results['x_predict']):.4f} A\n"
        f"sigma = {results['sigma']:.4g}\n"
        f"delta = {results['delta']:.4g}\n"
        f"lengthscale = {results['lengthscale']:.4g}\n"
        f"alpha_RQ = {results['alpha_RQ']:.4g}\n"
        f"mean std(dF/dx) = {np.mean(results['mean_force_std']):.4g} eV/A"
    )
    ax6.text(
        0.04,
        0.96,
        summary,
        va="top",
        transform=ax6.transAxes,
        family="monospace",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="#f4f1ea", alpha=0.9),
    )

    diag.suptitle(f"{output_dir.parent.name}: {label} diff-GP analysis", fontsize=15, fontweight="bold")
    for ext in ("png", "pdf"):
        analysis_out = analysis_dir / f"umbrella_integration_{model_tag}_analysis.{ext}"
        diag.savefig(analysis_out, dpi=300)
        outputs.append(analysis_out)
        out = analysis_dir / f"umbrella_gpr_{model_tag}_diagnostics.{ext}"
        diag.savefig(out, dpi=300)
        outputs.append(out)
    plt.close(diag)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate malonaldehyde umbrella plots from cached diff_GP predictions by default."
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Retrain diff_GP before plotting. Without this flag, existing umbrella_gpr_* CSVs are reused.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_style()
    for system_name in SYSTEMS:
        system_dir = ROOT / system_name
        grouped = gather_windows(system_dir)
        for model_tag, windows in sorted(grouped.items()):
            output_dir = system_dir / "analysis"
            diff_gp_results = get_diff_gp_results(output_dir, system_dir, model_tag, windows, args.refit)
            save_ui_outputs(system_dir / "analysis", system_dir, model_tag, windows, diff_gp_results)
            save_gpr_outputs(system_dir / "analysis", system_dir, model_tag, windows, diff_gp_results)

        for model_tag, csv_path, raw_outputs_dir in VIPER_ANALYSIS_INPUTS[system_name]:
            if not csv_path.exists():
                continue
            windows = load_windows_from_analysis_csv(model_tag, csv_path, raw_outputs_dir)
            output_dir = csv_path.parent
            diff_gp_results = get_diff_gp_results(output_dir, system_dir, model_tag, windows, args.refit)
            save_ui_outputs(output_dir, system_dir, model_tag, windows, diff_gp_results)
            save_gpr_outputs(output_dir, system_dir, model_tag, windows, diff_gp_results)


if __name__ == "__main__":
    main()
