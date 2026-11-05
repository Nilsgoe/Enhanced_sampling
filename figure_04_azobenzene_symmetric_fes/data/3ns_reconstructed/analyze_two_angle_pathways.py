from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from azobenzene.scripts.barrier_extraction.basins_mfep import minimax_path, nearest_grid
from azobenzene.scripts.barrier_extraction.uncertainty import _pick_basin


HERE = ROOT / "azobenzene" / "analysis" / "model_comparison" / "3ns_reconstructed"
OUT_CSV = HERE / "azobenzene_2d_3.0ns_two_angle_pathways.csv"
OUT_TEX = HERE / "azobenzene_2d_3.0ns_two_angle_pathways_table.tex"
OUT_SUMMARY_CSV = HERE / "azobenzene_2d_3.0ns_two_angle_pathways_summary.csv"
OUT_CHANNEL_SUMMARY_CSV = HERE / "azobenzene_2d_3.0ns_two_angle_pathways_channel_summary.csv"
PATH_DIR = HERE / "two_angle_paths"
PET_SPICE_STABLE_CV_DIR = ROOT / "azobenzene" / "analysis" / "model_comparison" / "pet_spice_cis_cv_comparison"
EV_TO_KJMOL = 96.485
EV_TO_KCALMOL = 23.0609
TEMPERATURE_K = 300.0
KB_EV_PER_K = 8.617333262145e-5
PHI_SEPARATOR_DEG = 94.0

MODELS = ("off", "mh1", "polar", "pet_spice", "so3lr", "scratch", "ft_so3lr", "ft_mh1")
SYSTEMS = ("cis", "trans")
LABELS = {
    "off": "MACE-OFF",
    "mh1": "MACE-MH1",
    "polar": "MACE-Polar",
    "pet_spice": "PET-SPICE",
    "so3lr": "SO3LR",
    "scratch": "MACE from scratch",
    "ft_so3lr": "Fine-tuned SO3LR",
    "ft_mh1": "Fine-tuned MACE-MH1",
}

PATH_SPECS = {
    "low_angle_120": {
        "label": r"$\theta\approx120^\circ$",
        "waypoints": ((90.0, 120.0), (-90.0, 120.0)),
    },
    "high_angle_160": {
        "label": r"$\theta\approx160^\circ$",
        "waypoints": ((90.0, 160.0), (-90.0, 160.0)),
    },
}


def _load_surface(system: str, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    path = HERE / f"azobenzene_{system}_{model}_2d_3.0ns_reconstructed_fes.npz"
    if system == "cis" and model == "pet_spice":
        path = PET_SPICE_STABLE_CV_DIR / "pet_spice_cis_stable_cv_3.0ns_reconstructed_fes.npz"
    data = np.load(path)
    return (
        np.asarray(data["cv1_deg"], dtype=float),
        np.asarray(data["cv2_deg"], dtype=float),
        np.asarray(data["free_energy_eV"], dtype=float),
        float(data["actual_time_fs"]) / 1_000_000.0,
        int(data["n_hills"]),
    )


def _surface_source(system: str, model: str) -> str:
    if system == "cis" and model == "pet_spice":
        return str(PET_SPICE_STABLE_CV_DIR / "pet_spice_cis_stable_cv_3.0ns_reconstructed_fes.npz")
    return str(HERE / f"azobenzene_{system}_{model}_2d_3.0ns_reconstructed_fes.npz")


def _path_ts(cv1: np.ndarray, cv2: np.ndarray, F: np.ndarray, path_idx: np.ndarray) -> dict[str, float | int]:
    n2 = F.shape[1]
    ii = path_idx // n2
    jj = path_idx % n2
    F_path = F[ii, jj]
    ts_k = int(np.argmax(F_path))
    return {
        "path_index": ts_k,
        "grid_i": int(ii[ts_k]),
        "grid_j": int(jj[ts_k]),
        "cv1_CNNC_deg": float(cv1[ii[ts_k]]),
        "cv2_CNN_deg": float(cv2[jj[ts_k]]),
        "F_eV": float(F_path[ts_k]),
        "path_max_eV": float(F_path[ts_k]),
        "path_min_eV": float(F_path.min()),
    }


def _write_path_csv(
    system: str,
    model: str,
    path_name: str,
    cv1: np.ndarray,
    cv2: np.ndarray,
    F: np.ndarray,
    path_idx: np.ndarray,
) -> Path:
    PATH_DIR.mkdir(parents=True, exist_ok=True)
    n2 = F.shape[1]
    ii = path_idx // n2
    jj = path_idx % n2
    out = PATH_DIR / f"azobenzene_{system}_{model}_3.0ns_{path_name}_path.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("step", "cv1_CNNC_deg", "cv2_CNN_deg", "F_eV", "F_kJmol", "F_kcalmol"),
        )
        writer.writeheader()
        for step, (i, j) in enumerate(zip(ii, jj, strict=True)):
            f_ev = float(F[i, j])
            writer.writerow(
                {
                    "step": step,
                    "cv1_CNNC_deg": float(cv1[i]),
                    "cv2_CNN_deg": float(cv2[j]),
                    "F_eV": f_ev,
                    "F_kJmol": f_ev * EV_TO_KJMOL,
                    "F_kcalmol": f_ev * EV_TO_KCALMOL,
                }
            )
    return out


def _best_waypoint_path(
    F: np.ndarray,
    cv1: np.ndarray,
    cv2: np.ndarray,
    basin_cis: tuple[int, int],
    basin_trans: tuple[int, int],
    waypoints: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], tuple[int, int], np.ndarray, np.ndarray]:
    candidates = []
    for cv1_target, cv2_target in waypoints:
        waypoint_ij = nearest_grid(cv1, cv2, cv1_target, cv2_target)
        path_idx, F_path, barrier = minimax_path(F, basin_cis, basin_trans, waypoint_ij=waypoint_ij)
        candidates.append((float(F_path.max()), float(barrier), (cv1_target, cv2_target), waypoint_ij, path_idx, F_path))
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, waypoint_target, waypoint_ij, path_idx, F_path = candidates[0]
    return waypoint_target, waypoint_ij, path_idx, F_path


def _energy_columns(prefix: str, value_eV: float) -> dict[str, float]:
    return {
        f"{prefix}_eV": value_eV,
        f"{prefix}_kJmol": value_eV * EV_TO_KJMOL,
        f"{prefix}_kcalmol": value_eV * EV_TO_KCALMOL,
    }


def _logsumexp(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot integrate an empty/non-finite basin")
    vmax = float(np.max(finite))
    return float(vmax + np.log(np.sum(np.exp(finite - vmax))))


def _periodic_phi(cv1: np.ndarray) -> np.ndarray:
    wrapped = ((cv1.astype(float) + 180.0) % 360.0) - 180.0
    wrapped[np.isclose(wrapped, -180.0)] = 180.0
    return wrapped


def _boltzmann_basin_free_energies(F: np.ndarray, cv1: np.ndarray) -> dict[str, float | int]:
    """Boltzmann integrate cis/trans basins over the 2D FES grid.

    Cis is |CNNC| <= PHI_SEPARATOR_DEG; trans is the periodic complement.
    The common grid-area factor cancels in differences and is omitted. The
    returned basin free energies are shifted so the lower basin is zero.
    """
    kT = KB_EV_PER_K * TEMPERATURE_K
    phi = _periodic_phi(cv1)
    cis_phi = np.abs(phi) <= PHI_SEPARATOR_DEG
    trans_phi = ~cis_phi
    cis_mask = np.broadcast_to(cis_phi[:, None], F.shape)
    trans_mask = np.broadcast_to(trans_phi[:, None], F.shape)

    g_cis_raw = -kT * _logsumexp(-F[cis_mask] / kT)
    g_trans_raw = -kT * _logsumexp(-F[trans_mask] / kT)
    offset = min(g_cis_raw, g_trans_raw)
    return {
        "temperature_K": TEMPERATURE_K,
        "phi_separator_deg": PHI_SEPARATOR_DEG,
        "cis_basin_grid_points": int(np.count_nonzero(cis_mask)),
        "trans_basin_grid_points": int(np.count_nonzero(trans_mask)),
        "F_cis": float(g_cis_raw - offset),
        "F_trans": float(g_trans_raw - offset),
        "deltaG_cis_minus_trans": float(g_cis_raw - g_trans_raw),
    }


def analyze_surface(system: str, model: str) -> list[dict[str, str | int | float]]:
    cv1, cv2, F, actual_time_ns, n_hills = _load_surface(system, model)
    basin_cis = _pick_basin(F, cv1, cv2, target=(0.0, 120.0))
    basin_trans = _pick_basin(F, cv1, cv2, target=(180.0, 120.0))
    F_cis_point = float(F[basin_cis])
    F_trans_point = float(F[basin_trans])
    basin_free_energy = _boltzmann_basin_free_energies(F, cv1)
    F_cis = float(basin_free_energy["F_cis"])
    F_trans = float(basin_free_energy["F_trans"])
    dG_cis_minus_trans = float(basin_free_energy["deltaG_cis_minus_trans"])

    rows = []
    ts_by_path = {}
    for path_name, spec in PATH_SPECS.items():
        waypoint_target, waypoint_ij, path_idx, F_path = _best_waypoint_path(
            F, cv1, cv2, basin_cis, basin_trans, spec["waypoints"]
        )
        path_csv = _write_path_csv(system, model, path_name, cv1, cv2, F, path_idx)
        ts = _path_ts(cv1, cv2, F, path_idx)
        ts_by_path[path_name] = ts["F_eV"]
        row: dict[str, str | int | float] = {
            "system": system,
            "model": model,
            "label": LABELS[model],
            "path": path_name,
            "path_label": spec["label"],
            "actual_time_ns": actual_time_ns,
            "n_hills": n_hills,
            "surface_source": _surface_source(system, model),
            "cis_min_cv1_CNNC_deg": float(cv1[basin_cis[0]]),
            "cis_min_cv2_CNN_deg": float(cv2[basin_cis[1]]),
            "trans_min_cv1_CNNC_deg": float(cv1[basin_trans[0]]),
            "trans_min_cv2_CNN_deg": float(cv2[basin_trans[1]]),
            "temperature_K": basin_free_energy["temperature_K"],
            "phi_separator_deg": basin_free_energy["phi_separator_deg"],
            "cis_basin_grid_points": basin_free_energy["cis_basin_grid_points"],
            "trans_basin_grid_points": basin_free_energy["trans_basin_grid_points"],
            "selected_waypoint_cv1_CNNC_deg": float(waypoint_target[0]),
            "selected_waypoint_cv2_CNN_deg": float(waypoint_target[1]),
            "selected_waypoint_grid_cv1_CNNC_deg": float(cv1[waypoint_ij[0]]),
            "selected_waypoint_grid_cv2_CNN_deg": float(cv2[waypoint_ij[1]]),
            "ts_cv1_CNNC_deg": ts["cv1_CNNC_deg"],
            "ts_cv2_CNN_deg": ts["cv2_CNN_deg"],
            "path_csv": str(path_csv),
        }
        row.update(_energy_columns("F_cis_point", F_cis_point))
        row.update(_energy_columns("F_trans_point", F_trans_point))
        row.update(_energy_columns("F_cis", F_cis))
        row.update(_energy_columns("F_trans", F_trans))
        row.update(_energy_columns("deltaG_cis_minus_trans", dG_cis_minus_trans))
        row.update(_energy_columns("F_TS", float(ts["F_eV"])))
        row.update(_energy_columns("TS_minus_cis", float(ts["F_eV"]) - F_cis))
        row.update(_energy_columns("TS_minus_trans", float(ts["F_eV"]) - F_trans))
        rows.append(row)

    if set(ts_by_path) == set(PATH_SPECS):
        delta = float(ts_by_path["high_angle_160"]) - float(ts_by_path["low_angle_120"])
        for row in rows:
            row.update(_energy_columns("TS_high160_minus_low120", delta))
    return rows


def write_csv(rows: list[dict[str, str | int | float]]) -> None:
    with OUT_CSV.open("w", newline="") as handle:
        fieldnames = [str(field).strip() for field in rows[0]]
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fieldnames)
        for row in rows:
            clean = {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items()}
            writer.writerow([clean.get(field, "") for field in fieldnames])


def _mean_std(values: list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    std = float(data.std(ddof=1)) if data.size > 1 else 0.0
    return float(data.mean()), std


def summarize_by_model(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | float | int]]:
    summary_rows: list[dict[str, str | float | int]] = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        if not subset:
            continue

        seen_systems: set[tuple[str, str]] = set()
        stability_eV: list[float] = []
        stability_kJmol: list[float] = []
        stability_kcalmol: list[float] = []
        for row in subset:
            key = (str(row["system"]), str(row["model"]))
            if key in seen_systems:
                continue
            seen_systems.add(key)
            stability_eV.append(float(row["deltaG_cis_minus_trans_eV"]))
            stability_kJmol.append(float(row["deltaG_cis_minus_trans_kJmol"]))
            stability_kcalmol.append(float(row["deltaG_cis_minus_trans_kcalmol"]))

        c2t_eV: list[float] = []
        c2t_kJmol: list[float] = []
        c2t_kcalmol: list[float] = []
        t2c_eV: list[float] = []
        t2c_kJmol: list[float] = []
        t2c_kcalmol: list[float] = []
        selected_path_rows: list[dict[str, str | int | float]] = []
        for system in SYSTEMS:
            system_rows = [row for row in subset if row["system"] == system]
            if not system_rows:
                continue
            best_c2t = min(system_rows, key=lambda row: float(row["TS_minus_cis_eV"]))
            best_t2c = min(system_rows, key=lambda row: float(row["TS_minus_trans_eV"]))
            selected_path_rows.extend([best_c2t, best_t2c])
            c2t_eV.append(float(best_c2t["TS_minus_cis_eV"]))
            c2t_kJmol.append(float(best_c2t["TS_minus_cis_kJmol"]))
            c2t_kcalmol.append(float(best_c2t["TS_minus_cis_kcalmol"]))
            t2c_eV.append(float(best_t2c["TS_minus_trans_eV"]))
            t2c_kJmol.append(float(best_t2c["TS_minus_trans_kJmol"]))
            t2c_kcalmol.append(float(best_t2c["TS_minus_trans_kcalmol"]))

        dg_eV_mean, dg_eV_std = _mean_std(stability_eV)
        dg_kJ_mean, dg_kJ_std = _mean_std(stability_kJmol)
        dg_kcal_mean, dg_kcal_std = _mean_std(stability_kcalmol)
        c2t_eV_mean, c2t_eV_std = _mean_std(c2t_eV)
        c2t_kJ_mean, c2t_kJ_std = _mean_std(c2t_kJmol)
        c2t_kcal_mean, c2t_kcal_std = _mean_std(c2t_kcalmol)
        t2c_eV_mean, t2c_eV_std = _mean_std(t2c_eV)
        t2c_kJ_mean, t2c_kJ_std = _mean_std(t2c_kJmol)
        t2c_kcal_mean, t2c_kcal_std = _mean_std(t2c_kcalmol)

        summary_rows.append(
            {
                "model": model,
                "label": LABELS[model],
                "n_stability_replicates": len(stability_eV),
                "n_barrier_estimates": len(c2t_eV),
                "barrier_selection": "minimum_pathway_per_starting_ensemble",
                "selected_cis_start_c2t_path": next(
                    row["path"] for row in selected_path_rows if row["system"] == "cis" and float(row["TS_minus_cis_eV"]) == min(float(r["TS_minus_cis_eV"]) for r in subset if r["system"] == "cis")
                ),
                "selected_trans_start_c2t_path": next(
                    row["path"] for row in selected_path_rows if row["system"] == "trans" and float(row["TS_minus_cis_eV"]) == min(float(r["TS_minus_cis_eV"]) for r in subset if r["system"] == "trans")
                ),
                "selected_cis_start_t2c_path": next(
                    row["path"] for row in selected_path_rows if row["system"] == "cis" and float(row["TS_minus_trans_eV"]) == min(float(r["TS_minus_trans_eV"]) for r in subset if r["system"] == "cis")
                ),
                "selected_trans_start_t2c_path": next(
                    row["path"] for row in selected_path_rows if row["system"] == "trans" and float(row["TS_minus_trans_eV"]) == min(float(r["TS_minus_trans_eV"]) for r in subset if r["system"] == "trans")
                ),
                "deltaG_cis_minus_trans_eV_mean": dg_eV_mean,
                "deltaG_cis_minus_trans_eV_std": dg_eV_std,
                "deltaG_cis_minus_trans_kJmol_mean": dg_kJ_mean,
                "deltaG_cis_minus_trans_kJmol_std": dg_kJ_std,
                "deltaG_cis_minus_trans_kcalmol_mean": dg_kcal_mean,
                "deltaG_cis_minus_trans_kcalmol_std": dg_kcal_std,
                "barrier_cis_to_trans_eV_mean": c2t_eV_mean,
                "barrier_cis_to_trans_eV_std": c2t_eV_std,
                "barrier_cis_to_trans_kJmol_mean": c2t_kJ_mean,
                "barrier_cis_to_trans_kJmol_std": c2t_kJ_std,
                "barrier_cis_to_trans_kcalmol_mean": c2t_kcal_mean,
                "barrier_cis_to_trans_kcalmol_std": c2t_kcal_std,
                "barrier_trans_to_cis_eV_mean": t2c_eV_mean,
                "barrier_trans_to_cis_eV_std": t2c_eV_std,
                "barrier_trans_to_cis_kJmol_mean": t2c_kJ_mean,
                "barrier_trans_to_cis_kJmol_std": t2c_kJ_std,
                "barrier_trans_to_cis_kcalmol_mean": t2c_kcal_mean,
                "barrier_trans_to_cis_kcalmol_std": t2c_kcal_std,
            }
        )
    return summary_rows


def summarize_by_model_and_path(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | float | int]]:
    summary_rows: list[dict[str, str | float | int]] = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue

        seen_systems: set[tuple[str, str]] = set()
        stability_kJmol: list[float] = []
        for row in model_rows:
            key = (str(row["system"]), str(row["model"]))
            if key in seen_systems:
                continue
            seen_systems.add(key)
            stability_kJmol.append(float(row["deltaG_cis_minus_trans_kJmol"]))
        dg_kJ_mean, dg_kJ_std = _mean_std(stability_kJmol)

        out: dict[str, str | float | int] = {
            "model": model,
            "label": LABELS[model],
            "n_stability_replicates": len(stability_kJmol),
            "deltaG_cis_minus_trans_kJmol_mean": dg_kJ_mean,
            "deltaG_cis_minus_trans_kJmol_std": dg_kJ_std,
        }
        for path_name in PATH_SPECS:
            path_rows = [row for row in model_rows if row["path"] == path_name]
            c2t = [float(row["TS_minus_cis_kJmol"]) for row in path_rows]
            t2c = [float(row["TS_minus_trans_kJmol"]) for row in path_rows]
            c2t_mean, c2t_std = _mean_std(c2t)
            t2c_mean, t2c_std = _mean_std(t2c)
            prefix = "low120" if path_name == "low_angle_120" else "high160"
            out[f"{prefix}_barrier_cis_to_trans_kJmol_mean"] = c2t_mean
            out[f"{prefix}_barrier_cis_to_trans_kJmol_std"] = c2t_std
            out[f"{prefix}_barrier_trans_to_cis_kJmol_mean"] = t2c_mean
            out[f"{prefix}_barrier_trans_to_cis_kJmol_std"] = t2c_std
            out[f"{prefix}_n_barrier_estimates"] = len(path_rows)

        c2t_deltas: list[float] = []
        t2c_deltas: list[float] = []
        for system in SYSTEMS:
            low = next(row for row in model_rows if row["system"] == system and row["path"] == "low_angle_120")
            high = next(row for row in model_rows if row["system"] == system and row["path"] == "high_angle_160")
            c2t_deltas.append(float(high["TS_minus_cis_kJmol"]) - float(low["TS_minus_cis_kJmol"]))
            t2c_deltas.append(float(high["TS_minus_trans_kJmol"]) - float(low["TS_minus_trans_kJmol"]))
        c2t_delta_mean, c2t_delta_std = _mean_std(c2t_deltas)
        t2c_delta_mean, t2c_delta_std = _mean_std(t2c_deltas)
        out["delta_barrier_160_minus_120_cis_to_trans_kJmol_mean"] = c2t_delta_mean
        out["delta_barrier_160_minus_120_cis_to_trans_kJmol_std"] = c2t_delta_std
        out["delta_barrier_160_minus_120_trans_to_cis_kJmol_mean"] = t2c_delta_mean
        out["delta_barrier_160_minus_120_trans_to_cis_kJmol_std"] = t2c_delta_std
        summary_rows.append(out)
    return summary_rows


def write_summary_csv(summary_rows: list[dict[str, str | float | int]]) -> None:
    with OUT_SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)


def write_channel_summary_csv(summary_rows: list[dict[str, str | float | int]]) -> None:
    with OUT_CHANNEL_SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)


def write_tex(summary_rows: list[dict[str, str | float | int]]) -> None:
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Azobenzene stability and pathway-resolved barrier estimates from the reconstructed 2D free-energy surfaces. The stability term is $\Delta G_{\mathrm{cis-trans}}=G_{\mathrm{cis}}-G_{\mathrm{trans}}$, where basin free energies are obtained by Boltzmann integration with $|\phi|\leq94^\circ$ defining the cis basin and the periodic complement defining the trans basin. Directional barriers are reported separately for constrained minimax pathways through $\theta\approx120^\circ$ and $\theta\approx160^\circ$, each referenced to the corresponding Boltzmann-integrated basin free energy. Reported uncertainties are sample standard deviations over the two starting ensembles.}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Model & $\Delta G_{\mathrm{cis-trans}}$ & $\Delta G^\ddagger_{120,\mathrm{c\to t}}$ & $\Delta G^\ddagger_{160,\mathrm{c\to t}}$ & $\Delta\Delta G^\ddagger_{\mathrm{c\to t}}$ & $\Delta G^\ddagger_{120,\mathrm{t\to c}}$ & $\Delta G^\ddagger_{160,\mathrm{t\to c}}$ & $\Delta\Delta G^\ddagger_{\mathrm{t\to c}}$ \\",
        r" & kJ mol$^{-1}$ & kJ mol$^{-1}$ & kJ mol$^{-1}$ & kJ mol$^{-1}$ & kJ mol$^{-1}$ & kJ mol$^{-1}$ & kJ mol$^{-1}$ \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['label']} & "
            f"{float(row['deltaG_cis_minus_trans_kJmol_mean']):.2f} $\\pm$ {float(row['deltaG_cis_minus_trans_kJmol_std']):.2f} & "
            f"{float(row['low120_barrier_cis_to_trans_kJmol_mean']):.2f} $\\pm$ {float(row['low120_barrier_cis_to_trans_kJmol_std']):.2f} & "
            f"{float(row['high160_barrier_cis_to_trans_kJmol_mean']):.2f} $\\pm$ {float(row['high160_barrier_cis_to_trans_kJmol_std']):.2f} & "
            f"{float(row['delta_barrier_160_minus_120_cis_to_trans_kJmol_mean']):.2f} $\\pm$ {float(row['delta_barrier_160_minus_120_cis_to_trans_kJmol_std']):.2f} & "
            f"{float(row['low120_barrier_trans_to_cis_kJmol_mean']):.2f} $\\pm$ {float(row['low120_barrier_trans_to_cis_kJmol_std']):.2f} & "
            f"{float(row['high160_barrier_trans_to_cis_kJmol_mean']):.2f} $\\pm$ {float(row['high160_barrier_trans_to_cis_kJmol_std']):.2f} & "
            f"{float(row['delta_barrier_160_minus_120_trans_to_cis_kJmol_mean']):.2f} $\\pm$ {float(row['delta_barrier_160_minus_120_trans_to_cis_kJmol_std']):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    OUT_TEX.write_text("\n".join(lines))


def main() -> None:
    rows: list[dict[str, str | int | float]] = []
    for system in SYSTEMS:
        for model in MODELS:
            rows.extend(analyze_surface(system, model))
    write_csv(rows)
    summary_rows = summarize_by_model(rows)
    write_summary_csv(summary_rows)
    channel_summary_rows = summarize_by_model_and_path(rows)
    write_channel_summary_csv(channel_summary_rows)
    write_tex(channel_summary_rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY_CSV}")
    print(f"Wrote {OUT_CHANNEL_SUMMARY_CSV}")
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote individual path CSV files in {PATH_DIR}")


if __name__ == "__main__":
    main()
