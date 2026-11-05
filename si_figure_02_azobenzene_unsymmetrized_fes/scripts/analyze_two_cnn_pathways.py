#!/usr/bin/env python3
"""Compare fixed-CNN azobenzene rotation routes on the wall-MetaD FES."""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "data"
GRID_POINTS = 181
EV_TO_KJMOL = 96.48533212331002
LOW_CNN_DEG = 120.0
HIGH_CNN_DEG = 160.0
CNN_FLEX_DEG = 15.0
HIGH_CNN_MIN_DEG = 155.0
HIGH_CNN_MAX_DEG = 175.0


@dataclass(frozen=True)
class Model:
    label: str
    directory: str


MODELS = (
    Model("MACE-OFF23(M)", "mace_off"),
    Model("MACE-MH1", "mace_mh1"),
    Model("MACE-POLAR-1-M", "mace_polar_m"),
    Model("PET-SPICE", "pet_spice"),
    Model("SO3LR", "sol3r"),
    Model("MACE from scratch", "mace_scratch"),
    Model("Fine-tuned SO3LR", "ft_so3lr"),
    Model("Fine-tuned MACE-MH1", "mace_ft_mh1"),
)


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def read_hills(path: Path) -> tuple[np.ndarray, ...]:
    rows: list[list[float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] in {"time", "!#"}:
            continue
        try:
            rows.append([float(value) for value in fields[:8]])
        except ValueError:
            continue
    data = np.asarray(rows, dtype=float)
    data = data[np.isfinite(data).all(axis=1)]
    if not len(data):
        raise ValueError(f"No valid hills in {path}")
    return data[:, 0], wrap_degrees(data[:, 1]), data[:, 2], data[:, 3], data[:, 4], data[:, 5]


def reconstruct(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    time_fs, phi_hills, cnn_hills, sigma_phi, sigma_cnn, height = read_hills(path)
    phi = np.linspace(-180.0, 180.0, GRID_POINTS)
    cnn = np.linspace(60.0, 180.0, GRID_POINTS)
    bias = np.zeros((GRID_POINTS, GRID_POINTS), dtype=float)
    for start in range(0, len(height), 512):
        stop = min(start + 512, len(height))
        dphi = wrap_degrees(phi[None, :] - phi_hills[start:stop, None])
        dcnn = cnn[None, :] - cnn_hills[start:stop, None]
        bias += np.einsum(
            "n,nx,ny->xy",
            height[start:stop],
            np.exp(-0.5 * (dphi / sigma_phi[start:stop, None]) ** 2),
            np.exp(-0.5 * (dcnn / sigma_cnn[start:stop, None]) ** 2),
            optimize=True,
        )
    fes = -bias.T
    fes -= fes.min()
    return phi, cnn, fes * EV_TO_KJMOL, time_fs[-1] / 1e6


def nearest(values: np.ndarray, value: float) -> int:
    return int(np.abs(values - value).argmin())


def smooth_band_path(fes: np.ndarray, x_path: np.ndarray, y_min: int, y_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Find a low-FES, monotonic path with at most one CNN grid-step per phi step."""
    y_values = np.arange(y_min, y_max + 1)
    cost = np.full((len(x_path), len(y_values)), np.inf)
    previous = np.full((len(x_path), len(y_values)), -1, dtype=int)
    cost[0] = fes[y_values, x_path[0]]
    for x_index, x in enumerate(x_path[1:], start=1):
        for y_index, y in enumerate(y_values):
            source_start = max(0, y_index - 1)
            source_stop = min(len(y_values), y_index + 2)
            source = cost[x_index - 1, source_start:source_stop]
            source_offset = int(np.argmin(source))
            previous[x_index, y_index] = source_start + source_offset
            cost[x_index, y_index] = source[source_offset] + fes[y, x]
    y_index = int(np.argmin(cost[-1]))
    path_y = np.empty(len(x_path), dtype=int)
    for x_index in range(len(x_path) - 1, -1, -1):
        path_y[x_index] = y_values[y_index]
        if x_index:
            y_index = previous[x_index, y_index]
    return x_path, path_y


def smooth_minimax_high_path(
    fes: np.ndarray,
    x_path: np.ndarray,
    start_y: int,
    end_y: int,
    x_min: int,
    x_max: int,
    high_y_min: int,
    high_y_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize the path maximum while requiring a smooth high-CNN crossing."""
    y_values = np.arange(fes.shape[0])
    inf = np.inf
    # State 0 has not crossed the high-CNN window; state 1 has crossed it.
    barrier = np.full((len(x_path), len(y_values), 2), inf)
    roughness = np.full_like(barrier, inf)
    previous_y = np.full((len(x_path), len(y_values), 2), -1, dtype=int)
    previous_state = np.full((len(x_path), len(y_values), 2), -1, dtype=int)
    initial_state = int(x_min <= x_path[0] <= x_max and high_y_min <= start_y <= high_y_max)
    barrier[0, start_y, initial_state] = fes[start_y, x_path[0]]
    roughness[0, start_y, initial_state] = 0.0
    for x_index, x in enumerate(x_path[1:], start=1):
        is_crossing_column = x_min <= x <= x_max
        for y in y_values:
            for state in (0, 1):
                next_state = int(state or (is_crossing_column and high_y_min <= y <= high_y_max))
                for prior_y in range(max(0, y - 2), min(len(y_values), y + 3)):
                    prior_barrier = barrier[x_index - 1, prior_y, state]
                    if not np.isfinite(prior_barrier):
                        continue
                    candidate_barrier = max(prior_barrier, fes[y, x])
                    candidate_roughness = roughness[x_index - 1, prior_y, state] + (y - prior_y) ** 2
                    old_barrier = barrier[x_index, y, next_state]
                    old_roughness = roughness[x_index, y, next_state]
                    if candidate_barrier < old_barrier - 1.0e-10 or (
                        abs(candidate_barrier - old_barrier) <= 1.0e-10 and candidate_roughness < old_roughness
                    ):
                        barrier[x_index, y, next_state] = candidate_barrier
                        roughness[x_index, y, next_state] = candidate_roughness
                        previous_y[x_index, y, next_state] = prior_y
                        previous_state[x_index, y, next_state] = state
    if not np.isfinite(barrier[-1, end_y, 1]):
        raise RuntimeError("No smooth path satisfies the high-CNN crossing constraint")
    path_y = np.empty(len(x_path), dtype=int)
    y, state = end_y, 1
    for x_index in range(len(x_path) - 1, -1, -1):
        path_y[x_index] = y
        if x_index:
            y, state = previous_y[x_index, y, state], previous_state[x_index, y, state]
    return x_path, path_y


def vertical_path(x: int, start_y: int, end_y: int) -> tuple[np.ndarray, np.ndarray]:
    step = 1 if end_y >= start_y else -1
    y = np.arange(start_y, end_y + step, step)
    return np.full(len(y), x, dtype=int), y


def minimax_path(fes: np.ndarray, allowed: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra path that minimizes the maximum FES encountered."""
    distance = np.full_like(fes, np.inf)
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    start_x, start_y = start
    end_x, end_y = end
    distance[start_y, start_x] = fes[start_y, start_x]
    queue = [(distance[start_y, start_x], start_x, start_y)]
    while queue:
        cost, x, y = heapq.heappop(queue)
        if (x, y) == (end_x, end_y):
            break
        if cost != distance[y, x]:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_x, next_y = x + dx, y + dy
            if not (0 <= next_x < fes.shape[1] and 0 <= next_y < fes.shape[0] and allowed[next_y, next_x]):
                continue
            next_cost = max(cost, fes[next_y, next_x])
            if next_cost < distance[next_y, next_x]:
                distance[next_y, next_x] = next_cost
                previous[(next_x, next_y)] = (x, y)
                heapq.heappush(queue, (next_cost, next_x, next_y))
    path = [(end_x, end_y)]
    while path[-1] != (start_x, start_y):
        try:
            path.append(previous[path[-1]])
        except KeyError as error:
            raise RuntimeError("No minimax route through the selected FES domain") from error
    path.reverse()
    return np.asarray([point[0] for point in path]), np.asarray([point[1] for point in path])


def region_minimum(fes: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> tuple[int, int]:
    local_y, local_x = np.unravel_index(np.argmin(fes[y_min : y_max + 1, x_min : x_max + 1]), (y_max - y_min + 1, x_max - x_min + 1))
    return x_min + int(local_x), y_min + int(local_y)


def pathway_indices(phi: np.ndarray, cnn: np.ndarray, fes: np.ndarray, system: str, high: bool) -> tuple[np.ndarray, np.ndarray]:
    phi0, phi180 = nearest(phi, 0.0), nearest(phi, 180.0)
    y_low, y_high = nearest(cnn, LOW_CNN_DEG), nearest(cnn, HIGH_CNN_DEG)
    low_min, low_max = nearest(cnn, LOW_CNN_DEG - CNN_FLEX_DEG), nearest(cnn, LOW_CNN_DEG + CNN_FLEX_DEG)
    high_min, high_max = nearest(cnn, HIGH_CNN_MIN_DEG), nearest(cnn, HIGH_CNN_MAX_DEG)
    start_x, end_x = (phi0, phi180) if system == "cis" else (phi180, phi0)
    x_path = np.arange(start_x, end_x + (1 if end_x >= start_x else -1), 1 if end_x >= start_x else -1)
    if not high:
        return smooth_band_path(fes, x_path, low_min, low_max)
    x50, x130 = nearest(phi, 50.0), nearest(phi, 130.0)
    cis_minimum = region_minimum(fes, phi0, phi0, low_min, low_max)
    trans_minimum = region_minimum(fes, phi180, phi180, low_min, low_max)
    first, second = (cis_minimum, trans_minimum) if system == "cis" else (trans_minimum, cis_minimum)
    return smooth_minimax_high_path(
        fes,
        x_path,
        first[1],
        second[1],
        x50,
        x130,
        high_min,
        high_max,
    )


def profile(fes: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    values = fes[y, x]
    return values - values[0], float(values.max() - values[0])


def bias_file(model: Model, system: str) -> Path:
    matches = sorted(
        path
        for path in (ROOT / "runs" / model.directory / system / "outputs").glob("*.bias")
        if "_failure_" not in path.name
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one bias file for {model.directory}/{system}")
    return matches[0]


def plot_system(system: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    profiles, surfaces = [], []
    for model in MODELS:
        phi, cnn, fes, time_ns = reconstruct(bias_file(model, system))
        low_x, low_y = pathway_indices(phi, cnn, fes, system, high=False)
        high_x, high_y = pathway_indices(phi, cnn, fes, system, high=True)
        low_profile, low_barrier = profile(fes, low_x, low_y)
        high_profile, high_barrier = profile(fes, high_x, high_y)
        preferred = "CNN 120 +/- 15 deg" if low_barrier <= high_barrier else "CNN 155-175 deg"
        rows.append({
            "system": system,
            "model": model.label,
            "time_ns": time_ns,
            "barrier_cnn120_kjmol": low_barrier,
            "barrier_cnn160_kjmol": high_barrier,
            "delta_160_minus_120_kjmol": high_barrier - low_barrier,
            "preferred_path": preferred,
        })
        profiles.append((model, low_profile, high_profile, low_barrier, high_barrier))
        surfaces.append((model, phi, cnn, fes, low_x, low_y, high_x, high_y))

    vmax = max(np.percentile(surface[3], 99.5) for surface in surfaces)
    levels = np.linspace(0.0, max(vmax, 1.0), 18)
    figure, axes = plt.subplots(2, 4, figsize=(15.6, 7.2), sharex=True, sharey=True)
    contour = None
    for axis, (model, phi, cnn, fes, low_x, low_y, high_x, high_y) in zip(axes.ravel(), surfaces, strict=True):
        contour = axis.contourf(phi, cnn, np.clip(fes, 0, vmax), levels=levels, cmap="RdBu_r")
        axis.plot(phi[low_x], cnn[low_y], color="black", linewidth=1.8, label="CNN 120 +/- 15 deg")
        axis.plot(phi[high_x], cnn[high_y], color="#d18b00", linewidth=1.8, label="CNN 155-175 deg")
        axis.set_title(model.label, fontsize=11)
        axis.set(xlim=(-180, 180), ylim=(60, 180), xticks=[-180, -90, 0, 90, 180], yticks=[60, 90, 120, 150, 180])
        axis.tick_params(direction="in", top=True, right=True)
    for axis in axes[-1, :]:
        axis.set_xlabel("CNNC dihedral (deg)")
    for axis in axes[:, 0]:
        axis.set_ylabel("CNN angle (deg)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.colorbar(contour, ax=axes.ravel().tolist(), pad=0.018, label="Relative free energy (kJ mol-1)")
    figure.suptitle(f"Azobenzene {system}: smooth minimax CNN pathways", y=0.995)
    figure.savefig(ROOT / f"azobenzene_{system}_constrained_minimax_cnn_pathways_fes.png", dpi=500, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(15.6, 6.8), sharey=False)
    for axis, (model, low, high, low_barrier, high_barrier) in zip(axes.ravel(), profiles, strict=True):
        axis.plot(np.linspace(0, 1, len(low)), low, color="black", label=f"CNN 120 +/- 15: {low_barrier:.1f}")
        axis.plot(np.linspace(0, 1, len(high)), high, color="#d18b00", label=f"CNN 155-175: {high_barrier:.1f}")
        axis.set_title(model.label, fontsize=11)
        axis.set_xlabel("Path progress")
        axis.set_ylabel("Delta F (kJ mol-1)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Azobenzene {system}: minimax route free-energy profiles", y=0.995)
    figure.savefig(ROOT / f"azobenzene_{system}_constrained_minimax_cnn_pathways_profiles.png", dpi=500, bbox_inches="tight")
    plt.close(figure)
    return rows


def main() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    rows = plot_system("cis") + plot_system("trans")
    with (ROOT / "azobenzene_constrained_minimax_cnn_pathways_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
