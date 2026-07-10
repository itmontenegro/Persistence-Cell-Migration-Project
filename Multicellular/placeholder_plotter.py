import argparse
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


REQUIRED_KEYS = {
    "x_array",
    "y_array",
    "theta_array",
    "fmpi_x_array",
    "fmpi_y_array",
    "xi_x_array",
    "xi_y_array",
}

FOLLOW_MIN_WINDOW = 20.0
FOLLOW_PADDING = 0.25
PLOT_CELL_RADIUS = 1.0
PLOT_W_S = 1.0
PLOT_W_C = 1.0


@dataclass
class SimulationRecord:
    file: Path
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    fmpi_x: np.ndarray
    fmpi_y: np.ndarray
    xi_x: np.ndarray
    xi_y: np.ndarray
    sim_id: int
    alpha: str
    h1: str
    h2: str
    dr: str
    boundary: str
    fcil: str
    start: str
    cluster_size: np.ndarray | None = None
    n_clusters: np.ndarray | None = None


@dataclass
class ParameterFilters:
    alpha: set[str] | None = None
    h1: set[str] | None = None
    h2: set[str] | None = None
    dr: set[str] | None = None
    boundary: set[str] | None = None
    fcil: set[str] | None = None
    start: set[str] | None = None
    sim: set[int] | None = None


def _normalize_param_value(value: str) -> str:
    return value.strip().replace("_", ".")


def _normalize_filter_values(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized = {_normalize_param_value(v) for v in values if v.strip()}
    return normalized or None


def _normalize_sim_filter(values: list[int] | None) -> set[int] | None:
    if not values:
        return None
    return {int(v) for v in values}


def _parse_from_path(npz_file: Path) -> tuple[int, str, str, str, str, str, str, str]:
    sim_id = -1
    alpha, h1, h2, dr, boundary, fcil, start = "?", "?", "?", "?", "?", "?", "?"

    path_text = str(npz_file)
    dir_match = re.search(
        r"Alpha_([0-9_]+)/H1_([0-9_]+)_H2_([0-9_]+)/Dr_([0-9_]+)/Boundary_([A-Za-z]+)/fcil_([0-9_]+)/Start_([A-Za-z0-9_]+)",
        path_text.replace("\\", "/"),
    )
    if dir_match:
        alpha = dir_match.group(1).replace("_", ".")
        h1 = dir_match.group(2).replace("_", ".")
        h2 = dir_match.group(3).replace("_", ".")
        dr = dir_match.group(4).replace("_", ".")
        boundary = dir_match.group(5).lower()
        fcil = dir_match.group(6).replace("_", ".")
        start = dir_match.group(7)

    file_match = re.match(
        r"Sim_([0-9]+)_Dr_([0-9_]+)_H1_([0-9_]+)_H2_([0-9_]+)_Alpha_([0-9_]+)(?:_Boundary_([A-Za-z]+))?(?:_fcil_([0-9_]+))?(?:_Start_?([A-Za-z0-9_]+))?\.npz$",
        npz_file.name,
    )
    if file_match:
        sim_id = int(file_match.group(1))
        dr = file_match.group(2).replace("_", ".")
        h1 = file_match.group(3).replace("_", ".")
        h2 = file_match.group(4).replace("_", ".")
        alpha = file_match.group(5).replace("_", ".")
        if file_match.group(6):
            boundary = file_match.group(6).lower()
        if file_match.group(7):
            fcil = file_match.group(7).replace("_", ".")
        if file_match.group(8):
            start = file_match.group(8)

    return sim_id, alpha, h1, h2, dr, boundary, fcil, start


def _matches_filters(
    sim_id: int,
    alpha: str,
    h1: str,
    h2: str,
    dr: str,
    boundary: str,
    fcil: str,
    start: str,
    filters: ParameterFilters | None,
) -> bool:
    if filters is None:
        return True
    if filters.sim is not None and sim_id not in filters.sim:
        return False
    if filters.alpha is not None and alpha not in filters.alpha:
        return False
    if filters.h1 is not None and h1 not in filters.h1:
        return False
    if filters.h2 is not None and h2 not in filters.h2:
        return False
    if filters.dr is not None and dr not in filters.dr:
        return False
    if filters.boundary is not None and boundary not in filters.boundary:
        return False
    if filters.fcil is not None and fcil not in filters.fcil:
        return False
    if filters.start is not None and start not in filters.start:
        return False
    return True


def _is_multicellular_file(npz_file: Path) -> bool:
    try:
        with np.load(npz_file) as data:
            keys = set(data.files)
            if not REQUIRED_KEYS.issubset(keys):
                return False
            x = data["x_array"]
            y = data["y_array"]
    except Exception:
        return False

    return x.ndim == 2 and y.ndim == 2 and x.shape == y.shape and x.size > 0


def find_multicellular_files(data_dir: Path, filters: ParameterFilters | None = None) -> list[Path]:
    files = sorted(data_dir.rglob("Sim_*.npz"))
    selected = []
    for file in files:
        if not _is_multicellular_file(file):
            continue
        sim_id, alpha, h1, h2, dr, boundary, fcil, start = _parse_from_path(file)
        if _matches_filters(sim_id, alpha, h1, h2, dr, boundary, fcil, start, filters):
            selected.append(file)
    return selected


def _load_records(npz_files: list[Path], max_simulations: int | None = None) -> list[SimulationRecord]:
    records = []
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            x = data["x_array"]
            y = data["y_array"]
            theta = data["theta_array"]
            fmpi_x = data["fmpi_x_array"]
            fmpi_y = data["fmpi_y_array"]
            xi_x = data["xi_x_array"]
            xi_y = data["xi_y_array"]
            cluster_size = data["cluster_size_array"] if "cluster_size_array" in data.files else None
            n_clusters = data["n_clusters_array"] if "n_clusters_array" in data.files else None

        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            continue

        if cluster_size is not None and (cluster_size.ndim != 2 or cluster_size.shape != x.shape):
            cluster_size = None
        if n_clusters is not None and n_clusters.ndim != 1:
            n_clusters = None

        sim_id, alpha, h1, h2, dr, boundary, fcil, start = _parse_from_path(npz_file)
        records.append(
            SimulationRecord(
                file=npz_file,
                x=x,
                y=y,
                theta=theta,
                fmpi_x=fmpi_x,
                fmpi_y=fmpi_y,
                xi_x=xi_x,
                xi_y=xi_y,
                sim_id=sim_id,
                alpha=alpha,
                h1=h1,
                h2=h2,
                dr=dr,
                boundary=boundary,
                fcil=fcil,
                start=start,
                cluster_size=cluster_size,
                n_clusters=n_clusters,
            )
        )

    if max_simulations is None:
        return records
    return records[:max_simulations]


def _record_label(rec: SimulationRecord) -> str:
    return f"Sim={rec.sim_id} | alpha={rec.alpha}, H1={rec.h1}, H2={rec.h2}, Dr={rec.dr}, boundary={rec.boundary}, fcil={rec.fcil}, Start={rec.start}"


def _compute_intercellular_force(
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    cell_radius: float,
    adhesion_substrate: float,
    adhesion_cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    d_x = x_curr[:, None] - x_curr[None, :]
    d_y = y_curr[:, None] - y_curr[None, :]
    dist = np.sqrt(d_x**2 + d_y**2)
    np.fill_diagonal(dist, np.inf)

    mask = (dist >= cell_radius) & (dist <= 2.0 * cell_radius)
    n_x = np.zeros_like(dist)
    n_y = np.zeros_like(dist)
    n_x[mask] = d_x[mask] / dist[mask]
    n_y[mask] = d_y[mask] / dist[mask]

    f_cc = np.zeros_like(dist)
    f_cc[mask] = (2.0 / cell_radius) * (
        adhesion_substrate - ((adhesion_substrate + adhesion_cell) / cell_radius) * (dist[mask] - cell_radius)
    )

    return np.sum(f_cc * n_x, axis=1), np.sum(f_cc * n_y, axis=1)


def _build_summary(records: list[SimulationRecord]) -> str:
    grouped: dict[tuple[str, str, str, str, str, str], int] = {}
    for rec in records:
        key = (rec.alpha, rec.h1, rec.h2, rec.dr, rec.boundary, rec.fcil)
        grouped[key] = grouped.get(key, 0) + 1

    lines = []
    for alpha, h1, h2, dr, boundary, fcil in sorted(grouped.keys()):
        lines.append(
            f"alpha={alpha}, H1={h1}, H2={h2}, Dr={dr}, boundary={boundary}, fcil={fcil}: n={grouped[(alpha, h1, h2, dr, boundary, fcil)]}"
        )
    return "\n".join(lines)


def _build_record_colors(records: list[SimulationRecord]) -> dict[Path, tuple[float, float, float, float]]:
    files = sorted({rec.file for rec in records})
    if not files:
        return {}
    cmap = plt.get_cmap("tab20")
    denom = max(1, len(files) - 1)
    return {file: cmap(idx / denom) for idx, file in enumerate(files)}


def _cluster_marker_area(cluster_size: float) -> float:
    return 30.0 + 18.0 * max(float(cluster_size), 1.0)


def _cluster_circle_radius(cluster_size: float, base_radius: float) -> float:
    return base_radius * (0.7 + 0.12 * np.sqrt(max(float(cluster_size), 1.0)))


def _cluster_sizes_from_positions(x_curr: np.ndarray, y_curr: np.ndarray, cell_radius: float) -> np.ndarray:
    d_x = x_curr[:, None] - x_curr[None, :]
    d_y = y_curr[:, None] - y_curr[None, :]
    dist = np.sqrt(d_x**2 + d_y**2)
    np.fill_diagonal(dist, np.inf)

    adjacency = dist <= 2.0 * cell_radius
    n_cells = adjacency.shape[0]
    labels = -np.ones(n_cells, dtype=int)
    cluster_id = 0

    for start in range(n_cells):
        if labels[start] != -1:
            continue

        stack = [start]
        labels[start] = cluster_id
        while stack:
            cell = stack.pop()
            neighbors = np.flatnonzero(adjacency[cell] & (labels == -1))
            if neighbors.size == 0:
                continue
            labels[neighbors] = cluster_id
            stack.extend(neighbors.tolist())

        cluster_id += 1

    counts = np.bincount(labels, minlength=cluster_id)
    return counts[labels]


def _cluster_components_from_positions(
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    cell_radius: float,
) -> list[np.ndarray]:
    d_x = x_curr[:, None] - x_curr[None, :]
    d_y = y_curr[:, None] - y_curr[None, :]
    dist = np.sqrt(d_x**2 + d_y**2)
    np.fill_diagonal(dist, np.inf)

    adjacency = dist <= 2.0 * cell_radius
    n_cells = adjacency.shape[0]
    labels = -np.ones(n_cells, dtype=int)
    components: list[np.ndarray] = []

    for start in range(n_cells):
        if labels[start] != -1:
            continue

        stack = [start]
        component = []
        labels[start] = len(components)

        while stack:
            cell = stack.pop()
            component.append(cell)
            neighbors = np.flatnonzero(adjacency[cell] & (labels == -1))
            if neighbors.size == 0:
                continue
            labels[neighbors] = len(components)
            stack.extend(neighbors.tolist())

        components.append(np.array(component, dtype=int))

    return components


def _draw_cluster_overlays(
    ax: plt.Axes,
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    cell_radius: float,
    color: str,
    show_circle: bool = True,
    show_number: bool = True,
) -> None:
    components = _cluster_components_from_positions(x_curr, y_curr, cell_radius)
    for component in components:
        if component.size == 0:
            continue

        xs = x_curr[component]
        ys = y_curr[component]
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
        spread = float(np.max(np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2))) if component.size > 1 else 0.0
        overlay_radius = max(cell_radius * 0.9, spread + cell_radius * 0.6)

        if show_circle:
            ring = Circle(
                (center_x, center_y),
                radius=overlay_radius,
                facecolor="none",
                edgecolor=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
            )
            ax.add_patch(ring)
        if show_number:
            ax.text(
                center_x,
                center_y,
                str(component.size),
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.85},
            )


def _cluster_sizes_for_record(rec: SimulationRecord, t: int, cell_radius: float) -> np.ndarray:
    if rec.cluster_size is not None:
        return rec.cluster_size[:, t]
    return _cluster_sizes_from_positions(rec.x[:, t], rec.y[:, t], cell_radius)


def _add_overlay_artists(
    ax: plt.Axes,
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    cell_radius: float,
    color: str,
    label_prefix: str,
    show_circle: bool = True,
    show_number: bool = True,
) -> list[object]:
    artists: list[object] = []
    for component in _cluster_components_from_positions(x_curr, y_curr, cell_radius):
        if component.size == 0:
            continue

        xs = x_curr[component]
        ys = y_curr[component]
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
        spread = float(np.max(np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2))) if component.size > 1 else 0.0
        overlay_radius = max(cell_radius * 0.9, spread + cell_radius * 0.6)

        if show_circle:
            ring = Circle(
                (center_x, center_y),
                radius=overlay_radius,
                facecolor="none",
                edgecolor=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.8,
            )
            ax.add_patch(ring)
            artists.append(ring)

        if show_number:
            text = ax.text(
                center_x,
                center_y,
                f"{label_prefix}{component.size}",
                color=color,
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.85},
            )
            artists.append(text)

    return artists


def _create_cluster_overlay_slots(
    ax: plt.Axes,
    records: list[SimulationRecord],
) -> dict[Path, list[tuple[Circle, object]]]:
    slots: dict[Path, list[tuple[Circle, object]]] = {}
    for rec in records:
        record_slots: list[tuple[Circle, object]] = []
        for _ in range(rec.x.shape[0]):
            ring = Circle(
                (0.0, 0.0),
                radius=1.0,
                facecolor="none",
                edgecolor="none",
                linestyle="--",
                linewidth=1.6,
                alpha=0.0,
                visible=False,
            )
            ax.add_patch(ring)
            text = ax.text(
                0.0,
                0.0,
                "",
                color="black",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.0},
                visible=False,
            )
            record_slots.append((ring, text))
        slots[rec.file] = record_slots
    return slots


def _update_cluster_overlay_slots(
    record_slots: list[tuple[Circle, object]],
    components: list[np.ndarray],
    x_curr: np.ndarray,
    y_curr: np.ndarray,
    cell_radius: float,
    color: str,
    label_prefix: str,
    show_circle: bool,
    show_number: bool,
) -> list[object]:
    artists: list[object] = []

    for slot_idx, (ring, text) in enumerate(record_slots):
        if slot_idx >= len(components):
            ring.set_visible(False)
            text.set_visible(False)
            continue

        component = components[slot_idx]
        if component.size == 0:
            ring.set_visible(False)
            text.set_visible(False)
            continue

        xs = x_curr[component]
        ys = y_curr[component]
        center_x = float(np.mean(xs))
        center_y = float(np.mean(ys))
        spread = float(np.max(np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2))) if component.size > 1 else 0.0
        overlay_radius = max(cell_radius * 0.9, spread + cell_radius * 0.6)

        if show_circle:
            ring.center = (center_x, center_y)
            ring.radius = overlay_radius
            ring.set_visible(True)
            ring.set_facecolor("none")
            ring.set_edgecolor(color)
            ring.set_linestyle("--")
            ring.set_linewidth(1.6)
            ring.set_alpha(0.8)
            artists.append(ring)
        else:
            ring.set_visible(False)

        if show_number:
            text.set_position((center_x, center_y))
            text.set_text(f"{label_prefix}{component.size}")
            text.set_color(color)
            text.set_visible(True)
            text.set_bbox({"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.85})
            artists.append(text)
        else:
            text.set_visible(False)

    return artists


def _resolve_arrow_visibility(
    motor: bool | None,
    noise: bool | None,
    intercellular: bool | None,
) -> tuple[bool, bool, bool]:
    # If no arrow flags are provided, default to showing all three.
    if motor is None and noise is None and intercellular is None:
        return True, True, True
    return bool(motor), bool(noise), bool(intercellular)


def plot_multicellular(
    npz_files: list[Path],
    output_path: Path,
    title: str,
    show_plot: bool,
    max_simulations: int,
    show_motor: bool,
    show_noise: bool,
    show_intercellular: bool,
    cell_radius: float,
    adhesion_substrate: float,
    adhesion_cell: float,
    intercellular_time: str,
    arrow_scale: float,
    arrow_gain: float,
    show_path: bool,
    show_cluster_circles: bool = True,
    show_cluster_numbers: bool = True,
) -> None:
    records = _load_records(npz_files, max_simulations=max_simulations)
    if not records:
        raise ValueError("No valid multicellular simulations to plot.")

    colors = _build_record_colors(records)
    fig, ax = plt.subplots(figsize=(10, 8))

    for rec in records:
        color = colors[rec.file]
        n_cells, n_steps = rec.x.shape
        start_cluster_sizes = _cluster_sizes_for_record(rec, 0, cell_radius)
        end_cluster_sizes = _cluster_sizes_for_record(rec, n_steps - 1, cell_radius)

        if intercellular_time == "start":
            t_force = 0
        elif intercellular_time == "mid":
            t_force = n_steps // 2
        else:
            t_force = n_steps - 1

        fx_cc, fy_cc = _compute_intercellular_force(
            rec.x[:, t_force],
            rec.y[:, t_force],
            cell_radius,
            adhesion_substrate,
            adhesion_cell,
        )

        for cell_id in range(n_cells):
            x = rec.x[cell_id]
            y = rec.y[cell_id]

            label = _record_label(rec) if cell_id == 0 else None
            if show_path:
                ax.plot(x, y, linewidth=1.2, alpha=0.9, color=color, label=label)
            ax.scatter(
                x[0],
                y[0],
                s=_cluster_marker_area(start_cluster_sizes[cell_id]),
                marker="o",
                facecolors="white",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.9,
            )
            ax.scatter(
                x[-1],
                y[-1],
                s=_cluster_marker_area(end_cluster_sizes[cell_id]),
                marker="o",
                facecolors=color,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
            )

            if show_intercellular:
                ax.quiver(
                    rec.x[cell_id, t_force],
                    rec.y[cell_id, t_force],
                    arrow_gain * fx_cc[cell_id],
                    arrow_gain * fy_cc[cell_id],
                    angles="xy",
                    scale_units="xy",
                    scale=arrow_scale,
                    width=0.003,
                    pivot="mid",
                    color="tab:green",
                    alpha=0.9,
                )

        _draw_cluster_overlays(
            ax,
            rec.x[:, t_force],
            rec.y[:, t_force],
            cell_radius,
            color,
            show_circle=show_cluster_circles,
            show_number=show_cluster_numbers,
        )

    handles = []
    if show_motor:
        handles.append(Line2D([0], [0], color="tab:blue", lw=2, label="Motor force"))
    if show_noise:
        handles.append(Line2D([0], [0], color="tab:red", lw=2, label="fBm noise"))
    if show_intercellular:
        handles.append(Line2D([0], [0], color="tab:green", lw=2, label="Intercellular force"))
    handles.append(Line2D([0], [0], color="black", lw=2, label="Trajectory (per simulation color)"))
    handles.append(Line2D([0], [0], marker="o", linestyle="None", color="gray", markersize=8, label="Marker size ~ cluster size"))
    if show_cluster_circles:
        handles.append(Line2D([0], [0], color="gray", lw=1.6, ls="--", label="Cluster outline"))
    if show_cluster_numbers:
        handles.append(Line2D([0], [0], marker="", linestyle="None", color="none", label="Cluster size label"))

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    sim_handles, sim_labels = ax.get_legend_handles_labels()
    ax.legend(sim_handles + handles, sim_labels + [h.get_label() for h in handles], loc="best", fontsize=8)

    summary_text = _build_summary(records)
    if summary_text:
        ax.text(
            1.01,
            0.5,
            summary_text,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.9},
        )

    ax.text(
        0.02,
        -0.08,
        "Cluster size is encoded by marker area; larger circles mean larger connected components.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()

    plt.close(fig)


def animate_multicellular(
    npz_files: list[Path],
    title: str,
    show_plot: bool,
    save_gif: Path | None,
    max_simulations: int,
    cell_radius: float,
    adhesion_substrate: float,
    adhesion_cell: float,
    show_motor: bool,
    show_noise: bool,
    show_intercellular: bool,
    interval: int = 30,
    step: int = 5,
    arrow_scale: float = 1.0,
    arrow_gain: float = 15.0,
    follow_cells: bool = True,
    show_path: bool = True,
    show_cluster_circles: bool = True,
    show_cluster_numbers: bool = True,
) -> None:
    records = _load_records(npz_files, max_simulations=max_simulations)
    if not records:
        raise ValueError("No valid multicellular simulations to animate.")

    colors = _build_record_colors(records)
    record_index = {id(rec): idx for idx, rec in enumerate(records)}

    all_x = np.concatenate([rec.x.ravel() for rec in records])
    all_y = np.concatenate([rec.y.ravel() for rec in records])
    pad_x = (all_x.max() - all_x.min()) * 0.05 or 1.0
    pad_y = (all_y.max() - all_y.min()) * 0.05 or 1.0
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    n_steps = min(rec.x.shape[1] for rec in records)
    frame_indices = list(range(0, n_steps, step))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)

    trail_lines = []
    cell_markers = []
    motor_quivers = []
    noise_quivers = []
    inter_quivers = []
    cluster_overlay_slots = _create_cluster_overlay_slots(ax, records)

    for rec in records:
        color = colors[rec.file]
        n_cells = rec.x.shape[0]

        for cell_id in range(n_cells):
            (line,) = ax.plot([], [], linewidth=1.0, alpha=0.6, color=color)
            trail_lines.append((line, rec, cell_id))
            line.set_visible(show_path)
            circle = Circle(
                (rec.x[cell_id, 0], rec.y[cell_id, 0]),
                radius=cell_radius,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.35,
            )
            ax.add_patch(circle)
            cell_markers.append((circle, rec, cell_id))

        x0 = rec.x[:, 0]
        y0 = rec.y[:, 0]
        zeros = np.zeros_like(x0)
        motor_quivers.append(
            ax.quiver(
                x0,
                y0,
                zeros,
                zeros,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                width=0.0025,
                pivot="mid",
                color="tab:blue",
                alpha=0.85,
            )
        )
        noise_quivers.append(
            ax.quiver(
                x0,
                y0,
                zeros,
                zeros,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                width=0.0025,
                pivot="mid",
                color="tab:red",
                alpha=0.75,
            )
        )
        inter_quivers.append(
            ax.quiver(
                x0,
                y0,
                zeros,
                zeros,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                width=0.003,
                pivot="mid",
                color="tab:green",
                alpha=0.9,
            )
        )

    legend_handles = []
    if show_motor:
        legend_handles.append(Line2D([0], [0], color="tab:blue", lw=2, label="Motor force"))
    if show_noise:
        legend_handles.append(Line2D([0], [0], color="tab:red", lw=2, label="fBm noise"))
    if show_intercellular:
        legend_handles.append(Line2D([0], [0], color="tab:green", lw=2, label="Intercellular force"))
    legend_handles.append(Line2D([0], [0], color="black", lw=2, label="Trajectory"))
    legend_handles.append(Line2D([0], [0], marker="o", linestyle="None", color="gray", markersize=8, label="Marker size ~ cluster size"))
    ax.legend(handles=legend_handles, loc="best", fontsize=8)

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top")
    summary_text = _build_summary(records)
    if summary_text:
        ax.text(
            1.01,
            0.5,
            summary_text,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.9},
        )

    def _init():
        artists: list[object] = []
        if show_path:
            for line, _, _ in trail_lines:
                line.set_data([], [])
                artists.append(line)
        for marker, rec, cell_id in cell_markers:
            marker.center = (rec.x[cell_id, 0], rec.y[cell_id, 0])
            artists.append(marker)
        for rec in records:
            artists.extend(
                _update_cluster_overlay_slots(
                    cluster_overlay_slots[rec.file],
                    _cluster_components_from_positions(rec.x[:, 0], rec.y[:, 0], cell_radius),
                    rec.x[:, 0],
                    rec.y[:, 0],
                    cell_radius,
                    colors[rec.file],
                    "n=",
                    show_circle=show_cluster_circles,
                    show_number=show_cluster_numbers,
                )
            )
        time_text.set_text("")
        artists.append(time_text)
        return artists

    def _update(frame_idx: int):
        k = frame_indices[frame_idx]
        artists: list[object] = []

        if follow_cells:
            x_now = np.concatenate([rec.x[:, k] for rec in records])
            y_now = np.concatenate([rec.y[:, k] for rec in records])

            center_x = float(np.mean(x_now))
            center_y = float(np.mean(y_now))

            span_x = max(float(np.max(x_now) - np.min(x_now)), FOLLOW_MIN_WINDOW)
            span_y = max(float(np.max(y_now) - np.min(y_now)), FOLLOW_MIN_WINDOW)
            half_wx = 0.5 * span_x * (1.0 + FOLLOW_PADDING)
            half_wy = 0.5 * span_y * (1.0 + FOLLOW_PADDING)

            ax.set_xlim(center_x - half_wx, center_x + half_wx)
            ax.set_ylim(center_y - half_wy, center_y + half_wy)

        if show_path:
            for line, rec, cell_id in trail_lines:
                line.set_data(rec.x[cell_id, : k + 1], rec.y[cell_id, : k + 1])
                artists.append(line)

        for marker, rec, cell_id in cell_markers:
            marker.center = (rec.x[cell_id, k], rec.y[cell_id, k])
            marker.radius = cell_radius
            artists.append(marker)

        for rec_idx, rec in enumerate(records):
            pos = np.column_stack((rec.x[:, k], rec.y[:, k]))

            if show_motor:
                motor_quivers[rec_idx].set_offsets(pos)
                motor_quivers[rec_idx].set_UVC(arrow_gain * rec.fmpi_x[:, k], arrow_gain * rec.fmpi_y[:, k])
                artists.append(motor_quivers[rec_idx])
            else:
                motor_quivers[rec_idx].set_offsets(pos)
                motor_quivers[rec_idx].set_UVC(np.zeros(rec.x.shape[0]), np.zeros(rec.x.shape[0]))

            if show_noise:
                noise_quivers[rec_idx].set_offsets(pos)
                noise_quivers[rec_idx].set_UVC(arrow_gain * rec.xi_x[:, k], arrow_gain * rec.xi_y[:, k])
                artists.append(noise_quivers[rec_idx])
            else:
                noise_quivers[rec_idx].set_offsets(pos)
                noise_quivers[rec_idx].set_UVC(np.zeros(rec.x.shape[0]), np.zeros(rec.x.shape[0]))

            if show_intercellular:
                fx_cc, fy_cc = _compute_intercellular_force(
                    rec.x[:, k],
                    rec.y[:, k],
                    cell_radius,
                    adhesion_substrate,
                    adhesion_cell,
                )
                inter_quivers[rec_idx].set_offsets(pos)
                inter_quivers[rec_idx].set_UVC(arrow_gain * fx_cc, arrow_gain * fy_cc)
                artists.append(inter_quivers[rec_idx])
            else:
                inter_quivers[rec_idx].set_offsets(pos)
                inter_quivers[rec_idx].set_UVC(np.zeros(rec.x.shape[0]), np.zeros(rec.x.shape[0]))

        time_text.set_text(f"Step {k}/{n_steps - 1}")
        artists.append(time_text)

        for rec in records:
            artists.extend(
                _update_cluster_overlay_slots(
                    cluster_overlay_slots[rec.file],
                    _cluster_components_from_positions(rec.x[:, k], rec.y[:, k], cell_radius),
                    rec.x[:, k],
                    rec.y[:, k],
                    cell_radius,
                    colors[rec.file],
                    "n=",
                    show_circle=show_cluster_circles,
                    show_number=show_cluster_numbers,
                )
            )
        return artists

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        init_func=_init,
        interval=interval,
        blit=False,
    )

    if save_gif is not None:
        save_gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_gif, writer="pillow", fps=max(1, 1000 // interval))
        print(f"Saved animation to: {save_gif}")

    if show_plot:
        plt.show()

    plt.close(fig)


def frame_by_frame_multicellular(
    npz_files: list[Path],
    title: str,
    max_simulations: int,
    cell_radius: float,
    adhesion_substrate: float,
    adhesion_cell: float,
    show_motor: bool,
    show_noise: bool,
    show_intercellular: bool,
    step: int = 1,
    arrow_scale: float = 1.0,
    arrow_gain: float = 15.0,
    follow_cells: bool = True,
    show_path: bool = True,
    show_cluster_circles: bool = True,
    show_cluster_numbers: bool = True,
) -> None:
    records = _load_records(npz_files, max_simulations=max_simulations)
    if not records:
        raise ValueError("No valid multicellular simulations to inspect.")

    colors = _build_record_colors(records)
    record_index = {id(rec): idx for idx, rec in enumerate(records)}

    all_x = np.concatenate([rec.x.ravel() for rec in records])
    all_y = np.concatenate([rec.y.ravel() for rec in records])
    pad_x = (all_x.max() - all_x.min()) * 0.05 or 1.0
    pad_y = (all_y.max() - all_y.min()) * 0.05 or 1.0
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    n_steps = min(rec.x.shape[1] for rec in records)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)

    trail_lines = []
    cell_markers = []
    motor_quivers = []
    noise_quivers = []
    inter_quivers = []
    cluster_overlay_slots = _create_cluster_overlay_slots(ax, records)

    for rec in records:
        color = colors[rec.file]
        n_cells = rec.x.shape[0]

        for cell_id in range(n_cells):
            (line,) = ax.plot([], [], linewidth=1.0, alpha=0.6, color=color)
            trail_lines.append((line, rec, cell_id))
            line.set_visible(show_path)
            circle = Circle(
                (rec.x[cell_id, 0], rec.y[cell_id, 0]),
                radius=cell_radius,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.35,
            )
            ax.add_patch(circle)
            cell_markers.append((circle, rec, cell_id))

        x0 = rec.x[:, 0]
        y0 = rec.y[:, 0]
        zeros = np.zeros_like(x0)
        motor_quivers.append(
            ax.quiver(
                x0,
                y0,
                zeros,
                zeros,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                width=0.0025,
                pivot="mid",
                color="tab:blue",
                alpha=0.85,
            )
        )
        noise_quivers.append(
            ax.quiver(
                x0,
                y0,
                zeros,
                zeros,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                width=0.0025,
                pivot="mid",
                color="tab:red",
                alpha=0.75,
            )
        )
        inter_quivers.append(
            ax.quiver(
                x0,
                y0,
                zeros,
                zeros,
                angles="xy",
                scale_units="xy",
                scale=arrow_scale,
                width=0.003,
                pivot="mid",
                color="tab:green",
                alpha=0.9,
            )
        )

    legend_handles = []
    if show_motor:
        legend_handles.append(Line2D([0], [0], color="tab:blue", lw=2, label="Motor force"))
    if show_noise:
        legend_handles.append(Line2D([0], [0], color="tab:red", lw=2, label="fBm noise"))
    if show_intercellular:
        legend_handles.append(Line2D([0], [0], color="tab:green", lw=2, label="Intercellular force"))
    legend_handles.append(Line2D([0], [0], color="black", lw=2, label="Trajectory"))
    legend_handles.append(Line2D([0], [0], marker="o", linestyle="None", color="gray", markersize=8, label="Marker size ~ cluster size"))
    ax.legend(handles=legend_handles, loc="best", fontsize=8)

    help_text = "Controls: Left/Right = +/-1 frame | Down/Up = -/+10 frames | Home/End = first/last"
    ax.text(
        0.02,
        0.02,
        help_text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "0.8", "alpha": 0.9},
    )

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top")
    summary_text = _build_summary(records)
    if summary_text:
        ax.text(
            1.01,
            0.5,
            summary_text,
            transform=ax.transAxes,
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.9},
        )

    state = {"k": 0}

    def _render_frame(k: int) -> None:
        if follow_cells:
            x_now = np.concatenate([rec.x[:, k] for rec in records])
            y_now = np.concatenate([rec.y[:, k] for rec in records])

            center_x = float(np.mean(x_now))
            center_y = float(np.mean(y_now))
            span_x = max(float(np.max(x_now) - np.min(x_now)), FOLLOW_MIN_WINDOW)
            span_y = max(float(np.max(y_now) - np.min(y_now)), FOLLOW_MIN_WINDOW)
            half_wx = 0.5 * span_x * (1.0 + FOLLOW_PADDING)
            half_wy = 0.5 * span_y * (1.0 + FOLLOW_PADDING)

            ax.set_xlim(center_x - half_wx, center_x + half_wx)
            ax.set_ylim(center_y - half_wy, center_y + half_wy)

        if show_path:
            for line, rec, cell_id in trail_lines:
                line.set_data(rec.x[cell_id, : k + 1], rec.y[cell_id, : k + 1])

        for marker, rec, cell_id in cell_markers:
            marker.center = (rec.x[cell_id, k], rec.y[cell_id, k])
            marker.radius = cell_radius

        for rec_idx, rec in enumerate(records):
            pos = np.column_stack((rec.x[:, k], rec.y[:, k]))

            if show_motor:
                motor_quivers[rec_idx].set_offsets(pos)
                motor_quivers[rec_idx].set_UVC(arrow_gain * rec.fmpi_x[:, k], arrow_gain * rec.fmpi_y[:, k])
            else:
                motor_quivers[rec_idx].set_offsets(pos)
                motor_quivers[rec_idx].set_UVC(np.zeros(rec.x.shape[0]), np.zeros(rec.x.shape[0]))

            if show_noise:
                noise_quivers[rec_idx].set_offsets(pos)
                noise_quivers[rec_idx].set_UVC(arrow_gain * rec.xi_x[:, k], arrow_gain * rec.xi_y[:, k])
            else:
                noise_quivers[rec_idx].set_offsets(pos)
                noise_quivers[rec_idx].set_UVC(np.zeros(rec.x.shape[0]), np.zeros(rec.x.shape[0]))

            if show_intercellular:
                fx_cc, fy_cc = _compute_intercellular_force(
                    rec.x[:, k],
                    rec.y[:, k],
                    cell_radius,
                    adhesion_substrate,
                    adhesion_cell,
                )
                inter_quivers[rec_idx].set_offsets(pos)
                inter_quivers[rec_idx].set_UVC(arrow_gain * fx_cc, arrow_gain * fy_cc)
            else:
                inter_quivers[rec_idx].set_offsets(pos)
                inter_quivers[rec_idx].set_UVC(np.zeros(rec.x.shape[0]), np.zeros(rec.x.shape[0]))

        for rec in records:
            _update_cluster_overlay_slots(
                cluster_overlay_slots[rec.file],
                _cluster_components_from_positions(rec.x[:, k], rec.y[:, k], cell_radius),
                rec.x[:, k],
                rec.y[:, k],
                cell_radius,
                colors[rec.file],
                "n=",
                show_circle=show_cluster_circles,
                show_number=show_cluster_numbers,
            )

        time_text.set_text(f"Frame {k}/{n_steps - 1}")
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        k = state["k"]
        if event.key == "right":
            k += step
        elif event.key == "left":
            k -= step
        elif event.key == "up":
            k += 10 * step
        elif event.key == "down":
            k -= 10 * step
        elif event.key == "home":
            k = 0
        elif event.key == "end":
            k = n_steps - 1
        else:
            return

        k = max(0, min(n_steps - 1, k))
        state["k"] = k
        _render_frame(k)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _render_frame(0)
    plt.show()
    plt.close(fig)


def _parse_npz_list(npz_values: list[str] | None) -> list[Path]:
    if not npz_values:
        return []
    paths = [Path(v).expanduser() for v in npz_values if v.strip()]
    return [p for p in paths if p.exists() and p.suffix.lower() == ".npz"]


def _parse_filter_tokens(values: str) -> list[str]:
    if not values.strip():
        return []
    return [tok for tok in re.split(r"[\s,]+", values.strip()) if tok]


class PlotterGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Multicellular Tools GUI")
        self.selected_files: list[Path] = []
        self.presets_path = Path(__file__).with_name("gui_presets.json")
        self.model_script_path = Path(__file__).with_name("model_gpu.py")

        self.data_dir_var = tk.StringVar(value=str(Path("DATA")))
        self.output_var = tk.StringVar(value=str(Path("DATA") / "trajectory_plot_multicellular.png"))
        self.save_gif_var = tk.StringVar(value="")
        self.title_var = tk.StringVar(value="Multicellular Migration Trajectories")
        self.max_sim_var = tk.StringVar(value="3")
        self.interval_var = tk.StringVar(value="30")
        self.step_var = tk.StringVar(value="5")
        self.search_var = tk.StringVar(value="")
        self.filter_alpha_var = tk.StringVar(value="Any")
        self.filter_h1_var = tk.StringVar(value="Any")
        self.filter_h2_var = tk.StringVar(value="Any")
        self.filter_dr_var = tk.StringVar(value="Any")
        self.filter_boundary_var = tk.StringVar(value="Any")
        self.filter_fcil_var = tk.StringVar(value="Any")
        self.filter_start_var = tk.StringVar(value="Any")
        self.filter_sim_var = tk.StringVar(value="Any")
        self.status_var = tk.StringVar(value="No DATA folder loaded.")
        self.arrow_scale_var = tk.StringVar(value="1.0")
        self.arrow_gain_var = tk.StringVar(value="15.0")
        self.intercellular_time_var = tk.StringVar(value="end")
        self.show_var = tk.BooleanVar(value=True)
        self.motor_var = tk.BooleanVar(value=False)
        self.noise_var = tk.BooleanVar(value=False)
        self.intercellular_var = tk.BooleanVar(value=False)
        self.follow_var = tk.BooleanVar(value=True)
        self.show_path_var = tk.BooleanVar(value=False)
        self.show_cluster_circles_var = tk.BooleanVar(value=True)
        self.show_cluster_numbers_var = tk.BooleanVar(value=True)

        self.all_metadata: list[dict[str, object]] = []
        self.visible_metadata: list[dict[str, object]] = []

        self.sim_boundary_var = tk.StringVar(value="hard")
        self.sim_replicates_var = tk.StringVar(value="1")
        self.sim_n_cells_var = tk.StringVar(value="50")
        self.sim_r_var = tk.StringVar(value="1.0")
        self.sim_ws_var = tk.StringVar(value="1.0")
        self.sim_wc_var = tk.StringVar(value="1.0")
        self.sim_f_cil_var = tk.StringVar(value="0.1")
        self.sim_distribution_var = tk.StringVar(value="4")
        self.sim_dr_values_var = tk.StringVar(value="0.1")
        self.sim_h1_values_var = tk.StringVar(value="0.5")
        self.sim_h2_values_var = tk.StringVar(value="0.5")
        self.sim_fm_var = tk.StringVar(value="1.0")
        self.sim_gamma_s_var = tk.StringVar(value="1.0")
        self.sim_gamma_c_var = tk.StringVar(value="0.0")
        self.sim_alpha_var = tk.StringVar(value="0.0")
        self.sim_dt_var = tk.StringVar(value="0.1")
        self.sim_t_var = tk.StringVar(value="100.0")
        self.sim_preset_name_var = tk.StringVar(value="default")
        self.sim_selected_preset_var = tk.StringVar(value="")

        self._build_ui()

    def _add_labeled_entry(self, parent: tk.Widget, row: int, label: str, variable: tk.StringVar, width: int = 18) -> None:
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        tk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky="ew", padx=4, pady=2)

    def _update_arrow_controls(self) -> None:
        if self.motor_var.get() or self.noise_var.get() or self.intercellular_var.get():
            self.arrow_controls_frame.grid()
        else:
            self.arrow_controls_frame.grid_remove()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.plot_tab = tk.Frame(notebook)
        notebook.add(self.plot_tab, text="Plotter")

        self._build_plot_tab()

    def _build_plot_tab(self) -> None:
        self.plot_tab.columnconfigure(0, weight=1)

        frame_top = tk.LabelFrame(self.plot_tab, text="Data")
        frame_top.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        frame_top.columnconfigure(1, weight=1)

        self._add_labeled_entry(frame_top, 0, "Data dir", self.data_dir_var, width=60)
        tk.Button(frame_top, text="Browse Dir", command=self._browse_data_dir).grid(row=0, column=2, padx=4, pady=2)
        tk.Button(frame_top, text="Load NPZ Files", command=self._load_npz_files).grid(row=1, column=2, padx=4, pady=2)
        tk.Button(frame_top, text="Scan DATA Folder", command=self._scan_data_folder).grid(row=2, column=2, padx=4, pady=2)

        self._add_labeled_entry(frame_top, 1, "Output image", self.output_var, width=60)
        self._add_labeled_entry(frame_top, 2, "Output GIF", self.save_gif_var, width=60)

        tk.Label(frame_top, textvariable=self.status_var, anchor="w").grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=4)

        frame_filters = tk.LabelFrame(self.plot_tab, text="Search And Parameter Filters")
        frame_filters.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        frame_filters.columnconfigure(1, weight=1)
        self._add_labeled_entry(frame_filters, 0, "Search", self.search_var, width=40)

        tk.Label(frame_filters, text="alpha").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.alpha_combo = ttk.Combobox(frame_filters, textvariable=self.filter_alpha_var, state="readonly", width=20)
        self.alpha_combo.grid(row=1, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="H1").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.h1_combo = ttk.Combobox(frame_filters, textvariable=self.filter_h1_var, state="readonly", width=20)
        self.h1_combo.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="H2").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.h2_combo = ttk.Combobox(frame_filters, textvariable=self.filter_h2_var, state="readonly", width=20)
        self.h2_combo.grid(row=3, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="Dr").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        self.dr_combo = ttk.Combobox(frame_filters, textvariable=self.filter_dr_var, state="readonly", width=20)
        self.dr_combo.grid(row=4, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="boundary").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        self.boundary_combo = ttk.Combobox(frame_filters, textvariable=self.filter_boundary_var, state="readonly", width=20)
        self.boundary_combo.grid(row=5, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="fcil").grid(row=6, column=0, sticky="w", padx=4, pady=2)
        self.fcil_combo = ttk.Combobox(frame_filters, textvariable=self.filter_fcil_var, state="readonly", width=20)
        self.fcil_combo.grid(row=6, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="Start").grid(row=7, column=0, sticky="w", padx=4, pady=2)
        self.start_combo = ttk.Combobox(frame_filters, textvariable=self.filter_start_var, state="readonly", width=20)
        self.start_combo.grid(row=7, column=1, sticky="w", padx=4, pady=2)

        tk.Label(frame_filters, text="sim id").grid(row=8, column=0, sticky="w", padx=4, pady=2)
        self.sim_combo = ttk.Combobox(frame_filters, textvariable=self.filter_sim_var, state="readonly", width=20)
        self.sim_combo.grid(row=8, column=1, sticky="w", padx=4, pady=2)

        tk.Button(frame_filters, text="Apply Filters", command=self._apply_parameter_filters).grid(row=9, column=0, padx=4, pady=4, sticky="w")
        tk.Button(frame_filters, text="Reset", command=self._reset_filters).grid(row=9, column=1, padx=4, pady=4, sticky="w")

        frame_params = tk.LabelFrame(self.plot_tab, text="Plot Parameters")
        frame_params.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        frame_params.columnconfigure(1, weight=1)
        self._add_labeled_entry(frame_params, 0, "Title", self.title_var, width=40)
        self._add_labeled_entry(frame_params, 1, "max simulations", self.max_sim_var)
        self._add_labeled_entry(frame_params, 2, "interval", self.interval_var)
        self._add_labeled_entry(frame_params, 3, "step", self.step_var)

        frame_toggles = tk.LabelFrame(self.plot_tab, text="Flags")
        frame_toggles.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        tk.Checkbutton(frame_toggles, text="Show window", variable=self.show_var).grid(row=0, column=0, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Motor", variable=self.motor_var, command=self._update_arrow_controls).grid(row=0, column=1, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Noise", variable=self.noise_var, command=self._update_arrow_controls).grid(row=0, column=2, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Intercellular", variable=self.intercellular_var, command=self._update_arrow_controls).grid(row=0, column=3, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Follow cells", variable=self.follow_var).grid(row=0, column=4, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Show path", variable=self.show_path_var).grid(row=0, column=5, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Cluster circles", variable=self.show_cluster_circles_var).grid(row=0, column=6, sticky="w", padx=6)
        tk.Checkbutton(frame_toggles, text="Cluster numbers", variable=self.show_cluster_numbers_var).grid(row=0, column=7, sticky="w", padx=6)

        self.arrow_controls_frame = tk.LabelFrame(self.plot_tab, text="Arrow Settings")
        self.arrow_controls_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=6)
        self.arrow_controls_frame.columnconfigure(1, weight=1)
        self._add_labeled_entry(self.arrow_controls_frame, 0, "arrow scale", self.arrow_scale_var)
        self._add_labeled_entry(self.arrow_controls_frame, 1, "arrow gain", self.arrow_gain_var)
        tk.Label(self.arrow_controls_frame, text="intercellular time").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        tk.OptionMenu(self.arrow_controls_frame, self.intercellular_time_var, "start", "mid", "end").grid(row=2, column=1, sticky="w", padx=4, pady=2)
        self._update_arrow_controls()

        frame_files = tk.LabelFrame(self.plot_tab, text="Selected Simulations (.npz)")
        frame_files.grid(row=5, column=0, sticky="nsew", padx=8, pady=6)
        self.plot_tab.rowconfigure(5, weight=1)

        self.file_list = tk.Listbox(frame_files, selectmode=tk.EXTENDED, width=120, height=12)
        self.file_list.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        scrollbar = tk.Scrollbar(frame_files, orient="vertical", command=self.file_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.configure(yscrollcommand=scrollbar.set)
        frame_files.columnconfigure(0, weight=1)
        frame_files.rowconfigure(0, weight=1)

        frame_actions = tk.Frame(self.plot_tab)
        frame_actions.grid(row=6, column=0, sticky="ew", padx=8, pady=8)
        tk.Button(frame_actions, text="Static Plot", command=self._run_static).grid(row=0, column=0, padx=4)
        tk.Button(frame_actions, text="Animate", command=self._run_animate).grid(row=0, column=1, padx=4)
        tk.Button(frame_actions, text="Frame-by-Frame", command=self._run_frame_by_frame).grid(row=0, column=2, padx=4)

    def _build_sim_tab(self) -> None:
        self.sim_tab.columnconfigure(0, weight=1)
        self.sim_tab.rowconfigure(2, weight=1)

        frame_preset = tk.LabelFrame(self.sim_tab, text="Presets")
        frame_preset.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        frame_preset.columnconfigure(4, weight=1)

        tk.Label(frame_preset, text="Preset name").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        tk.Entry(frame_preset, textvariable=self.sim_preset_name_var, width=22).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        tk.Button(frame_preset, text="Save Preset", command=self._save_sim_preset).grid(row=0, column=2, padx=4, pady=2)

        tk.Label(frame_preset, text="Load preset").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.preset_combo = ttk.Combobox(frame_preset, textvariable=self.sim_selected_preset_var, state="readonly", width=24)
        self.preset_combo.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        tk.Button(frame_preset, text="Load", command=self._load_sim_preset).grid(row=1, column=2, padx=4, pady=2)
        tk.Button(frame_preset, text="Refresh", command=self._refresh_preset_list).grid(row=1, column=3, padx=4, pady=2)

        frame_params = tk.LabelFrame(self.sim_tab, text="Simulation Parameters (model_gpu.py)")
        frame_params.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        frame_params.columnconfigure(1, weight=1)
        frame_params.columnconfigure(3, weight=1)

        self._add_labeled_entry(frame_params, 0, "boundary", self.sim_boundary_var)
        self._add_labeled_entry(frame_params, 1, "replicates", self.sim_replicates_var)
        self._add_labeled_entry(frame_params, 2, "n_cells", self.sim_n_cells_var)
        self._add_labeled_entry(frame_params, 3, "R", self.sim_r_var)
        self._add_labeled_entry(frame_params, 4, "W_s", self.sim_ws_var)
        self._add_labeled_entry(frame_params, 5, "W_c", self.sim_wc_var)
        self._add_labeled_entry(frame_params, 6, "f_cil", self.sim_f_cil_var)
        self._add_labeled_entry(frame_params, 7, "distribution", self.sim_distribution_var)
        self._add_labeled_entry(frame_params, 8, "Dr values", self.sim_dr_values_var)
        self._add_labeled_entry(frame_params, 9, "H1 values", self.sim_h1_values_var)
        self._add_labeled_entry(frame_params, 10, "H2 values", self.sim_h2_values_var)
        self._add_labeled_entry(frame_params, 11, "Fm", self.sim_fm_var)
        self._add_labeled_entry(frame_params, 12, "gamma_s", self.sim_gamma_s_var)
        self._add_labeled_entry(frame_params, 13, "gamma_c", self.sim_gamma_c_var)
        self._add_labeled_entry(frame_params, 14, "alpha", self.sim_alpha_var)
        self._add_labeled_entry(frame_params, 15, "dt", self.sim_dt_var)
        self._add_labeled_entry(frame_params, 16, "T", self.sim_t_var)

        tk.Button(frame_params, text="Run Simulation", command=self._run_simulation).grid(row=17, column=0, padx=4, pady=6, sticky="w")

        frame_log = tk.LabelFrame(self.sim_tab, text="Simulation Log")
        frame_log.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        frame_log.columnconfigure(0, weight=1)
        frame_log.rowconfigure(0, weight=1)

        self.sim_log_text = tk.Text(frame_log, height=14, wrap="word")
        self.sim_log_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        sim_scroll = tk.Scrollbar(frame_log, orient="vertical", command=self.sim_log_text.yview)
        sim_scroll.grid(row=0, column=1, sticky="ns")
        self.sim_log_text.configure(yscrollcommand=sim_scroll.set)

        self._refresh_preset_list()

    def _simulation_payload(self) -> dict[str, str]:
        return {
            "boundary": self.sim_boundary_var.get().strip(),
            "replicates": self.sim_replicates_var.get().strip(),
            "n_cells": self.sim_n_cells_var.get().strip(),
            "R": self.sim_r_var.get().strip(),
            "W_s": self.sim_ws_var.get().strip(),
            "W_c": self.sim_wc_var.get().strip(),
            "f_cil": self.sim_f_cil_var.get().strip(),
            "distribution": self.sim_distribution_var.get().strip(),
            "dr_values": self.sim_dr_values_var.get().strip(),
            "h1_values": self.sim_h1_values_var.get().strip(),
            "h2_values": self.sim_h2_values_var.get().strip(),
            "Fm": self.sim_fm_var.get().strip(),
            "gamma_s": self.sim_gamma_s_var.get().strip(),
            "gamma_c": self.sim_gamma_c_var.get().strip(),
            "alpha": self.sim_alpha_var.get().strip(),
            "dt": self.sim_dt_var.get().strip(),
            "T": self.sim_t_var.get().strip(),
        }

    def _apply_sim_payload(self, payload: dict[str, str]) -> None:
        self.sim_boundary_var.set(payload.get("boundary", self.sim_boundary_var.get()))
        self.sim_replicates_var.set(payload.get("replicates", self.sim_replicates_var.get()))
        self.sim_n_cells_var.set(payload.get("n_cells", self.sim_n_cells_var.get()))
        self.sim_r_var.set(payload.get("R", self.sim_r_var.get()))
        self.sim_ws_var.set(payload.get("W_s", self.sim_ws_var.get()))
        self.sim_wc_var.set(payload.get("W_c", self.sim_wc_var.get()))
        self.sim_f_cil_var.set(payload.get("f_cil", self.sim_f_cil_var.get()))
        self.sim_distribution_var.set(payload.get("distribution", self.sim_distribution_var.get()))
        self.sim_dr_values_var.set(payload.get("dr_values", self.sim_dr_values_var.get()))
        self.sim_h1_values_var.set(payload.get("h1_values", self.sim_h1_values_var.get()))
        self.sim_h2_values_var.set(payload.get("h2_values", self.sim_h2_values_var.get()))
        self.sim_fm_var.set(payload.get("Fm", self.sim_fm_var.get()))
        self.sim_gamma_s_var.set(payload.get("gamma_s", self.sim_gamma_s_var.get()))
        self.sim_gamma_c_var.set(payload.get("gamma_c", self.sim_gamma_c_var.get()))
        self.sim_alpha_var.set(payload.get("alpha", self.sim_alpha_var.get()))
        self.sim_dt_var.set(payload.get("dt", self.sim_dt_var.get()))
        self.sim_t_var.set(payload.get("T", self.sim_t_var.get()))
        self._update_arrow_controls()

    def _read_presets(self) -> dict[str, dict[str, str]]:
        if not self.presets_path.exists():
            return {}
        try:
            content = json.loads(self.presets_path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                return {}
            return {str(k): v for k, v in content.items() if isinstance(v, dict)}
        except Exception:
            return {}

    def _write_presets(self, presets: dict[str, dict[str, str]]) -> None:
        self.presets_path.write_text(json.dumps(presets, indent=2), encoding="utf-8")

    def _refresh_preset_list(self) -> None:
        presets = self._read_presets()
        names = sorted(presets.keys())
        self.preset_combo["values"] = names
        if names and self.sim_selected_preset_var.get() not in names:
            self.sim_selected_preset_var.set(names[0])

    def _save_sim_preset(self) -> None:
        name = self.sim_preset_name_var.get().strip()
        if not name:
            messagebox.showerror("Preset", "Please enter a preset name.")
            return
        presets = self._read_presets()
        presets[name] = self._simulation_payload()
        self._write_presets(presets)
        self._refresh_preset_list()
        self.sim_selected_preset_var.set(name)
        messagebox.showinfo("Preset", f"Saved preset '{name}'.")

    def _load_sim_preset(self) -> None:
        name = self.sim_selected_preset_var.get().strip()
        if not name:
            messagebox.showerror("Preset", "Please select a preset.")
            return
        presets = self._read_presets()
        payload = presets.get(name)
        if payload is None:
            messagebox.showerror("Preset", f"Preset '{name}' not found.")
            return
        self._apply_sim_payload(payload)
        messagebox.showinfo("Preset", f"Loaded preset '{name}'.")

    def _append_sim_log(self, text: str) -> None:
        self.sim_log_text.insert(tk.END, text)
        self.sim_log_text.see(tk.END)

    def _build_model_command(self) -> list[str]:
        payload = self._simulation_payload()
        boundary = payload["boundary"].lower()
        if boundary not in {"hard", "periodic", "none"}:
            raise ValueError("boundary must be one of: hard, periodic, none")

        return [
            sys.executable,
            str(self.model_script_path),
            "--boundary",
            boundary,
            "--replicates",
            payload["replicates"],
            "--n-cells",
            payload["n_cells"],
            "--R",
            payload["R"],
            "--W-s",
            payload["W_s"],
            "--W-c",
            payload["W_c"],
            "--f-cil",
            payload["f_cil"],
            "--distribution",
            payload["distribution"],
            "--dr-values",
            payload["dr_values"],
            "--h1-values",
            payload["h1_values"],
            "--h2-values",
            payload["h2_values"],
            "--Fm",
            payload["Fm"],
            "--gamma-s",
            payload["gamma_s"],
            "--gamma-c",
            payload["gamma_c"],
            "--alpha",
            payload["alpha"],
            "--dt",
            payload["dt"],
            "--T",
            payload["T"],
        ]

    def _run_simulation(self) -> None:
        if not self.model_script_path.exists():
            messagebox.showerror("Simulation", f"model_gpu.py not found at: {self.model_script_path}")
            return

        try:
            cmd = self._build_model_command()
        except Exception as exc:
            messagebox.showerror("Simulation", str(exc))
            return

        self._append_sim_log("\n=== Starting simulation ===\n")
        self._append_sim_log("Command: " + " ".join(cmd) + "\n")

        def _worker() -> None:
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.model_script_path.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.root.after(0, self._append_sim_log, line)
                code = proc.wait()
                self.root.after(0, self._append_sim_log, f"=== Simulation finished (exit code {code}) ===\n")
            except Exception as exc:
                self.root.after(0, self._append_sim_log, f"Simulation failed: {exc}\n")

        threading.Thread(target=_worker, daemon=True).start()

    def _browse_data_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.data_dir_var.get() or ".")
        if selected:
            self.data_dir_var.set(selected)

    def _load_npz_files(self) -> None:
        files = filedialog.askopenfilenames(
            initialdir=self.data_dir_var.get() or ".",
            title="Select NPZ simulations",
            filetypes=[("NPZ files", "*.npz")],
        )
        if files:
            self.selected_files = [Path(p) for p in files]
            self.all_metadata = []
            for p in self.selected_files:
                sim_id, alpha, h1, h2, dr, boundary, fcil, start = _parse_from_path(p)
                self.all_metadata.append(
                    {
                        "file": p,
                        "sim_id": sim_id,
                        "alpha": alpha,
                        "h1": h1,
                        "h2": h2,
                        "dr": dr,
                        "boundary": boundary,
                        "fcil": fcil,
                        "start": start,
                    }
                )
            self._populate_filter_options()
            self._apply_parameter_filters()

    def _scan_data_folder(self) -> None:
        data_dir = Path(self.data_dir_var.get()).expanduser()
        if not data_dir.exists():
            messagebox.showerror("Invalid data dir", f"Directory does not exist: {data_dir}")
            return

        npz_files = find_multicellular_files(data_dir)
        self.all_metadata = []
        for p in npz_files:
            sim_id, alpha, h1, h2, dr, boundary, fcil, start = _parse_from_path(p)
            self.all_metadata.append(
                {
                    "file": p,
                    "sim_id": sim_id,
                    "alpha": alpha,
                    "h1": h1,
                    "h2": h2,
                    "dr": dr,
                    "boundary": boundary,
                    "fcil": fcil,
                    "start": start,
                }
            )

        self._populate_filter_options()
        self._apply_parameter_filters()
        self.status_var.set(f"Indexed {len(self.all_metadata)} simulations from {data_dir}")

    def _set_filter_values(
        self,
        alphas: list[str],
        h1s: list[str],
        h2s: list[str],
        drs: list[str],
        boundaries: list[str],
        fcils: list[str],
        starts: list[str],
        sims: list[str],
    ) -> None:
        self.alpha_combo["values"] = ["Any"] + alphas
        self.h1_combo["values"] = ["Any"] + h1s
        self.h2_combo["values"] = ["Any"] + h2s
        self.dr_combo["values"] = ["Any"] + drs
        self.boundary_combo["values"] = ["Any"] + boundaries
        self.fcil_combo["values"] = ["Any"] + fcils
        self.start_combo["values"] = ["Any"] + starts
        self.sim_combo["values"] = ["Any"] + sims

        if self.filter_alpha_var.get() not in self.alpha_combo["values"]:
            self.filter_alpha_var.set("Any")
        if self.filter_h1_var.get() not in self.h1_combo["values"]:
            self.filter_h1_var.set("Any")
        if self.filter_h2_var.get() not in self.h2_combo["values"]:
            self.filter_h2_var.set("Any")
        if self.filter_dr_var.get() not in self.dr_combo["values"]:
            self.filter_dr_var.set("Any")
        if self.filter_boundary_var.get() not in self.boundary_combo["values"]:
            self.filter_boundary_var.set("Any")
        if self.filter_fcil_var.get() not in self.fcil_combo["values"]:
            self.filter_fcil_var.set("Any")
        if self.filter_start_var.get() not in self.start_combo["values"]:
            self.filter_start_var.set("Any")
        if self.filter_sim_var.get() not in self.sim_combo["values"]:
            self.filter_sim_var.set("Any")

    def _populate_filter_options(self) -> None:
        alphas = sorted({str(m["alpha"]) for m in self.all_metadata})
        h1s = sorted({str(m["h1"]) for m in self.all_metadata})
        h2s = sorted({str(m["h2"]) for m in self.all_metadata})
        drs = sorted({str(m["dr"]) for m in self.all_metadata})
        boundaries = sorted({str(m["boundary"]) for m in self.all_metadata})
        fcils = sorted({str(m["fcil"]) for m in self.all_metadata})
        starts = sorted({str(m["start"]) for m in self.all_metadata})
        sims = sorted({str(m["sim_id"]) for m in self.all_metadata}, key=lambda v: int(v) if v.lstrip("-").isdigit() else 999999)
        self._set_filter_values(alphas, h1s, h2s, drs, boundaries, fcils, starts, sims)

    def _metadata_label(self, m: dict[str, object]) -> str:
        return (
            f"Sim={m['sim_id']} | alpha={m['alpha']}, H1={m['h1']}, H2={m['h2']}, Dr={m['dr']}, boundary={m['boundary']}, fcil={m['fcil']}, Start={m['start']} | {m['file']}"
        )

    def _refresh_listbox(self) -> None:
        self.file_list.delete(0, tk.END)
        for m in self.visible_metadata:
            self.file_list.insert(tk.END, self._metadata_label(m))

    def _match_exact_or_any(self, selected: str, value: object) -> bool:
        return selected == "Any" or str(value) == selected

    def _apply_parameter_filters(self) -> None:
        search = self.search_var.get().strip().lower()
        filtered = []
        for m in self.all_metadata:
            if not self._match_exact_or_any(self.filter_alpha_var.get(), m["alpha"]):
                continue
            if not self._match_exact_or_any(self.filter_h1_var.get(), m["h1"]):
                continue
            if not self._match_exact_or_any(self.filter_h2_var.get(), m["h2"]):
                continue
            if not self._match_exact_or_any(self.filter_dr_var.get(), m["dr"]):
                continue
            if not self._match_exact_or_any(self.filter_boundary_var.get(), m["boundary"]):
                continue
            if not self._match_exact_or_any(self.filter_fcil_var.get(), m["fcil"]):
                continue
            if not self._match_exact_or_any(self.filter_start_var.get(), m["start"]):
                continue
            if not self._match_exact_or_any(self.filter_sim_var.get(), m["sim_id"]):
                continue

            label = self._metadata_label(m).lower()
            if search and search not in label:
                continue
            filtered.append(m)

        self.visible_metadata = filtered
        self.selected_files = [Path(m["file"]) for m in self.visible_metadata]
        self._refresh_listbox()
        self.status_var.set(f"Showing {len(self.visible_metadata)} / {len(self.all_metadata)} simulations")

    def _reset_filters(self) -> None:
        self.search_var.set("")
        self.filter_alpha_var.set("Any")
        self.filter_h1_var.set("Any")
        self.filter_h2_var.set("Any")
        self.filter_dr_var.set("Any")
        self.filter_boundary_var.set("Any")
        self.filter_fcil_var.set("Any")
        self.filter_start_var.set("Any")
        self.filter_sim_var.set("Any")
        self._apply_parameter_filters()

    def _chosen_files(self) -> list[Path]:
        selected_indices = self.file_list.curselection()
        if not selected_indices:
            return [Path(m["file"]) for m in self.visible_metadata]
        return [Path(self.visible_metadata[i]["file"]) for i in selected_indices]

    def _common_kwargs(self) -> dict[str, object]:
        return {
            "title": self.title_var.get(),
            "show_plot": bool(self.show_var.get()),
            "max_simulations": int(self.max_sim_var.get()),
            "show_motor": bool(self.motor_var.get()),
            "show_noise": bool(self.noise_var.get()),
            "show_intercellular": bool(self.intercellular_var.get()),
            "step": int(self.step_var.get()),
            "arrow_scale": float(self.arrow_scale_var.get()),
            "arrow_gain": float(self.arrow_gain_var.get()),
            "follow_cells": bool(self.follow_var.get()),
            "show_path": bool(self.show_path_var.get()),
            "show_cluster_circles": bool(self.show_cluster_circles_var.get()),
            "show_cluster_numbers": bool(self.show_cluster_numbers_var.get()),
        }

    def _run_static(self) -> None:
        try:
            files = self._chosen_files()
            if not files:
                raise ValueError("No simulation files selected.")

            plot_multicellular(
                files,
                output_path=Path(self.output_var.get()).expanduser(),
                title=self.title_var.get(),
                show_plot=bool(self.show_var.get()),
                max_simulations=int(self.max_sim_var.get()),
                show_motor=bool(self.motor_var.get()),
                show_noise=bool(self.noise_var.get()),
                show_intercellular=bool(self.intercellular_var.get()),
                cell_radius=PLOT_CELL_RADIUS,
                adhesion_substrate=PLOT_W_S,
                adhesion_cell=PLOT_W_C,
                intercellular_time=self.intercellular_time_var.get(),
                arrow_scale=float(self.arrow_scale_var.get()),
                arrow_gain=float(self.arrow_gain_var.get()),
                show_path=bool(self.show_path_var.get()),
                show_cluster_circles=bool(self.show_cluster_circles_var.get()),
                show_cluster_numbers=bool(self.show_cluster_numbers_var.get()),
            )
            messagebox.showinfo("Success", "Static plot generated.")
        except Exception as exc:
            messagebox.showerror("Plot failed", str(exc))

    def _run_animate(self) -> None:
        try:
            files = self._chosen_files()
            if not files:
                raise ValueError("No simulation files selected.")

            save_gif = self.save_gif_var.get().strip()
            animate_multicellular(
                files,
                title=self.title_var.get(),
                show_plot=bool(self.show_var.get()),
                save_gif=Path(save_gif).expanduser() if save_gif else None,
                max_simulations=int(self.max_sim_var.get()),
                cell_radius=PLOT_CELL_RADIUS,
                adhesion_substrate=PLOT_W_S,
                adhesion_cell=PLOT_W_C,
                show_motor=bool(self.motor_var.get()),
                show_noise=bool(self.noise_var.get()),
                show_intercellular=bool(self.intercellular_var.get()),
                interval=int(self.interval_var.get()),
                step=int(self.step_var.get()),
                arrow_scale=float(self.arrow_scale_var.get()),
                arrow_gain=float(self.arrow_gain_var.get()),
                follow_cells=bool(self.follow_var.get()),
                show_path=bool(self.show_path_var.get()),
                show_cluster_circles=bool(self.show_cluster_circles_var.get()),
                show_cluster_numbers=bool(self.show_cluster_numbers_var.get()),
            )
            messagebox.showinfo("Success", "Animation complete.")
        except Exception as exc:
            messagebox.showerror("Animation failed", str(exc))

    def _run_frame_by_frame(self) -> None:
        try:
            files = self._chosen_files()
            if not files:
                raise ValueError("No simulation files selected.")

            frame_by_frame_multicellular(
                files,
                title=self.title_var.get(),
                max_simulations=int(self.max_sim_var.get()),
                cell_radius=PLOT_CELL_RADIUS,
                adhesion_substrate=PLOT_W_S,
                adhesion_cell=PLOT_W_C,
                show_motor=bool(self.motor_var.get()),
                show_noise=bool(self.noise_var.get()),
                show_intercellular=bool(self.intercellular_var.get()),
                step=int(self.step_var.get()),
                arrow_scale=float(self.arrow_scale_var.get()),
                arrow_gain=float(self.arrow_gain_var.get()),
                follow_cells=bool(self.follow_var.get()),
                show_path=bool(self.show_path_var.get()),
                show_cluster_circles=bool(self.show_cluster_circles_var.get()),
                show_cluster_numbers=bool(self.show_cluster_numbers_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Frame mode failed", str(exc))

    def run(self) -> None:
        self.root.minsize(980, 760)
        self.root.mainloop()


def main() -> None:
    if len(sys.argv) == 1:
        PlotterGUI().run()
        return

    parser = argparse.ArgumentParser(description="Plot multicellular model_gpu.py trajectory outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"), help="Root folder containing multicellular NPZ files.")
    parser.add_argument("--files", nargs="+", default=None, help="Optional explicit NPZ file paths to load.")
    parser.add_argument("--max-simulations", type=int, default=3, help="Maximum number of simulations to plot.")
    parser.add_argument("--output", type=Path, default=Path("DATA") / "trajectory_plot_multicellular.png", help="Saved static figure path.")
    parser.add_argument("--title", type=str, default="Multicellular Migration Trajectories", help="Plot title.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    parser.add_argument("--gui", action="store_true", help="Launch GUI app for simulation browsing and plotting.")
    parser.add_argument("--animate", action="store_true", help="Show animation instead of static plot.")
    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="Interactive frame-by-frame mode controlled by keyboard arrows.",
    )
    parser.add_argument("--save-gif", type=Path, default=None, help="GIF path (only with --animate).")
    parser.add_argument("--interval", type=int, default=30, help="Milliseconds between animation frames.")
    parser.add_argument("--step", type=int, default=5, help="Render every Nth time step in animation.")
    parser.add_argument("--alpha", nargs="+", default=None, help="Filter alpha values (example: --alpha 0.25).")
    parser.add_argument("--h1", nargs="+", default=None, help="Filter H1 values (example: --h1 0.5).")
    parser.add_argument("--h2", nargs="+", default=None, help="Filter H2 values (example: --h2 0.75).")
    parser.add_argument("--dr", nargs="+", default=None, help="Filter Dr values (example: --dr 0.1).")
    parser.add_argument("--fcil", nargs="+", default=None, help="Filter fcil values (example: --fcil 0 0.1).")
    parser.add_argument("--start", nargs="+", default=None, help="Filter Start values (example: --start 0 0.5).")
    parser.add_argument(
        "--boundary",
        nargs="+",
        default=None,
        choices=["hard", "periodic", "none"],
        help="Filter boundary type(s): hard, periodic, none.",
    )
    parser.add_argument("--sim", nargs="+", type=int, default=None, help="Filter simulation ids (example: --sim 0 2).")
    parser.add_argument(
        "--motor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show motor force arrows.",
    )
    parser.add_argument(
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show fBm noise arrows.",
    )
    parser.add_argument(
        "--intercellular",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show intercellular force arrows.",
    )
    parser.add_argument(
        "--intercellular-time",
        type=str,
        choices=["start", "mid", "end"],
        default="end",
        help="Time used for intercellular force arrows in static plot.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=1.0,
        help="Matplotlib quiver scale. Increase to shrink arrows; decrease to enlarge.",
    )
    parser.add_argument(
        "--arrow-gain",
        type=float,
        default=15.0,
        help="Multiply all force vectors for display only (useful when arrows are too small to see).",
    )
    parser.add_argument(
        "--follow-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In animation mode, move the camera to follow the cell cluster.",
    )
    parser.add_argument(
        "--show-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show full past trajectory paths (default: hidden).",
    )
    parser.add_argument(
        "--show-cluster-circles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show dashed circle overlays for clusters (default: True).",
    )
    parser.add_argument(
        "--show-cluster-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show cluster size labels at cluster centers (default: True).",
    )
    args = parser.parse_args()

    if args.gui:
        PlotterGUI().run()
        return

    if args.max_simulations <= 0:
        raise ValueError("--max-simulations must be a positive integer")
    if args.step <= 0:
        raise ValueError("--step must be a positive integer")
    if args.interval <= 0:
        raise ValueError("--interval must be a positive integer")
    if args.arrow_scale <= 0:
        raise ValueError("--arrow-scale must be a positive number")
    if args.arrow_gain <= 0:
        raise ValueError("--arrow-gain must be a positive number")

    filters = ParameterFilters(
        alpha=_normalize_filter_values(args.alpha),
        h1=_normalize_filter_values(args.h1),
        h2=_normalize_filter_values(args.h2),
        dr=_normalize_filter_values(args.dr),
        fcil=_normalize_filter_values(args.fcil),
        start=_normalize_filter_values(args.start),
        boundary=set(args.boundary) if args.boundary else None,
        sim=_normalize_sim_filter(args.sim),
    )

    cli_files = _parse_npz_list(args.files)
    if cli_files:
        npz_files = []
        for p in cli_files:
            sim_id, alpha, h1, h2, dr, boundary, fcil, start = _parse_from_path(p)
            if _is_multicellular_file(p) and _matches_filters(sim_id, alpha, h1, h2, dr, boundary, fcil, start, filters):
                npz_files.append(p)
    else:
        npz_files = find_multicellular_files(args.data_dir, filters=filters)

    if not npz_files:
        raise FileNotFoundError(
            "No multicellular trajectory NPZ files found for selected filters under "
            f"{args.data_dir} (alpha={args.alpha}, H1={args.h1}, H2={args.h2}, Dr={args.dr}, fcil={args.fcil}, Start={args.start}, boundary={args.boundary}, sim={args.sim})"
        )

    show_motor, show_noise, show_intercellular = _resolve_arrow_visibility(
        args.motor,
        args.noise,
        args.intercellular,
    )

    if args.animate and args.frame_by_frame:
        raise ValueError("Use either --animate or --frame-by-frame, not both.")

    if args.animate:
        animate_multicellular(
            npz_files,
            args.title,
            show_plot=args.show,
            save_gif=args.save_gif,
            max_simulations=args.max_simulations,
            cell_radius=PLOT_CELL_RADIUS,
            adhesion_substrate=PLOT_W_S,
            adhesion_cell=PLOT_W_C,
            show_motor=show_motor,
            show_noise=show_noise,
            show_intercellular=show_intercellular,
            interval=args.interval,
            step=args.step,
            arrow_scale=args.arrow_scale,
            arrow_gain=args.arrow_gain,
            follow_cells=args.follow_cells,
            show_path=args.show_path,
            show_cluster_circles=args.show_cluster_circles,
            show_cluster_numbers=args.show_cluster_numbers,
        )
    elif args.frame_by_frame:
        frame_by_frame_multicellular(
            npz_files,
            args.title,
            max_simulations=args.max_simulations,
            cell_radius=PLOT_CELL_RADIUS,
            adhesion_substrate=PLOT_W_S,
            adhesion_cell=PLOT_W_C,
            show_motor=show_motor,
            show_noise=show_noise,
            show_intercellular=show_intercellular,
            step=args.step,
            arrow_scale=args.arrow_scale,
            arrow_gain=args.arrow_gain,
            follow_cells=args.follow_cells,
            show_path=args.show_path,
            show_cluster_circles=args.show_cluster_circles,
            show_cluster_numbers=args.show_cluster_numbers,
        )
    else:
        plot_multicellular(
            npz_files,
            args.output,
            args.title,
            args.show,
            max_simulations=args.max_simulations,
            show_motor=show_motor,
            show_noise=show_noise,
            show_intercellular=show_intercellular,
            cell_radius=PLOT_CELL_RADIUS,
            adhesion_substrate=PLOT_W_S,
            adhesion_cell=PLOT_W_C,
            intercellular_time=args.intercellular_time,
            arrow_scale=args.arrow_scale,
            arrow_gain=args.arrow_gain,
            show_path=args.show_path,
            show_cluster_circles=args.show_cluster_circles,
            show_cluster_numbers=args.show_cluster_numbers,
        )
        print(f"Plotted up to {args.max_simulations} multicellular simulations.")
        print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()