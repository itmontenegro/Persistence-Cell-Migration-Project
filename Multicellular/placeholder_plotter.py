import argparse
import re
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class ParameterFilters:
    alpha: set[str] | None = None
    h1: set[str] | None = None
    h2: set[str] | None = None
    dr: set[str] | None = None
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


def _parse_from_path(npz_file: Path) -> tuple[int, str, str, str, str]:
    sim_id = -1
    alpha, h1, h2, dr = "?", "?", "?", "?"

    path_text = str(npz_file)
    dir_match = re.search(r"Alpha_([0-9_]+)/H1_([0-9_]+)_H2_([0-9_]+)/Dr_([0-9_]+)", path_text.replace("\\", "/"))
    if dir_match:
        alpha = dir_match.group(1).replace("_", ".")
        h1 = dir_match.group(2).replace("_", ".")
        h2 = dir_match.group(3).replace("_", ".")
        dr = dir_match.group(4).replace("_", ".")

    file_match = re.match(
        r"Sim_([0-9]+)_Dr_([0-9_]+)_H1_([0-9_]+)_H2_([0-9_]+)_Alpha_([0-9_]+)\.npz$",
        npz_file.name,
    )
    if file_match:
        sim_id = int(file_match.group(1))
        dr = file_match.group(2).replace("_", ".")
        h1 = file_match.group(3).replace("_", ".")
        h2 = file_match.group(4).replace("_", ".")
        alpha = file_match.group(5).replace("_", ".")

    return sim_id, alpha, h1, h2, dr


def _matches_filters(
    sim_id: int,
    alpha: str,
    h1: str,
    h2: str,
    dr: str,
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
        sim_id, alpha, h1, h2, dr = _parse_from_path(file)
        if _matches_filters(sim_id, alpha, h1, h2, dr, filters):
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

        if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
            continue

        sim_id, alpha, h1, h2, dr = _parse_from_path(npz_file)
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
            )
        )

    if max_simulations is None:
        return records
    return records[:max_simulations]


def _record_label(rec: SimulationRecord) -> str:
    return f"Sim={rec.sim_id} | alpha={rec.alpha}, H1={rec.h1}, H2={rec.h2}, Dr={rec.dr}"


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
    grouped: dict[tuple[str, str, str, str], int] = {}
    for rec in records:
        key = (rec.alpha, rec.h1, rec.h2, rec.dr)
        grouped[key] = grouped.get(key, 0) + 1

    lines = []
    for alpha, h1, h2, dr in sorted(grouped.keys()):
        lines.append(f"alpha={alpha}, H1={h1}, H2={h2}, Dr={dr}: n={grouped[(alpha, h1, h2, dr)]}")
    return "\n".join(lines)


def _build_record_colors(records: list[SimulationRecord]) -> dict[Path, tuple[float, float, float, float]]:
    files = sorted({rec.file for rec in records})
    if not files:
        return {}
    cmap = plt.get_cmap("tab20")
    denom = max(1, len(files) - 1)
    return {file: cmap(idx / denom) for idx, file in enumerate(files)}


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
) -> None:
    records = _load_records(npz_files, max_simulations=max_simulations)
    if not records:
        raise ValueError("No valid multicellular simulations to plot.")

    colors = _build_record_colors(records)
    fig, ax = plt.subplots(figsize=(10, 8))

    for rec in records:
        color = colors[rec.file]
        n_cells, n_steps = rec.x.shape

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
            ax.scatter(x[0], y[0], s=10, marker="o", color=color, alpha=0.85)
            ax.scatter(x[-1], y[-1], s=20, marker="x", color=color, alpha=0.95)

            if show_motor:
                ax.quiver(
                    x[-1],
                    y[-1],
                    arrow_gain * rec.fmpi_x[cell_id, -1],
                    arrow_gain * rec.fmpi_y[cell_id, -1],
                    angles="xy",
                    scale_units="xy",
                    scale=arrow_scale,
                    width=0.0025,
                    pivot="mid",
                    color="tab:blue",
                    alpha=0.85,
                )

            if show_noise:
                ax.quiver(
                    x[-1],
                    y[-1],
                    arrow_gain * rec.xi_x[cell_id, -1],
                    arrow_gain * rec.xi_y[cell_id, -1],
                    angles="xy",
                    scale_units="xy",
                    scale=arrow_scale,
                    width=0.0025,
                    pivot="mid",
                    color="tab:red",
                    alpha=0.75,
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

    handles = [Line2D([0], [0], color="tab:blue", lw=2, label="Motor force")]
    if show_noise:
        handles.append(Line2D([0], [0], color="tab:red", lw=2, label="fBm noise"))
    if show_intercellular:
        handles.append(Line2D([0], [0], color="tab:green", lw=2, label="Intercellular force"))
    handles.append(Line2D([0], [0], color="black", lw=2, label="Trajectory (per simulation color)"))

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
) -> None:
    records = _load_records(npz_files, max_simulations=max_simulations)
    if not records:
        raise ValueError("No valid multicellular simulations to animate.")

    colors = _build_record_colors(records)

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

    legend_handles = [Line2D([0], [0], color="tab:blue", lw=2, label="Motor force")]
    if show_noise:
        legend_handles.append(Line2D([0], [0], color="tab:red", lw=2, label="fBm noise"))
    if show_intercellular:
        legend_handles.append(Line2D([0], [0], color="tab:green", lw=2, label="Intercellular force"))
    legend_handles.append(Line2D([0], [0], color="black", lw=2, label="Trajectory"))
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
) -> None:
    records = _load_records(npz_files, max_simulations=max_simulations)
    if not records:
        raise ValueError("No valid multicellular simulations to inspect.")

    colors = _build_record_colors(records)

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

    legend_handles = [Line2D([0], [0], color="tab:blue", lw=2, label="Motor force")]
    if show_noise:
        legend_handles.append(Line2D([0], [0], color="tab:red", lw=2, label="fBm noise"))
    if show_intercellular:
        legend_handles.append(Line2D([0], [0], color="tab:green", lw=2, label="Intercellular force"))
    legend_handles.append(Line2D([0], [0], color="black", lw=2, label="Trajectory"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multicellular model_gpu.py trajectory outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"), help="Root folder containing multicellular NPZ files.")
    parser.add_argument("--max-simulations", type=int, default=3, help="Maximum number of simulations to plot.")
    parser.add_argument("--output", type=Path, default=Path("DATA") / "trajectory_plot_multicellular.png", help="Saved static figure path.")
    parser.add_argument("--title", type=str, default="Multicellular Migration Trajectories", help="Plot title.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
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
    parser.add_argument("--sim", nargs="+", type=int, default=None, help="Filter simulation ids (example: --sim 0 2).")
    parser.add_argument("--R", type=float, default=1.0, help="Cell radius for intercellular force calculation.")
    parser.add_argument("--W-s", type=float, default=1.0, dest="w_s", help="Cell-substrate adhesion term W_s.")
    parser.add_argument("--W-c", type=float, default=0.5, dest="w_c", help="Cell-cell adhesion term W_c.")
    parser.add_argument("--no-motor", action="store_true", help="Hide motor force arrows.")
    parser.add_argument("--no-noise", action="store_true", help="Hide fBm noise arrows.")
    parser.add_argument("--no-intercellular", action="store_true", help="Hide intercellular force arrows.")
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
        default=True,
        help="Show full past trajectory paths.",
    )
    args = parser.parse_args()

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
        sim=_normalize_sim_filter(args.sim),
    )

    npz_files = find_multicellular_files(args.data_dir, filters=filters)
    if not npz_files:
        raise FileNotFoundError(
            "No multicellular trajectory NPZ files found for selected filters under "
            f"{args.data_dir} (alpha={args.alpha}, H1={args.h1}, H2={args.h2}, Dr={args.dr}, sim={args.sim})"
        )

    show_motor = not args.no_motor
    show_noise = not args.no_noise
    show_intercellular = not args.no_intercellular

    if args.animate and args.frame_by_frame:
        raise ValueError("Use either --animate or --frame-by-frame, not both.")

    if args.animate:
        animate_multicellular(
            npz_files,
            args.title,
            show_plot=args.show,
            save_gif=args.save_gif,
            max_simulations=args.max_simulations,
            cell_radius=args.R,
            adhesion_substrate=args.w_s,
            adhesion_cell=args.w_c,
            show_motor=show_motor,
            show_noise=show_noise,
            show_intercellular=show_intercellular,
            interval=args.interval,
            step=args.step,
            arrow_scale=args.arrow_scale,
            arrow_gain=args.arrow_gain,
            follow_cells=args.follow_cells,
            show_path=args.show_path,
        )
    elif args.frame_by_frame:
        frame_by_frame_multicellular(
            npz_files,
            args.title,
            max_simulations=args.max_simulations,
            cell_radius=args.R,
            adhesion_substrate=args.w_s,
            adhesion_cell=args.w_c,
            show_motor=show_motor,
            show_noise=show_noise,
            show_intercellular=show_intercellular,
            step=args.step,
            arrow_scale=args.arrow_scale,
            arrow_gain=args.arrow_gain,
            follow_cells=args.follow_cells,
            show_path=args.show_path,
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
            cell_radius=args.R,
            adhesion_substrate=args.w_s,
            adhesion_cell=args.w_c,
            intercellular_time=args.intercellular_time,
            arrow_scale=args.arrow_scale,
            arrow_gain=args.arrow_gain,
            show_path=args.show_path,
        )
        print(f"Plotted up to {args.max_simulations} multicellular simulations.")
        print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()