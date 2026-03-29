import argparse
import re
from pathlib import Path
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


TAXIS_REQUIRED_KEYS = {
    "theta",
    "theta_org",
    "x",
    "y",
    "xi_x",
    "xi_y",
    "fmpi_x",
    "fmpi_y",
}


@dataclass
class TrajectoryRecord:
    file: Path
    x: np.ndarray
    y: np.ndarray
    sim_id: int
    alpha: str
    h: str
    dr: str
    f: str


@dataclass
class ParameterFilters:
    alpha: set[str] | None = None
    h: set[str] | None = None
    dr: set[str] | None = None
    f: set[str] | None = None


def _param_key(record: TrajectoryRecord) -> tuple[str, str, str, str]:
    return (record.alpha, record.h, record.dr, record.f)


def _param_label(record: TrajectoryRecord) -> str:
    return f"alpha={record.alpha}, H={record.h}, Dr={record.dr}, f={record.f}"


def _normalize_param_value(value: str) -> str:
    return value.strip().replace("_", ".")


def _normalize_filter_values(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    normalized = {_normalize_param_value(v) for v in values if v.strip()}
    return normalized or None


def _matches_filters(alpha: str, h: str, dr: str, f: str, filters: ParameterFilters | None) -> bool:
    if filters is None:
        return True
    if filters.alpha is not None and alpha not in filters.alpha:
        return False
    if filters.h is not None and h not in filters.h:
        return False
    if filters.dr is not None and dr not in filters.dr:
        return False
    if filters.f is not None and f not in filters.f:
        return False
    return True


def _parse_from_path(npz_file: Path) -> tuple[str, str, str, str]:
    """Extract alpha, H, Dr and f from taxis output path and file name."""
    alpha, h, dr, f = "?", "?", "?", "?"

    path_text = str(npz_file)
    alpha_match = re.search(r"Alpha_([0-9_]+)_f_tax_([0-9_]+)", path_text)
    if alpha_match:
        alpha = alpha_match.group(1).replace("_", ".")
        f = alpha_match.group(2).replace("_", ".")

    h_match = re.search(r"[/\\]H_([0-9_]+)", path_text)
    if h_match:
        h = h_match.group(1).replace("_", ".")

    dr_match = re.search(r"[/\\]Dr_([0-9_]+)", path_text)
    if dr_match:
        dr = dr_match.group(1).replace("_", ".")

    file_match = re.match(r"Batch_Dr_([0-9_]+)_H_([0-9_]+)_f_([0-9_]+)\.npz$", npz_file.name)
    if file_match:
        dr = file_match.group(1).replace("_", ".")
        h = file_match.group(2).replace("_", ".")
        f = file_match.group(3).replace("_", ".")

    return alpha, h, dr, f


def _is_taxis_file(npz_file: Path) -> bool:
    try:
        with np.load(npz_file) as data:
            keys = set(data.files)
            if not TAXIS_REQUIRED_KEYS.issubset(keys):
                return False
            x = data["x"]
            y = data["y"]
    except Exception:
        return False

    # taxis_gpu.py stores [num_simulations, time_steps] arrays.
    return x.ndim == 2 and y.ndim == 2 and x.shape == y.shape and x.size > 0


def find_taxis_trajectory_files(data_dir: Path, filters: ParameterFilters | None = None) -> list[Path]:
    files = sorted(data_dir.rglob("Batch_*.npz"))
    selected = []
    for file in files:
        if not _is_taxis_file(file):
            continue
        alpha, h, dr, f = _parse_from_path(file)
        if _matches_filters(alpha, h, dr, f, filters):
            selected.append(file)
    return selected


def _load_records(npz_files: list[Path], max_trajectories: int | None = None) -> list[TrajectoryRecord]:
    records = []
    batches = []
    for npz_file in npz_files:
        with np.load(npz_file) as data:
            x_batch = data["x"]
            y_batch = data["y"]

        if x_batch.ndim != 2 or y_batch.ndim != 2 or x_batch.shape != y_batch.shape:
            continue

        alpha, h, dr, f = _parse_from_path(npz_file)
        batches.append((npz_file, x_batch, y_batch, alpha, h, dr, f))

    if max_trajectories is None:
        for npz_file, x_batch, y_batch, alpha, h, dr, f in batches:
            for sim_id in range(x_batch.shape[0]):
                records.append(
                    TrajectoryRecord(
                        file=npz_file,
                        x=x_batch[sim_id],
                        y=y_batch[sim_id],
                        sim_id=sim_id,
                        alpha=alpha,
                        h=h,
                        dr=dr,
                        f=f,
                    )
                )
        return records

    # Round-robin across batches so multiple selected parameter sets are represented.
    next_sim_by_batch = [0] * len(batches)
    while len(records) < max_trajectories:
        added_this_round = 0
        for idx, (npz_file, x_batch, y_batch, alpha, h, dr, f) in enumerate(batches):
            sim_id = next_sim_by_batch[idx]
            if sim_id >= x_batch.shape[0]:
                continue

            records.append(
                TrajectoryRecord(
                    file=npz_file,
                    x=x_batch[sim_id],
                    y=y_batch[sim_id],
                    sim_id=sim_id,
                    alpha=alpha,
                    h=h,
                    dr=dr,
                    f=f,
                )
            )
            next_sim_by_batch[idx] += 1
            added_this_round += 1

            if len(records) >= max_trajectories:
                break

        if added_this_round == 0:
            break

    return records


def _build_grouped_colors(records: list[TrajectoryRecord]) -> dict[Path, tuple[float, float, float, float]]:
    batch_files = sorted({rec.file for rec in records})
    if not batch_files:
        return {}

    # One stable color per batch file so all cells in the same batch share a color.
    cmap = plt.get_cmap("tab20")
    denom = max(1, len(batch_files) - 1)
    return {batch_file: cmap(idx / denom) for idx, batch_file in enumerate(batch_files)}


def _build_parameter_summary(records: list[TrajectoryRecord]) -> str:
    grouped: dict[tuple[str, str, str, str], int] = {}
    for rec in records:
        grouped[_param_key(rec)] = grouped.get(_param_key(rec), 0) + 1

    lines = []
    for alpha, h, dr, f in sorted(grouped.keys()):
        lines.append(f"alpha={alpha}, H={h}, Dr={dr}, f={f}: n={grouped[(alpha, h, dr, f)]}")
    return "\n".join(lines)


def _add_legend(
    ax: plt.Axes,
    records: list[TrajectoryRecord],
    colors: dict[Path, tuple[float, float, float, float]],
    legend_mode: str,
) -> None:
    if legend_mode == "none":
        return

    if legend_mode == "trajectory":
        if len(records) <= 18:
            ax.legend(loc="best", fontsize=8)
        return

    if legend_mode == "batch":
        by_batch: dict[Path, TrajectoryRecord] = {}
        for rec in records:
            by_batch.setdefault(rec.file, rec)

        handles = []
        labels = []
        for batch_file in sorted(by_batch.keys()):
            rec = by_batch[batch_file]
            handles.append(Line2D([0], [0], color=colors[batch_file], lw=2))
            labels.append(f"{batch_file.name} | {_param_label(rec)}")

        if handles:
            ax.legend(handles, labels, loc="best", fontsize=8)


def plot_trajectories(
    npz_files: list[Path],
    output_path: Path,
    title: str,
    show_plot: bool,
    max_trajectories: int,
    legend_mode: str,
) -> None:
    records = _load_records(npz_files, max_trajectories=max_trajectories)
    colors = _build_grouped_colors(records)

    fig, ax = plt.subplots(figsize=(10, 8))

    for rec in records:
        x, y = rec.x, rec.y
        color = colors[rec.file]
        label = f"Sim {rec.sim_id} | {_param_label(rec)}" if legend_mode == "trajectory" else None

        ax.plot(x, y, linewidth=1.2, alpha=0.9, label=label, color=color)
        ax.scatter(x[0], y[0], s=10, marker="o", color=color, alpha=0.8)
        ax.scatter(x[-1], y[-1], s=20, marker="x", color=color, alpha=0.9)
        ax.annotate(
            _param_label(rec),
            xy=(x[-1], y[-1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            alpha=0.75,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.55},
        )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    _add_legend(ax, records, colors, legend_mode)

    summary_text = _build_parameter_summary(records)
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


def animate_trajectories(
    npz_files: list[Path],
    title: str,
    show_plot: bool,
    save_gif: Path | None,
    max_trajectories: int,
    legend_mode: str,
    interval: int = 30,
    step: int = 5,
) -> None:
    records = _load_records(npz_files, max_trajectories=max_trajectories)
    colors_by_file = _build_grouped_colors(records)
    trajectories = [(rec.x, rec.y) for rec in records]

    all_x = np.concatenate([t[0] for t in trajectories])
    all_y = np.concatenate([t[1] for t in trajectories])
    pad_x = (all_x.max() - all_x.min()) * 0.05 or 1.0
    pad_y = (all_y.max() - all_y.min()) * 0.05 or 1.0
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    n_frames = len(trajectories[0][0])
    frame_indices = list(range(0, n_frames, step))
    if frame_indices[-1] != n_frames - 1:
        frame_indices.append(n_frames - 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)

    trail_lines = []
    head_dots = []
    for rec, (x, y) in zip(records, trajectories):
        color = colors_by_file[rec.file]
        label = f"Sim {rec.sim_id} | {_param_label(rec)}" if legend_mode == "trajectory" else None
        (line,) = ax.plot([], [], linewidth=1.0, alpha=0.6, color=color, label=label)
        (dot,) = ax.plot([], [], "o", ms=5, color=color)
        trail_lines.append((line, x, y))
        head_dots.append((dot, x, y))

    _add_legend(ax, records, colors_by_file, legend_mode)

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top")
    summary_text = _build_parameter_summary(records)
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
        for line, _, _ in trail_lines:
            line.set_data([], [])
        for dot, _, _ in head_dots:
            dot.set_data([], [])
        time_text.set_text("")
        return [l for l, _, _ in trail_lines] + [d for d, _, _ in head_dots] + [time_text]

    def _update(frame_idx: int):
        k = frame_indices[frame_idx]
        for line, x, y in trail_lines:
            line.set_data(x[: k + 1], y[: k + 1])
        for dot, x, y in head_dots:
            dot.set_data([x[k]], [y[k]])
        time_text.set_text(f"Step {k}/{n_frames - 1}")
        return [l for l, _, _ in trail_lines] + [d for d, _, _ in head_dots] + [time_text]

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        init_func=_init,
        interval=interval,
        blit=True,
    )

    if save_gif is not None:
        save_gif.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_gif, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to: {save_gif}")

    if show_plot:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot taxis.py trajectory outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"), help="Root folder containing taxis NPZ files.")
    parser.add_argument("--max-trajectories", type=int, default=20, help="Maximum number of simulation trajectories to plot.")
    parser.add_argument("--output", type=Path, default=Path("DATA") / "trajectory_plot_taxis.png", help="Saved figure path.")
    parser.add_argument("--title", type=str, default="Taxis Cell Trajectories", help="Plot title.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    parser.add_argument("--animate", action="store_true", help="Show animation instead of static plot.")
    parser.add_argument("--save-gif", type=Path, default=None, help="GIF path (only with --animate).")
    parser.add_argument("--interval", type=int, default=30, help="Milliseconds between animation frames.")
    parser.add_argument("--step", type=int, default=5, help="Render every Nth time step.")
    parser.add_argument("--alpha", nargs="+", default=None, help="Filter alpha values (example: --alpha 0.25).")
    parser.add_argument("--h", nargs="+", default=None, help="Filter H values (example: --h 0.5 0.9).")
    parser.add_argument("--dr", nargs="+", default=None, help="Filter Dr values (example: --dr 1 5).")
    parser.add_argument("--f", nargs="+", default=None, help="Filter f values (example: --f 0.1 0.5).")
    parser.add_argument(
        "--legend-mode",
        type=str,
        choices=["trajectory", "batch", "none"],
        default="trajectory",
        help="Legend style: trajectory (one per cell), batch (one per batch file), or none.",
    )
    args = parser.parse_args()

    if args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be a positive integer")

    filters = ParameterFilters(
        alpha=_normalize_filter_values(args.alpha),
        h=_normalize_filter_values(args.h),
        dr=_normalize_filter_values(args.dr),
        f=_normalize_filter_values(args.f),
    )

    npz_files = find_taxis_trajectory_files(args.data_dir, filters=filters)
    if not npz_files:
        raise FileNotFoundError(
            "No taxis trajectory NPZ files found for selected filters under "
            f"{args.data_dir} (alpha={args.alpha}, H={args.h}, Dr={args.dr}, f={args.f})"
        )

    if args.animate:
        animate_trajectories(
            npz_files,
            args.title,
            show_plot=args.show,
            save_gif=args.save_gif,
            max_trajectories=args.max_trajectories,
            legend_mode=args.legend_mode,
            interval=args.interval,
            step=args.step,
        )
    else:
        plot_trajectories(
            npz_files,
            args.output,
            args.title,
            args.show,
            max_trajectories=args.max_trajectories,
            legend_mode=args.legend_mode,
        )
        print(f"Plotted up to {args.max_trajectories} taxis trajectories.")
        print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
