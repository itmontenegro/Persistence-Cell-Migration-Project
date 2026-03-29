import argparse
import re
from pathlib import Path
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TAXIS_REQUIRED_COLUMNS = {
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


def _param_key(record: TrajectoryRecord) -> tuple[str, str, str, str]:
    return (record.alpha, record.h, record.dr, record.f)


def _param_label(record: TrajectoryRecord) -> str:
    return f"alpha={record.alpha}, H={record.h}, Dr={record.dr}, f={record.f}"


def _parse_from_path(csv_file: Path) -> tuple[str, str, str, str, int]:
    """Extract alpha, H, Dr, f and sim id from taxis output path and file name."""
    alpha, h, dr, f = "?", "?", "?", "?"
    sim_id = -1

    path_text = str(csv_file)
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

    file_match = re.match(r"Sim_(\d+)_Dr_([0-9_]+)_H_([0-9_]+)_f_([0-9_]+)\.csv$", csv_file.name)
    if file_match:
        sim_id = int(file_match.group(1))
        dr = file_match.group(2).replace("_", ".")
        h = file_match.group(3).replace("_", ".")
        f = file_match.group(4).replace("_", ".")

    return alpha, h, dr, f, sim_id


def _normalize_column_name(name: str) -> str:
    return name.strip().lstrip("#").strip().lower()


def _normalized_columns(data: pd.DataFrame) -> set[str]:
    return {_normalize_column_name(col) for col in data.columns}


def _extract_xy(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    normalized_to_original = {_normalize_column_name(col): col for col in data.columns}
    x_col = normalized_to_original.get("x")
    y_col = normalized_to_original.get("y")
    if x_col is None or y_col is None:
        raise ValueError(f"Missing x/y columns. Available columns: {list(data.columns)}")
    return data[x_col].to_numpy(), data[y_col].to_numpy()


def _is_taxis_file(csv_file: Path) -> bool:
    try:
        header_df = pd.read_csv(csv_file, nrows=0)
    except Exception:
        return False

    return TAXIS_REQUIRED_COLUMNS.issubset(_normalized_columns(header_df))


def find_taxis_trajectory_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("Sim_*.csv"))
    return [file for file in files if _is_taxis_file(file)]


def _load_records(csv_files: list[Path]) -> list[TrajectoryRecord]:
    records = []
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        x, y = _extract_xy(data)
        alpha, h, dr, f, sim_id = _parse_from_path(csv_file)
        records.append(
            TrajectoryRecord(
                file=csv_file,
                x=x,
                y=y,
                sim_id=sim_id,
                alpha=alpha,
                h=h,
                dr=dr,
                f=f,
            )
        )
    return records


def _build_grouped_colors(records: list[TrajectoryRecord]) -> dict[Path, tuple[float, float, float, float]]:
    grouped: dict[tuple[str, str, str, str], list[TrajectoryRecord]] = {}
    for rec in records:
        grouped.setdefault(_param_key(rec), []).append(rec)

    cmaps = [
        "Blues",
        "Greens",
        "Oranges",
        "Purples",
        "Reds",
        "cividis",
        "magma",
        "viridis",
    ]

    colors_by_file: dict[Path, tuple[float, float, float, float]] = {}
    for group_idx, key in enumerate(sorted(grouped.keys())):
        cmap = plt.get_cmap(cmaps[group_idx % len(cmaps)])
        members = sorted(grouped[key], key=lambda rec: rec.sim_id)
        count = len(members)
        for idx, rec in enumerate(members):
            t = (idx + 1) / (count + 1)
            colors_by_file[rec.file] = cmap(t)

    return colors_by_file


def _build_parameter_summary(records: list[TrajectoryRecord]) -> str:
    grouped: dict[tuple[str, str, str, str], int] = {}
    for rec in records:
        grouped[_param_key(rec)] = grouped.get(_param_key(rec), 0) + 1

    lines = []
    for alpha, h, dr, f in sorted(grouped.keys()):
        lines.append(f"alpha={alpha}, H={h}, Dr={dr}, f={f}: n={grouped[(alpha, h, dr, f)]}")
    return "\n".join(lines)


def plot_trajectories(csv_files: list[Path], output_path: Path, title: str, show_plot: bool) -> None:
    records = _load_records(csv_files)
    colors = _build_grouped_colors(records)

    fig, ax = plt.subplots(figsize=(10, 8))

    for rec in records:
        x, y = rec.x, rec.y
        color = colors[rec.file]
        label = f"Sim {rec.sim_id} | {_param_label(rec)}"

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

    if len(csv_files) <= 18:
        ax.legend(loc="best", fontsize=8)

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
    csv_files: list[Path],
    title: str,
    show_plot: bool,
    save_gif: Path | None,
    interval: int = 30,
    step: int = 5,
) -> None:
    records = _load_records(csv_files)
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
        (line,) = ax.plot([], [], linewidth=1.0, alpha=0.6, color=color)
        (dot,) = ax.plot([], [], "o", ms=5, color=color)
        trail_lines.append((line, x, y))
        head_dots.append((dot, x, y))

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
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"), help="Root folder containing taxis CSV files.")
    parser.add_argument("--max-trajectories", type=int, default=20, help="Maximum number of trajectory files to plot.")
    parser.add_argument("--output", type=Path, default=Path("DATA") / "trajectory_plot_taxis.png", help="Saved figure path.")
    parser.add_argument("--title", type=str, default="Taxis Cell Trajectories", help="Plot title.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    parser.add_argument("--animate", action="store_true", help="Show animation instead of static plot.")
    parser.add_argument("--save-gif", type=Path, default=None, help="GIF path (only with --animate).")
    parser.add_argument("--interval", type=int, default=30, help="Milliseconds between animation frames.")
    parser.add_argument("--step", type=int, default=5, help="Render every Nth time step.")
    args = parser.parse_args()

    if args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be a positive integer")

    csv_files = find_taxis_trajectory_files(args.data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No taxis trajectory CSV files found under {args.data_dir}")

    selected_files = csv_files[: args.max_trajectories]

    if args.animate:
        animate_trajectories(
            selected_files,
            args.title,
            show_plot=args.show,
            save_gif=args.save_gif,
            interval=args.interval,
            step=args.step,
        )
    else:
        plot_trajectories(selected_files, args.output, args.title, args.show)
        print(f"Plotted {len(selected_files)} taxis trajectories.")
        print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
