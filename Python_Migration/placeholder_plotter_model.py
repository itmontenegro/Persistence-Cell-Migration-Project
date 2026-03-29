import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_REQUIRED_COLUMNS = {
    "x",
    "y",
    "theta",
    "fmpi_x",
    "fmpi_y",
    "xi_x",
    "xi_y",
}

MODEL_REQUIRED_NPZ_KEYS = {
    "x_array",
    "y_array",
    "theta_array",
    "fmpi_x_array",
    "fmpi_y_array",
    "xi_x_array",
    "xi_y_array",
}


@dataclass
class TrajectoryRecord:
    file: Path
    x: np.ndarray
    y: np.ndarray
    sim_id: int
    source: str


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


def _extract_xy_from_npz(npz_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_file) as data:
        x = data["x_array"]
        y = data["y_array"]

    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or x.size == 0:
        raise ValueError(f"Invalid model NPZ trajectory arrays in {npz_file}")

    return x, y


def _is_model_file(csv_file: Path) -> bool:
    try:
        header_df = pd.read_csv(csv_file, nrows=0)
    except Exception:
        return False

    normalized = _normalized_columns(header_df)
    # Exclude taxis CSVs that include model columns plus theta_org.
    if "theta_org" in normalized:
        return False
    return MODEL_REQUIRED_COLUMNS.issubset(normalized)


def _is_model_npz_file(npz_file: Path) -> bool:
    try:
        with np.load(npz_file) as data:
            keys = set(data.files)
            if not MODEL_REQUIRED_NPZ_KEYS.issubset(keys):
                return False
            # Exclude taxis_gpu batches.
            if "theta_org" in keys:
                return False
            x = data["x_array"]
            y = data["y_array"]
    except Exception:
        return False

    return x.ndim == 2 and y.ndim == 2 and x.shape == y.shape and x.size > 0


def find_model_trajectory_files(data_dir: Path) -> list[Path]:
    csv_files = sorted(data_dir.rglob("Sim_*.csv"))
    npz_files = sorted(data_dir.rglob("Batch_*.npz"))
    selected_csv = [file for file in csv_files if _is_model_file(file)]
    selected_npz = [file for file in npz_files if _is_model_npz_file(file)]
    return selected_csv + selected_npz


def _load_records(files: list[Path], max_trajectories: int) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []

    for file in files:
        suffix = file.suffix.lower()
        if suffix == ".csv":
            data = pd.read_csv(file)
            x, y = _extract_xy(data)
            records.append(TrajectoryRecord(file=file, x=x, y=y, sim_id=0, source="csv"))
            if len(records) >= max_trajectories:
                break
            continue

        if suffix == ".npz":
            x_batch, y_batch = _extract_xy_from_npz(file)
            for sim_id in range(x_batch.shape[0]):
                records.append(
                    TrajectoryRecord(
                        file=file,
                        x=x_batch[sim_id],
                        y=y_batch[sim_id],
                        sim_id=sim_id,
                        source="npz",
                    )
                )
                if len(records) >= max_trajectories:
                    break
            if len(records) >= max_trajectories:
                break

    return records


def plot_trajectories(records: list[TrajectoryRecord], output_path: Path, title: str, show_plot: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for rec in records:
        x, y = rec.x, rec.y
        if rec.source == "npz":
            label = f"{rec.file.stem} | sim={rec.sim_id}"
        else:
            label = rec.file.stem
        ax.plot(x, y, linewidth=1.2, alpha=0.85, label=label)
        ax.scatter(x[0], y[0], s=10, marker="o", color=ax.lines[-1].get_color(), alpha=0.8)
        ax.scatter(x[-1], y[-1], s=20, marker="x", color=ax.lines[-1].get_color(), alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    if len(records) <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()

    plt.close(fig)


def animate_trajectories(
    records: list[TrajectoryRecord],
    title: str,
    show_plot: bool,
    save_gif: Path | None,
    interval: int = 30,
    step: int = 5,
) -> None:
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

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    trail_lines = []
    head_dots = []
    for idx, (x, y) in enumerate(trajectories):
        color = colors[idx % len(colors)]
        (line,) = ax.plot([], [], linewidth=1.0, alpha=0.6, color=color)
        (dot,) = ax.plot([], [], "o", ms=5, color=color)
        trail_lines.append((line, x, y))
        head_dots.append((dot, x, y))

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top")

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
    parser = argparse.ArgumentParser(description="Plot model.py trajectory outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("DATA"), help="Root folder containing model CSV/NPZ files.")
    parser.add_argument("--max-trajectories", type=int, default=20, help="Maximum number of trajectory files to plot.")
    parser.add_argument("--output", type=Path, default=Path("DATA") / "trajectory_plot_model.png", help="Saved figure path.")
    parser.add_argument("--title", type=str, default="Model Cell Trajectories", help="Plot title.")
    parser.add_argument("--show", action="store_true", help="Display the plot window.")
    parser.add_argument("--animate", action="store_true", help="Show animation instead of static plot.")
    parser.add_argument("--save-gif", type=Path, default=None, help="GIF path (only with --animate).")
    parser.add_argument("--interval", type=int, default=30, help="Milliseconds between animation frames.")
    parser.add_argument("--step", type=int, default=5, help="Render every Nth time step.")
    args = parser.parse_args()

    if args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be a positive integer")

    trajectory_files = find_model_trajectory_files(args.data_dir)
    if not trajectory_files:
        raise FileNotFoundError(f"No model trajectory CSV/NPZ files found under {args.data_dir}")

    records = _load_records(trajectory_files, max_trajectories=args.max_trajectories)
    if not records:
        raise FileNotFoundError(f"No readable model trajectories found under {args.data_dir}")

    if args.animate:
        animate_trajectories(
            records,
            args.title,
            show_plot=args.show,
            save_gif=args.save_gif,
            interval=args.interval,
            step=args.step,
        )
    else:
        plot_trajectories(records, args.output, args.title, args.show)
        print(f"Plotted {len(records)} model trajectories.")
        print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
