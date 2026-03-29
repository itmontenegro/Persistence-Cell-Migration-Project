import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


def _normalize_column_name(name: str) -> str:
    """Normalize CSV header names for robust column lookup."""
    # np.savetxt prepends '# ' to header by default; strip that and normalize casing.
    return name.strip().lstrip("#").strip().lower()


def _extract_xy(data: pd.DataFrame, source: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract x/y arrays from a trajectory dataframe with flexible header handling."""
    normalized_to_original = {_normalize_column_name(col): col for col in data.columns}

    x_col = normalized_to_original.get("x")
    y_col = normalized_to_original.get("y")

    if x_col is None or y_col is None:
        raise ValueError(
            f"Missing x/y columns in {source}. Available columns: {list(data.columns)}"
        )

    return data[x_col].to_numpy(), data[y_col].to_numpy()


def find_trajectory_files(data_dir: Path) -> list[Path]:
    """Return sorted trajectory CSV files under the data directory."""
    return sorted(data_dir.rglob("Sim_*.csv"))


def plot_trajectories(csv_files: list[Path], output_path: Path, title: str, show_plot: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)
        x, y = _extract_xy(data, csv_file)

        label = csv_file.stem
        ax.plot(x, y, linewidth=1.2, alpha=0.85, label=label)

        # Mark initial and final position for each cell trajectory.
        ax.scatter(x[0], y[0], s=10, marker="o", color=ax.lines[-1].get_color(), alpha=0.8)
        ax.scatter(x[-1], y[-1], s=20, marker="x", color=ax.lines[-1].get_color(), alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # Keep legend readable when plotting many trajectories.
    if len(csv_files) <= 12:
        ax.legend(loc="best", fontsize=8)

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
    """Animate cells moving along their trajectories."""
    trajectories = []
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        trajectories.append(_extract_xy(data, csv_file))

    # Common axis limits with some padding.
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

    def _update(frame_idx):
        k = frame_indices[frame_idx]
        for line, x, y in trail_lines:
            line.set_data(x[:k+1], y[:k+1])
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
    parser = argparse.ArgumentParser(description="Plot simulated cell trajectories from CSV outputs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DATA"),
        help="Root folder that contains trajectory CSV files (default: DATA).",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=20,
        help="Maximum number of trajectory files to plot (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DATA") / "trajectory_plot.png",
        help="Path of the saved figure image.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Simulated Cell Trajectories",
        help="Plot title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the image.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show an animation of cells moving instead of a static plot.",
    )
    parser.add_argument(
        "--save-gif",
        type=Path,
        default=None,
        help="Save the animation as a GIF to this path (requires --animate).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Milliseconds between animation frames (default: 30).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Only render every Nth time step in the animation (default: 5).",
    )

    args = parser.parse_args()

    if args.max_trajectories <= 0:
        raise ValueError("--max-trajectories must be a positive integer")

    csv_files = find_trajectory_files(args.data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No trajectory CSV files found under {args.data_dir}")

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
        print(f"Plotted {len(selected_files)} trajectories.")
        print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
