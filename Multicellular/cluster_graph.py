from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_from_path(npz_file: Path) -> tuple[int, str, str, str, str, str, str, str]:
	sim_id = -1
	alpha, h1, h2, dr, boundary, fcil, start = "?", "?", "?", "?", "?", "?", "?"

	path_text = str(npz_file).replace("\\", "/")
	dir_match = re.search(
		r"Alpha_([0-9_]+)/H1_([0-9_]+)_H2_([0-9_]+)/Dr_([0-9_]+)/Boundary_([A-Za-z]+)/fcil_([0-9_]+)/Start_([A-Za-z0-9_]+)",
		path_text,
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
		r"Sim_([0-9]+)_Dr_([0-9_]+)_H1_([0-9_]+)_H2_([0-9_]+)_Alpha_([0-9_]+)(?:_Boundary_([A-Za-z]+))?(?:_fcil_([0-9_]+))?(?:_Start_([A-Za-z0-9_]+))?\.npz$",
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


def _normalize_for_filename(value: str) -> str:
	return value.replace(".", "_").replace(" ", "_")


def _group_paths_by_arguments(selected_paths: list[Path]) -> list[tuple[tuple[str, str, str, str, str, str, str], list[Path]]]:
	grouped: dict[tuple[str, str, str, str, str, str, str], list[Path]] = {}
	for path in sorted(selected_paths):
		_, alpha, h1, h2, dr, boundary, fcil, start = _parse_from_path(path)
		key = (alpha, h1, h2, dr, boundary, fcil, start)
		grouped.setdefault(key, []).append(path)
	return sorted(grouped.items(), key=lambda item: item[0])


def _discover_npz_files(data_root: Path) -> list[Path]:
	if not data_root.exists():
		raise FileNotFoundError(f"Data root not found: {data_root}")
	return sorted(path for path in data_root.rglob("*.npz") if path.is_file())


def _build_output_name(group_key: tuple[str, str, str, str, str, str, str], sim_count: int) -> str:
	alpha, h1, h2, dr, boundary, fcil, start = group_key
	parts = [
		"cluster_summary",
		f"H2_{_normalize_for_filename(h2)}",
		f"f_cil_{_normalize_for_filename(fcil)}",
		f"sims_{sim_count}",
		f"boundary_{_normalize_for_filename(boundary)}",
		f"start_{_normalize_for_filename(start)}",
	]
	return "_".join(parts) + ".png"


def _render_cluster_summary(selected_paths: list[Path], output_path: Path) -> None:
	time_series = []
	clusters_gt2_series = []
	max_cluster_series = []
	avg_cluster_series = []

	for path in selected_paths:
		time_steps, clusters_gt2, max_cluster_size, avg_cluster_size = _load_single_simulation(path)
		time_series.append(time_steps)
		clusters_gt2_series.append(clusters_gt2)
		max_cluster_series.append(max_cluster_size)
		avg_cluster_series.append(avg_cluster_size)

	aligned_clusters_gt2 = _align_series(clusters_gt2_series)
	aligned_max_cluster = _align_series(max_cluster_series)
	aligned_avg_cluster = _align_series(avg_cluster_series)
	min_length = min(series.shape[0] for series in time_series)
	time_steps = time_series[0][:min_length]

	fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

	_plot_mean_with_dispersion(
		axes[0],
		time_steps,
		aligned_clusters_gt2,
		"Time vs Number of Clusters with Size > 2",
		"Number of clusters",
		"tab:blue",
	)

	_plot_mean_with_dispersion(
		axes[1],
		time_steps,
		aligned_max_cluster,
		"Time vs Max Cluster Size",
		"Max cluster size",
		"tab:green",
	)

	_plot_mean_with_dispersion(
		axes[2],
		time_steps,
		aligned_avg_cluster,
		"Time vs Average Cluster Size",
		"Average cluster size",
		"tab:orange",
		ytick_step=5,
	)

	axes[2].set_xlabel("Time step")

	group_key = _parse_from_path(selected_paths[0])[1:]
	alpha, h1, h2, dr, boundary, fcil, start = group_key
	fig.suptitle(
		f"Cluster summary over {len(selected_paths)} simulation(s) | H2={h2}, f_cil={fcil}, boundary={boundary}, start={start}"
	)
	fig.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved figure to: {output_path}")


def _count_clusters_larger_than_two(cluster_size_array: np.ndarray) -> np.ndarray:
	# Each cluster size is repeated once per cell in that cluster.
	# Summing 1 / cluster_size over cells belonging to clusters of size > 2
	# gives the number of distinct clusters at each time step.
	counts = []
	for t in range(cluster_size_array.shape[1]):
		sizes = cluster_size_array[:, t]
		valid = sizes > 2
		counts.append(float(np.sum(1.0 / sizes[valid])) if np.any(valid) else 0.0)
	return np.asarray(counts)


def _max_cluster_size(cluster_size_array: np.ndarray) -> np.ndarray:
	return np.max(cluster_size_array, axis=0)


def _average_cluster_size(cluster_size_array: np.ndarray) -> np.ndarray:
	# Each cluster size is repeated once per cell in that cluster.
	# The number of distinct clusters at time t is sum(1 / cluster_size) across cells.
	# Average cluster size = total cells / number of distinct clusters.
	cell_count = cluster_size_array.shape[0]
	counts = []
	for t in range(cluster_size_array.shape[1]):
		sizes = cluster_size_array[:, t]
		valid = sizes > 0
		n_clusters = float(np.sum(1.0 / sizes[valid])) if np.any(valid) else 0.0
		counts.append(cell_count / n_clusters if n_clusters > 0 else 0.0)
	return np.asarray(counts)


def _load_single_simulation(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if not npz_path.exists():
		raise FileNotFoundError(f"File not found: {npz_path}")

	with np.load(npz_path) as data:
		if "cluster_size_array" not in data.files:
			raise KeyError(f"cluster_size_array not found in {npz_path}")

		cluster_size_array = np.asarray(data["cluster_size_array"])

	if cluster_size_array.ndim != 2:
		raise ValueError(f"cluster_size_array must be 2D, got shape {cluster_size_array.shape}")

	time_steps = np.arange(cluster_size_array.shape[1])
	clusters_gt2 = _count_clusters_larger_than_two(cluster_size_array)
	max_cluster_size = _max_cluster_size(cluster_size_array)
	avg_cluster_size = _average_cluster_size(cluster_size_array)
	return time_steps, clusters_gt2, max_cluster_size, avg_cluster_size


def _align_series(series_list: list[np.ndarray]) -> np.ndarray:
	if not series_list:
		raise ValueError("No simulation series were loaded.")

	min_length = min(series.shape[0] for series in series_list)
	if min_length <= 0:
		raise ValueError("Loaded series have no time steps.")

	trimmed = [series[:min_length] for series in series_list]
	return np.vstack(trimmed)


def _plot_mean_with_dispersion(
	ax: plt.Axes,
	time_steps: np.ndarray,
	values_stack: np.ndarray,
	title: str,
	ylabel: str,
	color: str,
	ytick_step: int = 10,
) -> None:
	mean_values = np.mean(values_stack, axis=0)
	lower_values = np.min(values_stack, axis=0)
	upper_values = np.max(values_stack, axis=0)

	ax.fill_between(time_steps, lower_values, upper_values, color=color, alpha=0.15, linewidth=0)
	ax.plot(time_steps, lower_values, color=color, alpha=0.35, linewidth=1.0)
	ax.plot(time_steps, upper_values, color=color, alpha=0.35, linewidth=1.0)
	ax.plot(
		time_steps,
		mean_values,
		color=color,
		linewidth=2.2,
		marker="o",
		markersize=3,
		markevery=max(1, len(time_steps) // 25),
		label="mean",
	)

	ax.set_title(title)
	ax.set_ylabel(ylabel)
	upper_limit = max(float(ytick_step), float(np.ceil(np.max(upper_values) / ytick_step) * ytick_step))
	ax.set_yticks(np.arange(0, upper_limit + 1, ytick_step))
	ax.grid(True, axis="both", alpha=0.3)
	ax.legend(loc="best")


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate cluster summary plots for each simulation argument set.")
	parser.add_argument("paths", nargs="*", help="Optional .npz files or directories to scan. If omitted, --data-root is scanned.")
	parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "DATA", help="Root directory to scan for .npz files when no explicit paths are provided.")
	parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "graphs", help="Directory where plots are saved.")
	args = parser.parse_args()

	if args.paths:
		selected_paths: list[Path] = []
		for raw_path in args.paths:
			path = Path(raw_path)
			if path.is_dir():
				selected_paths.extend(path.rglob("*.npz"))
			elif path.is_file() and path.suffix.lower() == ".npz":
				selected_paths.append(path)
	else:
		selected_paths = _discover_npz_files(args.data_root)

	selected_paths = sorted({path.resolve() for path in selected_paths if path.exists() and path.suffix.lower() == ".npz"})
	if not selected_paths:
		raise FileNotFoundError("No .npz files were found. Provide explicit paths or point --data-root at the data directory.")

	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	grouped_paths = _group_paths_by_arguments(selected_paths)
	if not grouped_paths:
		raise ValueError("No valid simulation groups were found.")

	for group_key, group_paths in grouped_paths:
		output_path = output_dir / _build_output_name(group_key, len(group_paths))
		_render_cluster_summary(group_paths, output_path)


if __name__ == "__main__":
	main()
