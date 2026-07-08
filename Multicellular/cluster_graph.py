from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



file_paths = [
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_0_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_1_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
    Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_2_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_3_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_4_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_5_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_6_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_7_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
	Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_8_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
    Path("DATA/Alpha_0_25/H1_0_5_H2_0_99/Dr_0_1/Boundary_Periodic/Sim_9_Dr_0_1_H1_0_5_H2_0_99_Alpha_0_25_Boundary_Periodic.npz"),
]


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
	selected_paths = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else file_paths
	if not selected_paths:
		raise ValueError("Provide at least one .npz file path.")

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
		ytick_step=2,
	)

	axes[2].set_xlabel("Time step")

	fig.suptitle(f"Cluster summary over {len(selected_paths)} simulation(s)")
	fig.tight_layout()

	first_path = selected_paths[0]
	graph_dir = Path(__file__).resolve().parent.parent / "graphs"
	stem_parts = first_path.stem.split("_")
	if len(stem_parts) > 2 and stem_parts[0].lower() == "sim":
		name_tail = "_".join(stem_parts[2:])
	else:
		name_tail = first_path.stem
	output_path = graph_dir / f"sims_{len(selected_paths)}_{name_tail}_cluster_summary_multi.png"
	graph_dir.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.show()
	print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
	main()
