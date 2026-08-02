# predict_experimental.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models import get_model
from src.experimental_dataset import ExperimentalMigrationDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Experimental Cell Migration Model Inference")
    parser.add_argument('--csv_path', type=str, default='data/test.csv', help='Path to experimental tracking CSV')
    parser.add_argument('--model_path', type=str, default='saved_models/split_70_20_10/joint.pth', help='Path to saved .pth checkpoint')
    parser.add_argument('--out_dir', type=str, default='results_experimental', help='Output directory')
    parser.add_argument('--time_interval', type=float, default=15.0, help='Microscope image capture interval in minutes')
    parser.add_argument('--alpha_val', type=float, default=1.0, help='Alpha parameter encoding')
    return parser.parse_args()

def run_inference():
    args = parse_args()
    
    csv_path = Path(args.csv_path)
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Experimental dataset not found at: {csv_path.resolve()}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found at: {model_path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"RUNNING EXPERIMENTAL INFERENCE | Device: {device}")
    print(f"Dataset Path: {csv_path.resolve()}")
    print(f"Model Path:   {model_path.resolve()}")
    print("=" * 80)

    # Load Dataset
    dataset = ExperimentalMigrationDataset(str(csv_path), alpha_val=args.alpha_val)
    inputs, cell_ids = dataset.inputs, dataset.cell_ids
    x_tensor = torch.tensor(inputs).to(device)

    # Initialize Model & Load Weights
    model = get_model(target_mode='joint').to(device)
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Model Evaluation
    with torch.no_grad():
        raw_preds = model(x_tensor).cpu().numpy()

    # Inverse Target Transformations
    # Dr target inverse: log1p scaling -> expm1
    dr_preds = np.expm1(raw_preds[:, 0])
    dr_preds = np.maximum(dr_preds, 0.0)  # Physical non-negativity constraint
    
    # H2 target: bounded sigmoid output in range [0.5, 1.0]
    h_preds = raw_preds[:, 1]

    # Calculate Derived Biological / Biophysical Metrics
    results = []
    for cid, dr, h in zip(cell_ids, dr_preds, h_preds):
        tau_p_steps = 1.0 / dr if dr > 1e-6 else np.inf
        tau_p_min = tau_p_steps * args.time_interval if not np.isinf(tau_p_steps) else np.inf
        anomalous_alpha = 2.0 * h
        
        # Categorize Cell Migration Dynamic Mode
        if h > 0.70:
            mode = "Highly Persistent / Super-diffusive"
        elif h > 0.55:
            mode = "Moderately Persistent Walk"
        else:
            mode = "Constrained / Normal Random Walk"

        results.append({
            'Cell_No': cid,
            'Predicted_Dr': float(dr),
            'Predicted_H': float(h),
            'Anomalous_Exponent_2H': float(anomalous_alpha),
            'Persistence_Steps_Tau': float(tau_p_steps) if not np.isinf(tau_p_steps) else "Infinite",
            'Persistence_Time_Mins': float(tau_p_min) if not np.isinf(tau_p_min) else "Infinite",
            'Biological_Migration_Mode': mode
        })

    results_df = pd.DataFrame(results)
    csv_out_path = out_dir / "predictions_summary.csv"
    results_df.to_csv(csv_out_path, index=False)

    print("\n=== EXPERIMENTAL INFERENCE RESULTS SUMMARY ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV report to: {csv_out_path.resolve()}")

    # Visual Plot Generation
    plot_results(csv_path, results_df, out_dir / "trajectory_predictions_plot.png")

def plot_results(csv_path: Path, results_df: pd.DataFrame, save_path: Path):
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Trajectory paths
    for cell_id, group in df.groupby('Cell_No'):
        x = group['X'].values - group['X'].values[0]
        y = group['Y'].values - group['Y'].values[0]
        axes[0].plot(x, y, linewidth=2, marker='o', markersize=3, label=f'Cell {cell_id}')

    axes[0].set_title("Experimental Cell Trajectories (Zero-Centered Origin)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("X Displacement (pixels / µm)", fontsize=10)
    axes[0].set_ylabel("Y Displacement (pixels / µm)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best')

    # Subplot 2: Joint Predicted Parameters (H & Dr)
    cells = results_df['Cell_No'].values
    h_vals = results_df['Predicted_H'].values
    dr_vals = results_df['Predicted_Dr'].values
    x_indices = np.arange(len(cells))

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    p1 = ax2.bar(x_indices - 0.15, h_vals, width=0.3, color='royalblue', label='Hurst Exponent (H)')
    p2 = ax2_twin.bar(x_indices + 0.15, dr_vals, width=0.3, color='crimson', label='Rotational Diffusion (Dr)')

    ax2.set_xlabel("Cell ID", fontsize=10)
    ax2.set_ylabel("Hurst Exponent (H)", color='royalblue', fontsize=10)
    ax2_twin.set_ylabel("Rotational Diffusion (Dr)", color='crimson', fontsize=10)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"Cell {c}" for c in cells])
    ax2.set_title("Joint Model Predictions per Cell", fontsize=12, fontweight='bold')
    ax2.set_ylim(0.4, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved visualization summary plot to: {save_path.resolve()}")

if __name__ == "__main__":
    run_inference()