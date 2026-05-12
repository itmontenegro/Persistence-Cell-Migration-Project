import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dataset import UniversalMigrationDataset
from src.models import get_model

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'figure.figsize': (12, 6)
})

def evaluate_and_plot(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    test_ds = UniversalMigrationDataset(args.data_path, mode='test', target_vars=args.target_vars)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    var_str = "_".join(args.target_vars)

    model = get_model().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # --- THE CRITICAL FIX ---
    # Un-scale BOTH predictions and targets back to their physical values (0 to 100)
    all_preds[:, 0] = np.expm1(all_preds[:, 0]) 
    all_targets[:, 0] = np.expm1(all_targets[:, 0]) 
    # ------------------------

    os.makedirs('results', exist_ok=True)
    
    metrics_log = f"Results for Model: HYBRID | Targets: {args.target_vars}\n"
    metrics_log += "="*60 + "\n"

    for i, var_name in enumerate(args.target_vars):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        log = f"Variable: {var_name}\n  RMSE : {rmse:.4f}\n  MAE  : {mae:.4f}\n  R2   : {r2:.4f}\n"
        metrics_log += log

        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        plt.figure(figsize=(12, 6))
        
        plt.plot(range(len(y_true_sorted)), y_true_sorted, 
                 color='royalblue', linewidth=3, label='Original (True) Value', zorder=1)
        
        plt.scatter(range(len(y_pred_sorted)), y_pred_sorted, 
                    color='crimson', marker='x', s=50, linewidths=2.0, 
                    label='Model Prediction', alpha=0.8, zorder=2)
        
        plt.title(f"HYBRID: Original vs. Predicted for {var_name}")
        plt.xlabel("Test Simulation Files (Sorted by True Value)")
        plt.ylabel(f"{var_name} Parameter Value")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plot_path = f"results/ladder_phase2_hybrid_{var_name}.png"
        plt.savefig(plot_path)
        plt.close()

    txt_path = f"results/metrics_phase2_hybrid_{var_str}.txt"
    with open(txt_path, "w") as f:
        f.write(metrics_log)
    
    print(metrics_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--target_vars', nargs='+', default=['Dr', 'H'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    evaluate_and_plot(args)