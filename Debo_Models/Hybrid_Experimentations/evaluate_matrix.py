import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dataset_npz import UniversalMigrationDataset
from src.models import get_model

DATA_PATH = "/media/nacho/Station_s Vault/DATA/2026/Persistence-Cell-Migration-Project/DATA/"
#SPLIT_TYPES = ['60_30_10', '70_20_10', '80_15_5']
SPLIT_TYPES = ['60_30_10', '70_20_10']
#TARGET_MODES = ['joint', 'input_H2_predict_Dr', 'input_Dr_predict_H2']
TARGET_MODES = ['joint']

def evaluate_matrix(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_metrics = []

    for split in SPLIT_TYPES:
        for mode in TARGET_MODES:
            checkpoint_path = os.path.join("saved_models", f"split_{split}", f"{mode}.pth")
            if not os.path.exists(checkpoint_path):
                print(f"Skipping unbuilt checkpoint context: {checkpoint_path}")
                continue
                
            print(f"Processing Inference: Split={split} | Target Mode={mode}")
            
            test_ds = UniversalMigrationDataset(args.data_path, mode='test', split_type=split, target_mode=mode)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            
            model = get_model(target_mode=mode).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    preds = model(inputs.to(device))
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.numpy())
                    
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            # Core Fix: Safe variable target extraction prevents index crashes across configurations
            if mode == 'joint':
                all_preds[:, 0] = np.expm1(all_preds[:, 0])
                vars_to_eval = [('Dr', 0), ('H2', 1)]
            elif mode == 'input_H2_predict_Dr':
                all_preds[:, 0] = np.expm1(all_preds[:, 0])
                vars_to_eval = [('Dr', 0)]
            elif mode == 'input_Dr_predict_H2':
                vars_to_eval = [('H2', 0)]

            out_dir = os.path.join("results_matrix", f"split_{split}")
            os.makedirs(out_dir, exist_ok=True)
            
            for var_name, idx in vars_to_eval:
                y_true = all_targets[:, idx]
                y_pred = all_preds[:, idx]
                
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                summary_metrics.append({
                    'split': split, 'mode': mode, 'variable': var_name,
                    'rmse': rmse, 'mae': mae, 'r2': r2
                })
                
                plt.figure(figsize=(10, 5))
                sort_idx = np.argsort(y_true)
                plt.plot(y_true[sort_idx], color='royalblue', linewidth=2.5, label='True Ground Truth')
                plt.scatter(range(len(y_pred)), y_pred[sort_idx], color='crimson', marker='x', s=12, alpha=0.6, label='Prediction Match')
                plt.title(f"Model Matrix Evaluation ({split}): {mode} - {var_name}")
                plt.legend(loc='best')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"chart_{mode}_{var_name}.png"), dpi=300)
                plt.close()

    report_path = "results_matrix/comprehensive_report.txt"
    os.makedirs("results_matrix", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("=== STRUCTURAL MATRIX SYSTEM SCORING LEDGER ===\n\n")
        for m in summary_metrics:
            log_line = f"Split: {m['split']:<10} | Mode: {m['mode']:<22} | Var: {m['variable']:<3} -> R2: {m['r2']:6.4f} | RMSE: {m['rmse']:6.4f} | MAE: {m['mae']:6.4f}\n"
            print(log_line.strip())
            f.write(log_line)
            
    print(f"\nFinalized report exported successfully to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    evaluate_matrix(args)