import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

from src.dataset_npz import UniversalMigrationDataset
from src.models import ParamPredictorCNN, ParamPredictorLSTM

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14, 'figure.dpi': 150, 'figure.autolayout': True
})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['cnn', 'lstm'], required=True)
    parser.add_argument('--task_name', type=str, choices=['dr_only', 'h_only', 'joint'], required=True)
    parser.add_argument('--split_name', type=str, choices=['60_30_10', '70_20_10', '80_15_5'], required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    split_map = {
        '60_30_10': (0.60, 0.30, 0.10),
        '70_20_10': (0.70, 0.20, 0.10),
        '80_15_5' : (0.80, 0.15, 0.05)
    }
    ratios = split_map[args.split_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = UniversalMigrationDataset(args.data_path, mode='test', split_ratios=ratios, task_name=args.task_name)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = 9 if args.task_name in ['dr_only', 'h_only'] else 8
    output_dim = 2 if args.task_name == 'joint' else 1

    if args.model_type == 'cnn':
        model = ParamPredictorCNN(input_dim=input_dim, output_dim=output_dim).to(device)
    else:
        model = ParamPredictorLSTM(input_dim=input_dim, output_dim=output_dim).to(device)
        
    checkpoint_path = f"saved_models/{args.split_name}/best_{args.model_type}_{args.task_name}.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint weight target '{checkpoint_path}' missing.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    test_pbar = tqdm(test_loader, desc=f"Evaluating {args.model_type.upper()}", bar_format="{l_bar}{bar:30}{r_bar}")

    with torch.no_grad():
        for x, y in test_pbar:
            pred = model(x.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    if args.task_name == 'dr_only':
        all_preds[:, 0] = np.expm1(all_preds[:, 0])
        target_vars = ['Dr']
    elif args.task_name == 'h_only':
        target_vars = ['H2']
    else:
        all_preds[:, 0] = np.expm1(all_preds[:, 0])
        target_vars = ['Dr', 'H2']

    results_dir = os.path.join("results", args.split_name)
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_log = f"Model: {args.model_type.upper()} | Task: {args.task_name.upper()} | Split: {args.split_name}\n"
    metrics_log += "="*60 + "\n"

    for i, var_name in enumerate(target_vars):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics_log += f"Variable: {var_name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}\n"

        plt.figure(figsize=(10, 5))
        sort_idx = np.argsort(y_true)
        plt.plot(y_true[sort_idx], color='royalblue', linewidth=2.5, label='Simulation Truth')
        plt.scatter(range(len(y_pred)), y_pred[sort_idx], color='crimson', marker='x', s=15, alpha=0.6, label='Predicted')
        
        plt.title(f"{args.model_type.upper()} - {args.task_name.upper()} [{args.split_name}]: {var_name}")
        plt.xlabel("Sorted Test Samples")
        plt.ylabel("Value Space")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plot_path = os.path.join(results_dir, f"plot_{args.model_type}_{args.task_name}_{var_name}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()

    metrics_txt_path = os.path.join(results_dir, f"metrics_{args.model_type}_{args.task_name}.txt")
    with open(metrics_txt_path, "w") as f:
        f.write(metrics_log)
    print(f"[COMPLETED] Results saved to {metrics_txt_path}")

if __name__ == "__main__":
    main()