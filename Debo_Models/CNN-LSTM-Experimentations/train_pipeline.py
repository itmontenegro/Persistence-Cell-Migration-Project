import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_npz import UniversalMigrationDataset
from src.models import ParamPredictorCNN, ParamPredictorLSTM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['cnn', 'lstm'], required=True)
    parser.add_argument('--task_name', type=str, choices=['dr_only', 'h_only', 'joint'], required=True)
    parser.add_argument('--split_name', type=str, choices=['60_30_10', '70_20_10', '80_15_5'], required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    split_map = {
        '60_30_10': (0.60, 0.30, 0.10),
        '70_20_10': (0.70, 0.20, 0.10),
        '80_15_5' : (0.80, 0.15, 0.05)
    }
    ratios = split_map[args.split_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*70)
    print(f" LAUNCHING TRAINING PIPELINE")
    print(f" -> Model: {args.model_type.upper()} | Task: {args.task_name.upper()} | Split: {args.split_name}")
    print(f" -> Device: {device}")
    print("="*70 + "\n")

    train_ds = UniversalMigrationDataset(args.data_path, mode='train', split_ratios=ratios, task_name=args.task_name)
    val_ds = UniversalMigrationDataset(args.data_path, mode='val', split_ratios=ratios, task_name=args.task_name)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = 9 if args.task_name in ['dr_only', 'h_only'] else 8
    output_dim = 2 if args.task_name == 'joint' else 1

    if args.model_type == 'cnn':
        model = ParamPredictorCNN(input_dim=input_dim, output_dim=output_dim).to(device)
    else:
        model = ParamPredictorLSTM(input_dim=input_dim, output_dim=output_dim).to(device)
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_loss = float('inf')
    saved_model_dir = os.path.join("saved_models", args.split_name)
    os.makedirs(saved_model_dir, exist_ok=True)
    checkpoint_path = os.path.join(saved_model_dir, f"best_{args.model_type}_{args.task_name}.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1:02d}/{args.epochs} [{args.model_type.upper()}-TRAIN]", 
            bar_format="{l_bar}{bar:30}{r_bar}"
        )
        
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            if args.task_name in ['dr_only', 'joint']:
                y[:, 0] = torch.log1p(y[:, 0])

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss = loss.item()
            train_loss += running_loss
            train_pbar.set_postfix({"loss": f"{running_loss:.4f}"})

        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [VAL]", bar_format="{l_bar}{bar:30}{r_bar}", leave=False)
        
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                if args.task_name in ['dr_only', 'joint']:
                    y[:, 0] = torch.log1p(y[:, 0])
                
                pred = model(x)
                val_loss += criterion(pred, y).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f" -> Train HuberLoss: {avg_train:.5f} | Val HuberLoss: {avg_val:.5f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    * Saved best model checkpoint to: {checkpoint_path}")
        print("-" * 75)

if __name__ == "__main__":
    main()