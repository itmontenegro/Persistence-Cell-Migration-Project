import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_npz import UniversalMigrationDataset 
from src.models import get_model

DATA_PATH = "/media/nacho/Station_s Vault/DATA/2026/Persistence-Cell-Migration-Project/DATA/"
SPLIT_TYPES = ['80_15_5']
TARGET_MODES = ['joint', 'input_H2_predict_Dr', 'input_Dr_predict_H2']

def train_one_setting(split_type, target_mode, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*80)
    print(f"STARTING CONFIGURATION: Split [{split_type}] | Target Mode [{target_mode}]")
    print("="*80)

    # Instantiate datasets
    train_ds = UniversalMigrationDataset(args.data_path, mode='train', split_type=split_type, target_mode=target_mode)
    val_ds   = UniversalMigrationDataset(args.data_path, mode='val', split_type=split_type, target_mode=target_mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Initialize model, optimizer, and training metrics
    model = get_model(target_mode=target_mode).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.HuberLoss()

    best_loss = float('inf')
    model_dir = os.path.join("saved_models", f"split_{split_type}")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{target_mode}.pth")

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Setup real-time progress bar tracking for batches within the current epoch
        pbar_desc = f"Epoch {epoch+1:03d}/{args.epochs:03d} [Train]"
        train_pbar = tqdm(train_loader, desc=pbar_desc, leave=False)
        
        for x, y in train_pbar:
            x, y = x.to(device), y.to(device)
            y_scaled = y.clone()

            # Target scaling transformations
            if target_mode in ['joint', 'input_H2_predict_Dr']:
                y_scaled[:, 0] = torch.log1p(y_scaled[:, 0])

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_scaled)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss = loss.item()
            train_loss += running_loss
            train_pbar.set_postfix({'batch_loss': f"{running_loss:.4f}"})

        # Validation Run
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs:03d} [Val]  ", leave=False)
        
        with torch.no_grad():
            for x, y in val_pbar:
                x, y = x.to(device), y.to(device)
                y_scaled = y.clone()
                if target_mode in ['joint', 'input_H2_predict_Dr']:
                    y_scaled[:, 0] = torch.log1p(y_scaled[:, 0])
                
                pred = model(x)
                v_loss = criterion(pred, y_scaled).item()
                val_loss += v_loss
                val_pbar.set_postfix({'batch_loss': f"{v_loss:.4f}"})

        epoch_train_avg = train_loss / len(train_loader)
        epoch_val_avg = val_loss / len(val_loader)
        scheduler.step(epoch_val_avg)

        # Print detailed progress output to console for the completed epoch
        checkpoint_status = ""
        if epoch_val_avg < best_loss:
            best_loss = epoch_val_avg
            torch.save(model.state_dict(), save_path)
            checkpoint_status = " Best Model Saved!"
            
        print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | Avg Train Loss: {epoch_train_avg:.5f} | Avg Val Loss: {epoch_val_avg:.5f}{checkpoint_status}")
            
    print(f"\n COMPLETED CONFIGURATION: Saved optimal checkpoint -> {save_path} [Best Loss: {best_loss:.6f}]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    # Execute training loops over the grid space
    for split in SPLIT_TYPES:
        for mode in TARGET_MODES:
            train_one_setting(split, mode, args)