import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  

from src.dataset_2 import UniversalMigrationDataset
from src.models_2 import get_model

def multitask_loss(pred, target):
    mse = nn.MSELoss()
    dr_loss = mse(pred[:, 0], target[:, 0])
    h_loss = mse(pred[:, 1], target[:, 1])

    return dr_loss + 8.0 * h_loss

def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Training on device: {device}")
    train_ds = UniversalMigrationDataset(args.data_path, mode='train')
    val_ds = UniversalMigrationDataset(args.data_path, mode='val')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = get_model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best = 1e18

    for epoch in range(args.epochs):
        model.train()
        total = 0

        train_pbar = tqdm(train_loader, desc="Training  ", leave=False)

        for x, y in train_pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = multitask_loss(pred, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_loss = total / len(train_loader)
        model.eval()
        vtotal = 0
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for x, y in val_pbar:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = multitask_loss(pred, y)
                vtotal += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        val_loss = vtotal / len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best:
            best = val_loss
            os.makedirs("saved_models", exist_ok=True)
            save_path = "saved_models/best_hybrid_Dr_H_2.pth"
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    train(args)