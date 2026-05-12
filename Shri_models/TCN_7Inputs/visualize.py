import os
import re
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#1)Mimicing TCN structure
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, dropout=0.2):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.ln = nn.GroupNorm(1, out_channels) 
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        res = x
        out = self.conv(x)
        out = out[:, :, :-self.padding] 
        out = self.ln(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.downsample is not None: res = self.downsample(x)
        return out + res

class BaseTCNRegressor(nn.Module):
    def __init__(self, input_size=7):
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32]
        channels = [64, 128, 128, 256, 256, 512]
        layers = []
        curr_in = input_size
        for d, c in zip(dilations, channels):
            layers.append(CausalResidualBlock(curr_in, c, dilation=d))
            curr_in = c
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(512, 2) 

    def forward(self, x, lengths):
        x = self.encoder(x.permute(0, 2, 1))

        pooled = torch.mean(x, dim=2) 
        return self.fc(pooled)

#data management and visualization
class DataManager:
    def __init__(self, base_path=".", batch_size=16):
        self.base_path, self.batch_size = base_path, batch_size
        self.input_scaler, self.target_scaler = StandardScaler(), StandardScaler()
        self.files = []
        self._load_files()

    def _load_files(self):
        paths = glob.glob(os.path.join(self.base_path, "**", "Sim_*.csv"), recursive=True)
        for p in paths:
            m = re.search(r"Alpha_([\d_]+)__Dr_([\d_]+)__H_([\d_]+)", p)
            if m: self.files.append((p, *[float(x.replace('_', '.')) for x in m.groups()]))

    def get_test_loader(self):
        
        strats = [f"{f[2]}_{f[3]}" for f in self.files]
        tv, test_info = train_test_split(self.files, test_size=0.15, stratify=strats, random_state=42)
        train_info, _ = train_test_split(tv, test_size=0.15, stratify=[f"{f[2]}_{f[3]}" for f in tv], random_state=42)
        
        all_targets = []
        for p, _, d, h in train_info:
            df = pd.read_csv(p).head(1001)
            dx, dy = df['X'].diff().fillna(0), df['Y'].diff().fillna(0)
            sl = np.sqrt(dx**2 + dy**2)
            eff = np.sqrt(df['X']**2 + df['Y']**2) / (np.cumsum(sl) + 1e-6)
            self.input_scaler.partial_fit(np.stack([dx, dy, sl, eff], axis=1))
            all_targets.append([np.log10(d + 1), h]) 
        self.target_scaler.fit(all_targets)

        class VisualizationDS(Dataset):
            def __init__(self, info, in_s, tg_s): self.info, self.in_s, self.tg_s = info, in_s, tg_s
            def __len__(self): return len(self.info)
            def __getitem__(self, idx):
                p, _, d, h = self.info[idx]
                df = pd.read_csv(p).fillna(0).head(1001)
                x, y, t = df['X'].values, df['Y'].values, df['Theta'].values
                dx, dy = df['X'].diff().fillna(0).values, df['Y'].diff().fillna(0).values
                sl = np.sqrt(dx**2 + dy**2)
                eff = np.sqrt(x**2 + y**2) / (np.cumsum(sl) + 1e-6)
                
                coords_scaled = self.in_s.transform(np.stack([dx, dy, sl, eff], axis=1))
                feat = np.stack([
                    coords_scaled[:,0], coords_scaled[:,1], coords_scaled[:,2], coords_scaled[:,3],
                    np.sin(t), np.cos(t), np.sqrt(x**2 + y**2) / (np.max(np.sqrt(x**2+y**2)) + 1e-6)
                ], axis=1)
                
                target = self.tg_s.transform(np.array([[np.log10(d + 1), h]])).flatten()
                return torch.tensor(feat, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

        def collate(batch):
            s, t = zip(*batch)
            return pad_sequence(s, batch_first=True), torch.stack(t), torch.tensor([len(x) for x in s])

        return DataLoader(VisualizationDS(test_info, self.input_scaler, self.target_scaler), 
                          batch_size=self.batch_size, collate_fn=collate)



def generate_performance_plots(num_display_samples=150):#no. of samples
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")#for NVIDIA graphics cards
    else:
        device = torch.device("cpu")
    
    print(f"Generating plots using device: {device}")
    
    manager = DataManager()
    test_loader = manager.get_test_loader()
    
    
    model = BaseTCNRegressor(input_size=7).to(device)
    model.load_state_dict(torch.load("base_tcn_7in.pth", map_location=device))
    model.eval()

    all_p, all_y = [], []
    with torch.no_grad():
        for x, y, lens in test_loader:
            all_p.append(model(x.to(device), lens.to(device)).cpu().numpy())
            all_y.append(y.numpy())
    
    p_phys = manager.target_scaler.inverse_transform(np.concatenate(all_p))
    y_phys = manager.target_scaler.inverse_transform(np.concatenate(all_y))
    
    # Reverse log scaling for Dr 
    p_phys[:, 0] = 10**(p_phys[:, 0]) - 1 
    y_phys[:, 0] = 10**(y_phys[:, 0]) - 1

    names = ['Dr', 'H']
    
    for i in range(2):
        sort_idx = np.argsort(y_phys[:, i])
        y_sorted, p_sorted = y_phys[sort_idx, i], p_phys[sort_idx, i]
        
        step = max(1, len(y_sorted) // num_display_samples)
        y_plot, p_plot = y_sorted[::step][:num_display_samples], p_sorted[::step][:num_display_samples]

        mae = mean_absolute_error(y_sorted, p_sorted)
        mse = mean_squared_error(y_sorted, p_sorted)
        r2 = r2_score(y_sorted, p_sorted)

        plt.figure(figsize=(12, 7))
        plt.plot(y_plot, label='Ground Truth (Simulation)', color='#1f77b4', linewidth=3, alpha=0.8)
        plt.scatter(range(len(p_plot)), p_plot, label='Base TCN Prediction', color='#d62728', s=25, edgecolors='black', zorder=5)
        
        text_str = f"R² Score: {r2:.5f}\nMAE: {mae:.6f}\nMSE: {mse:.6f}"
        plt.gca().text(0.05, 0.92, text_str, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.title(f"Base TCN Performance: {names[i]} Recovery", fontsize=16, fontweight='bold')
        plt.xlabel("Test Samples(Sorted)", fontsize=12)
        plt.ylabel("Physical Units", fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        
        if names[i] == 'Dr': plt.yscale('linear') #Change to 'log'for better visuals)
        
        plt.tight_layout()
        plt.savefig(f"vis_{names[i].lower()}_base.png", dpi=300)
        print(f" Plot saved for {names[i]}: R2={r2:.5f}")
        plt.close()

if __name__ == "__main__":
    generate_performance_plots()