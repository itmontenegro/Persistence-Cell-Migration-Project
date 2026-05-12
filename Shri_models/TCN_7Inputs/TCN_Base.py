import os
import re
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


def seed_everything(seed=42):#imp. 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

# TCN-Arch.
class CausalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=self.padding)
        self.ln = nn.GroupNorm(1, out_channels) 
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out[:, :, :-self.padding] 
        out = self.ln(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        return out + residual 

class BaseTCNRegressor(nn.Module):
    def __init__(self, input_size=7): 
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32]#2^i
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
        
        #Global Avg. Pooling
        pooled = torch.mean(x, dim=2) 
        
        return self.fc(pooled)


class TrajectoryDataset(Dataset):  
    def __init__(self, file_info, input_scaler, target_scaler):
        self.file_info = file_info
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    def __len__(self): return len(self.file_info)

    def __getitem__(self, idx):
        path, _, d, h = self.file_info[idx]
        df = pd.read_csv(path).fillna(0).head(1001) #1001 steps
        
        x, y = df['X'].values, df['Y'].values
        dx = df['X'].diff().fillna(0).values
        dy = df['Y'].diff().fillna(0).values
        
        step_len = np.sqrt(dx**2 + dy**2)
        cum_dist = np.cumsum(step_len)
        net_disp = np.sqrt(x**2 + y**2)
        efficiency = net_disp / (cum_dist + 1e-6)
        
        inputs_raw = np.stack([dx, dy, step_len, efficiency], axis=1)
        inputs_scaled = self.input_scaler.transform(inputs_raw)
        
        t = df['Theta'].values
        #  7 inputs
        feat = np.stack([
            inputs_scaled[:, 0], # dx
            inputs_scaled[:, 1], # dy
            inputs_scaled[:, 2], # step_len
            inputs_scaled[:, 3], # efficiency
            np.sin(t),           # sin(theta)
            np.cos(t),           # cos(theta)
            net_disp / (np.max(net_disp) + 1e-6) # net_disp_normed
        ], axis=1)
        
        target_norm = self.target_scaler.transform(np.array([[np.log10(d + 1), h]])).flatten()
        
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target_norm, dtype=torch.float32)

def collate_fn(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), torch.stack(targets), lengths

class DataManager:
    def __init__(self, base_path=".", batch_size=16): #16 files at a time 
        self.base_path, self.batch_size = base_path, batch_size
        self.input_scaler = StandardScaler() 
        self.target_scaler = StandardScaler()
        self.files = []
        self._load_files()

    def _load_files(self):#amazing 
        paths = glob.glob(os.path.join(self.base_path, "**", "Sim_*.csv"), recursive=True)
        for p in paths:
            m = re.search(r"Alpha_([\d_]+)__Dr_([\d_]+)__H_([\d_]+)", p)
            if m:
                vals = [float(x.replace('_', '.')) for x in m.groups()]
                self.files.append((p, *vals))

    def get_loaders(self):
        strats = [f"{f[2]}_{f[3]}" for f in self.files]
        tv, test_info = train_test_split(self.files, test_size=0.15, stratify=strats, random_state=42)#Tst-15%, stratify:-equal combo. for all
        train_info, val_info = train_test_split(tv, test_size=0.15, stratify=[f"{f[2]}_{f[3]}" for f in tv], random_state=42)#Val-12.75%
        
        all_targets = []
        for p, _, d, h in train_info:#leakage preventer
            df = pd.read_csv(p).head(1001)
            dx, dy = df['X'].diff().fillna(0), df['Y'].diff().fillna(0)
            sl = np.sqrt(dx**2 + dy**2)
            eff = np.sqrt(df['X']**2 + df['Y']**2) / (np.cumsum(sl) + 1e-6)
            self.input_scaler.partial_fit(np.stack([dx, dy, sl, eff], axis=1))
            all_targets.append([np.log10(d + 1), h])
        
        self.target_scaler.fit(all_targets)
        
        loader_args = {'batch_size': self.batch_size, 'collate_fn': collate_fn}
        return {
            'train': DataLoader(TrajectoryDataset(train_info, self.input_scaler, self.target_scaler), shuffle=True, **loader_args),
            'val': DataLoader(TrajectoryDataset(val_info, self.input_scaler, self.target_scaler), **loader_args),
            'test': DataLoader(TrajectoryDataset(test_info, self.input_scaler, self.target_scaler), **loader_args),
            'manager': self
        }


def run_pipeline():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")#for NVIDIA graphics cards :-device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_gen = DataManager(batch_size=16)
    loaders = data_gen.get_loaders()
    
    model = BaseTCNRegressor(input_size=7).to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = nn.HuberLoss() 
    
    best_v, patience, counter = float('inf'), 25, 0 #early stopping based on val loss

    print(" Training-TCN (7INPUTS)")
    for epoch in range(1, 201):
        model.train()
        t_loss = 0
        for x, y, lens in loaders['train']:
            x, y, lens = x.to(device), y.to(device), lens.to(device)
            optimizer.zero_grad()
            p = model(x, lens)
            loss = criterion(p, y) 
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vy, vlens in loaders['val']:
                vx, vy, vlens = vx.to(device), vy.to(device), vlens.to(device)
                v_loss += criterion(model(vx, vlens), vy).item() * vx.size(0)
        
        avg_t, avg_v = t_loss/len(loaders['train'].dataset), v_loss/len(loaders['val'].dataset)
        scheduler.step(avg_v)
        
        if epoch % 5 == 0:
            print(f"Ep {epoch:03d} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

        if avg_v < best_v:
            best_v, counter = avg_v, 0
            torch.save(model.state_dict(), "base_tcn_7in.pth")
        else:
            counter += 1
            if counter >= patience: break

    
    model.load_state_dict(torch.load("base_tcn_7in.pth"))
    model.eval()
    all_p_norm, all_y_norm = [], []
    with torch.no_grad():
        for tx, ty, tlen in loaders['test']:
            all_p_norm.append(model(tx.to(device), tlen.to(device)).cpu().numpy())
            all_y_norm.append(ty.numpy())
    
    p_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_p_norm))
    y_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_y_norm))

    
    p_final[:, 0] = 10**(p_final[:, 0]) - 1 
    y_final[:, 0] = 10**(y_final[:, 0]) - 1

    print("\n" + "="*30)
    print("BASE TCN PERFORMANCE-7 INP.")
    print("="*30)
    for i, n in enumerate(['Dr', 'H']):
        print(f"{n} MAE: {mean_absolute_error(y_final[:, i], p_final[:, i]):.6f}")
        print(f"{n} R² Score: {r2_score(y_final[:, i], p_final[:, i]):.4f}")

if __name__ == "__main__":
    run_pipeline()