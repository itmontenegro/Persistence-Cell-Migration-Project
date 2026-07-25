import os
import re
import glob
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

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
MODEL_MODE = "A"   # "A" -> predict Dr (H is fed in as the known 8th input feature)
                   # "B" -> predict H  (Dr is fed in as the known 8th input feature)

# TCN-Arch (IDENTICAL to original)
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
    def __init__(self, input_size=8): # <-- Changed to 8
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32] # 2^i
        channels = [64, 128, 128, 256, 256, 512]
        
        layers = []
        curr_in = input_size
        for d, c in zip(dilations, channels):
            layers.append(CausalResidualBlock(curr_in, c, dilation=d))
            curr_in = c
            
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(512, 1) # <-- Changed to 1

    def forward(self, x, lengths):
        x = self.encoder(x.permute(0, 2, 1))
        # Global Avg. Pooling
        pooled = torch.mean(x, dim=2) 
        return self.fc(pooled)


class TrajectoryDataset(Dataset):  
    def __init__(self, samples, input_scaler, target_scaler, file_cache):
        self.samples = samples
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.file_cache = file_cache 

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path, traj_idx, dr, h2 = self.samples[idx]
        
        if path not in self.file_cache:
            data = np.load(path)
            x = data['x_array']
            y = data['y_array']
            theta = data['theta_array']
            
            # Exact replication of df['X'].diff().fillna(0) for statistical stability
            dx = np.zeros_like(x)
            dx[:, 1:] = x[:, 1:] - x[:, :-1]
            dy = np.zeros_like(y)
            dy[:, 1:] = y[:, 1:] - y[:, :-1]
            
            step_len = np.sqrt(dx**2 + dy**2)
            cum_dist = np.cumsum(step_len, axis=1)
            net_disp = np.sqrt(x**2 + y**2)
            efficiency = net_disp / (cum_dist + 1e-6)
            
            inputs_raw = np.stack([dx, dy, step_len, efficiency], axis=-1)
            B, T, F = inputs_raw.shape
            inputs_scaled = self.input_scaler.transform(inputs_raw.reshape(-1, F)).reshape(B, T, F)
            
            net_disp_normed = net_disp / (np.max(net_disp, axis=1, keepdims=True) + 1e-6)
            
            # Preserving the original 7 features 
            all_file_features = np.stack([
                inputs_scaled[:, :, 0], # dx
                inputs_scaled[:, :, 1], # dy
                inputs_scaled[:, :, 2], # step_len
                inputs_scaled[:, :, 3], # persistance
                np.sin(theta),          # sin(theta)
                np.cos(theta),          # cos(theta)
                net_disp_normed         # net_disp_normed
            ], axis=-1)
            
            self.file_cache[path] = all_file_features

        feat = self.file_cache[path][traj_idx]
        T = feat.shape[0]
        if MODEL_MODE == "A":
            known_param = h2 
        else:  
             known_param = np.log10(dr + 1)   
        known_col = np.full((T, 1), known_param, dtype=feat.dtype)
        feat = np.concatenate([feat, known_col], axis=1)
        if MODEL_MODE == "A":
            target_raw = np.array([[np.log10(dr + 1)]])
        else:
            target_raw = np.array([[h2]])

        target_norm = self.target_scaler.transform(target_raw).flatten()
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(target_norm, dtype=torch.float32)

    
        
       
def collate_fn(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True), torch.stack(targets), lengths

class DataManager:
    def __init__(self, base_path=".", batch_size=16): 
        self.base_path = base_path
        self.batch_size = batch_size
        self.input_scaler = StandardScaler() 
        self.target_scaler = StandardScaler()
        self.files = []
        self._load_files()

    def _load_files(self):
        print("\n" + "="*50)
        print("🔍 DIAGNOSTIC: DATA LOADING")
        print("="*50)
        print(f"Target Directory: {self.base_path}")
        
        if not os.path.exists(self.base_path):
            print(" FATAL: Directory does not exist! Please check the absolute path.")
            print("="*50 + "\n")
            return
            
        all_npz = []
        # os.walk is 100% reliable for deep directory traversal
        for root, dirs, files in os.walk(self.base_path):
            for f in files:
                if f.endswith('.npz'):
                    all_npz.append(os.path.join(root, f))
                    
        print(f"Found {len(all_npz)} total .npz files.")
        
        for p in all_npz:
             
            m = re.search(r"Batch_Dr_([\d_]+)_H1_([\d_]+)_H2_([\d_]+)_Alpha_([\d_]+)", os.path.basename(p))
            if m:
                vals = [float(x.replace('_', '.')) for x in m.groups()]
                self.files.append((p, *vals))
            else:
                print(f" Regex failed on: {os.path.basename(p)}")

        print(f" Successfully loaded {len(self.files)} files into the pipeline.")
        print("="*50 + "\n")

    def get_loaders(self):
        train_samples, test_samples, val_samples = [], [], []
        all_targets = []
        shared_cache = {} 
        
        for p, dr, h1, h2, alpha in self.files:
            # 70/20/10 Row-wise split maintaining array correspondence
            for i in range(350): train_samples.append((p, i, dr, h2))
            for i in range(350, 450): test_samples.append((p, i, dr, h2))
            for i in range(450, 500): val_samples.append((p, i, dr, h2))
           
            # Scaler fitted strictly on the 350 training rows per file to prevent leakage
            data = np.load(p)
            x_train = data['x_array'][:350]
            y_train = data['y_array'][:350]
            
            dx = np.zeros_like(x_train)
            dx[:, 1:] = x_train[:, 1:] - x_train[:, :-1]
            dy = np.zeros_like(y_train)
            dy[:, 1:] = y_train[:, 1:] - y_train[:, :-1]
            
            sl = np.sqrt(dx**2 + dy**2)
            eff = np.sqrt(x_train**2 + y_train**2) / (np.cumsum(sl, axis=1) + 1e-6)
            
            inputs_raw = np.stack([dx, dy, sl, eff], axis=-1)
            self.input_scaler.partial_fit(inputs_raw.reshape(-1, 4))
            
            for _ in range(350):
                if MODEL_MODE == "A":
                    all_targets.append([np.log10(dr + 1)])
                else:
                    all_targets.append([h2])
                
        self.target_scaler.fit(all_targets)
        
        loader_args = {'batch_size': self.batch_size, 'collate_fn': collate_fn}
        return {
            'train': DataLoader(TrajectoryDataset(train_samples, self.input_scaler, self.target_scaler, shared_cache), shuffle=True, **loader_args),
            'val': DataLoader(TrajectoryDataset(val_samples, self.input_scaler, self.target_scaler, shared_cache), **loader_args),
            'test': DataLoader(TrajectoryDataset(test_samples, self.input_scaler, self.target_scaler, shared_cache), **loader_args),
            'manager': self
        }


def run_pipeline():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_gen = DataManager(base_path="/Users/karan/Downloads/CellMigration_Project/DATA_TEST", batch_size=16)
    loaders = data_gen.get_loaders()
    if MODEL_MODE == "A":
                ckpt_name = "tcn_8in_H2.pth"  
    elif MODEL_MODE == "B":
                ckpt_name = "tcn_8in_Dr.pth" 
    else:
                raise ValueError(f"Unknown MODEL_MODE: {MODEL_MODE}")
    
    # strictly 8 inputs, outputting exactly 1 predictions
    model = BaseTCNRegressor(input_size=8).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = nn.HuberLoss() 
    
    best_v, patience, counter = float('inf'), 25, 0 

    target_name = "Dr" if MODEL_MODE == "A" else "H2"
    print(f" Training-TCN (8 INPUTS | 1 OUTPUT: {target_name})")
    for epoch in range(1, 201):
        model.train()
        t_loss = 0
        total_train_samples = 0
        for x, y, lens in loaders['train']:
            x, y, lens = x.to(device), y.to(device), lens.to(device)
            optimizer.zero_grad()
            p = model(x, lens)
            loss = criterion(p, y) 
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item() * x.size(0)
            total_train_samples += x.size(0)
            
        model.eval()
        v_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for vx, vy, vlens in loaders['val']:
                vx, vy, vlens = vx.to(device), vy.to(device), vlens.to(device)
                batch_loss = criterion(model(vx, vlens), vy).item()
                v_loss += batch_loss * vx.size(0)
                total_val_samples += vx.size(0)
        
        avg_t = t_loss / total_train_samples
        avg_v = v_loss / total_val_samples
        scheduler.step(avg_v)
        
        if epoch % 5 == 0:
            print(f"Ep {epoch:03d} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f}")

        if avg_v < best_v:
            best_v, counter = avg_v, 0
            torch.save(model.state_dict(), ckpt_name)
        else:
            counter += 1
            if counter >= patience: break

    model.load_state_dict(torch.load(ckpt_name))
    model.eval()
    all_p_norm, all_y_norm = [], []
    with torch.no_grad():
        for tx, ty, tlen in loaders['test']:
            all_p_norm.append(model(tx.to(device), tlen.to(device)).cpu().numpy())
            all_y_norm.append(ty.numpy())
    
    p_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_p_norm))
    y_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_y_norm))

    if MODEL_MODE == "A":
        # Target was log10(Dr + 1) -> invert
        p_final[:, 0] = 10**(p_final[:, 0]) - 1
        y_final[:, 0] = 10**(y_final[:, 0]) - 1
        param_names = ['Dr']
    else:
        # Target was raw H -> no inverse log transform
        param_names = ['H2']

    print("\n" + "="*30)
    print(f"BASE TCN PERFORMANCE - 8 IN -> 1 OUT ({param_names[0]})")
    print("="*30)
    for i, n in enumerate(param_names):
        print(f"{n} MAE: {mean_absolute_error(y_final[:, i], p_final[:, i]):.6f}")
        print(f"{n} R² Score: {r2_score(y_final[:, i], p_final[:, i]):.4f}")

if __name__ == "__main__":
    run_pipeline()