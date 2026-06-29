import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# --- TCN Architecture (Identical to training) ---
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


# --- Data Management (Identical to training to perfectly match scalers) ---
class TrajectoryDataset(Dataset):  
    def __init__(self, samples, input_scaler, target_scaler, file_cache):
        self.samples = samples
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.file_cache = file_cache 

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, traj_idx, dr, h2 = self.samples[idx]
        
        if path not in self.file_cache:
            data = np.load(path)
            x, y, theta = data['x_array'], data['y_array'], data['theta_array']
            
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
            
            all_file_features = np.stack([
                inputs_scaled[:, :, 0], inputs_scaled[:, :, 1], inputs_scaled[:, :, 2], 
                inputs_scaled[:, :, 3], np.sin(theta), np.cos(theta), net_disp_normed
            ], axis=-1)
            self.file_cache[path] = all_file_features

        feat = self.file_cache[path][traj_idx]
        target_raw = np.array([[np.log10(dr + 1), h2]])
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
        print(f"Loading data from {self.base_path} to recreate scalers...")
        all_npz = []
        for root, dirs, files in os.walk(self.base_path):
            for f in files:
                if f.endswith('.npz'):
                    all_npz.append(os.path.join(root, f))
        for p in all_npz:
            m = re.search(r"Batch_Dr_([\d_]+)_H1_([\d_]+)_H2_([\d_]+)_Alpha_([\d_]+)", os.path.basename(p))
            if m:
                vals = [float(x.replace('_', '.')) for x in m.groups()]
                self.files.append((p, *vals))

    def get_loaders(self):
        train_samples, test_samples = [], []
        all_targets = []
        shared_cache = {} 
        
        for p, dr, h1, h2, alpha in self.files:
            for i in range(350): train_samples.append((p, i, dr, h2))
            for i in range(350, 450): test_samples.append((p, i, dr, h2))
            
            data = np.load(p)
            x_train, y_train = data['x_array'][:350], data['y_array'][:350]
            
            dx = np.zeros_like(x_train)
            dx[:, 1:] = x_train[:, 1:] - x_train[:, :-1]
            dy = np.zeros_like(y_train)
            dy[:, 1:] = y_train[:, 1:] - y_train[:, :-1]
            
            sl = np.sqrt(dx**2 + dy**2)
            eff = np.sqrt(x_train**2 + y_train**2) / (np.cumsum(sl, axis=1) + 1e-6)
            
            inputs_raw = np.stack([dx, dy, sl, eff], axis=-1)
            self.input_scaler.partial_fit(inputs_raw.reshape(-1, 4))
            for _ in range(350): all_targets.append([np.log10(dr + 1), h2])
                
        self.target_scaler.fit(all_targets)
        loader_args = {'batch_size': self.batch_size, 'collate_fn': collate_fn}
        
        # We only need the test loader for evaluation
        return {
            'test': DataLoader(TrajectoryDataset(test_samples, self.input_scaler, self.target_scaler, shared_cache), **loader_args),
            'manager': self
        }

# --- Evaluation & Plotting Logic ---
def evaluate_and_plot():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # 1. Initialize DataManager to perfectly recreate test dataset & scalers
    data_gen = DataManager(base_path="/Users/karan/Downloads/CellMigration_Project/DATA_TEST", batch_size=16)
    loaders = data_gen.get_loaders()
    
    # 2. Load the Model
    model = BaseTCNRegressor(input_size=7).to(device)
    
    weights_path = "base_tcn_7in.pth"
    if not os.path.exists(weights_path):
        print(f" Could not find {weights_path}! Make sure the script is in the same folder as your saved model.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(" Model weights loaded successfully! Running inference on Test Set...")

    # 3. Collect Predictions
    all_p_norm, all_y_norm = [], []
    with torch.no_grad():
        for tx, ty, tlen in loaders['test']:
            all_p_norm.append(model(tx.to(device), tlen.to(device)).cpu().numpy())
            all_y_norm.append(ty.numpy())
    
    # 4. Inverse Scale & Fix Dr
    p_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_p_norm))
    y_final = data_gen.target_scaler.inverse_transform(np.concatenate(all_y_norm))

    p_final[:, 0] = 10**(p_final[:, 0]) - 1 
    y_final[:, 0] = 10**(y_final[:, 0]) - 1

    # 5. Generate Graphs
    param_names = ['Dr', 'H2']
    for i, n in enumerate(param_names):
        y_true = y_final[:, i]
        y_pred = p_final[:, i]
        
        # Calculate Metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n--- {n} Metrics ---")
        print(f"MAE: {mae:.6f} | MSE: {mse:.6f} | R²: {r2:.4f}")

        # Sort based on Ground Truth values to create the step-line
        sort_indices = np.argsort(y_true)
        y_true_sorted = y_true[sort_indices]
        y_pred_sorted = y_pred[sort_indices]
        x_axis = np.arange(len(y_true_sorted))

        # Setup the Matplotlib figure
        plt.figure(figsize=(12, 7), dpi=300)
        
        # Ground Truth as a thick, continuous blue step-line
        plt.plot(x_axis, y_true_sorted, label='Ground Truth (Simulation)', 
                 color='#348ABD', linewidth=3.5, alpha=0.9, zorder=1)
        
        # Predictions as red scatter dots with black borders
        plt.scatter(x_axis, y_pred_sorted, label='Base TCN Prediction', 
                    color='#E24A33', edgecolors='black', s=25, zorder=2)

        # Aesthetics
        plt.title(f'Base TCN Performance: {n} Recovery', fontsize=18, fontweight='bold', pad=10)
        plt.xlabel('Test Samples (Sorted)', fontsize=12)
        plt.ylabel('Physical Units', fontsize=12)
        
        # Text Box
        text_str = f"R² Score: {r2:.5f}\nMAE: {mae:.6f}\nMSE: {mse:.6f}"
        props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=1.0)
        plt.gca().text(0.04, 0.95, text_str, transform=plt.gca().transAxes, fontsize=14,
                       verticalalignment='top', bbox=props)

        # Grid and Legend
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right', fontsize=12)
        plt.tight_layout()
        
        # Save output
        filename = f"Evaluation_{n}_Recovery.png"
        plt.savefig(filename)
        print(f"📉 Saved graph locally as: {filename}")
        plt.close()

if __name__ == "__main__":
    evaluate_and_plot()
