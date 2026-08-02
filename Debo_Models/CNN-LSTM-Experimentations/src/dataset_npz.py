import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

_RE_ALPHA = re.compile(r'Alpha_(\d+(?:_\d+)*)')
_RE_H     = re.compile(r'H1_(\d+(?:_\d+)*)_H2_(\d+(?:_\d+)*)')
_RE_DR    = re.compile(r'Dr_(\d+(?:_\d+)*)')

def _parse_float(token: str) -> float:
    return float(token.replace('_', '.'))

def _discover_samples(data_dir: str) -> list:
    # CRITICAL FIX: Forces automatic conversion of shorthand character tildes to system folders
    data_dir = os.path.expanduser(data_dir)
    samples = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Provided data directory does not exist: {data_dir}")
        
    for alpha_entry in os.scandir(data_dir):
        if not alpha_entry.is_dir(): continue
        m_alpha = _RE_ALPHA.search(alpha_entry.name)
        if m_alpha is None: continue
        alpha = _parse_float(m_alpha.group(1))

        for h_entry in os.scandir(alpha_entry.path):
            if not h_entry.is_dir(): continue
            m_h = _RE_H.search(h_entry.name)
            if m_h is None: continue
            h2 = _parse_float(m_h.group(2))

            for dr_entry in os.scandir(h_entry.path):
                if not dr_entry.is_dir(): continue
                m_dr = _RE_DR.search(dr_entry.name)
                if m_dr is None: continue
                dr = _parse_float(m_dr.group(1))

                for npz_path in sorted(glob.glob(os.path.join(dr_entry.path, '*.npz'))):
                    samples.append((npz_path, alpha, dr, h2))
    return samples

def _build_sim_features(x: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, scaler: StandardScaler) -> np.ndarray:
    dx   = np.diff(x, prepend=x[0])
    dy   = np.diff(y, prepend=y[0])
    step = np.sqrt(dx ** 2 + dy ** 2)

    scaled = scaler.transform(np.stack([dx, dy, step], axis=1))
    net_disp   = np.sqrt(x ** 2 + y ** 2)
    cum_dist   = np.cumsum(step)
    efficiency = net_disp / (cum_dist + 1e-6)
    alpha_col  = np.full(len(x), alpha)

    features = np.stack([
        scaled[:, 0],                          # 0: dx
        scaled[:, 1],                          # 1: dy
        scaled[:, 2],                          # 2: step
        np.sin(theta),                         # 3: sin θ
        np.cos(theta),                         # 4: cos θ
        net_disp / (np.max(net_disp) + 1e-6),  # 5: normalized displacement
        efficiency,                            # 6: path efficiency
        alpha_col,                             # 7: alpha
    ], axis=1)
    return features.astype(np.float32)

class UniversalMigrationDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', split_ratios=(0.70, 0.20, 0.10), task_name='joint'):
        samples = _discover_samples(data_dir)
        train_prop, val_prop, test_prop = split_ratios
        
        # Pass 1: Secure structural scaling information from pure isolated training chunks
        self.scaler = StandardScaler()
        dyn_chunks = []
        for path, *_ in samples:
            data = np.load(path)
            x_arr = data['x_array']
            y_arr = data['y_array']
            n_sims = x_arr.shape[0]
            train_idx_end = int(n_sims * train_prop)
            
            x_train = x_arr[:train_idx_end]
            y_train = y_arr[:train_idx_end]
            dx = np.diff(x_train, axis=1, prepend=x_train[:, :1])
            dy = np.diff(y_train, axis=1, prepend=y_train[:, :1])
            step = np.sqrt(dx ** 2 + dy ** 2)
            dyn_chunks.append(np.stack([dx.ravel(), dy.ravel(), step.ravel()], axis=1))

        self.scaler.fit(np.concatenate(dyn_chunks, axis=0))

        # Pass 2: Process arrays conforming with matrix conditional task profiles
        inputs = []
        targets = []

        for path, alpha, dr, h2 in samples:
            data = np.load(path)
            x_arr, y_arr, theta_arr = data['x_array'], data['y_array'], data['theta_array']
            n_sims = x_arr.shape[0]
            
            split_1 = int(n_sims * train_prop)
            split_2 = int(n_sims * (train_prop + val_prop))

            if mode == 'train':
                sim_indices = range(0, split_1)
            elif mode == 'val':
                sim_indices = range(split_1, split_2)
            elif mode == 'test':
                sim_indices = range(split_2, n_sims)

            for i in sim_indices:
                base_feat = _build_sim_features(x_arr[i], y_arr[i], theta_arr[i], alpha, self.scaler)
                seq_len = base_feat.shape[0]
                
                if task_name == 'dr_only':
                    h2_col = np.full((seq_len, 1), h2, dtype=np.float32)
                    feat = np.hstack([base_feat, h2_col])
                    target = np.array([dr], dtype=np.float32)
                elif task_name == 'h_only':
                    log_dr = np.log1p(dr)
                    dr_col = np.full((seq_len, 1), log_dr, dtype=np.float32)
                    feat = np.hstack([base_feat, dr_col])
                    target = np.array([h2], dtype=np.float32)
                else:  # 'joint'
                    feat = base_feat
                    target = np.array([dr, h2], dtype=np.float32)
                    
                inputs.append(feat)
                targets.append(target)

        max_len = max(seq.shape[0] for seq in inputs)
        self.inputs = np.array([np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0))) for seq in inputs], dtype=np.float32)
        self.targets = np.array(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])