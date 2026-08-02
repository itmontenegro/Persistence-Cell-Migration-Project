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
    samples = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    for alpha_entry in os.scandir(data_dir):
        if not alpha_entry.is_dir():
            continue
        m_alpha = _RE_ALPHA.search(alpha_entry.name)
        if m_alpha is None:
            continue
        alpha = _parse_float(m_alpha.group(1))

        for h_entry in os.scandir(alpha_entry.path):
            if not h_entry.is_dir():
                continue
            m_h = _RE_H.search(h_entry.name)
            if m_h is None:
                continue
            h2 = _parse_float(m_h.group(2))

            for dr_entry in os.scandir(h_entry.path):
                if not dr_entry.is_dir():
                    continue
                m_dr = _RE_DR.search(dr_entry.name)
                if m_dr is None:
                    continue
                dr = _parse_float(m_dr.group(1))

                for npz_path in sorted(glob.glob(os.path.join(dr_entry.path, '*.npz'))):
                    samples.append((npz_path, alpha, dr, h2))
    return samples

def _build_sim_features(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    scaler: StandardScaler,
    target_mode: str,
    dr_val: float,
    h2_val: float
) -> np.ndarray:
    dx   = np.diff(x, prepend=x[0])   
    dy   = np.diff(y, prepend=y[0])
    step = np.sqrt(dx ** 2 + dy ** 2)

    # Scaler transformation step
    scaled = scaler.transform(np.stack([dx, dy, step], axis=1))

    net_disp   = np.sqrt(x ** 2 + y ** 2)
    cum_dist   = np.cumsum(step)
    efficiency = net_disp / (cum_dist + 1e-6)
    alpha_col  = np.full(len(x), alpha)

    feature_list = [
        scaled[:, 0],                          # 0 dx
        scaled[:, 1],                          # 1 dy
        scaled[:, 2],                          # 2 step
        np.sin(theta),                         # 3 sin θ
        np.cos(theta),                         # 4 cos θ
        net_disp / (np.max(net_disp) + 1e-6),  # 5 normalized net displacement
        efficiency,                            # 6 path efficiency
        alpha_col,                             # 7 alpha
    ]

    # Conditional injection based on operational configuration
    if target_mode == 'input_H2_predict_Dr':
        feature_list.append(np.full(len(x), h2_val))
    elif target_mode == 'input_Dr_predict_H2':
        feature_list.append(np.full(len(x), np.log1p(dr_val)))

    features = np.stack(feature_list, axis=1)
    return features.astype(np.float32)

class UniversalMigrationDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', split_type: str = '70_20_10', target_mode: str = 'joint'):
        self.target_mode = target_mode
        samples = _discover_samples(data_dir)

        if not samples:
            raise RuntimeError(f"No valid structured folders found under '{data_dir}'")

        if split_type == '60_30_10':
            r_train, r_val = 0.60, 0.30
        elif split_type == '70_20_10':
            r_train, r_val = 0.70, 0.20
        elif split_type == '80_15_5':
            r_train, r_val = 0.80, 0.15
        else:
            raise ValueError(f"Unknown split_type configuration: {split_type}")

        # Core Fix: Memory safe partial_fit avoids catastrophic memory allocation crashing
        self.scaler = StandardScaler()
        for path, *_ in samples:
            data = np.load(path)
            x_arr, y_arr = data['x_array'], data['y_array']
            train_idx_end = int(x_arr.shape[0] * r_train)
            
            x_train = x_arr[:train_idx_end]
            y_train = y_arr[:train_idx_end]

            dx   = np.diff(x_train, axis=1, prepend=x_train[:, :1])
            dy   = np.diff(y_train, axis=1, prepend=y_train[:, :1])
            step = np.sqrt(dx ** 2 + dy ** 2)
            
            self.scaler.partial_fit(np.stack([dx.ravel(), dy.ravel(), step.ravel()], axis=1))

        inputs  = []
        targets = []

        for path, alpha, dr, h2 in samples:
            data      = np.load(path)
            x_arr, y_arr, theta_arr = data['x_array'], data['y_array'], data['theta_array']

            n_sims = x_arr.shape[0]
            split_1 = int(n_sims * r_train)
            split_2 = int(n_sims * (r_train + r_val))

            if mode == 'train':
                sim_indices = range(0, split_1)
            elif mode == 'val':
                sim_indices = range(split_1, split_2)
            elif mode == 'test':
                sim_indices = range(split_2, n_sims)
            else:
                raise ValueError(f"Unknown mode assignment: {mode}")

            for i in sim_indices:
                feat = _build_sim_features(
                    x_arr[i], y_arr[i], theta_arr[i], alpha, self.scaler, target_mode, dr, h2
                )
                inputs.append(feat)
                
                if target_mode == 'joint':
                    targets.append(np.array([dr, h2], dtype=np.float32))
                elif target_mode == 'input_H2_predict_Dr':
                    targets.append(np.array([dr], dtype=np.float32))
                elif target_mode == 'input_Dr_predict_H2':
                    targets.append(np.array([h2], dtype=np.float32))

        max_len = max(seq.shape[0] for seq in inputs)
        self.inputs = np.array(
            [np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0))) for seq in inputs],
            dtype=np.float32,
        )
        self.targets = np.array(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])