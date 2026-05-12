import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class UniversalMigrationDataset(Dataset):
    def __init__(self, data_dir, mode='train', target_vars=['Dr', 'H']):
        self.inputs = []
        self.targets = []

        folders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        samples = []

        for folder in folders:
            name = os.path.basename(folder)
            m = re.search(r'Alpha_(\d+(?:_\d+)*)__Dr_(\d+(?:_\d+)*)__H_(\d+(?:_\d+)*)', name)

            if m is None:
                continue

            alpha = float(m.group(1).replace('_', '.'))
            dr = float(m.group(2).replace('_', '.'))
            h = float(m.group(3).replace('_', '.'))

            files = sorted(glob.glob(os.path.join(folder, "*.csv")))

            for f in files:
                samples.append((f, alpha, dr, h))

        train_files, test_files = train_test_split(samples, test_size=0.30, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=0.50, random_state=42)

        if mode == 'train':
            selected = train_files
        elif mode == 'val':
            selected = val_files
        else:
            selected = test_files

        self.scaler = StandardScaler()
        dyn = []

        for path, alpha, dr, h in train_files:
            df = pd.read_csv(path)
            dx = df['X'].diff().fillna(0).values
            dy = df['Y'].diff().fillna(0).values
            step = np.sqrt(dx**2 + dy**2)
            dyn.append(np.stack([dx, dy, step], axis=1))

        dyn = np.concatenate(dyn, axis=0)
        self.scaler.fit(dyn)

        for path, alpha, dr, h in selected:
            df = pd.read_csv(path).fillna(0)

            x = df['X'].values
            y = df['Y'].values
            theta = df['Theta'].values

            dx = df['X'].diff().fillna(0).values
            dy = df['Y'].diff().fillna(0).values
            step = np.sqrt(dx**2 + dy**2)

            scaled = self.scaler.transform(np.stack([dx, dy, step], axis=1))

            net_disp = np.sqrt(x**2 + y**2)
            cum_dist = np.cumsum(step)
            efficiency = net_disp / (cum_dist + 1e-6)
            alpha_col = np.full(len(df), alpha)

            features = np.stack([
                scaled[:, 0], 
                scaled[:, 1], 
                scaled[:, 2], 
                np.sin(theta), 
                np.cos(theta), 
                net_disp / (np.max(net_disp)+1e-6), 
                efficiency, 
                alpha_col 
            ], axis=1)
            
            self.inputs.append(features.astype(np.float32))
            dr_scaled = np.log1p(dr) 
            self.targets.append(np.array([dr_scaled, h], dtype=np.float32))

        max_len = max(len(x) for x in self.inputs)
        padded = []

        for seq in self.inputs:
            pad = max_len - len(seq)
            padded.append(np.pad(seq, ((0, pad), (0, 0))))

        self.inputs = np.array(padded, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])