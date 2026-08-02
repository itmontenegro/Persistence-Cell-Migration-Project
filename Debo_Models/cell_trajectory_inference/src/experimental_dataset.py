# src/experimental_dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ExperimentalMigrationDataset(Dataset):
    """
    Dataset class to convert raw experimental cell coordinate CSVs 
    into input tensors compatible with ParamPredictorHybrid model.
    """
    def __init__(self, csv_path: str, alpha_val: float = 1.0):
        self.csv_path = csv_path
        self.alpha_val = alpha_val
        
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist
        required_cols = {'Cell_No', 'X', 'Y', 'Timepoint'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        self.cell_ids = []
        self.trajectories = []
        self.timepoints = []
        
        # Process trajectory sequence per cell
        for cell_id, group in df.groupby('Cell_No'):
            group = group.sort_values('Timepoint')
            x = group['X'].values.astype(np.float32)
            y = group['Y'].values.astype(np.float32)
            tp = group['Timepoint'].values.astype(np.float32)
            
            self.cell_ids.append(cell_id)
            self.timepoints.append(tp)
            self.trajectories.append((x, y))

        # Build feature sequence for each cell trajectory
        inputs_list = []
        for x, y in self.trajectories:
            feat = self._build_features(x, y)
            inputs_list.append(feat)

        # Sequence padding to uniform length across cell batch
        max_len = max(seq.shape[0] for seq in inputs_list)
        self.inputs = np.array(
            [np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0))) for seq in inputs_list],
            dtype=np.float32,
        )

    def _build_features(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Relative trajectory coordinates zero-centered at origin
        x_rel = x - x[0]
        y_rel = y - y[0]

        # Step displacements
        dx = np.diff(x_rel, prepend=x_rel[0])
        dy = np.diff(y_rel, prepend=y_rel[0])
        step = np.sqrt(dx**2 + dy**2)

        # Instantaneous movement direction angle (theta)
        theta = np.arctan2(dy, dx)
        if len(theta) > 1:
            theta[0] = theta[1]  # Smooth boundary condition at t=0

        # Step-size feature standard scaling
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.stack([dx, dy, step], axis=1))

        # Displacement and path efficiency calculations
        net_disp = np.sqrt(x_rel**2 + y_rel**2)
        cum_dist = np.cumsum(step)
        efficiency = net_disp / (cum_dist + 1e-6)
        alpha_col = np.full(len(x), self.alpha_val, dtype=np.float32)

        # 8-Channel Feature List (matches training pipeline format)
        feature_list = [
            scaled[:, 0],                                         # 0: dx (scaled)
            scaled[:, 1],                                         # 1: dy (scaled)
            scaled[:, 2],                                         # 2: step (scaled)
            np.sin(theta).astype(np.float32),                     # 3: sin(theta)
            np.cos(theta).astype(np.float32),                     # 4: cos(theta)
            (net_disp / (np.max(net_disp) + 1e-6)).astype(np.float32), # 5: normalized net disp
            efficiency.astype(np.float32),                        # 6: path efficiency
            alpha_col,                                            # 7: alpha channel
        ]

        return np.stack(feature_list, axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return torch.tensor(self.inputs[idx]), self.cell_ids[idx]