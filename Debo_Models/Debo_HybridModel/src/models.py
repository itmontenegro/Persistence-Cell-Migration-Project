import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        w = self.score(x)
        w = torch.softmax(w, dim=1)
        pooled = (x * w).sum(dim=1)
        return pooled

class OutputHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU()
        )
        self.dr = nn.Linear(128, 1)
        self.h = nn.Linear(128, 1)

    def forward(self, x):
        z = self.net(x)
        dr = self.dr(z)
        
        # Bound H between [0.5, 1.0] using scaled sigmoid
        h_raw = self.h(z)
        h = torch.sigmoid(h_raw) * 0.5 + 0.5 
        
        return torch.cat([dr, h], dim=1)

class ParamPredictorHybrid(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        
        self.embed = nn.Linear(input_dim, 128)
        self.pos_encoder = PositionalEncoding(128)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.15,
            batch_first=True,
            activation='gelu'
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.pool = AttentionPooling(128)

        self.alpha_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )

        # UPDATED: Input dimension changed from 6 to 8 to accommodate angular stats
        self.stats_branch = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        self.head = OutputHead(224)

    def extract_stats(self, x):
        dx = x[:, :, 0]
        dy = x[:, :, 1]
        step = x[:, :, 2]
        sin_t = x[:, :, 3]
        cos_t = x[:, :, 4]
        eff = x[:, :, 6]

        # --- 1. Translational Stats (Governs H) ---
        mask = (step != 0.0).float()
        actual_lengths = mask.sum(dim=1).clamp(min=1)

        mean_step = step.sum(dim=1) / actual_lengths
        step_centered = (step - mean_step.unsqueeze(1)) * mask
        std_step = torch.sqrt((step_centered**2).sum(dim=1) / actual_lengths)
        
        dir_var = (dx.std(dim=1) + dy.std(dim=1)) / 2 
        mean_eff = (eff * mask).sum(dim=1) / actual_lengths
        
        persistence = torch.sqrt(
            (dx.sum(dim=1) / actual_lengths)**2 +
            (dy.sum(dim=1) / actual_lengths)**2
        )

        eff_masked = eff.masked_fill(mask == 0, -1e9)
        max_eff = eff_masked.max(dim=1).values

        # --- 2. Rotational Stats (Governs Dr) ---
        # Calculate step-to-step changes in the orientation vectors
        d_sin = sin_t[:, 1:] - sin_t[:, :-1]
        d_cos = cos_t[:, 1:] - cos_t[:, :-1]
        
        # Magnitude of the angular change
        angular_change = torch.sqrt(d_sin**2 + d_cos**2 + 1e-8)
        
        # Adjust mask for the n-1 length of diff arrays
        mask_diff = mask[:, 1:]
        actual_lengths_diff = mask_diff.sum(dim=1).clamp(min=1)

        # Calculate mean and standard deviation of angular changes
        mean_ang_change = (angular_change * mask_diff).sum(dim=1) / actual_lengths_diff
        ang_centered = (angular_change - mean_ang_change.unsqueeze(1)) * mask_diff
        std_ang_change = torch.sqrt((ang_centered**2).sum(dim=1) / actual_lengths_diff)

        # --- 3. Stack all 8 features ---
        stats = torch.stack([
            mean_step,
            std_step,
            dir_var,
            mean_eff,
            persistence,
            max_eff,
            mean_ang_change, # Explicit proxy for Dr
            std_ang_change   # Explicit proxy for Dr
        ], dim=1)

        return stats

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.002

        z = self.embed(x)
        z = self.pos_encoder(z) 
        z = self.encoder(z)
        traj = self.pool(z)

        alpha = x[:, 0, 7].unsqueeze(1)
        alpha_vec = self.alpha_branch(alpha)

        stats = self.extract_stats(x)
        stats_vec = self.stats_branch(stats)

        fused = torch.cat([traj, alpha_vec, stats_vec], dim=1)

        return self.head(fused)

def get_model():
    return ParamPredictorHybrid()