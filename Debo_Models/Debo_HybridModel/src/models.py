import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU()
        )
        self.dr = nn.Linear(128, 1)
        self.h = nn.Linear(128, 1)

    def forward(self, x):
        z = self.net(x)
        dr = self.dr(z)
        h = self.h(z)
        return torch.cat([dr, h], dim=1)

class ParamPredictorHybrid(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        
        self.embed = nn.Linear(input_dim, 128)

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

        self.stats_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

        self.head = OutputHead(224)

    def extract_stats(self, x):
        dx = x[:, :, 0]
        dy = x[:, :, 1]
        step = x[:, :, 2]
        eff = x[:, :, 6]

        mean_step = step.mean(dim=1)
        std_step = step.std(dim=1)
        dir_var = (dx.std(dim=1) + dy.std(dim=1)) / 2
        mean_eff = eff.mean(dim=1)
        
        persistence = torch.sqrt(
            dx.mean(dim=1)**2 +
            dy.mean(dim=1)**2
        )

        max_eff = eff.max(dim=1).values

        stats = torch.stack([
            mean_step,
            std_step,
            dir_var,
            mean_eff,
            persistence,
            max_eff
        ], dim=1)

        return stats

    def forward(self, x):
        if self.training:
            x = x + torch.randn_like(x) * 0.002

        z = self.embed(x)
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