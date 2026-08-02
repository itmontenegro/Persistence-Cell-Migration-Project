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
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)

class FlexibleOutputHead(nn.Module):
    def __init__(self, dim, target_mode):
        super().__init__()
        self.target_mode = target_mode
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU()
        )
        if target_mode == 'joint':
            self.dr_layer = nn.Linear(128, 1)
            self.h_layer  = nn.Linear(128, 1)
        elif target_mode == 'input_H2_predict_Dr':
            self.dr_layer = nn.Linear(128, 1)
        elif target_mode == 'input_Dr_predict_H2':
            self.h_layer  = nn.Linear(128, 1)

    def forward(self, x):
        z = self.net(x)
        if self.target_mode == 'joint':
            dr = self.dr_layer(z)
            h  = torch.sigmoid(self.h_layer(z)) * 0.5 + 0.5
            return torch.cat([dr, h], dim=1)
        elif self.target_mode == 'input_H2_predict_Dr':
            return self.dr_layer(z)
        elif self.target_mode == 'input_Dr_predict_H2':
            return torch.sigmoid(self.h_layer(z)) * 0.5 + 0.5
        raise ValueError(f"Invalid target mode: {self.target_mode}")

class ParamPredictorHybrid(nn.Module):
    def __init__(self, target_mode='joint'):
        super().__init__()
        self.target_mode = target_mode
        self.input_dim = 8 if target_mode == 'joint' else 9
        
        self.embed = nn.Linear(self.input_dim, 128)
        self.pos_encoder = PositionalEncoding(128)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, dropout=0.15, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.pool = AttentionPooling(128)

        self.alpha_branch = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32)
        )

        stats_in_dim = 8 if target_mode == 'joint' else 9
        self.stats_branch = nn.Sequential(
            nn.Linear(stats_in_dim, 64), nn.GELU(), nn.Linear(64, 64)
        )
        
        # Core Fix: 128 (pooled features) + 32 (alpha mapped features) + 64 (handcrafted stats) = 224
        self.head = FlexibleOutputHead(224, target_mode)

    def extract_stats(self, x):
        dx     = x[:, :, 0]
        dy     = x[:, :, 1]
        step   = x[:, :, 2]
        sin_t  = x[:, :, 3]
        cos_t  = x[:, :, 4]
        eff    = x[:, :, 6]

        mask = (step != 0.0).float()
        actual_lengths = mask.sum(dim=1).clamp(min=1)

        mean_step = step.sum(dim=1) / actual_lengths
        step_centered = (step - mean_step.unsqueeze(1)) * mask
        std_step = torch.sqrt((step_centered**2).sum(dim=1) / actual_lengths)
        
        dir_var = (dx.std(dim=1) + dy.std(dim=1)) / 2 
        mean_eff = (eff * mask).sum(dim=1) / actual_lengths
        
        persistence = torch.sqrt((dx.sum(dim=1) / actual_lengths)**2 + (dy.sum(dim=1) / actual_lengths)**2)
        eff_masked = eff.masked_fill(mask == 0, -1e9)
        max_eff = eff_masked.max(dim=1).values

        d_sin = sin_t[:, 1:] - sin_t[:, :-1]
        d_cos = cos_t[:, 1:] - cos_t[:, :-1]
        angular_change = torch.sqrt(d_sin**2 + d_cos**2 + 1e-8)
        
        mask_diff = mask[:, 1:]
        actual_lengths_diff = mask_diff.sum(dim=1).clamp(min=1)

        mean_ang_change = (angular_change * mask_diff).sum(dim=1) / actual_lengths_diff
        ang_centered = (angular_change - mean_ang_change.unsqueeze(1)) * mask_diff
        std_ang_change = torch.sqrt((ang_centered**2).sum(dim=1) / actual_lengths_diff)

        stats_list = [mean_step, std_step, dir_var, mean_eff, persistence, max_eff, mean_ang_change, std_ang_change]

        if self.target_mode in ['input_H2_predict_Dr', 'input_Dr_predict_H2']:
            stats_list.append(x[:, 0, 8])

        return torch.stack(stats_list, dim=1)

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

def get_model(target_mode='joint'):
    return ParamPredictorHybrid(target_mode=target_mode)