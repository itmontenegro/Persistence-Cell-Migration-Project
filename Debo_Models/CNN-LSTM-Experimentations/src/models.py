import torch
import torch.nn as nn

class ParamPredictorCNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ParamPredictorCNN, self).__init__()
        
        # Pure 1D Dilated Convolutional Network (Preserves temporal features)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=5, dilation=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, dilation=4, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # Input: [Batch, Seq_Len, Features] -> Transpose for Conv1d: [Batch, Features, Seq_Len]
        x = x.permute(0, 2, 1)
        features = self.conv_block(x)
        features = features.squeeze(-1)  # Drop temporal dimension after global pooling
        return self.fc(features)


class ParamPredictorLSTM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super(ParamPredictorLSTM, self).__init__()
        
        # Pure Recurrent Neural Network
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.2
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Input shape: [Batch, Seq_Len, Features]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        pooled = lstm_out.mean(dim=1)  # Global temporal sequence pooling
        return self.fc(pooled)