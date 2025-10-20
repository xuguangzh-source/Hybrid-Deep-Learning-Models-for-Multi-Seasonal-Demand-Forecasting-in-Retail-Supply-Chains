import torch
import torch.nn as nn
from .tcn_layer import TCN
from .attention_layer import AdditiveAttention

class HybridTCNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, tcn_channels, tcn_kernel_size, tcn_dropout,
                 lstm_hidden, lstm_layers, attention_dim, dense_hidden, horizon, dropout=0.2):
        super().__init__()
        self.tcn = TCN(num_inputs=input_dim, channels=tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        tcn_out = tcn_channels[-1]

        self.bilstm = nn.LSTM(input_size=tcn_out, hidden_size=lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.attn = AdditiveAttention(query_dim=2*lstm_hidden, key_dim=2*lstm_hidden, attn_dim=attention_dim)

        self.head = nn.Sequential(
            nn.Linear(2*lstm_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, horizon)
        )

    def forward(self, x):  # x: (B, T, F)
        z = self.tcn(x)                            # (B, T, C_tcn)
        h, _ = self.bilstm(z)                      # (B, T, 2*H)
        q = h[:, -1, :]                            # use last step as query
        context, attn = self.attn(q, h, h)         # (B, 2H), (B, T)
        y = self.head(context)                     # (B, HORIZON)
        return y, attn
