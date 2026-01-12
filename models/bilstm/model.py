import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    """
    Bi-directional LSTM for PPG -> ECG regression
    Input:  (B, L) or (B, 1, L)
    Output: (B, L)
    """
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: (B, 1, L) → (B, L, 1)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)        # (B, L, 2H)
        out = self.fc(lstm_out)           # (B, L, 1)
        return out.squeeze(-1)            # (B, L)
