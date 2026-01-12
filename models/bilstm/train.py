import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from models.bilstm.model import BiLSTM

# -------------------------------
# Arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--save_dir", default="saved_models/bilstm")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

device = torch.device("cpu")

# -------------------------------
# Load data
# -------------------------------
ppg = np.load("data/ppg_train.npy")   # (N, L)
ecg = np.load("data/ecg_train.npy")   # (N, L)

ppg = torch.tensor(ppg, dtype=torch.float32).unsqueeze(1)
ecg = torch.tensor(ecg, dtype=torch.float32)

dataset = TensorDataset(ppg, ecg)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# -------------------------------
# Model
# -------------------------------
model = BiLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch}/{args.epochs}]  MSE: {total_loss / len(loader):.6f}")

    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f"{args.save_dir}/bilstm_epoch_{epoch}.pth"
        )

print("✅ BiLSTM training complete")
