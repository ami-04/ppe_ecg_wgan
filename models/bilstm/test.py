import os
import torch
import numpy as np
from models.bilstm.model import BiLSTM

device = torch.device("cpu")

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "saved_models/bilstm/bilstm_epoch_200.pth"
OUT_DIR = "evaluation/bilstm"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# Load data
# -------------------------------
ppg = np.load("data/ppg_eval.npy")
ecg = np.load("data/ecg_eval.npy")

ppg_t = torch.tensor(ppg, dtype=torch.float32).unsqueeze(1)
ecg_t = torch.tensor(ecg, dtype=torch.float32)

# -------------------------------
# Load model
# -------------------------------
model = BiLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -------------------------------
# Inference
# -------------------------------
with torch.no_grad():
    pred = model(ppg_t).cpu().numpy()

np.save(f"{OUT_DIR}/generated_ecg.npy", pred)
np.save(f"{OUT_DIR}/gt_ecg.npy", ecg)

mse = np.mean((pred - ecg) ** 2)
np.save(f"{OUT_DIR}/mse.npy", np.array([mse]))

print(f"✅ Test complete | Mean MSE: {mse:.6f}")
