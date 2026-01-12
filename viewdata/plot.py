import numpy as np
import matplotlib.pyplot as plt
import os

# --- paths ---
base = "/Users/abhiramibhargan/Documents/p2e_samples"
results = "/Users/abhiramibhargan/Documents/p2e_samples/test_results_run200"

# Load evaluation set (ground truth)
ecg_eval = np.load(os.path.join(base, "ecg_eval.npy"))   # shape (995, 375)
ppg_eval = np.load(os.path.join(base, "ppg_eval.npy"))   # shape (995, 375)

# Pick sample index you want to inspect
idx = 0  # <--- change this to view gen_00001, gen_00002, etc.

# Load generated sample
gen_file = f"gen_{idx:05d}.npy"
gen_ecg = np.load(os.path.join(results, gen_file))

# Ground truth ECG
true_ecg = ecg_eval[idx]

# Plotting
plt.figure(figsize=(12, 5))
plt.plot(true_ecg, label="Ground Truth ECG", linewidth=2)
plt.plot(gen_ecg, label="Generated ECG", linestyle='dashed')
plt.legend()
plt.title(f"ECG Comparison for Sample {idx}")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.grid(True)
plt.show()
