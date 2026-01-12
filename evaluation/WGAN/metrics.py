import numpy as np
import os

folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_results_run200'))
all_mse = []

for file in os.listdir(folder):
    if file.startswith("gen_") and file.endswith(".npy"):
        idx = file.split("_")[1].split(".")[0]
        gen = np.load(os.path.join(folder, f"gen_{idx}.npy"))
        gt = np.load(os.path.join(folder, f"gt_{idx}.npy"))
        mse = np.mean((gen - gt) ** 2)
        all_mse.append(mse)

print(f"Mean MSE over {len(all_mse)} samples: {np.mean(all_mse):.6f}")

