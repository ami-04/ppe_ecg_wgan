import numpy as np
import matplotlib.pyplot as plt
import os
import os
folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_results_run200'))
num_samples_to_plot = 5

for i in range(num_samples_to_plot):
    gen = np.load(os.path.join(folder, f"gen_{i:05d}.npy"))
    gt = np.load(os.path.join(folder, f"gt_{i:05d}.npy"))

    plt.figure(figsize=(10,4))
    plt.plot(gt, label='Ground Truth', color='blue')
    plt.plot(gen, label='Generated', color='orange')
    plt.title(f"Sample {i}")
    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.show()
