import numpy as np
import matplotlib.pyplot as plt
import os

# The folder is the current folder
folder = os.path.dirname(__file__)  # points to evaluation/BILSTM

# Load files
predictions = np.load(os.path.join(folder, 'generated_ecg.npy'))  # shape: (num_samples, sequence_length)
ground_truth = np.load(os.path.join(folder, 'gt_ecg.npy'))        # shape: (num_samples, sequence_length)
mse_values = np.load(os.path.join(folder, 'mse.npy'))

# Plot first 5 samples
num_samples_to_plot = min(5, predictions.shape[0])

for i in range(num_samples_to_plot):
    plt.figure(figsize=(10, 4))
    plt.plot(ground_truth[i], label='Ground Truth', color='blue')
    plt.plot(predictions[i], label='Predicted (BiLSTM)', color='orange')
    plt.title(f"Sample {i} - MSE: {mse_values[i]:.5f}")
    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.show()
