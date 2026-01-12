import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. DEFINE FILE PATHS ---
# Define the base directory where your p2e_samples folder is located
BASE_DIR = '/Users/abhiramibhargan/Documents/p2e_samples/'

# Define the full paths to the two relevant training files
ppg_file_path = os.path.join(BASE_DIR, 'ppg_train.npy')
ecg_file_path = os.path.join(BASE_DIR, 'ecg_train.npy')

print("Attempting to load data...")

try:
    # --- 2. LOAD DATA ---
    ppg_data = np.load(ppg_file_path)
    ecg_data = np.load(ecg_file_path)
    
    # --- 3. INSPECT DATA STRUCTURE ---
    print(f"✅ Data loaded successfully!")
    print(f"PPG Data Shape: {ppg_data.shape}")
    print(f"ECG Data Shape: {ecg_data.shape}")
    print(f"Data Type (dtype): {ppg_data.dtype}")
    print("-" * 30)

    # Assuming the shape is (N_samples, N_timesteps)
    
    # --- 4. SELECT SAMPLE ---
    sample_index = 0
    
    # Select the first sample from both arrays
    ppg_sample = ppg_data[sample_index]
    ecg_sample = ecg_data[sample_index]
    
    # Create the time steps array based on the length of the sample
    time_steps = np.arange(ppg_sample.shape[0])

    # --- 5. PLOT THE WAVEFORMS ---
    plt.figure(figsize=(14, 8))

    # Subplot 1: PPG Waveform
    plt.subplot(2, 1, 1) # 2 rows, 1 column, position 1
    plt.plot(time_steps, ppg_sample, label=f'PPG Waveform (Sample {sample_index})', color='blue')
    plt.title('Photoplethysmogram (PPG) Signal', fontsize=14)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: ECG Waveform
    plt.subplot(2, 1, 2) # 2 rows, 1 column, position 2
    plt.plot(time_steps, ecg_sample, label=f'ECG Waveform (Sample {sample_index})', color='red', alpha=0.8)
    plt.title('Electrocardiogram (ECG) Signal', fontsize=14)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Normalized Amplitude', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('P2E-WGAN Training Sample Pair', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

except FileNotFoundError as e:
    print(f"❌ ERROR: File not found: {e.filename}")
    print("Please ensure your 'p2e_samples' folder is exactly here: /Users/abhiramibhargan/Documents/p2e_samples/")
except IndexError:
    print("❌ ERROR: Data structure unexpected. The indexing might be wrong.")
    print("Check the printed shapes and adjust 'sample_index' or the plotting logic.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")