import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# === Load a sample MFCC file ===
mfcc_path = "/Users/shrey/Documents/GitHub/Ai-Powered-Respiratory-Anomaly-Detection/data/segment_mfcc_npy/101_101_1b1_Pr_sc_Meditron_seg0.npy"  # change this to your file
mfcc = np.load(mfcc_path)

# === Print shape and a few values ===
print(f"File: {mfcc_path}")
print("Shape:", mfcc.shape)        # Should be (50, 13)
print("First 5 rows:\n", mfcc[:5]) # Preview of values

# === Visualize MFCC as a heatmap ===
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc.T, x_axis='time')
plt.colorbar(label='MFCC Coefficient Value')
plt.title("MFCC Heatmap")
plt.tight_layout()
plt.show()
