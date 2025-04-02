import os
import numpy as np
import soundfile as sf
import librosa
import pywt
import scipy.signal as signal

# Paths
INPUT_FOLDER = "Enter your input folder path here"
OUTPUT_FOLDER = "Enter your output folder path here"
#TXT_FILE = "/Users/shrey/Documents/Mini_Project_Dataset/txt check/101_1b1_Al_sc_Meditron copy.txt"  # Path to TXT file

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Bandpass filter (80 Hz - 2000 Hz) with fix
def bandpass_filter(audio, sr, lowcut=80, highcut=2000, order=4):
    nyquist = 0.5 * sr
    if highcut >= nyquist:
        highcut = nyquist - 1  # Ensure highcut is below Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid cutoff frequencies: low={lowcut}Hz, high={highcut}Hz (Nyquist={nyquist}Hz)")
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, audio)

# Remove heart sounds using Wavelet Transform
def remove_heart_sounds(audio, wavelet='db6', level=5):
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # Remove low-frequency components
    coeffs[1] = np.zeros_like(coeffs[1])
    return pywt.waverec(coeffs, wavelet)

# Process each file in the input folder
for file_name in os.listdir(INPUT_FOLDER):
    if file_name.endswith('.wav'):
        file_path = os.path.join(INPUT_FOLDER, file_name)
        audio, sr = librosa.load(file_path, sr=None)
        filtered_audio = bandpass_filter(audio, sr)
        heart_removed_audio = remove_heart_sounds(filtered_audio)
        output_path = os.path.join(OUTPUT_FOLDER, file_name)
        sf.write(output_path, heart_removed_audio, sr)
        print(f"Processed and saved: {output_path}")