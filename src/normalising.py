import librosa
import numpy as np
import os
import soundfile as sf  # To save the normalized files

def normalize_audio(y):
    #Normalize audio to the range [-1,1]
    original_max = np.max(np.abs(y))  # Get original max amplitude
    
    if original_max == 0:
        print("Warning: Audio is silent. Normalization skipped.")
        return y  # Return unchanged audio

    y_normalized = y / original_max  # Normalize
    new_max = np.max(np.abs(y_normalized))  # New max after normalization

    print(f"Original max amplitude: {original_max}")
    print(f"New max amplitude after normalization: {new_max}")
    
    return y_normalized

# Define the input and output folder paths
input_folder = "Enter your input folder path"  # Folder with resampled WAV files
output_folder = "Enter your output folder path"  # Folder for normalized WAV files

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each .wav file in the folder
for filename in os.listdir(input_folder):      
    if filename.endswith(".wav"):  # Process only .wav files
        file_path = os.path.join(input_folder, filename)

        # Load the audio (keep original 4000 Hz)
        y, sr = librosa.load(file_path, sr=4000)

        print(f"\nProcessing file: {filename}")
        print(f"Loaded sample rate: {sr} Hz")

        # Normalize the audio
        y_normalized = normalize_audio(y)

        # Save normalized audio in the output folder
        normalized_path = os.path.join(output_folder, f"normalized_{filename}")
        sf.write(normalized_path, y_normalized, sr)

        print(f"Saved normalized audio: {normalized_path}")

print("\nProcessing complete! Your original .txt files remain unchanged.")