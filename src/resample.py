#resampling

import librosa
import soundfile as sf
import os

# Define the input and output folder paths

input_folder = "Enter your input file path here"
output_folder = "Enter your output file path here"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Target sampling rate
target_sr = 4000

# Process all WAV files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):  # Process only WAV files
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"resampled_{filename}")
        
        # Load original audio
        audio, sr = librosa.load(input_path, sr=None)
        
        # Resample audio
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Save resampled audio
        sf.write(output_path, audio_resampled, target_sr)
        print(f"Resampled and saved: {output_path}")

print("All files have been processed and saved in the output folder.")