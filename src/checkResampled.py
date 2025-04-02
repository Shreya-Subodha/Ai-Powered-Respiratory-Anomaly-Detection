#checking resampled

import os
import soundfile as sf

# Set the folder path where your .wav files are stored
folder_path =  "Enter your folder path here"

# Counters for specific sampling rates
count_44100 = 0
count_4000 = 0

# List all .wav files in the folder
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Process each file
for wav_file in wav_files:
    file_path = os.path.join(folder_path, wav_file)
    try:
        data, samplerate = sf.read(file_path)
        print(f"File: {wav_file}, Sampling Rate: {samplerate} Hz")

        # Count occurrences of specific sampling rates
        if samplerate == 44100:
            count_44100 += 1
        elif samplerate == 4000:
            count_4000 += 1

    except Exception as e:
        print(f"Error reading {wav_file}: {e}")

# Print final counts
print("\nSummary:")
print(f"Number of files with 44100 Hz sampling rate: {count_44100}")
print(f"Number of files with 4000 Hz sampling rate: {count_4000}")