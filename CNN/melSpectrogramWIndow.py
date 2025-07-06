import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Use PIL for good quality resizing

# Input and output directories
input_base_dir = "/Users/shrey/Documents/Mini_Project_Dataset/Diseases"
output_base_dir = "/Users/shrey/Documents/Mini_Project_Dataset/Spectrograms_Mel_Corrected"

os.makedirs(output_base_dir, exist_ok=True)

for disease_folder in os.listdir(input_base_dir):
    input_folder_path = os.path.join(input_base_dir, disease_folder)
    output_folder_path = os.path.join(output_base_dir, disease_folder)

    if not os.path.isdir(input_folder_path):
        continue

    os.makedirs(output_folder_path, exist_ok=True)

    for file_name in os.listdir(input_folder_path):
        if file_name.lower().endswith(".wav"):
            file_path = os.path.join(input_folder_path, file_name)

            try:
                y, sr = librosa.load(file_path, sr=None)

                # Parameters
                window_duration = 0.060  # 60ms
                hop_duration = 0.030     # 50% overlap

                n_fft = int(window_duration * sr)
                hop_length = int(hop_duration * sr)

                # Create Mel spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)

                # Plot and save spectrogram with natural aspect
                fig, ax = plt.subplots()
                ax.axis('off')
                librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, cmap='plasma', x_axis=None, y_axis=None)
                
                temp_path = os.path.join(output_folder_path, "temp.png")
                plt.savefig(temp_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Save at high resolution
                plt.close()

                # Now resize the image to 128x128 using PIL
                img = Image.open(temp_path)
                img = img.resize((128, 128), Image.LANCZOS)  # High quality anti-aliasing resize
                final_path = os.path.join(output_folder_path, file_name.replace(".wav", ".png"))
                img.save(final_path)

                os.remove(temp_path)  # Delete temp

                print(f"Saved clean resized spectrogram: {final_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
