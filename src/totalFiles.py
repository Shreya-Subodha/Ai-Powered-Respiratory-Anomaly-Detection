import os

audio_folder = "/Users/shrey/Documents/Mini_Project_Dataset/archive/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"

# Get list of all WAV and TXT files
wav_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
txt_files = [f for f in os.listdir(audio_folder) if f.endswith('.txt')]

print(f"Total WAV files: {len(wav_files)}")
print(f"Total TXT files: {len(txt_files)}")