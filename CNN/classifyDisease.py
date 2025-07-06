import os
import pandas as pd
import shutil

# === Paths ===
csv_file_path = "/Users/shrey/Documents/Mini_Project_Dataset/archive copy/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
files_folder_path = "/Users/shrey/Documents/Mini_Project_Dataset/final"
destination_base_path = "/Users/shrey/Documents/Mini_Project_Dataset/Diseases"

# === Load CSV ===
df = pd.read_csv(csv_file_path, header=None)  # No header row in your CSV
df.columns = ['patient_id', 'disease']  # Add column names

# === Create folders and copy files ===
for index, row in df.iterrows():
    patient_id = str(row['patient_id']).strip()       # E.g., '101'
    disease_name = str(row['disease']).strip()        # E.g., 'COPD'

    # Find all .wav files that start with patient_id_
    matching_files = [f for f in os.listdir(files_folder_path)
                      if f.startswith(f"{patient_id}_") and f.endswith(".wav")]

    # Destination disease folder
    dest_folder = os.path.join(destination_base_path, disease_name)
    os.makedirs(dest_folder, exist_ok=True)

    # Copy each file
    for file in matching_files:
        source_path = os.path.join(files_folder_path, file)
        dest_path = os.path.join(dest_folder, file)
        shutil.copy(source_path, dest_path)
        # print(f"Copied {file} to {disease_name}")

print("\nAll matching .wav files copied successfully to their disease folders.")