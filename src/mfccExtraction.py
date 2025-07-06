import os
import librosa
import numpy as np
import pandas as pd

# ==== Set Your Paths ====
audio_and_annotation_dir = "/Users/shrey/Documents/Mini_Project_Dataset/final"
output_feature_dir = "data/segment_mfcc_npy"
metadata_csv_path = "data/rnn_mfcc_metadata.csv"
os.makedirs(output_feature_dir, exist_ok=True)

# ==== Patient Diagnosis Mapping ====
patient_diagnosis = {
    "101": "URTI", "102": "Healthy", "103": "Asthma", "104": "COPD", "105": "URTI", "106": "COPD", "107": "COPD", "108": "LRTI",
    "109": "COPD", "110": "COPD", "111": "Bronchiectasis", "112": "COPD", "113": "COPD", "114": "COPD", "115": "LRTI",
    "116": "Bronchiectasis", "117": "COPD", "118": "COPD", "119": "URTI", "120": "COPD", "121": "Healthy", "122": "Pneumonia",
    "123": "Healthy", "124": "COPD", "125": "Healthy", "126": "Healthy", "127": "Healthy", "128": "COPD", "129": "URTI",
    "130": "COPD", "131": "URTI", "132": "COPD", "133": "COPD", "134": "COPD", "135": "Pneumonia", "136": "Healthy",
    "137": "URTI", "138": "COPD", "139": "COPD", "140": "Pneumonia", "141": "COPD", "142": "COPD", "143": "Healthy",
    "144": "Healthy", "145": "COPD", "146": "COPD", "147": "COPD", "148": "URTI", "149": "Bronchiolitis", "150": "URTI",
    "151": "COPD", "152": "Healthy", "153": "Healthy", "154": "COPD", "155": "COPD", "156": "COPD", "157": "COPD",
    "158": "COPD", "159": "Healthy", "160": "COPD", "161": "Bronchiolitis", "162": "COPD", "163": "COPD", "164": "URTI",
    "165": "URTI", "166": "COPD", "167": "Bronchiolitis", "168": "Bronchiectasis", "169": "Bronchiectasis", "170": "COPD",
    "171": "Healthy", "172": "COPD", "173": "Bronchiolitis", "174": "COPD", "175": "COPD", "176": "COPD", "177": "COPD",
    "178": "COPD", "179": "Healthy", "180": "COPD", "181": "COPD", "182": "Healthy", "183": "Healthy", "184": "Healthy",
    "185": "COPD", "186": "COPD", "187": "Healthy", "188": "URTI", "189": "COPD", "190": "URTI", "191": "Pneumonia",
    "192": "COPD", "193": "COPD", "194": "Healthy", "195": "COPD", "196": "Bronchiectasis", "197": "URTI", "198": "COPD",
    "199": "COPD", "200": "COPD", "201": "Bronchiectasis", "202": "Healthy", "203": "COPD", "204": "COPD", "205": "COPD",
    "206": "Bronchiolitis", "207": "COPD", "208": "Healthy", "209": "Healthy", "210": "URTI", "211": "COPD", "212": "COPD",
    "213": "COPD", "214": "Healthy", "215": "Bronchiectasis", "216": "Bronchiolitis", "217": "Healthy", "218": "COPD",
    "219": "Pneumonia", "220": "COPD", "221": "COPD", "222": "COPD", "223": "COPD", "224": "Healthy", "225": "Healthy",
    "226": "Pneumonia"
}

diagnosis_to_risk = {
    "Healthy": 0,
    "URTI": 1,
    "Asthma": 1,
    "Bronchiolitis": 1,
    "Bronchiectasis": 2,
    "LRTI": 2,
    "COPD": 2,
    "Pneumonia": 2
}

# ==== Extract and Save Full MFCC Sequences ====
metadata = []

for file in os.listdir(audio_and_annotation_dir):
    if file.endswith(".wav"):
        try:
            audio_path = os.path.join(audio_and_annotation_dir, file)
            txt_path = os.path.join(audio_and_annotation_dir, file.replace(".wav", ".txt"))
            if not os.path.exists(txt_path):
                print(f"Annotation missing for {file}, skipping.")
                continue

            y, sr = librosa.load(audio_path, sr=4000)

            patient_id = file.split("_")[0]
            diagnosis = patient_diagnosis.get(patient_id, "Healthy")
            risk_level = diagnosis_to_risk.get(diagnosis, 0)

            with open(txt_path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                start_time, end_time, wheeze, crackle = map(float, line.strip().split())
                wheeze, crackle = int(wheeze), int(crackle)
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = y[start_sample:end_sample]

                mfcc = librosa.feature.mfcc(y=segment.astype(float), sr=sr, n_mfcc=13).T

                desired_length = 50
                if mfcc.shape[0] < desired_length:
                    pad_width = desired_length - mfcc.shape[0]
                    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
                elif mfcc.shape[0] > desired_length:
                    mfcc = mfcc[:desired_length, :]

                segment_id = f"{patient_id}_{file[:-4]}_seg{i}"
                mfcc_path = os.path.join(output_feature_dir, f"{segment_id}.npy")
                np.save(mfcc_path, mfcc)

                metadata.append({
                    "segment_id": segment_id,
                    "mfcc_path": mfcc_path,
                    "patient_id": patient_id,
                    "wheeze": wheeze,
                    "crackle": crackle,
                    "risk_level": risk_level
                })

        except Exception as e:
            print(f"Error processing {file}: {e}")

# ==== Save metadata CSV ====
df = pd.DataFrame(metadata)
df.to_csv(metadata_csv_path, index=False)
print(f"✅ Saved metadata CSV to: {metadata_csv_path}")
