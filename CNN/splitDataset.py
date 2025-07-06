import os
import random
import shutil

# Base folder where the current 5 disease folders are located
input_base_dir = "/Users/shrey/Documents/Mini_Project_Dataset/Stitched_Spectrograms"

# Target folders
output_base_dir = "/Users/shrey/Documents/Mini_Project_Dataset/Final_Split"
train_dir = os.path.join(output_base_dir, "train")
val_dir = os.path.join(output_base_dir, "val")
test_dir = os.path.join(output_base_dir, "test")

# Create target folders
for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)

# Target diseases
disease_classes = ["Pneumonia", "Bronchiectasis", "URTI", "Healthy", "Bronchiolitis", "COPD"]

# Splitting ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

for disease in disease_classes:
    disease_folder = os.path.join(input_base_dir, disease)
    images = [f for f in os.listdir(disease_folder) if f.endswith('.png')]

    random.shuffle(images)

    # 🛑 Very important: limit to 100 images if COPD (or any class with >100 images)
    if len(images) > 100:
        images = images[:100]

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # Remaining for test

    splits = {
        train_dir: images[:n_train],
        val_dir: images[n_train:n_train+n_val],
        test_dir: images[n_train+n_val:]
    }

    for split_folder, split_images in splits.items():
        split_disease_folder = os.path.join(split_folder, disease)
        os.makedirs(split_disease_folder, exist_ok=True)

        for img_name in split_images:
            src_path = os.path.join(disease_folder, img_name)
            dst_path = os.path.join(split_disease_folder, img_name)
            shutil.copy(src_path, dst_path)

    print(f"{disease}: {n_train} train, {n_val} val, {n_test} test images created.")
