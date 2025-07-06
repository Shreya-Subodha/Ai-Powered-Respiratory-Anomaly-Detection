import os
import random
from PIL import Image

# Input spectrogram folder
input_base_dir = "/Users/shrey/Documents/Mini_Project_Dataset/Spectrograms_Mel_Corrected"
# Output stitched spectrogram folder
output_base_dir = "/Users/shrey/Documents/Mini_Project_Dataset/Stitched_Spectrograms"

# Target diseases for stitching
target_diseases = ["Pneumonia", "Bronchiectasis", "URTI", "Healthy", "Bronchiolitis"]

# Parameters
target_total_images_per_class = 100  # including originals
image_size = (128, 128)  # final output size

os.makedirs(output_base_dir, exist_ok=True)

for disease in target_diseases:
    input_folder = os.path.join(input_base_dir, disease)
    output_folder = os.path.join(output_base_dir, disease)

    os.makedirs(output_folder, exist_ok=True)

    # List original images
    original_images = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]

    # Copy original images first
    for img_name in original_images:
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(image_size)
        img.save(os.path.join(output_folder, img_name))

    print(f"Copied {len(original_images)} original images for {disease}.")

    # How many more images needed
    additional_needed = target_total_images_per_class - len(original_images)

    if additional_needed <= 0:
        print(f"No stitching needed for {disease}. Already {len(original_images)} images.")
        continue

    augment_count = 0
    while augment_count < additional_needed:
        # Pick two random different images
        img_name1, img_name2 = random.sample(original_images, 2)
        img_path1 = os.path.join(input_folder, img_name1)
        img_path2 = os.path.join(input_folder, img_name2)

        img1 = Image.open(img_path1).convert('RGB').resize(image_size)
        img2 = Image.open(img_path2).convert('RGB').resize(image_size)

        # Decide randomly whether to stitch horizontally or vertically
        stitch_type = random.choice(["horizontal", "vertical"])

        if stitch_type == "horizontal":
            # Take left half of img1 + right half of img2
            half = image_size[0] // 2
            left = img1.crop((0, 0, half, image_size[1]))
            right = img2.crop((half, 0, image_size[0], image_size[1]))
            new_img = Image.new('RGB', (image_size[0], image_size[1]))
            new_img.paste(left, (0, 0))
            new_img.paste(right, (half, 0))
        else:
            # Take top half of img1 + bottom half of img2
            half = image_size[1] // 2
            top = img1.crop((0, 0, image_size[0], half))
            bottom = img2.crop((0, half, image_size[0], image_size[1]))
            new_img = Image.new('RGB', (image_size[0], image_size[1]))
            new_img.paste(top, (0, 0))
            new_img.paste(bottom, (0, half))

        # Save stitched image
        new_name = f"{img_name1.replace('.png','')}_{img_name2.replace('.png','')}_stitched{augment_count}.png"
        new_img.save(os.path.join(output_folder, new_name))

        augment_count += 1

    print(f"Created {augment_count} stitched images for {disease}.")
