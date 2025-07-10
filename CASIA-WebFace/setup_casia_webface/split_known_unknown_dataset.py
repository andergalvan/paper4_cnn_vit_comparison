import os
import shutil
from random import sample

# Input and output directories
original_dir = "/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn"
output_dir = "/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn_ordered"

# Create training, validation, CSR testing and OSR testing directories
os.makedirs(f"{output_dir}/train", exist_ok=True)
os.makedirs(f"{output_dir}/validation", exist_ok=True)
os.makedirs(f"{output_dir}/close_test", exist_ok=True)
os.makedirs(f"{output_dir}/open_test", exist_ok=True)

# Filter known identities with at least 90 images (the first 100 known people)
known_people = [person for person in os.listdir(original_dir) if len(os.listdir(os.path.join(original_dir, person))) >= 90][:100]

# Select unknown identities (no image count requirement, 1000 unknown people)
unknown_people = [person for person in os.listdir(original_dir) if person not in known_people][:1000]

# Distribute the 90 images of each known person into 70 for training, 10 for validation and 10 for  CSR testing
for person in known_people:
    
    # Distribute images
    images = os.listdir(os.path.join(original_dir, person))
    train_images = sample(images, 70)
    remaining_images = list(set(images) - set(train_images))
    val_images = sample(remaining_images, 10)
    remaining_images = list(set(remaining_images) - set(val_images))
    close_test_images = sample(remaining_images, 10)

    # Create directories
    os.makedirs(f"{output_dir}/train/{person}", exist_ok=True)
    os.makedirs(f"{output_dir}/validation/{person}", exist_ok=True)
    os.makedirs(f"{output_dir}/close_test/{person}", exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(original_dir, person, img), f"{output_dir}/train/{person}")
    for img in val_images:
        shutil.copy(os.path.join(original_dir, person, img), f"{output_dir}/validation/{person}")
    for img in close_test_images:
        shutil.copy(os.path.join(original_dir, person, img), f"{output_dir}/close_test/{person}")

# Prepare the OSR testing scenario which involves known and unknown people
os.makedirs(f"{output_dir}/open_test/unknown", exist_ok=True)
for person in known_people:
    images = sample(os.listdir(os.path.join(original_dir, person)), 10)
    os.makedirs(f"{output_dir}/open_test/{person}", exist_ok=True)
    for img in images:
        shutil.copy(os.path.join(original_dir, person, img), f"{output_dir}/open_test/{person}")
for person in unknown_people:
    images = os.listdir(os.path.join(original_dir, person))
    if len(images) > 0:  # Ensure there is at least one image
        img = sample(images, 1)[0]
        shutil.copy(os.path.join(original_dir, person, img), f"{output_dir}/open_test/unknown")
