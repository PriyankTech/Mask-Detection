import os
import shutil
import random
from tqdm import tqdm

# ===========================
# CONFIGURE DIRECTORIES
# ===========================
LOCAL_IMAGE_DIR = "./Images"
 # Path to your local image directory

BASE_DIR = "/content/dataset"  # Destination base directory in your project
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

categories = ["Masked", "No Mask", "Incorrect"]
train_split = 0.8  # 80% for training, 20% for validation

# ===========================
# CREATE NECESSARY DIRECTORIES
# ===========================
def create_directories():
    for category in categories:
        train_category_dir = os.path.join(TRAIN_DIR, category)
        if not os.path.exists(train_category_dir):
            os.makedirs(train_category_dir)
            print(f"📁 Created: {train_category_dir}")

    if not os.path.exists(VAL_DIR):
        os.makedirs(VAL_DIR)
        print(f"📁 Created: {VAL_DIR}")

create_directories()

# ===========================
# UPLOAD IMAGES
# ===========================
def upload_images():
    for category in categories:
        source_dir = os.path.join(LOCAL_IMAGE_DIR, category)

        if not os.path.exists(source_dir):
            print(f"❌ Source directory not found: {source_dir}")
            continue
        
        images = os.listdir(source_dir)
        random.shuffle(images)

        # Split images into train and val
        split_index = int(len(images) * train_split)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Copy images to train directory (with class subfolders)
        for img in tqdm(train_images, desc=f"Uploading {category} to train"):
            src_path = os.path.join(source_dir, img)
            dest_path = os.path.join(TRAIN_DIR, category, img)
            shutil.copy(src_path, dest_path)
        
        # Copy images to val directory (flat structure)
        for img in tqdm(val_images, desc=f"Uploading {category} to val"):
            src_path = os.path.join(source_dir, img)
            dest_path = os.path.join(VAL_DIR, img)
            shutil.copy(src_path, dest_path)

upload_images()
