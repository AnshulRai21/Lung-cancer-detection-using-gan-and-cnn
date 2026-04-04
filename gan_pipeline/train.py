# ==============================
# STEP 1: Create GAN structure
# ==============================

import os
import shutil
from pathlib import Path

base_dir = "/content/Lung-cancer-detection-using-gan-and-cnn"
data_dir = f"{base_dir}/uploads"

input_dir = f"{data_dir}/input"
target_dir = f"{data_dir}/target"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

# ==============================
# STEP 2: Collect images from dataset
# ==============================

source_dir = f"{base_dir}/uploads/Data/train"

count = 0

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(root, file)
            
            # Copy to input
            dst_input = os.path.join(input_dir, f"{count}.png")
            shutil.copy(src, dst_input)

            # Copy to target (same image for now)
            dst_target = os.path.join(target_dir, f"{count}.png")
            shutil.copy(src, dst_target)

            count += 1

print(f"Total images prepared: {count}")

# ==============================
# STEP 3: Verify structure
# ==============================

print("Input samples:", len(os.listdir(input_dir)))
print("Target samples:", len(os.listdir(target_dir)))

# ==============================
# STEP 4: Install dependencies
# ==============================

!pip install -r /content/Lung-cancer-detection-using-gan-and-cnn/requirements.txt

# ==============================
# STEP 5: Run training
# ==============================

%cd /content/Lung-cancer-detection-using-gan-and-cnn

!python -m gan_pipeline.train \
--data-dir uploads \
--epochs 3 \
--batch-size 4 \
--image-size 128 \
--checkpoint-dir models \
--visual-dir static/outputs
