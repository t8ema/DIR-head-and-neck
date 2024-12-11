### FOLDER STRUCTURE ###

# Dissertation-repo
#   saved_models
#   dataset
#       RADCURE-0005.npy
#       RADCURE-0007.npy
#       RADCURE-0009.npy
#       RADCURE-0010.npy
#       RADCURE-0012.npy
#       ...
#   test
#   train
#   data_prep.py
#   model_test.py



# Setup imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Set the data directory
data_dir = os.path.join(os.path.dirname(__file__), "dataset")
modified_data_dir = os.path.join(os.path.dirname(__file__), "modified_dataset")

# Ensure the modified dataset directory exists
if not os.path.exists(modified_data_dir):
    os.makedirs(modified_data_dir)

# Load and print all .npy files for debugging
files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
print(f"Found {len(files)} files in the dataset folder.")



# Process each file
for file in files:
    filepath = os.path.join(data_dir, file)
    data = np.load(filepath)  # Shape: (3, 128, 128, 128)

    # Validate the shape
    if data.shape != (3, 128, 128, 128):
        print(f"Skipping {file} due to unexpected shape: {data.shape}")
        continue

    # Extract channels
    image = data[0]  # CT scan
    label_brain_stem = data[1]  # Brain stem mask
    label_spinal_cord = data[2]  # Spinal cord mask

    # Create a folder for this file
    folder_name = os.path.splitext(file)[0]  # Remove .npy suffix
    folder_path = os.path.join(modified_data_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the subfiles
    np.save(os.path.join(folder_path, "image.npy"), image)
    np.save(os.path.join(folder_path, "label_brain_stem.npy"), label_brain_stem)
    np.save(os.path.join(folder_path, "label_spinal_cord.npy"), label_spinal_cord)

    print(f"Processed and saved files for {file}.")
