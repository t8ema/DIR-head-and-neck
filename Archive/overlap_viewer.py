# Used to check if structures have any overlap

import os
import matplotlib.pyplot as plt
import numpy as np

data_dir = os.path.join(os.path.dirname(__file__), "modified_dataset")

# Choose folder
example_folder = os.path.join(data_dir, 'RADCURE-0007')

# Load the saved files for visualization
image = np.load(os.path.join(example_folder, "image.npy"))
label_2 = np.load(os.path.join(example_folder, "label_2.npy"))

# Combine the masks (assign distinct values to different structures)
combined_mask = np.zeros(label_2[0].shape, dtype=np.uint8)
combined_mask[label_2[0] > 0] = 1  # Brain Stem (value = 1)
combined_mask[label_2[1] > 0] = 2  # Spinal Cord (value = 2)

# Display one slice (e.g., slice 64)
slice_idx = 70
plt.figure(figsize=(10, 5))

# First subplot: Original CT Image
plt.subplot(1, 2, 1)
plt.imshow(image[slice_idx], cmap='gray')
plt.title("CT Image (Slice 64)")
plt.axis('off')

# Second subplot: Combined Mask
plt.subplot(1, 2, 2)
plt.imshow(combined_mask[slice_idx], cmap='tab20', alpha=0.7)  # Different colors for values 1 and 2
plt.title("Brain Stem & Spinal Cord (Slice 64)")
plt.axis('off')

# Add legend (optional)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='tab:blue', label='Brain Stem'),
    Patch(facecolor='tab:orange', label='Spinal Cord')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
