# DIR-head-and-neck

Processed 3-channel (CT scan, brainstem mask, spinal cord mask) numpy files are expected to be placed in the dataset folder

data_prep.py creates a new folder called modified_dataset (if it doesn't exist already) and extracts the aforementioned data into a folder for each file, which contains the ct scan, brainstem mask and spinal cord mask as separate numpy files.

model.py then runs a neural network on the data (adapted from source: https://github.com/Project-MONAI/tutorials/blob/main/3d_registration/learn2reg_oasis_unpaired_brain_mr.ipynb)
