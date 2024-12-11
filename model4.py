# python imports
import os
import glob
import tempfile
import time
import warnings
from pprint import pprint
import shutil
import random

# data science imports
import matplotlib.pyplot as plt
import numpy as np

# PyTorch imports
import torch
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter

# MONAI imports
from monai.apps import extractall
from monai.data import Dataset, DataLoader, CacheDataset
from monai.losses import BendingEnergyLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.blocks import Warp
from monai.networks.nets import VoxelMorph
from monai.networks.utils import one_hot
from monai.utils import set_determinism, first
from monai.visualize.utils import blend_images
from monai.config import print_config
from monai.transforms import LoadImaged, Compose, Lambdad
print('--------------------------------------')
print('Imports done!')
print('--------------------------------------')

set_determinism(seed=0)
torch.backends.cudnn.benchmark = True

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")

#print_config() # Turn on if you want to see the configuration

print('\n \n \n \n \n')
###########################################################
################## ^ Environment setup ^ ##################
###########################################################





# Function to get files
def get_files(data_dir):
    """
    Get train/val files from the modified RADCURE dataset, 
    with keys referencing specific channels.
    """
    # List all subdirectories in the dataset directory
    subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # Sort the subfolders to maintain a consistent order (if necessary)
    subfolders.sort()

    # Split into training and validation sets manually (based on current code logic)
    train_files = []
    for subfolder in subfolders[:8]:  # First 8 subfolders for training
        subfolder_path = os.path.join(data_dir, subfolder)

        # Construct file paths for image and labels
        train_files.append(
            {
                "image": os.path.join(subfolder_path, "image.npy"),
                "label_brain_stem": os.path.join(subfolder_path, "label_brain_stem.npy"),
                "label_spinal_cord": os.path.join(subfolder_path, "label_spinal_cord.npy"),
            }
        )

    # Validation set
    val_files = []
    for i, subfolder in enumerate(subfolders[8:-1]):  # Next subfolders for validation
        next_subfolder = subfolders[8:][i + 1]  # Get the next subfolder for comparison

        # Construct file paths for fixed and moving images and labels
        fixed_folder_path = os.path.join(data_dir, subfolder)
        moving_folder_path = os.path.join(data_dir, next_subfolder)

        val_files.append(
            {
                "fixed_image": os.path.join(fixed_folder_path, "image.npy"),
                "moving_image": os.path.join(moving_folder_path, "image.npy"),
                "fixed_label_brain_stem": os.path.join(fixed_folder_path, "label_brain_stem.npy"),
                "moving_label_brain_stem": os.path.join(moving_folder_path, "label_brain_stem.npy"),
                "fixed_label_spinal_cord": os.path.join(fixed_folder_path, "label_spinal_cord.npy"),
                "moving_label_spinal_cord": os.path.join(moving_folder_path, "label_spinal_cord.npy"),
            }
        )

    return train_files, val_files

# Specify your dataset directory
data_dir = "modified_dataset"

# Create train and validation files
train_files, val_files = get_files(data_dir)

# Print 1 training sample and 1 validation sample to illustrate the contents of the datalist
pprint(train_files[0])
pprint(val_files[0])








# Image and label transforms
transform_train = LoadImaged(keys=["image", "label_brain_stem", "label_spinal_cord"], ensure_channel_first=True)
transform_val = LoadImaged(
    keys=["fixed_image", "moving_image", 
          "fixed_label_brain_stem", "moving_label_brain_stem", 
          "fixed_label_spinal_cord", "moving_label_spinal_cord"], 
    ensure_channel_first=True,
)

set_determinism(seed=0)

# Create the dataset and dataloader
check_ds = Dataset(data=train_files, transform=transform_train)
check_loader = DataLoader(check_ds, batch_size=1, shuffle=True)
check_data = first(check_loader)

# Check the structure of check_data to verify what's inside
print('Data keys: ', check_data.keys())  # Check the keys to make sure it contains 'image'
print(check_data["image"].shape)  # Check the shape of the image tensor - expect 1x1x128x128x128 for batch size 1


# Check by viewing
# Use [0] to select first in batch
image = check_data["image"][0]
label_brain_stem = check_data["label_brain_stem"][0]
label_spinal_cord = check_data["label_spinal_cord"][0]

# Extract slice 64 of first set in the batch
print('Shape of image: ', image.shape)
image = image.permute(1, 2, 3, 0)[64, :, :, :] # Permute puts singleton grayscale channel at the end to allow plotting
label_brain_stem = label_brain_stem.permute(1, 2, 3, 0)[64, :, :, :]
label_spinal_cord = label_spinal_cord.permute(1, 2, 3, 0)[64, :, :, :]

fig, axs = plt.subplots(1, 3)
axs[0].imshow(image, cmap="gray")
axs[0].title.set_text("Image")
axs[1].imshow(label_brain_stem, cmap="gray")
axs[1].title.set_text("Brain stem")
axs[2].imshow(label_spinal_cord, cmap="gray")
axs[2].title.set_text("Spinal cord")

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

plt.suptitle("Image and label visualization")
plt.tight_layout()
plt.show()




# Set hyperparameters:
# device, optimizer, epoch and batch settings
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
lr = 1e-4
weight_decay = 1e-5
max_epochs = 10

# Use mixed precision feature of GPUs for faster training
amp_enabled = True

# loss weights (set to zero to disable loss term)
lam_sim = 1e0  # MSE (image similarity)
lam_smooth = 1e-2  # Bending loss (smoothness)
lam_dice = 2e-2  # Dice (auxiliary)

#  Write model and tensorboard logs?
do_save = True
dir_save = os.path.join(os.getcwd(), "models", "voxelmorph")
if do_save and not os.path.exists(dir_save):
    os.makedirs(dir_save)

print('Successfully set hyperparameters')




# Create custom forward pass
def forward(fixed_image, moving_image, moving_label, model, warp_layer, num_classes):
    """
    Model forward pass:
        - predict DDF,
        - convert moving label to one-hot format, and
        - warp one-hot encoded moving label using predicted DDF
    """

    # predict DDF and warp moving image using predicted DDF
    pred_image, ddf_image = model(moving_image, fixed_image)

    # warp moving label
    # num_classes + 1 to include background as a channel
    moving_label_one_hot = one_hot(moving_label, num_classes=num_classes + 1)
    pred_label_one_hot = warp_layer(moving_label_one_hot, ddf_image)

    return ddf_image, pred_image, pred_label_one_hot

print('Successfully created a forward pass')





# Define a flexible multi-target loss function
def loss_fun(
    fixed_image,
    pred_image,
    fixed_label,
    pred_label_one_hot,
    ddf_image,
    lam_sim,
    lam_smooth,
    lam_dice,
):
    """
    Custom multi-target loss:
        - Image similarity: MSELoss
        - Deformation smoothness: BendingEnergyLoss
        - Auxiliary loss: DiceLoss
    """
    # Instantiate where necessary
    if lam_sim > 0:
        mse_loss = MSELoss()
    if lam_smooth > 0:
        regularization = BendingEnergyLoss()
    if lam_dice > 0:
        # we exclude the first channel (i.e., background) when calculating dice
        label_loss = DiceLoss(include_background=False)

    num_classes = 2 # Change depending on number of classes
    # If you are using one label, e.g. the brain stem, we have 2 classes - background and brain_stem
    # If you are using two labels, e.g. brain stem and spinal cord, we have 3 classes - background, brain stem and spinal cord

    # Compute loss components
    sim = mse_loss(pred_image, fixed_image) if lam_sim > 0 else 0
    smooth = regularization(ddf_image) if lam_smooth > 0 else 0
    dice = label_loss(pred_label_one_hot, one_hot(fixed_label, num_classes=num_classes + 1)) if lam_dice > 0 else 0

    # Weighted combination:
    return lam_sim * sim + lam_smooth * smooth + lam_dice * dice

print('Successfully defined loss_fun()')




# Define CacheDataset and DataLoader for training and validation:

# Cached datasets for high performance during batch generation
# train_ds = CacheDataset(data=train_files, transform=transform_train, cache_rate=1.0, num_workers=1)
# val_ds = CacheDataset(data=val_files, transform=transform_val, cache_rate=1.0, num_workers=1)
train_ds = Dataset(data=train_files, transform=transform_train)
val_ds = Dataset(data=val_files, transform=transform_val)
print('Successfully ran Dataset() for train and val files')

# By setting batch_size=2 * batch_size, we randomly sample two images for each training iteration from
# the training set. During training, we manually split along the batch dimension to obtain the fixed
# and moving images.
train_loader = DataLoader(train_ds, batch_size=2 * batch_size, shuffle=True, num_workers=1)
print('Successfully ran train_loader')

# We obtain one sample for each validation iteration since the validation set is already arranged
# into pairs of fixed and moving images.
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)
print('Successfully ran val_loader')





# Create model/optimizer/metrics

# Model
model = VoxelMorph().to(device)
warp_layer = Warp().to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# Metrics
dice_metric_before = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_metric_after = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
print('Created model, warp layer, oltimzer, scheduler, dice metrics')




###################################################################################################
###################################################################################################
###################################################################################################
# Execute a typical PyTorch training process
print('Starting training')

# Automatic mixed precision (AMP) for faster training
amp_enabled = True
scaler = torch.cuda.amp.GradScaler()

# Tensorboard
if do_save:
    writer = SummaryWriter(log_dir=dir_save)

# Start torch training loop
val_interval = 1
best_eval_dice = 0
log_train_loss = []
log_val_dice = []
pth_best_dice, pth_latest = "", ""

for epoch in range(max_epochs):
    # ==============================================
    # Train
    # ==============================================
    model.train()

    epoch_loss, n_steps = 0, 0
    t0_train = time.time()
    for batch_data in train_loader:
        # for batch_data in tqdm(train_loader):
        # Get data: manually slicing along the batch dimension to obtain the fixed and moving images
        fixed_image = batch_data["image"][0:1, ...].to(device)
        moving_image = batch_data["image"][1:, ...].to(device)
        fixed_label = batch_data["label_brain_stem"][0:1, ...].to(device)
        moving_label = batch_data["label_brain_stem"][1:, ...].to(device)
        n_steps += 1

        # Forward pass and loss
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            ddf_image, pred_image, pred_label_one_hot = forward(
                fixed_image, moving_image, moving_label, model, warp_layer, num_classes=2 # Make sure to set num_classes
            )
            loss = loss_fun(
                fixed_image,
                pred_image,
                fixed_label,
                pred_label_one_hot,
                ddf_image,
                lam_sim,
                lam_smooth,
                lam_dice,
            )
        # Optimise
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    # Scheduler step
    lr_scheduler.step()
    # Loss
    epoch_loss /= n_steps
    log_train_loss.append(epoch_loss)
    if do_save:
        writer.add_scalar("train_loss", epoch_loss, epoch)
    print(f"{epoch + 1} | loss = {epoch_loss:.6f} " f"elapsed time: {time.time()-t0_train:.2f} sec.")
    # ==============================================
    # Eval
    # ==============================================
    if (epoch + 1) % val_interval == 0:
        model.eval()

        n_steps = 0
        with torch.no_grad():
            for batch_data in val_loader:
                # Get data
                fixed_image = batch_data["fixed_image"].to(device)
                moving_image = batch_data["moving_image"].to(device)
                fixed_label_brain_stem = batch_data["fixed_label_brain_stem"].to(device)
                moving_label_brain_stem = batch_data["moving_label_brain_stem"].to(device)
                n_steps += 1
                # Infer
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    ddf_image, pred_image, pred_label_one_hot = forward(
                        fixed_image, moving_image, moving_label_brain_stem, model, warp_layer, num_classes=2 # Make sure to set num_classes
                    )
                # Dice
                dice_metric_before(y_pred=moving_label_brain_stem, y=fixed_label_brain_stem)
                dice_metric_after(y_pred=pred_label_one_hot.argmax(dim=1, keepdim=True), y=fixed_label_brain_stem)
                # Note: DiceMetric does work with one-hot encoded inputs. However, recall that when we
                # defined the forward pass, we first converted the discrete moving label into one-hot
                # format, and then warp the one-hot encoded moving label (which by default uses
                # bilinear interpolation). Passing a non-binary one-hot label as y_pred is not
                # supported by DiceMetric. DiceHelper does, but we are not using DiceHelper since it
                # does not inherit from CumulativeIterationMetric and is, thus, unable to accumulate
                # statistics.

                # # uncomment to show a pair of fixed and moved image at each validation
                if n_steps == 1:
                    fig, axs = plt.subplots(1, 2)
                    fixed = fixed_image.cpu().squeeze()[64, :, :]
                    axs[0].imshow(fixed, cmap="gray")
                    axs[0].title.set_text("Fixed")
                    moving = pred_image.detach().cpu().squeeze()[64, :, :]
                    axs[1].imshow(moving, cmap="gray")
                    axs[1].title.set_text("Moved")
                    plt.show()

        # Dice
        dice_before = dice_metric_before.aggregate().item()
        dice_metric_before.reset()
        dice_after = dice_metric_after.aggregate().item()
        dice_metric_after.reset()
        if do_save:
            writer.add_scalar("val_dice", dice_after, epoch)
        log_val_dice.append(dice_after)
        print(f"{epoch + 1} | dice_before = {dice_before:.3f}, dice_after = {dice_after:.3f}")

        if dice_after > best_eval_dice:
            best_eval_dice = dice_after
            if do_save:
                # Save best model based on Dice
                if pth_best_dice != "":
                    os.remove(os.path.join(dir_save, pth_best_dice))
                pth_best_dice = f"voxelmorph_loss_best_dice_{epoch + 1}_{best_eval_dice:.3f}.pth"
                torch.save(model.state_dict(), os.path.join(dir_save, pth_best_dice))
                print(f"{epoch + 1} | Saving best Dice model: {pth_best_dice}")

    if do_save:
        # Save latest model
        if pth_latest != "":
            os.remove(os.path.join(dir_save, pth_latest))
        pth_latest = "voxelmorph_loss_latest.pth"
        torch.save(model.state_dict(), os.path.join(dir_save, pth_latest))

print('TRAINING COMPLETE')



fig, axs = plt.subplots(2, 1, figsize=(4, 6))
axs[0].plot(log_train_loss)
# axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Training Loss")
axs[1].plot(log_val_dice)
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Dice Score on Validation Set")
plt.suptitle(f"VoxelMorph\n(Smoothness weight: {lam_smooth}, Dice weight: {lam_dice})")
plt.tight_layout()
plt.show()
