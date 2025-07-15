import glob
import os
import random

import numpy as np
import tifffile as tiff  # We will use tifffile to open TIFF images
import torch
import torch.nn as nn
import torch.optim as optim

#from  _prof_reader import prof_proc_meta
# from UNet import UNet
from jj_nn_framework.yakub_complex_conjugate_unet import UNet
from napari_cool_tools_io._prof_reader import prof_proc_meta
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def random_horizontal_flip(image, label, p=0.5):
    """Applies random horizontal flip to both image and label."""
    if random.random() < p:
        image = torch.flip(image, dims=[-1])  # Flip horizontally
        label = torch.flip(label, dims=[-1])

    return image, label


def random_vertical_flip(image, label, p=0.5):
    """Applies random horizontal flip to both image and label."""
    if random.random() < p:
        image = torch.flip(image, dims=[-2])  # Flip vertically
        label = torch.flip(label, dims=[-2])

    return image, label


def adjust_brightness(batch, brightness):
    return torch.clamp(batch * brightness, 0, 1)


def adjust_contrast(batch, contrast):
    mean = batch.mean(dim=(-2, -1), keepdim=True)  # shape [B, C, 1, 1]
    return torch.clamp((batch - mean) * contrast + mean, 0, 1)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder):
        self.input_path = "dispersed"
        self.label_path = "normal"
        self.input_list = glob.glob(
            os.path.join(image_folder, self.input_path, "*.prof")
        )
        self.label_list = glob.glob(
            os.path.join(image_folder, self.label_path, "*.prof")
        )

        # Define the resize transformation
        self.resize_transform = transforms.Resize((512, 1024))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_name = self.input_list[idx]
        label_name = self.label_list[idx]

        """read the interpolation image"""
        meta = prof_proc_meta(input_name, ".prof")
        h, w, d, bmscan, w_param, dtype, layer_type = meta

        dot_prof = np.dtype(("<f4", (h, w)))
        input_image = np.fromfile(input_name, dtype=dot_prof, count=-1)

        """read the normal image"""
        meta = prof_proc_meta(label_name, ".prof")
        h, w, d, bmscan, w_param, dtype, layer_type = meta

        dot_prof = np.dtype(("<f4", (h, w)))
        label_image = np.fromfile(label_name, dtype=dot_prof, count=-1)

        input_image = np.flip(input_image, axis=-1)

        input_image = input_image + label_image
        input_image = input_image * 0.5

        mmin = input_image.min()
        mmax = input_image.max()
        delta = mmax - mmin

        input_image = input_image - mmin
        input_image = input_image / delta

        mmin = label_image.min()
        mmax = label_image.max()
        delta = mmax - mmin

        label_image = label_image - mmin
        label_image = label_image / delta

        input_image = torch.from_numpy(input_image)
        label_image = torch.from_numpy(label_image)

        # shuffle the image in the volume
        idx = torch.randperm(input_image.shape[0])
        idx = idx[0 : int(input_image.shape[0] // 2)]
        input_image = input_image[idx]
        label_image = label_image[idx]

        # Apply the transformation to the image
        input_image = self.resize_transform(input_image)
        label_image = self.resize_transform(label_image)

        # random flip
        # input_image, label_image = random_horizontal_flip(input_image,label_image)
        input_image, label_image = random_vertical_flip(input_image, label_image)

        # Apply color jitter only to image
        brightness = random.uniform(1.0 - 0.2, 1.0 + 0.2)
        contrast = random.uniform(1.0 - 0.2, 1.0 + 0.2)
        input_image = adjust_brightness(input_image, brightness)
        input_image = adjust_contrast(input_image, contrast)
        label_image = adjust_brightness(label_image, brightness)
        label_image = adjust_contrast(label_image, contrast)

        return input_image, label_image


chekpoint = torch.load("saved_model/unet_denoise_epoch161.pth", weights_only=False)

# Define model, loss, and optimizer
model = UNet(in_channels=1, out_channels=1)  # Assuming RGB images

model.load_state_dict(chekpoint["model_state_dict"])

model = model.cuda()

criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

optimizer.load_state_dict(chekpoint["optimizer_state_dict"])

# Set up TensorBoard writer
writer = SummaryWriter("runs/mnist_unet")  # Logs go to "runs/mnist_unet"

# Create Dataset
train_dataset = CustomDataset(image_folder="train")
validation_dataset = CustomDataset(image_folder="validation")
# test_dataset = CustomDataset(image_folder='test')

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Training loop
save_interval = 10
num_epochs = 50
batch_size = 8

epoch = 160

for ii in range(num_epochs):
    epoch = epoch + 1

    model.train()
    running_loss = 0.0
    counter = 0

    for i, (noisy_vol, clean_vol) in enumerate(train_dataloader, 0):
        noisy_vol = noisy_vol[0]
        clean_vol = clean_vol[0]

        # #shuffle the image in the volume
        # idx = torch.randperm(noisy_vol.shape[0])
        # noisy_vol = noisy_vol[idx]
        # clean_vol = clean_vol[idx]

        # expand dim for channel
        noisy_vol = torch.unsqueeze(noisy_vol, 1)
        clean_vol = torch.unsqueeze(clean_vol, 1)

        # split for batch training
        noisy_vol = torch.split(noisy_vol, batch_size)
        clean_vol = torch.split(clean_vol, batch_size)

        for noisy, clean in zip(noisy_vol, clean_vol):
            noisy, clean = noisy.cuda(), clean.cuda()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy)

            # Compute loss
            loss = criterion(outputs, clean)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Log training loss to TensorBoard
            running_loss += loss.item()
            counter = counter + 1

        print(
            f"Loss/train, {running_loss / (counter + 1)}, Epoch: {epoch},  Vol: {i} / {len(train_dataloader)}"
        )
        writer.add_scalar(
            "Loss/train",
            running_loss / (counter + 1),
            epoch * len(train_dataloader) + i,
        )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {running_loss / (counter + 1):.4f}"
    )

    # Validation phase
    model.eval()

    validation_noisy_samples = []
    validation_clean_samples = []
    validation_output_samples = []

    with torch.no_grad():
        val_loss = 0.0
        counter = 0

        for i, (noisy_vol, clean_vol) in enumerate(validation_dataloader, 0):
            noisy_vol = noisy_vol[0]
            clean_vol = clean_vol[0]

            # shuffle the image in the volume
            idx = torch.randperm(noisy_vol.shape[0])
            noisy_vol = noisy_vol[idx]
            clean_vol = clean_vol[idx]

            # expand dim for channel
            noisy_vol = torch.unsqueeze(noisy_vol, 1)
            clean_vol = torch.unsqueeze(clean_vol, 1)

            # split for batch training
            noisy_vol = torch.split(noisy_vol, batch_size)
            clean_vol = torch.split(clean_vol, batch_size)

            for noisy, clean in zip(noisy_vol, clean_vol):
                noisy, clean = noisy.cuda(), clean.cuda()

                outputs = model(noisy)
                loss = criterion(outputs, clean)
                val_loss += loss.item()
                counter = counter + 1

                # take some image samples to save
                if counter == 1 and i < 5:
                    validation_noisy_samples.append(noisy[0].cpu())
                    validation_clean_samples.append(clean[0].cpu())
                    validation_output_samples.append(outputs[0].cpu())

                break
            break

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss / (counter + 1):.4f}"
        )
        writer.add_scalar("Loss/validation", val_loss / (counter + 1), epoch)

        # Log images to TensorBoard (optional)
        if epoch % save_interval == 0:
            # # Log some input and output images
            # writer.add_images('Noisy Images', validation_noisy_samples, epoch)
            # writer.add_images('Denoised Images', validation_clean_samples, epoch)
            # writer.add_images('Clean Images', validation_output_samples, epoch)

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(state, f"saved_model/unet_denoise_epoch{epoch + 1}.pth")

            # save output to output
            for i, (noisy_img, clean_img, output_img) in enumerate(
                zip(
                    validation_noisy_samples,
                    validation_clean_samples,
                    validation_output_samples,
                )
            ):
                file_name = os.path.join(
                    "validation", "output", f"epoch_{epoch}_noisy_{i}.tif"
                )
                noisy_img = noisy_img.squeeze().numpy()
                tiff.imwrite(file_name, noisy_img)

                file_name = os.path.join(
                    "validation", "output", f"epoch_{epoch}_clean_{i}.tif"
                )
                clean_img = clean_img.squeeze().numpy()
                tiff.imwrite(file_name, clean_img)

                file_name = os.path.join(
                    "validation", "output", f"epoch_{epoch}_output_{i}.tif"
                )
                output_img = output_img.squeeze().numpy()
                tiff.imwrite(file_name, output_img)


# # Final test loss
# test_loss = 0
# with torch.no_grad():
#     for noisy, clean, names in test_dataloader:
#         noisy, clean = noisy.cuda(), clean.cuda()
#         outputs = model(noisy)
#         test_loss += criterion(outputs, clean).item()

#         #save output to output
#         for output_img, name in zip(outputs.cpu().squeeze().numpy(),names):
#             file_name = os.path.join('test','output',os.path.basename(name))
#             tiff_img = output_img*65535
#             tiff_img = tiff_img.astype(np.uint16)
#             tiff.imwrite(file_name,output_img)


# test_loss /= len(test_dataloader)
# writer.add_scalar("Loss/Test", test_loss, epoch)
# print(f"Test Loss: {test_loss:.6f}")

# # After training, log final results and close the writer
# writer.close()
