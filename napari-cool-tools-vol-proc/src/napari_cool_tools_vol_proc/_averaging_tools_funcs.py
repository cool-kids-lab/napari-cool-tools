"""
This module contains code for averaging 2D slices
"""

from enum import Enum

from napari_cool_tools_oct_preproc.__init__ import OCTACalc
import numpy as np
from napari.types import ImageData
from napari.utils.notifications import show_info
from napari_cool_tools_io import torch
from skimage.measure import block_reduce
from tqdm import tqdm
from napari_cool_tools_io import device
from napari_cool_tools_registration.__init__ import RegistrationMode
# from napari_cool_tools_registration._registration_tools import register_frame_2d
from skimage.registration import optical_flow_ilk, phase_cross_correlation
from skimage.transform import warp
from scipy.ndimage import shift as ndimage_shift
import cv2
from napari_cool_tools_oct_preproc._oct_preproc_func import apply_spatial_axial_averaging, generate_octa


class Implementation(Enum):
    NUMPY = "Numpy"
    TORCH = "Pytorch"
    KORNIA = "Kornia"
    CUPY = "Cupy"


def average_bscans(vol: ImageData, scans_per_avg: int = 3) -> ImageData:
    """Function averaging every scans_per_avg images/B-scans togehter.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        ImageData volume where values have been averaged every scans_per_avg images/B-scans along the depth dimension
    """

    averaged_array = block_reduce(vol, block_size=(scans_per_avg, 1, 1), func=np.mean)

    return averaged_array


def average_bscans_torch(vol: ImageData, scans_per_avg: int = 3) -> ImageData:
    """Function averaging every scans_per_avg images/B-scans together using PyTorch.
    Args:
        vol (ImageData): vol representing volumetric or image stack data (Z, Y, X)
        scans_per_avg (int): number of consecutive images/B-scans to average together

    Returns:
        ImageData volume averaged every scans_per_avg images/B-scans along the depth dimension
    """
    if scans_per_avg <= 0:
        raise ValueError("scans_per_avg must be a positive integer")

    #vol_t = torch.as_tensor(vol.copy()).to(device)
    vol_t = torch.as_tensor(vol).to(device)
    z, y, x = vol_t.shape

    if z % scans_per_avg != 0:
        raise ValueError(
            f"Depth dimension ({z}) must be divisible by scans_per_avg ({scans_per_avg})"
        )

    out = (
        vol_t.view(z // scans_per_avg, scans_per_avg, y, x)
        .mean(dim=1)
        .cpu()
        .numpy()
    )

    return out


def average_per_bscan(
    vol: ImageData, scans_per_avg: int = 3, axis=0, trim: bool = False
) -> ImageData:
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim: (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)

    Returns:
        ImageData volume where values at each index each slice is an average of the surrounding bscans from vol
    """

    if scans_per_avg % 2 == 1:
        offset = int((scans_per_avg - 1) / 2)

        length = vol.shape[axis]

        averaged_slices = []

        for i in tqdm(range(length), desc="Avg per B-scan"):
            if i >= offset and i < length - offset:
                # print(f"Averaging slices...\nGenerating new slice by averaging slices {i-offset} through {i+offset} of {length-1}")

                if axis == 0:
                    start0 = i - offset
                    end0 = i + offset + 1
                    start1 = 0
                    end1 = vol.shape[1]
                    start2 = 0
                    end2 = vol.shape[2]
                elif axis == 1:
                    start0 = 0
                    end0 = vol.shape[0]
                    start1 = i - offset
                    end1 = i + offset + 1
                    start2 = 0
                    end2 = vol.shape[2]
                elif axis == 2:
                    start0 = 0
                    end0 = vol.shape[0]
                    start1 = 0
                    end1 = vol.shape[1]
                    start2 = i - offset
                    end2 = i + offset + 1
                else:
                    print("You done effed up!!")

                averaged_slice = vol[start0:end0, start1:end1, start2:end2].mean(axis)

                # averaged_slice = data[i-offset:i+offset+1,:,:].mean(axis)
                averaged_slices.append(averaged_slice)
            else:
                if not trim:
                    if i < offset:
                        if axis == 0:
                            start0 = i
                            end0 = i + 1
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 1:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = i
                            end1 = i + 1
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 2:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = i
                            end2 = i + 1
                        else:
                            print("You done effed up!!")

                        averaged_slice = vol[
                            start0:end0, start1:end1, start2:end2
                        ].squeeze(axis)

                        averaged_slices.append(averaged_slice)

                        pass

                    elif i >= length - offset:
                        if axis == 0:
                            start0 = i
                            end0 = i + 1
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 1:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = i
                            end1 = i + 1
                            start2 = 0
                            end2 = vol.shape[2]
                        elif axis == 2:
                            start0 = 0
                            end0 = vol.shape[0]
                            start1 = 0
                            end1 = vol.shape[1]
                            start2 = i
                            end2 = i + 1
                        else:
                            print("You done effed up!!")

                        averaged_slice = vol[
                            start0:end0, start1:end1, start2:end2
                        ].squeeze(axis)

                        averaged_slices.append(averaged_slice)

                        pass
                else:
                    # print(f"You shouldn't be here {average_per_bscan}!!")
                    pass

        averaged_array = np.stack(averaged_slices, axis=axis)

        return averaged_array
    else:
        print(
            "scans_per_avg should be an odd number please use an odd number for this value"
        )
        show_info(
            "scans_per_avg should be an odd number please use an odd number for this value"
        )
        return vol


def optical_flow_registration_cv(
    target_image: np.ndarray, moving_image: np.ndarray
) -> np.ndarray:

    # print(target_image.shape, moving_image.shape)

    # 1. Upload the grayscale numpy arrays to GPU memory
    gpu_target = cv2.cuda.GpuMat()
    gpu_moving = cv2.cuda.GpuMat()
    gpu_target.upload(target_image)
    gpu_moving.upload(moving_image)

    # #min max normalization of the images to 0-255 range for better optical flow results
    # gpu_target_u8 = cv2.cuda.normalize(gpu_target, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    # gpu_moving_u8 = cv2.cuda.normalize(gpu_moving, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # 2. Compute Dense Optical Flow entirely on the GPU
    # Returns a 2-channel GpuMat containing (dx, dy)
    # Allocate a GpuMat for the output flow map
    gpu_flow = cv2.cuda.GpuMat(gpu_target.size(), cv2.CV_32FC2)  # 2-channel float32 for flow

    # 2. Compute dense optical flow on the GPU
    # cuda_flow_algo = cv2.cuda.DensePyrLKOpticalFlow.create(winSize=(13, 13), iters=10)
    cuda_flow_algo = cv2.cuda.FarnebackOpticalFlow.create(
    numLevels=5,
    pyrScale=0.5,
    fastPyramids=False,
    winSize=13,
    numIters=10,
    polyN=5,
    polySigma=1.1,
    flags=0
    )
    
    cuda_flow_algo.calc(gpu_target, gpu_moving, gpu_flow)

    row, col = target_image.shape
    grid_y, grid_x = np.mgrid[0:row, 0:col].astype(np.float32)

    grid_x = grid_x.astype(np.float32)
    grid_y = grid_y.astype(np.float32)

    # 3. Upload the meshgrid to GPU memory
    gpu_grid_x = cv2.cuda.GpuMat(grid_x.shape, cv2.CV_32FC1)
    gpu_grid_y = cv2.cuda.GpuMat(grid_y.shape, cv2.CV_32FC1)
    gpu_grid_x.upload(grid_x)
    gpu_grid_y.upload(grid_y)

    # 4. Split the 2-channel flow vector into separate X and Y channels
    gpu_dx = cv2.cuda.GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    gpu_dy = cv2.cuda.GpuMat(gpu_flow.size(), cv2.CV_32FC1)
    cv2.cuda.split(gpu_flow, [gpu_dx, gpu_dy]) #

    # 5. Map offsets to absolute coordinates on the GPU VRAM
    # FIX: Adding flow to grid maps moving to target coordinates
    gpu_map_x = cv2.cuda.GpuMat(gpu_grid_x.size(), cv2.CV_32FC1)
    gpu_map_y = cv2.cuda.GpuMat(gpu_grid_y.size(), cv2.CV_32FC1)
    gpu_registered = cv2.cuda.GpuMat(gpu_moving.size(), gpu_moving.type())

    # 6. Execute in-place (no variables returned or reassigned)
    cv2.cuda.add(gpu_grid_x, gpu_dx, gpu_map_x)
    cv2.cuda.add(gpu_grid_y, gpu_dy, gpu_map_y)
    cv2.cuda.remap(gpu_moving, gpu_map_x, gpu_map_y, cv2.INTER_LINEAR, dst=gpu_registered)

    registered_image = gpu_registered.download()

    # 7. Download the final registered image back to a CPU numpy array
    return registered_image


def kernel_average_pt(
    vol: np.ndarray,
    slow_window: int = 3,  # lateral window size in z direction
    fast_window: int = 3,  # lateral window size
    axial_window: int = 3,  # axial window size
) -> np.ndarray:
    """Function to perform kernel averaging on a volume.
    Args:
        vol (np.ndarray): vol representing volumetric or image stack data
        slow_window (int): lateral window size in z direction
        fast_window (int): lateral window size
        axial_window (int): axial window size

    Returns:
        np.ndarray: volume after kernel averaging
    """
    vol_t = torch.as_tensor(vol, device=device)
    if not torch.is_floating_point(vol_t):
        vol_t = vol_t.float()
    pad = (
        axial_window // 2,
        axial_window // 2,
        fast_window // 2,
        fast_window // 2,
        slow_window // 2,
        slow_window // 2,
    )
    vol_padded = torch.nn.functional.pad(vol_t.unsqueeze(0).unsqueeze(0), pad, mode="reflect")
    kernel = torch.ones(
        (1, 1, slow_window, fast_window, axial_window),
        device=vol_padded.device,
        dtype=vol_padded.dtype,
    ) / float(slow_window * fast_window * axial_window)
    averaged_vol = torch.nn.functional.conv3d(vol_padded, kernel, stride=1)
    return averaged_vol.squeeze(0).squeeze(0).cpu().numpy()

def average_per_bscan_pt(
    vol: np.ndarray,
    scans_per_avg: int = 3,
    registration: bool = False,
    registration_mode: RegistrationMode = RegistrationMode.Phase_Correlation,
    OCTA: bool = False,
    octa_mode: OCTACalc = OCTACalc.STD,
    kernel_averaging: bool = False,
    slow_window: int = 3, #lateral window size in z direction
    fast_window: int = 3, #lateral window size
    axial_window: int = 3, #axial window size
    sigma_slow: float = 0.0,
    sigma_fast: float = 0.0,
    sigma_axial: float = 0.0,
    enable_axial_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:


    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (np.ndarray): vol representing volumetric or image stack data
    """

    vol_t = vol.copy()
    buffer = int(scans_per_avg / 2) #5 -> 2

    # buffer_vol = torch.zeros((scans_per_avg, vol_t.shape[1], vol_t.shape[2]), device=vol_t.device, dtype=vol_t.dtype)
    # calc indicies
    axis_len = vol_t.shape[0]
    buffer_idxs = np.arange(0, scans_per_avg, dtype=np.int64) # [0, 1, 2, 3, 4] for scans_per_avg = 5

    avg_out = np.zeros_like(vol_t)
    octa_out = np.zeros_like(vol_t) if OCTA else None

    for i in range(axis_len):
        idxs = i - buffer + buffer_idxs 
        #clamp the indices to be within the valid range of the volume
        idxs = np.clip(idxs, 0, axis_len - 1)
        # idxs = idxs % axis_len  # wrap around for circular indexing
        buffer_vol = vol_t[idxs].copy()  # shape: (scans_per_avg, height, width)


        #do some registration here if needed based on the registration parameter
        if registration:
            if registration_mode == RegistrationMode.Phase_Correlation:

                print(f"Performing registration for buffer centered at index {i} with target index {buffer} using {registration_mode.value}")

                #always use the middle image as reference for registration
                target_idx = buffer
                target_image = buffer_vol[target_idx]
                moving_images = np.concatenate([buffer_vol[:target_idx], buffer_vol[target_idx + 1:]], axis=0)

                for j in range(moving_images.shape[0]):
                    moving_image = moving_images[j]
                    #Calculate the translation using phase cross-correlation
                    detected_shift, _, _ = phase_cross_correlation(target_image, moving_image, upsample_factor=10)

                    corrected_image = ndimage_shift(moving_image, detected_shift, mode='constant', cval=0)

                    moving_images[j] = corrected_image

                #copy back the registered images into the buffer_vol
                buffer_vol[:target_idx] = moving_images[:target_idx]
                buffer_vol[target_idx + 1:] = moving_images[target_idx:]

            elif registration_mode == RegistrationMode.Optical_Flow:
                print(f"Performing registration for buffer centered at index {i} with target index {buffer} using {registration_mode.value}")

                #always use the middle image as reference for registration
                target_idx = buffer
                target_image = buffer_vol[target_idx]
                moving_images = np.concatenate([buffer_vol[:target_idx], buffer_vol[target_idx + 1:]], axis=0)

                # Create a meshgrid for pixel coordinates
                row, col = target_image.shape
                row_coords, col_coords = np.meshgrid(
                    np.arange(row), np.arange(col), indexing="ij"
                )
            
                # Use the optical flow registration function
                for j in range(moving_images.shape[0]):

                    moving_image = moving_images[j]
                    # Calculate the optical flow
                    v, u = optical_flow_ilk(target_image, moving_image)

                    registered_image = warp(
                        moving_image, np.array([row_coords + v, col_coords + u]), mode="edge"
                    )

                    # moving_image = moving_images[j]
                    # registered_image = optical_flow_registration_cv(target_image, moving_image)
                    moving_images[j] = registered_image

                #copy back the registered images into the buffer_vol
                buffer_vol[:target_idx] = moving_images[:target_idx]
                buffer_vol[target_idx + 1:] = moving_images[target_idx:]

            elif registration_mode == RegistrationMode.Combination:
                print(f"Performing registration for buffer centered at index {i} with target index {buffer} using {registration_mode.value}")

                #always use the middle image as reference for registration
                target_idx = buffer
                target_image = buffer_vol[target_idx]
                moving_images = np.concatenate([buffer_vol[:target_idx], buffer_vol[target_idx + 1:]], axis=0)

                # Create a meshgrid for pixel coordinates
                row, col = target_image.shape
                row_coords, col_coords = np.meshgrid(
                    np.arange(row), np.arange(col), indexing="ij"
                )
            
                # Use the optical flow registration function
                for j in range(moving_images.shape[0]):

                    moving_image = moving_images[j]

                    # First, try phase correlation
                    detected_shift, _, _ = phase_cross_correlation(target_image, moving_image, upsample_factor=10)
                    corrected_image = ndimage_shift(moving_image, detected_shift, mode='constant', cval=0)

                    # Calculate the optical flow
                    registered_image = optical_flow_registration_cv(target_image, corrected_image)

                    moving_images[j] = registered_image

                #copy back the registered images into the buffer_vol
                buffer_vol[:target_idx] = moving_images[:target_idx]
                buffer_vol[target_idx + 1:] = moving_images[target_idx:]
                

            else:
                raise ValueError(f"Unsupported registration mode: {registration_mode}")

        if kernel_averaging:
            print(f"Performing kernel averaging for buffer centered at index {i} with window sizes (slow: {slow_window}, fast: {fast_window}, axial: {axial_window})")
            # buffer_vol = kernel_average_pt(buffer_vol, slow_window=slow_window, fast_window=fast_window, axial_window=axial_window)

            buffer_vol = apply_spatial_axial_averaging(buffer_vol, slow_window=slow_window, fast_window=fast_window, axial_window=axial_window, 
                                                       sigma_slow=sigma_slow, sigma_fast=sigma_fast, sigma_axial=sigma_axial, enable_axial_filter=enable_axial_filter,
                                                       return_as_numpy=True)


            #don't do mean. take the middle slice of the kernel averaged volume as the output for this index
            mid_idx = buffer_vol.shape[0] // 2
            avg_out[i] = buffer_vol[mid_idx]
        else:
            print(f"Skipping kernel averaging for buffer centered at index {i}")
            avg_out[i] = buffer_vol.mean(axis=0)

        if OCTA:
            print(f"Generating OCTA for buffer centered at index {i} using {octa_mode.value}")
            octa_out[i] = generate_octa(buffer_vol, mscans=len(buffer_vol),calc=octa_mode)

    return avg_out , octa_out


