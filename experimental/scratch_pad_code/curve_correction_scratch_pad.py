""""""
import gc
from datetime import datetime

import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import map_coordinates
from skimage.measure import block_reduce
from tqdm import tqdm


def spherical2cartesian_chunked(
    r, thx, thy, grid, x, y, z, sweep_angle=140, order=3, chunk_size=64
):
    # log_time("Starting spherical to Cartesian conversion")

    # Prepare the output array
    cartesian_shape = (len(x), len(y), len(z))
    output = np.zeros(cartesian_shape, dtype=grid.dtype)

    # Iterate over chunks along the z-axis
    # for z_start in range(0, len(z), chunk_size):
    for z_start in tqdm(range(0, len(z), chunk_size), desc="Processing", unit="chunk"):
        z_end = min(z_start + chunk_size, len(z))
        # log_time(f"Processing chunk: z[{z_start}:{z_end}]")
        # log_time(f"% conversion processing: {round(100*z_end/len(z),1)}% ")
        # Create the chunk-specific z-axis slice{
        Z = z[z_start:z_end][None, None, :]  # Shape: (1, 1, z_chunk_size)

        # Broadcast X and Y to match the chunk size
        # X = x[:, None, None]  # Shape: (len(x), 1, 1)
        # Y = y[None, :, None]  # Shape: (1, len(y), 1)
        X = x[:, None, None].repeat(Z.shape[-1], axis=-1)  # Broadcast X
        Y = y[None, :, None].repeat(Z.shape[-1], axis=-1)

        X, Y, Z = cp.broadcast_arrays(X, Y, Z)
        # Compute spherical coordinates for this chunk
        new_r = cp.sqrt(X**2 + Y**2 + Z**2).astype(cp.float16)
        new_thx = cp.arctan2(X, Z).astype(cp.float16)
        new_thy = cp.arctan2(Y, Z).astype(cp.float16)

        # Interpolate on GPU
        new_ir = cp.interp(new_r.ravel(), r, cp.arange(len(r)))
        new_ithx = cp.interp(
            new_thx.ravel(), thx, cp.arange(len(thx)), period=2 * np.pi
        )
        new_ithy = cp.interp(
            new_thy.ravel(), thy, cp.arange(len(thy)), period=2 * np.pi
        )

        # Compute valid indices for this chunk
        valid_mask = (new_r.ravel() <= r.max()) & (new_r.ravel() >= r.min())
        valid_mask &= (new_thx.ravel() <= thx.max()) & (new_thx.ravel() >= thx.min())
        valid_mask &= (new_thy.ravel() <= thy.max()) & (new_thy.ravel() >= thy.min())

        if valid_mask.any():
            # Get the bounding indices for the valid region
            r_min, r_max = (
                int(new_ir[valid_mask].min()),
                int(new_ir[valid_mask].max()) + 1,
            )
            thetax_indxsin, thetax_indxsax = (
                int(new_ithx[valid_mask].min()),
                int(new_ithx[valid_mask].max()) + 1,
            )
            thetay_indxsin, thetay_indxsax = (
                int(new_ithy[valid_mask].min()),
                int(new_ithy[valid_mask].max()) + 1,
            )

            # Slice the grid to the valid region
            grid_slice = cp.asarray(grid[r_min:r_max, thetax_indxsin:thetax_indxsax, thetay_indxsin:thetay_indxsax])
            # Adjust indices to fit within the local grid slice
            local_ir = new_ir[valid_mask] - r_min
            local_ithx = new_ithx[valid_mask] - thetax_indxsin
            local_ithy = new_ithy[valid_mask] - thetay_indxsin

            # Map coordinates in the local grid slice
            valid_points = cp.array([local_ir, local_ithx, local_ithy])
            interpolated = map_coordinates(grid_slice, valid_points, order=order)

            # Place interpolated values in the output array for this chunk
            interpolated_chunk = cp.zeros_like(new_r.ravel(), dtype=grid.dtype)
            interpolated_chunk[valid_mask] = interpolated
            output[:, :, z_start:z_end] = interpolated_chunk.get().resolutionhape(new_r.shape)
        # del X_chunk, Y_chunk, Z_chunk, new_r, new_th, new_phi
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        del X, Y, Z, new_r, new_thx, new_thy

    log_time("Completed spherical to Cartesian conversion")
    return output

# Function to print a message with a timestamp
def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def cartify(
    data: np.ndarray,
    sweep_angle=102,
    downsample=1, #downsample factor
    resolution=1 / 6, # resolutionolution
    threshold=5,
    chunk_size=16,
    save=False,
    circleCrop=1,
):
    log_time("Starting cartify function")
    # data = np.load(file)
    log_time("Data loaded. Shape is:")
    if downsample > 1:
        data = block_reduce(data, block_size=(downsample, downsample, downsample), func=np.mean)
        # data = data[::downsample, ::downsample, ::downsample]

    thetax, r, thetay = data.shape
    r_pad = int(round(r * 1.66)) # what is the constant 1.66 is this distance to the pivot point?
    zeros_array_dimensions = (thetax, r_pad, thetay)
    data = np.pad(
        data,
        ((0, 0), (zeros_array_dimensions[1], 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    data = data.transpose((1, 0, 2))
    print(data.shape)
    ################## !!!!!!!!!!! data  = (data * 255).astype(np.uint8)
    # Compute center and radius
    thetax_center, thetay_center = thetax // 2, thetay // 2 # center of the top of the Cscan is the pivot point
    radius = (min(thetax, thetay) // 2) * circleCrop # radius is half the distance along top of the Cscan take smallest distance ?should we calculate both radii separately?

    # Create a meshgrid of coordinates
    thetax_indxs = np.arange(thetax) - thetax_center # exclude center index because the center does not move
    thetay_indxs = np.arange(thetay) - thetay_center # exclude the center index because the center does not move
    thetax_i, thetay_i = np.meshgrid(thetax_indxs, thetay_indxs, indexing="ij")

    # Compute the circular mask
    distance_squared = thetax_i**2 + thetay_i**2
    circular_mask = distance_squared <= radius**2

    # Apply the mask across all z-slices  Mask is applied orthoganly to the en face axis
    data[:, ~circular_mask] = 0  # Values outside the circle are set to 0
    data[data < threshold] = 0  # Apply threshold

    num_r, num_thx, num_thy = data.shape
    radians = sweep_angle * np.pi / 180 # conversion from radians to degrees

    # Really not sure why its radians/4, but that's what works...
    r = cp.linspace(0, num_r, int(num_r))
    thx = cp.linspace(-radians / 4, radians / 4, int(num_thx)) # this is a symmetrical curve it is changing in 2 dimensions simultaneously it is twice the distance linearly
    thy = cp.linspace(-radians / 4, radians / 4, int(num_thy)) # this is a symetrical curve it is changing in 2 dimensions simultaneously it is twice the distance linearly

    x_dim = y_dim = int(num_r * np.sin(radians / 2)) # proportional to maximum of pi/2 from center ? should this be symmetrical or calculated separately acorrding to the axis?
    z_dim = int(num_r)

    x_resolution = y_resolution = int(num_r * resolution)
    z_resolution = int(z_dim * resolution / 2) # presumably because we are currently only using the bottom half of the image
    x = cp.linspace(-x_dim, x_dim, x_resolution)
    y = cp.linspace(-y_dim, y_dim, y_resolution)
    z = cp.linspace(0, z_dim, z_resolution)

    # Determine optimal chunk size
    log_time("Calculating optimal chunk size")
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    dtype_size = cp.dtype(data.dtype).itemsize
    #chunk_size = get_optimal_chunk_size(len(x), len(y), dtype_size, free_mem)
    print(f"optimal chunk size is: {chunk_size}")
    chunk_size = 16 #32
    print(f"using chunk size {chunk_size}")

    log_time("Warping to Cartesian coordinates")
    cart_image = spherical2cartesian_chunked(
        r, thx, thy, data, x, y, z, sweep_angle, order=1, chunk_size=chunk_size
    )
    log_time("Completed warping to Cartesian coordinates. Shape is:")
    print(cart_image.shape)

    log_time("Cropping the Cartesian volume")
    valid_mask = cart_image > 0
    z_min, z_max = np.where(valid_mask.any(axis=(0, 1)))[0][[0, -1]]
    y_min, y_max = np.where(valid_mask.any(axis=(0, 2)))[0][[0, -1]]
    x_min, x_max = np.where(valid_mask.any(axis=(1, 2)))[0][[0, -1]]

    # Crop the volume
    cart_image = cart_image[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    # print(cart_image.max())
    # if save:
    #    log_time("Saving file")
    #    np.save("rendered.npy", cart_image)
    return cart_image