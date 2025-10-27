import gc
from datetime import datetime
from pathlib import Path
from typing import Literal

import cupy as cp
import napari
import numpy as np
import pyvista as pv
from cupyx.scipy.ndimage import map_coordinates
from magicgui import magicgui
# from napari_cool_tools_img_proc import DType
# from napari_cool_tools_img_proc._equalization_funcs import (
#     init_bscan_preproc,
#     normalize_data_in_range_func,
# )
# from napari_cool_tools_registration._registration_tools_funcs import (
#     a_scan_correction_func2,
# )
from pypcd4 import PointCloud
from skimage.measure import block_reduce, marching_cubes
from tqdm import tqdm


# Function to print a message with a timestamp
def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Add timestamps to your functions
def spherical2cartesian_chunked(
    r, thx, thy, grid, x, y, z, order=3, chunk_size=64
):
    # log_time("Starting spherical to Cartesian conversion")

    # Prepare the output array
    cartesian_shape = (len(x), len(y), len(z))
    output = np.zeros(cartesian_shape, dtype=grid.dtype)

    # Iterate over chunks along the z-axis
    # for z_start in range(0, len(z), chunk_size):
    for z_start in tqdm(range(0, len(z), chunk_size), desc="Processing", unit="chunk"):
        z_end = min(z_start + chunk_size, len(z))

        #this is similar with meshgrid but more memory efficient
        Z = z[z_start:z_end][None, None, :]  # Shape: (1, 1, z_chunk_size)
        X = x[:, None, None].repeat(Z.shape[-1], axis=-1)  # Broadcast X
        Y = y[None, :, None].repeat(Z.shape[-1], axis=-1)
        X, Y, Z = cp.broadcast_arrays(X, Y, Z)

        # Compute scan coordinates for this chunk
        new_r = cp.sqrt(X**2 + Y**2 + Z**2).astype(cp.float32)
        new_thx = cp.arctan2(X, Z).astype(cp.float32)
        new_thy = cp.arctan2(Y, Z).astype(cp.float32)

        # Interpolate on GPU
        new_ir = cp.interp(new_r.ravel(), r, cp.arange(len(r)), left=len(r)+1, right=len(r)+1)
        new_ithx = cp.interp(
            new_thx.ravel(), thx, cp.arange(len(thx)), left=len(thx)+1, right=len(thx)+1
        )
        new_ithy = cp.interp(
            new_thy.ravel(), thy, cp.arange(len(thy)), left=len(thy)+1, right=len(thy)+1
        )

        # Map coordinates in the local grid slice
        valid_points = cp.array([new_ir, new_ithx, new_ithy])
        interpolated = map_coordinates(cp.asarray(grid), valid_points, 
                                        order=order, mode="constant",
                                        cval=0.0)
        
        output[:, :, z_start:z_end] = interpolated.get().reshape(new_r.shape)

        del X, Y, Z, new_r, new_thx, new_thy

    log_time("Completed spherical to Cartesian conversion")
    return output


# def cartify(file, sweep=102, ds=1, res=1/6, threshold=5, chunk_size=16, save = False, circleCrop = 1):
def cartify(
    data: np.ndarray,
    #sweep=105,
    angle=105,
    refractive_index=1.33, #1.33
    imaging_range=12.0,
    pivot_point= 19.0,
    ref_motor_location=0.0,
    img_motor_location=0.0,
    ds=1, #downsample factor
    down_sample_factor = 0.5,
    res=1 / 6, # resolution
    threshold=5,
    chunk_size=8, #16,
    save=False,
    circleCrop=1,
):
    log_time("Starting cartify function")
    # data = np.load(file)
    log_time("Data loaded. Shape is:")

    
    print(f"initial shape: {data.shape}\n")

    #initial shape is [x, r, y]

    # #block_size = (data.shape[0]//int(data.shape[0]*down_sample_factor),data.shape[1]//int(data.shape[1]*down_sample_factor),data.shape[2]//int(data.shape[2]*down_sample_factor))
    # if ds > 1:
    # #if ds:
    #     data = block_reduce(data, block_size=(ds, ds, ds), func=np.mean)
    #     #data = block_reduce(data, block_size=block_size, func=np.mean)
    #     # data = data[::ds, ::ds, ::ds]

    print(f"downsampled shape: {data.shape}\n")

    thetax, r, thetay = data.shape

    imaging_range = imaging_range / refractive_index

    pixel_spacing = imaging_range / data.shape[1]

    reference_arm_shift = (ref_motor_location - img_motor_location) #convert micrometers to milimeters
    
    reference_arm_shift = 0

    reference_arm_shift = (
        reference_arm_shift * 0.5 / refractive_index
    )

    padding = pivot_point - imaging_range + reference_arm_shift

    padding_pixel = int(padding / pixel_spacing)

    data = np.pad(
        data,
        ((0, 0), (padding_pixel, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    data = data.transpose((1, 0, 2))

    print(data.shape)

    # Compute center and radius
    thx_center, thy_center = thetax // 2, thetay // 2
    radius = (min(thetax, thetay) // 2)

    # Create a meshgrid of coordinates
    thx_m = np.arange(thetax) - thx_center
    thy_m = np.arange(thetay) - thy_center
    thxthx, thythy = np.meshgrid(thx_m, thy_m, indexing="ij")

    # Compute the circular mask
    distance_squared = thxthx**2 + thythy**2
    circular_mask = distance_squared <= radius**2

    # Apply the mask across all z-slices
    data[:, ~circular_mask] = 0  # Values outside the circle are set to 0

    num_r, num_thx, num_thy = data.shape
    radians = angle * (np.pi / 180)

    r = cp.linspace(0, num_r, int(num_r))
    thx = cp.linspace(-radians/2, radians/2 , num_thx)
    thy = cp.linspace(-radians/2, radians/2, num_thy)

    # x_dim = num_r# * np.sin(radians / 2)
    # y_dim = num_r# * np.sin(radians / 2)
    # z_dim = num_r

    # x_res = int(x_dim * 2) #output resolution
    # y_res = int(y_dim * 2)
    # z_res = int(z_dim * 1)

    # #this is the target output grid
    # x = cp.linspace(-x_dim, x_dim, x_res)
    # y = cp.linspace(-y_dim, y_dim, y_res)
    # z = cp.linspace(0, z_dim, z_res)
    
    th = np.linspace(-np.pi/2, np.pi/2, num_r*2)
    ph = np.linspace(-np.pi/2, np.pi/2, num_r*2)
    rz = np.linspace(0, num_r, num_r)

    x = rz[:, None, None] *np.sin(ph[None,:,None]) * np.cos(th[None, None, :])
    y = rz[:, None, None] *np.sin(ph[None,:,None]) * np.sin(th[None, None, :])
    z = rz[:, None, None] *np.cos(ph[None,:,None])

    # print(f"output shape will be: {x_res},{y_res},{z_res}\n")

    # # Determine optimal chunk size
    # log_time("Calculating optimal chunk size")
    # free_mem, total_mem = cp.cuda.Device(0).mem_info
    # dtype_size = cp.dtype(data.dtype).itemsize
    # chunk_size = get_optimal_chunk_size(len(x), len(y), dtype_size, free_mem)
    # print(f"optimal chunk size is: {chunk_size}")
    chunk_size = 8 #16 #32
    print(f"using chunk size {chunk_size}")

    log_time("Warping to Cartesian coordinates")
    cart_image = spherical2cartesian_chunked(
        r, thx, thy, data, x, y, z, order=3, chunk_size=64
    )
    # log_time("Completed warping to Cartesian coordinates. Shape is:")
    # print(cart_image.shape)

    # log_time("Cropping the Cartesian volume")
    # valid_mask = cart_image > 0
    # z_min, z_max = np.where(valid_mask.any(axis=(0, 1)))[0][[0, -1]]
    # y_min, y_max = np.where(valid_mask.any(axis=(0, 2)))[0][[0, -1]]
    # x_min, x_max = np.where(valid_mask.any(axis=(1, 2)))[0][[0, -1]]

    # # Crop the volume
    # cart_image = cart_image[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    # print(cart_image.max())
    # if save:
    #    log_time("Saving file")
    #    np.save("rendered.npy", cart_image)
    return cart_image


def estimate_chunk_memory(len_x, len_y, dtype_size, z_chunk_size):
    # Approx for data grid, final grid, interpollation buffers (approx. 100x grid size, Ben's back of the napkin)
    return len_x * len_y * z_chunk_size * dtype_size * 100


def get_optimal_chunk_size(len_x, len_y, dtype_size, free_memory, safety_factor=0.7):
    # Use a safety factor to leave some room for other processes
    available_memory = free_memory * safety_factor
    z_chunk_size = 1  # Start with a small chunk size
    while True:
        memory_per_chunk = estimate_chunk_memory(len_x, len_y, dtype_size, z_chunk_size)
        if memory_per_chunk > available_memory:
            break
        z_chunk_size += 1
    return max(1, z_chunk_size - 1)  # Ensure at least 1


def surfacerender(cart_image, color="lightblue", isovalue=5):
    log_time("Starting model rendering")
    # Step 4: Perform isosurface extraction using marching_cubes
    # isovalue = 5
    vertices, faces, normals, _ = marching_cubes(cart_image, level=isovalue)
    log_time("Marching cubes algorithm complete")
    subset_faces = faces

    # Prepend the number 3 to each face
    faces_pv_subset = np.hstack([np.full((subset_faces.shape[0], 1), 3), subset_faces])
    faces_pv_subset = faces_pv_subset.flatten()

    # Try creating the PyVista mesh
    try:
        mesh = pv.PolyData(vertices, faces_pv_subset)
        log_time("Mesh Generated")
        # mesh.plot(show_edges=True)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color=color)
        plotter.show_grid()  # Add grid lines to the background
        plotter.add_axes()  # Add axes for orientation)
        plotter.show()
    except Exception as e:
        print(f"Subset mesh creation failed: {e}")


def volumerender(data, threshold=9, opac=0.3):
    log_time("Normalizing data for volume render")
    data_min = data.min()
    # data_max = data.max()
    data_max = 25
    data = np.clip(data, a_min=None, a_max=data_max)
    # Step 2: Normalize to the range [0, 255] (or [0, 255] as uint8)
    ########## !!!!!!!!!! data = ((data - data_min) / (data_max - data_min) * 255)
    plotter = pv.Plotter()
    plotter.background_color = "black"
    opacity = [
        0 if val < threshold else opac for val in np.linspace(data.min(), data.max())
    ]
    log_time("Adding volume to GPU")
    plotter.add_volume(data, mapper="gpu", cmap="binary_r", opacity=opacity)
    plotter.show_grid()  # Add grid lines to the background
    plotter.add_axes()  # Add axes for orientation)
    log_time("Displaying volume")
    plotter.show()


def pointcloud_renderer(data, threshold=9):
    log_time("Normalizing data for point cloud render")
    data_max = 25
    data = np.clip(data, a_min=None, a_max=data_max)

    pcd_numpy_data = numpy_to_pcd_format(data)

    point_cloud = pv.PolyData(pcd_numpy_data[:, :3])
    point_cloud["intensity"] = pcd_numpy_data[:, -1]

    plotter = pv.Plotter()
    plotter.add_mesh(
        point_cloud, render_points_as_spheres=True, color="red"
    )  # ,eye_dome_lighting=True)
    plotter.enable_eye_dome_lighting()
    plotter.show()


def numpy_to_pcd_format(data: np.ndarray, threshold=9):
    """"""
    x, y, z = np.where(data > threshold)
    print(f"There are {len(x)} points in the pointcloud.\n")
    # x,y,z = np.where(data != 0)
    # x,y,z = np.arange(data.shape[0],np.arange(data.shape[1]),np.arange(data.shape[2]))
    intensity_data = data[x, y, z]
    assert len(x) == len(y) == len(z) == len(intensity_data)

    return np.stack([x, y, z, intensity_data], axis=1)


# Just put the filename and path like below (keep the 'r' before the file name)
# file = r""

# ds = downsampling factor. 1 means no downsampling, 2 means 50% downsampling, et
# res = resolution. Just leave this as 1
# threshold = lowest pixel intensity to process. Will need to change this for each image (different images have different intensities of noise)
# threshold number likely needs to be between 5 and 25
# save can save the 3d file as a .npy if you want it

# cart = cartify(file, sweep = 102, ds = 2, res = 1,threshold = 15, save = False)

# Volumerender below is more "reliable", but surfacerender can generate "prettier" images

# surfacerender(cart, isovalue = 3)
# volumerender(cart)


@magicgui(
    image_path={"label": "Image File", "mode": "r"},
    label_path={"label": "Label File", "mode": "r"},
    output_dir={"label": "Output Directory", "mode": "d"},
    call_button="2 Fast 2 Curvious" #"Uncle Ben's Fast Curve Corrector",
)
def generate_fast_curve_correction(
    image_path: Path = Path(
        r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan\Figure_Sample_Scans\fold_4_img_predictions.prof"
    ),
    label_path: Path = Path(
        r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan\Figure_Sample_Scans\fold_4_label_predictions.prof"
    ),
    output_dir: Path = Path(r"D:\JJ\Projects\RT_Registration\Data\Test_Output"),
    output_filename: str = "output.pt",
    #sweep: int = 105,
    angle: int = 105,
    refractive_index=1.0, #1.33
    imaging_range=6.0, #12.0
    pivot_point= 19.0,
    ref_motor_location=0.0,
    img_motor_location=0.0,
    downsampling: int = 1, #3,
    down_sample_factor: float = 0.25, #0.5
    resolution: float = 1.0, #1 / 6,
    chunk_size: int = 8, #16
    threshold: int = 0, #15,  # 60
    isovalue: int = 35,
    init_preproc: bool = False,
    render_style: Literal["volume", "surface", "points","none"] = "none", #"volume",
    display_in_napari: bool = True,
    save_pcd: bool = False,
    save_npy: bool = False,
    use_gpu: bool = True,
    sin_correct: bool = False,
):
    """ """
    viewer = napari.Viewer(show=False)
    viewer.open(image_path, plugin="napari-cool-tools-io")
    image_data = viewer.layers[-1].data
    image_name = viewer.layers[-1].name
    # normalized_data = normalize_data_in_range_func(image_data,0,255).astype(np.uint8)

    # if sin_correct:
    #     image_data = a_scan_correction_func2(image_data)

    # if init_preproc:
    #     preproc_data = init_bscan_preproc(
    #         image_data,
    #         num_std=16,
    #         min_intensity=0.0,
    #         max_intensity=255.0,
    #         dtype=DType.NP_UINT8,
    #     )
    # else:
    #     preproc_data = normalize_data_in_range_func(image_data,min_val=0.0,max_val=255.0).astype(np.uint8)

    print(type(image_data), image_data.shape)
    # del viewer
    # gc.collect()

    # cart = cartify(data=normalized_data,sweep=sweep,ds=downsampling,res=resolution,threshold=threshold,save=save)
    # cart = cartify(
    #     data=image_data,
    #     #sweep=sweep,
    #     angle=angle,
    #     refractive_index=refractive_index,
    #     imaging_range=imaging_range,
    #     pivot_point=pivot_point,
    #     ref_motor_location=ref_motor_location,
    #     img_motor_location=img_motor_location,
    #     ds=downsampling,
    #     down_sample_factor=down_sample_factor,
    #     res=resolution,
    #     chunk_size=chunk_size,
    #     threshold=threshold,
    #     # save=save,
    #     save=False,
    # )
    image_data = image_data[:, ::-1, :]  # flip the B-scan data to correct orientation
    cart = cartify(data = image_data)

    # # orient for proper viewing in Napari
    # cart = cart.transpose(0,2,1)

    # cart = image_data

    print(f"print cartesian stats: {type(cart)}, {cart.dtype},{cart.shape}\n")

    if render_style == "volume":
        volumerender(cart, 0.2)
    elif render_style == "points":
        pointcloud_renderer(cart, threshold=9)
    elif render_style == "surface":
        surfacerender(cart, isovalue=isovalue)

    if display_in_napari:
        # viewer.add_image(cart,name="uncle_ben-s_curve_correction")
        viewer.add_image(cart, name="curve_correction")
        # viewer.add_labels(cart,name="uncle_ben-s_curve_correction")
        viewer.show()
        napari.run()

    # if save_pcd:
    #     pcd_numpy_data = numpy_to_pcd_format(data=cart, threshold=threshold)
    #     pointcloud = PointCloud.from_xyzi_points(pcd_numpy_data)
    #     output_file_path = output_dir / f"{image_name}.pcd"
    #     pointcloud.save(output_file_path)

    # if save_npy:
    #     log_time("Saving file")
    #     output_file_path = output_dir / f"{image_name}.npy"
    #     np.save(output_file_path, cart)


generate_fast_curve_correction.show(run=True)
