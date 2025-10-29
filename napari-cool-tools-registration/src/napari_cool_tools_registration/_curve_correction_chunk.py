import gc
from datetime import datetime

import cupy as cp
import numpy as np
import pyvista as pv
from cupyx.scipy.ndimage import map_coordinates
from magicgui import magic_factory
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error
from napari_cool_tools_io import viewer
from pyvistaqt import BackgroundPlotter
from skimage.measure import block_reduce, marching_cubes
from tqdm import tqdm


# Function to print a message with a timestamp
def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def spherical2cartesian_chunked(r, thx, thy, grid, x, y, z, order=1, chunk_size=64):
    log_time("Starting spherical to Cartesian conversion")

    # Prepare the output array
    cartesian_shape = (len(x), len(y), len(z))
    output = np.zeros(cartesian_shape, dtype=grid.dtype)

    # Iterate over chunks along the z-axis
    for z_start in tqdm(range(0, len(z), chunk_size), desc="Processing", unit="chunk"):
        z_end = min(z_start + chunk_size, len(z))

        # Create the chunk-specific z-axis slice
        Z = z[z_start:z_end][None, None, :]  # Shape: (1, 1, z_chunk_size)

        # Broadcast X and Y to match the chunk size
        X = x[:, None, None].repeat(Z.shape[-1], axis=-1)  # Broadcast X
        Y = y[None, :, None].repeat(Z.shape[-1], axis=-1)

        # Broadcast all axes
        X, Y, Z = cp.broadcast_arrays(X, Y, Z)

        # Compute spherical coordinates for this chunk
        new_r = cp.sqrt(X**2 + Y**2 + Z**2).astype(cp.float16)
        new_thx = cp.arctan2(X, Z).astype(cp.float16)
        new_thy = cp.arctan2(Y, Z).astype(cp.float16)

        # Normalize angles to [-π, π]
        new_thx = ((new_thx + cp.pi) % (2 * cp.pi)) - cp.pi
        new_thy = ((new_thy + cp.pi) % (2 * cp.pi)) - cp.pi

        # Rescale spherical coordinates to grid indices
        new_ir = (new_r - r.min()) / (r.max() - r.min()) * (len(r) - 1)
        new_ithx = (new_thx - thx.min()) / (thx.max() - thx.min()) * (len(thx) - 1)
        new_ithy = (new_thy - thy.min()) / (thy.max() - thy.min()) * (len(thy) - 1)

        # Compute valid range of indices
        valid_mask = (
            (new_ir >= 0)
            & (new_ir < len(r))
            & (new_ithx >= 0)
            & (new_ithx < len(thx))
            & (new_ithy >= 0)
            & (new_ithy < len(thy))
        )

        if valid_mask.any():
            # Get the bounding indices for the valid region
            r_min, r_max = (
                int(new_ir[valid_mask].min()),
                int(new_ir[valid_mask].max()) + 1,
            )
            thx_min, thx_max = (
                int(new_ithx[valid_mask].min()),
                int(new_ithx[valid_mask].max()) + 1,
            )
            thy_min, thy_max = (
                int(new_ithy[valid_mask].min()),
                int(new_ithy[valid_mask].max()) + 1,
            )

            # Slice the grid to the valid region
            grid_slice = cp.asarray(grid[r_min:r_max, thx_min:thx_max, thy_min:thy_max])

            # Adjust indices to fit within the local grid slice
            local_ir = new_ir[valid_mask] - r_min
            local_ithx = new_ithx[valid_mask] - thx_min
            local_ithy = new_ithy[valid_mask] - thy_min

            # Combine local indices into a single array for map_coordinates
            valid_points = cp.array([local_ir, local_ithx, local_ithy])

            # Perform trilinear interpolation on the grid slice
            interpolated_chunk = map_coordinates(
                grid_slice, valid_points, mode="constant", cval=0
            )

            # Insert interpolated values back into the Cartesian output
            interpolated_chunk_full = cp.zeros(
                new_r.size, dtype=grid.dtype
            )  # Flat array for all points
            interpolated_chunk_full[valid_mask.ravel()] = interpolated_chunk
            output[:, :, z_start:z_end] = interpolated_chunk_full.reshape(
                new_r.shape
            ).get()

        # Free GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    log_time("Completed spherical to Cartesian conversion")
    return output


def estimate_chunk_memory(len_x, len_y, dtype_size, z_chunk_size):
    # Approx for data grid, final grid, interpollation buffers (approx. 100x grid size, Ben's back of the napkin)
    return len_x * len_y * z_chunk_size * dtype_size * 100

def get_optimal_chunk_size(len_x, len_y, dtype_size, free_memory, safety_factor=0.3):
    # Use a safety factor to leave some room for other processes
    available_memory = free_memory * safety_factor
    z_chunk_size = 1  # Start with a small chunk size
    while True:
        memory_per_chunk = estimate_chunk_memory(len_x, len_y, dtype_size, z_chunk_size)
        if memory_per_chunk > available_memory:
            break
        z_chunk_size += 1
    return max(1, z_chunk_size - 1)  # Ensure at least 1


def _on_init(widget):
    @widget.curve_correct_button.clicked.connect
    def do_curve_correct_button():
        if len(widget.input_vol) == 0:
            show_error("Please select an Image")
            return

        input_image = viewer.layers[widget.input_vol.current_choice]
        cartify_function(input_image, widget.ref_indx.value, widget.scan_angle.value,  widget.down_sample.value, widget.res.value, widget.threshold.value, widget.circleCrop.value)

    @widget.vis_3d_button.clicked.connect
    def do_vis_3d_button():
        if len(widget.input_vol_3d) == 0:
            show_error("Please select an Image")
            return

        input_image = viewer.layers[widget.input_vol_3d.current_choice]
        volumerender_function(
            input_image, widget.vol_threshold.value, widget.vol_opac.value
        )

    @widget.surf_3d_button.clicked.connect
    def do_surf_3d_button():
        if len(widget.input_vol_3d) == 0:
            show_error("Please select an Image")
            return

        input_image = viewer.layers[widget.input_surf_3d.current_choice]
        surfacerender_function(input_image, widget.isovalue.value)


@magic_factory(
    call_button=False,
    widget_init=_on_init,
    curve_correct_button=dict(widget_type="PushButton", text="Curve_Correct"),
    scan_angle=dict(widget_type="FloatSpinBox", value=102),
    ref_indx=dict(widget_type="FloatSpinBox", value=1.0),
    imaging_range=dict(widget_type="FloatSpinBox", value=6.0),
    pivot_point=dict(widget_type="FloatSpinBox", value=19.0),
    ref_motor_loc=dict(widget_type="FloatSpinBox", value=0.0),
    img_motor_loc=dict(widget_type="FloatSpinBox", value=0.0),
    down_sample=dict(widget_type="SpinBox", value=2),
    res=dict(widget_type="FloatSpinBox", value=1.0),
    threshold=dict(widget_type="SpinBox", value=5),
    circleCrop=dict(widget_type="SpinBox", value=1),
    vol_threshold=dict(widget_type="FloatSpinBox", value=1.0),
    vol_opac=dict(widget_type="FloatSpinBox", value=0.5),
    isovalue=dict(widget_type="FloatSpinBox", value=5.0),
    vis_3d_button=dict(widget_type="PushButton", text="Volume Render"),
    surf_3d_button=dict(widget_type="PushButton", text="Surface Render"),
    save_button=dict(widget_type="PushButton", text="Save as BigTiff")
    )
def cartify(input_vol: Image, scan_angle, ref_indx, imaging_range, pivot_point, ref_motor_loc, img_motor_loc, 
            down_sample, res, threshold, circleCrop, curve_correct_button, 
            input_vol_3d: Image, vol_threshold, vol_opac, vis_3d_button, 
            input_surf_3d: Image, isovalue, surf_3d_button, save_button):
    return


# TODO
"""what about the refractive index and pivot point location and imaging range???"""


@thread_worker(connect={"returned": viewer.add_layer})
# def cartify_function(input_image: Image, sweep=102, ds=1, res=1/6, threshold=5, chunk_size=16, circleCrop = 1):
def cartify_function(input_image: Image, ref_indx, sweep, ds, res, threshold, circleCrop):

    log_time("Starting cartify function")
    data = input_image.data
    log_time("Data loaded. Shape is:")

    if ds > 1:
        data = block_reduce(data, block_size=(ds, ds, ds), func=np.mean)

    thetax, r, thetay = data.shape
    r_pad = int(round(r * 1.66))
    zeros_array_dimensions = (thetax, r_pad, thetay)
    data = np.pad(
        data,
        ((0, 0), (zeros_array_dimensions[1], 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    data = data.transpose((1, 0, 2))
    print(data.shape)
    data = (data * 255).astype(np.uint8)

    # Compute center and radius
    thx_center, thy_center = thetax // 2, thetay // 2
    radius = (min(thetax, thetay) // 2) * circleCrop

    # Create a meshgrid of coordinates
    thx_m = np.arange(thetax) - thx_center
    thy_m = np.arange(thetay) - thy_center
    thxthx, thythy = np.meshgrid(thx_m, thy_m, indexing="ij")

    # Compute the circular mask
    distance_squared = thxthx**2 + thythy**2
    circular_mask = distance_squared <= radius**2

    # Apply the mask across all z-slices
    data[:, ~circular_mask] = 0  # Values outside the circle are set to 0
    data[data < threshold] = 0  # Apply threshold

    num_r, num_thx, num_thy = data.shape
    angle = sweep * np.pi / 180

    r = cp.linspace(0, num_r, num_r)
    thx = cp.linspace(-angle / 2, angle / 2, num_thx)
    thy = cp.linspace(-angle / 2, angle / 2, num_thy)

    x_dim = y_dim = int(num_r * np.sin(angle / 2))
    z_dim = int(num_r)

    x_res = y_res = int(x_dim * res * 2)
    z_res = int(z_dim * res)
    x = cp.linspace(-x_dim, x_dim, x_res)
    y = cp.linspace(-y_dim, y_dim, y_res)
    z = cp.linspace(0, z_dim, z_res)

    # Determine optimal chunk size
    log_time("Calculating optimal chunk size")
    free_mem, total_mem = cp.cuda.Device(0).mem_info
    dtype_size = cp.dtype(data.dtype).itemsize
    chunk_size = get_optimal_chunk_size(len(x), len(y), dtype_size, free_mem)
    print(chunk_size)

    log_time("Warping to Cartesian coordinates")
    cart_image = spherical2cartesian_chunked(
        r, thx, thy, data, x, y, z, sweep, chunk_size=chunk_size
    )
    log_time("Completed warping to Cartesian coordinates.")

    cart_image = cart_image.transpose((0, 2, 1))

    add_kwargs = {"name": input_image.name + "_output"}
    layer_type = "image"
    out_layer = Layer.create(cart_image, add_kwargs, layer_type)

    return out_layer


def surfacerender_function(input_image: Image, isovalue, color="lightblue"):
    log_time("Starting model rendering")
    # Step 4: Perform isosurface extraction using marching_cubes
    # isovalue = 5
    cart_image = input_image.data
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
        plotter = BackgroundPlotter()
        plotter.add_mesh(mesh, color=color)
        # plotter.show_grid()  # Add grid lines to the background
        # plotter.add_axes()   # Add axes for orientation)
        plotter.show()
        return "Rendering"
    except Exception as e:
        print(f"Subset mesh creation failed: {e}")
        return f"Subset mesh creation failed: {e}"


def volumerender_function(input_image: Image, threshold, opac):
    log_time("Normalizing data for volume render")
    data = input_image.data
    data = data * 255
    data = data.astype(np.uint8)
    # print(data.dtype)
    data_min = data.min()
    # data_max = data.max()
    data_max = 255 * threshold
    # data = np.clip(data, a_min=None, a_max=data_max)
    # print(data.dtype)
    # Step 2: Normalize to the range [0, 255] (or [0, 255] as uint8)
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.uint8)
    plotter = BackgroundPlotter()
    plotter.background_color = "black"
    opacity = [
        0 if val < threshold else opac for val in np.linspace(data.min(), data.max())
    ]
    log_time("Adding volume to GPU")
    plotter.add_volume(data, mapper="gpu", cmap="Blues", opacity=opacity)
    # plotter.show_grid()  # Add grid lines to the background
    # plotter.add_axes()   # Add axes for orientation)
    log_time("Displaying volume")
    plotter.show()
