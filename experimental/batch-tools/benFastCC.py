from cupyx.scipy.ndimage import map_coordinates
import cupy as cp
import numpy as np
from skimage.measure import block_reduce
from datetime import datetime
import gc
import pyvista as pv
from skimage.measure import marching_cubes
from tqdm import tqdm


# Function to print a message with a timestamp
def log_time(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

# Add timestamps to your functions
def spherical2cartesian_chunked(r, thx, thy, grid, x, y, z, sweep=140, order=3, chunk_size=64):
    #log_time("Starting spherical to Cartesian conversion")
    
    # Prepare the output array
    cartesian_shape = (len(x), len(y), len(z))
    output = np.zeros(cartesian_shape, dtype=grid.dtype)
    
    # Iterate over chunks along the z-axis
    #for z_start in range(0, len(z), chunk_size):
    for z_start in tqdm(range(0, len(z), chunk_size), desc="Processing", unit="chunk"):
        z_end = min(z_start + chunk_size, len(z))
        #log_time(f"Processing chunk: z[{z_start}:{z_end}]")
        #log_time(f"% conversion processing: {round(100*z_end/len(z),1)}% ")
        # Create the chunk-specific z-axis slice{
        Z = z[z_start:z_end][None, None, :]  # Shape: (1, 1, z_chunk_size)

        # Broadcast X and Y to match the chunk size
        #X = x[:, None, None]  # Shape: (len(x), 1, 1)
        #Y = y[None, :, None]  # Shape: (1, len(y), 1)
        X = x[:, None, None].repeat(Z.shape[-1], axis=-1)  # Broadcast X
        Y = y[None, :, None].repeat(Z.shape[-1], axis=-1)

        X, Y, Z = cp.broadcast_arrays(X, Y, Z)
        # Compute spherical coordinates for this chunk
        new_r = cp.sqrt(X**2 + Y**2 + Z**2).astype(cp.float16)
        new_thx = cp.arctan2(X,Z).astype(cp.float16)
        new_thy = cp.arctan2(Y,Z).astype(cp.float16)
        
        # Interpolate on GPU
        new_ir = cp.interp(new_r.ravel(), r, cp.arange(len(r)))
        new_ithx = cp.interp(new_thx.ravel(), thx, cp.arange(len(thx)), period=2*np.pi)
        new_ithy = cp.interp(new_thy.ravel(), thy, cp.arange(len(thy)), period =2*np.pi)

        # Compute valid indices for this chunk
        valid_mask = (new_r.ravel() <= r.max()) & (new_r.ravel() >= r.min())
        valid_mask &= (new_thx.ravel() <= thx.max()) & (new_thx.ravel() >= thx.min())
        valid_mask &= (new_thy.ravel() <= thy.max()) & (new_thy.ravel() >= thy.min())

        if valid_mask.any():
            # Get the bounding indices for the valid region
            r_min, r_max = int(new_ir[valid_mask].min()), int(new_ir[valid_mask].max()) + 1
            thx_min, thx_max = int(new_ithx[valid_mask].min()), int(new_ithx[valid_mask].max()) + 1
            thy_min, thy_max = int(new_ithy[valid_mask].min()), int(new_ithy[valid_mask].max()) + 1

            # Slice the grid to the valid region
            grid_slice = cp.asarray(grid[r_min:r_max, thx_min:thx_max, thy_min:thy_max])
            # Adjust indices to fit within the local grid slice
            local_ir = new_ir[valid_mask] - r_min
            local_ithx = new_ithx[valid_mask] - thx_min
            local_ithy = new_ithy[valid_mask] - thy_min

            # Map coordinates in the local grid slice
            valid_points = cp.array([local_ir, local_ithx, local_ithy])
            interpolated = map_coordinates(grid_slice, valid_points, order=order)
            
            # Place interpolated values in the output array for this chunk
            interpolated_chunk = cp.zeros_like(new_r.ravel(), dtype=grid.dtype)
            interpolated_chunk[valid_mask] = interpolated
            output[:, :, z_start:z_end] = interpolated_chunk.get().reshape(new_r.shape)
        #del X_chunk, Y_chunk, Z_chunk, new_r, new_th, new_phi
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()    
        del X, Y, Z, new_r, new_thx, new_thy
        
    log_time("Completed spherical to Cartesian conversion")
    return output

def cartify(file, sweep=102, ds=1, res=1/6, threshold=5, chunk_size=16, save = False, circleCrop = 1):
    log_time("Starting cartify function")
    data = np.load(file)
    log_time("Data loaded. Shape is:")
    if ds > 1:
        data = block_reduce(data, block_size=(ds, ds, ds), func=np.mean)
        #data = data[::ds, ::ds, ::ds]

    thetax, r, thetay = data.shape
    r_pad = int(round(r*1.66))
    zeros_array_dimensions = (thetax, r_pad, thetay)
    data = np.pad(data, ((0, 0), (zeros_array_dimensions[1], 0), (0, 0)),
                  mode='constant', constant_values=0)
    data = data.transpose((1, 0, 2))
    print(data.shape)
    data  = (data * 255).astype(np.uint8)
    # Compute center and radius
    thx_center, thy_center = thetax // 2, thetay // 2
    radius = (min(thetax, thetay) // 2) * circleCrop

    # Create a meshgrid of coordinates
    thx_m = np.arange(thetax) - thx_center
    thy_m = np.arange(thetay) - thy_center
    thxthx, thythy = np.meshgrid(thx_m, thy_m, indexing='ij')

    # Compute the circular mask
    distance_squared = thxthx**2 + thythy**2
    circular_mask = distance_squared <= radius**2

    # Apply the mask across all z-slices
    data[:, ~circular_mask] = 0    # Values outside the circle are set to 0
    data[data < threshold] = 0  # Apply threshold
    
    num_r, num_thx, num_thy = data.shape
    angle = sweep * np.pi / 180

    #Really not sure why its angle/4, but that's what works...
    r = cp.linspace(0, num_r, int(num_r))
    thx = cp.linspace(-angle/4, angle/4, int(num_thx))
    thy = cp.linspace(-angle/4, angle/4, int(num_thy))
    
    x_dim = y_dim = int(num_r * np.sin(angle / 2)) 
    z_dim = int(num_r)
 
    x_res = y_res= int(num_r * res)
    z_res = int(z_dim*res/2)
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
    cart_image = spherical2cartesian_chunked(r, thx, thy, data, x, y, z, sweep, order=1, chunk_size=chunk_size)
    log_time("Completed warping to Cartesian coordinates. Shape is:")
    print(cart_image.shape)

    log_time("Cropping the Cartesian volume")
    valid_mask = cart_image > 0
    z_min, z_max = np.where(valid_mask.any(axis=(0, 1)))[0][[0, -1]]
    y_min, y_max = np.where(valid_mask.any(axis=(0, 2)))[0][[0, -1]]
    x_min, x_max = np.where(valid_mask.any(axis=(1, 2)))[0][[0, -1]]

    # Crop the volume
    cart_image = cart_image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    #print(cart_image.max())
    if save:
        log_time("Saving file")
        np.save("rendered.npy", cart_image)
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

def surfacerender(cart_image, color = 'lightblue', isovalue=5):
    log_time("Starting model rendering")
    # Step 4: Perform isosurface extraction using marching_cubes
    #isovalue = 5
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
        #mesh.plot(show_edges=True)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color=color)
        plotter.show_grid()  # Add grid lines to the background
        plotter.add_axes()   # Add axes for orientation)
        plotter.show()
    except Exception as e:
        print(f"Subset mesh creation failed: {e}")
    
def volumerender(data, threshold = 9, opac = 0.3):
    log_time("Normalizing data for volume render")
    data_min = data.min()
    #data_max = data.max()
    data_max = 25
    data = np.clip(data, a_min=None, a_max=data_max)
    # Step 2: Normalize to the range [0, 255] (or [0, 255] as uint8)
    data = ((data - data_min) / (data_max - data_min) * 255)
    plotter = pv.Plotter()
    plotter.background_color = "black"
    opacity = [0 if val < threshold else opac for val in np.linspace(data.min(), data.max())]
    log_time("Adding volume to GPU")
    plotter.add_volume(data, mapper = "gpu", cmap = "binary_r", opacity = opacity)
    plotter.show_grid()  # Add grid lines to the background
    plotter.add_axes()   # Add axes for orientation)
    log_time("Displaying volume")
    plotter.show()
 
#Just put the filename and path like below (keep the 'r' before the file name)
file = r""

#ds = downsampling factor. 1 means no downsampling, 2 means 50% downsampling, et
#res = resolution. Just leave this as 1
#threshold = lowest pixel intensity to process. Will need to change this for each image (different images have different intensities of noise)
#threshold number likely needs to be between 5 and 25
#save can save the 3d file as a .npy if you want it 

cart = cartify(file, sweep = 102, ds = 2, res = 1,threshold = 15, save = False)

#Volumerender below is more "reliable", but surfacerender can generate "prettier" images

#surfacerender(cart, isovalue = 3)
volumerender(cart)