from pathlib import Path
from typing import Literal

import napari
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import splprep, splev
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from magicgui import magicgui



def curve_length_notinterpolated(coords,  n, reference_arm_shift=-5.8, imaging_range = 6,  fast_axis_shape = 800, slow_axis_shape=840, axial_shape=992, pivot_point = 15.5, scan_angle = 102):
    #coords = array of three dimensional coordinates in theta_x, theta_y, r
    #Entered coords need to be consecutive
    #reference_arm_shift = location relative to the position at pivot point (known to be 85000)
    center = [int(fast_axis_shape*0.5), int(slow_axis_shape*0.5)]
    imaging_range = imaging_range / n
    pixel_spacing = imaging_range/axial_shape

    reference_arm_shift = (reference_arm_shift  *0.5) / n
    padding = pivot_point - imaging_range + reference_arm_shift
    padding_pixel = int(padding/pixel_spacing)

    cart_coords = []
    cart_coords_PIX = []

    for t_y,r,t_x in coords:
        #convert coord input to centered theta_x, theta_y,  and r (with padding)

        p_t_x = ((t_x- center[0])/(fast_axis_shape/2))*(scan_angle/2)* (np.pi/180)
        p_t_y = ((t_y- center[1])/(slow_axis_shape/2))*(scan_angle/2)* (np.pi/180)
        p_r = r + padding_pixel
        p = [p_t_x, p_t_y, p_r]
        cart_point = convert_spherical(p)
        #cart_point_PIX = cart_point
        cart_point = [x * pixel_spacing for x in cart_point]
        cart_point_PIX = cart_point

        cart_coords.append(cart_point)
        cart_coords_PIX.append(cart_point_PIX)

    length = 0
    for i in range(len(cart_coords)-1):

        x1,y1,z1 = cart_coords[i]
        x2,y2,z2 = cart_coords[i+1]
        distance = np.sqrt(((x2-x1)**2 + (y2-y1)**2 +(z2-z1)**2))
        length += distance

    length = length#*pixel_spacing
    return length,  np.array(cart_coords), np.array(cart_coords_PIX)

def lengthBetween2Pts_new(interfaceMap, pts):
    #pts_new = np.round(pts[:, 1:3]).astype(int)
    pts_new = np.round(pts).astype(int)

    x1,y1 = pts_new[0]
    x2,y2 = pts_new[1]

    start = np.round(pts[0]).astype(int)
    end = np.round(pts[1]).astype(int)
    #print(start, end)

    # Use Bresenham's line algorithm to get the nearest neighbor points along the line
    y_values, x_values = bresenham_line(start[1], start[0], end[1], end[0])  # `line` expects (y, x) format

    # Retrieve the corresponding z-values from the interfaceMap
    z_values = interfaceMap[x_values, y_values]

    #Calculate vector distance
    z1 = interfaceMap[x1,y1]
    z2 = interfaceMap[x2,y2]

    start = np.array([x1, y1, z1])
    end = np.array([x2, y2, z2])
    vector_distance = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2 + (end[2] - start[2])**2)

    pts_3d = np.array([[y1, z1,x1], [y2,z2, x2]])
    fullArray = np.array([y_values, z_values, x_values])

    # Stack y, x, and z values into an array of 3D coordinates
    interpolated_coordinates = np.vstack((y_values, z_values,x_values)).T
    CC_length, cart_coords, cart_coords_PIX = curve_length_notinterpolated(interpolated_coordinates, 1.33)


    return CC_length, vector_distance,interpolated_coordinates, pts_3d, cart_coords, cart_coords_PIX

# def convert_spherical(coord):
#     #coord = [theta_x, theta_y, r]
#     #theta_x and theta_y are in radians units
#     theta_x = coord[0]
#     theta_y = coord[1]
#     r = coord[2]
#     theta = np.arctan2(theta_x,theta_y)
#     phi = np.sqrt(theta_x**2 + theta_y**2)

#     x = r*np.sin(phi)*np.cos(theta)
#     y = r*np.sin(phi)*np.sin(theta)
#     z = r*np.cos(phi)
#     return (x,y,z)

def convert_spherical(coord):
    #coord = [theta_x, theta_y, r]
    #theta_x and theta_y are in radians units
    theta_fast = coord[0]
    theta_slow = coord[1]
    axial = coord[2]
    theta = np.arctan2(theta_fast,theta_slow)
    phi = np.sqrt(theta_fast**2 + theta_slow**2)

    x = axial*np.sin(phi)*np.cos(theta)
    y = axial*np.sin(phi)*np.sin(theta)
    z = axial*np.cos(phi)
    return (x,y,z)

def RPE_layer_coords(mask): # TODO replace with existing vectorized code for this
    mask_r = mask == 1
    interfaceMap = np.where(mask_r.any(axis=1), mask_r.shape[1] -1- np.argmax(mask_r[:, ::-1, :], axis=1), -1)
    coords = valid_coordinates(interfaceMap)
    return coords

def Retinal_surface_coords(mask): # TODO replace with existing vectorized code for this
    mask_r = mask ==1
    interfaceMap = np.where(mask_r.any(axis =1), np.argmax(mask_r, axis =1), -1)
    coords = valid_coordinates(interfaceMap)
    return coords


def valid_coordinates(z_array):
    # Find where z is not -1
    mask = z_array != -1

    # Get x and y coordinates
    y_indices, x_indices = np.where(mask)

    # Get the corresponding z values
    z_values = z_array[y_indices, x_indices]

    # Stack into (N, 3) format: (x, y, z)
    coordinates = np.column_stack((y_indices,z_values, x_indices))

    return coordinates

def fit_sphere(points):
    """Fit a sphere to 3D points and return the sphere's center and radius."""
    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Formulate the linear system
    A = np.c_[2*x, 2*y, 2*z, np.ones(len(points))]
    b = x**2 + y**2 + z**2

    # Solve the linear system using least squares
    c, _, _, _ = lstsq(A, b)
    x_c, y_c, z_c, d = c

    # Calculate the radius
    radius = np.sqrt(x_c**2 + y_c**2 + z_c**2 + d)

    return (x_c, y_c, z_c), radius

def sphere_fit(data, mask, rad_clip=2.3, index_of_refraction=1.333, reference_arm_shift=8.5, imaging_range=6, pivot_point=19.2, scan_angle=102):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """

    #y_shape, z_shape, x_shape = mask.shape
    slow_axis_shape, axial_shape, fast_axis_shape = mask.shape
    #imaging_range /= index_of_refraction
    imaging_range = imaging_range / index_of_refraction
    # pixel_spacing = imaging_range / z_shape
    # center = np.array([x_shape // 2 - 60, y_shape // 2])
    pixel_spacing = imaging_range / axial_shape
    center = np.array([fast_axis_shape // 2 - 60, slow_axis_shape // 2]) # what is the magic number 60?
    
    shift = center* (scan_angle / 2) * (np.pi / 180)
    print(f"center,shift: {center,shift}\n")

    # Adjust reference arm shift
    reference_arm_shift = (reference_arm_shift * 0.5) / index_of_refraction
    padding = pivot_point - imaging_range - reference_arm_shift
    padding_pixel = int(padding / pixel_spacing)

    # Circular masking to remove artifacts
    #radius = int(x_shape / rad_clip)
    radius = int(fast_axis_shape / rad_clip)
    circ_mask = circular_mask_3D(mask, center, radius) # TODO replace with existing mask projection function
    mask = np.where(circ_mask, mask, 0)
    data = np.where(circ_mask, data, 0)

    # Extract layer coordinates
    rpe_coords = np.array(RPE_layer_coords(mask))  # Shape: (N, 3)
    retina_coords = np.array(Retinal_surface_coords(mask))  # Shape: (M, 3)

    print(f"rpe coords shape: {rpe_coords.shape}\n")
    print(f"retina coords shape: {retina_coords.shape}\n")
    maximum_intensity_projection = np.max(data, axis =1) # TODO replace with existing projection function
    opacity = 1-maximum_intensity_projection

    # Convert 3D coordinates to 2D indices
    # rpe_y, rpe_z, rpe_x = rpe_coords.T
    # retina_y, retina_z, retina_x = retina_coords.T
    rpe_slow, rpe_axial, rpe_fast = rpe_coords.T
    retina_slow, retina_axial, retina_fast = retina_coords.T
    
    # Vectorized spherical to Cartesian conversion
    def spherical_to_cartesian(slow_axis_coords, z, fast_axis_coords,fast_axis_shape,slow_axis_shape):
        slow_axis_shape = data.shape[0]
        fast_axis_shape = data.shape[2]
        # p_t_x = ((fast_axis_coords - center[0]) / (x_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
        # p_t_y = ((slow_axis_coords - center[1]) / (y_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_t_x = ((fast_axis_coords - center[0]) / (fast_axis_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_t_y = ((slow_axis_coords - center[1]) / (slow_axis_shape / 2)) * (scan_angle / 2) * (np.pi / 180)
        p_r = z + padding_pixel
        #return np.array(convert_spherical([p_t_x, p_t_y, p_r])) * pixel_spacing
        return np.array(convert_spherical([p_t_x, p_t_y, p_r])) #* pixel_spacing


    # Apply spherical-to-Cartesian conversion vectorized
    cart_coords = np.array([spherical_to_cartesian(y, z, x,fast_axis_shape,slow_axis_shape) for y, z, x in retina_coords])

    cart_coords = np.round(cart_coords,1).astype(np.int16)

    print(f"cart_coords (fast,slow,axial): ({cart_coords[:,0],cart_coords[:,1]},{cart_coords[:,2]})\n")

    fast_min_max = np.array((cart_coords[:,0].min(),cart_coords[:,0].max()))
    slow_min_max = np.array((cart_coords[:,1].min(),cart_coords[:,1].max()))
    axial_min_max = np.array((cart_coords[:,2].min(),cart_coords[:,2].max()))

    # fast_min_max = np.round(fast_min_max,1).astype(np.int16)
    # slow_min_max = np.round(slow_min_max,1).astype(np.int16)
    # axial_min_max = np.round(axial_min_max,1).astype(np.int16)


    print(f"fast min/max: {fast_min_max}\n")
    print(f"slow min/max: {slow_min_max}\n")
    print(f"axial min/max: {axial_min_max}\n")

    fast_shape_max = np.abs(fast_min_max).max() #.astype(np.uint16)
    slow_shape_max = np.abs(slow_min_max).max() #.astype(np.uint16)
    axial_shape_max = np.abs(axial_min_max).max()   #.astype(np.uint16)

    print(f"fast shape_max: {fast_shape_max}\n")
    print(f"slow shape_max: {slow_shape_max}\n")
    print(f"axial shape_max: {axial_shape_max}\n")

    fast_shift = (np.round(fast_min_max[0],1)*-1)#.astype(np.uint16)
    slow_shift = (np.round(slow_min_max[0],1)*-1)#.astype(np.uint16)
    print(fast_shape_max,slow_shape_max,axial_shape_max)
    print(f"fast_shift: {fast_shift}\n")
    print(f"slow_shift: {slow_shift}\n")


    


    # cart_coords[:,0] = cart_coords[:,0] + (np.round(fast_min_max[0],1)*-1)
    # cart_coords[:,1] = cart_coords[:,1] + (np.round(slow_min_max[0],1)*-1)
    cart_coords[:,0] = cart_coords[:,0] + (fast_min_max[0]*-1)
    cart_coords[:,1] = cart_coords[:,1] + (slow_min_max[0]*-1)
    cart_coords[:,2] = cart_coords[:,2] -1
    #cart_coords = np.round(cart_coords,1).astype(np.uint16)

    print(f"fast axis (min,max,len): ({cart_coords,cart_coords[:,0].min(),cart_coords[:,0].max(),len(cart_coords[:,0])})\n")
    print(f"slow axis (min,max,len): ({cart_coords,cart_coords[:,1].min(),cart_coords[:,1].max(),len(cart_coords[:,1])})\n")    
    print(f"axial axis (min,max,len): ({cart_coords,cart_coords[:,2].min(),cart_coords[:,2].max(),len(cart_coords[:,2])})\n")    

    new_volume = np.zeros((fast_shape_max*2,slow_shape_max*2,axial_shape_max),dtype=np.uint8)

    new_volume[cart_coords[:,0],cart_coords[:,1],cart_coords[:,2]] = 1

    print(f"new_volume_1: {np.count_nonzero(new_volume)}\n")

    Z_diff = retina_axial - rpe_axial
 
    return cart_coords, Z_diff, opacity, new_volume
    

def circular_mask_3D(data, center, radius):
    """
    Creates a 3D mask where values outside a circular region in the XY plane are masked out.
    
    Parameters:
        array_3d (np.ndarray): Input 3D array with shape (Z, Y, X).
        center (tuple): (y, x) coordinates of the circle center.
        radius (float): Radius of the circular mask.
    
    Returns:
        np.ndarray: Boolean 3D mask with the same shape as array_3d.
    """
    y_dim, z_dim, x_dim = data.shape  # Extract shape
    Y, X = np.ogrid[:y_dim, :x_dim]  # Create 2D coordinate grids
    dist_from_center = (X - center[1])**2 + (Y - center[0])**2  # Compute squared distance
    mask_2d = dist_from_center <= radius**2  # Create 2D mask
    
    return mask_2d[:, None, :].repeat(z_dim, axis=1)  # Expand to 3D

def renderSphere(mask, data, corneal_diam = 8.5):
    cart_coords, z_diffs, opacity = sphere_fit(mask,data)
    # print(f"Center of the fitted sphere: {sphere_center}")
    # print(f"Radius of the fitted sphere: {sphere_radius}")

    plotter = pv.Plotter()
    
    # Define your key values and colors
    z_values = np.array([40, 60.05, 75.95, 112.5, 150])  # Z-diff reference points
    color_labels = {60.05: "Stage 1", 75.95: "Stage 2", 112.5: "Stage 3"}
    colors = np.array([
        [42, 111, 219, 255],  # Soft Blue (#2a6fdb)
        [60, 179, 113, 255],  # Teal-Green (#3cb371)
        [249, 212, 127, 255],  # Soft Yellow (#F9D47F)
        [209, 73, 91, 255],    # Soft Red (#d1495b)
        [139, 0, 0, 255]       # Dark Red for values beyond 112.5 (#8b0000)
    ], dtype=np.uint8)

    # Create a Lookup Table (LUT) with smooth interpolation
    lut = pv.LookupTable()
    #lut.build()  # Ensure it's ready for use
    lut.scalar_range = (z_values.min(), z_values.max())

    # Define the number of interpolated colors
    n_colors = 256  # More values create smoother blending
    interp_colors = np.zeros((n_colors, 4), dtype=np.uint8)

    # Interpolate the colors smoothly
    for i in range(4):  # Iterate over RGBA channels
        interp_colors[:, i] = np.interp(
            np.linspace(z_values.min(), z_values.max(), n_colors), 
            z_values, 
            colors[:, i]
        )

    # Assign interpolated colors to LUT
    lut.values = interp_colors
    rgb_colors = lut.values[z_diffs]
    print(z_diffs.shape)
    print(rgb_colors.shape)
    print(opacity.shape)
    rgba_colors = np.column_stack((rgb_colors, opacity))

    cloud = pv.PolyData(points_chor)
    cloud["RGBA"] = rgba_colors
    mesh = cloud.delaunay_2d()
    plotter.add_mesh(mesh, scalars = "RGBA", rgba= True)
    #cmap = plt.get_cmap("terrain")  # Use "jet" colormap
    
    # shift = np.mean(z_diffs)*(4.33/mask.shape[2]) + 0.2
    # print(shift)
    # # Create the sphere using PyVista
    # sphere = pv.Sphere(radius=sphere_radius+shift, center=sphere_center)
    # # Define the normal vector for the clipping plane (pointing along the z-axis)
    # normal = [0, 0, -1]

    # z_cut_height = sphere_cut(sphere_center, sphere_radius+shift, corneal_diam)

    # # Define the point on the plane at the calculated cut height
    # plane_point = [sphere_center[0], sphere_center[1], z_cut_height]

    # # Clip the sphere at the desired z-height
    # clipped_sphere = sphere.clip(normal=normal, origin=plane_point)

    # rad_curv_cornea = 6.2
    # corneal_z = z_cut_height + np.sqrt(rad_curv_cornea**2 - (corneal_diam/2)**2)
    # corn_center = [sphere_center[0], sphere_center[1], corneal_z]
    # cornea = pv.Sphere(radius = rad_curv_cornea, center = corn_center)
    # clipped_cornea = cornea.clip(normal = [0,0,1], origin=plane_point)
    # #plotter.add_mesh(clipped_cornea, color = 'lightblue', opacity = 0.7, show_edges=False)
    # # Add the clipped sphere to the plotter
    # #plotter.add_mesh(clipped_sphere, color='white', opacity=0.7, show_edges=False)
    # # Create the iris
    # iris_center = [sphere_center[0], sphere_center[1], z_cut_height]
    # disc = pv.Disc(center = iris_center, outer=corneal_diam/2, inner=0.5*corneal_diam/2, c_res = 60)
    
    # #plotter.add_mesh(sphere, color='white', opacity=0.5, show_edges=False, label='Fitted Sphere')
    # # Create a PointCloud from the Cartesian coordinates
    # points_mesh = pv.PolyData(cart_coords)
    # #plotter.add_mesh(disc, opacity = 0.7, color='brown')
    # points_mesh["z_diffs"] = z_diffs
    # plotter.add_points(points_mesh, scalars="z_diffs", cmap=cmap, point_size=2,specular = 1.0, specular_power = 100, show_scalar_bar=False)

    # # Calculate and display the spherical diameter as text
    # diameter = 2 * sphere_radius
    # text_position = sphere_center + np.array([sphere_radius, sphere_radius, sphere_radius])  # Positioning text near the sphere
    # plotter.add_point_labels([text_position], [f"Diameter: {diameter:.2f}"], font_size=24, text_color="black")

    # Add lighting
    light = pv.Light(position=(10, 10, 10), focal_point=(0, 0, 0), intensity=1.0)
    light.positional = True
    light.cast_shadows = True
    plotter.add_light(light)

    plotter.enable_eye_dome_lighting()  # Enhance depth perception

    # Set axes labels
    #plotter.add_axes()
    plotter.set_background('black')

    # Create annotation dictionary (maps z_diff values to labels)
    lut.annotations = {
        60.05: "Stage 1",
        75.95: "Stage 2",
        112.5: "Stage 3"
    }

    # Add scalar bar with the LUT (which now has labels)
    plotter.add_scalar_bar(title="", n_labels=3, color = 'white', bold = True, interactive = True, vertical=True, position_y=0.5)

    # Show coordinate axes and add grid lines
    #plotter.add_axes(line_width=2, color='black')

    # Show the plot
    plotter.show()

def sphere_cut(sphere_center, sphere_radius, hole_diameter):
    """Cut a spherical cap off the top of the sphere (away from the data)."""
    # The hole diameter defines the hole's radius
    hole_radius = hole_diameter / 2
    
    # Find the height at which to cut the sphere (z-coordinate)
    # Using the formula for the height of a spherical cap:
    h = np.sqrt(sphere_radius**2 - hole_radius**2)
    
    # Create a mask where we exclude points above the cut height
    return sphere_center[2]-h

def animateVolume(mask):
    cart_coords, z_diffs, sphere_center, sphere_radius = sphere_fit(mask)
    center = [sphere_center[0], sphere_center[1], sphere_center[2]+sphere_radius/2]
    # Create PyVista plotter
    plotter = pv.Plotter()
    plotter.background_color = "black"

    cmap = plt.get_cmap("jet")  # Use "jet" colormap
    norm = plt.Normalize(vmin=np.min(z_diffs), vmax=np.max(z_diffs))
    colors = cmap(norm(z_diffs))[:, :3]

    # Create the sphere using PyVista
    #sphere = pv.Sphere(radius=sphere_radius+0.4, center=sphere_center)
    #plotter.add_mesh(sphere, color='white', opacity=0.5, show_edges=False, label='Fitted Sphere')
    # Create a PointCloud from the Cartesian coordinates
    points_mesh = pv.PolyData(cart_coords)
    points_mesh["z_diffs"] = z_diffs
    plotter.add_points(points_mesh, scalars="z_diffs", cmap="jet", point_size=3,specular = 1.0, specular_power = 50)

    # Add lighting
    light = pv.Light(position=[center[0]-10, center[1]-10, center[2]], focal_point=center, intensity=1.0)
    light.positional = True
    light.cast_shadows = True
    plotter.add_light(light)
    plotter.enable_eye_dome_lighting()

    # Define camera movement parameters
    n_frames = 120  # Number of frames for smooth rotation
    radius = 25  # Distance from the object
    z_offset = -25 
    viewup = [0, 0, -1]  # Ensure the camera stays upright



    # Save animation as a GIF
    plotter.open_gif("rotation.gif")

    # Function to update camera position for animation
    def update_frame(i):
        angle = i * (360 / n_frames)  # Full 360Â° rotation
        x = center[0] + radius * np.cos(np.radians(angle))  # Move around X-Y plane
        y = center[1] + radius * np.sin(np.radians(angle))
        z = center[2] + z_offset  # Maintain a slightly elevated view

        # Update camera position and ensure it looks at the center
        plotter.camera_position = [(x, y, z), center, viewup]
        plotter.camera.focal_point = center  # Ensure camera stays aimed at the object
        plotter.render()

    for i in range(n_frames):
        update_frame(i)
        plotter.write_frame()

    plotter.close()  # Close the plotter

def load_files():
    """ Open a file dialog to select the data and mask files. """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt user to select the data file
    data_file = filedialog.askopenfilename(title="Select Data File", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    if not data_file:
        print("No data file selected.")
        return None, None

    # Prompt user to select the mask file
    mask_file = filedialog.askopenfilename(title="Select Mask File", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    if not mask_file:
        print("No mask file selected.")
        return None, None

    # Load the NumPy arrays
    data = np.load(data_file)
    mask = np.load(mask_file)

    print(f"Loaded data shape: {data.shape}")
    print(f"Loaded mask shape: {mask.shape}")

    return data, mask

#data = np.load(r"C:\Users\benja\OneDrive - Oregon Health & Science University\Documents\Data Files\ManisData\OHSU-0540_S6.0_OD_14_51_40\volumes_and_labels\OCT_base.npy")
#mask = np.load(r"C:\Users\benja\OneDrive - Oregon Health & Science University\Documents\Data Files\ManisData\OHSU-0540_S6.0_OD_14_51_40\volumes_and_labels\RetChor_Labels.npy")

# data, mask = load_files()

# renderSphere(mask,data)
# #animateVolume(mask)

@magicgui(
    prof_dir={"label": "Path to folder containing ridge masks.", "mode": "r"},
    label_dir={"label": "Path to folder containing retchor masks", "mode": "r"},
    output_dir_path={"label": "Path to output results", "mode": "d"},
    call_button="Fit Sphere Calc Thickness",
)
def sphere_fit_calc(
    prof_dir: Path = Path(r"D:\JJ\Projects\Segmentation_Paper\Data\Bscan\08364574_Complete_PT\08364574-2022_12_06-13_47_38_prof_AS_Corr_Images.pt"),
    label_dir: Path = Path(r"D:\JJ\Projects\Beth_Automation\sphere_fit_test_data\08364574-2022_12_06-13_47_38_retina_label.prof"),
    output_dir_path: Path = Path(r"F:\Beth_RetChor_Stuff\output"),
    output: Literal[".xlsx", ".csv", "none"] = ".xlsx",
    display_dataframe: bool = True,
    scan_angle: float = 106.0,
    imaging_range: float = 6.0,
    refractive_index: float = 1.33,
    mode: Literal["center", "optic disk", "fovea"] = "center",
    incedence_correction: bool = True,
    micrometer_output: bool = False,
    display_in_napari: bool = False,
    verbose: bool = True,
    debug: bool = False,
):
    
    viewer = napari.Viewer(show=False)
    viewer.open(prof_dir, plugin="napari-cool-tools-io")
    viewer.open(label_dir, plugin="napari-cool-tools-io")

    labels = viewer.layers[-1].data
    data = viewer.layers[-2].data

    cart_coords, Z_diff, opacity, new_volume = sphere_fit(data,labels,imaging_range=imaging_range,scan_angle=scan_angle)

    print(f"new_volume shape: {new_volume.shape}\n")
    
    #viewer.add_points(new_volume)
    viewer.add_labels(new_volume)

    print(f"new_volume nonzeros:\n{new_volume.nonzero()}\n")

    viewer.show()
    napari.run()

sphere_fit_calc.show(run=True)