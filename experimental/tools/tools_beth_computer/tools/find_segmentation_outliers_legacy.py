import pathlib
from pathlib import Path
import sys
from magicgui import widgets
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import napari
from napari.utils.notifications import show_info
import cc3d

from napari_cool_tools_vol_proc._masking_tools_funcs import project_2d_mask
from batch_tools_utils import generate_depth_map, generate_thick_map, load_bits_labels

# def load_bits_labels(file_path:Path,key,shape_key="shape",value:int=1):
#     """
#     """
#     npzfile = np.load(file_path)
#     bits_label = npzfile[key]
#     shape = npzfile[shape_key]
#     label = np.unpackbits(bits_label)*value
#     return label.reshape(shape)

# def generate_depth_map(retina_mask,normalized:bool=True,axis=1):
#     """"""
#     from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func
#     depth = retina_mask.shape[1]
#     raw_depth = retina_mask.sum(axis=axis)

#     if not normalized:
#         return raw_depth
#     else:
#         normalized_depth = raw_depth.astype("float32") / float(depth)
#         normalized_depth = normalize_data_in_range_func(normalized_depth,min_val=0.0,max_val=1.0)
#         return normalized_depth

# 1. Create a dummy text file to process
dummy_file_path = pathlib.Path("dummy.txt")
with open(dummy_file_path, "w") as f:
    f.write("This is the first line.\n")
    f.write("This is the second line.\n")


def process_label(label: np.ndarray):
    """A custom function to process the file."""
    raw_depth_map = generate_depth_map(label, normalized=False)
    retina_coords, rpe_coords, thick_map = generate_thick_map(label)
    difference_map = thick_map - raw_depth_map

    return (
        raw_depth_map,
        retina_coords,
        rpe_coords,
        thick_map,
        difference_map,
    )  # thick_map,difference_map


def get_points_centroid_along_axis(
    points: np.ndarray, points2: np.ndarray, target_axes: list[int] = [1]
):
    """ """
    axes = np.arange(len(points[0]))
    out_data = []
    for ax in axes:
        if ax in target_axes:
            out_data.append((points[:, ax] + points2[:, ax]) / 2)
        else:
            out_data.append(points[:, ax])
    return np.column_stack(out_data)


def mask_components_within_theshold_of_points_distribution(
    binary_mask: np.ndarray,
    point_distirbution: np.ndarray,
    threshold: float,
    verbose: bool = False,
):
    """ """
    # ensure point distribution is of the proper dimenionality [num_points,dimensions/axes]

    # get connected components
    labeled_components = cc3d.connected_components(binary_mask, binary_image=True)
    # get statistics
    stats = cc3d.statistics(labeled_components)
    # get centroids
    centroids = stats["centroids"]
    # build kdtree for point_distribution
    point_distribution_kdtree = KDTree(point_distirbution)
    # find closest points for centroids
    distance_from_centroid, neighbor_index = point_distribution_kdtree.query(
        centroids, k=1, workers=-1
    )
    # filter centroids beyond the theshold distance
    indices_meet_criteria = (distance_from_centroid < threshold).nonzero()
    # build mask of components that meet the criteria
    if verbose:
        print(
            f"centroids: {centroids}\ndistance from centroid: {distance_from_centroid}\nindices that meet criteria: {indices_meet_criteria}\n"
        )
    out_mask = np.isin(labeled_components, indices_meet_criteria)
    return out_mask


def generate_height_map_from_3D_points(
    points_3D: np.ndarray, shape: tuple, axis=1
) -> np.ndarray:
    """"""
    # def depth_from_points_3D(points_3D:np.ndarray,shape:tuple)->np.ndarray:
    # """
    # """
    depth_out = np.zeros(shape)
    # print(depth_out.shape)
    coordinates = np.arange(len(points_3D))
    # print(points_3D[coordinates,0],points_3D[coordinates,2],points_3D[coordinates,1])
    if axis == 1:
        depth_out[points_3D[coordinates, 0], points_3D[coordinates, 2]] = points_3D[
            coordinates, 1
        ]
    elif axis == 2:
        depth_out[points_3D[coordinates, 0], points_3D[coordinates, 1]] = points_3D[
            coordinates, 2
        ]
    elif axis == 0:
        depth_out[points_3D[coordinates, 1], points_3D[coordinates, 2]] = points_3D[
            coordinates, 0
        ]
    return depth_out


def mask_3D_points_with_2D_boolean_mask(
    points_3D: np.ndarray,
    mask: np.ndarray,
    axis: int = 1,
    invert_mask: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Filters a 3D array of points using a 2D boolean mask by projection.

    This function takes a 3D array of points and applies a 2D boolean mask to it.
    The mask is projected along the specified `axis` to generate a 3D boolean mask
    of the same shape as `points_3D`. This mask is then used to filter the
    `points_3D` array. Points where the mask is `True` are kept, while points
    where the mask is `False` are set to 0.

    Args:
        points_3D: A NumPy array of shape `(N, 3, ...)` or similar. The shape
        must conform to the specified checks, particularly that the second
        dimension must have a size of 3.
        mask: A 2D NumPy boolean array of shape `(H, W)`.
        axis: An integer specifying the dimension along which the 2D mask is
        broadcast to create the 3D mask.

    Returns:
        A NumPy array of the same shape as `points_3D` with the filtered points.
        If an error occurs, the original `points_3D` array is returned.

    Raises:
        ValueError: If the second dimension of `points_3D` is not 3.
        ValueError: If the `mask` is not a 2D array.
        ValueError: If the `mask` contains no `True` values.
        ValueError: If the total number of elements in the `mask` does not match
        the number of elements in the `points_3D` dimensions for the specified
        `axis`.
    """
    # --- Error checks ---
    if points_3D.shape[1] != 3:
        print(
            "Error: 'points_3D' must have a second dimension of size 3. "
            "Returning original array."
        )
        return points_3D

    if mask.ndim != 2:
        print("Error: 'mask' must be a 2D array. Returning original array.")
        return points_3D

    if not np.any(mask):
        print("Error: 'mask' contains no positive values. Returning original array.")
        return points_3D

    input_points = points_3D.copy().astype(int)

    # Build voxel grid from points data
    depth_min = int(input_points[:, 1].min())
    depth_max = int(input_points[:, 1].max())
    depth_shape = len(np.arange(depth_max - depth_min)) + 1  # adjust for 0 indexing

    if verbose:
        print(
            f"depth min: {depth_min}, depth max: {depth_max}, depth_shape: {depth_shape}\n"
        )

    voxel_grid = np.zeros((mask.shape[0], depth_shape, mask.shape[1])).astype(bool)
    if verbose:
        print(f"voxel grid shape: {voxel_grid.shape}\n")
    voxel_grid[
        input_points[:, 0], input_points[:, 1] - depth_min, input_points[:, 2]
    ] = 1  # offset depth points by min value to set it to zero

    # Expand mask to cover depth
    mask_3D = np.repeat(mask[:, None, :], depth_shape, axis=axis)
    if invert_mask:
        mask_3D = ~mask_3D
    voxel_grid = voxel_grid * mask_3D
    height, depth, width = voxel_grid.nonzero()
    masked_points_3D = np.column_stack((height, depth + depth_min, width))

    # return voxel_grid,masked_points_3D
    return masked_points_3D


def outlier_correction(points_3D: np.ndarray, outlier_map: np.ndarray) -> np.ndarray:
    """ """
    in_points = points_3D.copy()
    outlier_mask = outlier_map > 0
    outlier_coords = np.column_stack(outlier_mask.nonzero())

    # remove outlier points
    # out_points = remove_outliers_by_xz_coords(out_points,outlier_xz_coords=outlier_coords)
    outlier_free_points = mask_3D_points_with_2D_boolean_mask(
        in_points, mask=outlier_map, axis=1, invert_mask=True
    )

    # Interpolate depth values
    height_map = generate_height_map_from_3D_points(
        points_3D=points_3D, shape=outlier_mask.shape, axis=1
    )
    # print(outlier_coords,outlier_coords.shape)
    height_outlier_nans = height_map.astype(float).copy()
    # height_outlier_nans[outlier_coords] = np.nan
    height_outlier_nans[outlier_mask] = np.nan
    interpolated_nans = fill_nan_nearest_neighbor_2D(height_outlier_nans)
    new_depth_vals = interpolated_nans[outlier_mask].flatten()
    print(new_depth_vals.shape, outlier_coords.shape)

    # create points from depth values
    interpolated_points = np.column_stack(
        (outlier_coords[:, 0], new_depth_vals, outlier_coords[:, 1])
    )
    # interpolated_points = create_3d_points_from_depth(interpolated_nans,depth_axis_index=1)
    # return height_map,height_outlier_nans,interpolated_nans,create_3d_points_from_depth(interpolated_nans,depth_axis_index=1)
    # filtered_points = filter_points_with_2d_mask(interpolated_points,mask_2d=mask)
    # out_points[outlier_coords,1] = new_depth_vals
    out_points = np.concatenate((outlier_free_points, interpolated_points))
    return out_points, outlier_free_points, interpolated_nans


def find_and_process_outliers(
    label_to_process,
    raw_depth_map: np.ndarray,
    difference_map: np.ndarray,
    ret_surf_coords: np.ndarray,
    rpe_surf_coords: np.ndarray,
    gap_threshold: float,
    pixel_thickness_threshold: float,
    component_to_central_retina_threshold: float,
):
    """ """
    # find outlier values
    show_info("Finding Outlier Values\n")
    diff = (
        difference_map > gap_threshold
    ) * 9  # mismatch outliers indicating gaps in the segmentaion
    diff2 = (
        (raw_depth_map >= pixel_thickness_threshold) & ~(diff)
    ) * 4  # mismatch from overly thick measurments 112 is slightly greater than 500 um
    outlier_map = (diff + diff2).astype(bool)
    # print(f"diff shape: {diff.shape}, label shape: {label.shape}\n")

    # Process surface outliers
    # replace outlier values in retina surface points
    ret_surf_replaced, ret_outlier_free, outlier_filled_height_map = outlier_correction(
        ret_surf_coords, outlier_map=outlier_map
    )
    ret_outliers = mask_3D_points_with_2D_boolean_mask(
        ret_surf_coords, mask=outlier_map, axis=1, invert_mask=False
    )
    rpe_outlier_free = mask_3D_points_with_2D_boolean_mask(
        rpe_surf_coords, mask=outlier_map, axis=1, invert_mask=True
    )

    # generate centroid surface for retina
    retina_surf_centroids = get_points_centroid_along_axis(
        ret_outlier_free, rpe_outlier_free, target_axes=[1]
    )
    # generate 3D outlier labels intersections
    diff_3D = project_2d_mask(label_to_process, diff)
    diff2_3D = project_2d_mask(label_to_process, diff2)
    intersect = diff_3D.astype(bool) & label_to_process.astype(bool)
    # recover label components close to the center of the retinal surface
    recovered_components = mask_components_within_theshold_of_points_distribution(
        intersect,
        point_distirbution=retina_surf_centroids,
        threshold=component_to_central_retina_threshold,
    )
    intersect2 = (diff2_3D.astype(bool) & label_to_process.astype(bool)) * 4
    clean = (
        label_to_process.astype(bool)
        & ~intersect.astype(bool)
        & ~intersect2.astype(bool)
    ) + recovered_components

    return (
        clean,
        intersect2,
        recovered_components,
        intersect,
        diff2_3D,
        diff_3D,
        retina_surf_centroids,
        rpe_outlier_free,
        ret_outliers,
        ret_surf_replaced,
        ret_outlier_free,
        outlier_filled_height_map,
        outlier_map,
        diff2,
        diff,
    )


######## LLM AIDED CODE CLEANUP ########


def filter_points_with_2d_mask(points_3d, mask_2d):
    """
    Filters a (N, 3) point array using a (H, W) boolean mask.

    The function assumes the (N, 3) points array can be mapped to a (H, W) grid.

    Args:
        points_3d (np.ndarray): A NumPy array of shape (N, 3) containing 3D points.
        mask_2d (np.ndarray): A NumPy boolean array of shape (H, W),
                              where True values indicate points to keep.

    Returns:
        np.ndarray: A new NumPy array containing only the 3D points that
                    correspond to the True values in the mask, with shape (M, 3),
                    where M is the number of True values.

    Raises:
        ValueError: If the total number of points in the 3D array (N) does not
                    match the number of elements in the 2D mask (H * W).
    """
    # Verify that the total number of points matches the mask dimensions
    if points_3d.shape[0] != mask_2d.size:
        raise ValueError(
            "The number of points in the 3D array must match the number of "
            "elements in the 2D mask. "
            f"Expected {mask_2d.size} points, but got {points_3d.shape[0]}."
        )

    # Flatten the 2D mask into a 1D boolean array
    flat_mask = mask_2d.flatten()

    # Use the 1D boolean mask to index and filter the rows of the 3D point array
    return points_3d[flat_mask]


def remove_outliers_by_xz_coords(points_3d, outlier_xz_coords):
    """
    Removes 3D points from an array if their 2D (x, z) coordinates match
    any of the coordinates in a list of 2D outlier points.

    Args:
        points_3d (np.ndarray): A NumPy array of shape (N, 3) representing 3D points.
        outlier_xz_coords (np.ndarray): A NumPy array of shape (M, 2) containing
                                        the 2D (x, z) coordinates of the outlier points.

    Returns:
        np.ndarray: A new NumPy array of 3D points with the outliers removed.
    """
    # Validate input shapes
    if (
        not isinstance(points_3d, np.ndarray)
        or points_3d.ndim != 2
        or points_3d.shape[1] != 3
    ):
        raise ValueError("points_3d must be a NumPy array of shape (N, 3)")
    if (
        not isinstance(outlier_xz_coords, np.ndarray)
        or outlier_xz_coords.ndim != 2
        or outlier_xz_coords.shape[1] != 2
    ):
        raise ValueError("outlier_xz_coords must be a NumPy array of shape (M, 2)")

    # Extract the 2D (x, z) coordinates from the 3D points array
    # The x-coordinate is at index 0 and the z-coordinate is at index 2
    points_xz = points_3d[:, [0, 2]]

    # Find the indices of the 3D points whose x,z coordinates are in the outlier list
    outlier_mask = np.isin(points_xz, outlier_xz_coords).all(axis=1)

    # Use a boolean mask to keep only the points that are not outliers
    cleaned_points_array = points_3d[~outlier_mask]

    return cleaned_points_array


def create_3d_points_from_depth(depth_array_2d, depth_axis_index, mask=None):
    """
    Transforms a 2D array of depth values into a 3D array of points,
    where the depth is placed along the specified axis.

    Args:
        depth_array_2d (numpy.ndarray): A 2D NumPy array containing depth values.
        depth_axis_index (int): The index (0, 1, or 2) representing the desired
                                 axis for depth in the output 3D points.

    Returns:
        numpy.ndarray: A 3D NumPy array of shape (rows, cols, 3) where each
                       element is a [x, y, z] point, with the depth value
                       placed at the specified depth_axis_index.
                       Returns None if depth_axis_index is invalid.
    """
    if depth_axis_index not in [0, 1, 2]:
        print("Error: depth_axis_index must be 0, 1, or 2.")
        return None

    if mask is None:
        mask = np.ones_like(depth_array_2d).astype(bool)
    elif mask.shape != depth_array_2d.shape:
        print(
            f"Error: The depth array shape {depth_array_2d.shape} and mask shape {mask.shape} do not match!"
        )
        return None

    rows, cols = depth_array_2d.shape
    output_3d_points = np.zeros((rows, cols, 3))

    # Create coordinate grids for x and y
    x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    # x_coords, y_coords = np.meshgrid(np.arange(rows), np.arange(cols))

    # Assign coordinates and depth based on the specified depth_axis_index

    if depth_axis_index == 0:  # Depth along X-axis
        output_3d_points[:, :, 0] = depth_array_2d
        output_3d_points[:, :, 1] = y_coords
        output_3d_points[:, :, 2] = x_coords
    elif depth_axis_index == 1:  # Depth along Y-axis
        output_3d_points[:, :, 0] = x_coords
        output_3d_points[:, :, 1] = depth_array_2d
        output_3d_points[:, :, 2] = y_coords
    elif depth_axis_index == 2:  # Depth along Z-axis
        output_3d_points[:, :, 0] = x_coords
        output_3d_points[:, :, 1] = y_coords
        output_3d_points[:, :, 2] = depth_array_2d

    output_points = output_3d_points.reshape((-1, 3))

    return output_points


def get_largest_contiguous_groups_along_axis(
    binary_data: np.ndarray, axis: int
) -> np.ndarray:
    """
    Finds the largest contiguous group of True values along a specified axis
    in a 3D binary NumPy array and returns a mask for that group.

    Args:
        binary_data (np.ndarray): A 3D NumPy array of boolean (binary) data.
        axis (int): The axis along which to find the largest contiguous groups.
                    0 for z-axis, 1 for y-axis, 2 for x-axis.

    Returns:
        np.ndarray: A 3D NumPy array of boolean masks, where each slice along the
                    specified axis contains only the largest contiguous component.
    """
    if binary_data.ndim != 3:
        raise ValueError("Input data must be a 3D array.")
    if axis not in [0, 1, 2]:
        raise ValueError("Axis must be 0, 1, or 2.")

    # Create an empty array to store the result masks
    output_masks = np.zeros_like(binary_data, dtype=bool)

    # Use a permutation to move the desired axis to the last position for slicing
    permuted_axes = [ax for ax in range(3) if ax != axis] + [axis]
    transposed_data = np.transpose(binary_data, permuted_axes)

    # Loop through each slice along the specified axis
    for i in range(transposed_data.shape[-1]):
        slice_2d = transposed_data[..., i]

        # Use cc3d to find connected components in the 2D slice
        labels_out, N = cc3d.connected_components(
            slice_2d, connectivity=8, return_N=True
        )

        if N > 0:
            # Find the largest component in the slice
            largest_component_label = -1
            largest_component_size = 0

            # Count the size of each component and find the largest
            for segid in range(1, N + 1):
                component_size = np.sum(labels_out == segid)
                if component_size > largest_component_size:
                    largest_component_size = component_size
                    largest_component_label = segid

            if largest_component_label != -1:
                # Create a mask for the largest component and store it
                largest_component_mask = labels_out == largest_component_label
                transposed_data[..., i] = largest_component_mask

    # Transpose the data back to its original orientation
    inverse_permuted_axes = np.argsort(permuted_axes)
    output_masks = np.transpose(transposed_data, inverse_permuted_axes)

    return output_masks


def fill_nan_nearest_neighbor_2D(arr):
    """
    Fills NaN values in a 2D array using the nearest non-NaN neighbor.
    """
    filled_arr = np.copy(arr)
    rows, cols = filled_arr.shape

    # Create a mask for NaN values
    nan_mask = np.isnan(filled_arr)

    # Iterate until no more NaNs can be filled
    while np.any(nan_mask):
        new_nan_mask = np.copy(nan_mask)

        for r in range(rows):
            for c in range(cols):
                if nan_mask[r, c]:
                    # Check 8-directional neighbors
                    neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip self

                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and not np.isnan(filled_arr[nr, nc])
                            ):
                                neighbors.append(filled_arr[nr, nc])

                    if neighbors:
                        # Replace NaN with the first found non-NaN neighbor (or mean/median)
                        filled_arr[r, c] = neighbors[0]
                        new_nan_mask[r, c] = False  # Mark as filled

        # Update the NaN mask for the next iteration
        if np.array_equal(nan_mask, new_nan_mask):  # No more NaNs filled in this pass
            break
        nan_mask = new_nan_mask

    return filled_arr


########


class FileProcessorWidget(widgets.Container):
    """
    A custom widget with a file input and a button to process the file.
    """

    def __init__(self):
        super().__init__()

        # Create child widgets
        self.file_input = widgets.FileEdit(
            label="Select file:",
            mode="r",  # Set to "r" for reading an existing file
            tooltip="Select a file to process.",
            # value=Path(r"F:\38 peak stage ret_chor crop\UNPs_08874712-2023_09_06-14_02_11_processed_ret_chor_seg.npz") #dummy_file_path # Pre-fill with the dummy file path
            value=Path(
                r"E:\38 peak stage ret_chor crop\UNPs_08874712-2023_09_06-14_02_11_processed_ret_chor_seg.npz"
            ),  # dummy_file_path # Pre-fill with the dummy file path
        )
        self.process_button = widgets.PushButton(text="Run Function")

        # Add the widgets to the container
        self.append(self.file_input)
        self.append(self.process_button)

        # Connect the button's click event to a method
        self.process_button.changed.connect(self._on_button_clicked)

    def _on_button_clicked(self):
        """
        This method is called when the button is pressed.
        It runs the custom function with the current file path.
        """
        file_path = self.file_input.value
        # label = load_bits_labels(file_path,"retina")
        label = load_bits_labels(file_path, "choroid")
        raw_depth_map, ret_surf_coords, rpe_surf_coords, thick_map, difference_map = (
            process_label(label)
        )

        viewer.add_image(label, visible=False)
        viewer.add_image(raw_depth_map, visible=False)
        viewer.add_image(thick_map, visible=False)
        viewer.add_image(difference_map, visible=False)
        viewer.add_points(
            ret_surf_coords,
            size=4,
            face_color="red",
            name="retinal_surface",
            visible=False,
        )
        viewer.add_points(
            rpe_surf_coords,
            size=4,
            face_color="blue",
            name="rpe_surface",
            visible=False,
        )
        # viewer.add_points(retina_surf_centroids,size=4,face_color="magenta",name="retina_surf_centroids",visible=True)

        imaging_range = 12  # mm
        refractive_index = 1.33
        conv_factor = (
            imaging_range / label.shape[1] * 1000 / refractive_index
        )  # mm/pixel * um/mm /refractive index = um/pixel

        # threshold values
        gap_threshold = 8
        thickness_theshold = 500  # micrometers
        incedence_allowance = 1 / np.sin(np.pi / 4)
        # pixel_thickness_threshold = int(thickness_theshold / conv_factor)
        pixel_thickness_threshold = (
            int(thickness_theshold / conv_factor) * incedence_allowance
        )
        component_to_pixel_thickness_ratio = 1 / 3  # 1/2 #1/6 #1/8 #1/4 #1/2
        component_to_central_retina_threshold = (
            pixel_thickness_threshold * component_to_pixel_thickness_ratio
        )
        dust_threshold = 1e6
        print(
            f"Thresholdss\ngap: {gap_threshold}\nthickness theshold: {thickness_theshold}\nincedence allowance = {incedence_allowance}\npixel thicknes theshold: {pixel_thickness_threshold}\ndust threshold: {dust_threshold}\n"
        )
        print(
            f"component to pixel thickness ratio: {component_to_pixel_thickness_ratio}\ncomponent to central retina threshold: {component_to_central_retina_threshold}\n"
        )

        # def find_and_process_outliers(label_to_process,raw_depth_map:np.ndarray,difference_map:np.ndarray,ret_surf_coords:np.ndarray,rpe_surf_coords:np.ndarray):
        #     """
        #     """
        #     # find outlier values
        #     show_info("Finding Outlier Values\n")
        #     diff = (difference_map > gap_theshold)*9 # mismatch outliers indicating gaps in the segmentaion
        #     diff2 = ((raw_depth_map >= pixel_thickness_threshold) & ~(diff))*4 # mismatch from overly thick measurments 112 is slightly greater than 500 um
        #     outlier_map = (diff + diff2).astype(bool)
        #     #print(f"diff shape: {diff.shape}, label shape: {label.shape}\n")

        #     # Process surface outliers
        #     # replace outlier values in retina surface points
        #     ret_surf_replaced, ret_outlier_free, outlier_filled_height_map = outlier_correction(ret_surf_coords,outlier_map=outlier_map)
        #     ret_outliers = mask_3D_points_with_2D_boolean_mask(ret_surf_coords,mask=outlier_map,axis=1,invert_mask=False)
        #     rpe_outlier_free = mask_3D_points_with_2D_boolean_mask(rpe_surf_coords,mask=outlier_map,axis=1,invert_mask=True)

        #     # generate centroid surface for retina
        #     retina_surf_centroids = get_points_centroid_along_axis(ret_outlier_free,rpe_outlier_free,target_axes=[1])
        #     # generate 3D outlier labels intersections
        #     diff_3D = project_2d_mask(label_to_process,diff)
        #     diff2_3D = project_2d_mask(label_to_process,diff2)
        #     intersect = (diff_3D.astype(bool) & label_to_process.astype(bool))
        #     # recover label components close to the center of the retinal surface
        #     recovered_components = mask_components_within_theshold_of_points_distribution(intersect,point_distirbution=retina_surf_centroids,threshold=component_to_central_retina_threshold)
        #     intersect2 = (diff2_3D.astype(bool) & label_to_process.astype(bool))*4
        #     clean = (label_to_process.astype(bool) & ~intersect.astype(bool) & ~intersect2.astype(bool)) + recovered_components

        #     return clean,intersect2,recovered_components,intersect,diff2_3D,diff_3D,retina_surf_centroids,rpe_outlier_free,ret_outliers,ret_surf_replaced,ret_outlier_free,outlier_filled_height_map,outlier_map,diff2,diff

        (
            clean,
            intersect2,
            recovered_components,
            intersect,
            diff2_3D,
            diff_3D,
            retina_surf_centroids,
            rpe_outlier_free,
            ret_outliers,
            ret_surf_replaced,
            ret_outlier_free,
            outlier_filled_height_map,
            outlier_map,
            diff2,
            diff,
        ) = find_and_process_outliers(
            label,
            raw_depth_map,
            difference_map,
            ret_surf_coords,
            rpe_surf_coords,
            gap_threshold=gap_threshold,
            pixel_thickness_threshold=pixel_thickness_threshold,
            component_to_central_retina_threshold=component_to_central_retina_threshold,
        )

        viewer.add_labels(diff, name="diff", visible=False)
        viewer.add_labels(diff2, name="diff2", visible=False)
        viewer.add_points(
            ret_surf_replaced,
            size=4,
            face_color="green",
            border_color="yellow",
            name="ret_surf_replaced",
            visible=True,
        )
        viewer.add_points(
            ret_outliers, size=4, face_color="red", border_color="orange", visible=True
        )
        viewer.add_points(
            ret_outlier_free,
            size=4,
            face_color="green",
            border_color="blue",
            visible=True,
        )
        viewer.add_labels(diff_3D, visible=False)
        viewer.add_labels(diff2_3D, visible=False)
        viewer.add_labels(intersect, visible=False)
        viewer.add_labels(recovered_components, visible=True)
        viewer.add_labels(intersect2, visible=False)
        viewer.add_image(outlier_filled_height_map, visible=False)
        viewer.add_labels(clean, visible=False)

        # process small remnants
        show_info("Processing Small Components\n")
        dust_free = cc3d.dust(clean, threshold=dust_threshold).astype("bool") * 6
        outliers = (label.astype("bool") & ~dust_free.astype("bool")) * 10
        outliers_coord_mask = (
            outliers.sum(axis=1).astype(bool) * 1
            - recovered_components.sum(axis=1).astype(bool) * 1
        ).astype(bool)
        num_outliers = np.count_nonzero(outliers)
        num_labeled = np.count_nonzero(label)
        show_info(
            f"{num_outliers}/{num_labeled} outliers {(num_outliers / num_labeled) * 100}%\n"
        )

        viewer.add_labels(dust_free, visible=True)
        viewer.add_labels(outliers)
        viewer.add_image(outliers_coord_mask)

        # TODO revisit this intersect would need to be recalculated to get an accurate reading with the recovered data
        # TODO verify that outlier_coord_mask is accurate and also recalculate. Then perform thickness calculations!!
        show_info("Processing Final Cleanup\n")
        (
            raw_depth_map2,
            ret_surf_coords2,
            rpe_surf_coords2,
            thick_map2,
            difference_map2,
        ) = process_label(dust_free.astype("bool") * 1)

        (
            clean2,
            intersect2_2,
            recovered_components2,
            intersect_2,
            diff2_3D2,
            diff_3D2,
            retina_surf_centroids2,
            rpe_outlier_free2,
            ret_outliers2,
            ret_surf_replaced2,
            ret_outlier_free2,
            outlier_filled_height_map2,
            outlier_map2,
            diff2_2,
            diff_2,
        ) = find_and_process_outliers(
            dust_free,
            raw_depth_map2,
            difference_map2,
            ret_surf_coords2,
            rpe_surf_coords2,
            gap_threshold=gap_threshold,
            pixel_thickness_threshold=pixel_thickness_threshold,
            component_to_central_retina_threshold=component_to_central_retina_threshold,
        )

        # retina_surf_centroids2 = get_points_centroid_along_axis(ret_surf_coords2,rpe_surf_coords2,target_axes=[1])
        recovered_components2 = mask_components_within_theshold_of_points_distribution(
            intersect_2,
            point_distirbution=retina_surf_centroids2,
            threshold=component_to_central_retina_threshold,
        )
        # squeaky_clean = (dust_free.astype(bool) + recovered_components2)*36
        squeaky_clean = (clean2.astype(bool) + recovered_components2) * 36

        # outliers2 = (label.astype("bool") & ~squeaky_clean.astype("bool"))*10
        outliers2 = (dust_free.astype("bool") & ~squeaky_clean.astype("bool")) * 10
        num_outliers2 = np.count_nonzero(outliers2)
        num_labeled2 = np.count_nonzero(dust_free)
        show_info(
            f"{num_outliers2}/{num_labeled2} outliers {(num_outliers2 / num_labeled2) * 100}%\n"
        )

        viewer.add_image(raw_depth_map2, visible=False)
        viewer.add_image(thick_map2, visible=False)
        viewer.add_image(difference_map2, visible=False)
        viewer.add_image(outlier_filled_height_map2, visible=False)
        viewer.add_labels(diff_3D2, visible=False)
        viewer.add_labels(diff2_3D2, visible=False)
        viewer.add_labels(outliers2)
        viewer.add_labels(recovered_components2, visible=True)
        viewer.add_labels(squeaky_clean)
        viewer.add_points(
            ret_surf_coords2,
            size=4,
            face_color="yellow",
            name="retinal_surface2",
            visible=True,
        )
        viewer.add_points(
            rpe_surf_coords2,
            size=4,
            face_color="purple",
            name="rpe_surface2",
            visible=True,
        )
        viewer.add_points(
            retina_surf_centroids2,
            size=4,
            face_color="magenta",
            name="retina_surf_centroids",
            visible=True,
        )
        viewer.add_points(
            ret_surf_replaced2,
            size=4,
            face_color="green",
            border_color="yellow",
            name="ret_surf_replaced",
            visible=True,
        )

    # def _on_button_clicked_current(self):
    #     """
    #     This method is called when the button is pressed.
    #     It runs the custom function with the current file path.
    #     """
    #     file_path = self.file_input.value
    #     label = load_bits_labels(file_path,"retina")
    #     raw_depth_map,ret_surf_coords,rpe_surf_coords,thick_map,difference_map = process_label(label)

    #     viewer.add_image(label,visible=False)
    #     viewer.add_image(raw_depth_map,visible=False)
    #     viewer.add_image(thick_map,visible=False)
    #     viewer.add_image(difference_map,visible=False)
    #     viewer.add_points(ret_surf_coords,size=4,face_color="red",name="retinal_surface",visible=False)
    #     viewer.add_points(rpe_surf_coords,size=4,face_color="blue",name="rpe_surface",visible=False)
    #    #viewer.add_points(retina_surf_centroids,size=4,face_color="magenta",name="retina_surf_centroids",visible=True)

    #     imaging_range = 12 # mm
    #     refractive_index = 1.33
    #     conv_factor = imaging_range / label.shape[1] * 1000 / refractive_index # mm/pixel * um/mm /refractive index = um/pixel

    #     # threshold values
    #     gap_theshold = 8
    #     thickness_theshold = 500 # micrometers
    #     incedence_allowance = 1 / np.sin(np.pi/4)
    #     #pixel_thickness_threshold = int(thickness_theshold / conv_factor)
    #     pixel_thickness_threshold = int(thickness_theshold / conv_factor) * incedence_allowance
    #     component_to_pixel_thickness_ratio = 1/2 #1/2 #1/6 #1/8 #1/4 #1/2
    #     component_to_central_retina_threshold = pixel_thickness_threshold * component_to_pixel_thickness_ratio
    #     dust_threshold = 1e6
    #     print(f"Thresholdss\ngap: {gap_theshold}\nthickness theshold: {thickness_theshold}\nincedence allowance = {incedence_allowance}\npixel thicknes theshold: {pixel_thickness_threshold}\ndust threshold: {dust_threshold}\n")
    #     print(f"component to pixel thickness ratio: {component_to_pixel_thickness_ratio}\ncomponent to central retina threshold: {component_to_central_retina_threshold}\n")

    #     # find outlier values
    #     show_info("Finding Outlier Values\n")
    #     diff = (difference_map > gap_theshold)*9 # mismatch outliers indicating gaps in the segmentaion
    #     diff2 = ((raw_depth_map >= pixel_thickness_threshold) & ~(diff))*4 # mismatch from overly thick measurments 112 is slightly greater than 500 um
    #     outlier_map = (diff + diff2).astype(bool)

    #     #print(f"diff shape: {diff.shape}, label shape: {label.shape}\n")
    #     # viewer.add_labels(diff,name="diff",visible=False)
    #     # viewer.add_labels(diff2,name="diff2",visible=False)

    #     # Process surface outliers
    #     # replace outlier values in retina surface points
    #     ret_surf_replaced, ret_outlier_free, outlier_filled_height_map = outlier_correction(ret_surf_coords,outlier_map=outlier_map)
    #     ret_outliers = mask_3D_points_with_2D_boolean_mask(ret_surf_coords,mask=outlier_map,axis=1,invert_mask=False)
    #     rpe_outlier_free = mask_3D_points_with_2D_boolean_mask(rpe_surf_coords,mask=outlier_map,axis=1,invert_mask=True)

    #     # generate centroid surface for retina
    #     retina_surf_centroids = get_points_centroid_along_axis(ret_outlier_free,rpe_outlier_free,target_axes=[1])

    #     # visualize outliers and valid points
    #     # viewer.add_points(ret_outliers,size=4,face_color="red",border_color="orange",visible=True)
    #     # viewer.add_points(ret_outlier_free,size=4,face_color="green",border_color="blue",visible=True)
    #     viewer.add_points(ret_surf_replaced,size=4,face_color="green",border_color="yellow",name="ret_surf_replaced",visible=True)

    #     diff_3D = project_2d_mask(label,diff)
    #     diff2_3D = project_2d_mask(label,diff2)

    #     intersect = (diff_3D.astype(bool) & label.astype(bool))
    #     recovered_components = mask_components_within_theshold_of_points_distribution(intersect,point_distirbution=retina_surf_centroids,threshold=component_to_central_retina_threshold)
    #     intersect2 = (diff2_3D.astype(bool) & label.astype(bool))*4

    #     viewer.add_labels(diff_3D,visible=False)
    #     viewer.add_labels(diff2_3D,visible=False)
    #     viewer.add_labels(intersect,visible=False)
    #     viewer.add_labels(recovered_components,visible=True)
    #     viewer.add_labels(intersect2,visible=False)

    #     clean = (label.astype(bool) & ~intersect.astype(bool) & ~intersect2.astype(bool)) + recovered_components
    #     #clean = (label.astype(bool) & ~intersect.astype(bool) & ~intersect2.astype(bool) & ~largest_intersect.astype(bool))*36
    #     #clean = (label.astype(bool) & ~largest_intersect.astype(bool) & ~intersect2.astype(bool))*36
    #     viewer.add_labels(clean,visible=False)

    #     # process small remnants
    #     show_info("Processing Small Components\n")
    #     dust_free = cc3d.dust(clean,threshold=dust_threshold).astype("bool")*6
    #     viewer.add_labels(dust_free,visible=True)
    #     outliers = (label.astype("bool") & ~dust_free.astype("bool"))*10
    #     outliers_coord_mask = (outliers.sum(axis=1).astype(bool)*1 - recovered_components.sum(axis=1).astype(bool)*1).astype(bool)
    #     viewer.add_labels(outliers)
    #     viewer.add_image(outliers_coord_mask)
    #     num_outliers = np.count_nonzero(outliers)
    #     num_labeled = np.count_nonzero(label)
    #     show_info(f"{num_outliers}/{num_labeled} outliers {(num_outliers/num_labeled)*100}%\n")

    #     #TODO revisit this intersect would need to be recalculated to get an accurate reading with the recovered data
    #     #TODO verify that outlier_coord_mask is accurate and also recalculate. Then perform thickness calculations!!
    #     show_info("Processing Final Cleanup\n")
    #     raw_depth_map2,ret_surf_coords2,rpe_surf_coords2,thick_map2,difference_map2 = process_label(dust_free.astype("bool")*1)

    #     retina_surf_centroids2 = get_points_centroid_along_axis(ret_surf_coords2,rpe_surf_coords2,target_axes=[1])
    #     recovered_components2 = mask_components_within_theshold_of_points_distribution(intersect,point_distirbution=retina_surf_centroids2,threshold=component_to_central_retina_threshold)
    #     squeaky_clean = (dust_free.astype(bool) + recovered_components2)*36

    #     outliers2 = (label.astype("bool") & ~squeaky_clean.astype("bool"))*10
    #     num_outliers2 = np.count_nonzero(outliers2)
    #     show_info(f"{num_outliers2}/{num_labeled} outliers {(num_outliers2/num_labeled)*100}%\n")

    #     viewer.add_image(raw_depth_map2,visible=False)
    #     viewer.add_image(thick_map2,visible=False)
    #     viewer.add_image(difference_map2,visible=False)
    #     viewer.add_labels(outliers2)
    #     viewer.add_labels(recovered_components2,visible=True)
    #     viewer.add_labels(squeaky_clean)
    #     viewer.add_points(ret_surf_coords2,size=4,face_color="yellow",name="retinal_surface2",visible=True)
    #     viewer.add_points(rpe_surf_coords2,size=4,face_color="purple",name="rpe_surface2",visible=True)
    #     viewer.add_points(retina_surf_centroids2,size=4,face_color="magenta",name="retina_surf_centroids",visible=True)

    #     # raw_depth_map3,ret_surf_coords3,rpe_surf_coords3,thick_map3,difference_map3 = process_label(squeaky_clean.astype("bool")*1)

    #     # show_info("Replacing outlier points\n")

    #     # viewer.add_image(raw_depth_map2,visible=False)
    #     # viewer.add_image(thick_map2,visible=False)
    #     # viewer.add_image(difference_map2,visible=False)
    #     # viewer.add_image(raw_depth_map3,visible=False)
    #     # viewer.add_image(thick_map3,visible=False)
    #     # viewer.add_image(difference_map3,visible=False)
    #     # viewer.add_labels(recovered_components,visible=True)
    #     # viewer.add_labels(squeaky_clean)
    #     # viewer.add_points(ret_surf_coords2,size=4,face_color="yellow",name="retinal_surface2",visible=True)
    #     # viewer.add_points(rpe_surf_coords2,size=4,face_color="purple",name="rpe_surface2",visible=True)
    #     # viewer.add_points(retina_surf_centroids,size=4,face_color="magenta",name="retina_surf_centroids",visible=True)


if __name__ == "__main__":
    try:
        viewer = napari.Viewer()
        # Create and show the custom widget
        processor_widget = FileProcessorWidget()
        viewer.window.add_dock_widget(processor_widget)
        viewer.show()
        napari.run()
        # processor_widget.show(run=True)
    finally:
        # Clean up the dummy file when the widget is closed
        if dummy_file_path.exists():
            dummy_file_path.unlink()
