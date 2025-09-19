


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

########