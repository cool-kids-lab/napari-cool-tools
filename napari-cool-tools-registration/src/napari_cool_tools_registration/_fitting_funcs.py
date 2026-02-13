""" """

from typing import Callable

from jax import grad
import jax.numpy as jnp
import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.spatial import KDTree
import torch

from napari_cool_tools_registration import CurvCorrectSettings
from napari_cool_tools_segmentation._label_cleaning_funcs_v2 import (
    Retinal_surface_coords,
    RPE_layer_coords,
)


def scan_angle_fit_func(
    indices_to_map: int, bb: float = 0.7669, cc: float = 0.05, dd=0.0063, ee=0.0107
):
    """"""
    sign = np.sign
    x = np.linspace(-1.0, 1.0, indices_to_map)
    return (
        bb * sign(x) * abs(x) ** 1
        + cc * sign(x) * abs(x) ** 2
        + dd * sign(x) * abs(x) ** 3
        + ee * sign(x) * abs(x) ** 4
    )


def convert_spherical(coord):
    # coord = [theta_x, theta_y, r]
    # theta_x and theta_y are in radians units
    theta_x = coord[0]
    theta_y = coord[1]
    r = coord[2]
    theta = np.arctan2(theta_x, theta_y)
    phi = np.sqrt(theta_x**2 + theta_y**2)
    # phi = np.arctan(np.sqrt(np.tan(theta_x) + np.tan(theta_x)))
    # phi = np.arctan2(np.sqrt(np.tan(theta_x)**2 + np.tan(theta_y)**2), 1.0)  # polar

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return (x, y, z)


def spherical_to_cartesian_corrected(
    points_3D: np.ndarray, input_shape: tuple[int], angle_func: Callable, padding_pixel
):
    """ """
    slow_axis = points_3D[:, 0]
    fast_axis = points_3D[:, 2]
    axial_axis = points_3D[:, 1]
    slow_center = input_shape[0] // 2
    fast_center = input_shape[2] // 2
    # p_t_x = ((x - x_center) / (input_shape[0] / 2)) * (scan_angle / 2) * (np.pi / 180)
    # p_t_y = ((y - y_center) / (input_shape[2] / 2)) * (scan_angle / 2) * (np.pi / 180)
    slow_axis_nonlinear_degree_map = angle_func(input_shape[0])
    fast_axis_nonlinear_degree_map = angle_func(input_shape[2])
    # slow_axis_nonlinear_degree_map = scan_angle_fit_func(input_shape[0])
    # fast_axis_nonlinear_degree_map = scan_angle_fit_func(input_shape[2])
    slow_axis_points = slow_axis_nonlinear_degree_map[slow_axis.astype(int)]
    fast_axis_points = fast_axis_nonlinear_degree_map[fast_axis.astype(int)]
    axial_points = axial_axis + padding_pixel
    # return np.array(convert_spherical([p_t_x, p_t_y, p_r])) * pixel_spacing

    # spherical_tuple = convert_spherical([p_t_x,p_t_y,p_r])
    spherical_tuple = convert_spherical(
        [fast_axis_points, slow_axis_points, axial_points]
    )

    return np.column_stack(
        (spherical_tuple[0], spherical_tuple[2], spherical_tuple[1])
    )  # * pixel_spacing
    # return np.array((spherical_tuple[0],spherical_tuple[2],spherical_tuple[1])) #* pixel_spacing


def sphere_fit_thick_map_corrected_v2(
    mask,
    pixel_spacing: float,
    padding_pixel: float,
    refractive_index: float = 1.33,
    ret_to_rpe: bool = True,
    micron_output: bool = True,
    debug: bool = False,
):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """

    y_shape, z_shape, x_shape = mask.shape
    if debug:
        print(f"retchor_mask shape: {mask.shape}\n")

    # Extract layer coordinates
    # note to self, this essentially calls valid_coordinates three separate times; for efficiency can probably condense
    rpe_coords = RPE_layer_coords(
        mask, use_accelerator=True, return_numpy=True
    )  # Shape: (N, 3)
    retina_coords = Retinal_surface_coords(
        mask, use_accelerator=True, return_numpy=True
    )  # Shape: (M, 3)

    if debug:
        print(
            f"retina_coords len: {len(retina_coords)}\n\nrpe_coords len: {len(rpe_coords)}\n"
        )

    # Convert 3D coordinates to 2D indices
    rpe_y, rpe_z, rpe_x = rpe_coords.T
    retina_y, retina_z, retina_x = retina_coords.T
    standard_height = rpe_z - retina_z

    curv_ret_coords = spherical_to_cartesian_corrected(
        retina_coords,
        input_shape=mask.shape,
        angle_func=scan_angle_fit_func,
        padding_pixel=padding_pixel,
    )
    curv_rpe_coords = spherical_to_cartesian_corrected(
        rpe_coords,
        input_shape=mask.shape,
        angle_func=scan_angle_fit_func,
        padding_pixel=padding_pixel,
    )

    if debug:
        print(
            f"retina_coords len: {len(curv_ret_coords)}\n\nrpe_coords len: {len(curv_rpe_coords)}\n"
        )

    if ret_to_rpe:
        tree = KDTree(curv_rpe_coords)
    else:
        tree = KDTree(curv_ret_coords)

    # Query nearest neighbor for each inner point

    if ret_to_rpe:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)
    else:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)

    # convert from mm to pixels

    if micron_output:
        conv_factor = (
            pixel_spacing * 1000 / refractive_index
        )  # mm/pixel * 1000 um/mm / refractive index = um/pixel
        # conv_factor = pixel_spacing * 1000 # mm/pixel * 1000 um/mm = um/pixel
    else:
        conv_factor = 1.0

    raw_pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    raw_pixel_thickness_map[rpe_y, rpe_x] = standard_height
    pixel_thickness_map[rpe_y, rpe_x] = curve_correct_height
    curve_correct_height = curve_correct_height * conv_factor

    # z_diffs = rpe_z-retina_z
    thickness_map = np.full(
        (y_shape, x_shape), 0.0
    )  # np.nan)  # use NaN for missing pixels

    # Fill thickness map at (y, x) positions
    thickness_map[rpe_y, rpe_x] = curve_correct_height

    # return thickness_map
    # return thickness_map, cart_coords, rpe_coords
    return (
        thickness_map,
        retina_coords,
        rpe_coords,
        curv_ret_coords,
        curv_rpe_coords,
        raw_pixel_thickness_map,
        pixel_thickness_map,
    )


def sphere_fit_thick_map_corrected_v3(
    mask,
    pixel_spacing: float,
    padding_pixel: float,
    refractive_index: float = 1.33,
    ret_to_rpe: bool = True,
    micron_output: bool = True,
    debug: bool = False,
):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """

    y_shape, z_shape, x_shape = mask.shape
    if debug:
        print(f"retchor_mask shape: {mask.shape}\n")

    # Extract layer coordinates
    # note to self, this essentially calls valid_coordinates three separate times; for efficiency can probably condense
    rpe_coords = RPE_layer_coords(
        mask, use_accelerator=True, return_numpy=True
    )  # Shape: (N, 3)
    retina_coords = Retinal_surface_coords(
        mask, use_accelerator=True, return_numpy=True
    )  # Shape: (M, 3)

    if debug:
        print(
            f"retina_coords len: {len(retina_coords)}\n\nrpe_coords len: {len(rpe_coords)}\n"
        )

    # Convert 3D coordinates to 2D indices
    rpe_y, rpe_z, rpe_x = rpe_coords.T
    retina_y, retina_z, retina_x = retina_coords.T
    standard_height = rpe_z - retina_z

    curv_ret_coords = spherical_to_cartesian_corrected(
        retina_coords,
        input_shape=mask.shape,
        angle_func=scan_angle_fit_func,
        padding_pixel=padding_pixel,
    )
    curv_rpe_coords = spherical_to_cartesian_corrected(
        rpe_coords,
        input_shape=mask.shape,
        angle_func=scan_angle_fit_func,
        padding_pixel=padding_pixel,
    )

    if debug:
        print(
            f"retina_coords len: {len(curv_ret_coords)}\n\nrpe_coords len: {len(curv_rpe_coords)}\n"
        )

    if ret_to_rpe:
        tree = KDTree(curv_rpe_coords)
    else:
        tree = KDTree(curv_ret_coords)

    # Query nearest neighbor for each inner point

    if ret_to_rpe:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)
    else:
        curve_correct_height, _ = tree.query(curv_ret_coords, k=1)

    # convert from mm to pixels

    if micron_output:
        conv_factor = (
            pixel_spacing * 1000 / refractive_index
        )  # mm/pixel * 1000 um/mm / refractive index = um/pixel
        # conv_factor = pixel_spacing * 1000 # mm/pixel * 1000 um/mm = um/pixel
    else:
        conv_factor = 1.0

    raw_pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    pixel_thickness_map = np.full((y_shape, x_shape), 0.0)
    raw_pixel_thickness_map[rpe_y, rpe_x] = standard_height
    pixel_thickness_map[rpe_y, rpe_x] = curve_correct_height
    raw_micrometer_thickness_map = raw_pixel_thickness_map * conv_factor
    curve_correct_height = curve_correct_height * conv_factor

    # z_diffs = rpe_z-retina_z
    thickness_map = np.full(
        (y_shape, x_shape), 0.0
    )  # np.nan)  # use NaN for missing pixels

    # Fill thickness map at (y, x) positions
    thickness_map[rpe_y, rpe_x] = curve_correct_height

    # return thickness_map
    # return thickness_map, cart_coords, rpe_coords
    return (
        thickness_map,
        retina_coords,
        rpe_coords,
        curv_ret_coords,
        curv_rpe_coords,
        raw_pixel_thickness_map,
        pixel_thickness_map,
        raw_micrometer_thickness_map,
    )


def extract_surface_coordinates_from_mask(mask: np.ndarray) -> np.ndarray:
    """"""
    far_surface_coords = RPE_layer_coords(
        mask, use_accelerator=True, return_numpy=True
    )  # Shape: (N, 3)
    near_surface_coords = Retinal_surface_coords(
        mask, use_accelerator=True, return_numpy=True
    )
    return near_surface_coords, far_surface_coords


def calculate_distance_between_point_distirbutions(
    main_distribution: np.ndarray,
    query_distribution: np.ndarray,
    k: int = 1,
    return_nearsest_neighbor_coords: bool = True,
):
    """"""
    tree = KDTree(main_distribution)
    distances, indices = tree.query(query_distribution, k=k)

    if not return_nearsest_neighbor_coords:
        return distances, indices
    else:
        return distances, indices, main_distribution[indices]

def extract_surfaces_and_curve_correct_coordinates(
    mask,
    pixel_spacing: float,
    padding_pixel: float,
    refractive_index: float = 1.33,
    micron_output: bool = True,
    debug: bool = False,
):
    """
    Optimized function to fit a sphere to the retinal surface and compute z-difference.
    Returns Cartesian coordinates and color values for visualization.
    """

    # get input mask shape
    slow_shape, axial_shape, fast_shape = mask.shape

    if debug:
        print(f"retchor_mask shape: {mask.shape}\n")

    # Extract layer coordinates
    retina_coords, rpe_coords = extract_surface_coordinates_from_mask(mask)

    if debug:
        print(
            f"retina_coords len: {len(retina_coords)}\n\nrpe_coords len: {len(rpe_coords)}\n"
        )

    # Convert 3D coordinates to 2D indices
    rpe_slow, rpe_axial, rpe_fast = rpe_coords.T
    _, retina_axial, _ = retina_coords.T

    # calculate uncorrected surface difference
    standard_height = rpe_axial - retina_axial

    curv_ret_coords = spherical_to_cartesian_corrected(
        retina_coords,
        input_shape=mask.shape,
        angle_func=scan_angle_fit_func,
        padding_pixel=padding_pixel,
    )
    curv_rpe_coords = spherical_to_cartesian_corrected(
        rpe_coords,
        input_shape=mask.shape,
        angle_func=scan_angle_fit_func,
        padding_pixel=padding_pixel,
    )

    if debug:
        print(
            f"retina_coords len: {len(curv_ret_coords)}\n\nrpe_coords len: {len(curv_rpe_coords)}\n"
        )

    # Find closest point in rpe surface distribution to retinal surface distribution
    curve_correct_height, _, retina_nearest_neighbor_coords = (
        calculate_distance_between_point_distirbutions(
            main_distribution=curv_rpe_coords,
            query_distribution=curv_ret_coords,
            k=1,
            return_nearsest_neighbor_coords=True,
        )
    )

    # Calculate conversion factor from pixels to um
    if micron_output:
        conv_factor = (
            pixel_spacing * 1000 / refractive_index
        )  # mm/pixel * 1000 um/mm / refractive index = um/pixel
    else:
        conv_factor = 1.0

    raw_pixel_thickness_map = np.full((slow_shape, fast_shape), 0.0)
    curve_correct_pixel_thickness_map = np.full((slow_shape, fast_shape), 0.0)
    raw_pixel_thickness_map[rpe_slow, rpe_fast] = standard_height
    curve_correct_pixel_thickness_map[rpe_slow, rpe_fast] = curve_correct_height
    raw_micrometer_thickness_map = raw_pixel_thickness_map * conv_factor
    curve_correct_height = curve_correct_height * conv_factor

    # z_diffs = rpe_z-retina_z
    curve_correct_micrometer_thickness_map = np.full(
        (slow_shape, fast_shape), 0.0
    )  # np.nan)  # use NaN for missing pixels

    # Fill thickness map at (y, x) positions
    curve_correct_micrometer_thickness_map[rpe_slow, rpe_fast] = (
        curve_correct_height
    )

    # return thickness_map
    # return thickness_map, cart_coords, rpe_coords
    return (
        curve_correct_micrometer_thickness_map,
        curve_correct_pixel_thickness_map,
        curv_ret_coords,
        curv_rpe_coords,
        retina_coords,
        rpe_coords,
        retina_nearest_neighbor_coords,
        raw_micrometer_thickness_map,
        raw_pixel_thickness_map,
    )

def equidistant_loss(center_point_3D:np.ndarray, points_3D:np.ndarray):
    """
    """
    distances = jnp.sqrt(jnp.sum((points_3D.astype("float64") - center_point_3D.astype("float64"))**2,axis=1))
    return jnp.var(distances)

def generate_center_point_guess(
    curve_corrected_points: np.ndarray,
    cc_settings: CurvCorrectSettings,
    pixel_spacing: float,
):
    """"""

    # maximum distance down axial axis
    max_axial = curve_corrected_points[:, 1].max()
    # generate inital guess for center of ellipsoid
    center_guess_slow = curve_corrected_points[:, 0].mean()
    center_guess_fast = curve_corrected_points[:, 2].mean()
    center_guess_axial = max_axial - ((cc_settings.pivot_point / 2.0) / pixel_spacing)

    return np.array([center_guess_slow, center_guess_axial, center_guess_fast])

def sphere_fit_points(points_to_fit: np.ndarray, center_point_guess: np.ndarray):
    """"""
    # perform sphere fitting on data
    (
        center_point_result,
        min_variance,
        gradient,
        hessian_matrix,
        function_calls,
        gradient_calls,
        warning_flag,
    ) = fmin_bfgs(
        f=equidistant_loss,
        x0=center_point_guess.astype("float64"),
        fprime=grad(equidistant_loss),
        norm=2.0,
        args=(points_to_fit.astype("float64"),),
        # gtol=1e-17,
        maxiter=None,
        full_output=True,
        disp=False,  # True False
        retall=False,
        callback=None,
    )

    return center_point_result, min_variance

def generate_angle_of_incidence_map(points_to_map:np.ndarray,angles_of_incidence:np.ndarray,map_shape:tuple[int]):
    angle_of_incidence_map = np.zeros(map_shape, dtype=np.float32)
    angle_of_incidence_map[points_to_map[:, 0], points_to_map[:, 2]] = (
        angles_of_incidence.to(torch.float32).numpy()
    )
    return angle_of_incidence_map
