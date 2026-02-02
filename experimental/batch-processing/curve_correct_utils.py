


from dataclasses import dataclass
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import grad
import napari
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from current_dev import create_patch_processor, generate_patch_indices
from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration._fitting_funcs import sphere_fit_thick_map_corrected_v3
from napari_cool_tools_segmentation._label_cleaning_funcs_v2 import (
    generate_elliptical_mask,
)

logging.basicConfig(
    level=logging.INFO, # Capture INFO and above
    format='%(levelname)s: %(message)s'
)

@dataclass
class CurvCorrectSettings:
    pivot_point: float = 19.2
    imaging_range: float = 12.0
    reference_motor_position: float = 85.0
    imaging_motor_position: float = 85.0
    imaging_motor_position_delta: float = 0.0
    refractive_index: float = 1.33
    scan_angle: float = 100

def equidistant_loss(center_point_3D:np.ndarray, points_3D:np.ndarray):
    """
    """
    distances = jnp.sqrt(jnp.sum((points_3D.astype("float64") - center_point_3D.astype("float64"))**2,axis=1))
    return jnp.var(distances)

def get_incidence_angle_torch(ray_batch, normal_batch, use_degrees: bool = False):
    """Calculates angle of incidence for batches of tensors.

    Args:
        ray_batch (torch.Tensor): Tensor of shape (N, 3) or (3,).
        normal_batch (torch.Tensor): Tensor of same shape as ray_batch.
        use_degrees (bool): Whether to return degrees. Defaults to False.

    Returns:
        torch.Tensor: Angles for each ray in the batch.
    """
    # functional.normalize is efficient for batched unit vector calculation
    u_ray = F.normalize(ray_batch, p=2, dim=-1)
    u_normal = F.normalize(normal_batch, p=2, dim=-1)

    # Batched dot product via element-wise multiplication and reduction
    dot_val = torch.abs(torch.sum(u_ray * u_normal, dim=-1))
    
    # torch.clamp is critical to avoid NaN in acos at the boundaries
    angle_rad = torch.acos(torch.clamp(dot_val, -1.0, 1.0))

    if use_degrees:
        return torch.rad2deg(angle_rad) # Modern alias for degrees
    return angle_rad


def get_pixel_spacing_and_padding(
    cc_settings: CurvCorrectSettings, axial_data_shape: int, verbose: bool = False
):
    """"""
    imaging_range_in_substance = (
        cc_settings.imaging_range / cc_settings.refractive_index
    )
    pixel_spacing = imaging_range_in_substance / axial_data_shape
    if verbose:
        logging.info(
            f"imaging range in water / A-scan pixels = pixel spacing: {imaging_range_in_substance} / {axial_data_shape} = {pixel_spacing}\n"
        )

    base_padding = cc_settings.pivot_point - imaging_range_in_substance

    reference_arm_shift = (
        cc_settings.imaging_motor_position - cc_settings.imaging_motor_position_delta
    ) - cc_settings.reference_motor_position
    reference_arm_shift_in_water = (
        reference_arm_shift * 0.5
    ) / cc_settings.refractive_index
    if verbose:
        logging.info(
            f"(imaging motor position - imaging motor position delta) - reference motor postition = raw refereence arm shift: ({cc_settings.imaging_motor_position} - {cc_settings.imaging_motor_position_delta}) - {cc_settings.reference_motor_position} = {reference_arm_shift}\n"
        )
        logging.info(
            f"(raw reference arm shift / 2) / refractive index = refereence arm shift in water: ({reference_arm_shift} * 0.5) / {cc_settings.refractive_index} = {reference_arm_shift_in_water}\n"
        )

    padding = base_padding + reference_arm_shift_in_water
    if verbose:
        logging.info(
            f"base_padding + reference arm shift in air = padding: {base_padding} + {reference_arm_shift_in_water} = {padding}\n"
        )

    padding_pixel = int(padding / pixel_spacing)
    if verbose:
        logging.info(
            f"padding / pixel spacing = padding pixels: {padding / pixel_spacing} = {padding_pixel}\n"
        )

    return pixel_spacing, padding_pixel


def generate_noisy_ellipsoid_sample_data(center:tuple[int,int,int]=(0,0,0),semi_axes:tuple[float,float,float]=(1.0,1.0,800/840),radius:float=1.0,theta_samples:int=20,seed:int=42,add_noise:bool=True):
    """
    modified from https://jekel.me/2021/A-better-way-to-fit-Ellipsoids/
    """
    # create noise
    if add_noise:
        np.random.seed(seed)
        noise = np.random.normal(size=(theta_samples*theta_samples), loc=0, scale=1e-2)
    else:
        noise = 0
        #noise = np.zeros((theta_samples*theta_samples,))

    # define u,v space which equate to theta and phi in spherical coordinates
    u = np.linspace(0.,np.pi*2,theta_samples)
    v = np.linspace(0.,np.pi, theta_samples)
    u,v = np.meshgrid(u,v,sparse=True)
    a = semi_axes[0] #1.0
    b = semi_axes[1] #0.5,
    c = semi_axes[2] #800/840

    # calculate cartesian coordinates from spherical
    x = a*np.cos(u)*np.sin(v)*radius
    y = b*np.cos(v)*radius
    z = c*np.sin(u)*np.sin(v)*radius

    x = x.flatten() + noise
    y = np.repeat(y.flatten(),theta_samples) + noise
    z = z.flatten() + noise

    x = x+center[0]
    y = y+center[1]
    z = z+center[2]

    return np.column_stack((x,y,z))