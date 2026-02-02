


from dataclasses import dataclass
from pathlib import Path

import numpy as np

from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration._fitting_funcs import CurvCorrectSettings, scan_angle_fit_func, spherical_to_cartesian_corrected
from curve_correct_utils import (
    # CurvCorrectSettings,
    equidistant_loss,
    generate_noisy_ellipsoid_sample_data,
    get_incidence_angle_torch,
    get_pixel_spacing_and_padding,
)

test_data_path = Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Project\All ridge project Outputs\Structure_Raw_RetChor\08810999-2023_05_24-13_05_28_structure.npz")
test_data = npz_file_reader(test_data_path,return_layer=False,verbose=True)
test_coords = np.vstack((test_data > 0).nonzero()).T

print(f"Test coords: {test_coords}\n")

# initiallize curve correction settings
cc_settings = CurvCorrectSettings(
    imaging_range=12., #6., #12.,
    imaging_motor_position_delta=6., #0., #6.,
)

pixel_spacing, padding_pixel = get_pixel_spacing_and_padding(cc_settings=cc_settings,axial_data_shape=test_data.shape[1])

curv_coords = spherical_to_cartesian_corrected(
    points_3D=test_coords,
    input_shape=test_data.shape,
    angle_func=scan_angle_fit_func,
    padding_pixel=padding_pixel
)

pass



