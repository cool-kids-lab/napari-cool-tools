""" """

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from magicgui.widgets import (
    create_widget,
    Container,
    FileEdit,
    FloatSpinBox,
    LineEdit,
    PushButton,
)
from magicgui import magicgui
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import napari

# from napari.utils.notifications import show_info
import numpy as np
import polars as pl
from scipy.stats import normaltest, probplot
import torch
from tqdm import tqdm

from experimental_to_incorporate import (
    generate_patch_maximum_intensity_projections,
    get_top_k_patch_intensity_indices,
    get_top_k_slices_radial_symmetry,
    get_top_k_slices_perpendicular_radial,
    calculate_height_density_profile,
)
from experimental_utils import find_spatial_outliers
from napari_cool_tools_img_proc._normalization_funcs import (
    convert_dtype_and_rescale,
    DType,
)
from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration import CurvCorrectSettings
from napari_cool_tools_registration._fitting_funcs import (
    sphere_fit_thick_map_corrected_v3,
)
from napari_cool_tools_segmentation._label_cleaning_funcs_v2 import (
    generate_elliptical_mask,
)  # fill_ellipse TODO fix error in this function


@dataclass
class CurvCorrectSettings:
    pivot_point: float = 19.2
    imaging_range: float = 12.0
    reference_motor_position: float = 85.0
    imaging_motor_position: float = 85.0
    imaging_motor_position_delta: float = 0.0
    refractive_index: float = 1.33
    scan_angle: float = 100


def equidistant_pixel_error_by_degree(
    error_map: np.ndarray, scan_angle: float = 100, scan_angle_pad: int = 2
):
    """ """
    from scipy import ndimage

    # replace nan values with zero
    data = np.nan_to_num(error_map)
    # get mask of nonzero values
    mask = data.astype(bool)
    # create coordinate grid
    slow_axis, fast_axis = data.shape[:2]
    slow_coords, fast_coords = np.ogrid[0:slow_axis, 0:fast_axis]
    # get centroid
    # center_slow,center_fast = ndimage.center_of_mass(mask)
    center_slow, center_fast = slow_axis // 2, fast_axis // 2
    # shift coordinates for the center
    slow_coords = slow_coords - center_slow
    fast_coords = fast_coords - center_fast
    # TODO add in support for rotations
    # ellipsoid adjustment parameters
    major_axis = slow_axis
    minor_axis = fast_axis
    # calculate distances
    # radii_map = np.sqrt(slow_coords**2 + fast_coords**2)
    elliptical_distance_squared = (slow_coords / major_axis) ** 2 + (
        fast_coords / minor_axis
    ) ** 2
    radii_map = np.sqrt(elliptical_distance_squared) * major_axis
    # define boundaries
    max_distance = np.max(radii_map)
    # num_rings = int(scan_angle/2.0)
    rings_per_pixel = scan_angle / slow_axis
    num_rings = round(rings_per_pixel * max_distance) - 2
    zone_boundaries = np.linspace(0, max_distance, num_rings + 1)
    # zone_boundaries = np.linspace(0,max_distance, num_rings+1)

    # initial_ring_mask = radii_map < zone_boundaries[0]
    # iniitial_pixels_in_ring = data[initial_ring_mask]

    sampled_pixels = []
    sampled_masks = []
    # sampled_pixels = {"Ring_0": data[int(center_slow),int(center_fast)]}
    # sampled_pixels = [data[int(center_slow),int(center_fast)]]
    # sampled_pixels = [iniitial_pixels_in_ring.mean()]

    # for i in range(num_rings):
    for i in range((scan_angle // 2 + scan_angle_pad)):
        inner_radius = zone_boundaries[i]
        outer_radius = zone_boundaries[i + 1]
        # create mask for actively selected pixels
        ring_mask = (radii_map >= inner_radius) & (radii_map < outer_radius)
        sampled_masks.append(ring_mask)
        # extract pixels
        pixels_in_ring = data[ring_mask]
        if pixels_in_ring.sum() > 0:
            nonzero_mask = pixels_in_ring > 0
            # sampled_pixels[f"Ring_{i+1}"] = pixels_in_ring[nonzero_mask].mean()
            sampled_pixels.append(pixels_in_ring[nonzero_mask].mean())
        else:
            # sampled_pixels.append(0.0)
            pass

    return sampled_pixels, sampled_masks


class FileGeneratorWidget(Container):
    """A magicgui widget to generate and step through files in a directory."""

    def __init__(self):
        super().__init__()

        # Widgets for user input
        self.clean_labels_directory_input = FileEdit(
            label="Choose a Structure directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
            # value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.topology_directory_input = FileEdit(
            label="Choose a Topology map directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
            # value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.mask_directory_input = FileEdit(
            label="Choose a data mask directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"E:\_Beth_Thickness_Calculations\Ridge Labels"),
        )
        self.output_directory_input = FileEdit(
            label="Choose an output directory",
            mode="d",  # 'd' specifies directory selection
            # value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output"),
            # value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.output_file_name = LineEdit(
            label="Choose a output directory", value="ridge_heights.csv"
        )
        self.clean_labels_extension_input = LineEdit(
            label="Structure File extension", value="*_structure.npz"
        )
        # self.topology_extension_input = LineEdit(
        #     label="Topology File extension",
        #     value="*_retina_nn_topo_map.npy"
        # )
        # self.mask_extension_input = LineEdit(
        #     label="Mask File extension",
        #     value="*_processed_en_face_ridge_labels.npy"
        # )
        # self.mask_type = LineEdit(
        #     label="Mask type",
        #     value="ridge"
        # )
        # self.outlier_cutoff = FloatSpinBox(
        #     label="Outlier Cutoff",
        #     value=500.,
        # )
        # self.selected_tissue = create_widget(
        #     annotation=Literal["retina","choroid"],
        #     label="Tissue Options",
        #     value="retina",
        # )
        # self.imaging_range = FloatSpinBox(
        #     label="Imaging Range",
        #     value=12.,
        # )
        # self.imaging_motor_position_delta = FloatSpinBox(
        #     label="Imaging motor position offset",
        #     value=6.,
        # )
        self.generate_button = PushButton(text="Create File Generator")

        # Widget to display the current file
        self.next_file_button = PushButton(text="Next File")

        # Widget to process all files
        self.process_all_files_button = PushButton(text="Process All Files")

        # Layout the widgets
        self.extend(
            [
                self.clean_labels_directory_input,
                self.topology_directory_input,
                self.mask_directory_input,
                self.output_directory_input,
                self.output_file_name,
                self.clean_labels_extension_input,
                # self.topology_extension_input,
                # self.mask_extension_input,
                # self.mask_type,
                # self.outlier_cutoff,
                self.generate_button,
                self.next_file_button,
                self.process_all_files_button,
            ]
        )

        # output file path
        self.output_file_path = (
            self.output_directory_input.value / self.output_file_name.value
        )

        # curve correction settings
        self.cc_settings = CurvCorrectSettings(
            imaging_range=12.0,  # 6., #12.,
            imaging_motor_position_delta=6.0,  # 0., #6.,
        )

        # State for the file generator
        self.file_generator = None

        # Connect button signals to methods
        self.generate_button.clicked.connect(self._create_generator)
        # self.generate_button.clicked.connect(self._generate_mask_list)
        self.next_file_button.clicked.connect(self._next_file)
        self.process_all_files_button.clicked.connect(self._process_all_files)

    def _create_generator(self):
        """Creates a generator for files with the specified extension."""
        directory = self.clean_labels_directory_input.value
        extension = self.clean_labels_extension_input.value
        # directory = self.topology_directory_input.value
        # extension = self.topology_extension_input.value
        self.extension = extension

        if not directory or not Path(directory).is_dir():
            print("Error: Invalid directory.")
            return

        # Create a generator for the files
        self.file_generator = (
            f for f in Path(directory).rglob(extension) if f.is_file()
        )
        print(f"Generator created for '{extension}' files in {directory}.")

        self.file_list = list(
            f for f in Path(directory).rglob(extension) if f.is_file()
        )
        print(
            f"List containing {len(self.file_list)} entries created for '{extension}' files in {directory}."
        )

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            next_file = next(self.file_generator)
            print(str(next_file))

            self._process_file(next_file)

            print("File Processed.")
            # TODO add viewer back
            # viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    # def _generate_mask_list(self):
    #     # Get mask list
    #     directory = self.mask_directory_input.value
    #     extension = self.mask_extension_input.value
    #     self.mask_list = list(
    #         f for f in Path(self.mask_directory_input.value).rglob(extension)
    #         if f.is_file()
    #     )
    #     print(f"List containing {len(self.file_list)} mask entries created for '{extension}' files in {directory}.")

    def _process_all_files(self):
        """Processes all files in the generator"""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            file_path_pbar = tqdm(self.file_generator)
            for file_path in file_path_pbar:
                file_path_pbar.set_description(f"Processing: {file_path}")

                self._process_file(file_path=file_path)

            print("End of files. Processing complete.")
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_file(self, file_path):
        """"""

        output_dir = self.output_directory_input.value

        verbose = False  # True # TODO make this an option on the widget
        save_plots = False
        show_plots = True

        extension = Path(self.extension).stem.replace("*", "")

        # get id
        id = file_path.stem.replace(extension, "")

        # load data
        if file_path.suffix == ".npz":
            data, attributes, layer_type = npz_file_reader(
                file_path, return_layer=True, verbose=True
            )[0]
        elif file_path.suffix == ".npy":
            data = np.load(file_path)

        viewer.add_image(data, name=f"{id}_structure")
        
        cropped_data = data.copy()
        preproc_data_mask = generate_elliptical_mask(data.shape,use_input_depth=True)
        cropped_data[~preproc_data_mask] = 0

        viewer.add_image(cropped_data, name=f"{id}_cropped")

        diff_data = data - cropped_data

        viewer.add_image(diff_data, name=f"{id}_diff")



if __name__ == "__main__":
    # Create the widget and show it
    viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    viewer.window.add_dock_widget(my_widget)
    viewer.show()
    napari.run()
    # my_widget.show(run=True)
