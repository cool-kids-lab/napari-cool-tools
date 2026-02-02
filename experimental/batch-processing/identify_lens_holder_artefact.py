""" """

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cv2 import getStructuringElement, MORPH_ELLIPSE
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
    clean_hardware_resources,
    apply_morphology,
    apply_patch_morphology,
)
from experimental_utils import (
    find_spatial_outliers,
    generate_targeted_3d_fill_optimized,
    linear_quantization,
    iterative_3d_inpaint_with_grid,
    iterative_3d_inpaint_cropped_grid,
    iterative_3d_inpaint_hybrid,
    iterative_3d_inpaint_with_noise,
)
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
    remove_lens_holder,
    reclaim_memory,
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

        # # convert data to float and rescale to 0-1 range
        # data = convert_dtype_and_rescale(data, datatype=DType.NP_UINT8)
        # og_data = data.copy()
        # viewer.add_image(og_data, name=f"{id}_structure",visible=False)

        # k = 1
        percentile = 99.0
        density_range = 48  # 32 #64
        pixel_offset = 48  # 32
        quantization = 6

        # # get_top_k high intensity density
        # top_k_values, top_k_indices = get_top_k_patch_intensity_indices(data,patch_height_size=density_range,intensity_threshold=0,stride_step=1,top_k_count=5,use_symmetric_padding=True,device_name="cpu",return_numpy=True)
        # high_intensity_theshold = data[:,top_k_indices[0],:].mean()
        # top_k_values, top_k_indices = get_top_k_patch_intensity_indices(data,patch_height_size=density_range,intensity_threshold=high_intensity_theshold,stride_step=1,top_k_count=3,use_symmetric_padding=True,device_name="cpu",return_numpy=True)

        # mean_idx = round(top_k_indices.mean())
        # target_density_start = round(mean_idx-(density_range/2))
        # target_density_stop = round(mean_idx+(density_range/2))

        # density_range_label = np.zeros_like(data,dtype="uint8")
        # density_range_label[:,target_density_start:target_density_stop,:] = 10
        # density_range_label[:,mean_idx,:] = 6

        # def remove_lens_holder(data:np.ndarray|torch.Tensor,percentile:float=99.0,holder_height:int=64,pixel_offset:int=24):
        #     """"""

        #     preproc_data = np.zeros_like(data,dtype="uint8")
        #     preproc_data[data >= np.percentile(data, percentile)] = 1

        #     preproc_data_mask = generate_elliptical_mask(data.shape,use_input_depth=True)
        #     preproc_data[preproc_data_mask] = 0

        #     cleanup_mask = generate_elliptical_mask(data.shape,symmetric_axis_offset=(pixel_offset,pixel_offset),use_input_depth=True)

        #     _,density_label = calculate_height_density_profile(torch.as_tensor(preproc_data,device="cuda"),return_as_numpy=True,enable_visualization=True,window_height_voxels=holder_height,center_slice_value=6)
        #     density_mask = density_label > 0

        #     data[~preproc_data_mask] = 0
        #     volume_to_remove = (preproc_data_mask & ~(cleanup_mask)) & density_mask
        #     #volume_to_remove = density_mask & ~cleanup_mask
        #     data[volume_to_remove] = 0
        #     return data

        lens_holder_free, density_mask, area_of_interest = remove_lens_holder(
            data.copy(),
            percentile=99.0,
            holder_height=density_range,
            pixel_offset=pixel_offset,
            use_elliptical_mask=True,
            return_mask=True,
        )
        reclaim_memory()

        cleaned, density_mask, hole_to_interpolate = remove_lens_holder(
            data.copy(),
            percentile=99.0,
            holder_height=density_range,
            pixel_offset=pixel_offset,
            use_elliptical_mask=False,
            return_mask=True,
        )
        reclaim_memory()

        # preproc_data = np.zeros_like(data, dtype="uint8")
        # preproc_data[data >= np.percentile(data, percentile)] = 1
        # # preproc_data[data >= np.percentile(data, 99.7)] = 1

        # viewer.add_labels(preproc_data * 9, name=f"{id}_{percentile}th", visible=False)

        # # major_axis_len = data.shape[0]
        # # minor_axis_len = data.shape[2]
        # # center = major_axis_len//2,minor_axis_len//2
        # # preproc_data = fill_ellipse(data,center=center,major_axis_len=major_axis_len,minor_axis_len=minor_axis_len,fill_value=0)

        # preproc_data_mask = generate_elliptical_mask(data.shape, use_input_depth=True)
        # og_preproc_data = preproc_data.copy()
        # preproc_data[preproc_data_mask] = 0
        # # data[~(preproc_data_mask > 0)] = 0
        # viewer.add_image(data.copy(), name=f"{id}_cropped", visible=False)

        # cleanup_mask = generate_elliptical_mask(
        #     data.shape,
        #     symmetric_axis_offset=(pixel_offset, pixel_offset),
        #     use_input_depth=True,
        # )

        # # stucturing_element = torch.as_tensor(getStructuringElement(MORPH_ELLIPSE,(5,5)))

        # # eroded_mask = apply_morphology(
        # #     preproc_data_mask > 0,
        # #     kernel=stucturing_element,
        # #     batch_dimension="height",
        # #     morphology_type="erosion",
        # #     inference=True,
        # #     return_numpy=True,
        # #     keep_on_gpu=False,
        # #     compute_device=None,
        # # )

        # _, density_label = calculate_height_density_profile(
        #     torch.as_tensor(preproc_data, device="cuda"),
        #     return_as_numpy=True,
        #     enable_visualization=True,
        #     window_height_voxels=density_range,
        #     center_slice_value=6,
        # )
        # clean_hardware_resources()

        # viewer.add_labels(density_label, visible=False)

        # density_mask = density_label > 0

        # annular_mask = np.zeros_like(density_mask, dtype="uint8")
        # # annular_mask[:] = og_data * density_mask
        # annular_mask[:] = og_preproc_data * density_mask
        # annular_mask = annular_mask.sum(axis=1) > 0  # & ~cleanup_mask[:,0,:]
        # viewer.add_labels(
        #     (annular_mask * 4).astype("uint8"), name="annular_mask", visible=False
        # )

        # area_of_interest = (preproc_data_mask & ~(cleanup_mask)) & density_mask
        # hole_to_interpolate = density_mask & ~cleanup_mask

        sample = data.copy()
        sample[~area_of_interest] = 0

        # # viewer.add_image(linear_quantization(sample,quantizations=quantization),name="sample_quantized",visible=False)

        # data[hole_to_interpolate] = 0
        # # data[area_of_interest] = 0

        print(f"Attempting iterative 3D inpainting.")
        # attempt iterative 3D inpainting of data
        fill_input = convert_dtype_and_rescale(cleaned.copy(), DType.NP_FLOAT32)
        # fill_input = convert_dtype_and_rescale(data.copy(), DType.NP_FLOAT32)

        # in_filled_volume = iterative_3d_inpaint_with_grid(torch.tensor(fill_input),torch.tensor(hole_to_interpolate),iterations=2,verbose=True).detach().squeeze().cpu().numpy()

        # in_filled_volume = iterative_3d_inpaint_cropped_grid(torch.tensor(fill_input),torch.tensor(hole_to_interpolate),iterations=32,verbose=True).detach().squeeze().cpu().numpy()

        # in_filled_volume = (
        #     iterative_3d_inpaint_hybrid(
        #         torch.tensor(fill_input),
        #         torch.tensor(hole_to_interpolate),
        #         iterations=64,
        #         device="cuda",
        #         verbose=True,
        #     )
        #     .detach()
        #     .squeeze()
        #     .cpu()
        #     .numpy()
        # )

        # in_filled_volume = (
        #     iterative_3d_inpaint_with_noise(
        #         torch.tensor(fill_input),
        #         torch.tensor(hole_to_interpolate),
        #         # torch.tensor(hole_to_interpolate),
        #         iterations=64,
        #         device="cuda",
        #         noise_std=0.01,
        #         clump_ratio=0.8,
        #         clump_size=2,
        #         intensity_offset=0.001,
        #         sharpness=0.001,
        #         verbose=False,
        #     )
        #     .detach()
        #     .squeeze()
        #     .cpu()
        #     .numpy()
        # )

        in_filled_volume = (
            generate_targeted_3d_fill_optimized(
                torch.tensor(fill_input),
                torch.tensor(hole_to_interpolate),
                perlin_weight=0.8,
                context_margin=12,
                base_freq=1.0,
                octaves=2,
                persistence=0.2,
                lacunarity=2.0,
            )
            .detach()
            .squeeze()
            .cpu()
            .numpy()
        )

        reclaim_memory()
        # clean_hardware_resources()

        print("Infilling complete.")
        in_filled_volume = convert_dtype_and_rescale(in_filled_volume, DType.NP_UINT8)

        viewer.add_image(data, visible=False)
        viewer.add_image(in_filled_volume, visible=True)
        viewer.add_image(cleaned, name=f"{id}_cleaned", visible=True)
        viewer.add_image(
            lens_holder_free, name="lens holder free", opacity=1.0, visible=True
        )
        viewer.add_image(sample, name="sample", visible=True)
        viewer.add_labels(
            (area_of_interest * 9).astype("uint8"),
            name="area of interest",
            opacity=0.34,
            visible=False,
        )
        viewer.add_labels(
            (hole_to_interpolate * 36).astype("uint8"),
            name="hole to interpolate",
            opacity=0.34,
            visible=True,
        )

        # top_k_values, top_k_indices, ring_window, scores = get_top_k_slices_perpendicular_radial(
        #     preproc_data,
        #     k=k,
        #     window_size=density_range,
        #     stride=1,
        #     batch_size=-1,
        #     device="cpu",
        #     return_numpy=True,
        #     angle_bins=12,
        #     density_percentile=0.1,
        #     generate_mask=True,
        #     window_label=10,
        #     center_label=6,
        #     return_scores=True
        # )

        # top_k_values, top_k_indices, ring_window = get_top_k_slices_radial_symmetry(
        #     preproc_data,
        #     k=k,
        #     window_size=density_range,
        #     stride=1,
        #     batch_size=-1,
        #     device="cpu",
        #     return_numpy=True,
        #     angle_bins=12,
        #     density_percentile=0.5,
        #     generate_mask=True,
        #     window_label=10,
        #     center_label=6,
        # )
        # with np.printoptions(threshold=np.inf):
        #     print(f'Scores: {scores}\n')
        #     print(f"Max score: {scores.max()} @ idx {scores.argmax()}\n")
        # print(f"Top {k} Values: {top_k_values}\nTop {k} Indices: {top_k_indices}\n")

        # viewer.add_labels(ring_window, name=f"{id}_ring_window")
        # viewer.add_labels(density_range_label,name=f"{id}_density_range_label")


if __name__ == "__main__":
    # Create the widget and show it
    viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    viewer.window.add_dock_widget(my_widget)
    viewer.show()
    napari.run()
    # my_widget.show(run=True)
