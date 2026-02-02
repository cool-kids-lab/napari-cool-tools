"""
"""

from pathlib import Path
from typing import Literal

from magicgui.widgets import create_widget, Container, FileEdit, FloatSpinBox, LineEdit, PushButton
from magicgui import magicgui
import napari
from napari.utils.notifications import show_info
import numpy as np
import torch
from tqdm import tqdm

from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration import CurvCorrectSettings
from napari_cool_tools_registration._fitting_funcs import sphere_fit_thick_map_corrected_v2

def calculate_nearest_neighbor_thickness(tissue_label:np.ndarray|torch.Tensor,cc_settings:CurvCorrectSettings,viewer:None,verbose:bool=False):
    """"""
    if verbose:
        show_info(f"\ntissue_label shape,dtype: {tissue_label.shape,tissue_label.dtype}\nCurve Correction settings:\n{cc_settings}\n")

    # calculate values for curve correction
    # TODO clairify 
    imaging_range_in_water = cc_settings.imaging_range / cc_settings.refractive_index
    pixel_spacing = imaging_range_in_water / tissue_label.shape[1]
    if verbose:
        print(f"imaging range in water / A-scan pixels = pixel spacing: {imaging_range_in_water} / {tissue_label.shape[1]} = {pixel_spacing}\n")
        #print(f"pixel_spacing: {pixel_spacing}\n")

    base_padding = cc_settings.pivot_point-imaging_range_in_water

    reference_arm_shift = ( cc_settings.imaging_motor_position - cc_settings.imaging_motor_position_delta) - cc_settings.reference_motor_position
    reference_arm_shift_in_water = (reference_arm_shift * 0.5) / cc_settings.refractive_index 
    if verbose:
        print(f"(imaging motor position - imaging motor position delta) - reference motor postition = raw refereence arm shift: ({cc_settings.imaging_motor_position} - {cc_settings.imaging_motor_position_delta}) - {cc_settings.reference_motor_position} = {reference_arm_shift}\n")
        print(f"(raw reference arm shift / 2) / refractive index = refereence arm shift in water: ({reference_arm_shift} * 0.5) / {cc_settings.refractive_index} = {reference_arm_shift_in_water}\n")

    padding = base_padding + reference_arm_shift_in_water
    if verbose:
        print(f"base_padding + reference arm shift in air = padding: {base_padding} + {reference_arm_shift_in_water} = {padding}\n")

    padding_pixel = int(padding / pixel_spacing)
    if verbose:
        print(f"padding / pixel spacing = padding pixels: {padding / pixel_spacing} = {padding_pixel}\n")

    # get surface points and curve correct there positions
    if tissue_label.sum() > 0:
        (
            thickness_map, retina_points, rpe_points, curv_ret_points, curv_rpe_points, raw_pixel_thickness_map, pixel_thickness_map
        ) = sphere_fit_thick_map_corrected_v2(tissue_label,pixel_spacing=pixel_spacing,padding_pixel=padding_pixel,refractive_index=cc_settings.refractive_index)
        #) = sphere_fit_thick_map_corrected(retina_data,imaging_motor_position=cc_settings.imaging_motor_position-cc_settings.imaging_motor_position_delta,pivot_point=cc_settings.pivot_point,scan_angle=cc_settings.scan_angle)
    else:
        return
    
    # micron_conv_factor = pixel_spacing * 1000 / cc_settings.refractive_index
    # curv_rpe_points_microns = curv_rpe_points*micron_conv_factor
    # curv_ret_points_microns = curv_ret_points*micron_conv_factor
    
    viewer.add_points(curv_ret_points,size=0.5,face_color="magenta",border_color="magenta",name="retina",blending="translucent_no_depth")
    viewer.add_points(curv_rpe_points,size=0.5,face_color="yellow",border_color="yellow",name="rpe",blending="translucent_no_depth")

    return

def process_file(file_path:Path,cc_settings:CurvCorrectSettings,target_label:int,viewer:None): #,logger=''):
    """"""
    label_data,attributes,layer_type = npz_file_reader(path=file_path,return_layer=True,verbose=False)[0]
    name = attributes["name"]

    if "motor_position" in attributes["metadata"]:
        cc_settings.imaging_motor_position = float(attributes["metadata"]["motor_position"])/1000.

    else:
        show_info(f"Skipping {name} because this file lacks motor_position data.")
        return
    
    # isolate retina data
    tissue_label = label_data == target_label
    calculate_nearest_neighbor_thickness(tissue_label,cc_settings=cc_settings,viewer=viewer,verbose=False)
    return

class FileGeneratorWidget(Container):
    """A magicgui widget to generate and step through files in a directory."""

    def __init__(self):
        super().__init__()

        # Widgets for user input
        self.directory_input = FileEdit(
            label="Choose a directory",
            mode='d',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2")
        )
        self.extension_input = LineEdit(
            label="File extension",
            value="*_ret_chor_seg_clean.npz"
        )
        self.selected_tissue = create_widget(
            annotation=Literal["retina","choroid"],
            label="Tissue Options",
            value="retina",
        )
        self.imaging_range = FloatSpinBox(
            label="Imaging Range",
            value=12.,
        )
        self.imaging_motor_position_delta = FloatSpinBox(
            label="Imaging motor position offset",
            value=6.,
        )
        self.generate_button = PushButton(
            text="Create File Generator"
        )

        # Widget to display the current file
        self.next_file_button = PushButton(
            text="Next File"
        )

        # Widget to process all files
        self.process_all_files_button = PushButton(
            text="Process All Files"
        )

        self.tissue_options = ("retina","choroid")

        # Layout the widgets
        self.extend([
            self.directory_input,
            self.extension_input,
            self.selected_tissue,
            self.imaging_range,
            self.imaging_motor_position_delta,
            self.generate_button,
            self.next_file_button,
            self.process_all_files_button
        ])

        # State for the file generator
        self.file_generator = None

        # Connect button signals to methods
        self.generate_button.clicked.connect(self._create_generator)
        self.next_file_button.clicked.connect(self._next_file)
        self.process_all_files_button.clicked.connect(self._process_all_files)

    def _create_generator(self):
        """Creates a generator for files with the specified extension."""
        directory = self.directory_input.value
        extension = self.extension_input.value

        if not directory or not Path(directory).is_dir():
            show_info("Error: Invalid directory.")
            return

        # Create a generator for the files
        self.file_generator = (
            f for f in Path(directory).rglob(extension)
            if f.is_file()
        )
        show_info(f"Generator created for '{extension}' files in {directory}.")

        self.file_list = list(
            f for f in Path(directory).rglob(extension)
            if f.is_file()
        )
        show_info(f"List containing {len(self.file_list)} entries created for '{extension}' files in {directory}.")

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            show_info("Please create a generator first.")
            return

        try:
            next_file = next(self.file_generator)
            show_info(str(next_file))

            if next_file.suffix == ".npz":
                label_data,attributes,layer_type = npz_file_reader(next_file,return_layer=True,verbose=True)[0]
            elif next_file.suffix == ".npy":
                label_data = np.load(next_file)

            viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])
        except StopIteration:
            show_info("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_all_files(self):
        """Processes all files in the generator"""
        if self.file_generator is None:
            show_info("Please create a generator first.")
            return

        try:
            # # setup logging
            # logger = logging.getLogger(name="clean_segmentations_logger")
            # logger.setLevel(logging.DEBUG)
            # save_dir_path = Path(f"{self.file_list[0].parent}\\processed2")
            # save_dir_path.mkdir(parents=False,exist_ok=True)
            # log_file_path = save_dir_path/"empty_labels.log"
            # #log_file_path = Path(f"{self.file_list[0].parent}\\processed2")/"empty_labels.log"
            # handler = logging.FileHandler(log_file_path, mode='a')
            # handler.setLevel(logging.DEBUG)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            # handler.setFormatter(formatter)
            # logger.addHandler(handler)

            # # get list of files to skip from the log file
            # file_stems_to_skip = []
            # with open(log_file_path,"r") as f:
            #     for line in f:
            #         if "This file will not be saved." in line:
            #             #end_idx = line.find("_processed")
            #             # get file stems
            #             end_idx = line.find(".npy")
            #             start_idx = line.find("UNPs")
            #             file_stems_to_skip.append(line[start_idx:end_idx])
            #             #print(line[start_idx:end_idx])

            #for file_path in progress(self.file_generator,desc="Processing_Files"):

            # initialize curve correction settings
            cc_settings = CurvCorrectSettings(
                imaging_range=self.imaging_range.value,
                imaging_motor_position_delta=self.imaging_motor_position_delta.value
            )

            target_label:int = self.tissue_options.index(self.selected_tissue.value)

            file_path_pbar = tqdm(self.file_generator)
            for file_path in file_path_pbar:
                file_path_pbar.set_description(f"Processing: {file_path}")
            #for file_path in tqdm(reversed(self.file_list),desc="Processing_Files"):
                # save_file_path = Path(f"{file_path.parent}\\processed2\\{file_path.with_suffix('.npz').name}")

                # if not save_file_path.exists() and file_path.stem not in file_stems_to_skip:
                process_file(file_path=file_path,cc_settings=cc_settings,target_label=target_label,viewer=viewer) # ,logger=logger)

                # else:
                #     print(f"skipping {file_path}\n")
            show_info("End of files. Processing complete.")
        except StopIteration:
            show_info("End of files.")
            self.file_generator = None  # Reset the generator

if __name__ == "__main__":
    # Create the widget and show it
    viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    viewer.window.add_dock_widget(my_widget)
    viewer.show()
    napari.run()
    #my_widget.show(run=True)
