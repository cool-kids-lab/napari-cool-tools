"""
"""

from pathlib import Path
from typing import Literal

from magicgui.widgets import create_widget, Container, FileEdit, FloatSpinBox, LineEdit, PushButton
from magicgui import magicgui
import napari
# from napari.utils.notifications import show_info
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_registration import CurvCorrectSettings
from napari_cool_tools_registration._fitting_funcs import sphere_fit_thick_map_corrected_v2

class FileGeneratorWidget(Container):
    """A magicgui widget to generate and step through files in a directory."""

    def __init__(self):
        super().__init__()

        # Widgets for user input
        self.topology_directory_input = FileEdit(
            label="Choose a Topology map directory",
            mode='d',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
            #value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.mask_directory_input = FileEdit(
            label="Choose a data mask directory",
            mode='d',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"E:\_Beth_Thickness_Calculations\Ridge Labels")
        )
        self.output_directory_input = FileEdit(
            label="Choose an output directory",
            mode='d',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
            #value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.output_file_name = LineEdit(
            label="Choose a output directory",
            value="ridge_heights.csv"
        )
        self.topology_extension_input = LineEdit(
            label="Topology File extension",
            value="*_retina_nn_topo_map.npy"
        )
        self.mask_extension_input = LineEdit(
            label="Mask File extension",
            value="*_processed_en_face_ridge_labels.npy"
        )
        self.mask_type = LineEdit(
            label="Mask type",
            value="ridge"
        )
        self.outlier_cutoff = FloatSpinBox(
            label="Outlier Cutoff",
            value=500.,
        )
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

        # Layout the widgets
        self.extend([
            self.topology_directory_input,
            self.mask_directory_input,
            self.output_directory_input,
            self.output_file_name,
            self.topology_extension_input,
            self.mask_extension_input,
            self.mask_type,
            self.outlier_cutoff,
            self.generate_button,
            self.next_file_button,
            self.process_all_files_button
        ])

        # State for the file generator
        self.file_generator = None

        # Connect button signals to methods
        self.generate_button.clicked.connect(self._generate_output_file_path)
        self.generate_button.clicked.connect(self._create_generator)
        self.generate_button.clicked.connect(self._generate_mask_list)
        self.next_file_button.clicked.connect(self._next_file)
        self.process_all_files_button.clicked.connect(self._process_all_files)

    def _generate_output_file_path(self):
        """"""
        # output file path
        self.output_file_path = self.output_directory_input.value / self.output_file_name.value

    def _create_generator(self):
        """Creates a generator for files with the specified extension."""
        directory = self.topology_directory_input.value
        extension = self.topology_extension_input.value
        self.extension = extension

        if not directory or not Path(directory).is_dir():
            print("Error: Invalid directory.")
            return

        # Create a generator for the files
        self.file_generator = (
            f for f in Path(directory).rglob(extension)
            if f.is_file()
        )
        print(f"Generator created for '{extension}' files in {directory}.")

        self.file_list = list(
            f for f in Path(directory).rglob(extension)
            if f.is_file()
        )
        print(f"List containing {len(self.file_list)} entries created for '{extension}' files in {directory}.")

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            next_file = next(self.file_generator)
            print(str(next_file))

            if next_file.suffix == ".npz":
                label_data,attributes,layer_type = npz_file_reader(next_file,return_layer=True,verbose=True)[0]
            elif next_file.suffix == ".npy":
                label_data = np.load(next_file)

            # TODO add viewer back
            #viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    def _generate_mask_list(self):
        # Get mask list
        directory = self.mask_directory_input.value
        extension = self.mask_extension_input.value
        self.mask_list = list(
            f for f in Path(self.mask_directory_input.value).rglob(extension)
            if f.is_file()
        )
        print(f"List containing {len(self.file_list)} mask entries created for '{extension}' files in {directory}.")

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

            print(self.dataframe)
            print(self.dataframe.filter(pl.col("outlier") == 0))
            print(self.dataframe.filter(pl.col("outlier") == 1))    
            print(
                self.dataframe.filter(
                    pl.col("outlier") == 0,
                ).select
                (
                    pl.min("min_ridge_thickness"),
                    pl.mean("mean_ridge_thickness"),
                    pl.max("max_ridge_thickness")
                )
            )
            print("End of files. Processing complete.")
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_file(self,file_path):
        """"""
        extension = Path(self.extension).stem.replace("*","")
        mask_name = self.mask_type.value
        # get id
        id = file_path.stem.replace(extension,"")
        mask_path = [mask_path for mask_path in self.mask_list if id in str(mask_path)]

        # verify there is a mask
        if len(mask_path) == 0:
            print(f"\n{file_path.stem} has no mask\n")
            return
        else:
            mask_path = mask_path[0]
    
        # load mask and confirm it contains values
        mask = np.load(mask_path) > 0
        if mask.sum() == 0:
            print(f"\n{mask_path.stem} is empty.")
            return
        
        # load topomap
        topo_map = np.load(file_path)
        if (topo_map > 0).sum() == 0:
            print(f"\n{file_path.stem} is empty.")
            return
        
        if topo_map.shape != mask.shape:
            print(f"\n{id}'s topographic map shape {topo_map.shape} and the corresponding mask shape {mask.shape} do not match.")
            return

        target_mask = (topo_map*mask) > 0

        if (target_mask > 0).sum() == 0:
            print(f"\n{file_path.stem} does not overlap with mask {mask_path.stem}.")
            return

        target_values = topo_map[target_mask].reshape(-1)

        if target_values.max() < self.outlier_cutoff.value:
            outlier = 0
        else:
            outlier = 1

        output_df = pl.DataFrame({
            "id": id,
            "outlier": outlier,
            f"min_{mask_name}_thickness": target_values.min().astype(float),
            f"mean_{mask_name}_thickness": target_values.mean().astype(float),
            f"max_{mask_name}_thickness": target_values.max().astype(float),
        })

        self.output_file_path
        if self.output_file_path.exists():
            existing_df = pl.read_csv(self.output_file_path)
            #output_dataframe = pl.concat([existing_dataframe,output_dataframe])
            output_df = existing_df.vstack(output_df).unique(maintain_order=True)


        output_df.write_csv(self.output_file_path)
        self.dataframe = output_df

if __name__ == "__main__":
    # Create the widget and show it
    # viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    # viewer.window.add_dock_widget(my_widget)
    # viewer.show()
    # napari.run()
    my_widget.show(run=True)
