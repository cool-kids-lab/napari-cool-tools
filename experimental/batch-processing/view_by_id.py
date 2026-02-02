"""
"""

from pathlib import Path
from typing import Literal

import cv2
from magicgui.widgets import create_widget, Container, FileEdit, FloatSpinBox, LineEdit, PushButton
from magicgui import magicgui
import napari
from napari.layers import Layer
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
        self.search_directory_input = FileEdit(
            label="Choose a file directory to search",
            mode='d',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output")
            #value=Path(r"E:\_rebatch_conjugate_test_12_01_2025\Temp_Thickness_Output")
        )
        self.target_file_list_input = FileEdit(
            label="Choose a .csv file containing the desired scan ids",
            mode='r',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"\\192.168.1.3\coolkid\Beth Roti\Ridge Height Output\missing_topo_paths.csv")
        )
        self.extension_input = LineEdit(
            label="Limit search to file extension",
            value="*.*"
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

        # Layout the widgets
        self.extend([
            self.search_directory_input,
            self.target_file_list_input,
            self.generate_button,
            self.next_file_button,
            self.process_all_files_button
        ])

        # output file path
        # self.search_dataframe = pl.read_csv(self.target_file_list_input.value)

        # State for the file generator
        self.file_generator = None

        # Connect button signals to methods
        self.generate_button.clicked.connect(self._create_generator)
        self.next_file_button.clicked.connect(self._next_file)

    def _create_generator(self):
        """Creates a generator for files with the specified extension."""
        directory = self.search_directory_input.value
        extension = self.extension_input.value
        self.extension = extension
        self.search_dataframe = pl.read_csv(self.target_file_list_input.value)
        target_id_list = self.search_dataframe.to_series().to_list()
        self.target_id_generator = (id for id in target_id_list)


        if not directory or not Path(directory).is_dir():
            print("Error: Invalid directory.")
            return

        # Create a generator for the files
        self.file_generator = (file for file in Path(directory).rglob(extension) if any(val in file.stem for val in target_id_list)) 
        # self.file_generator = (
        #     f for f in Path(directory).rglob(extension)
        #     if f.is_file()
        # )
        print(f"Generator created for '{extension}' files in {directory}.")


        self.file_list = list(file for file in Path(directory).rglob(extension) if any(val in file.stem for val in target_id_list)) 
        # self.file_list = list(
        #     f for f in Path(directory).rglob(extension)
        #     if f.is_file()
        # )
        self.file_list.sort()
        self.sorted_file_generator = (file for file in self.file_list)
        print(f"List containing {len(self.file_list)} entries created for '{extension}' files in {directory}.")

    # def _next_file(self):
    #     """Advances the generator and displays the next file."""
    #     if self.file_generator is None:
    #         print("Please create a generator first.")
    #         return

    #     try:
    #         next_file = next(self.file_generator)
    #         next_file = next(self.sorted_file_generator)
    #         print(str(next_file))

    #         if next_file.suffix == ".npz":
    #             loaded_data,attributes,layer_type = npz_file_reader(next_file,return_layer=True,verbose=True)[0]
    #         elif next_file.suffix == ".npy":
    #             layer_type = "image"
    #             loaded_data = np.load(next_file)
    #         elif next_file.suffix == ".png":
    #             layer_type = "image"
    #             loaded_data = cv2.imread(next_file,cv2.IMREAD_GRAYSCALE)

    #         # TODO add viewer back
    #         loaded_layer = Layer.create(loaded_data,meta={"name":next_file.stem},layer_type=layer_type)
    #         viewer.add_layer(loaded_layer)
    #         #viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])
    #     except StopIteration:
    #         print("End of files.")
    #         self.file_generator = None  # Reset the generator

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            id = next(self.target_id_generator)
           
            id_paths = [path for path in self.file_list if id in str(path)]
            for id_path in id_paths:

                if id_path.suffix == ".npz":
                    loaded_data,attributes,layer_type = npz_file_reader(id_path,return_layer=True,verbose=True)[0]
                elif id_path.suffix == ".npy":
                    layer_type = "image"
                    loaded_data = np.load(id_path)
                elif id_path.suffix == ".png":
                    layer_type = "image"
                    loaded_data = cv2.imread(id_path,cv2.IMREAD_GRAYSCALE)

                # TODO add viewer back
                loaded_layer = Layer.create(loaded_data,meta={"name":id_path.stem},layer_type=layer_type)
                viewer.add_layer(loaded_layer)
                #viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

if __name__ == "__main__":
    # Create the widget and show it
    viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    viewer.window.add_dock_widget(my_widget)
    viewer.show()
    napari.run()
    # my_widget.show(run=True)
