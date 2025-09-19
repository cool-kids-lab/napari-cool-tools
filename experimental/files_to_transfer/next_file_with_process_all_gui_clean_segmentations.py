from pathlib import Path
from magicgui.widgets import Container, FileEdit, LineEdit, PushButton, Label
from magicgui import magicgui
import napari
from napari.utils import progress
from tqdm import tqdm
from napari.utils.notifications import show_info
import numpy as np

from batch_tools_utils import clean_labels, save_bits_labels, load_bits_labels

def process_file(file_path:Path):
    show_info(f"Processing: {file_path}\n")
    if Path(file_path.parent).exists():
        save_dir_path = Path(f"{file_path.parent}\\processed")
        save_dir_path.mkdir(parents=True,exist_ok=True)
        save_name = f"{file_path.stem}.npz"
        save_file_path = save_dir_path / save_name
    else:
        show_info(f"{Path(file_path.parent)} does not exist\nTerminating program")
        return
    if Path.suffix == ".npy":
        label_data = np.load(file_path)
    elif file_path.suffix == ".npz":
        label_data = load_bits_labels(file_path,key="retina",shape_key="shape",value=1)
        label_data = label_data + load_bits_labels(file_path,key="choroid",shape_key="shape",value=2)

    label_values = np.unique(label_data)
    non_zero_labels = label_values > 0
    label_values = label_values[non_zero_labels]

    label_out_data = np.zeros_like(label_data)
    for label_value in tqdm(label_values,desc="processing labels in layer"):
        processed_labels = {}
        (
            processed_labels["depth_map"],
            processed_labels["ret_surf_coords"],
            processed_labels["rpe_surf_coords"],
            processed_labels["thick_map"],
            processed_labels["difference_map"],
            processed_labels["outlier_coordinate_mask"],
            processed_labels["clean_label"],
        ) = clean_labels(
            label_data=label_data == label_value,
            imaging_range=12.0,
            refractive_index=1.33,
            gap_threshold=8,
            thickness_threshold=600,
            incedence_allowance=1 / np.sin(np.pi / 4),
            component_to_pixel_thickness_ratio = (1 / 3),
            dust_threshold=1.0e6,            
        )
        label_out_data = (processed_labels["clean_label"] > 0)*label_value + label_out_data
    viewer.add_labels(label_out_data)
    save_bits_labels(save_file_path,label_data=label_out_data)

class FileGeneratorWidget(Container):
    """A magicgui widget to generate and step through files in a directory."""

    def __init__(self):
        super().__init__()

        # Widgets for user input
        self.directory_input = FileEdit(
            label="Choose a directory",
            mode='d',  # 'd' specifies directory selection
            value=Path.home()
        )
        self.extension_input = LineEdit(
            label="File extension",
            value="*.txt"
        )
        self.generate_button = PushButton(
            text="Create File Generator"
        )

        # Widget to display the current file
        self.next_file_button = PushButton(
            text="Next File"
        )
        #self.current_file_label = Label(value="No file selected.")

        # Widget to process all files
        self.process_all_files_button = PushButton(
            text="Process All Files"
        )

        # Layout the widgets
        self.extend([
            self.directory_input,
            self.extension_input,
            self.generate_button,
            self.next_file_button,
            self.process_all_files_button
            #self.current_file_label
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

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            show_info("Please create a generator first.")
            return

        try:
            next_file = next(self.file_generator)
            show_info(str(next_file))
        except StopIteration:
            show_info("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_all_files(self):
        """Processes all files in the generator"""
        if self.file_generator is None:
            show_info("Please create a generator first.")
            return

        try:
            #for file_path in progress(self.file_generator,desc="Processing_Files"):
            for file_path in tqdm(self.file_generator,desc="Processing_Files"):
                process_file(file_path=file_path)
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
