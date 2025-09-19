from pathlib import Path
import logging
from magicgui.widgets import Container, FileEdit, LineEdit, PushButton, Label
from magicgui import magicgui
import napari
from napari.utils import progress
from tqdm import tqdm
from napari.utils.notifications import show_info
import numpy as np

from batch_tools_utils import clean_labels, save_bits_labels, load_bits_labels, fill_ellipse

def process_file(file_path:Path,masks:tuple,logger=''):
    print(f"Processing: {file_path}\n")
    if Path(file_path.parent).exists():
        save_dir_path = Path(f"{file_path.parent}\\processed2")
        save_dir_path.mkdir(parents=True,exist_ok=True)
        save_name = f"{file_path.stem}.npz"
        save_file_path = save_dir_path / save_name
    else:
        print(f"{Path(file_path.parent)} does not exist\nTerminating program")
        return
    if file_path.suffix == ".npy":
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
        # apply lens mask to label data
        # for mask in masks:
        #     if label_data.sum(axis=1).shape == mask.shape:
        #         lens_mask = np.repeat(mask[:,None,:],label_data.shape[1])
        #         label_data[~lens_mask] = 0

        mask_0 = masks[0]
        mask_1 = masks[1]
        match label_data.sum(axis=1).shape:
            case mask_0.shape:
                lens_mask = np.repeat(mask_0[:,None,:],label_data.shape[1],axis=1)
                print(f"\nlens mask shape: {lens_mask.shape}")
                print(f"label data shape: {label_data.shape}")
                label_data[~lens_mask] = 0
            case mask_1.shape:
                lens_mask = np.repeat(mask_1[:,None,:],label_data.shape[1],axis=1)
                print(f"\nlens mask shape: {lens_mask.shape}")
                print(f"label data shape: {label_data.shape}")
                label_data[~lens_mask] = 0
            case _:
                message = f"Data in {file_path} does not match any of the lens masks.\nSkipping this file"
                print(message)
                if isinstance(logger,logging.Logger):
                    logger.warning(message)
                return
        (
            processed_labels["depth_map"],
            processed_labels["ret_surf_coords"],
            processed_labels["rpe_surf_coords"],
            processed_labels["thick_map"],
            processed_labels["difference_map"],
            processed_labels["outlier_coordinate_mask"],
            processed_labels["clean_label"],
            processed_labels["percent_outliers"],
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
    #viewer.add_labels(label_out_data)

    # log high outlier values
    if processed_labels['percent_outliers'] > 0.1:
        message = f"{file_path} contians labels consisting of greater than 10 percent outliers.\n"
        print(message)
        if isinstance(logger,logging.Logger):
            logger.warning(message)

    exit_code = save_bits_labels(save_file_path,label_data=label_out_data)
    if exit_code == 1:
        message = f"{file_path} has no nonzero labels to store. This file will not be saved.\n"
        print(message)
        if isinstance(logger,logging.Logger):
            logger.warning(message)

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
            print("Error: Invalid directory.")
            return

        # Create a generator for the files
        self.file_generator = (
            #f for f in Path(directory).rglob(extension)
            f for f in Path(directory).glob(extension)
            if f.is_file()
        )
        print(f"Generator created for '{extension}' files in {directory}.")

        self.file_list = list(
            #f for f in Path(directory).rglob(extension)
            f for f in Path(directory).glob(extension)
            if f.is_file()
        )
        print(f"List created for '{extension}' files in {directory}.")

    def _next_file(self):
        """Advances the generator and displays the next file."""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            next_file = next(self.file_generator)
            print(str(next_file))
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

    def _process_all_files(self):
        """Processes all files in the generator"""
        if self.file_generator is None:
            print("Please create a generator first.")
            return

        try:
            # setup logging
            logger = logging.getLogger(name="clean_segmentations_logger")
            logger.setLevel(logging.DEBUG)
            save_dir_path = Path(f"{self.file_list[0].parent}\\processed2")
            save_dir_path.mkdir(parents=False,exist_ok=True)
            log_file_path = save_dir_path/"empty_labels.log"
            #log_file_path = Path(f"{self.file_list[0].parent}\\processed2")/"empty_labels.log"
            handler = logging.FileHandler(log_file_path, mode='a')
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # get list of files to skip from the log file
            file_stems_to_skip = []
            with open(log_file_path,"r") as f:
                for line in f:
                    if "This file will not be saved." in line:
                        #end_idx = line.find("_processed")
                        # get file stems
                        end_idx = line.find(".npy")
                        start_idx = line.find("UNPs")
                        file_stems_to_skip.append(line[start_idx:end_idx])
                        #print(line[start_idx:end_idx])

            # generate ellipse masks for clipping data beyond lens
            mask_840x800 = np.zeros((840,800),dtype=bool)
            mask_800x800 = np.zeros((800,800),dtype=bool)
            fill_ellipse(mask_840x800,center=(415,382.5),major_axis_len=(840/800)*700,minor_axis_len=700)
            fill_ellipse(mask_800x800,center=(405,395),major_axis_len=730,minor_axis_len=730)
            masks = (mask_840x800,mask_800x800)

            #for file_path in progress(self.file_generator,desc="Processing_Files"):
            #for file_path in tqdm(self.file_generator,desc="Processing_Files"):
            for file_path in tqdm(reversed(self.file_list),desc="Processing_Files"):
                save_file_path = Path(f"{file_path.parent}\\processed2\\{file_path.with_suffix('.npz').name}")
                #print(save_file_path)
                if "_ret_chor_seg.npy" in str(file_path):
                    if not save_file_path.exists() and file_path.stem not in file_stems_to_skip:
                        process_file(file_path=file_path,masks=masks,logger=logger)
                        #pass
                        #print(f"processing {file_path}\n")
                    else:
                        print(f"skipping {file_path}\n")
            print("End of files. Processing complete.")
        except StopIteration:
            print("End of files.")
            self.file_generator = None  # Reset the generator

if __name__ == "__main__":
    # Create the widget and show it
    # viewer = napari.Viewer()
    my_widget = FileGeneratorWidget()
    # viewer.window.add_dock_widget(my_widget)
    # viewer.show()
    # napari.run()
    my_widget.show(run=True)
