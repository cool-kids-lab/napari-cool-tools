import pathlib
from pathlib import Path
import time
from magicgui import widgets
import numpy as np
import napari
from napari.utils.notifications import show_info

#from napari_cool_tools_vol_proc._masking_tools_funcs import project_2d_mask
from batch_tools_utils import (
    clean_labels,
    load_bits_labels,
    load_bits_labels_v2,
    fill_ellipse,
)

# 1. Create a dummy text file to process
dummy_file_path = pathlib.Path("dummy.txt")
with open(dummy_file_path, "w") as f:
    f.write("This is the first line.\n")
    f.write("This is the second line.\n")

class FileProcessorWidget(widgets.Container):
    """
    A custom widget with a file input and a button to process the file.
    """

    def __init__(self):
        super().__init__()

        # Create child widgets
        self.file_input = widgets.FileEdit(
            label="Select file:",
            mode="r",  # Set to "r" for reading an existing file
            tooltip="Select a file to process.",
            # value=Path(r"F:\38 peak stage ret_chor crop\UNPs_08874712-2023_09_06-14_02_11_processed_ret_chor_seg.npz") #dummy_file_path # Pre-fill with the dummy file path
            value=Path(
                r"E:\38 peak stage ret_chor crop\UNPs_08874712-2023_09_06-14_02_11_processed_ret_chor_seg.npz"
            ),  # dummy_file_path # Pre-fill with the dummy file path
        )
        self.process_button = widgets.PushButton(text="Run Function")

        # Add the widgets to the container
        self.append(self.file_input)
        self.append(self.process_button)

        # Connect the button's click event to a method
        self.process_button.changed.connect(self._on_button_clicked)

    def _on_button_clicked(self):
        """
        This method is called when the button is pressed.
        It runs the custom function with the current file path.
        """
        start_time = time.time()
        file_path = self.file_input.value

        # generate ellipse masks for clipping data beyond lens
        mask_840x800 = np.zeros((840,800),dtype=bool)
        mask_800x800 = np.zeros((800,800),dtype=bool)
        fill_ellipse(mask_840x800,center=(415,382.5),major_axis_len=(840/800)*700,minor_axis_len=700)
        fill_ellipse(mask_800x800,center=(405,395),major_axis_len=730,minor_axis_len=730)
        masks = (mask_840x800,mask_800x800)

        #label_data = load_bits_labels(file_path,"retina")
        #label_data = load_bits_labels_v2(file_path)
        # retina_mask = label_data == 1
        # choroid_mask = label_data == 2
        # label_data[not retina_mask] = 0
        label_data = np.load(file_path)

        # apply mask
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
                # if isinstance(logger,logging.Logger):
                #     logger.warning(message)
                # return

        imaging_range = 12  # mm
        refractive_index = 1.33
        # threshold values
        gap_threshold = 8
        thickness_theshold = 500  # micrometers
        incedence_allowance = 1 / np.sin(np.pi / 4)
        component_to_pixel_thickness_ratio = 1 / 3  # 1/2 #1/6 #1/8 #1/4 #1/2
        dust_threshold = 1e6

        processed_labels = {}
        # (
        #     raw_depth_map2,
        #     ret_surf_coords2,
        #     rpe_surf_coords2,
        #     thick_map2,
        #     difference_map2,
        #     dust_free

        # )
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
            label_data=label_data,
            imaging_range=imaging_range,
            refractive_index=refractive_index,
            gap_threshold=gap_threshold,
            thickness_threshold=thickness_theshold,
            incedence_allowance=incedence_allowance,
            component_to_pixel_thickness_ratio=component_to_pixel_thickness_ratio,
            dust_threshold=dust_threshold,
            viewer="", #viewer,
        )

        for key,val in processed_labels.items():
            if isinstance(val,np.ndarray):
                print(f"{key}, {val.shape},{val.dtype}\n")
            else:
                print(f"{key}, {val}")

        end_time = time.time()
        elapsed_time = end_time - start_time 
        print(f"Label cleaning took {elapsed_time/60} minutes\n")

        viewer.add_image(processed_labels["depth_map"],name="depth_map")
        viewer.add_image(processed_labels["thick_map"],name="thick_map")
        viewer.add_image(processed_labels["difference_map"],name="differnce_map")
        viewer.add_image(processed_labels["outlier_coordinate_mask"],name="outlier_coordinate_mask")
        viewer.add_points(processed_labels["rpe_surf_coords"],size=4,face_color="red",border_color="orange",name="rpe_surf_coords")
        viewer.add_points(processed_labels["ret_surf_coords"],size=4,face_color="indigo",border_color="blue",name="ret_surf_coords")
        viewer.add_labels(processed_labels["clean_label"],name="clean_label")
        show_info(f"Outlier percentage: {processed_labels['percent_outliers']}\n")

if __name__ == "__main__":
    try:
        viewer = napari.Viewer()
        # Create and show the custom widget
        processor_widget = FileProcessorWidget()
        viewer.window.add_dock_widget(processor_widget)
        viewer.show()
        napari.run()
        # processor_widget.show(run=True)
    finally:
        # Clean up the dummy file when the widget is closed
        if dummy_file_path.exists():
            dummy_file_path.unlink()
