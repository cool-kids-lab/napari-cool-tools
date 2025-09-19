from magicgui.widgets import Container, LineEdit, PushButton, CheckBox, FileEdit
from pathlib import Path
import napari
from napari.utils.notifications import show_info, show_error
import numpy as np

def save_labels_as_npz(save_directory: Path, retina_key: str = "retina", choroid_key: str = "choroid", shape_key: str = "shape", compress_bits_flag: bool = True):
    """A custom function that processes inputs and a save directory."""

    # Get active labels layer
    active_layer = viewer.layers.selection.active

    # Check if the active layer is a labels layer and proceed
    if isinstance(active_layer, napari.layers.Labels):
        show_info(f"The active layer is: {active_layer.name}")
        # You can now access the data of the active labels layer
        # For example, to get its data:
        name = active_layer.name
        labels_data = active_layer.data
        shape = labels_data.shape

        # pack label data as bits
        retina_bits = np.packbits(labels_data==1)
        choroid_bits = np.packbits(labels_data==2)

        # create save dictionary
        npz_save_dict = {
            retina_key:retina_bits,
            choroid_key:choroid_bits,
            shape_key:shape
        }

        # get save path
        save_path = save_directory/f"{name}.npz"

        show_info("--- Saving as .npz ---")
        show_info(f"Active Layer: {save_directory}")
        show_info(f"Save Directory: {save_directory}")
        show_info(f"Retina_key: {retina_key}")
        show_info(f"Choroid_key: {choroid_key}")
        show_info(f"Shape_key: {shape_key}")
        show_info(f"Shape: {shape}")
        show_info(f"Compress bits: {compress_bits_flag}")
        show_info(f"Save_path: {save_path}")
        show_info("---------------------------------------")

        # save .npz
        #np.savez(save_path, retina_key=retina_bits,choroid_key=choroid_bits,shape_key=shape)
        np.savez(save_path, **npz_save_dict)
        show_info(f"{save_path} save is complete.")
    else:
        #raise ValueError("The active layer is not a labels layer.\nPlease select a labels layer")
        show_error("The active layer is not a labels layer.\nPlease select a labels layer")

class SaveLabelsAsNpzWidget(Container):
    """A custom magicgui widget with a specific order of inputs."""
    def __init__(self):
        super().__init__()
        
        # Create the widgets
        self.save_directory_picker = FileEdit(
            label="Save Directory",
            mode="d",
            value=Path.home()
        )
        self.retina_input = LineEdit(label="Retina",value="retina")
        self.choroid_key_input = LineEdit(label="Choroid_key",value="choroid")
        self.shape_key_input = LineEdit(label="Shape_key",value="shape")
        self.compress_bits_flag_checkbox = CheckBox(label="Compress Bits", value=True)
        self.run_button = PushButton(text="Save as .npz")

        # Append the widgets to the container in the desired order
        self.append(self.save_directory_picker)
        self.append(self.retina_input)
        self.append(self.choroid_key_input)
        self.append(self.shape_key_input)
        self.append(self.compress_bits_flag_checkbox)
        self.append(self.run_button)

        # Connect the button's clicked event to our handler method
        self.run_button.clicked.connect(self._on_button_clicked)

    def _on_button_clicked(self):
        """This method is called when the run button is clicked."""
        save_directory = self.save_directory_picker.value
        retina_key = self.retina_input.value
        choroid_key = self.choroid_key_input.value
        shape_key = self.shape_key_input.value
        compress_bits_flag_flag = self.compress_bits_flag_checkbox.value
        
        save_labels_as_npz(
            save_directory,
            retina_key,
            choroid_key,
            shape_key,
            compress_bits_flag_flag
        )

if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = SaveLabelsAsNpzWidget()
    viewer.window.add_dock_widget(widget,name="Save Labels as .npz") 
    #widget.show(run=True)
    viewer.show()
    napari.run()
