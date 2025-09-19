from pathlib import Path
import numpy as np
from magicgui.widgets import Container, FileEdit, LineEdit, PushButton, Label
from magicgui import magicgui
import napari
from napari.utils.notifications import show_info
from batch_tools_utils import load_bits_labels_v2

def load_bits_labels(file_path:Path,key,shape_key="shape",value:int=1):
    """
    """
    npzfile = np.load(file_path)
    bits_label = npzfile[key]
    shape = npzfile[shape_key]
    label = np.unpackbits(bits_label)*value
    return label.reshape(shape)

class FileGeneratorWidget(Container):
    """A magicgui widget to generate and step through files in a directory."""

    def __init__(self):
        super().__init__()

        # Widgets for user input
        self.directory_input = FileEdit(
            label="Choose a directory",
            mode='d',  # 'd' specifies directory selection
            #value=Path.home()
            value=Path(r"E:\38 peak stage ret_chor crop")
        )
        self.extension_input = LineEdit(
            label="File extension",
            value="*.npz"
        )
        self.retina_key = LineEdit(
            label="Retina Key",
            value="retina"
        )
        self.choroid_key = LineEdit(
            label="Choroid Key",
            value="choroid"
        )
        self.shape_key = LineEdit(
            label="Shape Key",
            value="shape"
        )
        self.generate_button = PushButton(
            text="Create File Generator"
        )

        # Widget to display the current file
        self.next_file_button = PushButton(
            text="Next File"
        )
        #self.current_file_label = Label(value="No file selected.")

        self.current_shape:tuple = ()

        # Layout the widgets
        self.extend([
            self.directory_input,
            self.extension_input,
            self.retina_key,
            self.choroid_key,
            self.shape_key,
            self.generate_button,
            self.next_file_button,
            #self.current_file_label
        ])

        # State for the file generator
        self.file_generator = None

        # Connect button signals to methods
        self.generate_button.clicked.connect(self._create_generator)
        self.next_file_button.clicked.connect(self._next_file)

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
            retina_key = self.retina_key.value
            choroid_key = self.choroid_key.value
            shape_key = self.shape_key.value
            print(retina_key,choroid_key,shape_key)
            #retina=load_bits_labels(next_file,retina_key,shape_key)
            #choroid=load_bits_labels(next_file,choroid_key,shape_key,value=2)
            if next_file.suffix == ".npz":
                label_data,metadata,layer_type = load_bits_labels_v2(next_file)
            elif next_file.suffix == ".npy":
                label_data = np.load(next_file)
            #viewer.add_labels(retina,name=self.directory_input.value)
            #viewer.add_labels(choroid,name=self.directory_input.value)
            #viewer.add_labels(retina+choroid,name=next_file.stem)
            show_info(f"{label_data.shape}")
            viewer.add_labels(label_data,name=next_file.stem) #,name=metadata["name"],properties=metadata["properties"])


            # if self.current_shape != label_data.shape:
            #     self.current_shape = label_data.shape
            #     viewer.add_labels(label_data,name=metadata["name"],properties=metadata["properties"])
            # else:
            #     while self.current_shape == label_data.shape:
            #         next_file = next(self.file_generator)

            #     self.current_shape = label_data.shape
            #     viewer.add_labels(label_data,name=metadata["name"],properties=metadata["properties"])

            # layer = napari.layers.Layer.create((label_data,metadata,layer_type))
            # viewer.add_layer(layer)
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
