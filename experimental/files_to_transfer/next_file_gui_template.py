from pathlib import Path
from magicgui.widgets import Container, FileEdit, LineEdit, PushButton, Label
from magicgui import magicgui
import napari
from napari.utils.notifications import show_info

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

        # Layout the widgets
        self.extend([
            self.directory_input,
            self.extension_input,
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
