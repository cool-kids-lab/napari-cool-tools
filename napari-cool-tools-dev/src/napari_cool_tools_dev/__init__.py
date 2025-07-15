__version__ = "0.0.1"

__all__ = ()

import napari
from napari.utils.notifications import show_info
from qtreload import QtReloadWidget

viewer = napari.current_viewer()

## specify list of modules that should be monitored
list_of_modules = [
    "napari",
    "napari_cool_tools_img_proc",
    "napari_cool_tools_io",
    "napari_cool_tools_oct_preproc",
    "napari_cool_tools_registration",
    "napari_cool_tools_segmentation",
    "napari_cool_tools_vol_proc",
]

widget = QtReloadWidget(list_of_modules)

# add the widget to your application (or keep reference to it so it's not garbage collected)
qt_viewer = napari.qt.QtViewer(viewer)
qt_viewer.addWidget(widget)

show_info("\n\nHOT RELOAD ACTIVATED!!\n\n")
# print(f"\n\nHOT RELOAD ACTIVATED!!\n\n")
