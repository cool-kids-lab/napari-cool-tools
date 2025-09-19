"""
"""

import numpy as np
import napari
from napari.utils.notifications import show_info
import napari_cool_tools_io
from magicgui import magicgui

viewer = napari.Viewer()

#@labels_layer.mouse_drag_callbacks.append
def get_label_info(layer, event):
    # The `event` object provides useful information
    data_coordinates = layer.world_to_data(event.position)
    label_value = layer.get_value(data_coordinates)
    show_info(f"Clicked on label: {label_value} at coordinates: {data_coordinates}")

    yield
    show_info("Drag complete.")

@magicgui(call_button="Toggle Eraser Tool")
def toggle_custom_eraser(custom_eraser: bool = False):
    """
    A magicgui function that toggles the visibility of the "My Image" layer.
    
    The `visible` parameter is automatically converted to a checkbox.
    The function body is executed when the button is toggled.
    """

    # Get active layer
    active_layer = viewer.layers.selection.active

    # You can access viewer objects directly within the decorated function.
    if not custom_eraser:
        show_info("Custom Eraser Tool Activated")
        #custom_eraser = False
        toggle_custom_eraser.custom_eraser.value=True

        active_layer.mouse_drag_callbacks.append(get_label_info)
        
    else:
        show_info("Custom Eraser Tool Deactivated")
        #custom_eraser = False
        toggle_custom_eraser.custom_eraser.value=False
        active_layer.mouse_drag_callbacks.remove(get_label_info)

# Add the widget to the napari viewer as a dockable widget
viewer.window.add_dock_widget(toggle_custom_eraser)

# Start the napari event loop
napari.run()