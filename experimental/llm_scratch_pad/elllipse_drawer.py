import napari
import numpy as np
from magicgui import magicgui
from skimage.draw import ellipse_perimeter
from napari.utils.events import Event

@magicgui(
    call_button="Activate Ellipse Tool",
    minor_major_ratio={'widget_type': 'FloatSlider', 'min': 0.1, 'max': 1.0, 'step': 0.05}
)
def interactive_ellipse_border(viewer: napari.Viewer, minor_major_ratio: float = 0.5):
    """
    A magicgui widget for interactively drawing an ellipse border on a labels layer.
    """
    if not any(isinstance(layer, napari.layers.Labels) for layer in viewer.layers):
        print("Please add or select a Labels layer first.")
        return

    labels_layer = viewer.layers.selection.active
    if not isinstance(labels_layer, napari.layers.Labels):
        print("Please select a Labels layer to draw on.")
        return

    # A dictionary to store state during the drawing process
    state = {'is_drawing': False, 'start_point': None, 'preview_layer': None}

    def _mouse_press_callback(layer, event):
        """Called when the mouse button is pressed."""
        if not state['is_drawing'] or not labels_layer.visible:
            return

        state['start_point'] = np.round(event.position).astype(int)
        
        # Create a temporary shapes layer for preview
        if state['preview_layer'] is None:
            state['preview_layer'] = viewer.add_shapes(
                data=None,
                shape_type='ellipse',
                edge_width=2,
                edge_color=labels_layer.colormap[labels_layer.selected_label],
                face_color='transparent',
                name='ellipse_preview'
            )
            state['preview_layer'].editable = False
        else:
            state['preview_layer'].data = []
        
        state['preview_layer'].add_ellipse(
            center=state['start_point'],
            radii=(1, 1),
            edge_color=labels_layer.colormap[labels_layer.selected_label],
        )

    def _mouse_move_callback(layer, event):
        """Called when the mouse is moved while drawing."""
        if not state['is_drawing'] or not labels_layer.visible or state['start_point'] is None:
            return
        
        current_pos = np.round(event.position).astype(int)
        start_point = state['start_point']
        
        # Calculate major axis and orientation
        major_axis_vector = current_pos - start_point
        major_axis_length = np.linalg.norm(major_axis_vector)
        if major_axis_length == 0:
            return
            
        orientation = np.arctan2(major_axis_vector[0], major_axis_vector[1])
        
        # Calculate minor axis based on the ratio
        minor_axis_length = major_axis_length * minor_major_ratio
        
        # Update the preview shape
        state['preview_layer'].data[0] = {
            'center': start_point,
            'radii': np.array([major_axis_length, minor_axis_length]),
            'rotation': orientation
        }

    def _mouse_release_callback(layer, event):
        """Called when the mouse button is released."""
        if not state['is_drawing'] or not labels_layer.visible or state['start_point'] is None:
            return

        end_point = np.round(event.position).astype(int)
        center = state['start_point']

        # Calculate major axis and orientation
        major_axis_vector = end_point - center
        major_axis_length = np.linalg.norm(major_axis_vector)
        if major_axis_length == 0:
            # Draw a single point if there was no drag
            labels_layer.paint(center, labels_layer.selected_label)
        else:
            orientation = np.arctan2(major_axis_vector[0], major_axis_vector[1])

            # Calculate minor axis based on the ratio
            minor_axis_length = major_axis_length * minor_major_ratio

            # Generate the ellipse pixel coordinates and draw on the labels layer
            rr, cc = ellipse_perimeter(
                r=center[0], c=center[1],
                r_radius=minor_axis_length, c_radius=major_axis_length,
                orientation=orientation, shape=labels_layer.data.shape
            )
            labels_layer.data[rr, cc] = labels_layer.selected_label
            labels_layer.refresh()

        # Clean up temporary state
        state['start_point'] = None
        if state['preview_layer'] in viewer.layers:
            viewer.layers.remove(state['preview_layer'])
            state['preview_layer'] = None
        
        # Re-add mouse callbacks to re-enable drawing
        _connect_callbacks()

    def _connect_callbacks():
        """Connects the mouse callbacks to the labels layer."""
        labels_layer.mouse_press_callbacks.append(_mouse_press_callback)
        labels_layer.mouse_move_callbacks.append(_mouse_move_callback)
        labels_layer.mouse_release_callbacks.append(_mouse_release_callback)

    def _disconnect_callbacks():
        """Disconnects the mouse callbacks from the labels layer."""
        labels_layer.mouse_press_callbacks.remove(_mouse_press_callback)
        labels_layer.mouse_move_callbacks.remove(_mouse_move_callback)
        labels_layer.mouse_release_callbacks.remove(_mouse_release_callback)
    
    @viewer.layers.selection.events.active.connect
    def _on_active_layer_changed(event: Event):
        """Handles changes in the active layer to manage callbacks."""
        if state['is_drawing']:
            if event.value is not labels_layer:
                _disconnect_callbacks()
            else:
                _connect_callbacks()

    @viewer.layers.events.removed.connect
    def _on_layer_removed(event: Event):
        """Handles the case where the labels layer is removed."""
        if state['is_drawing'] and event.value is labels_layer:
            interactive_ellipse_border.reset_choices() # Reset the magicgui state
            _disconnect_callbacks()
            state['is_drawing'] = False
    
    # Toggle drawing state and connect/disconnect callbacks
    state['is_drawing'] = not state['is_drawing']
    if state['is_drawing']:
        _connect_callbacks()
        print("Interactive ellipse drawing tool activated. Click and drag on the labels layer to draw.")
    else:
        _disconnect_callbacks()
        if state['preview_layer'] in viewer.layers:
            viewer.layers.remove(state['preview_layer'])
        state['preview_layer'] = None
        state['start_point'] = None
        print("Interactive ellipse drawing tool deactivated.")

def main():
    """Starts a napari viewer with an interactive ellipse border drawing tool."""
    #with napari.Viewer() as viewer:
    viewer = napari.Viewer()
    # Add a sample labels layer to draw on
    labels_layer = viewer.add_labels(np.zeros((512, 512), dtype=int))
    labels_layer.selected_label = 1
    
    # Add the custom widget to the viewer
    viewer.window.add_dock_widget(interactive_ellipse_border)
    
    # Run the napari event loop
    viewer.show()
    napari.run()

if __name__ == '__main__':
    main()