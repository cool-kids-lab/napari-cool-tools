from napari.layers import Image, Layer
from napari_cool_tools_io import viewer
from napari_cool_tools_registration._bidirectional_bscan_registration_widget import Bidirectional_Bscan_Registration_Widget
from napari.utils.notifications import show_info, show_warning, show_error
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt

def bidirectional_bscan_registration_plugin(
    vol: Image,
):
    """Bidirectional A-scan registration plugin for napari.
    
    """

    if vol is None:
        # raise ValueError("No input volume provided.")
        show_error("No input volume provided.")
        return

    if vol.ndim != 3:
        show_error("Input volume must be 3D.")
        return
    

    print("Bidirectional B-scan registration plugin called")

    dialog = Bidirectional_Bscan_Registration_Widget(parent=viewer.window._qt_window)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    dialog.setWindowModality(Qt.NonModal)   # make sure it's non-modal
    dialog.setModal(False)                  # redundant but explicit
    dialog.set_input_volume(vol.data)

    def on_reg_dialog_accepted(dialog):
        output_volume = dialog.get_output_volume()

        add_kwargs = {"name": vol.name + "_registered"}
        layer_type = "image"
        bscan_layers = Layer.create(output_volume, add_kwargs, layer_type)
        vmin, vmax = np.percentile(output_volume, (1, 99))
        bscan_layers.contrast_limits = (float(vmin), float(vmax))
        viewer.add_layer(bscan_layers)

        if dialog.mipCheckBox.isChecked():
            enface_image = np.max(output_volume, axis=1)
        else:
            enface_image = np.mean(output_volume, axis=1)

        add_kwargs = {"name": vol.name + "_enface"}
        layer_type = "image"
        enface_layer = Layer.create(enface_image, add_kwargs, layer_type)
        vmin, vmax = np.percentile(enface_image, (1, 99))
        enface_layer.contrast_limits = (float(vmin), float(vmax))
        viewer.add_layer(enface_layer)

        # react to OK / Cancel asynchronously
    dialog.accepted.connect(lambda: on_reg_dialog_accepted(dialog))

    dialog.show()
