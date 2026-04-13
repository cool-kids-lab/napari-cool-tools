from qtpy.QtWidgets import QWidget, QDialog
from qtpy import QtWidgets
from qtpy.QtCore import Qt
import pyvista as pv
import pyvistaqt
import numpy as np
from napari_cool_tools_io import viewer
from napari.layers import Image
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Signal
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from napari_cool_tools_vol_render import cast_dtype
from typing import cast

def _normalize_to_uint(volume: np.ndarray, dtype: cast_dtype) -> np.ndarray:
    vmin = np.min(volume)
    vmax = np.max(volume)

    scaled = (volume - vmin) / (vmax - vmin)
    dtype_max = np.iinfo(dtype.value).max
    return np.clip(scaled * dtype_max, 0, dtype_max).astype(dtype.value)

class ValueSlider(QWidget):
    value_changed = Signal(int)

    def __init__(
        self,
        text: str = "Slider",
        min_value=0,
        max_value=255,
        parent=None,
    ):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout(self)

        self.min_label = QtWidgets.QLabel(self)
        self.min_label.setText(f"{text}: {min_value}")
        layout.addWidget(self.min_label)

        center_widget = QtWidgets.QWidget(self)
        center_layout = QtWidgets.QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(2)

        self.value_label = QtWidgets.QLabel(self)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.value_label.setText(str(min_value))
        center_layout.addWidget(self.value_label)

        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(min_value)
        center_layout.addWidget(self.slider)

        layout.addWidget(center_widget, 1)

        self.max_label = QtWidgets.QLabel(self)
        self.max_label.setText(f"{max_value}")
        layout.addWidget(self.max_label)

        self.slider.valueChanged.connect(self._on_slider_value_changed)

    def _on_slider_value_changed(self, value: int):
        self.value_label.setText(str(value))
        self.value_changed.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int):
        self.slider.setValue(value)


class ControlWidget(QWidget):
    value_changed = Signal(int)

    def __init__(
        self,
        min_value=0,
        max_value=255,
        parent=None,
    ):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        #add the min max slider
        self.min_slider = ValueSlider("Min", min_value, max_value, self)
        self.min_slider.set_value(min_value)
        self.max_slider = ValueSlider("Max", min_value, max_value, self)
        self.max_slider.set_value(max_value)

        layout.addWidget(self.min_slider)
        layout.addWidget(self.max_slider)

        spacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        layout.addItem(spacer)

        self.min_slider.value_changed.connect(self.synchronize_sliders)
        self.max_slider.value_changed.connect(self.synchronize_sliders)

    def synchronize_sliders(self, _value: int):
        min_value = self.min_slider.value()
        max_value = self.max_slider.value()
        sender = self.sender()

        if sender is self.min_slider and min_value >= max_value:
            self.max_slider.set_value(min_value)
        
        if sender is self.max_slider and max_value <= min_value:
            self.min_slider.set_value(max_value)

        self.value_changed.emit(_value)


class PyVistaDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QtWidgets.QHBoxLayout(self)

        self.plotter = pyvistaqt.QtInteractor(self)
        self.plotter.set_background("black")
        self.layout.addWidget(self.plotter)

    def set_volume(self, volume):
        self.control_menu = ControlWidget(min_value=np.min(volume), max_value=np.max(volume), parent=self)
        self.layout.addWidget(self.control_menu)
        self.control_menu.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Preferred,
        )

        grid = pv.ImageData()
        grid.dimensions = np.array(volume.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.origin = (0, 0, 0)
        grid.cell_data["values"] = volume.ravel(order="F")

        self.plotter.clear()
        plotter = cast(pv.Plotter, self.plotter)  # typing workaround for QtInteractor stubs
        self.actor = plotter.add_volume(
            grid,
            mapper="gpu",          # prefers GPU when available
            blending="maximum",    # faster than "maximum"
            cmap="gray",
            opacity="linear",
            shade=False,
            show_scalar_bar=False,
        )
        self.plotter.reset_camera()


        self.control_menu.value_changed.connect(self.update_opacity)

    def update_opacity(self, value):
        min_value = self.control_menu.min_slider.value()
        max_value = self.control_menu.max_slider.value()

        if min_value >= max_value:
            show_warning("Min value must be less than Max value.")
            return

        opacity_function = vtkPiecewiseFunction()
        opacity_function.AddPoint(min_value, 0.0)  # invisible
        opacity_function.AddPoint(max_value, 1.0)  # fully opaque

        self.actor.GetProperty().SetScalarOpacity(opacity_function)
        self.plotter.render()



def pyvista_render_plugin(input_volume: Image, cast_dtype: cast_dtype):

    # #if input_volume is not 3D, show error and return
    if input_volume.ndim != 3:
        show_warning("Not a 3D input volume provided.")
        return

    dialog = QDialog(parent=viewer.window._qt_window)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dialog.setWindowModality(Qt.WindowModality.NonModal)
    dialog.setModal(False)

    dialog.setWindowFlag(Qt.WindowType.Window, True)
    dialog.setWindowFlag(Qt.WindowType.WindowSystemMenuHint, True)
    dialog.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
    dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, True)

    dialog.setWindowTitle("3D Rendering PyVista")

    layout = QtWidgets.QVBoxLayout(dialog)

    pyvista_dock = PyVistaDock(dialog)
    layout.addWidget(pyvista_dock)

    # Example volume data
    volume_data = input_volume.data
    # Normalize the volume data to the specified dtype
    volume_data = _normalize_to_uint(np.asarray(volume_data), cast_dtype)

    #generate random volume data for testing
    # volume_data = np.random.rand(50, 50, 50)

    pyvista_dock.set_volume(volume_data)

    dialog.show()

    


