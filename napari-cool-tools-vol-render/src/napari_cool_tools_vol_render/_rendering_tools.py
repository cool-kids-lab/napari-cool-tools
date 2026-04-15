from qtpy.QtWidgets import QWidget, QDialog, QFileDialog
from qtpy import QtWidgets
from qtpy.QtCore import Qt
import pyvista as pv
import pyvistaqt
import numpy as np
from napari_cool_tools_io import viewer
from napari.layers import Image
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Signal
from napari_cool_tools_vol_render import cast_dtype
from typing import cast
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

        layout.addWidget(center_widget,1)

        self.max_label = QtWidgets.QLabel(self)
        self.max_label.setText(f"{max_value}")
        layout.addWidget(self.max_label)

        #add a spinner to the right of the slider
        self.spinner = QtWidgets.QSpinBox(self)
        self.spinner.setRange(min_value, max_value)
        self.spinner.setValue(min_value)
        layout.addWidget(self.spinner)

        self.slider.valueChanged.connect(lambda value: self.value_label.setText(str(value)))
        self.slider.valueChanged.connect(lambda value: self.spinner.setValue(value))
        self.spinner.valueChanged.connect(lambda value: self.slider.setValue(value))
        self.spinner.valueChanged.connect(self.value_changed)
        self.slider.valueChanged.connect(self.value_changed)

class ControlWidget(QWidget):
    min_max_value_changed = Signal(int)

    def __init__(
        self,
        min_value=0,
        max_value=255,
        parent=None,
    ):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        #add shading technique selection (between Maximum Intensity Projection and Iso Surface)
        self.technqiue_label = QtWidgets.QLabel(self)
        self.technqiue_label.setText("Shading Technique:")
        layout.addWidget(self.technqiue_label)

        self.technique_combo = QtWidgets.QComboBox(self)
        self.technique_combo.addItems(["Maximum Intensity Projection", "Surface"])
        layout.addWidget(self.technique_combo)

        #add the min max slider
        self.min_slider = ValueSlider("Min", min_value, max_value, self)
        self.min_slider.slider.setValue(min_value)
        self.max_slider = ValueSlider("Max", min_value, max_value, self)
        self.max_slider.slider.setValue(max_value)
        layout.addWidget(self.min_slider)
        layout.addWidget(self.max_slider)

        #add auto range button
        self.auto_range_button = QtWidgets.QPushButton(self)
        self.auto_range_button.setText("Auto Range")
        layout.addWidget(self.auto_range_button)

        #add background color selection
        bg_layout = QtWidgets.QHBoxLayout()
        bg_label = QtWidgets.QLabel(self)
        bg_label.setText("Background:")
        bg_layout.addWidget(bg_label)

        self.bg_combo = QtWidgets.QComboBox(self)
        self.bg_combo.addItems(["black", "white"])
        self.bg_combo.setCurrentText("black")
        bg_layout.addWidget(self.bg_combo)
        layout.addLayout(bg_layout)

        #add color map selection
        cmap_layout = QtWidgets.QHBoxLayout()
        cmap_label = QtWidgets.QLabel(self)
        cmap_label.setText("Colormap:")
        cmap_layout.addWidget(cmap_label)

        self.cmap_combo = QtWidgets.QComboBox(self)
        #list all the colormaps available in pyvista and add them to the combo box
        colormaps = plt.colormaps()
        self.cmap_combo.addItems(colormaps)
        self.cmap_combo.setCurrentText("gray")
        cmap_layout.addWidget(self.cmap_combo)
        layout.addLayout(cmap_layout)

        #add shading mode checkbox
        self.shading_checkbox = QtWidgets.QCheckBox(self)
        self.shading_checkbox.setText("Shading")
        self.shading_checkbox.setChecked(True)
        layout.addWidget(self.shading_checkbox)

        #add interpolation mode selection
        self.interpolation_combo = QtWidgets.QComboBox(self)
        self.interpolation_combo.addItems(["Linear", "Nearest"])
        self.interpolation_combo.setCurrentText("Linear")
        layout.addWidget(self.interpolation_combo)

        #add ambient slider using qwidgets.QSlider
        ambient_layout = QtWidgets.QHBoxLayout()
        ambient_label = QtWidgets.QLabel(self)
        ambient_label.setText("Ambient:")
        ambient_layout.addWidget(ambient_label)

        self.ambient_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(10)
        self.ambient_slider.setSingleStep(1)
        self.ambient_slider.setPageStep(10)
        ambient_layout.addWidget(self.ambient_slider)

        #add spinner to the right of the ambient slider
        self.ambient_spinner = QtWidgets.QDoubleSpinBox(self)
        self.ambient_spinner.setRange(0.0, 1.0)
        self.ambient_spinner.setValue(0.1)
        self.ambient_spinner.setSingleStep(0.01)
        ambient_layout.addWidget(self.ambient_spinner)
        
        self.ambient_spinner.valueChanged.connect(lambda value: self.ambient_slider.setValue(int(value*100)))
        self.ambient_slider.valueChanged.connect(lambda value: self.ambient_spinner.setValue(value/100))

        layout.addLayout(ambient_layout)

        #add diffuse slider using qwidgets.QSlider
        diffuse_layout = QtWidgets.QHBoxLayout()
        diffuse_label = QtWidgets.QLabel(self)
        diffuse_label.setText("Diffuse:")
        diffuse_layout.addWidget(diffuse_label)

        self.diffuse_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.diffuse_slider.setRange(0, 100)
        self.diffuse_slider.setValue(70)
        self.diffuse_slider.setSingleStep(1)
        self.diffuse_slider.setPageStep(10)
        diffuse_layout.addWidget(self.diffuse_slider)

        #add spinner to the right of the diffuse slider
        self.diffuse_spinner = QtWidgets.QDoubleSpinBox(self)
        self.diffuse_spinner.setRange(0.0, 1.0)
        self.diffuse_spinner.setValue(0.7)
        self.diffuse_spinner.setSingleStep(0.01)
        diffuse_layout.addWidget(self.diffuse_spinner)

        self.diffuse_spinner.valueChanged.connect(lambda value: self.diffuse_slider.setValue(int(value*100)))
        self.diffuse_slider.valueChanged.connect(lambda value: self.diffuse_spinner.setValue(value/100))

        layout.addLayout(diffuse_layout)

        #add specular slider using qwidgets.QSlider
        specular_layout = QtWidgets.QHBoxLayout()
        specular_label = QtWidgets.QLabel(self)
        specular_label.setText("Specular:")
        specular_layout.addWidget(specular_label)

        self.specular_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.specular_slider.setRange(0, 100)
        self.specular_slider.setValue(20)
        self.specular_slider.setSingleStep(1)
        self.specular_slider.setPageStep(10)
        specular_layout.addWidget(self.specular_slider)

        #add spinner to the right of the specular slider
        self.specular_spinner = QtWidgets.QDoubleSpinBox(self)
        self.specular_spinner.setRange(0.0, 1.0)
        self.specular_spinner.setValue(0.2)
        self.specular_spinner.setSingleStep(0.01)
        specular_layout.addWidget(self.specular_spinner)

        self.specular_spinner.valueChanged.connect(lambda value: self.specular_slider.setValue(int(value*100)))
        self.specular_slider.valueChanged.connect(lambda value: self.specular_spinner.setValue(value/100))

        layout.addLayout(specular_layout)

        #add specular power slider using qwidgets.QSlider
        specular_power_layout = QtWidgets.QHBoxLayout()
        specular_power_label = QtWidgets.QLabel(self)
        specular_power_label.setText("Specular Power:")
        specular_power_layout.addWidget(specular_power_label)

        self.specular_power_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.specular_power_slider.setRange(0, 256)
        self.specular_power_slider.setValue(20)
        specular_power_layout.addWidget(self.specular_power_slider)

        #add a spinner to the right of the specular power slider
        self.specular_power_spinner = QtWidgets.QDoubleSpinBox(self)
        self.specular_power_spinner.setRange(0.0, 128.0)
        self.specular_power_spinner.setValue(10.0)
        self.specular_power_spinner.setSingleStep(0.5)

        self.specular_power_spinner.valueChanged.connect(lambda value: self.specular_power_slider.setValue(int(value * 2)))
        self.specular_power_slider.valueChanged.connect(lambda value: self.specular_power_spinner.setValue(value * 0.5))

        specular_power_layout.addWidget(self.specular_power_spinner)

        layout.addLayout(specular_power_layout)

        #add presets for the sliders (ambient, diffuse, specular, specular power), just like in 3D slicer
        presets_layout = QtWidgets.QHBoxLayout()
        presets_label = QtWidgets.QLabel(self)
        presets_label.setText("Presets:")
        presets_layout.addWidget(presets_label)

        self.preset_combo = QtWidgets.QComboBox(self)
        self.preset_combo.addItems(["High", "Default", "Normal", "Medium", "Low"])
        self.preset_combo.setCurrentText("Default")
        presets_layout.addWidget(self.preset_combo)

        layout.addLayout(presets_layout)


        #add animation button        
        self.animation_button = QtWidgets.QPushButton(self)
        self.animation_button.setText("Animate")
        layout.addWidget(self.animation_button)


        #add record layout with a setting for the output file name and a button to start recording
        record_setting_layout_1 = QtWidgets.QHBoxLayout()
        self.record_start_angle_spinner = QtWidgets.QSpinBox(self)
        self.record_start_angle_spinner.setRange(-180, 0)
        self.record_start_angle_spinner.setValue(-30)
        record_setting_layout_1.addWidget(QtWidgets.QLabel("Start: ", self))
        record_setting_layout_1.addWidget(self.record_start_angle_spinner)

        self.record_end_angle_spinner = QtWidgets.QSpinBox(self)
        self.record_end_angle_spinner.setRange(0, 180)
        self.record_end_angle_spinner.setValue(30)
        record_setting_layout_1.addWidget(QtWidgets.QLabel("End: ", self))
        record_setting_layout_1.addWidget(self.record_end_angle_spinner)


        record_setting_layout_2 = QtWidgets.QHBoxLayout()
        self.record_rate_spinner = QtWidgets.QSpinBox(self)
        self.record_rate_spinner.setRange(1, 120)
        self.record_rate_spinner.setValue(30)
        record_setting_layout_2.addWidget(QtWidgets.QLabel("FPS: ", self))
        record_setting_layout_2.addWidget(self.record_rate_spinner)

        
        #add Record button
        self.record_button = QtWidgets.QPushButton(self)
        self.record_button.setText("Record")
        record_setting_layout_2.addWidget(self.record_button)

        record_setting_layout_2.setAlignment(Qt.AlignRight)

        layout.addLayout(record_setting_layout_1)
        layout.addLayout(record_setting_layout_2)

        spacer = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding
        )
        layout.addItem(spacer)

        self.min_slider.value_changed.connect(self.synchronize_sliders)
        self.max_slider.value_changed.connect(self.synchronize_sliders)

    def synchronize_sliders(self, _value: int):
        min_value = self.min_slider.slider.value()
        max_value = self.max_slider.slider.value()
        sender = self.sender()

        if sender is self.min_slider and min_value >= max_value:
            if min_value == self.min_slider.slider.maximum():
                self.min_slider.slider.setValue(min_value-1)
                self.max_slider.slider.setValue(min_value)
            else:
                self.max_slider.slider.setValue(min_value+1)

        if sender is self.max_slider and max_value <= min_value:
            if max_value == self.max_slider.slider.minimum():
                self.max_slider.slider.setValue(max_value+1)
                self.min_slider.slider.setValue(max_value)
            else:
                self.min_slider.slider.setValue(max_value-1)

        self.min_max_value_changed.emit(_value)


class PyVistaDock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QtWidgets.QHBoxLayout(self)

        self.plotter = pyvistaqt.QtInteractor(self)
        self.plotter.set_background("black")
        self.layout.addWidget(self.plotter)

        self.name = "PyVista Volume Renderer"

    def set_volume(self, volume):
        min_value = np.min(volume)
        max_value = np.max(volume)
        self.control_menu = ControlWidget(min_value=min_value, max_value=max_value, parent=self)
        self.layout.addWidget(self.control_menu)
        self.control_menu.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed,
            QtWidgets.QSizePolicy.Preferred,
        )

        self.grid = pv.ImageData()
        self.grid.dimensions = np.array(volume.shape) + 1
        self.grid.spacing = (1, 1, 1)
        self.grid.origin = (0, 0, 0)
        self.grid.cell_data["values"] = volume.ravel(order="F")
        self.plotter.clear()

        plotter = cast(pv.Plotter, self.plotter)  # typing workaround for QtInteractor stubs
        self.actor = plotter.add_volume(
            self.grid,
            mapper="gpu",          # prefers GPU when available
            blending="maximum",    # faster than "maximum"
            cmap="gray",
            opacity="linear",       # use linear opacity mapping
            shade=False,
            show_scalar_bar=False,
            ambient=0.1,
            diffuse=0.7,
            specular=0.2,
            specular_power=10.0,
        )
        self.actor.prop.interpolation_type = "linear"

        # self.plotter.enable_lightkit()

        self.light = pv.Light()
        self.light.set_headlight()
        self.light.intensity = 1.0
        self.light.switch_off()
        plotter.add_light(self.light)

        self.control_menu.min_max_value_changed.connect(self.update_range)
        self.control_menu.auto_range_button.clicked.connect(self.auto_range)
        self.control_menu.bg_combo.currentTextChanged.connect(self.update_background)
        self.control_menu.shading_checkbox.stateChanged.connect(self.update_shading)
        self.control_menu.technique_combo.currentIndexChanged.connect(self.update_technique)
        self.control_menu.interpolation_combo.currentTextChanged.connect(self.update_interpolation)
        self.control_menu.ambient_slider.valueChanged.connect(self.update_ambient)
        self.control_menu.diffuse_slider.valueChanged.connect(self.update_diffuse)
        self.control_menu.specular_slider.valueChanged.connect(self.update_specular)
        self.control_menu.specular_power_slider.valueChanged.connect(self.update_specular_power)
        self.control_menu.preset_combo.currentTextChanged.connect(self.apply_preset)
        self.control_menu.cmap_combo.currentTextChanged.connect(self.update_cmap)

        self.update_range(0)

        self.animation_running = False
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animation_rotate)

        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.animation_rotate_record)

        self.control_menu.animation_button.clicked.connect(self.toggle_animation)
        self.control_menu.record_button.clicked.connect(self.record_animation)

        self.plotter.reset_camera()

    def update_cmap(self, text):

        # vmin, vmax = self.actor.mapper.scalar_range

        # # remove and re-add the volume actor with the new colormap to ensure the mapper/LUT is updated
        # plotter = cast(pv.Plotter, self.plotter)
        # # try:
        # blend_mode = self.actor.mapper.blend_mode
        # # except Exception:
        # #     blend_mode = "maximum"
        # ambient = self.actor.prop.GetAmbient()
        # diffuse = self.actor.prop.GetDiffuse()
        # specular = self.actor.prop.GetSpecular()
        # specular_power = self.actor.prop.GetSpecularPower()
        # plotter.remove_actor(self.actor)
        # # add volume without immediate render to avoid expensive re-rendering
        # self.actor = plotter.add_volume(
        #     self.grid,
        #     mapper="gpu",
        #     blending=blend_mode,
        #     cmap=text,
        #     opacity="linear",
        #     shade=self.control_menu.shading_checkbox.isChecked(),
        #     show_scalar_bar=False,
        #     ambient=ambient,
        #     diffuse=diffuse,
        #     specular=specular,
        #     specular_power=specular_power,
        #     render=False,
        # )
        # self.actor.prop.interpolation_type = "linear"

        # self.plotter.update_scalar_bar_range(clim=(vmin, vmax))

        self.actor.mapper.lookup_table.cmap = text

        self.plotter.render()

    def get_save_video_path(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Video",
            "Unnamed.mp4",
            "MP4 Video (*.mp4)"
        )

        if not filename:
            return None

        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"

        return filename

    def record_animation(self):

        #disable any animations while recording
        if self.animation_running:
            self.toggle_animation()
        
        #disable the control_menu options while recording
        self.control_menu.setDisabled(True)

        self.animation_init_angle = self.plotter.camera.azimuth
        self.plotter.camera.azimuth = self.control_menu.record_start_angle_spinner.value()
        self.animation_direction = 1
        self.animation_step = 1
        fps = self.control_menu.record_rate_spinner.value()

        #open the file dialog to get the save path for the video
        video_name_path = self.get_save_video_path()
        print(f"Saving video to: {video_name_path}")

        #open the video writer with the specified path and frame rate
        self.plotter.open_movie(video_name_path, framerate=fps)

        #start timer to rotate the camera and record frames at the specified frame rate
        self.record_timer.start(int(1000 / fps))

        #the recording will automaticatlly stop when the camera azimuth reaches the end angle, which is handled in the animation_rotate_record function

    def animation_rotate_record(self):

        self.plotter.camera.azimuth = self.plotter.camera.azimuth + self.animation_direction * self.animation_step

        # Define the left and right maximum angles for the animation
        left_max_angle = self.control_menu.record_start_angle_spinner.value()
        right_max_angle = self.control_menu.record_end_angle_spinner.value()

        if self.plotter.camera.azimuth  >= right_max_angle:
            self.animation_direction = -1
        elif self.plotter.camera.azimuth  <= left_max_angle:
            self.animation_direction = 1
            self.record_timer.stop()

            self.plotter.render()
            self.plotter.write_frame()  # write the last frame at the end angle
            self.plotter.mwriter.close()

            #put back the camera to the initial angle and re-enable the control menu
            self.plotter.camera.azimuth = self.animation_init_angle
            self.control_menu.setDisabled(False)
            self.plotter.render()
            return

        self.plotter.render()
        self.plotter.write_frame()


    def toggle_animation(self):
        self.animation_running = not self.animation_running

        if self.animation_running:

            #get initial camera azimuth angle to use as the starting point for the animation
            self.animation_init_angle = self.plotter.camera.azimuth
            self.plotter.camera.azimuth = self.control_menu.record_start_angle_spinner.value()
            self.animation_direction = 1
            self.animation_step = 1
            self.control_menu.record_end_angle_spinner.setDisabled(True)
            self.control_menu.record_start_angle_spinner.setDisabled(True)
            fps = self.control_menu.record_rate_spinner.value()
            self.control_menu.record_rate_spinner.setDisabled(True)
            self.control_menu.record_button.setDisabled(True)
            self.animation_timer.start(int(1000 / fps))
        else:
            self.animation_timer.stop()

            #return to init angle
            self.plotter.camera.azimuth = self.animation_init_angle
            self.plotter.render()

            self.control_menu.record_end_angle_spinner.setDisabled(False)
            self.control_menu.record_start_angle_spinner.setDisabled(False)
            self.control_menu.record_rate_spinner.setDisabled(False)
            self.control_menu.record_button.setDisabled(False)

    def animation_rotate(self):
        if not self.animation_running:
            return

        self.plotter.camera.azimuth = self.plotter.camera.azimuth + self.animation_direction * self.animation_step

        # Define the left and right maximum angles for the animation
        left_max_angle = self.control_menu.record_start_angle_spinner.value()
        right_max_angle = self.control_menu.record_end_angle_spinner.value()

        if self.plotter.camera.azimuth  >= right_max_angle:
            self.animation_direction = -1
        elif self.plotter.camera.azimuth  <= left_max_angle:
            self.animation_direction = 1

        self.plotter.render()

    def apply_preset(self, text):
        if text == "High":
            self.control_menu.ambient_slider.setValue(100)
            self.control_menu.diffuse_slider.setValue(0)
            self.control_menu.specular_slider.setValue(0)
            self.control_menu.specular_power_slider.setValue(2)
        elif text == "Default":
            self.control_menu.ambient_slider.setValue(10)
            self.control_menu.diffuse_slider.setValue(70)
            self.control_menu.specular_slider.setValue(20)
            self.control_menu.specular_power_slider.setValue(20)
        elif text == "Normal":
            self.control_menu.ambient_slider.setValue(20)
            self.control_menu.diffuse_slider.setValue(100)
            self.control_menu.specular_slider.setValue(0)
            self.control_menu.specular_power_slider.setValue(2)
        elif text == "Medium":
            self.control_menu.ambient_slider.setValue(10)
            self.control_menu.diffuse_slider.setValue(90)
            self.control_menu.specular_slider.setValue(20)
            self.control_menu.specular_power_slider.setValue(20)
        elif text == "Low":
            self.control_menu.ambient_slider.setValue(10)
            self.control_menu.diffuse_slider.setValue(60)
            self.control_menu.specular_slider.setValue(50)
            self.control_menu.specular_power_slider.setValue(80)


    def auto_range(self):
        vmin, vmax = np.percentile(self.grid.active_scalars, (1, 99))

        #update the sliders to reflect the new range
        self.control_menu.min_slider.slider.setValue(int(vmin))
        self.control_menu.max_slider.slider.setValue(int(vmax))

    def update_diffuse(self, value):
        self.actor.prop.SetDiffuse(value/100.0)
        self.plotter.render()
    
    def update_specular(self, value):
        self.actor.prop.SetSpecular(value/100.0)
        self.plotter.render()

    def update_specular_power(self, value):
        self.actor.prop.SetSpecularPower(value*0.5)
        self.plotter.render()

    def update_ambient(self, value):
        self.actor.prop.SetAmbient(value/100.0)
        self.plotter.render()

    def update_interpolation(self, text):
        if text == "Linear":
            self.actor.prop.interpolation_type  = "linear"
        else:
            self.actor.prop.interpolation_type  = "nearest"
        self.plotter.render()

    def update_technique(self, index):
        if index == 0:  # Maximum Intensity Projection
            self.actor.mapper.blend_mode = "maximum"
            # self.actor.prop.ShadeOff()
        else:  # Surface
            self.actor.mapper.blend_mode = "composite"
            self.update_shading(self.control_menu.shading_checkbox.checkState())
            # self.actor.prop.ShadeOn()
        self.plotter.render()

    def update_shading(self, state):
        if state == 2:  # Checked
            self.actor.prop.ShadeOn()
            self.light.switch_on()
        else:
            self.actor.prop.ShadeOff()
            self.light.switch_off()

        self.plotter.render()

    def update_background(self, value):
        self.plotter.set_background(value)
        self.plotter.render()

    def update_range(self, value):
        min_value = self.control_menu.min_slider.slider.value()
        max_value = self.control_menu.max_slider.slider.value()

        with pv.vtk_verbosity('error'):
            self.plotter.update_scalar_bar_range(clim=(min_value, max_value))
            self.plotter.render()



def pyvista_render_plugin(input_volume: Image, cast_dtype: cast_dtype):
# def pyvista_render_plugin():

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

    # generate random volume data for testing
    # volume_data = np.random.rand(50, 50, 50)

    pyvista_dock.set_volume(volume_data)

    dialog.show()

    


