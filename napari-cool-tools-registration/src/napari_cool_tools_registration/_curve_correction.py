from typing import Generator

import cupy as cu
import napari
import numpy as np
import scipy.ndimage
import tifffile
from cupyx.scipy.ndimage import map_coordinates
from napari.layers import Image, Labels, Layer
from napari.qt.threading import create_worker
from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy import QtCore
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import interp1d
from skimage.transform import resize
import torch
from napari_cool_tools_io import viewer, device

# this is curve correction in 2D with cylindrical method
def curve_correction(
    image,
    imaging_range,
    pivot_point,
    reference_arm_shift,
    scan_angle,
    n,
    down_sample_factor=1.0
) -> Generator[Image,Image,Image]:

    show_info("Curve Correction in Progress.")
    data = image.data
    name = f"{image.name}_curve_corrected"

    data[-1,:,:] = data[-1,:,:]*0.0 #delete last frame to avoid some issue
    data[:,0:2,:] = data[:,0:2,:]*0.0 #delete last frame to avoid some issue
    data[:,-3:-1,:] = data[:,-3:-1,:]*0.0 #delete last frame to avoid some issue

    #################
    # curve correction 2D, This step is cylindrical coordinates convertion
    # pivot_point is in pixel

    data = data.transpose((0, 2, 1))  # [840, 1024,800] to [840, 800,1024]
    output_size = data.shape[1]

    ##############################
    #main function to resize using torch
    input_data = torch.Tensor(data).unsqueeze(0).unsqueeze(0).to(device)
    data = torch.nn.functional.interpolate(
    input_data, 
    size=(output_size, output_size, data.shape[-1]),
    mode='trilinear', 
    align_corners=True
    )
    data = data.squeeze(0).squeeze(0)
    print(f"data.shape: {data.shape}")
    ######################################

    # output_size = 800
    output_size_th = output_size*2#TODO #multiplie by 2 to improve resampling quality

    r = (
        np.linspace(0, output_size-1, output_size) - output_size * 0.5 + 0.5
    )

    th = np.linspace(0, output_size_th, output_size_th)
    th = np.pi * th / output_size_th # 180 degree scan
    # th = th + (th[1] - th[0]) * 0.5

    R, TH = np.meshgrid(r, th)

    #this is correct
    x = R * np.cos(TH) #X is range from -399.5 to 399.5
    x = x / (output_size * 0.5) # Normalize to [-1, 1] for pytorch
    y = R * np.sin(TH) #Y is range from -399.5 to 399.5
    y = y / (output_size * 0.5) # Normalize to [-1, 1] for pytorch

    coordinates = np.array([x, y],dtype=np.float32)

    # Convert coordinates to torch tensors
    coords_torch = torch.from_numpy(coordinates).to(device)
    
    # Normalize coordinates to [-1, 1] range for grid_sample
    coords_normalized = torch.stack([
        coords_torch[1],  # y/theta coordinate
        coords_torch[0]  # x/r coordinate
    ], dim=-1)
        
    # Reshape for grid_sample: [1, H, W, 2]
    coords_normalized = coords_normalized.unsqueeze(0)

    output_image = torch.zeros(
        (output_size_th, output_size, data.shape[-1]),dtype=torch.float32, device=device
    ) # [1600,800,1024]

    for fnum in range(0, data.shape[-1]):  # 1024 iteration
        image_torch = data[:, :, fnum]
        
        # Apply grid_sample (torch equivalent of map_coordinates)
        new_image = torch.nn.functional.grid_sample(
            image_torch.unsqueeze(0).unsqueeze(0),
            coords_normalized,
            mode='bilinear',
            align_corners=True # already half compensated
        )
        
        new_image = new_image.squeeze(0).squeeze(0)
        output_image[:, :, fnum] = new_image
        yield 1

    # output_image = output_image.cpu().numpy()
    #output_image = output_image.transpose((2, 0, 1))  #[1600,800,1024] to [1024, 800, 800]
    data = output_image# [1600,800,1024]

    print(f"data.shape: {data.shape}")

    #################
    # curve correction 2D, this is the curve correction for each individual image on the cylindrical coordinates
    # pivot_point is in pixel

    ####prepare all the parameters
    imaging_range = imaging_range / n  # the imaging range
    print("imaging_range")
    print(imaging_range)
    pixel_spacing = imaging_range / data.shape[-1]

    reference_arm_shift = (
        reference_arm_shift * 0.5 / n
    )  # this is the reference arm location relative to the position at pivot point (known to be 85000)
    print("pivot_point")
    print(pivot_point)
    print("reference_arm_shift")
    print(reference_arm_shift)
    padding = pivot_point - imaging_range + reference_arm_shift

    padding_pixel = int(padding / pixel_spacing)

    radius = data.shape[-1] + padding_pixel
    resolution = np.round(radius).astype(int)

    # grid for the target image
    x = np.linspace(0, radius, resolution)
    y = np.linspace(0, radius * 2, resolution*2)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # center the target
    # X = X# - radius #[-radius,+radius]
    Y = Y - radius#[0,+radius]

    # this is the location in the image in polar corrdinates
    new_r = np.sqrt(X * X + Y * Y)
    new_th = np.arctan2(Y, X)
    # removes some ugly values
    new_th[np.isnan(new_th)] = 0
    new_r[np.isnan(new_r)] = 0

    # Build ranges for function in polar coordinates
    num_theta = data.shape[1]

    r = np.linspace(0, radius - 1, radius)
    angle = 0.5 * scan_angle #TODO theorically scan angle should be recalculated for each scan (see paper)
    th = np.linspace(-np.deg2rad(angle), np.deg2rad(angle), num_theta)

    # interpolate the target location in the image polar coordinates
    #TODO add 0.5
    ir = interp1d(
        r, np.arange(radius), bounds_error=False, fill_value=-radius, kind="linear"
    )
    ith = interp1d(
        th, np.arange(num_theta), bounds_error=False, fill_value=-num_theta, kind="linear"
    )
    new_ir = ir(new_r) #this is now in image index 0 -> radius (1024)
    #normalize new_ir to [-1, 1] for torch
    new_ir = (new_ir / (radius - 1)) * 2 - 1

    new_ith = ith(new_th) #this is now in image index 0 -> num_theta (800)
    #normalize new_ith to [-1, 1] for torch
    new_ith = (new_ith / (num_theta - 1)) * 2 - 1

    top_image = int(padding_pixel* down_sample_factor * np.cos(np.deg2rad(angle)))
    right_image = int(down_sample_factor*radius*(1 - np.sin(angle / 180 * np.pi)))

    # make sure even number for better resampling
    new_width = int(down_sample_factor*resolution*2)
    new_width = new_width + (int(new_width) % 2)

    new_depth = int(down_sample_factor*resolution)
    new_depth = new_depth + (int(new_depth) % 2)

    output_image = np.zeros((data.shape[0], new_width, 
                             new_depth-top_image),dtype=np.float32)
    
    # [1600,800,1024]

    coordinates = np.array([new_ith, new_ir],dtype=np.float32) # [_,800,1024]

    # Convert coordinates to torch tensors
    coords_torch = torch.from_numpy(coordinates).to(device)
    coords_normalized = torch.stack([
        coords_torch[1],  # y/theta coordinate
        coords_torch[0]  # x/r coordinate
    ], dim=-1)
        
    # Reshape for grid_sample: [1, H, W, 2]
    coords_normalized = coords_normalized.unsqueeze(0)

    for frame, image_torch in enumerate(data):  # 840 iteration

        # Pad the image_torch tensor to match the padding applied to the numpy array
        image_torch = torch.nn.functional.pad(
            image_torch, 
            (padding_pixel, 0, 0, 0), 
            mode='constant', 
            value=0
        )

        # Apply grid_sample (torch equivalent of map_coordinates)
        new_image = torch.nn.functional.grid_sample(
            image_torch.unsqueeze(0).unsqueeze(0),
            coords_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        new_image = torch.nn.functional.interpolate(
            new_image,
            size=(new_width, 
                  new_depth),
            mode='bilinear',
            align_corners=True
        )

        new_image = new_image.squeeze(0).squeeze(0).cpu().numpy()
        output_image[frame] = new_image[:,top_image:]
        yield 1

    output_image = output_image[:, right_image:-right_image, :]
    # output_image = output_image.transpose((0, 2, 1))  # [800, 800, 1024] -> [1024, 800, 800]
    data = output_image
    print(f"data.shape: {data.shape}")
    # [1600,800,1024]

    #############################
    # put it back to cartesian
    output_size = data.shape[1] # 800

    x = np.linspace(0, output_size - 1, output_size)
    y = np.linspace(0, output_size - 1, output_size)

    # this is the target coordinates
    X, Y = np.meshgrid(x, y)

    X = X - output_size * 0.5 + 0.5 #center at 0 [-400.5,400.5]
    Y = Y - output_size * 0.5 + 0.5 #center at 0 [-400.5,400.5]

    # this is the new target location
    new_r = np.sign(Y) * np.sqrt(X * X + Y * Y)  #put sign to avoid all positive
    new_th = np.arctan2(Y, X)  # [avoid negative angle]
    new_th = np.mod(new_th, np.pi)
    new_th[np.isnan(new_th)] = 0

    # This is location in the polar image
    num_r = data.shape[1]  # [840]
    num_theta = data.shape[0]*2#TODO  # 1600 multiply by 2 for better quality

    r = (
        np.linspace(0, num_r - 1, num_r) - num_r * 0.5 + 0.5 #[-400.5,400.5]
    )
    
    th = np.linspace(0, num_theta, num_theta)
    th = np.pi * th / num_theta # 180 degree scan

    ir = interp1d(
        r,
        np.arange(num_r),#0->num_r-1
        bounds_error=False, fill_value=-num_r, kind="linear"
    )
    ith = interp1d(
        th, np.arange(len(th)), bounds_error=False, fill_value=-num_theta, kind="linear"
    )

    new_ir = ir(new_r)
    #normalize new_ir to [-1, 1] for torch
    new_ir = (new_ir / (num_r - 1)) * 2 - 1

    new_ith = ith(new_th)
    #normalize new_ith to [-1, 1] for torch
    new_ith = (new_ith / (num_theta - 1)) * 2 - 1

    coordinates = np.array([new_ith, new_ir],dtype=np.float32)

    # Convert coordinates to torch tensors
    coords_torch = torch.from_numpy(coordinates).to(device)
    coords_normalized = torch.stack([
        coords_torch[1],  # y/theta coordinate
        coords_torch[0]  # x/r coordinate
    ], dim=-1)
        
    # Reshape for grid_sample: [1, H, W, 2]
    coords_normalized = coords_normalized.unsqueeze(0)

    output_image = np.zeros((output_size, output_size, data.shape[-1]), dtype=np.float32)

    for fnum in range(0, data.shape[-1]):  # 1024 iteration
        image = data[:, :, fnum]

        image_torch = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(device)
        image_torch = torch.nn.functional.interpolate(
            image_torch,
            size=(num_r, num_theta),
            mode='bilinear',
            align_corners=True
        )

        # Apply grid_sample (torch equivalent of map_coordinates)
        new_image = torch.nn.functional.grid_sample(
            image_torch,
            coords_normalized,
            mode='bilinear',
            align_corners=True
        )

        new_image = new_image.squeeze(0).squeeze(0)
        output_image[:, :, fnum] = new_image.cpu().numpy()
        yield 1

    print(f"output_image.shape: {output_image.shape}")

    output_image = output_image.transpose((1, 2, 0))  # [1600, 800, 1024] -> [800, 1024, 800]

    print(f"output_image.shape: {output_image.shape}")


    add_kwargs = {"name": name}
    layer_type = "image"
    new_layer = Layer.create(output_image, add_kwargs, layer_type)

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Curve Correction is Finished.")

    return new_layer

class Curve_Correction_Widget(QWidget):
    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Correction Panel")

        # this spin box is used to change the axial resolution
        self.refractive_index_label = QLabel("Refractive Index (n)")
        self.layout().addWidget(self.refractive_index_label)
        self.refractive_index = QDoubleSpinBox()
        self.refractive_index.setSingleStep(1.0)
        self.refractive_index.setDecimals(2)
        self.refractive_index.setMinimum(0.0)
        self.refractive_index.setMaximum(10.0)
        self.refractive_index.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.refractive_index.setValue(1.0)
        self.layout().addWidget(self.refractive_index)

        # this spin box is used to change the axial resolution
        self.imaging_range_label = QLabel("Imaging Range in Air (mm)")
        self.layout().addWidget(self.imaging_range_label)
        self.imaging_range = QDoubleSpinBox()
        self.imaging_range.setSingleStep(1.0)
        self.imaging_range.setDecimals(2)
        self.imaging_range.setMinimum(-100000.00)
        self.imaging_range.setMaximum(100000.00)
        self.imaging_range.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.imaging_range.setValue(6.0)
        self.layout().addWidget(self.imaging_range)

        # this spin box is used to change the pivot point location
        self.pivot_point_label = QLabel("Pivot Point (mm)")
        self.layout().addWidget(self.pivot_point_label)
        self.pivot_point = QDoubleSpinBox()
        self.pivot_point.setSingleStep(0.1)
        self.pivot_point.setDecimals(2)
        self.pivot_point.setMinimum(0.00)
        self.pivot_point.setMaximum(1000.00)
        self.pivot_point.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pivot_point.setValue(15.5)
        self.layout().addWidget(self.pivot_point)

        self.ref_motor_label = QLabel("Reference Motor Position")
        self.layout().addWidget(self.ref_motor_label)
        self.ref_motor = QSpinBox()
        self.ref_motor.setMinimum(0)
        self.ref_motor.setMaximum(1000000)
        self.ref_motor.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ref_motor.setValue(85000)
        self.layout().addWidget(self.ref_motor)

        self.pos_motor_label = QLabel("Current Motor Position")
        self.layout().addWidget(self.pos_motor_label)
        self.pos_motor = QSpinBox()
        self.pos_motor.setMinimum(0)
        self.pos_motor.setMaximum(1000000)
        self.pos_motor.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.pos_motor.setValue(85000)
        self.layout().addWidget(self.pos_motor)

        # this spin box is used to change scan angle of the OCT
        self.degree_label = QLabel("Scan Angle in Air (<sup>o</sup>)")
        self.layout().addWidget(self.degree_label)
        self.scan_angle = QDoubleSpinBox()
        self.scan_angle.setSingleStep(0.1)
        self.scan_angle.setDecimals(2)
        self.scan_angle.setMinimum(0.00)
        self.scan_angle.setMaximum(360.00)
        self.scan_angle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scan_angle.setValue(140.0)
        self.layout().addWidget(self.scan_angle)

        # this is just a dummy function to initialize a worker thread
        dummy_function = lambda: 10
        self.worker = create_worker(dummy_function)

        # this spin box is used to change the down sample
        self.downsample_label = QLabel("Downsample Factor")
        self.layout().addWidget(self.downsample_label)
        self.downsample_factor = QDoubleSpinBox()
        self.downsample_factor.setSingleStep(0.1)
        self.downsample_factor.setDecimals(2)
        self.downsample_factor.setMinimum(0.00)
        self.downsample_factor.setMaximum(1.00)
        self.downsample_factor.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.downsample_factor.setValue(0.5)
        self.layout().addWidget(self.downsample_factor)

        self.curve_button = QPushButton("Correct Curve")
        self.layout().addWidget(self.curve_button)
        self.curve_button.clicked.connect(self.on_curve_button_clicked)


    def on_curve_button_clicked(self):
        if self.worker.is_running:
            show_info("A Curve Correction process is running. Please Wait!")
            return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a line shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active

        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return

        if not isinstance(current_layer, Image) and not isinstance(
            current_layer, Labels
        ):
            show_info("No Image is selected. Please select an image layer.")
            return

        ####prepare all the parameters
        imaging_range = self.imaging_range.value()
        pivot_point = self.pivot_point.value()
        reference_arm_shift = self.ref_motor.value() - self.pos_motor.value()
        reference_arm_shift = int(reference_arm_shift / 1000)
        n = self.refractive_index.value()
        scan_angle = self.scan_angle.value()

        imaging_range = imaging_range / n  # the imaging range
        pixel_spacing = imaging_range / current_layer.data.shape[1]
        reference_arm_shift = (
            reference_arm_shift * 0.5 / n
        )  # this is the reference arm location relative to the position at pivot point (known to be 85000)
        padding = pivot_point - imaging_range + reference_arm_shift
        padding_pixel = int(padding / pixel_spacing)

        output_size = np.min(current_layer.data[:, 0, :].shape) * 2
        total = (
            output_size
            + current_layer.data.shape[1]
            + (current_layer.data.shape[1] + padding_pixel)
        )
        progress_bar = progress(total=int(np.ceil(total)))
        progress_bar.set_description("Correcting Curvature")

        ######
        imaging_range = self.imaging_range.value()
        pivot_point = self.pivot_point.value()
        reference_arm_shift = self.ref_motor.value() - self.pos_motor.value()
        n = self.refractive_index.value()
        scan_angle = self.scan_angle.value()

        self.worker = create_worker(
            curve_correction,
            current_layer,
            imaging_range,
            pivot_point,
            reference_arm_shift / 1000,
            scan_angle,
            n,
            self.downsample_factor.value()
        )

        self.worker.returned.connect(self.viewer.add_layer)
        self.worker.yielded.connect(progress_bar.update)
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()
