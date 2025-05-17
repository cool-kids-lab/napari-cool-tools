import numpy as np
from napari.utils import progress
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
    QFileDialog,
    QCheckBox,
)

from napari.qt.threading import create_worker
from qtpy import QtCore
from napari.utils.notifications import show_info
from napari.layers import Image, Layer, Labels
import napari
from napari.utils import progress
from scipy.interpolate import interp1d
import cupy as cu
from cupyx.scipy.ndimage import map_coordinates, zoom
from skimage.transform import resize
import tifffile
import scipy.ndimage
from scipy.interpolate import interpn


#this is curve correction in 2D with cylindrical method
def curve_correction(image, imaging_range, pivot_point, reference_arm_shift, scan_angle, n,
                      down_sample=False, as_16bit = False) -> Image:

    def curve_correction_2D(image,coordinates_gpu,order):

        data_gpu = cu.asarray(image)
        data_gpu = map_coordinates(data_gpu, coordinates_gpu, order=order)

        #retrieve data from gpu
        new_image = cu.asnumpy(data_gpu)

        return new_image

    show_info("Curve Correction in Progress.")
    data = image.data
    name = f"{image.name}_curve_corrected"
    data[-1,:,:] = data[-2,:,:]#delete last layer



    #################
    #curve correction 2D, This step is cylindrical coordinates convertion
    # pivot_point is in pixel

    data = data.transpose((0, 2, 1))#[840, 840,1024]
    output_size = np.min(data[:,:,0].shape)
    output_size_th = output_size*2
    r = np.linspace(0, output_size -1 , output_size) - output_size*0.5 + 0.5#from 0 -> 839, don't put exactly at 0.0
    th = np.linspace(0, output_size_th, output_size_th)
    th = np.pi*th/output_size_th #180 degree scan

    R, TH = np.meshgrid(r,th)

    x = R*np.cos(TH) # put it in the center of the original image
    x = x + output_size*0.5 - 0.5
    y = R*np.sin(TH) # we need to consider the unequally spaced pixel
    y = y + output_size*0.5 - 0.5
    
    coordinates = np.array([x, y])
    coordinates_gpu = cu.asarray(coordinates)

    output_image = np.zeros((output_size_th,output_size,data.shape[2]))#[840, 840,1024]

    for fnum in range(0,data.shape[2]):#1024 iteration
        image = data[:,:,fnum]
        image = resize(image,(output_size,output_size),order=3)#this is important
        image_gpu = cu.asarray(image)

        new_image = map_coordinates(image_gpu, coordinates_gpu,order=1)#avoid oversample
        new_image = cu.asnumpy(new_image)

        output_image[:,:,fnum] = new_image#only take the last half
        yield 1
    
    output_image = output_image.transpose((0, 2, 1))#[840, 1024, 840]
    data = output_image

    #################
    #curve correction 2D, this is the curve correction for each individual image on the cylindrical coordinates
    # pivot_point is in pixel

    ####prepare all the parameters
    imaging_range = imaging_range / n #the imaging range
    print("imaging_range")
    print(imaging_range)
    pixel_spacing = imaging_range/data.shape[1]

    pivot_point = pivot_point #this is the reference pivot point, this is constant
    reference_arm_shift = reference_arm_shift*0.5/n#this is the reference arm location relative to the position at pivot point (known to be 85000)
    print("pivot_point")
    print(pivot_point)
    print("reference_arm_shift")
    print(reference_arm_shift)
    padding = pivot_point - imaging_range + reference_arm_shift

    padding_pixel = int(padding/pixel_spacing)

    radius = data.shape[1] + padding_pixel
    resolution = radius*2

    #grid for the target image
    x = np.linspace(0, radius*2, resolution)
    y = np.linspace(0, radius*2, resolution)
    X, Y = np.meshgrid(x,y)

    #center the target
    X = X - radius
    Y = Y - radius

    # this is the location in the image in polar corrdinates
    new_r = np.sqrt(X*X+Y*Y)
    new_th = np.arctan2(Y,X)

    #removes some ugly values
    new_th[np.isnan(new_th)] = 0
    new_r[np.isnan(new_r)] = 0

    # Build ranges for function in polar coordinates
    num_theta = data.shape[2]

    r = np.linspace(0, radius-1, radius)
    angle = 0.5*scan_angle
    # angle = np.arcsin(np.sin(0.5*scan_angle/180*np.pi)/n)*180/np.pi
    th = np.linspace(-angle/180*np.pi, angle/180*np.pi, num_theta)

    #interpolate the target location in the image polar coordinates
    ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value=-1.0,kind='linear')
    ith = interp1d(th, np.arange(len(th)), bounds_error=False, fill_value=-1.0,kind='linear')
    new_ir = ir(new_r)
    new_ith = ith(new_th)

    top_image = int(padding_pixel*np.cos(angle/180*np.pi))
    right_image = radius - int(radius*np.sin(angle/180*np.pi))

    output_image = np.zeros((data.shape[0], int(resolution/2) - top_image, resolution))

    coordinates = np.array([new_ir, new_ith])
    coordinates_gpu = cu.asarray(coordinates)

    for frame, image in enumerate(data):#840 iteration
        image = np.pad(image, ((padding_pixel, 0), (0, 0)), mode='constant', constant_values=0)
        image = curve_correction_2D(image, coordinates_gpu, order=1)
        image = image.T
        output_image[frame] = image[int(resolution/2)+top_image:,:]
        yield 1   

    # top_image = int(padding_pixel*np.cos(angle/180*np.pi))
    # output_image = output_image[:,:,:]#[840, 1024, 1024*2]
    # output_image = output_image.transpose((1, 2, 0))#[1024, 840, 840]
    # data = output_image


    #     output_image = output_image.transpose((1, 2, 0))#[1024, 840, 840]

    top_image = int(padding_pixel*np.cos(angle/180*np.pi))
    print(top_image)
    right_image = radius - int(radius*np.sin(angle/180*np.pi))
    print(right_image)
    output_image = output_image[:,:,right_image:-right_image]

    output_image = output_image.transpose((1, 2, 0))#[1024, 840, 840]

    data = output_image
    
    #############################
    #put it back to cartesian
    output_size = data.shape[1]

    x = np.linspace(0, output_size-1, output_size)
    y = np.linspace(0, output_size-1, output_size) 

    #this is the target coordinates
    X, Y = np.meshgrid(x,y)

    X = X - output_size*0.5 + 0.5
    Y = Y - output_size*0.5 + 0.5


    #this is the new target location
    new_r = np.sign(Y+0.1)*np.sqrt(X*X+Y*Y)#this is always positive
    new_th = np.arctan2(Y,X)#[avoid negative angle]
    new_th = np.mod(new_th,np.pi)
    new_th[np.isnan(new_th)] = 0

    # This is location in the polar image
    num_r = data.shape[1]#[840]
    num_theta = data.shape[2]*2#840*2

    r = np.linspace(0, num_r-1, num_r) - output_size*0.5 + 0.5#from 0 -> 839, don't put exactly at 0.0
    # th = np.linspace(0, np.pi, num_theta)

    # r = np.linspace(0, num_r - 1, num_r) - output_size*0.5#from 0 -> 839
    th = np.linspace(0, num_theta, num_theta)
    th = np.pi*th/num_theta

    ir = interp1d(r, np.arange(len(r)), bounds_error=False, fill_value=-1.0,kind='linear')
    ith = interp1d(th, np.arange(len(th)), bounds_error=False, fill_value=-1.0,kind='linear')

    new_ir = ir(new_r)
    new_ith = ith(new_th)

    coordinates = np.array([new_ir, new_ith])
    coordinates_gpu = cu.asarray(coordinates)

    output_image = np.zeros((data.shape[0], output_size, output_size));

    for fnum in range(0,data.shape[0]):#1024 iteration
        image = data[fnum,:,:]
        image = resize(image,(num_r,num_theta),order=3)#this is important

        image_gpu = cu.asarray(image)

        new_image = map_coordinates(image_gpu, coordinates_gpu,order=1)#avoid oversample
        new_image = cu.asnumpy(new_image)

        output_image[fnum, :,:] = new_image
        yield 1

    output_image = output_image.transpose((2, 0, 1))


    add_kwargs = {"name":name}
    layer_type = "image"
    new_layer = Layer.create(output_image,add_kwargs,layer_type)

    show_info("Curve Correction is Finished.")

    return new_layer

def downsample_image(image,downsample_factor):
    show_info("Down Sampling in Progress.")
    
    name = f"{image.name}_downsampled"

    output_image = scipy.ndimage.zoom(image.data, downsample_factor)

    add_kwargs = {"name":name}
    layer_type = "image"
    new_layer = Layer.create(output_image,add_kwargs,layer_type)

    show_info("Down Sampling is Finished.")

    return new_layer

def saveas_numpy(data,fname):
    show_info("Save as Numpy Started.")
    with open(fname, 'wb') as f:
        np.save(f, data)

    show_info("Save as Numpy Finished.")

def saveas_bigtiff(current_layer,fname, save_as_16bit=False):
    show_info("Save as TIFF Started.")
    
    tifffile.imwrite(fname, current_layer.data, bigtiff= True)


class Curve_Correction_Widget(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Correction Panel")

        #this spin box is used to change the axial resolution
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

        #this spin box is used to change the axial resolution
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

        #this spin box is used to change the pivot point location
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

        #this spin box is used to change scan angle of the OCT
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

        self.curve_button = QPushButton("Correct Curve")
        self.layout().addWidget(self.curve_button)
        self.curve_button.clicked.connect(self.on_curve_button_clicked)

        self.as16bit_checkbx = QCheckBox("As 16bit")
        self.as16bit_checkbx.setChecked(False)
        self.layout().addWidget(self.as16bit_checkbx)

        self.auto_downsample = QCheckBox("Downsample")
        self.auto_downsample.setChecked(False)
        self.layout().addWidget(self.auto_downsample)

        #this is just a dummy function to initialize a worker thread
        dummy_function = lambda : 10
        self.worker = create_worker(dummy_function)

        #this spin box is used to change the down sample
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

        self.downsample_button = QPushButton("Downsample")
        self.layout().addWidget(self.downsample_button)
        self.downsample_button.clicked.connect(self.on_downsample_button_clicked)

        self.savenumpy_button = QPushButton("Save as numpy")
        self.layout().addWidget(self.savenumpy_button)
        self.savenumpy_button.clicked.connect(self.on_savenumpy_button_clicked)

        self.savetiff_button = QPushButton("Save as tiff")
        self.layout().addWidget(self.savetiff_button)
        self.savetiff_button.clicked.connect(self.on_savetiff_button_clicked)

        self.save16bit_checkbox = QCheckBox("Save as 16bit")
        self.save16bit_checkbox.setChecked(False)
        self.layout().addWidget(self.save16bit_checkbox)

    def on_savetiff_button_clicked(self):
        if self.worker.is_running:
            show_info("A Curve Correction process is running. Please Wait!")
            return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a layers shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        name = f"{current_layer.name}.tif"
        fileName, filters = QFileDialog.getSaveFileName(self, "Save File", name, "TIFF File (*.tif)")

        if len(fileName) == 0:
            return

        progress_bar = progress()
        progress_bar.set_description("Save as TIFF")
        progress_bar.display()

        self.worker = create_worker(saveas_bigtiff, current_layer, fileName, self.save16bit_checkbox.isChecked())
        self.worker.returned.connect(progress_bar.close)
        self.worker.start()

    def on_savenumpy_button_clicked(self):
        if self.worker.is_running:
            show_info("A Curve Correction process is running. Please Wait!")
            return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a layers shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        name = f"{current_layer.name}.npy"
        fileName, filters = QFileDialog.getSaveFileName(self, "Save File", name, "Numpy Array (*.npy)")
        
        progress_bar = progress()
        progress_bar.set_description("Save as Numpy")
        progress_bar.display()

        data = current_layer.data
        self.worker = create_worker(saveas_numpy, data, fileName)
        self.worker.returned.connect(progress_bar.close)
        self.worker.start()

    def on_downsample_button_clicked(self):
        if self.worker.is_running:
             show_info("A Curve Correction process is running. Please Wait!")
             return

        # check if an image is opened
        if len(self.viewer.layers) == 0:
            show_info("No Image layer. Please open an image.")
            return

        # #check if a layers shape is selected, otherwise throw warning.
        current_layer = self.viewer.layers.selection.active
 
        if current_layer is None:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        if isinstance(current_layer, Image) is False:
            show_info("No Image is selected. Please select an image layer.")
            return
        
        # total = current_layer.data.shape[0] + current_layer.data.shape[1]
        progress_bar = progress()
        progress_bar.set_description("Down Sampling Image(s)")
        progress_bar.display()

        self.worker = create_worker(downsample_image, current_layer, self.downsample_factor.value())
        self.worker.returned.connect(self.viewer.add_layer)
        # self.worker.yielded.connect(progress_bar.update)
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()

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
        
        if not isinstance(current_layer, Image) and not isinstance(current_layer, Labels):
            show_info("No Image is selected. Please select an image layer.")
            return
        
        ####prepare all the parameters
        imaging_range = self.imaging_range.value()
        pivot_point = self.pivot_point.value()
        reference_arm_shift = self.ref_motor.value() - self.pos_motor.value()
        reference_arm_shift = int(reference_arm_shift/1000)
        n = self.refractive_index.value()
        scan_angle = self.scan_angle.value()

        imaging_range = imaging_range / n #the imaging range
        pixel_spacing = imaging_range/current_layer.data.shape[1]
        reference_arm_shift = reference_arm_shift * 0.5/n#this is the reference arm location relative to the position at pivot point (known to be 85000)
        padding = pivot_point - imaging_range + reference_arm_shift
        padding_pixel = int(padding/pixel_spacing)

        output_size = np.min(current_layer.data[:,0,:].shape)*2
        total = output_size + current_layer.data.shape[1] +  (current_layer.data.shape[1] + padding_pixel)
        progress_bar = progress(total=int(np.ceil(total)))
        progress_bar.set_description("Correcting Curvature")


        ######
        imaging_range = self.imaging_range.value()
        pivot_point = self.pivot_point.value()
        reference_arm_shift = self.ref_motor.value() - self.pos_motor.value()
        n = self.refractive_index.value()
        scan_angle = self.scan_angle.value()

        self.worker = create_worker(curve_correction, current_layer, imaging_range, pivot_point, reference_arm_shift/1000,
                                    scan_angle,n, self.auto_downsample.isChecked(), self.as16bit_checkbx.isChecked())
        
        self.worker.returned.connect(self.viewer.add_layer)
        self.worker.yielded.connect(progress_bar.update)
        self.worker.returned.connect(progress_bar.close)

        self.worker.start()