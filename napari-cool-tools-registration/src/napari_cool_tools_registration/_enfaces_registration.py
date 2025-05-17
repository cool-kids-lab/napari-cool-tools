import gc
import numpy as np
import segmentation_models_pytorch as smp
import onnxruntime as onrt
from pathlib import Path
from typing import List, Tuple, Generator
from magicgui import magic_factory
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer
from napari.types import ImageData
from napari_cool_tools_io import torch,viewer,device,memory_stats, _prof_reader
from napari_cool_tools_segmentation._segmentation_funcs import bscan_onnx_seg_func, enface_onnx_seg_func
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import generate_enface
import glob

import napari
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from magicgui.widgets import create_widget
from skimage.measure import EllipseModel
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale, resize
from skimage.util import img_as_float
from scipy.ndimage import rotate as ndimage_rotate
from scipy.ndimage import binary_dilation
from skimage.exposure import match_histograms



class EllipseProcessor:
    def __init__(self, ON, Img, Ves, viewer):
        self.ON = ON  # Original mask
        self.Img = Img  # Original image
        self.Ves = Ves  # Original image
        self.viewer = viewer
        self.rotated_ON = None  # Placeholder for the rotated mask
        self.rotated_Img = None  # Placeholder for the rotated image
        self.rotated_Ves = None  # Placeholder for the rotated image
        self.center_x_ON = None
        self.center_y_ON = None
        self.center_x_ON_rot = None
        self.center_y_ON_rot = None
        self.radius_major_ON = None
        self.radius_minor_ON = None
        self.orientation_ON = None
        self.major_axis_direction = None
        self.minor_axis_direction = None
        self.major_axis_start = None
        self.major_axis_end = None
        self.minor_axis_start = None
        self.minor_axis_end = None
        self.attributes = None  # To store all key attributes
        #self.process_image()
    
    def process_image(self,mask):
        coordinates = np.where(mask > 0)
        coordinates = np.array(coordinates)
        
        # Create the EllipseModel object
        ellipse_model = EllipseModel()
        ellipse_model.estimate(coordinates.T)
        
        # Extract ellipse parameters
        self.center_x_ON, self.center_y_ON, self.radius_major_ON, self.radius_minor_ON, self.orientation_ON = ellipse_model.params
        self.center_x_ON_rot = self.center_x_ON
        self.center_y_ON_rot = self.center_y_ON

        # Step 1: Define the bounding box around the ellipse
        box_points = np.array([
            [-self.radius_major_ON, -self.radius_minor_ON],
            [self.radius_major_ON, -self.radius_minor_ON],
            [self.radius_major_ON, self.radius_minor_ON],
            [-self.radius_major_ON, self.radius_minor_ON]
        ])
        
        # Step 2: Rotate the bounding box by theta
        rotation_matrix = np.array([
            [np.cos(self.orientation_ON), -np.sin(self.orientation_ON)],
            [np.sin(self.orientation_ON), np.cos(self.orientation_ON)]
        ])
        
        # Rotate each point of the bounding box
        rotated_box_points = np.dot(box_points, rotation_matrix.T)
        bounding_box = rotated_box_points + np.array([self.center_x_ON, self.center_y_ON])
        
        # # Add the bounding box to the viewer
        # self.viewer.add_shapes(
        #     bounding_box,
        #     shape_type='ellipse',
        #     edge_width=1,
        #     edge_color='coral',
        #     face_color='purple',
        #     name='ON'
        # )
        
        # Step 3: Define the major and minor axes
        self.major_axis_direction = np.array([np.cos(self.orientation_ON), np.sin(self.orientation_ON)])
        self.minor_axis_direction = np.array([-np.sin(self.orientation_ON), np.cos(self.orientation_ON)])
        
        self.major_axis_start = [self.center_x_ON - self.radius_major_ON * self.major_axis_direction[0], 
                                 self.center_y_ON - self.radius_major_ON * self.major_axis_direction[1]]
        self.major_axis_end = [self.center_x_ON + self.radius_major_ON * self.major_axis_direction[0], 
                               self.center_y_ON + self.radius_major_ON * self.major_axis_direction[1]]
        
        self.minor_axis_start = [self.center_x_ON - self.radius_minor_ON * self.minor_axis_direction[0], 
                                 self.center_y_ON - self.radius_minor_ON * self.minor_axis_direction[1]]
        self.minor_axis_end = [self.center_x_ON + self.radius_minor_ON * self.minor_axis_direction[0], 
                               self.center_y_ON + self.radius_minor_ON * self.minor_axis_direction[1]]
        
        # # Add the major axis to the viewer
        # self.viewer.add_shapes(
        #     np.array([self.major_axis_start, self.major_axis_end]),
        #     shape_type='line',
        #     edge_width=2,
        #     edge_color='blue',
        #     name='Major Axis'
        # )
        
        # # Add the minor axis to the viewer
        # self.viewer.add_shapes(
        #     np.array([self.minor_axis_start, self.minor_axis_end]),
        #     shape_type='line',
        #     edge_width=2,
        #     edge_color='green',
        #     name='Minor Axis'
        # )
        

       # Store attributes
        self.attributes = np.array([[self.center_x_ON, self.center_y_ON], self.major_axis_direction,
                                    self.minor_axis_direction, self.major_axis_start, self.major_axis_end,
                                    self.minor_axis_start, self.minor_axis_end, [self.orientation_ON, 1]])

    
    def rotate_image_and_elements(self):
        # Rotate the image to align the major axis to the 12 o'clock position (0 degrees)
        rotation_angle = np.degrees(self.orientation_ON * -1)  # Negative to rotate clockwise
        self.rotated_Img = ndimage_rotate(self.Img, rotation_angle, reshape=False)
        self.rotated_ON = ndimage_rotate(self.ON, rotation_angle, reshape=False)
        self.rotated_Ves = ndimage_rotate(self.Ves, rotation_angle, reshape=False)

        # Calculate new center after rotation (center of the rotated mask)

        

        # Process the rotated mask
        self.process_image(self.rotated_ON)
        self.process_image(self.rotated_Ves)

        print(self.center_x_ON_rot, self.center_x_ON)


        # Update the viewer with rotated image and mask
        #self.viewer.add_image(self.rotated_Img, name='Rotated Image')
        #self.viewer.add_labels(self.rotated_ON, name='Rotated Mask')

    def get_center(self):
        #Returns the center (x, y) of the processed ON shape.
        return np.array([self.center_x_ON, self.center_y_ON])

    def get_rotated_center(self):
            #Returns the new center (x, y) of the rotated ON shape.
        return np.array([self.center_x_ON_rot, self.center_y_ON_rot])

    def get_attributes(self):
        """Returns the computed attributes."""
        return self.attributes

def translationStation(processor1, processor2, test, viewer):
    center1 = processor1.get_rotated_center()
    center2 = processor2.get_rotated_center()
    
    # Calculate the necessary padding for Img2
    delta_x = center1[0] - center2[0]  # Difference in x-coordinates
    delta_y = center1[1] - center2[1]  # Difference in y-coordinates
    
    shift_x = delta_x  # Move based on the x difference
    shift_y = delta_y  # Move based on the y difference

    print(f"Shift in x: {shift_x}, Shift in y: {shift_y}")

    # Handle translation in the y direction
    if shift_y < 0:
        # Pad bottom, crop top (move up)
        padded_image = np.pad(test, ((0, int(shift_y*-1)), (0, 0)), mode='constant', constant_values=0)
        translated_image = padded_image[int(shift_y*-1):, :]  # Crop from the top
    else:
        # Pad top, crop bottom (move down)
        padded_image = np.pad(test, ((int(shift_y), 0), (0, 0)), mode='constant', constant_values=0)
        translated_image = padded_image[:test.shape[0], :]

    # Handle translation in the x direction
    if shift_x < 0:
        # Pad right, crop left (move left)
        padded_image = np.pad(translated_image, ((0, 0), (0, int(shift_x*-1))), mode='constant', constant_values=0)
        translated_image = padded_image[:, int(shift_x*-1):]  # Crop from the left
    else:
        # Pad left, crop right (move right)
        padded_image = np.pad(translated_image, ((0, 0), (int(shift_x), 0)), mode='constant', constant_values=0)
        translated_image = padded_image[:, :test.shape[1]]

    
    # Add the translated test image to the viewer
    #viewer.add_image(translated_image, name="Translated Test Image")

def center_point_in_image(img, x, y):
    # Step 1: Get image dimensions and find the center of the image
    rows, cols = img.shape  # Assuming img is a 2D array (grayscale)

    center_x = rows // 2
    center_y = cols // 2
    x = round(x)
    y = round(y)
    # Step 2: Calculate the shifts needed to center (x, y)
    shift_x = center_x - x
    shift_y = center_y - y

    print(x,y,shift_x,shift_y)

    # Step 3: Pad the image to shift the point (x, y) to the center
    # We will pad the image using np.pad and adjust the padding based on the shift
    pad_top = max(0, shift_x)
    pad_bottom = max(0, -shift_x)
    pad_left = max(0, shift_y)
    pad_right = max(0, -shift_y)

    # Apply the padding, this will shift the image so that (x, y) becomes the center
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Step 4: Crop the image if necessary to maintain the original size (optional)
    # If you want to keep the image size fixed after padding, crop back to original size
    start_row = abs(min(0, shift_x))
    start_col = abs(min(0, shift_y))
    centered_img = padded_img[start_row:start_row+rows, start_col:start_col+cols]

    return centered_img

def find_box_corners(center_x, center_y, distance=200):
    # Top-left corner
    top_left = (center_x - distance, center_y - distance)
    
    # Top-right corner
    top_right = (center_x - distance, center_y + distance)
    
    # Bottom-left corner
    bottom_left = (center_x + distance, center_y - distance)
    
    # Bottom-right corner
    bottom_right = (center_x + distance, center_y + distance)
    row_start, col_start = top_left
    row_end, col_end = bottom_right
    row_end=round(row_end)
    row_start=round(row_start)
    col_end=round(col_end)
    col_start= round(col_start)

    return row_start, col_start, row_end, col_end

def pad_images_to_align_features(ellipse_processor_1:EllipseProcessor, ellipse_processor_2: EllipseProcessor):
    
    
    # Calculate the necessary padding for Img2
    dy = int(ellipse_processor_2.center_x_ON - ellipse_processor_1.center_x_ON)  # Difference in x-coordinates
    dx = int(ellipse_processor_2.center_y_ON - ellipse_processor_1.center_y_ON)  # Difference in y-coordinates
    
    print("y,x",dx,dy) # for some reason this is switched^ above too.

    left = max(0, -dx)
    right = max(0, dx)
    canvas_width = max(ellipse_processor_1.Img.shape[1] + left, ellipse_processor_2.Img.shape[1] + right)
    
    # For height, calculate the max height considering dy offset
    top = max(0, -dy)
    bottom = max(0, dy)
    canvas_height = max(ellipse_processor_1.Img.shape[0] + top, ellipse_processor_2.Img.shape[0] + bottom)
    
    # Create a blank canvas with zeros (black)
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
   
    pos_image1_x = max(0, dx)
    pos_image1_y = max(0, dy)
    
    # Image2 will be pasted with top-left at (max(0, -dx), max(0, -dy))
    pos_image2_x = max(0, -dx)
    pos_image2_y = max(0, -dy)
    
    # Paste image1 onto the canvas
    canvas[pos_image1_y:pos_image1_y + ellipse_processor_1.Img.shape[0], pos_image1_x:pos_image1_x + ellipse_processor_1.Img.shape[1]] = ellipse_processor_1.Img
    finalIm1 = canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    canvas[pos_image1_y:pos_image1_y + ellipse_processor_1.Img.shape[0], pos_image1_x:pos_image1_x + ellipse_processor_1.Img.shape[1]] = ellipse_processor_1.ON
    finalON1 = canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    # Paste image2 onto the canvas
    canvas[pos_image2_y:pos_image2_y + ellipse_processor_2.Img.shape[0], pos_image2_x:pos_image2_x + ellipse_processor_2.Img.shape[1]] = ellipse_processor_2.Img
    finalIm2 = canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint)

    canvas[pos_image2_y:pos_image2_y + ellipse_processor_2.Img.shape[0], pos_image2_x:pos_image2_x + ellipse_processor_2.Img.shape[1]] = ellipse_processor_2.ON
    finalON2 = canvas
    #canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    return finalIm1, finalIm2, finalON1, finalON2

def pad_images_to_align_featuresinEdit(ellipse_processor_1:EllipseProcessor, ellipse_processor_2: EllipseProcessor):
    
    
    # Calculate the necessary padding for Img2
    dy = int(ellipse_processor_2.center_x_ON - ellipse_processor_1.center_x_ON)  # Difference in x-coordinates
    dx = int(ellipse_processor_2.center_y_ON - ellipse_processor_1.center_y_ON)  # Difference in y-coordinates
    
    print("y,x",dx,dy) # for some reason this is switched^ above too.

    print(ellipse_processor_1.Img.shape[0])
    canvas_height = abs(dy) + max(ellipse_processor_1.Img.shape[0], ellipse_processor_2.Img.shape[0])
    canvas_width = abs(dx) + max(ellipse_processor_1.Img.shape[1],ellipse_processor_2.Img.shape[1])


    # Create a blank canvas with zeros (black)
    canvas = np.zeros((canvas_height,canvas_width), dtype=np.float32)
    print(canvas.shape)
    print(canvas_height)

    pos_image1_x = max(0, dx)
    pos_image1_y = max(0, dy)
    
    # Image2 will be pasted with top-left at (max(0, -dx), max(0, -dy))
    pos_image2_x = max(0, -dx)
    pos_image2_y = max(0, -dy)
    
    # Paste image1 onto the canvas
    canvas[pos_image1_y:pos_image1_y + ellipse_processor_1.Img.shape[0], pos_image1_x:pos_image1_x + ellipse_processor_1.Img.shape[1]] = ellipse_processor_1.Img
    finalIm1 = canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    canvas[pos_image1_y:pos_image1_y + ellipse_processor_1.Img.shape[0], pos_image1_x:pos_image1_x + ellipse_processor_1.Img.shape[1]] = ellipse_processor_1.ON
    finalON1 = canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    # Paste image2 onto the canvas
    canvas[pos_image2_y:pos_image2_y + ellipse_processor_2.Img.shape[0], pos_image2_x:pos_image2_x + ellipse_processor_2.Img.shape[1]] = ellipse_processor_2.Img
    finalIm2 = canvas
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint)

    canvas[pos_image2_y:pos_image2_y + ellipse_processor_2.Img.shape[0], pos_image2_x:pos_image2_x + ellipse_processor_2.Img.shape[1]] = ellipse_processor_2.ON
    finalON2 = canvas
    #canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    return finalIm1, finalIm2, finalON1, finalON2

def center_point_in_image_Post(img, x, y):
    # Step 1: Get image dimensions and find the center of the image
    rows, cols = img.shape  # Assuming img is a 2D array (grayscale)

    center_x = rows // 2
    center_y = cols // 2
    x = round(x)
    y = round(y)
    # Step 2: Calculate the shifts needed to center (x, y)
    shift_x = center_x - x
    shift_y = center_y - y

    print(x,y,shift_x,shift_y)

    # Step 3: Pad the image to shift the point (x, y) to the center
    # We will pad the image using np.pad and adjust the padding based on the shift
    pad_top = max(0, shift_x)
    pad_bottom = max(0, -shift_x)
    pad_left = max(0, shift_y)
    pad_right = max(0, -shift_y)

    # Apply the padding, this will shift the image so that (x, y) becomes the center
    padded_img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Step 4: Crop the image if necessary to maintain the original size (optional)
    # If you want to keep the image size fixed after padding, crop back to original size
    start_row = abs(min(0, shift_x))
    start_col = abs(min(0, shift_y))
    centered_img = padded_img[start_row:start_row+rows, start_col:start_col+cols]

    return centered_img

# Build an array of ON masks from the Napari Viewer
def extract_masks_from_layers(layers):
    # Create an empty list to store mask arrays
    ONlist = []
    
    # Track the maximum dimensions
    max_shape = (0, 0)  # Assuming 2D masks; update as needed for 3D

    # Loop through each layer in the viewer
    for layer in layers:
        # Check if the layer type is 'labels' or similar that indicates a mask
        if isinstance(layer, napari.layers.Labels):  # 'Labels' layers often store masks
            mask = layer.data  # Access the mask data
            ONlist.append(mask)
            # Update max_shape if this mask is larger in any dimension
            max_shape = tuple(max(s, m) for s, m in zip(mask.shape, max_shape))

    # Pad masks to match max_shape using reflection padding
    padded_masks = []
    for mask in ONlist:
        # Calculate padding for each dimension
        padding = [(0, max_dim - mask_dim) for mask_dim, max_dim in zip(mask.shape, max_shape)]
        padded_mask = np.pad(mask, padding, mode='reflect')  # Pads with reflection
        padded_masks.append(padded_mask)
    
    # Stack all masks into a single 3D array if possible
    mask_array = np.stack(padded_masks) if padded_masks else np.array([])
    
    return mask_array

# Resize OCTA scans from 800x800 to 840x840
def print_layer_sizes(viewer):
    # Loop through each layer in the viewer
    for i, layer in enumerate(viewer.layers):
        # Print the name and size of the data array in each layer
        print(f"Layer {i} - {layer.name}: array size         {layer.data.shape}")

# Find the center of the image
def find_center_of_largest_array(arrays):
 
    # Initialize max_shape
    max_shape = (0, 0)  # Adjust to (0, 0, 0) if working with 3D arrays

    # Find the maximum shape among the arrays
    for array in arrays:
        max_shape = tuple(max(dim_size, max_dim) for dim_size, max_dim in zip(array.shape, max_shape))

    # Calculate the center coordinates of the largest array
    center_coordinates = tuple(dim_size // 2 for dim_size in max_shape)
    
    return center_coordinates

# This is to extract Ellipse Data 
class EllipseProcessor:
    def __init__(self, ON, Img, Ves, viewer):
        self.ON = ON  # Original mask
        self.Img = Img  # Original image
        self.Ves = Ves  # Original image
        self.viewer = viewer
        self.rotated_ON = None  # Placeholder for the rotated mask
        self.rotated_Img = None  # Placeholder for the rotated image
        self.rotated_Ves = None  # Placeholder for the rotated image
        self.center_x_ON = None
        self.center_y_ON = None
        self.center_x_ON_rot = None
        self.center_y_ON_rot = None
        self.radius_major_ON = None
        self.radius_minor_ON = None
        self.orientation_ON = None
        self.major_axis_direction = None
        self.minor_axis_direction = None
        self.major_axis_start = None
        self.major_axis_end = None
        self.minor_axis_start = None
        self.minor_axis_end = None
        self.attributes = None  # To store all key attributes
        #self.process_image()
    
    def process_image(self,mask):
        coordinates = np.where(mask > 0)
        coordinates = np.array(coordinates)
        
        # Create the EllipseModel object
        ellipse_model = EllipseModel()
        ellipse_model.estimate(coordinates.T)
        
        # Extract ellipse parameters
        self.center_x_ON, self.center_y_ON, self.radius_major_ON, self.radius_minor_ON, self.orientation_ON = ellipse_model.params
        self.center_x_ON_rot = self.center_x_ON
        self.center_y_ON_rot = self.center_y_ON

        # Step 1: Define the bounding box around the ellipse
        box_points = np.array([
            [-self.radius_major_ON, -self.radius_minor_ON],
            [self.radius_major_ON, -self.radius_minor_ON],
            [self.radius_major_ON, self.radius_minor_ON],
            [-self.radius_major_ON, self.radius_minor_ON]
        ])
        
        # Step 2: Rotate the bounding box by theta
        rotation_matrix = np.array([
            [np.cos(self.orientation_ON), -np.sin(self.orientation_ON)],
            [np.sin(self.orientation_ON), np.cos(self.orientation_ON)]
        ])
        
        # Rotate each point of the bounding box
        rotated_box_points = np.dot(box_points, rotation_matrix.T)
        bounding_box = rotated_box_points + np.array([self.center_x_ON, self.center_y_ON])
        
        # # Add the bounding box to the viewer
        # self.viewer.add_shapes(
        #     bounding_box,
        #     shape_type='ellipse',
        #     edge_width=1,
        #     edge_color='coral',
        #     face_color='purple',
        #     name='ON'
        # )
        
        # Step 3: Define the major and minor axes
        self.major_axis_direction = np.array([np.cos(self.orientation_ON), np.sin(self.orientation_ON)])
        self.minor_axis_direction = np.array([-np.sin(self.orientation_ON), np.cos(self.orientation_ON)])
        
        self.major_axis_start = [self.center_x_ON - self.radius_major_ON * self.major_axis_direction[0], 
                                 self.center_y_ON - self.radius_major_ON * self.major_axis_direction[1]]
        self.major_axis_end = [self.center_x_ON + self.radius_major_ON * self.major_axis_direction[0], 
                               self.center_y_ON + self.radius_major_ON * self.major_axis_direction[1]]
        
        self.minor_axis_start = [self.center_x_ON - self.radius_minor_ON * self.minor_axis_direction[0], 
                                 self.center_y_ON - self.radius_minor_ON * self.minor_axis_direction[1]]
        self.minor_axis_end = [self.center_x_ON + self.radius_minor_ON * self.minor_axis_direction[0], 
                               self.center_y_ON + self.radius_minor_ON * self.minor_axis_direction[1]]
        
        # # Add the major axis to the viewer
        # self.viewer.add_shapes(
        #     np.array([self.major_axis_start, self.major_axis_end]),
        #     shape_type='line',
        #     edge_width=2,
        #     edge_color='blue',
        #     name='Major Axis'
        # )
        
        # # Add the minor axis to the viewer
        # self.viewer.add_shapes(
        #     np.array([self.minor_axis_start, self.minor_axis_end]),
        #     shape_type='line',
        #     edge_width=2,
        #     edge_color='green',
        #     name='Minor Axis'
        # )
        

       # Store attributes
        self.attributes = np.array([[self.center_x_ON, self.center_y_ON], self.major_axis_direction,
                                    self.minor_axis_direction, self.major_axis_start, self.major_axis_end,
                                    self.minor_axis_start, self.minor_axis_end, [self.orientation_ON, 1]])

    def rotate_image_and_elements(self):
        # Rotate the image to align the major axis to the 12 o'clock position (0 degrees)
        rotation_angle = np.degrees(self.orientation_ON * -1)  # Negative to rotate clockwise
        self.rotated_Img = ndimage_rotate(self.Img, rotation_angle, reshape=False)
        self.rotated_ON = ndimage_rotate(self.ON, rotation_angle, reshape=False)
        self.rotated_Ves = ndimage_rotate(self.Ves, rotation_angle, reshape=False)

        # Calculate new center after rotation (center of the rotated mask)

        

        # Process the rotated mask
        self.process_image(self.rotated_ON)
        self.process_image(self.rotated_Ves)

        print(self.center_x_ON_rot, self.center_x_ON)


        # Update the viewer with rotated image and mask
        #self.viewer.add_image(self.rotated_Img, name='Rotated Image')
        #self.viewer.add_labels(self.rotated_ON, name='Rotated Mask')

    def get_center(self):
        #Returns the center (x, y) of the processed ON shape.
        return np.array([self.center_x_ON, self.center_y_ON])

    def get_rotated_center(self):
            #Returns the new center (x, y) of the rotated ON shape.
        return np.array([self.center_x_ON_rot, self.center_y_ON_rot])

    def get_attributes(self):
        """Returns the computed attributes."""
        return self.attributes

# This crops a circle from an image/label 
def crop_circle(testIm, center=None, radius=None):

    # Get image dimensions
    h, w = testIm.shape
    
    # Define the center if not provided
    if center is None:
        center = (w // 2, h // 2)
    
    # If radius is not provided, default to the largest possible inscribed circle
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    
    # Create a grid of coordinates with the same shape as the image
    Y, X = np.ogrid[:h, :w]
    
    # Calculate the distance from the center for each pixel
    distance_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    
    # Create a mask that is True for pixels within the circle and False outside
    mask = distance_from_center <= radius
    
    # Create a new image with the same dimensions, initialized to 0
    cropped_image = np.zeros_like(testIm)
    
    # Copy the pixels within the circular region
    cropped_image[mask] = testIm[mask]
    
    return cropped_image

# This crops all labels and images in the Napari Viewer
def crop_dilate_all_layers(viewer, center=None, radius=None):

    for layer in viewer.layers:
        # Check if the layer is an image or labels layer
        if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
            # Apply cropping
            cropped_data = crop_circle(layer.data, center=center, radius=radius)
            
            # Update the layer with the cropped data
            layer.data = cropped_data
            print(f"Cropped {layer.name} with shape {cropped_data.shape}")

    for layer in viewer.layers:
        # Check if the layer is a Labels layer, which commonly stores masks
        if isinstance(layer, napari.layers.Labels):
            original_mask = layer.data > 0  # Ensure binary mask
            dilated_mask = binary_dilation(original_mask)
            
            # Perform XOR to get only the edge
            edge_mask = dilated_mask ^ original_mask
            
            # Update the layer with the edge mask
            layer.data = edge_mask.astype(layer.data.dtype)
            #print(f"Updated {layer.name} with one-pixel-thick edge mask")

def outline_masks_in_viewer(viewer):
    """
    Cycles through the mask layers in the viewer, applies a dilation to the "ON" mask, 
    and updates each label layer with a one-pixel-thick edge version.

    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance containing the mask layers.
    """
    for layer in viewer.layers:
        # Check if the layer is a Labels layer and corresponds to an "ON" mask
        if isinstance(layer, napari.layers.Labels) and "Seg" in layer.name and not layer.name.endswith("[1]"):
            original_mask = layer.data > 0  # Ensure binary mask
            dilated_mask = binary_dilation(original_mask)
            
            # Perform XOR to get only the edge
            edge_mask = dilated_mask ^ original_mask
            
            # Update the layer with the edge mask
            layer.data = edge_mask.astype(layer.data.dtype)

def pair_images_and_masks(viewer):
    """
    Pairs images with both "ON" and "Vessels" masks in the Napari viewer based on their name patterns.

    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance containing the image and mask layers.
    
    Returns
    -------
    dict
        Dictionary mapping each image layer name to its corresponding "ON" and "Vessels" mask layer names.
    """
    image_layers = {}
    mask_layers = {"ON": {}, "Vessels": {}}

    # Separate image and mask layers based on naming convention
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image):
            image_layers[layer.name] = layer
        elif isinstance(layer, napari.layers.Labels):
            # Check for ON and Vessels suffixes to categorize masks
            if layer.name.endswith("_Seg [1]"):  # Vessels mask
                mask_layers["Vessels"][layer.name.replace("_Seg [1]", "")] = layer
            elif layer.name.endswith("_Seg"):  # ON mask
                mask_layers["ON"][layer.name.replace("_Seg", "")] = layer

    # Match each image with its "ON" and "Vessels" masks by exact prefix match
    paired_layers = {}
    for image_name in image_layers.keys():
        # Match the corresponding ON and Vessels masks by prefix
        on_mask = mask_layers["ON"].get(image_name)
        vessels_mask = mask_layers["Vessels"].get(image_name)

        # Store the matched pairs if both masks are found
        if on_mask and vessels_mask:
            paired_layers[image_name] = {"ON": on_mask.name, "Vessels": vessels_mask.name}
    
    return paired_layers

def extract_ellipse_centers(viewer):
    """
    Processes only "ON" masks for ellipse center extraction and stores the results.
    
    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance containing the layers.
    
    Returns
    -------
    dict
        Dictionary mapping each image layer name to the center coordinates of its "ON" mask.
    """
    centers = {}
    image_mask_pairs = pair_images_and_masks(viewer)

    for image_name, masks in image_mask_pairs.items():
        # Retrieve the "ON" mask and its corresponding image
        mask_layer_ON = viewer.layers[masks["ON"]]
        mask_layer_Ves = viewer.layers[masks["ON"]]
        image_layer = viewer.layers[image_name]
        
        # Initialize EllipseProcessor with the "ON" mask data and image data
        processor = EllipseProcessor(mask_layer_ON.data, image_layer.data, mask_layer_Ves.data,  viewer)
        
        # Process the "ON" mask to extract ellipse center
        processor.process_image(mask_layer_ON.data)
        
        # Store the center coordinates
        center = processor.get_center()
        if center is not None:
            centers[image_name] = center

    return centers

def pad_all_layers(viewer, target_shape=(800, 840)):
    """
    Pads all layers in the viewer that are smaller than the target shape with zeros.

    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance containing the layers.
    target_shape : tuple of int
        The target shape (height, width) to pad small layers to.
    """
    target_height, target_width = target_shape

    for layer in viewer.layers:
        current_shape = layer.data.shape
        
        # Check if the layer is smaller than the target shape
        if current_shape[0] < target_height or current_shape[1] < target_width:
            # Calculate the padding required for each dimension
            pad_height = (target_height - current_shape[0]) // 2
            pad_width = (target_width - current_shape[1]) // 2
            
            # Calculate padding for top, bottom, left, right
            padding = (
                (pad_height, target_height - current_shape[0] - pad_height),
                (pad_width, target_width - current_shape[1] - pad_width)
            )
            
            # Apply padding with zeros
            if layer.ndim == 3:
                # For 3D data (e.g., single-channel 3D image with shape (1, height, width))
                padding = ((0, 0), *padding)  # No padding for the first dimension

            padded_data = np.pad(layer.data, padding, mode='constant', constant_values=0)
            
            # Update the layer with the padded data
            layer.data = padded_data
            #print(f"Padded {layer.name} from {current_shape} to {target_shape}")

def find_closest_center(ellipse_centers, center=(400, 420)):
    """
    Finds the ellipse center closest to the specified point.
    
    Parameters
    ----------
    ellipse_centers : dict
        Dictionary where keys are image layer names and values are the coordinates of the ellipse centers.
    center : tuple of int
        The reference point (x, y) to which the distance is calculated. Default is (400, 420).
    
    Returns
    -------
    tuple
        A tuple containing the name of the layer with the closest center, the closest center coordinates,
        and the distance to the specified point.
    """
    min_distance = float('inf')
    closest_layer = None
    closest_center = None

    for layer_name, ellipse_center in ellipse_centers.items():
        # Calculate Euclidean distance to the specified center point
        distance = np.linalg.norm(np.array(ellipse_center) - np.array(center))
        
        # Check if this center is the closest so far
        if distance < min_distance:
            min_distance = distance
            closest_layer = layer_name
            closest_center = ellipse_center

    return closest_layer, closest_center, min_distance

def find_closest_centers_sorted(ellipse_centers, center=(400, 420)):
    """
    Finds and sorts ellipse centers by their distance to a specified reference point.
    
    Parameters
    ----------
    ellipse_centers : dict
        Dictionary where keys are image layer names and values are the coordinates of the ellipse centers.
    center : tuple of int
        The reference point (x, y) to which the distance is calculated. Default is (400, 420).
    
    Returns
    -------
    list of tuples
        A list of tuples containing the layer name, center coordinates, and distance to the specified point,
        sorted in ascending order by distance.
    """
    distances = []

    for layer_name, ellipse_center in ellipse_centers.items():
        # Calculate Euclidean distance to the specified center point
        distance = np.linalg.norm(np.array(ellipse_center) - np.array(center))
        distances.append((layer_name, ellipse_center, distance))

    # Sort by distance (ascending order)
    distances.sort(key=lambda x: x[2])

    return distances

def find_furthest_centers(ellipse_centers, center=(400, 420)):
    """
    Finds the ellipse centers that are furthest away from the specified point in +x, -x, +y, and -y directions.
    
    Parameters
    ----------
    ellipse_centers : dict
        Dictionary where keys are layer names and values are coordinates of the ellipse centers.
    center : tuple of int
        The reference point (x, y) from which distances are calculated. Default is (400, 420).
    
    Returns
    -------
    dict
        A dictionary with keys "+x", "-x", "+y", and "-y" mapping to the furthest center in each direction.
    """
    # Initialize variables to keep track of the furthest centers and their x or y values
    furthest_centers = {
        "+x": (None, float('-inf')),  # Highest x > 400
        "-x": (None, float('inf')),   # Lowest x < 400
        "+y": (None, float('-inf')),  # Highest y > 420
        "-y": (None, float('inf'))    # Lowest y < 420
    }

    ref_x, ref_y = center

    for layer_name, (x, y) in ellipse_centers.items():
        # Check +x direction (furthest right of 400)
        if x > ref_x and x > furthest_centers["+x"][1]:
            furthest_centers["+x"] = (layer_name, x)

        # Check -x direction (furthest left of 400)
        if x < ref_x and x < furthest_centers["-x"][1]:
            furthest_centers["-x"] = (layer_name, x)

        # Check +y direction (furthest above 420)
        if y > ref_y and y > furthest_centers["+y"][1]:
            furthest_centers["+y"] = (layer_name, y)

        # Check -y direction (furthest below 420)
        if y < ref_y and y < furthest_centers["-y"][1]:
            furthest_centers["-y"] = (layer_name, y)

    # Simplify output by just returning layer names and coordinates
    result = {
        "+x": (furthest_centers["+x"][0], furthest_centers["+x"][1]),
        "-x": (furthest_centers["-x"][0], furthest_centers["-x"][1]),
        "+y": (furthest_centers["+y"][0], furthest_centers["+y"][1]),
        "-y": (furthest_centers["-y"][0], furthest_centers["-y"][1])
    }

    return result

def calculate_canvas_size(ellipse_centers, center=(400, 420), padding=166):
    """
    Calculates the canvas size needed to accommodate all centers such that
    the furthest centers can be brought to the original center point (400, 420),
    with an additional padding of 166 pixels on each side.

    Parameters
    ----------
    ellipse_centers : dict
        Dictionary where keys are layer names and values are coordinates of the ellipse centers.
    center : tuple of int
        The reference point (x, y) to which the furthest centers will be aligned.
    padding : int, optional
        Padding to add to each side of the canvas. Default is 166 pixels.

    Returns
    -------
    canvas_shape : tuple
        Height and width required for the canvas.
    shift_offsets : dict
        Offsets for each direction to bring the furthest points to the reference center.
    """
    # Find furthest centers in each direction
    furthest_centers = find_furthest_centers(ellipse_centers, center=center)
    
    # Calculate maximum distances in each direction
    ref_x, ref_y = center
    max_x_right = abs(furthest_centers["+x"][1] - ref_x) if furthest_centers["+x"][0] else 0
    max_x_left = abs(furthest_centers["-x"][1] - ref_x) if furthest_centers["-x"][0] else 0
    max_y_up = abs(furthest_centers["+y"][1] - ref_y) if furthest_centers["+y"][0] else 0
    max_y_down = abs(furthest_centers["-y"][1] - ref_y) if furthest_centers["-y"][0] else 0

    # Canvas width and height calculations with padding
    canvas_width = 2 * max(max_x_right, max_x_left) + ref_x * 2 + 2 * padding
    canvas_height = 2 * max(max_y_up, max_y_down) + ref_y * 2 + 2 * padding

    # Create a blank canvas
    canvas_shape = (int(canvas_height), int(canvas_width))
    shift_offsets = {
        "+x": max_x_right + padding,
        "-x": max_x_left + padding,
        "+y": max_y_up + padding,
        "-y": max_y_down + padding
    }
    translations = {
        layer_name: (ref_x - x, ref_y - y)
        for layer_name, (x, y) in ellipse_centers.items()
    }
    
    return canvas_shape, shift_offsets, translations

def process_images_for_alignment(Img1, Img2, ON1, ON2, Ves1, Ves2, viewer, radius=200):

    # Initialize EllipseProcessor for Vessel Masks
    ellipse_processor_ves1 = EllipseProcessor(ON1, Ves1, viewer)
    ellipse_processor_ves1.process_image(ON1)
    ellipse_processor_ves2 = EllipseProcessor(ON2, Ves2, viewer)
    ellipse_processor_ves2.process_image(ON2)

    # Center Images based on ON mask center
    centered_img1 = center_point_in_image(ellipse_processor_ves1.Img, ellipse_processor_ves1.center_x_ON, ellipse_processor_ves1.center_y_ON)
    centered_img2 = center_point_in_image(ellipse_processor_ves2.Img, ellipse_processor_ves2.center_x_ON, ellipse_processor_ves2.center_y_ON)

    # Apply polar transformation
    image_polar1 = warp_polar(centered_img1, radius=radius)[:, 50:-1]
    image_polar2 = warp_polar(centered_img2, radius=radius)[:, 50:-1]

    # Calculate phase cross-correlation to find rotation angle
    shifts, error, phasediff = phase_cross_correlation(image_polar1, image_polar2, normalization=None)
    rotation_angle = -float(shifts[0])


    # Deal with the ON masks for Images
    ellipse_processor_on1 = EllipseProcessor(ON1, Img1, viewer)
    ellipse_processor_on1.process_image(ON1)
    ellipse_processor_on2 = EllipseProcessor(ON2, Img2, viewer)
    ellipse_processor_on2.process_image(ON2)

    # Rotate Img2 and ON2 to align with Img1 and ON1
    rotated_img2 = rotate(ellipse_processor_on2.Img, rotation_angle)
    rotated_on2 = rotate(ellipse_processor_on2.ON, rotation_angle)
    rotated_on2 = np.where(rotated_on2 > 1.21e-10, 1, 0)  # Binarize rotated ON mask

    # Reprocess rotated ON mask with EllipseProcessor to align features
    ellipse_processor_on2_rotated = EllipseProcessor(rotated_on2, rotated_img2, viewer)
    ellipse_processor_on2_rotated.process_image(rotated_on2)

    # Pad and align final images and ON masks
    finalIm1, finalIm2, finalON1, finalON2 = pad_images_to_align_features(ellipse_processor_on1, ellipse_processor_on2_rotated)

    return finalIm1, finalIm2, finalON1, finalON2

def process_images_for_alignmentPost(Img1, Img2, ON1, ON2, Ves1, Ves2, viewer, radius=200):

    # Initialize EllipseProcessor for Vessel Masks
    ellipse_processor_ves1 = EllipseProcessor(ON1, Ves1, viewer)
    ellipse_processor_ves1.process_image(ON1)
    ellipse_processor_ves2 = EllipseProcessor(ON2, Ves2, viewer)
    ellipse_processor_ves2.process_image(ON2)
    ellipse_processor_ves1.Img = np.squeeze(ellipse_processor_ves1.Img)  # This will result in shape (806, 924)
    ellipse_processor_ves2.Img = np.squeeze(ellipse_processor_ves2.Img)  # This will result in shape (806, 924)

    # Center Images based on ON mask center
    centered_img1 = center_point_in_image(ellipse_processor_ves1.Img, ellipse_processor_ves1.center_x_ON, ellipse_processor_ves1.center_y_ON)
    centered_img2 = center_point_in_image(ellipse_processor_ves2.Img, ellipse_processor_ves2.center_x_ON, ellipse_processor_ves2.center_y_ON)

    # Apply polar transformation
    image_polar1 = warp_polar(centered_img1, radius=radius)[:, 50:-1]
    image_polar2 = warp_polar(centered_img2, radius=radius)[:, 50:-1]

    # Calculate phase cross-correlation to find rotation angle
    shifts, error, phasediff = phase_cross_correlation(image_polar1, image_polar2, normalization=None)
    rotation_angle = -float(shifts[0])

    # Rotate centered_img1 to align with centered_img2
    rotated_img1 = rotate(centered_img1, rotation_angle)

    # Deal with the ON masks for Images
    ellipse_processor_on1 = EllipseProcessor(ON1, Img1, viewer)
    ellipse_processor_on1.process_image(ON1)
    ellipse_processor_on2 = EllipseProcessor(ON2, Img2, viewer)
    ellipse_processor_on2.process_image(ON2)

    # Rotate Img2 and ON2 to align with Img1 and ON1
    rotated_img2 = rotate(ellipse_processor_on2.Img, rotation_angle)
    rotated_on2 = rotate(ellipse_processor_on2.ON, rotation_angle)
    rotated_on2 = np.where(rotated_on2 > 1.21e-10, 1, 0)  # Binarize rotated ON mask
   
    # Reprocess rotated ON mask with EllipseProcessor to align features
    ellipse_processor_2_rotated = EllipseProcessor(rotated_on2, rotated_img2, viewer)
    ellipse_processor_2_rotated.process_image(rotated_on2)
    viewer.add_image(ellipse_processor_on1.Img)
    viewer.add_image(ellipse_processor_on1.ON)
    
    # Pad and align final images and ON masks
    finalIm1, finalIm2, finalON1, finalON2 = pad_images_to_align_featuresinEdit(ellipse_processor_on1, ellipse_processor_2_rotated)

    return finalIm1, finalIm2, finalON1, finalON2

def alignClosest(img_canvases, on_canvases, vessel_canvases, sorted_centers, viewer):
    """
    Aligns and centers the closest image, ON, and vessel layers on their canvas using EllipseProcessor.
    
    Parameters
    ----------
    img_canvases : list of np.ndarray
        List of image canvases to align.
    on_canvases : list of np.ndarray
        List of ON canvases to align.
    vessel_canvases : list of np.ndarray
        List of vessel canvases to align.
    sorted_centers : list of tuple
        List of tuples containing (layer_name, (center_y, center_x), distance) for sorted centers.
    viewer : napari.Viewer
        The Napari viewer instance for visualization.
    
    Returns
    -------
    tuple
        Rotated image, ON, and vessel canvases for the closest layer.
    """
    # Get the closest layer info
    closest_layer_name, closest_center, _ = sorted_centers[0]

    # Retrieve the closest image, ON, and vessel canvas
    closest_img_canvas = img_canvases[0]
    closest_on_canvas = on_canvases[0]
    closest_vessel_canvas = vessel_canvases[0]

    # Initialize EllipseProcessor with the ON, Image, and Vessel data for the closest layer
    ellipse_processor = EllipseProcessor(closest_on_canvas, closest_img_canvas, closest_vessel_canvas, viewer)
    ellipse_processor.process_image(closest_on_canvas)

    # Rotate to align the major axis of the ellipse to 12 o'clock
    ellipse_processor.rotate_image_and_elements()

    # Retrieve the rotated images and masks
    rotated_img_canvas = ellipse_processor.rotated_Img
    rotated_on_canvas = ellipse_processor.rotated_ON
    rotated_vessel_canvas = ellipse_processor.rotated_Ves

    # Display the rotated canvases in Napari
    #viewer.add_image(rotated_img_canvas, name=closest_layer_name + "_Rotated Canvas Img")
    #viewer.add_labels(rotated_on_canvas, name=closest_layer_name + "_Rotated Canvas ON")
    #viewer.add_labels(rotated_vessel_canvas, name=closest_layer_name + "_Rotated Canvas Vessels")

    return rotated_img_canvas, rotated_on_canvas, rotated_vessel_canvas

def center_layers_on_canvas(viewer, sorted_centers, canvas_shape):
    """
    Centers each image, ON, and vessel layer based on sorted center points, aligning each layer's center to the canvas center.

    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance containing the layers.
    sorted_centers : list of tuples
        List of tuples (layer_name, center_coords, distance) sorted by distance from the canvas center.
    canvas_shape : tuple
        Shape of the canvas (height, width) calculated from `calculate_canvas_size`.
    
    Returns
    -------
    tuple of lists
        Lists containing all centered canvases for images, ON, and vessels.
    """
    # Initialize lists to store the canvases
    img_canvases = []
    on_canvases = []
    vessel_canvases = []

    # Define canvas center coordinates
    canvas_center_y, canvas_center_x = canvas_shape[0] // 2, canvas_shape[1] // 2

    # Step 3: Process each layer in sorted order
    for layer_name, center_coords, _ in sorted_centers:  # Unpack only layer_name and center_coords
        center_y, center_x = center_coords  # Extract center coordinates
        # Retrieve data for the current image, ON, and vessels
        img_data = viewer.layers[layer_name].data
        on_data = viewer.layers[layer_name + "_Seg"].data
        vessel_data = viewer.layers[layer_name + "_Seg [1]"].data

        # Create a blank canvas for each type
        canvas_img = np.zeros(canvas_shape, dtype=img_data.dtype)
        canvas_on = np.zeros(canvas_shape, dtype=on_data.dtype)
        canvas_vessel = np.zeros(canvas_shape, dtype=vessel_data.dtype)

        # Calculate the shift needed to align the layer's center to the canvas center
        shift_y = int(canvas_center_y - center_y)
        shift_x = int(canvas_center_x - center_x)

        # Calculate start coordinates, ensuring the layer will be centered on the canvas
        start_y = max(0, shift_y)
        start_x = max(0, shift_x)

        # Ensure slices don't exceed canvas bounds
        end_y_img = min(start_y + img_data.shape[0], canvas_shape[0])
        end_x_img = min(start_x + img_data.shape[1], canvas_shape[1])
        
        end_y_on = min(start_y + on_data.shape[0], canvas_shape[0])
        end_x_on = min(start_x + on_data.shape[1], canvas_shape[1])
        
        end_y_vessel = min(start_y + vessel_data.shape[0], canvas_shape[0])
        end_x_vessel = min(start_x + vessel_data.shape[1], canvas_shape[1])

        # Place each layer onto the canvas, centered according to its shift
        canvas_img[start_y:end_y_img, start_x:end_x_img] = img_data[:end_y_img - start_y, :end_x_img - start_x]
        canvas_on[start_y:end_y_on, start_x:end_x_on] = on_data[:end_y_on - start_y, :end_x_on - start_x]
        canvas_vessel[start_y:end_y_vessel, start_x:end_x_vessel] = vessel_data[:end_y_vessel - start_y, :end_x_vessel - start_x]

        # Append each centered canvas to the respective list
        img_canvases.append(canvas_img)
        on_canvases.append(canvas_on)
        vessel_canvases.append(canvas_vessel)

    return img_canvases, on_canvases, vessel_canvases

def display_canvases_in_napari(viewer, img_canvases, on_canvases, vessel_canvases):
    """
    Displays lists of image, ON, and vessel canvases in the Napari viewer.

    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance where the canvases will be displayed.
    img_canvases : list of np.ndarray
        List of centered image canvases.
    on_canvases : list of np.ndarray
        List of centered ON canvases.
    vessel_canvases : list of np.ndarray
        List of centered vessel canvases.
    """
    for i, (canvas_img, canvas_on, canvas_vessel) in enumerate(zip(img_canvases, on_canvases, vessel_canvases)):
        # Display each canvas as a separate layer in the viewer
        viewer.add_image(canvas_img, name=f'Centered Image Canvas {i+1}', colormap='gray')
        viewer.add_labels(canvas_on, name=f'Centered ON Canvas {i+1}')
        #viewer.add_labels(canvas_vessel, name=f'Centered Vessel Canvas {i+1}')

def findRotationAngles(center_canv_ves, vessel_canvases, sorted_centers):
    """
    Uses warp_polar to compare each vessel layer to the center vessel layer and calculates the rotation angle.

    Parameters
    ----------
    center_canv_ves : np.ndarray
        The centered vessel canvas for the reference (centermost) layer.
    vessel_canvases : list of np.ndarray
        List of centered vessel canvases to be compared to the reference.
    sorted_centers : list of tuples
        Sorted list of tuples containing layer names, center coordinates, and distances.

    Returns
    -------
    rotation_angles : dict
        Dictionary with layer names as keys and rotation angles as values.
    """
    # Step 1: Convert the reference vessel canvas to polar coordinates
    radius = min(center_canv_ves.shape) // 2
    #radius = 200
    #output_shape = (radius, 720)
    reference_polar = warp_polar(center_canv_ves, radius=radius)[:, 50:-1]

    # Initialize a dictionary to store rotation angles with layer names
    rotation_angles = {}

    # Step 2: Compare each vessel canvas to the reference in the order of sorted_centers
    for (layer_name, _, _), vessel_canvas in zip(sorted_centers, vessel_canvases):
        # Convert the current vessel canvas to polar coordinates
        current_polar = warp_polar(vessel_canvas, radius=radius)[:, 50:-1]

        # Calculate the rotation angle using phase cross-correlation
        shifts, error, phasediff = phase_cross_correlation(reference_polar, current_polar, normalization=None)
        rotation_angle = -float(shifts[0])  # Negative to align to 12 o'clock

        # Store the calculated rotation angle with the layer name
        rotation_angles[layer_name] = rotation_angle

    return rotation_angles


def applyRotationAngles(img_canvases, on_canvases, vessel_canvases, rotation_angles, sorted_centers, center_index=0):
    """
    Rotates each Img, ON, and Vessel canvas according to its corresponding rotation angle,
    excluding the center canvas and starting rotations with the second layer.

    Parameters
    ----------
    img_canvases : list of np.ndarray
        List of canvases for images to be rotated.
    on_canvases : list of np.ndarray
        List of canvases for ON masks to be rotated.
    vessel_canvases : list of np.ndarray
        List of canvases for vessel masks to be rotated.
    rotation_angles : dict
        Dictionary of rotation angles with layer names as keys.
    sorted_centers : list of tuples
        List of tuples where each contains (layer_name, center_coordinates, distance).
    center_index : int, optional
        Index of the center canvas in the lists (default is 0).

    Returns
    -------
    rotated_img_canvases : list of np.ndarray
        List of rotated image canvases.
    rotated_on_canvases : list of np.ndarray
        List of rotated ON canvases.
    rotated_vessel_canvases : list of np.ndarray
        List of rotated vessel canvases.
    """
    rotated_img_canvases = []
    rotated_on_canvases = []
    rotated_vessel_canvases = []

    # Cycle through each canvas in sorted_centers, skipping rotation for the center canvas
    for i, ((layer_name, _, _), img, on, ves) in enumerate(zip(sorted_centers, img_canvases, on_canvases, vessel_canvases)):
        if i == center_index:
            # Keep the center canvas as is without rotation
            rotated_img_canvases.append(img)
            rotated_on_canvases.append(on)
            rotated_vessel_canvases.append(ves)
            continue

        # Retrieve the rotation angle for the current layer, starting rotations with the second item
        angle = rotation_angles.get(layer_name, 0) if i > center_index else 0

        # Apply rotation to each canvas
        rotated_img = rotate(img, angle, resize=False)
        rotated_on = rotate(on, angle, resize=False)
        rotated_vessel = rotate(ves, angle, resize=False)

        # Append rotated canvases to the respective lists
        rotated_img_canvases.append(rotated_img)
        rotated_on_canvases.append(rotated_on)
        rotated_vessel_canvases.append(rotated_vessel)

    return rotated_img_canvases, rotated_on_canvases, rotated_vessel_canvases

def display_aligned_images(viewer, rotated_img_canvases, rotated_on_canvases, rotated_vessel_canvases,
                           rotated_img_canvas, rotated_on_canvas, rotated_vessel_canvas):
    """
    Adds the aligned image, ON, and vessel canvases to Napari as separate layers, 
    excluding the first item in the rotated canvases lists and replacing it with reference canvases.
    
    Also generates new arrays with the updated values.

    Parameters
    ----------
    viewer : napari.Viewer
        The Napari viewer instance to which the canvases will be added.
    rotated_img_canvases : list of np.ndarray
        List of aligned (rotated) image canvases.
    rotated_on_canvases : list of np.ndarray
        List of aligned (rotated) ON canvases.
    rotated_vessel_canvases : list of np.ndarray
        List of aligned (rotated) vessel canvases.
    rotated_img_canvas : np.ndarray
        The reference aligned (rotated) image canvas.
    rotated_on_canvas : np.ndarray
        The reference aligned (rotated) ON canvas.
    rotated_vessel_canvas : np.ndarray
        The reference aligned (rotated) vessel canvas.

    Returns
    -------
    new_img_canvases : list of np.ndarray
        New list of image canvases with the first value replaced.
    new_on_canvases : list of np.ndarray
        New list of ON canvases with the first value replaced.
    new_vessel_canvases : list of np.ndarray
        New list of vessel canvases with the first value replaced.
    """
    # Create new arrays by replacing the first item with the reference canvas
    new_img_canvases = [rotated_img_canvas] + rotated_img_canvases[1:]
    new_on_canvases = [rotated_on_canvas] + rotated_on_canvases[1:]
    new_vessel_canvases = [rotated_vessel_canvas] + rotated_vessel_canvases[1:]

    # Display the canvases in Napari
    #for i, (img_canvas, on_canvas, vessel_canvas) in enumerate(zip(new_img_canvases, new_on_canvases, new_vessel_canvases)):
       # viewer.add_image(img_canvas, name=f'Aligned Image Canvas {i+1}', colormap='gray')
       # viewer.add_labels(on_canvas.astype(np.int32), name=f'Aligned ON Canvas {i+1}')
       # viewer.add_labels(vessel_canvas.astype(np.int32), name=f'Aligned Vessel Canvas {i+1}')

    return new_img_canvases, new_on_canvases, new_vessel_canvases

def match_histograms_to_max_variance(canvases):
    """
    Matches the histograms of all canvases to the one with the greatest variance.

    Parameters
    ----------
    canvases : list of np.ndarray
        List of image canvases to process.

    Returns
    -------
    matched_canvases : list of np.ndarray
        List of canvases with histograms matched to the one with the greatest variance.
    """
    # Calculate variances for each canvas
    variances = [np.var(canvas) for canvas in canvases]
    
    # Find the index of the canvas with the highest variance
    max_var_index = np.argmax(variances)
    
    # Use the canvas with the highest variance as the reference
    reference_canvas = canvases[max_var_index]
    
    # Initialize an empty list to store the matched canvases
    matched_canvases = []
    
    # Loop through each canvas and match its histogram to the reference
    for canvas in canvases:
        matched_canvas = match_histograms(canvas, reference_canvas)
        matched_canvases.append(matched_canvas)
    
    return matched_canvases

def normalize_and_overlay(canvases):
    # Initialize the result with the first normalized canvas
    result = (canvases[0] / canvases[0].max() * 255).astype(np.uint8) if canvases[0].max() > 0 else canvases[0].astype(np.uint8)
    
    # Loop through the rest of the canvases and overlay them
    for canvas in canvases[1:]:
        # Normalize the current canvas
        normalized_canvas = (canvas / canvas.max() * 255).astype(np.uint8) if canvas.max() > 0 else canvas.astype(np.uint8)
        
        # Overlay it onto the cumulative result
        result = np.where(result <16, normalized_canvas, result)
    
    return result

@magic_factory(
        folder={"label": "Target Directory","mode":"d","tooltip": "Directory Containing files or subdirectories with OCT .prof files to process"},
)
def enface_registration_plugin(folder=Path("."), 
                          onnx_path_ON=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                          onnx_path_Vessels=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                          radius:int=1,
                          threshold:float=0.0):
    """"""
    enface_registration_thread(folder,
                               onnx_path_ON,
                               onnx_path_Vessels,
                               radius,
                               threshold)
    
    return

#@thread_worker(connect={"returned": viewer.add_layer})
@thread_worker(connect={"yielded": viewer.add_layer})
def enface_registration_thread(folder, 
                          onnx_path_ON=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                          onnx_path_Vessels=Path("../onnx_models/bscan/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856/UWF_OCT_Bscan_seg_TD_Full_EP_250_PR_16-mixed_SD_60_06-23-2024_19h21m_top_10-epoch=0247-step=17856.onnx"),
                          radius:int=420,
                          threshold:float=0.0):
    """"""
    show_info("Enface Registration thread has started\n")

    #Glob Folder
    filelist = list(Path(folder).glob("*.prof"))
    enfaces = []
    #Generate Enface
    for filename in filelist:
        layer = _prof_reader.prof_get_reader(filename.__str__)
        print(filename)
        img = generate_enface(layer.data)
        enfaces.append(img)
        # ONseg = enface_onnx_seg_func(img,onnx_path=Path(onnx_path_ON), use_cpu =True,DoG=True,blur=True,log_adjust=True)
        
        
        
    # viewer = napari.current_viewer() 
    # mask_array = extract_masks_from_layers(viewer)                                  # Create Mask Array
    # center = find_center_of_largest_array(mask_array)                               # Find center of largest Image
    # crop_dilate_all_layers(viewer, radius=radius)                                      # Crop the background noise out
    # pad_all_layers(viewer)   
    
    # for img in enfaces:
    #     vessels = enface_onnx_seg_func(img, onnx_path=Path(onnx_path_Vessels),use_cpu=True,DoG=True,blur=True,log_adjust=True)


    # image_mask_pairs = pair_images_and_masks(viewer)                                # Get pairs of images and masks
    # ellipse_centers = extract_ellipse_centers(viewer)                               # Get centers for each image-mask pair
    # closest_layer, closest_center, distance = find_closest_center(ellipse_centers)  # Identify the center-most image
    # furthest_centers = find_furthest_centers(ellipse_centers)
    # canvas_shape, shift_offsets, translations = calculate_canvas_size(ellipse_centers)
    # sorted_centers = find_closest_centers_sorted(ellipse_centers)
    # img_canvases, on_canvases, vessel_canvases = center_layers_on_canvas(viewer, sorted_centers, canvas_shape)
    # center_rot_img_canvas, center_rot_on_canvas, center_rot_vessel_canvas = alignClosest(img_canvases, on_canvases, vessel_canvases, sorted_centers, viewer)
    # rotation_angles = findRotationAngles(center_rot_vessel_canvas, vessel_canvases, sorted_centers)
    # rotated_img_canvases, rotated_on_canvases, rotated_vessel_canvases = applyRotationAngles(img_canvases, on_canvases, vessel_canvases, rotation_angles, sorted_centers, center_index=0)
    # new_img_canvases, new_on_canvases, new_vessel_canvases = display_aligned_images(viewer, rotated_img_canvases, rotated_on_canvases, rotated_vessel_canvases, center_rot_img_canvas, center_rot_on_canvas, center_rot_vessel_canvas)
    # new_img_canvases = match_histograms_to_max_variance(new_img_canvases)

    # final_img_canvas = normalize_and_overlay(new_img_canvases)
    # viewer.add_image(final_img_canvas)



    for idx, output in enumerate(output):
        out_data = output[0]
        suffix = idx
        layer_type = "image"

        add_kwargs = {"name": f"{folder}_{suffix}"}
        layer = Layer.create(out_data,add_kwargs,layer_type)
        yield layer

    # #output_image will be the result
    # output_image = np.zeros(100,100)

    # name = folder
    # add_kwargs = {"name": f"{name}"}
    # layer_type = "image"
    # layer = Layer.create(output_image,add_kwargs,layer_type)
    # yield layer

    show_info("Enface Registration thread has completed\n")
    #yield out_layer


