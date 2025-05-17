"""
"""
import numpy as np
from typing import Tuple,List
from napari.types import ImageData, LabelsData

def circle_circumference_mask(center_x, center_y, radius, image_size):
    """
    Generates a mask representing the circumference of a circle with given center and radius. 
    
    Args:
        center_x (int): X coordinate of the circle center.
        center_y (int): Y coordinate of the circle center.
        radius (int): Radius of the circle.
        image_size (tuple): (width, height) of the image where the mask will be applied. 
    
    Returns:
        np.ndarray: A 2D boolean mask representing the circle circumference.
    """
    
    mask = np.zeros(image_size, dtype=bool)
    
    # Generate coordinates for the circle
    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    
    # Calculate distances from center
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Set mask to True only for points on the circumference 
    mask[(dist > radius - 1) & (dist < radius + 1)] = True 
    
    return mask 

def bscan_label_cleanup(data:LabelsData,input_label_vals:List=[0,1,2],lower_threshold:int=2,upper_threshold:int=400,output_label_val:int=10,hightlight_features:bool=True,debug:bool=True)->LabelsData:
    '''
    Args:
    Returns:
    Raises:
    '''
    import math
    import cv2
    from tqdm import tqdm
    from scipy import stats as sp_stats
    from napari_cool_tools_img_proc._morphology_funcs import morphological_dilation

    if not (data.ndim == 3 or data.ndim == 2):
        raise ValueError(f"Data with dimensions {data.ndim} not curently supported please suply data with 2 or 3 dimensions.")

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    out_data = np.zeros_like(data)

    for v_idx in tqdm(range(len(data)),desc="Cleaning Volume Slices:"):

        if hightlight_features:
            dilations = []
        
        if debug:
            fill_values = []

        for l_idx,label in enumerate(input_label_vals):

            #label_values = (data == label)
            label_values = (data[v_idx] == label)
            nb_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(label_values.astype(np.uint8),connectivity=8)

            feature_sizes = [stat[-1] for stat in stats]
            bb_radii = [((stat[-3]**2 + stat[-2]**2)**(1/2)) for stat in stats]
            #size_threshold = 50 #200
            feature_idxs_below_threshold = [idx for idx,feature in enumerate(feature_sizes) if (feature >lower_threshold and feature <= upper_threshold)]
            selected_sizes = [feature_sizes[idx] for idx in feature_idxs_below_threshold]
            selected_centroids = [centroids[idx] for idx in feature_idxs_below_threshold]
            selected_bb_radii = [bb_radii[idx] for idx in feature_idxs_below_threshold]

            if debug:
                print(f"\ncentorids: {selected_centroids}\nbb_radii: {selected_bb_radii}\nsizes: {selected_sizes}\n")

            for f_idx in range(len(feature_idxs_below_threshold)):
                component = labels == feature_idxs_below_threshold[f_idx]
                new_component = morphological_dilation(component)
                out_component = new_component.astype(bool) ^ component.astype(bool)
                #selected = data[out_component]
                selected = data[v_idx][out_component]
                value = sp_stats.mode(selected)[0]
                if debug:
                    fill_values.append(value)
                #data[component.astype(bool)] = value
                data[v_idx][component.astype(bool)] = value
                if hightlight_features:
                    dilations.append(out_component)

            if debug:
                print(f"fill_values: {fill_values}\n")

            if hightlight_features:
                #combined_dilated_mask = np.zeros_like(data).astype(bool)
                combined_dilated_mask = np.zeros_like(data[v_idx]).astype(bool)
                for mask in dilations:
                    combined_dilated_mask = mask | combined_dilated_mask
                
                out_data[v_idx] = combined_dilated_mask*output_label_val
            else:
                #out_data = data
                out_data[v_idx] = data[v_idx]

    return  out_data #combined_dilated_mask*out_label_val

def group_labels(data:LabelsData,input_label_vals:List=[0,],output_label_val=10)->LabelsData:
    '''
    Args:
    Returns:
    Raises:
    '''
    if len(input_label_vals):
        non_zero_mask = (data > 0)
    else:
        pass

    return non_zero_mask*output_label_val


def mask_relative_to_existing_label(data:ImageData,occurence="first",relative="before",axis:int=0,input_label_val=1,output_label_val=10,volumetric_calc:bool=False)->ImageData:
    '''
    Args:
    Returns:
    Raises:
    '''
    from jj_nn_framework.ndarray_tensor_utils import mask_indicies_along_axis
    
    #for i,d in enumerate(data):
    #    out_data[i] = mask_indicies_along_axis(d,val=input_label_val,axis=axis,occurrence=occurence,relative=relative)

    if not volumetric_calc and data.ndim > 2:
        from tqdm import tqdm
        out_data = np.zeros_like(data)
        for i,d in tqdm(enumerate(data),desc="Generating Relative Mask"):
            out_data[i] = mask_indicies_along_axis(d,val=input_label_val,axis=axis,occurrence=occurence,relative=relative)*output_label_val
    else:
        out_data = mask_indicies_along_axis(data,val=input_label_val,axis=axis,occurrence=occurence,relative=relative)*output_label_val
        
    
    return  out_data

def mask_interface_of_existing_label(data:ImageData,occurence="first",inverse:bool=False,axis:int=0,input_label_val=1,output_label_val=10,volumetric_calc:bool=False)->ImageData:
    '''
    Args:
    Returns:
    Raises:
    '''
    from jj_nn_framework.ndarray_tensor_utils import select_indicies_along_axis
    
    #for i,d in enumerate(data):
    #    out_data[i] = mask_indicies_along_axis(d,val=input_label_val,axis=axis,occurrence=occurence,relative=relative)

    if not volumetric_calc and data.ndim > 2:
        from tqdm import tqdm
        out_data = np.zeros_like(data)
        for i,d in tqdm(enumerate(data),desc="Generating Relative Mask"):
            out_data[i] = select_indicies_along_axis(d,val=input_label_val,axis=axis,occurrence=occurence,inverse=inverse)*output_label_val
    else:
        out_data = select_indicies_along_axis(data,val=input_label_val,axis=axis,occurrence=occurence,inverse=inverse)*output_label_val
        
    
    return  out_data

def create_blank_lable_from_layer(img_data:ImageData)->LabelsData:
    """"""
    empty_labels = np.zeros_like(img_data).astype(np.uint8)
    return empty_labels

def isolate_labeled_volume(img_data:ImageData,lbl_data:LabelsData,label:int)->Tuple[ImageData,ImageData]:
    """
    """    
    label_mask = lbl_data == label
    out_data = img_data.copy()
    out_data[~label_mask] = 0

    return out_data

def project_2d_mask(img_data:ImageData,lbl_data:LabelsData,axis:int=1,swap_axes:bool=True)->LabelsData:
    """
    """
    if axis == 1:
        lbl_data = lbl_data[:,np.newaxis,:]
        if swap_axes:
            lbl_data = lbl_data.transpose(2,1,0)
    elif axis == 0:
        pass
    elif axis == 2:
        pass

    assert(lbl_data.shape[-3] == img_data.shape[-1] and lbl_data.shape[-2] == img_data.shape[-1], f"Mask dimensions {lbl_data.shape} do not match Image dimensions {(img_data.shape[-3],img_data.shape[-1])}\n")
    out_lbl  = np.repeat(lbl_data,img_data.shape[1],axis=axis)

    return out_lbl


