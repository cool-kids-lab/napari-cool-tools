"""
optimize the label cleaning code/ trouble shoot
"""
import gc
from pathlib import Path
import time
import timeit
import threading
from tqdm import tqdm
from typing import Literal

import cc3d
import napari
import numpy as np

from napari_cool_tools_io import torch, device, memory_stats
from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_io._npz_writer import save_npz
from napari_cool_tools_segmentation._label_cleaning_funcs_v2 import generate_circular_mask, generate_depth_map, generate_elliptical_mask,generate_elliptical_masks, generate_maps, generate_thick_map, get_lens_mask_for_oct_volume, RPE_layer_coords, Retinal_surface_coords
from napari_cool_tools_vol_proc._masking_tools_funcs import project_2d_mask

def isolate_connected_components_above_threshold_v4(mask:np.ndarray,threshold:int,connectivity:Literal[4,8,6,18,26]=6,retain_label_values:bool=True,return_remainder:bool=True,verbose:bool=False):
    """
    """
    #TODO validate dimensionality of mask and process according to connectivity
    connectivity_map = cc3d.connected_components(mask,connectivity=connectivity,binary_image=True)
    #connectivity_map = torch.from_numpy(connectivity_map).to(torch.int)
    connectivity_map = torch.as_tensor(connectivity_map).to(torch.int)

    voxel_counts = torch.bincount(connectivity_map.view(-1))
    
    if verbose:
        print(f"number of labels: {len(voxel_counts)}")
  
    sorted_indices = torch.argsort(voxel_counts)
    threshold_idx = (voxel_counts[sorted_indices] > threshold).to(torch.int8).argmax() # This will break overflow error if there are > 255 components > than the threshold
    
    # always more background than signal for this data, but for general case mask 0 index as it is always the bakground
    label_indices_above_threshold = torch.flip(sorted_indices[threshold_idx:-1],dims=[0])

    if verbose:
        label_vals_above_threshold = voxel_counts[label_indices_above_threshold]
        print(f"Component indicies above {threshold}:\n{label_indices_above_threshold}\n")
        print(f"Components sizes above {threshold}:\n{label_vals_above_threshold}\n")
        del label_vals_above_threshold

    #create lookup table
    lookup_table = torch.zeros((len(voxel_counts)),dtype=torch.int64)
    for idx,label in enumerate(label_indices_above_threshold):
        if retain_label_values:
            lookup_table[label] = idx+1
        else:
            lookup_table[label] = 1

    major_components_by_size = lookup_table[connectivity_map].to(torch.uint8).numpy()
    if not return_remainder:
        return major_components_by_size
    else:
        minor_comoponents = np.zeros_like(mask,dtype=np.uint8)
        minor_comoponents[~(major_components_by_size>0) & ~(mask==0)] = len(label_indices_above_threshold)+1
        return(major_components_by_size,minor_comoponents)

@torch.inference_mode()    
def find_densest_voxel_window(mask:np.ndarray|torch.Tensor,axis:int=1,window_along_axis:int=100,use_accelerator:bool=True):
    """"""
    if use_accelerator:
        current_device = device
    else:
        current_device = "cpu"

    mask = torch.as_tensor(mask > 0,device=current_device).float()

    # Generate density map along desired axis
    sum_dims = [i for i in range(mask.ndim) if i != axis]
    density_map_along_axis = mask.sum(dim=sum_dims)

    # Shape density map for convolutions function (Batch, Channel, length)
    density_map_along_axis = density_map_along_axis.view(1,1,-1)
    kernel = torch.ones(1,1,window_along_axis,device=current_device)

    # Convolve and generate sliding window sums
    sliding_window_sums = torch.nn.functional.conv1d(density_map_along_axis,kernel)

    # Get start index of the window with max density
    max_density_window_start = torch.argmax(sliding_window_sums).item()

    if use_accelerator:
        del mask,kernel
        gc.collect()
        torch.cuda.empty_cache()

    return max_density_window_start

def main():
    use_accelerator = True
    voxel_threshold = 100_000
    # iccat_numpy_time = 0
    window_along_axis = 80 #100 #80 #150 #100
    window_offsets = (0,0) #(10,25)
    # iccat_torch_cpu_time = 0
    # iccat_torch_gpu_time = 0

    # oct_masks = generate_elliptical_masks()

    test_labels_path = Path(r"F:\_temp_test_data\08882298-2023_11_15-11_41_08_ret_chor_seg.npz")
    #test_labels_data_og = npz_file_reader(test_labels_path,return_layer=False)
    label_paths = [test_labels_path]

    viewer = None
    viewer = napari.Viewer(show=False)
    

    #label_paths = list(Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2").glob("*_retchor.npz"))
    label_paths = list(Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2").glob("*_ret_chor_seg.npz")) + label_paths
    #label_paths = list(Path(r"E:\_rebatch_conjugate_test_12_01_2025\output3").glob("*_ret_chor_seg.npz")) + label_paths

    print(f"File list: {label_paths}\n")

    label_progress = tqdm(label_paths[3:4])
    #label_progress = tqdm(label_paths[:])
    for test_labels_path in label_progress:
        name = test_labels_path.stem
        label_progress.set_description = f"Processing {name}"

        motor_position = None
        test_labels_data,attributes,layer_type = npz_file_reader(test_labels_path,return_layer=True)[0]
        if "motor_position" in attributes["metadata"]:
            motor_position = int(attributes["metadata"]["motor_position"])
        #test_labels_data, = npz_file_reader(test_labels_path,return_layer=False)
        #test_labels_data = torch.as_tensor(test_labels_data,dtype=torch.int16)

        #elliptical_mask = generate_elliptical_mask(test_labels_data.shape,use_input_depth=True,use_accelerator=False,return_numpy=True)
        elliptical_mask = generate_circular_mask(test_labels_data.shape,use_input_depth=True,use_accelerator=False,return_numpy=True)
        #circular_mask = generate_circular_mask(test_labels_data.shape,use_input_depth=True,use_accelerator=False,return_numpy=True)
        #lens_mask = get_lens_mask_for_oct_volume(test_labels_data.shape,mask_options=oct_masks,verbose=False).astype(bool)

        # #start_cowl_mask_idx = find_densest_voxel_window((desired_labels>0)[:,:1008,:],axis=1,window_along_axis=window_along_axis,use_accelerator=True)
        # start_cowl_mask_idx = find_densest_voxel_window((test_labels_data>0)[:,:1008,:],axis=1,window_along_axis=window_along_axis,use_accelerator=True)
        # cowl_window_mask = np.zeros_like(test_labels_data,dtype=np.uint8)
        # cowl_window_mask[:,start_cowl_mask_idx:(start_cowl_mask_idx+window_along_axis),:] = 10
        density_window_search_limit = (test_labels_data.shape[1]//2) + ((window_along_axis+window_offsets[0]+window_offsets[1])//2)
        #print(f"Density search window limit: {density_window_search_limit}\n")
        start_cowl_mask_idx2 = find_densest_voxel_window((test_labels_data>0)[:,:density_window_search_limit,:],axis=1,window_along_axis=window_along_axis,use_accelerator=use_accelerator)
        cowl_window_mask2 = np.zeros_like(test_labels_data,dtype=np.uint8)
        #cowl_window_mask2 = torch.zeros_like(test_labels_data,dtype=torch.uint8)
        cowl_window_mask2[:,start_cowl_mask_idx2-window_offsets[0]:(start_cowl_mask_idx2+window_along_axis+window_offsets[1]),:] = 9#6

        with torch.inference_mode():
            combo_mask = (~(torch.as_tensor(elliptical_mask,device=device,dtype=torch.uint8).to(device) > 0) & (torch.as_tensor(cowl_window_mask2,device=device,dtype=torch.uint8) > 0)).cpu().numpy().astype(np.uint8)
            # combo_mask = (~(torch.as_tensor(circular_mask,device=device,dtype=torch.uint8).to(device) > 0) & (torch.as_tensor(cowl_window_mask2,device=device,dtype=torch.uint8) > 0)).cpu().numpy()
            torch.cuda.empty_cache()
            lens_labels_data = torch.as_tensor(test_labels_data,device=device,dtype=torch.uint8)
            lens_labels_data[torch.as_tensor(combo_mask,device=device,dtype=torch.bool)] = 0
            lens_labels_data = lens_labels_data.cpu().numpy()
            torch.cuda.empty_cache()
        #print(lens_mask.shape,cowl_window_mask2.shape)
        
        #test_labels_data[~lens_mask & cowl_window_mask2] = 0
        #lens_labels_data = test_labels_data.copy()
        #lens_labels_data[(~lens_mask) & (cowl_window_mask2)] = 0

        #print("we got here!!")

        # start_torch_gpu = time.perf_counter()
        #desired_labels,remnants = isolate_connected_components_above_threshold_v4(test_labels_data==1,threshold=voxel_threshold,retain_label_values=True,return_remainder=True,verbose=False)
        desired_labels,remnants = isolate_connected_components_above_threshold_v4(lens_labels_data==1,threshold=voxel_threshold,retain_label_values=True,return_remainder=True,verbose=False)
        desired_labels = desired_labels.astype("uint8")
        #desired_labels = torch.as_tensor(desired_labels,dtype=torch.uint8,device="cpu")
        # end_torch_gpu = time.perf_counter()
        # iccat_torch_gpu_time = iccat_torch_cpu_time + (end_torch_gpu-start_torch_gpu)
        # print(f"torch_cpu: {iccat_torch_gpu_time/len(label_paths)}")

        # depth_map = generate_depth_map(desired_labels==1,axis=1,normalized=False,use_accelerator=use_accelerator,return_numpy=True)
        # # rpe_coords = RPE_layer_coords(desired_labels==1,use_accelerator=True,return_numpy=True)
        # # ret_surf_coords = Retinal_surface_coords(desired_labels==1,use_accelerator=True,return_numpy=True)
        # ret_surf_coords, rpe_coords, thick_map = generate_thick_map(desired_labels,use_accelerator=use_accelerator,return_numpy=True)

        depth_map,ret_surf_coords,rpe_coords,thick_map,difference_map = generate_maps(desired_labels==1,use_accelerator=use_accelerator,return_numpy=True)
        difference_map = difference_map.astype(np.float32)
        
        # depth_artefact_map = depth_map > 133
        # depth_artefact_projection = project_2d_mask(desired_labels,depth_artefact_map,axis=1,swap_axes=False,use_accelerator=True,return_numpy=False)
        overhang_artefact_map = (difference_map > 2) | (difference_map < 0)
        overhang_artefact_projection = project_2d_mask(desired_labels,overhang_artefact_map,axis=1,swap_axes=False,use_accelerator=True,return_numpy=True)
        overhang_artefact_projection = overhang_artefact_projection.astype("uint8")
        #overhanging_labels = torch.as_tensor(desired_labels==1,dtype=torch.bool)
        #overhanging_labels = torch.logical_and(overhang_artefact_projection,desired_labels==1)
        overhanging_labels = (desired_labels==1) & (overhang_artefact_projection > 0) & (cowl_window_mask2 > 0)
        overhanging_labels = overhanging_labels.astype("uint8")
        #overhanging_labels = torch.logical_and(overhanging_labels,cowl_window_mask2)
        # depth_artefact_labels = depth_artefact_projection & torch.as_tensor(desired_labels==1,dtype=torch.bool)

        out_labels = ~overhanging_labels & (desired_labels==1)
        out_labels = out_labels.astype("uint8")

        final_depth_map,final_ret_surf_coords,final_rpe_coords,final_thick_map,final_difference_map = generate_maps(out_labels,use_accelerator=use_accelerator,return_numpy=True)
        final_difference_map = final_difference_map.atype(np.float32)
        #out_labels = (desired_labels==1) == overhang_artefact_projection
        # #overhanging_labels[:,overhanging_labels.shape[1]//2:,:] = 0
        # out_labels = torch.as_tensor(desired_labels==1,dtype=torch.uint8)
        # out_labels[depth_artefact_labels.to(torch.bool)] = 0
        # out_labels[overhanging_labels>1] = 0
        #overhanging_labels.numpy()
        #out_labels=out_labels.numpy().astype(bool)
        final_depth_artefact_map = depth_map > 133
        final_depth_artefact_projection = project_2d_mask(desired_labels,final_depth_artefact_map,axis=1,swap_axes=False,use_accelerator=True,return_numpy=True)
        final_out_labels = out_labels & ~final_depth_artefact_projection
        final_out_labels.astype("uint8")

        #desired_mask = isolate_connected_components_above_threshold(test_labels_data==1,threshold=20_000,retain_label_values=False,verbose=True)
        #cleaned_retina = isolate_connected_components_above_threshold(test_labels_data==1,threshold=100,retain_label_values=False,verbose=False)
        if viewer:
            viewer.add_labels(test_labels_data,name=f"{name}_labels")
            # #viewer.add_labels(lens_labels_data,name=f"{name}_labels_v2")
            # viewer.add_labels(desired_labels,name=f"{name}_main_components")
            viewer.add_labels(remnants,name=f"{name}_minor_components",visible=False)
            # #viewer.add_labels(cowl_window_mask,name=f"{name}_cowl_mask")
            viewer.add_labels(cowl_window_mask2,name=f"{name}_cowl_mask2")
            # #viewer.add_labels(desired_mask,visible=False)
            # #viewer.add_labels(cleaned_retina,visible=False)
            # viewer.add_labels(circular_mask,name=f"{name}_cylinder_mask",visible=False)
            viewer.add_labels(combo_mask*4,name=f"{name}_combo_mask",visible=False)
            # #viewer.add_labels(lens_mask,name=f"{name}_lens_mask",visible=False)
            viewer.add_image(depth_map,name=f"{name}_depth_map")
            viewer.add_image(thick_map,name=f"{name}_thick_map")
            # viewer.add_labels(depth_artefact_map*6,name=f"{name}_depth_artefact_map")
            viewer.add_image(difference_map,name=f"{name}_difference_map")
            # viewer.add_labels(depth_artefact_projection*6,name=f"{name}_depth_artefact_projection")
            # viewer.add_labels(overhang_artefact_map*10,name=f"{name}_overhang_artefact_map")
            # viewer.add_labels(overhang_artefact_projection*10,name=f"{name}_overhang_artefact_projection")
            # viewer.add_labels(overhanging_labels*10,name=f"{name}_overhanging_labels")
            # viewer.add_labels(out_labels*6,name=f"{name}_out_labels")
            viewer.add_image(final_depth_map,name=f"{name}_final_depth_map")
            viewer.add_image(final_thick_map,name=f"{name}_final_thick_map")
            viewer.add_image(final_difference_map,name=f"{name}_final_difference_map")
            viewer.add_labels(final_out_labels*36,name=f"{name}_final_out_labels")
            # viewer.add_points(ret_surf_coords,size=0.1,border_color="green",face_color="green",blending="translucent",name=f"{name}_ret_surf_coords",visible=False)
            # viewer.add_points(rpe_coords,size=0.1,border_color="red",face_color="red",blending="translucent",name=f"{name}_rpe_coords",visible=False)
            # viewer.add_points(final_ret_surf_coords,size=0.1,border_color="green",face_color="green",blending="translucent",name=f"{name}_final_ret_surf_coords",visible=False)
            # viewer.add_points(final_rpe_coords,size=0.1,border_color="red",face_color="red",blending="translucent",name=f"{name}_final_rpe_coords",visible=False)

    #print(f"iccat numpy: {iccat_numpy_time/len(label_paths)}, torch_cpu: {iccat_torch_cpu_time/len(label_paths)}, torch_gpu: {iccat_torch_gpu_time/len(label_paths)}\n")

    if viewer:
        viewer.show()
        napari.run()

if __name__ == "__main__":
    main()
