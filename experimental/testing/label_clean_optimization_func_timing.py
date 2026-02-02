"""
optimize the label cleaning code/ trouble shoot
"""
import gc
from pathlib import Path
import time
import timeit
from tqdm import tqdm
from typing import Literal

import cc3d
import napari
import numpy as np

from napari_cool_tools_io import torch, device, memory_stats
from napari_cool_tools_io._npz_reader import npz_file_reader
from napari_cool_tools_vol_proc._masking_tools_funcs import project_2d_mask
from napari_cool_tools_segmentation._label_cleaning_funcs_v2 import generate_elliptical_mask, generate_circular_mask, generate_depth_map, generate_thick_map, RPE_layer_coords, Retinal_surface_coords

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

def main():
    #use_accelerator = True

    timeit_iterations = 20 # 10
    return_numpy = True
    normalized_depth_map = False

    test_labels_path = Path(r"F:\_temp_test_data\08882298-2023_11_15-11_41_08_ret_chor_seg.npz")
    test_labels_data_og = npz_file_reader(test_labels_path,return_layer=False)
    #label_paths = [test_labels_path]

    #viewer = napari.Viewer(show=False)
    total_test_time_start = time.perf_counter()
    print("Start time comparison for functions")
    # print("fdvw comparison start")
    # fdvw_cpu_time = timeit.timeit(lambda:find_densest_voxel_window(test_labels_data_og,axis=1,window_along_axis=100,use_accelerator=False),number=timeit_iterations)
    # fdvw_gpu_time = timeit.timeit(lambda:find_densest_voxel_window(test_labels_data_og,axis=1,window_along_axis=100,use_accelerator=True),number=timeit_iterations)
    # print("fdvw comparison end")

    # print("gcm comparison start")
    # gcm_cpu_time = timeit.timeit(lambda:generate_circular_mask(test_labels_data_og.shape,use_input_depth=True,use_accelerator=False,return_numpy=return_numpy),number=timeit_iterations)
    # gcm_gpu_time = timeit.timeit(lambda:generate_circular_mask(test_labels_data_og.shape,use_input_depth=True,use_accelerator=True,return_numpy=return_numpy),number=timeit_iterations)
    # print("gcm comparison end")

    print("gem comparison start")
    gem_cpu_time = timeit.timeit(lambda:generate_elliptical_mask(test_labels_data_og.shape,use_input_depth=True,use_accelerator=False,return_numpy=return_numpy),number=timeit_iterations)
    gem_gpu_time = timeit.timeit(lambda:generate_elliptical_mask(test_labels_data_og.shape,use_input_depth=True,use_accelerator=True,return_numpy=return_numpy),number=timeit_iterations)
    print("gem comparison end")

    # print("gdm comparison start")
    # gdm_cpu_time = timeit.timeit(lambda:generate_depth_map(test_labels_data_og,axis=1,normalized=normalized_depth_map,use_accelerator=False,return_numpy=return_numpy),number=timeit_iterations)
    # gdm_gpu_time = timeit.timeit(lambda:generate_depth_map(test_labels_data_og,axis=1,normalized=normalized_depth_map,use_accelerator=True,return_numpy=return_numpy),number=timeit_iterations)
    # print("gdm comparison end")

    # print("gtm comparison start")
    # gtm_cpu_time = timeit.timeit(lambda:generate_thick_map(test_labels_data_og,use_accelerator=False,return_numpy=return_numpy),number=timeit_iterations)
    # gtm_gpu_time = timeit.timeit(lambda:generate_thick_map(test_labels_data_og,use_accelerator=True,return_numpy=return_numpy),number=timeit_iterations)
    # print("gtm comparison end")

    # print("rpelc comparison start")
    # rpelc_cpu_time = timeit.timeit(lambda:RPE_layer_coords(test_labels_data_og==1,use_accelerator=False,return_numpy=return_numpy),number=timeit_iterations)
    # rpelc_gpu_time = timeit.timeit(lambda:RPE_layer_coords(test_labels_data_og==1,use_accelerator=True,return_numpy=return_numpy),number=timeit_iterations)
    # print("rpelc comparison end")

    # print("rslc comparison start")
    # rslc_cpu_time = timeit.timeit(lambda:Retinal_surface_coords(test_labels_data_og==1,use_accelerator=False,return_numpy=return_numpy),number=timeit_iterations)
    # rslc_gpu_time = timeit.timeit(lambda:Retinal_surface_coords(test_labels_data_og==1,use_accelerator=True,return_numpy=return_numpy),number=timeit_iterations)
    # print("rslc comparison end")

    total_test_time_end = time.perf_counter()
    print(f"Total test time: {total_test_time_end-total_test_time_start}")

    # print(f"fdvw cpu vs gpu per call: {fdvw_cpu_time/timeit_iterations} vs {fdvw_gpu_time/timeit_iterations}")
    # print(f"gcm cpu vs gpu per call: {gcm_cpu_time/timeit_iterations} vs {gcm_gpu_time/timeit_iterations}")
    print(f"gem cpu vs gpu per call: {gem_cpu_time/timeit_iterations} vs {gem_gpu_time/timeit_iterations}")
    # print(f"gdm cpu vs gpu per call: {gdm_cpu_time/timeit_iterations} vs {gdm_gpu_time/timeit_iterations}")
    # print(f"gtm cpu vs gpu per call: {gtm_cpu_time/timeit_iterations} vs {gtm_gpu_time/timeit_iterations}")
    # print(f"rpelc cpu vs gpu per call: {rpelc_cpu_time/timeit_iterations} vs {rpelc_gpu_time/timeit_iterations}")
    # print(f"rslc cpu vs gpu per call: {rslc_cpu_time/timeit_iterations} vs {rslc_gpu_time/timeit_iterations}")
    print("Time comparison for functions complete")


    # #label_paths = list(Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2").glob("*_retchor.npz"))
    # label_paths = list(Path(r"E:\_rebatch_conjugate_test_12_01_2025\output2").glob("*_retchor.npz")) + label_paths

    # iccat_numpy_time = 0
    # iccat_torch_cpu_time = 0
    # iccat_torch_gpu_time = 0

    # label_progress = tqdm(label_paths)
    # for test_labels_path in label_progress:
    #     name = test_labels_path.stem
    #     label_progress.set_description = f"Processing {name}"

    #     test_labels_data = npz_file_reader(test_labels_path,return_layer=False)

    #     circular_mask = generate_circular_mask(test_labels_data.shape,use_input_depth=True,use_accelerator=use_accelerator,return_numpy=True)
    #     depth_map = generate_depth_map(test_labels_data,axis=1,normalized=True,use_accelerator=use_accelerator,return_numpy=True)
    #     rpe_coords = RPE_layer_coords(test_labels_data==1,use_accelerator=True,return_numpy=True)
    #     ret_surf_coords = Retinal_surface_coords(test_labels_data==1,use_accelerator=True,return_numpy=True)
    #     #print(rpe_coords,rpe_coords.shape)

    #     start_numpy = time.perf_counter()
    #     desired_labels = isolate_connected_components_above_threshold(test_labels_data==1,threshold=90_000,retain_label_values=True,verbose=False)
    #     end_numpy = time.perf_counter()
    #     iccat_numpy_time = iccat_numpy_time + (end_numpy-start_numpy)

    #     start_torch_cpu = time.perf_counter()
    #     desired_labels = isolate_connected_components_above_threshold_v2(test_labels_data==1,threshold=90_000,retain_label_values=True,verbose=False)
    #     end_torch_cpu = time.perf_counter()
    #     iccat_torch_cpu_time = iccat_torch_cpu_time + (end_torch_cpu-start_torch_cpu)

    #     start_torch_gpu = time.perf_counter()
    #     desired_labels = isolate_connected_components_above_threshold_v3(test_labels_data==1,threshold=90_000,retain_label_values=True,verbose=False)
    #     end_torch_gpu = time.perf_counter()
    #     iccat_torch_gpu_time = iccat_torch_cpu_time + (end_torch_gpu-start_torch_gpu)

    #     #desired_mask = isolate_connected_components_above_threshold(test_labels_data==1,threshold=20_000,retain_label_values=False,verbose=True)
    #     #cleaned_retina = isolate_connected_components_above_threshold(test_labels_data==1,threshold=100,retain_label_values=False,verbose=False)
    #     if viewer:
    #         viewer.add_labels(test_labels_data,name=f"{name}_labels")
    #         viewer.add_labels(desired_labels,name=f"{name}_mani_components")
    #         #viewer.add_labels(desired_mask,visible=False)
    #         #viewer.add_labels(cleaned_retina,visible=False)
    #         viewer.add_labels(circular_mask,visible=False,name=f"{name}_cylinder_mask")
    #         viewer.add_image(depth_map,name=f"{name}_depth_map")
    #         viewer.add_points(ret_surf_coords,size=0.1,border_color="green",face_color="green",blending="translucent",name=f"{name}_ret_surf_coords")
    #         viewer.add_points(rpe_coords,size=0.1,border_color="red",face_color="red",blending="translucent",name=f"{name}_rpe_coords")

    # print(f"iccat numpy: {iccat_numpy_time/len(label_paths)}, torch_cpu: {iccat_torch_cpu_time/len(label_paths)}, torch_gpu: {iccat_torch_gpu_time/len(label_paths)}\n")

    # if viewer:
    #     viewer.show()
    #     napari.run()

if __name__ == "__main__":
    main()