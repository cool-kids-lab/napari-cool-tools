
import os
from multiprocessing import Pool
from pathlib import Path
from napari_cool_tools_io import getWindow, unp_meta
import numpy as np
import torch
from napari_cool_tools_io import device
from napari_cool_tools_io.process_unp import unpack12_torch, dc_subtraction_double_sweep_torch, comp_dis_phase_torch, desine
import torch.nn.functional as F
from tqdm import tqdm

def process_unp_sine_pause_batch_haoshen(unp_file_path:Path, meta: unp_meta, include_hires_in_lowres=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    print("Starting unp file processing.")

    indices = meta.sine_frame_indices
    pause_index = indices[0::2]
    stop_index = indices[1::2]

    print("Pause indices:", pause_index)
    print("Stop indices:", stop_index)

    hires_ratio = meta.sine_hires_ratio # 3
    hires_h = meta.height*hires_ratio
    hires_d = (indices[1] - indices[0])/hires_ratio
    hires_d = int(hires_d)

    ini_delay = meta.delay
    delay = round((ini_delay/10)*(hires_ratio-1) * 2)

    low_res_depth = meta.depth - len(pause_index) *  hires_d * hires_ratio # - 5 * 6 * 3 = 90 - 5
    print("Low res depth:", low_res_depth)
    
    # read 2 bytes size for uint16
    if meta.packed:
        data_size_bytes = int(1.5 * meta.width * meta.height)
    else:
        data_size_bytes = 2 * meta.width * meta.height

    if meta.full_range:
        oct_vol_array = torch.zeros((low_res_depth, meta.height, meta.width), dtype=torch.float32).to(device)
        oct_vol_array_hires = torch.zeros((hires_d*len(pause_index), hires_h, meta.width), dtype=torch.float32).to(device)

        oct_vol_array_lowres_pre = torch.zeros((hires_d*len(pause_index), meta.height, int(meta.width)), dtype=torch.float32).to(device)
        oct_vol_array_lowres_post = torch.zeros((hires_d*len(pause_index), meta.height, int(meta.width)), dtype=torch.float32).to(device)

    else:
        oct_vol_array = torch.zeros((low_res_depth, meta.height, int(meta.width/2)), dtype=torch.float32).to(device)
        oct_vol_array_hires = torch.zeros((hires_d*len(pause_index), hires_h, int(meta.width/2)), dtype=torch.float32).to(device)

        oct_vol_array_lowres_pre = torch.zeros((hires_d*len(pause_index), meta.height, int(meta.width/2)), dtype=torch.float32).to(device)
        oct_vol_array_lowres_post = torch.zeros((hires_d*len(pause_index), meta.height, int(meta.width/2)), dtype=torch.float32).to(device)

    # open file
    with open(unp_file_path, "rb", buffering=0) as byte_reader:
        
        # 1D Hamming window (like np.hamming)
        # hamming = torch.hamming_window(meta.width, periodic=False, dtype=torch.float32, device=device)
        hamming = getWindow(meta.width, meta.windowType, dtype=torch.float32, device=device)
        hamming = hamming.unsqueeze(0).repeat(meta.height, 1)
        # hamming_signal = subtracted_signal * hamming

        # hamming_hires = torch.hamming_window(meta.width, periodic=False, dtype=torch.float32, device=device)
        hamming_hires = getWindow(meta.width, meta.windowType, dtype=torch.float32, device=device)
        hamming_hires = hamming_hires.unsqueeze(0).repeat(hires_h, 1)

        dispMaxOrder = 3

        #TODO this function does not include autodispersion yet. It should be added in the future, but for now we can just use the same coefficients as the low-res frames.
        #Will add this function in the future for batch processing

        frame_counter = 0
        frame_counter_lowres = 0
        frame_counter_hires = 0

        byte_reader.seek(0,0) #reset to beginning of file
        
        # Main OCT Volume process
        for _ in tqdm(range(0, low_res_depth+len(pause_index)), desc="Processing Bscans"):

            if frame_counter in pause_index:
                for _ in range(hires_d):
                    if meta.packed:
                        raw_data = np.frombuffer(byte_reader.read(data_size_bytes * hires_ratio), dtype="<u1")
                        if raw_data.size != data_size_bytes * hires_ratio:
                            continue
                        raw_data = torch.tensor(raw_data).to(device)
                        raw = unpack12_torch(raw_data)
                        raw = raw.reshape((hires_h, meta.width))
                    else:
                        raw_data = np.frombuffer(byte_reader.read(data_size_bytes*hires_ratio), dtype=np.uint16)
                        if raw_data.size != meta.height * meta.width * hires_ratio:
                            continue
                        raw = raw_data.reshape((hires_h, meta.width)).astype(np.float32)
                        raw = torch.tensor(raw).to(device)

                    if meta.dcSubtract:
                    # Subtract the DC signal
                        subtracted_signal = dc_subtraction_double_sweep_torch(raw)
                    else:
                        subtracted_signal = raw

                    # Hamming windowing
                    hamming_signal = subtracted_signal * hamming_hires

                    img_disp_comp = torch.zeros_like(hamming_signal, dtype=torch.complex64, device=device)

                    if meta.split_dispersion:
                        dispCoeffsA = [meta.c2A, meta.c3A]
                        dispCoeffsB = [meta.c2B, meta.c3B]
                        img_disp_comp[0::2] = comp_dis_phase_torch(hamming_signal[0::2], dispMaxOrder, dispCoeffsA, mode=meta.dispersion_mode)
                        img_disp_comp[1::2] = comp_dis_phase_torch(hamming_signal[1::2], dispMaxOrder, dispCoeffsB, mode=meta.dispersion_mode)
                    else:
                        dispCoeffsA = [meta.c2A, meta.c3A]
                        img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffsA, mode=meta.dispersion_mode)

                    # Fourier Transform
                    if meta.split_spectrum:
                        # Split Spectrum Fourier Transform
                        half_point = img_disp_comp.shape[-1] // 2
                        img_disp_comp_split = torch.zeros((img_disp_comp.shape[0]*2, half_point), dtype=img_disp_comp.dtype)

                        img_disp_comp_split[0::4, :] = img_disp_comp[0::2, :half_point]
                        img_disp_comp_split[1::4, :] = img_disp_comp[0::2, half_point:]
                        img_disp_comp_split[3::4, :] = img_disp_comp[1::2, half_point:]
                        img_disp_comp_split[2::4, :] = img_disp_comp[1::2, :half_point]

                        fft_signal = torch.fft.ifft(img_disp_comp_split, dim=-1)

                    else:
                        # Standard Fourier Transform
                        fft_signal = torch.fft.ifft(img_disp_comp, dim=-1)

                    if meta.full_range:
                        temp_frame = torch.abs(fft_signal)  # full range
                    else:
                        temp_frame = torch.abs(fft_signal[:, int(fft_signal.shape[1] / 2):])  # take the negative part

                    if meta.log_scale:
                        temp_frame = 20 * torch.log10(temp_frame + 1e-6)  # Add a small value to avoid log(0)

                    oct_vol_array_hires[frame_counter_hires] = temp_frame

                    frame_counter_hires += 1

                frame_counter += hires_d*hires_ratio # 6*3 = 18

            else:

                # #if next frame is a pause frame or after a pause frame
                if (frame_counter+1 in pause_index) or (frame_counter in stop_index):
                    # print(f"Processing low-res pre frame at index {frame_counter_lowres} for pause index {frame_counter+1}")

                    if meta.packed:
                        raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype="<u1")
                        if raw_data.size != data_size_bytes:
                            continue
                        raw_data = torch.tensor(raw_data).to(device)
                        raw = unpack12_torch(raw_data)
                        raw = raw.reshape((meta.height, meta.width))
                    else:
                        raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype=np.uint16)
                        if raw_data.size != meta.height * meta.width:
                            continue
                        raw = raw_data.reshape((meta.height, meta.width)).astype(np.float32)
                        raw = torch.tensor(raw).to(device)

                    if meta.dcSubtract:
                    # Subtract the DC signal
                        subtracted_signal = dc_subtraction_double_sweep_torch(raw)
                    else:
                        subtracted_signal = raw

                    # Hamming windowing
                    hamming_signal = subtracted_signal * hamming

                    img_disp_comp = torch.zeros_like(hamming_signal, dtype=torch.complex64, device=device)
                    
                    if meta.split_dispersion:
                        dispCoeffsA = [meta.c2A, meta.c3A]
                        dispCoeffsB = [meta.c2B, meta.c3B]
                        img_disp_comp[0::2] = comp_dis_phase_torch(hamming_signal[0::2], dispMaxOrder, dispCoeffsA, mode=meta.dispersion_mode)
                        img_disp_comp[1::2] = comp_dis_phase_torch(hamming_signal[1::2], dispMaxOrder, dispCoeffsB, mode=meta.dispersion_mode)
                    else:
                        dispCoeffsA = [meta.c2A, meta.c3A]
                        img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffsA, mode=meta.dispersion_mode)

                    # Fourier Transform
                    if meta.split_spectrum:
                        # Split Spectrum Fourier Transform
                        half_point = img_disp_comp.shape[-1] // 2
                        img_disp_comp_split = torch.zeros((img_disp_comp.shape[0]*2, half_point), dtype=img_disp_comp.dtype)

                        img_disp_comp_split[0::4, :] = img_disp_comp[0::2, :half_point]
                        img_disp_comp_split[1::4, :] = img_disp_comp[0::2, half_point:]

                        img_disp_comp_split[3::4, :] = img_disp_comp[1::2, half_point:]
                        img_disp_comp_split[2::4, :] = img_disp_comp[1::2, :half_point]

                        fft_signal = torch.fft.ifft(img_disp_comp_split, dim=-1)

                    else:
                        # Standard Fourier Transform
                        fft_signal = torch.fft.ifft(img_disp_comp, dim=-1)

                    if meta.full_range:
                        temp_frame = torch.abs(fft_signal) #full range
                    else:
                        temp_frame = torch.abs(fft_signal[:, int(fft_signal.shape[1] / 2):]) #take the negative part

                    if meta.log_scale:
                        temp_frame = 20 * torch.log10(temp_frame + 1e-6)  # Add a small value to avoid log(0)

                    oct_vol_array[frame_counter_lowres] = temp_frame

                else:
                    byte_reader.seek(data_size_bytes, 1) #skip this frame

                frame_counter_lowres += 1
                frame_counter += 1


    #add delay to the high-res frames    
    for i in range(len(pause_index)):
        idx1 = i*hires_d
        idx2 = idx1 + hires_d
        # take a cloned block of 6 high-resolution b-scans and flatten (concatenate) along the first axis
        hires_block = oct_vol_array_hires[idx1:idx2].clone()
        hires_bscan = hires_block.reshape(-1, hires_block.shape[2])

        # roll and reshape back to (hires_d, hires_h, width) using torch
        hires_bscan = torch.roll(hires_bscan, shifts=(delay, 0), dims=(0, 1))
        hires_bscan = hires_bscan.reshape((hires_d, hires_h, hires_block.shape[2]))
        oct_vol_array_hires[idx1:idx2] = hires_bscan

    #double side the low-res volume
    if meta.double_side:
        # reverse the height axis for every odd B-scan (works for torch.Tensor)
        oct_vol_array[1::2, :, :] = torch.flip(oct_vol_array[1::2, :, :], dims=[1])
        oct_vol_array_hires[1::2, :, :] = torch.flip(oct_vol_array_hires[1::2, :, :], dims=[1])

    oct_vol_array = oct_vol_array.permute(0,2,1)
    oct_vol_array_hires = oct_vol_array_hires.permute(0,2,1)
    oct_vol_array_lowres_pre = oct_vol_array_lowres_pre.permute(0,2,1)
    oct_vol_array_lowres_post = oct_vol_array_lowres_post.permute(0,2,1)

    if meta.desine:
        oct_vol_array = desine(oct_vol_array, mode="bilinear", transpose=False, scale_fac=2)
        oct_vol_array_hires = desine(oct_vol_array_hires, mode="bilinear", transpose=False, scale_fac=2)

    if include_hires_in_lowres:
        target_size = oct_vol_array[0].shape
        for i in range(len(pause_index)):
            idx = pause_index[i] - i*hires_d*hires_ratio + i
            temp_frame = oct_vol_array_hires[i*hires_d].unsqueeze(0)
            temp_frame = F.interpolate(temp_frame.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            oct_vol_array = torch.cat((oct_vol_array[:idx], temp_frame, oct_vol_array[idx:]), dim=0)
   
    for idx, pause_idx in enumerate(pause_index):
        insert_idx = pause_idx - idx*hires_d*hires_ratio - 1
        for i in range(hires_d):
            oct_vol_array_lowres_pre[idx*hires_d + i] = oct_vol_array[insert_idx]
            oct_vol_array_lowres_post[idx*hires_d + i] = oct_vol_array[insert_idx+1]

    oct_vol_array = oct_vol_array.cpu().numpy() #this is the main low-res volume
    oct_vol_array_hires = oct_vol_array_hires.cpu().numpy() #this is the high-res volume

    #this is just for haoshen, he wants to see the low-res pre and post frames for each pause frame, so we will return them as well
    oct_vol_array_lowres_pre = oct_vol_array_lowres_pre.cpu().numpy()
    oct_vol_array_lowres_post = oct_vol_array_lowres_post.cpu().numpy()

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("Finished unp file processing.")
    
    return oct_vol_array_lowres_pre, oct_vol_array_lowres_post, oct_vol_array_hires



from skimage.registration import optical_flow_ilk, phase_cross_correlation
from skimage.transform import warp
from scipy.ndimage import shift as ndimage_shift

def optical_flow_registration(img: np.ndarray, img2: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray) -> np.ndarray:
    """Computes optical flow and warps img2 to match img."""
    # row, col = img.shape  # Fixed: extracted variables correctly from shape
    
    # Compute optical flow
    v, u = optical_flow_ilk(img, img2) #ilk is better than tvl1
    
    # Register coordinates
    # row_coords, col_coords = np.meshgrid(np.arange(row), np.arange(col), indexing="ij")
    flow_warp = warp(img2, np.array([row_coords + v, col_coords + u]), mode="edge")
    
    return flow_warp


def phase_correlation_registration(ref: np.ndarray, moving: np.ndarray, row_coords: np.ndarray, col_coords: np.ndarray) -> np.ndarray:

    #Calculate the translation using phase cross-correlation
    detected_shift, _, _ = phase_cross_correlation(
        ref, 
        moving
    )

    corrected_image = ndimage_shift(moving, detected_shift, mode='constant', cval=0)

    return corrected_image




# def optical_flow_registration(active: np.ndarray) -> np.ndarray:
#     """Registers a sequence of images to the first frame and returns the average."""
#     if active.ndim != 3:
#         raise ValueError("active must have shape (frames, height, width)")

#     registered = np.empty_like(active)
#     registered[0] = active[0].copy()  # Keep the reference frame intact

#     total_frames = active.shape[0] - 1
#     if total_frames <= 0:
#         return registered[0].copy()

#     reference = active[0]

#     references = np.expand_dims(reference, axis=0)
#     references = np.repeat(references, total_frames, axis=0)

#     #prepare the mashgrid for the optical flow registration
#     row, col = reference.shape
#     row_coord, col_coord = np.meshgrid(np.arange(row), np.arange(col), indexing="ij")

#     row_coords = np.expand_dims(row_coord, axis=0)
#     col_coords = np.expand_dims(col_coord, axis=0)

#     row_coords = np.repeat(row_coords, total_frames, axis=0)
#     col_coords = np.repeat(col_coords, total_frames, axis=0)

#     tasks = []
#     for ref, frame, row_coord, col_coord in zip(references, active[1:], row_coords, col_coords):
#         tasks.append((ref, frame, row_coord, col_coord))

#     with Pool(processes=min(8, max(1, (os.cpu_count() or 1)))) as pool:
#         results = pool.starmap(opti_flow_internal, tasks)

#     for i, result in enumerate(results):
#         registered[i + 1] = result
#         print(f"Index: {i} of {total_frames} registered.")

#     print("Optical flow registration complete.")

#     reg_out = registered.mean(axis=0)
#     return reg_out

# def optical_flow_registration_low_hi(low: np.ndarray, high: np.ndarray) -> np.ndarray:
#     '''Registers a sequence of low-resolution images to high-resolution images and returns the registered low-resolution images.'''

#     # both low and high should have shape (frames, height, width)
#     if low.shape != high.shape:
#         raise ValueError(
#             f"low and high must have the same shape (frames, height, width), "
#             f"low shape: {low.shape}, high shape: {high.shape}"
#         )

#     print(f"Registering {low.shape[0]} low-resolution frames to high-resolution frames.")

#     registered = np.empty_like(high)

#     #prepare the mashgrid for the optical flow registration
#     _, row, col = registered.shape
#     row_coord, col_coord = np.meshgrid(np.arange(row), np.arange(col), indexing="ij")

#     row_coords = np.expand_dims(row_coord, axis=0)
#     col_coords = np.expand_dims(col_coord, axis=0)

#     row_coords = np.repeat(row_coords, registered.shape[0], axis=0)
#     col_coords = np.repeat(col_coords, registered.shape[0], axis=0)

#     tasks = [
#         (ref, frame, row_coord, col_coord)
#         for ref, frame, row_coord, col_coord in zip(low, high, row_coords, col_coords)
#     ]

#     with Pool(processes=4) as pool:
#         results = pool.starmap(opti_flow_internal, tasks)

#     for i, result in enumerate(results):
#         registered[i] = result
#         # print(f"Index: {i} of {len(high)} registered.")

#     print("Optical flow registration complete.")

#     return registered

# import cv2

# def optical_flow_registration_cuda(active: np.ndarray) -> np.ndarray:
#     """Registers a sequence of images to the first frame using OpenCV CUDA 

#     and returns the average image. Input 'active' shape: (Frames, Height, Width).
#     """
#     total_frames = active.shape[0] - 1
#     num_frames, row, col = active.shape

#     # 1. Initialize the GPU Lucas-Kanade Dense solver
#     # gpu_lk = cv2.cuda.FarnebackOpticalFlow.create(
#     #     winSize=(15, 15), maxLevel=3, iters=5
#     # )

#     gpu_lk = cv2.cuda.FarnebackOpticalFlow.create(
#         numLevels=3, 
#         pyrScale=0.5, 
#         fastPyramids=False, 
#         winSize=13, 
#         numIters=10, 
#         polyN=5, 
#         polySigma=1.1, 
#         flags=0
#     )

#     # 2. Pre-allocate static GPU memory buffers to avoid memory churn
#     gpu_ref = cv2.cuda_GpuMat()
#     gpu_mov = cv2.cuda_GpuMat()
#     gpu_flow = cv2.cuda_GpuMat()
#     gpu_warped = cv2.cuda_GpuMat()

#     # Create the baseline coordinate meshgrid once on the CPU
#     grid_x, grid_y = np.meshgrid(np.arange(col, dtype=np.float32), np.arange(row, dtype=np.float32))
    
#     # Pre-allocate GPU mats for the remap coordinate lookup tables
#     gpu_map_x = cv2.cuda_GpuMat()
#     gpu_map_y = cv2.cuda_GpuMat()

#     # 3. Upload the base reference frame (index 0) to the GPU
#     # OpenCV algorithms expect data of type float32 or uint8
#     ref_frame_f32 = active[0].astype(np.float32)
#     gpu_ref.upload(ref_frame_f32)

#     # Initialize the output array on the CPU
#     registered = np.empty_like(active)
#     registered[0] = active[0].copy()

#     # 4. Process the remaining sequence in the loop
#     for i in range(1, num_frames):
#         mov_frame_f32 = active[i].astype(np.float32)
#         gpu_mov.upload(mov_frame_f32)

#         # Compute optical flow on the GPU (returns a 2-channel matrix of displacement vectors)
#         gpu_lk.calc(gpu_ref, gpu_mov, gpu_flow)

#         print(gpu_flow.size(), gpu_flow.type())  # Debug: Check the size and type of the flow matrix

#         # Split the 2-channel flow vector matrix into separate X (u) and Y (v) components
#         # gpu_flow_x, gpu_flow_y = cv2.cuda.split(gpu_flow)

#         gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
#         gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)

#         cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

#         # Update lookup tables: New coordinate = Base Meshgrid + Displacement Flow Vector
#         # Note: OpenCV maps X-flow (u) to columns and Y-flow (v) to rows
#         cv2.cuda.add(gpu_flow_x, grid_x, gpu_map_x)
#         cv2.cuda.add(gpu_flow_y, grid_y, gpu_map_y)

#         # Warp the image directly on the GPU using remap
#         # BORDER_REPLICATE acts exactly like skimage's mode="edge"
#         cv2.cuda.remap(
#             gpu_mov, 
#             gpu_map_x, 
#             gpu_map_y, 
#             interpolation=cv2.INTER_LINEAR, 
#             borderMode=cv2.BORDER_REPLICATE, 
#             dst=gpu_warped
#         )

#         # Download the final registered frame back to the CPU
#         registered[i] = gpu_warped.download()
#         print(f"Index: {i} of {total_frames} registered.")

#     # Compute the average over the sequence
#     reg_out = registered.mean(axis=0)
#     return reg_out
