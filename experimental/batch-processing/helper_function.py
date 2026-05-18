
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

    # if include_hires_in_lowres:
    #     target_size = oct_vol_array[0].shape
    #     for i in range(len(pause_index)):
    #         idx = pause_index[i] - i*hires_d*hires_ratio + i
    #         temp_frame = oct_vol_array_hires[i*hires_d].unsqueeze(0)
    #         temp_frame = F.interpolate(temp_frame.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    #         oct_vol_array = torch.cat((oct_vol_array[:idx], temp_frame, oct_vol_array[idx:]), dim=0)
   
    for idx, pause_idx in enumerate(pause_index):
        insert_idx = pause_idx - idx*hires_d*hires_ratio - 1
        for i in range(hires_d):
            oct_vol_array_lowres_pre[idx*hires_d + i] = oct_vol_array[insert_idx]
            oct_vol_array_lowres_post[idx*hires_d + i] = oct_vol_array[insert_idx+1]

    oct_vol_array = oct_vol_array.cpu().numpy()
    oct_vol_array_hires = oct_vol_array_hires.cpu().numpy()
    oct_vol_array_lowres_pre = oct_vol_array_lowres_pre.cpu().numpy()
    oct_vol_array_lowres_post = oct_vol_array_lowres_post.cpu().numpy()

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("Finished unp file processing.")
    
    return oct_vol_array_lowres_pre, oct_vol_array_lowres_post, oct_vol_array_hires