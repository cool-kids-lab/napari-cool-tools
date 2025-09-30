
import math
import pathlib
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
import napari
import numpy as np
from _prof_reader import prof_proc_meta
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile as tiff  # We will use tifffile to open TIFF images
import os
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

def torch_like_numpy_median(x: torch.Tensor, dim=None, keepdim=False):
    """
    Compute the median in PyTorch with NumPy's definition:
    - For odd n: middle element
    - For even n: average of the two middle elements

    Args:
        x (torch.Tensor): input tensor
        dim (int, optional): dimension along which to compute the median.
                             If None, flattens the tensor.
        keepdim (bool): whether the output has dim retained.

    Returns:
        torch.Tensor: median values (float if averaging needed).
    """
    if dim is None:
        x = x.flatten()
        dim = 0

    # Sort along the dimension
    x_sorted, _ = torch.sort(x, dim=dim)
    n = x.shape[dim]
    mid = n // 2

    if n % 2 == 1:  # odd length
        result = x_sorted.select(dim, mid)
    else:  # even length → average the two middle values
        left = x_sorted.select(dim, mid - 1)
        right = x_sorted.select(dim, mid)
        result = (left.to(torch.float64) + right.to(torch.float64)) / 2.0

    if keepdim:
        result = result.unsqueeze(dim)

    return result

def dc_subtraction_double_sweep_torch(data: torch.Tensor) -> torch.Tensor:
    """
    Remove the DC signal (DC subtraction) for double-sweep source signal.
    The number of A-scans per B-scan must be even, otherwise this function will raise an error.

    Args:
        data: Tensor of shape [numAscans, numPts], e.g. [800, 2016].

    Returns:
        subtracted_signal: Tensor of the same shape as input with DC component removed.
    """
    if data.shape[0] % 2 != 0:
        raise ValueError("Number of A-scans must be even for double-sweep subtraction")

    # Split into even (forward) and odd (reverse) A-scans
    corrected_1 = data[::2, :]  # every even A-scan
    # corrected_2 = data[1::2, :]                  # every odd A-scan
    corrected_2 = torch.flip(data[1::2, :], [1]) # every odd A-scan, reversed along spectral axis
    #this is to keep the phase in the same direction, so to make the dipersion correction monotonic
    
    # Remove DC component by subtracting the median spectrum
    # Subtract median (DC removal) along each spectrum (per column)
    corrected_1 = corrected_1 - torch_like_numpy_median(corrected_1, dim=0, keepdim=True)
    corrected_2 = corrected_2 - torch_like_numpy_median(corrected_2, dim=0, keepdim=True)

    # Recombine into full B-scan
    subtracted_signal = torch.zeros_like(data)
    subtracted_signal[::2, :] = corrected_1
    subtracted_signal[1::2, :] = corrected_2

    return subtracted_signal

def set_displacement_coefficients_torch(data: torch.Tensor, maxDispOrders, coefRange) -> torch.Tensor:
    """"""
    arrCountDispCoeff = torch.zeros((maxDispOrders - 1, 1),device=data.device, dtype=torch.float64)

    for idx_CounterDispCoef in tqdm(
        range(0, len(arrCountDispCoeff)), desc="Calculating Displacement Coefficients"
    ):
        arrDispCoeffRange = np.arange(-1 * coefRange, coefRange + 1, 1)
        arrCost = np.zeros((arrDispCoeffRange.shape[0]))

        for k in tqdm(range(0, len(arrDispCoeffRange)), desc="Calculating Costs"):
            arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[k]
            arrCost[k] = cal_cost_function_torch(data, maxDispOrders, arrCountDispCoeff)

        argMinCost = arrCost.argmax()
        arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost]

    return arrCountDispCoeff


def cal_cost_function_torch(data: torch.Tensor, maxDispOrders, arrCountDispCoeff: torch.Tensor):
    """"""
    data_disp_comp = comp_dis_phase_torch(data, maxDispOrders, arrCountDispCoeff)

    # FFT magnitude squared
    oct_img = torch.abs(torch.fft.fft(data_disp_comp, dim=-1)) ** 2
    
    # Avoid edges
    roi_oct = oct_img[:, 49 : int(data_disp_comp.shape[1] / 2) - 50]
    
    # Normalize
    norm_oct = roi_oct / torch.sum(roi_oct)
    
    # Shannon entropy
    eps = 1e-12#this is to avoid nan
    entropy = norm_oct * torch.log10(norm_oct + eps)
    
    # Final cost
    cost = torch.sum(entropy)
    return cost

def comp_dis_phase_torch(data: torch.Tensor, max_disp_orders, arrCountDispCoeff: torch.Tensor) -> torch.Tensor:

    # Amplitude/phase
    # data_c = data.to(torch.complex64)                    # ensure complex128
    amp   = torch.abs(data)
    phase = torch.angle(data)

    line_per_frame, scan_pts = data.shape

    # k-axis (broadcasted across lines)
    k_linear = torch.linspace(-1.0, 1.0, scan_pts, device=data.device, dtype=data.dtype)
    k_axis   = k_linear.unsqueeze(0).expand(line_per_frame, -1) - 1.0

    # Apply dispersion phase terms: i from 0..max_disp_orders-2 -> power i+2
    # (matches your NumPy loop)
    n_terms = max(0, max_disp_orders - 1)
    for i in range(n_terms):
        phase = phase + arrCountDispCoeff[i] * k_axis.pow(i + 2)

    # Recombine amplitude and phase: amp * exp(1j*phase)
    data_disp_comp = amp * torch.exp(1j*phase)
    return data_disp_comp


def reflect_shift2d(img: torch.Tensor, shift: tuple[int,int]) -> torch.Tensor:
    """
    Reflecting (mirror) shift for 2D image.

    img: (H, W) tensor
    shift: (dy, dx), must satisfy |dy| < H and |dx| < W
    """
    H, W = img.shape
    dy, dx = shift
    py, px = abs(dy), abs(dx)

    if py >= H or px >= W:
        raise ValueError(f"|dy| must be < H and |dx| < W (got {dy=}, {dx=}).")

    # pad with reflection
    img_pad = F.pad(img.unsqueeze(0).unsqueeze(0), (px, px, py, py), mode="reflect").squeeze()

    # extract shifted window
    y0, y1 = py - dy, py - dy + H
    x0, x1 = px - dx, px - dx + W

    return img_pad[y0:y1, x0:x1]


def shift2d_zeropad(img: np.ndarray, shift: tuple[int, int], fill_value=0.0) -> np.ndarray:

    H, W = img.shape
    dy, dx = shift

    # pad with zeros (or fill_value)
    pad_y, pad_x = abs(dy), abs(dx)
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="constant", constant_values=fill_value)

    # slicing window after shift
    y0 = pad_y - dy
    x0 = pad_x - dx
    return padded[y0:y0+H, x0:x0+W]


def process_unp(unp_file_path:Path):
    
    # pause_index = [236, 267 , 298 , 329, 358] #hires_h = 1200 hires_d = 10 delay = 55
    # pause_index = [236, 265 , 296 , 327, 358]
    #even, odd, even, odd, even
    pause_index = [234, 265 , 294 , 325, 356]#this will make flipping problem

    # Read the xml file
    meta = prof_proc_meta(Path(unp_file_path), ".unp")

    print(meta)

    h, w, d, bmscan, w_param, dtype, layer_type = meta
    
    reference_frame = 220 #math.ceil(d / 2)
    
    # read 2 bytes size for uint16
    data_size_bytes = int(2 * w_param * h)#packed=false
    
    hires_h = 1800
    hires_d = 6
    
    subdelay = -1
    
    #28 is from the ini file
    #it is 28*2 not 28*3 because the first 28 is already taken by the low res signal
    #times two, because delay should be even number, because of forward anc backward Ascan
    #times two again because it is forward and backward BScan
    #subdelay also multiplied by two beacuse of the dual ascan
    delay = (28*2)*2 + subdelay*2
    
    d = d - int(hires_h/h)*hires_d*5 #len(pause_index)=5
    
    hires_data_size_bytes = int(2 * w_param *  hires_h)#packed=false
    
    # open file
    with open(unp_file_path, "rb", buffering=0) as byte_reader:
        # Set reference A-scan to find the dispersion coefficients
        # Use center frame (b-scan) of the volume
        
        print(f"Reference frame: {reference_frame}\n")
        # move to center frame in binary file
        
        byte_reader.seek(data_size_bytes * (reference_frame - 1), 0)
       
        ref_RawData = byte_reader.read(data_size_bytes)
        array = np.frombuffer(ref_RawData, dtype='<u2')
        array = array.reshape((h, w_param)).astype(np.float64)
        
        array = torch.tensor(array).cuda()

        # Subtract the DC signal
        subtracted_signal = dc_subtraction_double_sweep_torch(array)
        
        # 1D Hamming window (like np.hamming)
        hamming = torch.hamming_window(w_param, periodic=False, dtype=torch.float64, device="cuda")
        hamming = hamming.unsqueeze(0).repeat(h, 1)
        
        hamming_double = torch.hamming_window(w_param, periodic=False, dtype=torch.float64, device="cuda")
        hamming_double = hamming_double.unsqueeze(0).repeat(hires_h, 1)
        hamming_signal = subtracted_signal * hamming

        dispMaxOrder = 3
        coeffRange = 100
        dispCoeffs = set_displacement_coefficients_torch(hamming_signal,dispMaxOrder,coeffRange)
        
        # dispCoeffs1 = set_displacement_coefficients_torch(hamming_signal[:,:],dispMaxOrder,coeffRange)
        
        print(f'Dispersion Corrections {dispCoeffs}')
        
        dispCoeffs = np.array([0, 0])  # PUT THIS BACK AFTER TESTING!!!!
        
        oct_vol_array = []
        hires_oct_vol_array = []
        
        seek_offset = 0
        
        frame_counter = 0
        
        # Main OCT Volume process
        for frame_num in tqdm(range(0, d), desc="Processing Bscans"):
                        
            if frame_num in pause_index:
                
                for hires_frame_num in range(0,hires_d):
                    #use it first and then update
                    byte_reader.seek(seek_offset, 0)
                    seek_offset = seek_offset + hires_data_size_bytes
                    
                    raw_data = np.frombuffer(byte_reader.read(hires_data_size_bytes), dtype='<u2')
                    raw = raw_data.reshape((hires_h, w_param)).astype(np.float64)
                    raw = torch.tensor(raw).cuda()
                    
                    # Subtract the DC signal
                    subtracted_signal = dc_subtraction_double_sweep_torch(raw)
                    
                    # subtracted_signal = subtracted_signal[:,1024:3072]
        
                    # Hamming windowing
                    hamming_signal = subtracted_signal * hamming_double
        
                    # Dispersion Correction
                    img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffs)
        
                    # Fourier Transform
                    # fft_signal = np.fft.fft(img_disp_comp)
                    fft_signal = torch.fft.fft(img_disp_comp, dim=-1)
                    
                    # double sided fast axis scans
                    temp_frame = torch.abs(fft_signal[:, : int(fft_signal.shape[1] / 2)])

                    # (optional) flip the image
                    temp_frame = torch.flip(temp_frame, [1])#horizontal flip
                    
                    #save it for the hires image
                    hires_oct_vol_array.append(temp_frame.cpu().numpy())
                    
                    #circular shift
                    temp_frame = torch.roll(temp_frame, shifts=(delay, 0), dims=(0, 1))
                    
                    if hires_frame_num % 2:
                        temp_frame = torch.flip(temp_frame, [0])#horizontal flip  
                    
                    if hires_frame_num == 0:
                    
                        # resize image
                        temp_frame = temp_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, 1024, 1200)
                        resized = F.interpolate(temp_frame, size=(h, temp_frame.shape[-1]), mode="bilinear", align_corners=False)
                        resized = resized.squeeze(0).squeeze(0)
                        
                        oct_vol_array.append(resized.cpu().numpy()) 
                        
                        frame_counter = frame_counter + 2
                
            else:
                byte_reader.seek(seek_offset, 0)
                seek_offset = seek_offset + data_size_bytes
            
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype='<u2')                
                raw = raw_data.reshape((h, w_param)).astype(np.float64)
                raw = torch.tensor(raw).cuda()
                
                # Subtract the DC signal
                subtracted_signal = dc_subtraction_double_sweep_torch(raw)
    
                # Hamming windowing
                hamming_signal = subtracted_signal * hamming
    
                # Dispersion Correction
                img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffs)
    
                # Fourier Transform
                # fft_signal = np.fft.fft(img_disp_comp)
                fft_signal = torch.fft.fft(img_disp_comp, dim=-1)
                
                # double sided fast axis scans
                temp_frame = torch.abs(fft_signal[:, : int(fft_signal.shape[1] / 2)])
                
                # (optional) flip the image
                # temp_frame = temp_frame[:, ::-1]#vertical flip
                temp_frame = torch.flip(temp_frame, [1])#horizontal flip
                                
                if frame_counter % 2:
                    temp_frame = torch.flip(temp_frame, [0])#horizontal flip  
    
                oct_vol_array.append(temp_frame.cpu().numpy())
                
                frame_counter = frame_counter + 1
                
    
    return oct_vol_array, hires_oct_vol_array


def desine(frame: np.ndarray, axis = 0) -> np.ndarray:
    
    if axis == 1:
        frame = frame.transpose(1,0)
    
    h, w = frame.shape
    results = np.zeros((h, w))

    Yn = np.arange(w)  # 0:h-1
    angles = (np.pi / w) * Yn - (np.pi / 2)
    y_org = (w / 2) * np.sin(angles) + (w / 2)

    for i in range(h):
        f = interp1d(y_org, frame[i], kind="linear", bounds_error=False, fill_value=0)
        results[i] = f(Yn)
        
        
    if axis == 1:
        results = results.transpose(1,0)

    return results


app = QApplication([])
parent = QWidget()  # can be your main window instead
start_dir = str(Path.home())

path, _ = QFileDialog.getOpenFileName(
    parent,
    "Open .unp file",
    start_dir,
    "Unprocessed Files (*.unp);"
)

if path:
    print("Selected:", path)
    processed_vol, processed_vol_hires = process_unp(Path(path))
    processed_vol = np.asanyarray(processed_vol)
    processed_vol_hires = np.asanyarray(processed_vol_hires)
    
    fname = path[:-4] + "_processed.tif"
    tiff.imwrite(fname,processed_vol)
        
    #generate enface
    enface = np.max(processed_vol,axis=-1).squeeze()
    # fname = os.path.basename(path)[:-5] + "_processed_enface.tif"
    # fname = os.path.join(".", fname)
    # tiff.imwrite(fname,enface)
    
    #correct ascan correction
    enface = desine(enface)
    fname = path[:-4] + "_processed_enface_ascancorrected_max.tif"
    tiff.imwrite(fname,enface)
    
    #generate enface
    enface = np.mean(processed_vol,axis=-1).squeeze()
    # fname = os.path.basename(path)[:-5] + "_processed_enface.tif"
    # fname = os.path.join(".", fname)
    # tiff.imwrite(fname,enface)
    
    #correct ascan correction
    enface = desine(enface)
    fname = path[:-4] + "_processed_enface_ascancorrected_mean.tif"
    tiff.imwrite(fname,enface)
    
    
    #generate hires
    fname = path[:-4] + "_processed_hires.tif"
    # processed_vol_hires_unwrapped = np.angle(processed_vol_hires)
    # # processed_vol_hires_unwrapped = processed_vol_hires_unwrapped - processed_vol_hires_unwrapped[:,:,0:1]
    
    save_vol_hires = np.zeros_like(processed_vol_hires)
    
    subdelay = -1
    
    #28 is from the ini file
    #it is 28*2 not 28*3 because the first 28 is already taken by the low res signal
    #times two, because delay should be even number, because of forward anc backward Ascan
    #times two again because it is forward and backward BScan
    #subdelay also multiplied by two beacuse of the dual ascan
    delay = (28*2)*2 + subdelay*2
    
    #this is now even delay
    
    for i in range(0,5):
        idx1 = i*6
        idx2 = i*6 + 6
        hires_bscan = np.concatenate(processed_vol_hires[idx1:idx2].copy(), axis=0)

        hires_bscan = np.roll(hires_bscan, shift=(delay, 0), axis=(0, 1))
        hires_bscan = np.reshape(hires_bscan,(6,1800,2048))
        save_vol_hires[idx1:idx2] = hires_bscan
        
    for idx, bscan in enumerate(save_vol_hires):
        if idx % 2 == 1:
            save_vol_hires[idx,:,:] = save_vol_hires[idx,::-1,:]
    
    tiff.imwrite(fname,save_vol_hires)
    
    # #correct ascan correction
    # processed_vol_hires_ascancorrected = np.zeros_like(processed_vol_hires)
    
    # for i in range(0,processed_vol_hires.shape[0]):
    #     in_img = processed_vol_hires[i].T
    #     out_img = desine(in_img).T
    #     processed_vol_hires_ascancorrected[i] = out_img
        
    # fname = os.path.basename(path)[:-5] + "_processed_hires_ascancorrected.tif"
    # fname = os.path.join(".", fname)
    # tiff.imwrite(fname,processed_vol_hires_ascancorrected)
    
    
    #generate averaged hires
    
