""" """

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
    corrected_1 = data[::2, :]                   # every even A-scan
    corrected_2 = torch.flip(data[1::2, :], [1]) # every odd A-scan, reversed along spectral axis
    
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
    oct = torch.abs(torch.fft.fft(data_disp_comp, dim=-1)) ** 2
    
    # Avoid edges
    roi_oct = oct[:, 49 : int(data_disp_comp.shape[1] / 2) - 50]
    
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


def process_unp(unp_file_path:Path):

    # Read the xml file
    meta = prof_proc_meta(Path(unp_file_path), ".unp")

    print(meta)

    h, w, d, bmscan, w_param, dtype, layer_type = meta
    
    # read 2 bytes size for uint16
    data_size_bytes = 2 * w_param * h

    oct_vol_array = torch.zeros((d,h,int(w_param/2)),dtype=torch.float64).cuda()

    # open file
    with open(unp_file_path, "rb", buffering=0) as byte_reader:
        # Set reference A-scan to find the dispersion coefficients
        # Use center frame (b-scan) of the volume
        reference_frame = math.ceil(d / 2)
        print(f"Reference frame: {reference_frame}\n")
        # move to center frame in binary file
        byte_reader.seek(data_size_bytes * (reference_frame - 1), 0)
       
        ref_RawData = byte_reader.read(data_size_bytes)
        array = np.frombuffer(ref_RawData, dtype=np.uint16)
        array = array.reshape((h, w_param)).astype(np.float64)
        array = torch.tensor(array).cuda()

        # Subtract the DC signal
        subtracted_signal = dc_subtraction_double_sweep_torch(array)
        
        # 1D Hamming window (like np.hamming)
        hamming = torch.hamming_window(subtracted_signal.shape[1], periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)
        hamming = hamming.unsqueeze(0).repeat(subtracted_signal.shape[0], 1)
        hamming_signal = subtracted_signal * hamming

        dispMaxOrder = 3
        coeffRange = 100
        dispCoeffs = set_displacement_coefficients_torch(hamming_signal,dispMaxOrder,coeffRange)
        # dispCoeffs = np.array([0, 0])  # PUT THIS BACK AFTER TESTING!!!!
        
        # Main OCT Volume process
        for frame_num in tqdm(range(0, d), desc="Processing Bscans"):

            byte_reader.seek(data_size_bytes * (frame_num), 0)
            raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype=np.uint16)
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
            if np.floor(frame_num / bmscan) % 2:
                # temp_frame = temp_frame[::-1, :]#horizontal flip  
                temp_frame = torch.flip(temp_frame, [0])#horizontal flip  

            # (optional) flip the image
            # temp_frame = temp_frame[:, ::-1]#vertical flip
            temp_frame = torch.flip(temp_frame, [1])#horizontal flip  
            
            oct_vol_array[frame_num] = temp_frame
    
    return oct_vol_array


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
    processed_vol = process_unp(Path(path))
    
    # fname = os.path.basename(path)[:-5] + "_processed_python.prof"
    # processed_vol.cpu().numpy().tofile(fname)
    
    
    
    # fname = os.path.basename(path)[:-5] + "_processed.tif"
    # fname = os.path.join(".", fname)
    # tiff.imwrite(fname,np.abs(processed_vol))
    
    
