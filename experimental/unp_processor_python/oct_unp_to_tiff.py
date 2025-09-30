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

def dc_subtraction_double_sweep(data: np.ndarray) -> np.ndarray:
    """
    Perform DC subtraction for double-sweep source signals.

    This function removes the DC component from spectral interferograms 
    acquired using a double-sweep source. The number of A-scans per B-scan 
    must be even; otherwise, an error will be raised.

    Parameters
    ----------
    data : np.ndarray
        Input spectral data of shape (num_ascans, num_points), e.g. (800, 2016).

    Returns
    -------
    np.ndarray
        DC-subtracted signal with the same shape as the input.
    """
    num_ascans, num_points = data.shape
    if num_ascans % 2 != 0:
        raise ValueError("Number of A-scans must be even for double-sweep subtraction.")

    # Split into forward (even index) and backward (odd index, reversed) sweeps
    forward = data[::2, :]
    backward = data[1::2, :][:, ::-1]

    # Remove DC component by subtracting the median spectrum
    forward = forward - np.median(forward, axis=0, keepdims=True)
    backward = backward - np.median(backward, axis=0, keepdims=True)

    # Recombine into the full B-scan
    subtracted_signal = np.empty_like(data)
    subtracted_signal[::2, :] = forward
    subtracted_signal[1::2, :] = backward

    return subtracted_signal


def set_displacement_coefficients(data: np.ndarray, maxDispOrders, coefRange) -> np.ndarray:
    """"""
    arrCountDispCoeff = np.zeros((maxDispOrders - 1, 1))

    for idx_CounterDispCoef in tqdm(
        range(0, len(arrCountDispCoeff)), desc="Calculating Displacement Coefficients"
    ):
        arrDispCoeffRange = np.arange(-1 * coefRange, coefRange + 1, 1)
        arrCost = np.zeros((arrDispCoeffRange.shape[0]))

        for k in tqdm(range(0, len(arrDispCoeffRange)), desc="Calculating Costs"):
            arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[k]
            arrCost[k] = cal_cost_function(data, maxDispOrders, arrCountDispCoeff)

        argMinCost = arrCost.argmax()
        arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost]

    return arrCountDispCoeff


def cal_cost_function(data: np.ndarray, maxDispOrders, arrCountDispCoeff):
    """"""
    data_disp_comp = comp_dis_phase(data, maxDispOrders, arrCountDispCoeff)

    oct = np.abs(np.fft.fft(data_disp_comp)) ** 2
    roi_oct = oct[:, 49 : int(data.shape[1] / 2) - 50]  # avoid edges
    norm_oct = roi_oct / np.sum(roi_oct[:])
    entropy = norm_oct * np.log10(norm_oct)
    cost = np.sum(entropy[:])  # %shenon entropy

    return cost


def comp_dis_phase(data: np.ndarray, maxDsipOrders, arrCountDispCoeff):
    """"""
    # shape [numAscan numPts] [800 2016]
    scan_pts = data.shape[1]
    line_per_frame = data.shape[0]
    k_linear = np.linspace(-1, 1, scan_pts)
    k_axis = np.tile(k_linear, (line_per_frame, 1)) - 1

    amp = np.abs(data)
    phase = np.angle(data)

    for i in range(maxDsipOrders - 1):
        phase = phase + arrCountDispCoeff[i] * (k_axis ** (i + 2))

    data_disp_comp = amp * np.exp(1j * phase)

    return data_disp_comp


def process_unp(unp_file_path:Path):

    # Read the xml file
    meta = prof_proc_meta(Path(unp_file_path), ".unp")

    print(meta)

    h, w, d, bmscan, w_param, dtype, layer_type = meta
    
    # read 2 bytes size for uint16
    data_size_bytes = 2 * w_param * h

    oct_vol_array = []

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

        # Subtract the DC signal
        subtracted_signal = dc_subtraction_double_sweep(array)

        # Hamming windowing
        hamming = np.tile(
            np.hamming(subtracted_signal.shape[1]), (subtracted_signal.shape[0], 1)
        )

        hamming_signal = subtracted_signal * hamming

        dispMaxOrder = 3
        coeffRange = 100
        dispCoeffs = set_displacement_coefficients(hamming_signal,dispMaxOrder,coeffRange)
        # dispCoeffs = np.array([0, 0])  # PUT THIS BACK AFTER TESTING!!!!

        # Main OCT Volume process
        for frame_num in tqdm(range(0, d), desc="Processing Bscans"):

            byte_reader.seek(data_size_bytes * (frame_num), 0)
            raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype=np.uint16)
            raw = raw_data.reshape((h, w_param)).astype(np.float64)
            
            # Subtract the DC signal
            subtracted_signal = dc_subtraction_double_sweep(raw)

            # Hamming windowing
            hamming_signal = subtracted_signal * hamming

            # Dispersion Correction
            img_disp_comp = comp_dis_phase(hamming_signal, dispMaxOrder, dispCoeffs)

            # Fourier Transform
            fft_signal = np.fft.fft(img_disp_comp)
            
            # double sided fast axis scans
            temp_frame = np.abs(fft_signal[:, : int(fft_signal.shape[1] / 2)])
            if np.floor(frame_num / bmscan) % 2:
                temp_frame = temp_frame[::-1, :]#horizontal flip
            else:
                pass

            # (optional) flip the image
            temp_frame = temp_frame[:, ::-1]#vertical flip

            # store image to oct array
            oct_vol_array.append(temp_frame)

    oct_vol_array = np.stack(oct_vol_array, axis=0)
    
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
    
    fname = os.path.basename(path)[:-5] + "_processed_python.prof"
    processed_vol.tofile(fname)
    
    # fname = os.path.basename(path)[:-5] + "_processed.tif"
    # fname = os.path.join(".", fname)
    # tiff.imwrite(fname,np.abs(processed_vol))
    
    
