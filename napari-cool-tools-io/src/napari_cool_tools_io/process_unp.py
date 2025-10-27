from napari.utils.notifications import show_info
import math
from pathlib import Path
from networkx import display
import numpy as np
from tqdm import tqdm
import torch
from napari_cool_tools_io import unp_meta
import torch.nn.functional as F

def desine_torch(frame: torch.Tensor,mode = "bilinear", transpose: bool = False) -> torch.Tensor:

    if mode == "bilinear":
        return desine_torch_bilinear(frame, transpose=transpose)
    else:
        return desine_torch_nearest(frame, transpose=transpose)

def desine_torch_nearest(frame: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    # Work along the width dimension by transposing to (W, H)
    if transpose:
        frame = torch.transpose(frame, 1, 0)  # shape now (W, H)

    H, W = frame.shape  # Note: H <- original W, W <- original H (naming kept for consistency)
    device = frame.device
    dtype  = frame.dtype

    # Target uniform coordinates (0..W-1)
    Yn = torch.arange(W, device=device, dtype=dtype)

    # Warped source coordinates along width: y_org in [0, W]
    angles = (torch.pi / W) * torch.arange(W, device=device, dtype=dtype) - (torch.pi / 2)
    y_org  = (W / 2) * torch.sin(angles) + (W / 2)  # strictly increasing

    # Find enclosing indices: y_org[idx-1] <= Yn < y_org[idx]
    idx = torch.bucketize(Yn, y_org)  # (W,), in [0..W]

    # Track out-of-bounds (below first or above last), for zero-fill
    oob = (idx == 0) | (idx == W)

    # Clamp to valid interior to form left/right candidates
    idx = idx.clamp(1, W - 1)
    left  = idx - 1
    right = idx

    # Choose nearest between y_org[left] and y_org[right]
    dleft  = (Yn - y_org[left]).abs()
    dright = (Yn - y_org[right]).abs()
    nearest = torch.where(dright < dleft, right, left)  # tie -> left kept

    # Gather nearest columns for all rows (broadcasted column indexing)
    out = frame[:, nearest]  # shape (H, W)

    # Zero fill outside interpolation domain (to match fill_value=0 behavior)
    if oob.any():
        out[:, oob] = 0

    # Restore original (H, W) orientation
    if transpose:
        out = torch.transpose(out, 1, 0)

    return out

def desine_torch_bilinear(frame: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    if transpose:
        frame = torch.transpose(frame, 1, 0)

    H, W = frame.shape
    device = frame.device
    dtype  = frame.dtype

    # Target uniform coordinates (like Yn = np.arange(W))
    Yn = torch.arange(W, device=device, dtype=dtype)

    # Source (warped) coordinates along width: y_org = (W/2) * sin(angles) + (W/2)
    angles = (torch.pi / W) * torch.arange(W, device=device, dtype=dtype) - (torch.pi / 2)
    y_org  = (W / 2) * torch.sin(angles) + (W / 2)  # strictly increasing

    # For each target x=Yn[j], find enclosing interval in y_org
    # bucketize gives index i such that y_org[i-1] <= Yn[j] < y_org[i]
    idx = torch.bucketize(Yn, y_org)  # shape (W,), in [0..W]
    # Identify out-of-range targets (to be filled with 0)
    oob = (idx == 0) | (idx == W)

    # Clamp to valid interior for interpolation
    idx = idx.clamp(1, W - 1)
    x0  = idx - 1
    x1  = idx

    y0 = y_org[x0]  # (W,)
    y1 = y_org[x1]  # (W,)
    denom = (y1 - y0)
    # Safe ratio (linear weight), denom should be > 0 since y_org is strictly increasing
    t = (Yn - y0) / denom

    # Gather source samples for all rows at columns x0, x1
    # frame[:, x0] yields (H, W) by broadcasting column indices across rows
    v0 = frame[:, x0]  # (H, W)
    v1 = frame[:, x1]  # (H, W)

    out = v0 + (v1 - v0) * t  # broadcast t over rows

    # Zero fill outside the interpolation domain (matches fill_value=0)
    if oob.any():
        out[:, oob] = 0

    if transpose:
        out = torch.transpose(out, 1, 0)

    return out

def torch_like_numpy_median(x: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
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
        result = (left + right) / 2.0

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
    corrected_1 = data[0::2, :]                   # every even A-scan
    corrected_2 = torch.flip(data[1::2, :], [1]) # every odd A-scan, reversed along spectral axis
    # corrected_2 = data[1::2, :] # every odd A-scan, reversed along spectral axis
    
    # Remove DC component by subtracting the median spectrum
    # Subtract median (DC removal) along each spectrum (per column)
    corrected_1 = corrected_1 - torch_like_numpy_median(corrected_1, dim=0, keepdim=True)
    corrected_2 = corrected_2 - torch_like_numpy_median(corrected_2, dim=0, keepdim=True)

    # Recombine into full B-scan
    subtracted_signal = torch.zeros_like(data)
    subtracted_signal[0::2, :] = corrected_1
    subtracted_signal[1::2, :] = corrected_2

    return subtracted_signal


def set_dispersion_coefficients_torch(data: torch.Tensor, maxDispOrders, coefRange) -> torch.Tensor:
    """
    Determine per-order dispersion coefficients by evaluating a cost function over an integer range.
    This function searches, for each dispersion order (from 1 to maxDispOrders-1), an integer coefficient
    within the closed interval [-coefRange, coefRange] that optimizes a provided cost function
    (cal_cost_function_torch). The search is performed by brute force: for each candidate coefficient
    value the cost is evaluated and the candidate that yields the best (largest) cost is selected and
    stored in the output coefficient array.
    Parameters
    ----------
    data : torch.Tensor
        Input tensor that is passed to the cost function. The returned coefficient tensor will use the
        same device and dtype as this tensor.
    maxDispOrders : int
        Number of dispersion orders to consider. The function returns coefficients for orders
        1 .. (maxDispOrders - 1). Must be an integer greater than 1.
    coefRange : int
        Non-negative integer specifying the search range for each coefficient. Candidate coefficients
        are the integers in [-coefRange, ..., coefRange].
    Returns
    -------
    torch.Tensor
        A tensor of shape (maxDispOrders - 1, 1) containing the selected integer coefficients for each
        dispersion order. The tensor uses the same device and dtype as `data`.
    Notes
    -----
    - The function initializes the coefficient array with zeros on the same device/dtype as `data`.
    - For each dispersion order index i, it constructs the candidate array of integers
      np.arange(-coefRange, coefRange + 1) and evaluates the cost for each candidate by calling
      cal_cost_function_torch(data, maxDispOrders, arrCountDispCoeff) after temporarily assigning
      the candidate to the i-th position.
    - The implementation selects the candidate that maximizes the returned cost (the code uses
      arrCost.argmax()). Variable names in the implementation (e.g., argMinCost) may suggest a
      minimization but the actual selection uses the maximum cost.
    - The function uses tqdm to present progress bars for the outer and inner loops.
    - cal_cost_function_torch is expected to accept (data, maxDispOrders, arrCountDispCoeff) and to
      return a scalar (or scalar-like) value for each candidate; the code converts those values into a
      NumPy array of costs for argmax selection.
    Raises
    ------
    ValueError
        If maxDispOrders <= 1 or if coefRange < 0 (caller should ensure valid inputs).
    Performance
    -----------
    - Time complexity is O((maxDispOrders-1) * (2*coefRange+1) * C) where C is the cost of a single
      cal_cost_function_torch evaluation. This is a brute-force search and may be slow for large
      maxDispOrders or large coefRange.
    Example
    -------
    # Example usage (assuming cal_cost_function_torch is defined and torch imported):
    # coeffs = set_dispersion_coefficients_torch(data_tensor, maxDispOrders=5, coefRange=3)
    """
    """"""
    arrCountDispCoeff = torch.zeros((maxDispOrders - 1, 1),device=data.device, dtype=data.dtype)

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


def cal_cost_function_torch(data: torch.Tensor, maxDispOrders, arrCountDispCoeff: torch.Tensor) -> torch.Tensor:
    """
    Compute an entropy-based cost for OCT data after dispersion phase compensation.
    This function:
    - Applies dispersion compensation via `comp_dis_phase_torch`.
    - Computes the magnitude-squared FFT along the last dimension.
    - Selects a region of interest (negative-frequency half, excluding 50 edge samples).
    - Normalizes the ROI to a probability distribution.
    - Returns the base-10 Shannon entropy sum (sum_i p_i * log10(p_i)) as the cost.
    Parameters
    ----------
    data : torch.Tensor
        Input interferometric data of shape (N, L), where FFT is computed along the
        last dimension (L). Typically complex-valued after dispersion compensation.
    maxDispOrders
        Dispersion model order(s) forwarded to `comp_dis_phase_torch`.
    arrCountDispCoeff : torch.Tensor
        Dispersion coefficient tensor consumed by `comp_dis_phase_torch`.
    Returns
    -------
    torch.Tensor
        A scalar tensor containing the entropy cost. Lower (more negative) values
        generally correspond to sharper spectra.
    Notes
    -----
    - ROI selection uses the negative-frequency half with edges excluded:
      indices [L/2 + 50 : L - 50]. This assumes `data` is 2D (N, L).
    - A small epsilon (1e-12) is added inside the logarithm to avoid NaNs.
    - The normalization divides by the sum over the ROI; ensure it is nonzero.
    - The entropy is computed with log base 10 and without a leading minus sign.
      For conventional Shannon entropy, negate the result.
    - Operations are differentiable and suitable for gradient-based optimization,
      assuming `comp_dis_phase_torch` is differentiable.
    """
    """"""
    data_disp_comp = comp_dis_phase_torch(data, maxDispOrders, arrCountDispCoeff)

    # FFT magnitude squared
    toct = torch.abs(torch.fft.fft(data_disp_comp, dim=-1)) ** 2
    
    # Avoid edges
    # roi_oct = toct[:, 49 : int(data_disp_comp.shape[1] / 2) - 50]#this is the positive half
    roi_oct = toct[:, int(data_disp_comp.shape[1] / 2) + 50 : -50] #take the negative part
    
    # Normalize
    norm_oct = roi_oct / torch.sum(roi_oct)
    
    # Shannon entropy
    eps = 1e-12#this is to avoid nan
    entropy = norm_oct * torch.log10(norm_oct + eps)
    
    # Final cost
    cost = torch.sum(entropy)
    return cost

def comp_dis_phase_torch(data: torch.Tensor, max_disp_orders, arrCountDispCoeff: torch.Tensor) -> torch.Tensor:
    """
    Dispersion-phase compensation for complex OCT data using PyTorch.
    This function decomposes a complex-valued signal into amplitude and phase, adds a
    polynomial dispersion phase term across a normalized k-axis, and recombines the
    amplitude with the corrected phase. The dispersion phase is modeled as:
        phase += sum_{i=0}^{n_terms-1} coeff[i] * k^(i+2)
    i.e., powers k^2, k^3, ..., k^{max_disp_orders}, where n_terms = max(0, max_disp_orders - 1).
    Parameters
    ----------
    data : torch.Tensor
        Complex-valued tensor of shape (line_per_frame, scan_pts). Must be a complex dtype
        (e.g., torch.complex64 or torch.complex128). The device and dtype drive internal computations.
    max_disp_orders : int
        Maximum polynomial order of the dispersion phase to apply. If <= 1, no dispersion phase is added.
    arrCountDispCoeff : torch.Tensor
        1D tensor of length >= max(0, max_disp_orders - 1) containing real-valued dispersion coefficients
        [c2, c3, ..., c_{max_disp_orders}]. Should reside on the same device as `data` and use a real dtype
        compatible with `data`'s real component.
    Returns
    -------
    torch.Tensor
        Complex-valued tensor of the same shape and dtype as `data`, with dispersion compensation applied.
    Notes
    -----
    - The k-axis is constructed as a linearly spaced vector in [-1, 1] of length `scan_pts`,
      broadcast across lines, and then shifted by -1.0, resulting in values in [-2, 0].
    - Computational complexity is O(line_per_frame * scan_pts * n_terms).
    - No in-place modifications are made to the input.
    Examples
    --------
    >>> import torch
    >>> data = torch.ones(2, 4, dtype=torch.complex64)
    >>> coeffs = torch.tensor([0.1, -0.01], dtype=data.real.dtype, device=data.device)
    >>> out = comp_dis_phase_torch(data, max_disp_orders=3, arrCountDispCoeff=coeffs)
    >>> out.shape
    torch.Size([2, 4])
    """

    # Amplitude/phase
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

def unpack12_torch(buf: torch.Tensor) -> torch.Tensor:
    assert buf.dtype == torch.uint8, "Input must be torch.uint8"
    n_triplets = buf.numel() // 3

    b0 = buf[0::3].to(torch.int32)   # promote to avoid overflow
    b1 = buf[1::3].to(torch.int32)
    b2 = buf[2::3].to(torch.int32)

    out = torch.empty(n_triplets * 2, dtype=torch.int32, device=buf.device)
    out[0::2] = b0 | ((b1 & 0x0F) << 8)
    out[1::2] = (b1 >> 4) | (b2 << 4)

    return out.to(dtype=torch.float32)  # or torch.uint16 if unsigned

def process_unp(unp_file_path:Path, meta: unp_meta) -> np.ndarray:

    if torch.cuda.is_available():
        show_info("CUDA is available. Using GPU for processing.")
        device = torch.device("cuda")
    else:
        show_info("CUDA is not available. Using CPU for processing.")
        device = torch.device("cpu")
    
    # read 2 bytes size for uint16
    if meta.packed:
        data_size_bytes = int(1.5 * meta.width * meta.height)
    else:
        data_size_bytes = 2 * meta.width * meta.height

    if meta.full_range:
        oct_vol_array = torch.zeros((meta.depth, meta.height, meta.width), dtype=torch.float32).to(device)
    else:
        oct_vol_array = torch.zeros((meta.depth, meta.height, int(meta.width/2)), dtype=torch.float32).to(device)

    # open file
    with open(unp_file_path, "rb", buffering=0) as byte_reader:
        # Set reference A-scan to find the dispersion coefficients
        # Use center frame (b-scan) of the volume
        reference_frame = math.ceil(meta.depth / 2)

        show_info(f"Reference frame: {reference_frame}\n")
        # move to center frame in binary file
        byte_reader.seek(data_size_bytes * (reference_frame-1), 0)

        if meta.packed:
            ref_RawData = byte_reader.read(data_size_bytes)
            array = np.frombuffer(ref_RawData, dtype="<u1")
            array = torch.tensor(array).to(device)
            array = unpack12_torch(array)
            array = array.reshape((meta.height, meta.width))
        else:    
            ref_RawData = byte_reader.read(data_size_bytes)
            array = np.frombuffer(ref_RawData, dtype=np.uint16)
            array = array.reshape((meta.height, meta.width)).astype(np.float32)
            array = torch.tensor(array).to(device)

        # Subtract the DC signal
        subtracted_signal = dc_subtraction_double_sweep_torch(array)
        
        # 1D Hamming window (like np.hamming)
        hamming = torch.hamming_window(subtracted_signal.shape[1], periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)
        hamming = hamming.unsqueeze(0).repeat(subtracted_signal.shape[0], 1)
        hamming_signal = subtracted_signal * hamming

        dispMaxOrder = 3
        coeffRange = 100

        if meta.auto_dispersion:
            dispCoeffs = set_dispersion_coefficients_torch(hamming_signal,dispMaxOrder,coeffRange)
        else:
            dispCoeffs = torch.tensor([0 , 0], device=hamming_signal.device) #disable dispersion compensation

        byte_reader.seek(0, 0)
        
        # Main OCT Volume process
        for frame_num in tqdm(range(0, meta.depth), desc="Processing Bscans"):

            if meta.packed:
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype="<u1")
                raw_data = torch.tensor(raw_data).to(device)
                raw = unpack12_torch(raw_data)
                raw = raw.reshape((meta.height, meta.width))
            else:
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype=np.uint16)
                raw = raw_data.reshape((meta.height, meta.width)).astype(np.float32)
                raw = torch.tensor(raw).to(device)

            # Subtract the DC signal
            subtracted_signal = dc_subtraction_double_sweep_torch(raw)

            # Hamming windowing
            hamming_signal = subtracted_signal * hamming
            
            img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffs)

            # Fourier Transform
            fft_signal = torch.fft.fft(img_disp_comp, dim=-1)

            if meta.full_range:
                temp_frame = torch.abs(fft_signal) #full range
            else:
                temp_frame = torch.abs(fft_signal[:, int(fft_signal.shape[1] / 2):]) #take the negative part

            oct_vol_array[frame_num] = temp_frame


    #apply double side
    if meta.double_side:
        # reverse the height axis for every odd B-scan (works for torch.Tensor)
        oct_vol_array[1::2, :, :] = torch.flip(oct_vol_array[1::2, :, :], dims=[1])

    #apply desine
    if meta.desine:
        for i in range(oct_vol_array.shape[0]):
            oct_vol_array[i] = desine_torch(oct_vol_array[i], mode="bilinear", transpose=True)
    
    return oct_vol_array.cpu().numpy()

def process_unp_sine_pause(unp_file_path:Path, meta: unp_meta) -> tuple[np.ndarray, np.ndarray]:

    indices = meta.sine_frame_indices
    pause_index = indices[::2]

    hires_ratio = meta.sine_hires_ratio
    hires_h = meta.height*hires_ratio
    hires_d = 2*hires_ratio

    ini_delay = meta.delay
    delay = round((ini_delay/10)*(hires_ratio-1) * 2)

    low_res_depth = meta.depth - len(pause_index)*hires_d*hires_ratio

    if torch.cuda.is_available():
        show_info("CUDA is available. Using GPU for processing.")
        device = torch.device("cuda")
    else:
        show_info("CUDA is not available. Using CPU for processing.")
        device = torch.device("cpu")
    
    # read 2 bytes size for uint16
    if meta.packed:
        data_size_bytes = int(1.5 * meta.width * meta.height)
    else:
        data_size_bytes = 2 * meta.width * meta.height

    if meta.full_range:
        oct_vol_array = torch.zeros((low_res_depth, meta.height, meta.width), dtype=torch.float32).to(device)
        oct_vol_array_hires = torch.zeros((hires_d*len(pause_index), hires_h, meta.width), dtype=torch.float32).to(device)

    else:
        oct_vol_array = torch.zeros((low_res_depth, meta.height, int(meta.width/2)), dtype=torch.float32).to(device)
        oct_vol_array_hires = torch.zeros((hires_d*len(pause_index), hires_h, int(meta.width/2)), dtype=torch.float32).to(device)

    # open file
    with open(unp_file_path, "rb", buffering=0) as byte_reader:
        # Set reference A-scan to find the dispersion coefficients
        # Use center frame (b-scan) of the volume
        reference_frame = math.ceil(meta.depth / 2)

        show_info(f"Reference frame: {reference_frame}\n")
        # move to center frame in binary file
        byte_reader.seek(data_size_bytes * (reference_frame-1), 0)
       
        if meta.packed:
            ref_RawData = byte_reader.read(data_size_bytes)
            array = np.frombuffer(ref_RawData, dtype="<u1")
            array = torch.tensor(array).to(device)
            array = unpack12_torch(array)
            array = array.reshape((meta.height, meta.width))
        else:
            ref_RawData = byte_reader.read(data_size_bytes)
            array = np.frombuffer(ref_RawData, dtype=np.uint16)
            array = array.reshape((meta.height, meta.width)).astype(np.float32)
            array = torch.tensor(array).to(device)

        # Subtract the DC signal
        subtracted_signal = dc_subtraction_double_sweep_torch(array)
        
        # 1D Hamming window (like np.hamming)
        hamming = torch.hamming_window(meta.width, periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)
        hamming = hamming.unsqueeze(0).repeat(subtracted_signal.shape[0], 1)
        hamming_signal = subtracted_signal * hamming

        hamming_hires = torch.hamming_window(meta.width, periodic=False, dtype=subtracted_signal.dtype, device=subtracted_signal.device)
        hamming_hires = hamming_hires.unsqueeze(0).repeat(hires_h, 1)

        dispMaxOrder = 3
        coeffRange = 100

        if meta.auto_dispersion:
            dispCoeffs = set_dispersion_coefficients_torch(hamming_signal,dispMaxOrder,coeffRange)
        else:
            dispCoeffs = torch.tensor([0 , 0], device=hamming_signal.device) #disable dispersion compensation

        
        frame_counter = 0
        frame_counter_lowres = 0
        frame_counter_hires = 0

        pause_index_lowres = []

        byte_reader.seek(0,0) #reset to beginning of file
        
        # Main OCT Volume process
        for _ in tqdm(range(0, low_res_depth), desc="Processing Bscans"):

            if frame_counter in pause_index:
                for _ in range(hires_d):
                    if meta.packed:
                        raw_data = np.frombuffer(byte_reader.read(data_size_bytes * hires_ratio), dtype="<u1")
                        raw_data = torch.tensor(raw_data).to(device)
                        raw = unpack12_torch(raw_data)
                        raw = raw.reshape((hires_h, meta.width))
                    else:
                        raw_data = np.frombuffer(byte_reader.read(data_size_bytes*hires_ratio), dtype=np.uint16)
                        raw = raw_data.reshape((hires_h, meta.width)).astype(np.float32)
                        raw = torch.tensor(raw).to(device)

                    # Subtract the DC signal
                    subtracted_signal = dc_subtraction_double_sweep_torch(raw)

                    # Hamming windowing
                    hamming_signal = subtracted_signal * hamming_hires

                    img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffs)

                    # Fourier Transform
                    fft_signal = torch.fft.fft(img_disp_comp, dim=-1)

                    if meta.full_range:
                        temp_frame = torch.abs(fft_signal)  # full range
                    else:
                        temp_frame = torch.abs(fft_signal[:, int(fft_signal.shape[1] / 2):])  # take the negative part

                    oct_vol_array_hires[frame_counter_hires] = temp_frame

                    frame_counter_hires += 1

                pause_index_lowres.append(frame_counter_lowres)

                frame_counter += hires_d*hires_ratio 
                frame_counter_lowres += 1

                continue
            
            else:
                if meta.packed:
                    raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype="<u1")
                    raw_data = torch.tensor(raw_data).to(device)
                    raw = unpack12_torch(raw_data)
                    raw = raw.reshape((meta.height, meta.width))
                else:
                    raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype=np.uint16)
                    raw = raw_data.reshape((meta.height, meta.width)).astype(np.float32)
                    raw = torch.tensor(raw).to(device)

                # Subtract the DC signal
                subtracted_signal = dc_subtraction_double_sweep_torch(raw)

                # Hamming windowing
                hamming_signal = subtracted_signal * hamming
                
                img_disp_comp = comp_dis_phase_torch(hamming_signal, dispMaxOrder, dispCoeffs)

                # Fourier Transform
                fft_signal = torch.fft.fft(img_disp_comp, dim=-1)

                if meta.full_range:
                    temp_frame = torch.abs(fft_signal) #full range
                else:
                    temp_frame = torch.abs(fft_signal[:, int(fft_signal.shape[1] / 2):]) #take the negative part

                if frame_counter % 2:
                    temp_frame = torch.flip(temp_frame, [0])#horizontal flip  

                oct_vol_array[frame_counter_lowres] = temp_frame

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

        # resize first image and insert back to the low-res volume
        temp_frame = hires_bscan[0]  # take the first B-scan for resizing
        temp_frame = temp_frame.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(temp_frame, size=(meta.height, temp_frame.shape[-1]), mode="bilinear", align_corners=False)
        resized = resized.squeeze(0).squeeze(0)
        oct_vol_array[pause_index_lowres[i]] = resized
    
    #double side the high-res volume
    if meta.double_side:
        oct_vol_array_hires[1::2, :, :] = torch.flip(oct_vol_array_hires[1::2, :, :], dims=[1])

    #double side the low-res volume
    if not meta.double_side:
        # reverse the height axis for every odd B-scan (works for torch.Tensor)
        oct_vol_array[1::2, :, :] = torch.flip(oct_vol_array[1::2, :, :], dims=[1])

    #applt desine both volumes
    if meta.desine:
        for i in range(oct_vol_array.shape[0]):
            oct_vol_array[i] = desine_torch(oct_vol_array[i], mode="bilinear", transpose=True)

        for i in range(oct_vol_array_hires.shape[0]):
            oct_vol_array_hires[i] = desine_torch(oct_vol_array_hires[i], mode="bilinear", transpose=True)
    
    return oct_vol_array.cpu().numpy(), oct_vol_array_hires.cpu().numpy()
    



