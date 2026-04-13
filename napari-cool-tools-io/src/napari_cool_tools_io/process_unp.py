import math
from pathlib import Path
from napari_cool_tools_oct_preproc._oct_preproc_func import desine
import numpy as np
from tqdm import tqdm
import torch
from napari_cool_tools_io import getWindow, unp_meta
import torch.nn.functional as F
from napari.utils.notifications import show_info
from napari_cool_tools_io import device

def reshuffle_vista_frames(ref_RawData: np.ndarray, nvista: int, numMScans: int) -> np.ndarray:
    """
    Reorders frames in ref_RawData for 'vista' grouping, matching the given MATLAB logic.

    Parameters
    ----------
    ref_RawData : np.ndarray
        3D array of shape (H, W, numBScans).
    nvista : int
        Number of vista views.
    numMScans : int
        Number of M-scans.

    Returns
    -------
    np.ndarray
        Reordered array with the same shape as ref_RawData.
    """
    numBScans  = ref_RawData.shape[0]
    if nvista <= 1:
        return ref_RawData.copy()

    block = nvista * numMScans
    if numBScans % block != 0:
        raise ValueError(
            f"numBScans ({numBScans}) must be divisible by nvista*numMScans ({block})."
        )

    # --- Vectorized permutation (fast) ---
    # Within each block of size (nvista x numMScans), MATLAB orders linear indices column-wise.
    # The MATLAB code remaps to row-wise order. Build that mapping:
    # perm[t] = source_index_within_block for target position t (0-based).
    perm = np.arange(block).reshape(nvista, numMScans, order='F').ravel(order='C')

    out = np.empty_like(ref_RawData)
    for start in range(0, numBScans, block):
        out[start:start+block,:,:] = ref_RawData[start + perm, :, :]

    return out

def reshuffle_vista_frames_torch(ref_RawData: torch.Tensor, nvista: int, numMScans: int) -> torch.Tensor:
    """
    Reorders frames in ref_RawData for 'vista' grouping using torch tensors.

    Parameters
    ----------
    ref_RawData : torch.Tensor
        3D tensor of shape (numBScans, H, W).
    nvista : int
        Number of vista views.
    numMScans : int
        Number of M-scans.

    Returns
    -------
    torch.Tensor
        Reordered tensor with the same shape as ref_RawData.
    """
    numBScans = ref_RawData.shape[0]
    if nvista <= 1:
        return ref_RawData.clone()

    block = nvista * numMScans
    if numBScans % block != 0:
        raise ValueError(
            f"numBScans ({numBScans}) must be divisible by nvista*numMScans ({block})."
        )

    # Build permutation indices using numpy, then convert to torch
    perm = np.arange(block).reshape(nvista, numMScans, order='F').ravel(order='C')
    perm = torch.from_numpy(perm).to(ref_RawData.device)

    out = torch.empty_like(ref_RawData)
    for start in range(0, numBScans, block):
        out[start:start+block] = ref_RawData[start + perm]

    return out


def reshuffle_vista_indices(indices: np.ndarray, nvista: int, numMScans: int) -> np.ndarray:
    """
    Reorders sorted indeces for 'vista' grouping, matching the given MATLAB logic.

    indices should be 1D array of indices.

    """

    numBScans  = len(indices)

    block = nvista * numMScans
    if numBScans % block != 0:
        raise ValueError(
            f"numBScans ({numBScans}) must be divisible by nvista*numMScans ({block})."
        )

    # --- Vectorized permutation (fast) ---
    # Within each block of size (nvista x numMScans), MATLAB orders linear indices column-wise.
    # The MATLAB code remaps to row-wise order. Build that mapping:
    # perm[t] = source_index_within_block for target position t (0-based).
    perm = np.arange(block).reshape(nvista, numMScans, order='F').ravel(order='C')

    out_indices = np.empty_like(indices)
    for start in range(0, numBScans, block):
        out_indices[start:start+block] = indices[start + perm]

    return out_indices #output is the reshuffled indices

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
    corrected_1 = data[0::2, :] # every even A-scan
    # corrected_2 = data[1::2, :] # every odd A-scan
    corrected_2 = torch.flip(data[1::2, :], dims=[1]) # flip the odd A-scans to align with the even ones

    # Remove DC component by subtracting the median spectrum
    # Subtract median (DC removal) along each spectrum (per column)
    corrected_1 = corrected_1 - torch.mean(corrected_1, dim=0, keepdim=True)
    corrected_2 = corrected_2 - torch.mean(corrected_2, dim=0, keepdim=True)

    # Recombine into full B-scan
    subtracted_signal = torch.zeros_like(data)
    subtracted_signal[0::2, :] = corrected_1
    subtracted_signal[1::2, :] = corrected_2

    return subtracted_signal


def set_dispersion_coefficients_torch(data: torch.Tensor, maxDispOrders, coefRange, dispersion_mode: int = 0) -> list:
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
    arrCountDispCoeff = [0,0]

    for idx_CounterDispCoef in tqdm(
        range(0, len(arrCountDispCoeff)), desc="Calculating Displacement Coefficients"
    ):
        arrDispCoeffRange = np.arange(-1 * coefRange, coefRange + 1, 1)
        arrCost = np.zeros((arrDispCoeffRange.shape[0]))

        for k in tqdm(range(0, len(arrDispCoeffRange)), desc="Calculating Costs"):
            arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[k]
            arrCost[k] = cal_cost_function_torch(data, maxDispOrders, arrCountDispCoeff, dispersion_mode=dispersion_mode)

        argMinCost = arrCost.argmax()
        arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost]

    return arrCountDispCoeff


def cal_cost_function_torch(data: torch.Tensor, maxDispOrders, arrCountDispCoeff: list, dispersion_mode: int = 0) -> torch.Tensor:
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
    arrCountDispCoeff : list
        1D list of length >= max(0, maxDispOrders - 1) containing real-valued dispersion coefficients.
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
    data_disp_comp = comp_dis_phase_torch(data, maxDispOrders, arrCountDispCoeff, mode=dispersion_mode)

    # FFT magnitude squared
    toct = torch.abs(torch.fft.ifft(data_disp_comp, dim=-1)) ** 2
    
    # Avoid edges
    # roi_oct = toct[:, 50 : int(data_disp_comp.shape[1] / 2) - 50]#this is the positive half
    roi_oct = toct[50:-50, int(data_disp_comp.shape[1] / 2) + 50 : -50] #take the negative part
    
    # Normalize
    norm_oct = roi_oct / torch.sum(roi_oct)
    
    # Shannon entropy
    eps = 1e-12#this is to avoid nan
    entropy = norm_oct * torch.log10(norm_oct + eps)
    
    # Final cost
    cost = torch.sum(entropy)
    return cost

def comp_dis_phase_torch(data: torch.Tensor, max_disp_orders, arrCountDispCoeff: list, mode: int = 3) -> torch.Tensor:
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
    arrCountDispCoeff : list
        1D list of length >= max(0, max_disp_orders - 1) containing real-valued dispersion coefficients
        [c2, c3, ..., c_{max_disp_orders}]. Should reside on the same device as `data` and use a real dtype
        compatible with `data`'s real component.
    mode : int, optional
        Mode of dispersion compensation. 0 = global, 1 = quadratic, 2=sinusoidal, 3=quadratic + sinusoidal.
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
    k_linear = torch.linspace(-1.0, 1.0, scan_pts, device=data.device, dtype=phase.dtype).unsqueeze(0) - 1.0
    k_axis   = k_linear.expand(line_per_frame, -1)

    if mode == 0:
        for i in range(2):
            phase = phase + arrCountDispCoeff[i] * k_axis.pow(i + 2)

    if mode == 1:
        h = line_per_frame
        a = arrCountDispCoeff[0]
        b = a + arrCountDispCoeff[1]

        # === Coefficients for y = A x^2 + B x + C ===
        A = 4.0 * (a - b) / (h ** 2)
        B = -4.0 * (a - b) / h
        C = a

        # Define f(x) analogous to MATLAB's function handle, using PyTorch ops
        f = lambda x: A * x**2 + B * x + C

        x = torch.linspace(0, h, h, device=data.device, dtype=phase.dtype).unsqueeze(-1)
        coeff= f(x)
        coeff = coeff.expand(-1, scan_pts)

        phase = phase + coeff * k_axis.pow(2)

    if mode == 2:
        h = line_per_frame
        a = arrCountDispCoeff[0]
        b = arrCountDispCoeff[1]

        x = torch.linspace(0, h, h, device=data.device, dtype=phase.dtype).unsqueeze(-1)
        x_norm = (x/h)*2*torch.pi
        coeff = b*0.5*(torch.sin(x_norm - torch.pi/2) + 1) + a
        coeff = coeff.expand(-1, scan_pts)

        phase = phase + coeff * k_axis.pow(2)

    
    if mode == 3:
        h = line_per_frame
        a = arrCountDispCoeff[0]
        b = a + arrCountDispCoeff[1]

        # === Coefficients for y = A x^2 + B x + C ===
        A = 4.0 * (a - b) / (h ** 2)
        B = -4.0 * (a - b) / h
        C = a

        # Define f(x) analogous to MATLAB's function handle, using PyTorch ops
        f = lambda x: A * x**2 + B * x + C

        x = torch.linspace(0, h, h, device=data.device, dtype=phase.dtype).unsqueeze(-1)
        x_org = (h/2) * torch.sin(torch.pi/h * x - torch.pi/2) + (h/2)

        coeff = f(x_org)
        coeff = coeff.expand(-1, scan_pts)

        phase = phase + coeff * k_axis.pow(2)

    # Recombine amplitude and phase: amp * exp(1j*phase)
    data_disp_comp = amp * torch.exp(1j*phase)
    return data_disp_comp

# def comp_dis_phase_torch(data: torch.Tensor, max_disp_orders, arrCountDispCoeff: torch.Tensor) -> torch.Tensor:
#     """
#     Dispersion-phase compensation for complex OCT data using PyTorch.
#     This function decomposes a complex-valued signal into amplitude and phase, adds a
#     polynomial dispersion phase term across a normalized k-axis, and recombines the
#     amplitude with the corrected phase. The dispersion phase is modeled as:
#         phase += sum_{i=0}^{n_terms-1} coeff[i] * k^(i+2)
#     i.e., powers k^2, k^3, ..., k^{max_disp_orders}, where n_terms = max(0, max_disp_orders - 1).
#     Parameters
#     ----------
#     data : torch.Tensor
#         Complex-valued tensor of shape (line_per_frame, scan_pts). Must be a complex dtype
#         (e.g., torch.complex64 or torch.complex128). The device and dtype drive internal computations.
#     max_disp_orders : int
#         Maximum polynomial order of the dispersion phase to apply. If <= 1, no dispersion phase is added.
#     arrCountDispCoeff : torch.Tensor
#         1D tensor of length >= max(0, max_disp_orders - 1) containing real-valued dispersion coefficients
#         [c2, c3, ..., c_{max_disp_orders}]. Should reside on the same device as `data` and use a real dtype
#         compatible with `data`'s real component.
#     Returns
#     -------
#     torch.Tensor
#         Complex-valued tensor of the same shape and dtype as `data`, with dispersion compensation applied.
#     Notes
#     -----
#     - The k-axis is constructed as a linearly spaced vector in [-1, 1] of length `scan_pts`,
#       broadcast across lines, and then shifted by -1.0, resulting in values in [-2, 0].
#     - Computational complexity is O(line_per_frame * scan_pts * n_terms).
#     - No in-place modifications are made to the input.
#     Examples
#     --------
#     >>> import torch
#     >>> data = torch.ones(2, 4, dtype=torch.complex64)
#     >>> coeffs = torch.tensor([0.1, -0.01], dtype=data.real.dtype, device=data.device)
#     >>> out = comp_dis_phase_torch(data, max_disp_orders=3, arrCountDispCoeff=coeffs)
#     >>> out.shape
#     torch.Size([2, 4])
#     """

#     # Amplitude/phase
#     amp   = torch.abs(data)
#     phase = torch.angle(data)

#     line_per_frame, scan_pts = data.shape

#     # k-axis (broadcasted across lines)
#     k_linear = torch.linspace(-1.0, 1.0, scan_pts, device=data.device, dtype=data.dtype)
#     k_axis   = k_linear.unsqueeze(0).expand(line_per_frame, -1) - 1.0

#     # Apply dispersion phase terms: i from 0..max_disp_orders-2 -> power i+2
#     # (matches your NumPy loop)
#     n_terms = max(0, max_disp_orders - 1)
#     for i in range(n_terms):
#         phase = phase + arrCountDispCoeff[i] * k_axis.pow(i + 2)

#     # Recombine amplitude and phase: amp * exp(1j*phase)
#     data_disp_comp = amp * torch.exp(1j*phase)
#     return data_disp_comp

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


def process_unp(unp_file_path:Path, meta: unp_meta, auto_dispersion:bool=False, flip_coeffs:bool=False) -> np.ndarray:

    show_info("Starting unp file processing...")
    
    # read 2 bytes size for uint16
    if meta.packed:
        data_size_bytes = int(1.5 * meta.width * meta.height)
    else:
        data_size_bytes = 2 * meta.width * meta.height

    if meta.full_range:
        if meta.split_spectrum:
            oct_vol_array = torch.zeros((meta.depth, meta.height*2, meta.width//2), dtype=torch.float32).to(device)
        else:
            oct_vol_array = torch.zeros((meta.depth, meta.height, meta.width), dtype=torch.float32).to(device)
    else:
        if meta.split_spectrum:
            oct_vol_array = torch.zeros((meta.depth, meta.height*2, meta.width//4), dtype=torch.float32).to(device)
        else:
            oct_vol_array = torch.zeros((meta.depth, meta.height, meta.width//2), dtype=torch.float32).to(device)

    # open file
    with open(unp_file_path, "rb", buffering=0) as byte_reader:
        
        # 1D Hamming window (like np.hamming)
        hamming = getWindow(meta.width, meta.windowType, dtype=torch.float32, device=device)
        hamming = hamming.unsqueeze(0).repeat(meta.height, 1)
        dispMaxOrder = 3

        # auto dispersion (does not support split dispersion yet, only calculates c2 and c3 for the whole volume, global mode)
        #this is for experimental only
        if auto_dispersion:
            # move to center frame in binary file
            byte_reader.seek(int(data_size_bytes) * int(meta.depth/2), 0)

            if meta.packed:
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype="<u1")
                if raw_data.size != data_size_bytes:
                    pass

                else:
                    raw_data = torch.tensor(raw_data).to(device)
                    raw = unpack12_torch(raw_data)
                    raw = raw.reshape((meta.height, meta.width))

                    if meta.dcSubtract:
                    # Subtract the DC signal
                        subtracted_signal = dc_subtraction_double_sweep_torch(raw)
                    else:
                        subtracted_signal = raw

                    # Hamming windowing
                    hamming_signal = subtracted_signal * hamming

                    dispersion_coeffs = set_dispersion_coefficients_torch(hamming_signal, maxDispOrders=dispMaxOrder, coefRange=100, dispersion_mode=meta.dispersion_mode)
                    c2,c3 = dispersion_coeffs
                    
                    if flip_coeffs:
                        if c2 < 0:
                            c2 = c2 * -1
                            c3 = c3 * -1
                            
                    meta.c2A = int(c2)
                    meta.c3A = int(c3)


            else:
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes), dtype=np.uint16)
                if raw_data.size != meta.height * meta.width:
                    pass

                else:
                    raw = raw_data.reshape((meta.height, meta.width)).astype(np.float32)
                    raw = torch.tensor(raw).to(device)

                    if meta.dcSubtract:
                    # Subtract the DC signal
                        subtracted_signal = dc_subtraction_double_sweep_torch(raw)
                    else:
                        subtracted_signal = raw

                    # Hamming windowing
                    hamming_signal = subtracted_signal * hamming

                    dispersion_coeffs = set_dispersion_coefficients_torch(hamming_signal, maxDispOrders=dispMaxOrder, coefRange=100, dispersion_mode=meta.dispersion_mode)
                    c2,c3 = dispersion_coeffs
                    
                    if flip_coeffs:
                        if c2 < 0:
                            c2 = c2 * -1
                            c3 = c3 * -1

                    meta.c2A = int(c2)
                    meta.c3A = int(c3)

        byte_reader.seek(0, 0)
        
        # Main OCT Volume process
        for frame_num in tqdm(range(0, meta.depth), desc="Processing Bscans"):

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
                dispCoeffsA = [meta.c2A, meta.c3A] #disable dispersion compensation
                dispCoeffsB = [meta.c2B, meta.c3B] #disable dispersion compensation
                img_disp_comp[0::2] = comp_dis_phase_torch(hamming_signal[0::2], dispMaxOrder, dispCoeffsA, mode=meta.dispersion_mode)
                img_disp_comp[1::2] = comp_dis_phase_torch(hamming_signal[1::2], dispMaxOrder, dispCoeffsB, mode=meta.dispersion_mode)

            else:
                dispCoeffsA = [meta.c2A, meta.c3A] #disable dispersion compensation
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

            oct_vol_array[frame_num] = temp_frame

    oct_vol_array = oct_vol_array.permute(0,2,1)

    #it should be double sided first, before desine
    if meta.double_side:
        if meta.bmscan >1:
            if meta.vista >1:
                oct_vol_array = reshuffle_vista_frames_torch(oct_vol_array, meta.vista, meta.bmscan)

                for frame_num in range(meta.depth):
                    cframe = int(np.floor(frame_num/(meta.bmscan)))
                    if (cframe % 2): #flip every odd frame (1,3,5,...) python is zero indexed
                        oct_vol_array[frame_num] = torch.flip(oct_vol_array[frame_num], dims=[-1])
            else:
                for frame_num in range(meta.depth):
                    cframe = int(np.floor(frame_num/(meta.bmscan)))
                    if (cframe % 2):
                        oct_vol_array[frame_num] = torch.flip(oct_vol_array[frame_num], dims=[-1])
        else:
            oct_vol_array[1::2, :, :] = torch.flip(oct_vol_array[1::2, :, :], dims=[-1])
        
    if meta.desine:
        oct_vol_array = desine(oct_vol_array, mode="bilinear", transpose=False, scale_fac=2)

    oct_vol_array = oct_vol_array.cpu().numpy()

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished unp file processing.")
    
    return oct_vol_array


def process_unp_sine_pause(unp_file_path:Path, meta: unp_meta, include_hires_in_lowres=True) -> tuple[np.ndarray, np.ndarray]:

    show_info("Starting unp file processing.")

    indices = meta.sine_frame_indices
    pause_index = indices[0::2]

    print("Pause indices:", pause_index)

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

    else:
        oct_vol_array = torch.zeros((low_res_depth, meta.height, int(meta.width/2)), dtype=torch.float32).to(device)
        oct_vol_array_hires = torch.zeros((hires_d*len(pause_index), hires_h, int(meta.width/2)), dtype=torch.float32).to(device)

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
                        # dispCoeffsB = [-1.0*meta.c2A, -1.0*meta.c3A]
                        # img_disp_comp[0::2] = comp_dis_phase_torch(hamming_signal[0::2], dispMaxOrder, dispCoeffsA, mode=meta.dispersion_mode)
                        # img_disp_comp[1::2] = comp_dis_phase_torch(hamming_signal[1::2], dispMaxOrder, dispCoeffsB, mode=meta.dispersion_mode)


                    # Fourier Transform
                    if meta.split_spectrum:
                        # Split Spectrum Fourier Transform
                        half_point = img_disp_comp.shape[-1] // 2
                        img_disp_comp_split = torch.zeros((img_disp_comp.shape[0]*2, half_point), dtype=img_disp_comp.dtype)

                        img_disp_comp_split[0::4, :] = img_disp_comp[0::2, :half_point]
                        img_disp_comp_split[1::4, :] = img_disp_comp[0::2, half_point:]

                        img_disp_comp_split[3::4, :] = img_disp_comp[1::2, half_point:]
                        # img_disp_comp_split[2::4, :] = torch.flip(img_disp_comp_split[2::4, :], dims=[1])
                        img_disp_comp_split[2::4, :] = img_disp_comp[1::2, :half_point]
                        # img_disp_comp_split[3::4, :] = torch.flip(img_disp_comp_split[3::4, :], dims=[1])

                        fft_signal = torch.fft.ifft(img_disp_comp_split, dim=-1)

                    else:
                        # Standard Fourier Transform
                        # img_disp_comp[1::2, :] = torch.flip(img_disp_comp[1::2, :], dims=[1]) #flip the odd frames
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

    oct_vol_array, oct_vol_array_hires = oct_vol_array.cpu().numpy(), oct_vol_array_hires.cpu().numpy()

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished unp file processing.")
    
    return oct_vol_array, oct_vol_array_hires