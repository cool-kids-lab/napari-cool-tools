__version__ = "0.0.1"

__all__ = ()

import napari
import torch
from napari.utils.notifications import show_info
from dataclasses import dataclass

viewer = napari.current_viewer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class unp_meta:
    width: int = 0
    height: int = 0
    depth: int = 0
    bmscan: int = 0
    vista: int = 0
    packed: bool = False
    double_side: bool = False
    pattern: str = "Sine"
    full_range: bool = False
    desine: bool = False
    dcSubtract: bool = True
    log_scale: bool = False
    max_projection: bool = False
    delay: int = 0
    sine_frame_indices: list[int] = None
    sine_hires_ratio: int = 0
    c2A: int = 0
    c3A: int = 0
    c2B: int = 0
    c3B: int = 0
    split_dispersion: bool = False
    dispersion_mode: int = 0
    octa: str = "none"
    structure: bool = False
    windowType: int = 0
    split_spectrum: bool = False
    motor_position: int | None = None

def memory_stats():
    show_info(f"Gpu memory allocated: {torch.cuda.memory_allocated() / 1024**2}")
    show_info(f"Gpu memory reserved: {torch.cuda.memory_reserved() / 1024**2}")

    gpu_mem_clear = torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0

    print(f"GPU memory is clear: {gpu_mem_clear}\n")
    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")



def gaussian_window(
    M: int,
    sigma: float,
    periodic: bool = False,
    *,
    dtype=torch.float32,
    device=torch.device("cpu")
) -> torch.Tensor:
    """
    Create a Gaussian window.

    Parameters
    ----------
    M : int
        Window length.
    sigma : float
        Standard deviation of the Gaussian in samples.
        Example: sigma = 0.4 * (M - 1)
    periodic : bool, default=False
        If True, return a periodic window (drop last point).
        If False, return a symmetric window.
    dtype : torch.dtype, default=torch.float32
    device : PyTorch device, default='cpu'

    Returns
    -------
    Tensor of shape (M,)
    """

    if M <= 0:
        return torch.empty((0,), dtype=dtype, device=device)

    # Convert periodic to symmetric length (PyTorch convention)
    M_sym = M + 1 if (periodic and M > 1) else M

    # Indices
    n = torch.arange(M_sym, dtype=dtype, device=device)

    # Center of the window
    mu = (M_sym - 1) / 2.0

    # Gaussian formula
    w = torch.exp(-0.5 * ((n - mu) / sigma) ** 2)

    # Remove last sample for periodic window
    if periodic and M > 1:
        w = w[:-1]

    return w

def tukey_window(M: int,
                 alpha: float = 0.5,
                 periodic: bool = False,
                 *,
                 dtype=torch.float32,
                 device=torch.device("cpu")) -> torch.Tensor:
    """
    Tukey (tapered cosine) window.

    Parameters
    ----------
    M : int
        Window length.
    alpha : float, default=0.5
        0 -> rectangular, 1 -> Hann.
    periodic : bool, default=False
        If True, return periodic form (drop last sample of symmetric window).
    dtype, device : optional
        Tensor dtype/device.

    Returns
    -------
    w : (M,) tensor
    """
    if M <= 0:
        return torch.empty((0,), dtype=dtype, device=device)

    M_sym = M + 1 if (periodic and M > 1) else M

    if dtype is None:
        dtype = torch.get_default_dtype()

    n = torch.arange(M_sym, dtype=dtype, device=device)

    if alpha <= 0:
        w = torch.ones(M_sym, dtype=dtype, device=device)
    elif alpha >= 1:
        # Hann window
        w = 0.5 * (1 - torch.cos(2 * torch.pi * n / (M_sym - 1)))
    else:
        # Piecewise Tukey
        w = torch.empty(M_sym, dtype=dtype, device=device)
        edge = alpha * (M_sym - 1) / 2

        m1 = n < edge
        m2 = (n >= edge) & (n <= (M_sym - 1) * (1 - alpha / 2))
        m3 = ~m1 & ~m2  # right taper

        w[m1] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[m1] / (alpha * (M_sym - 1)) - 1)))
        w[m2] = 1.0
        w[m3] = 0.5 * (1 + torch.cos(torch.pi * (2 * n[m3] / (alpha * (M_sym - 1)) - 2 / alpha + 1)))

    if periodic and M > 1:
        w = w[:-1]
    return w


def getWindow(width: int, type: int, dtype: torch.dtype, device:torch.device) -> torch.Tensor:
    if type == 0:
        return torch.hamming_window(width, periodic=False, dtype=dtype, device=device)
    elif type == 1:
        return tukey_window(width, alpha=0.5, periodic=False, dtype=dtype, device=device)
    elif type == 2:
        return gaussian_window(width, sigma=0.2 * (width - 1), periodic=False, dtype=dtype, device=device)
    else:
        raise ValueError("Invalid window type. Use 0 for Hanning and 1 for Hamming.")

# @napari.Viewer.bind_key("i")
# def shortcut(viewer):
#     # init layers and active selection
#     layers = viewer.layers
#     curr_sel = layers.selection

#     # check that label layer is selected

#     # case only one layer is selected
#     if len(curr_sel) == 1:
#         # get current layer
#         curr_layer = list(curr_sel)[0]
#         curr_layer_type = curr_layer.as_layer_data_tuple()[2]

#         # set default opacity
#         new_opacity = curr_layer.opacity

#         # case selected layer is a labels layer
#         if curr_layer_type == "labels":
#             # get current opacity
#             opacity = curr_layer.opacity

#             # case opacity is greater than 0
#             if opacity > 0:
#                 # increase opacity size
#                 new_opacity = opacity - 0.1

#                 if new_opacity < 0:
#                     new_opacity = 0
#                 else:
#                     pass

#                 curr_layer.opacity = new_opacity

#                 # update viewer with mesage
#                 msg = f"decrease opacity to {new_opacity}"
#                 viewer.status = msg

#             # case opacity is < 0
#             else:
#                 pass

#         # case selected layer is not a labels layer
#         else:
#             pass

#         pass

#     # case multiple layers are selected
#     else:
#         pass


# @napari.Viewer.bind_key("o")
# def shortcut2(viewer):
#     # init layers and active selection
#     layers = viewer.layers
#     curr_sel = layers.selection

#     # check that label layer is selected

#     # case only one layer is selected
#     if len(curr_sel) == 1:
#         # get current layer
#         curr_layer = list(curr_sel)[0]
#         curr_layer_type = curr_layer.as_layer_data_tuple()[2]

#         # set default opacity
#         new_opacity = curr_layer.opacity

#         # case selected layer is a labels layer
#         if curr_layer_type == "labels":
#             # get current opacity
#             opacity = curr_layer.opacity

#             # case opacity is less than 1
#             if opacity < 1:
#                 # increase opacity
#                 new_opacity = opacity + 0.1

#                 if new_opacity > 1:
#                     new_opacity = 1
#                 else:
#                     pass

#                 curr_layer.opacity = new_opacity

#                 # update viewer with mesage
#                 msg = f"increase opacity to {new_opacity}"
#                 viewer.status = msg

#             # case opacity is >= 1
#             else:
#                 pass

#         # case selected layer is not a labels layer
#         else:
#             pass

#         pass

#     # case multiple layers are selected
#     else:
#         pass


# @napari.Viewer.bind_key("[")
# def shortcut3(viewer):
#     # init layers and active selection
#     layers = viewer.layers
#     curr_sel = layers.selection

#     # check that label layer is selected

#     # case only one layer is selected
#     if len(curr_sel) == 1:
#         # get current layer
#         curr_layer = list(curr_sel)[0]
#         curr_layer_type = curr_layer.as_layer_data_tuple()[2]

#         # case selected layer is a labels layer
#         if curr_layer_type == "labels":
#             # get current brush size
#             brush_size = curr_layer.brush_size

#             # case brush size is greater than 1
#             if brush_size > 1:
#                 # case brush size is odd
#                 if brush_size % 2 == 1:
#                     brush_size = brush_size - 1

#                 # case brush size is even
#                 else:
#                     pass

#                 # decrease brush size
#                 curr_layer.brush_size = brush_size - 1

#                 # update viewer with mesage
#                 msg = f"decrease brush size to {brush_size - 1}"
#                 viewer.status = msg

#             # case brush size is <= 1
#             else:
#                 pass

#         # case selected layer is not a labels layer
#         else:
#             pass

#         pass

#     # case multiple layers are selected
#     else:
#         pass


# @napari.Viewer.bind_key("]")
# def shortcut4(viewer):
#     # init layers and active selection
#     layers = viewer.layers
#     curr_sel = layers.selection

#     # check that label layer is selected

#     # case only one layer is selected
#     if len(curr_sel) == 1:
#         # get current layer
#         curr_layer = list(curr_sel)[0]
#         curr_layer_type = curr_layer.as_layer_data_tuple()[2]

#         # case selected layer is a labels layer
#         if curr_layer_type == "labels":
#             # get current brush size
#             brush_size = curr_layer.brush_size

#             # case brush size is less than 40
#             if brush_size < 40:
#                 # case brush size is even
#                 if brush_size % 2 == 0:
#                     brush_size = brush_size + 1

#                 # case brush size is odd
#                 else:
#                     pass

#                 # increase brush size
#                 curr_layer.brush_size = brush_size + 1

#                 # update viewer with mesage
#                 msg = f"increase brush size to {brush_size + 1}"
#                 viewer.status = msg

#             # case brush size is >= 40
#             else:
#                 pass

#         # case selected layer is not a labels layer
#         else:
#             pass

#         pass

#     # case multiple layers are selected
#     else:
#         pass