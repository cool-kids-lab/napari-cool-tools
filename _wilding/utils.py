import numpy as np
import torch
import torch.nn.functional as F

def get_padding_params(shape, stride=32):
    h, w = shape[-2:]
    new_h = (h + stride - 1) // stride * stride
    new_w = (w + stride - 1) // stride * stride
    ph, pw = new_h - h, new_w - w
    return (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)

def apply_padding(tensor, pads):
    return F.pad(tensor, pads, mode="constant", value=0)

def undo_padding(tensor, pads):
    lw, rw, th, bh = pads
    if bh > 0 and rw > 0: return tensor[..., th:-bh, lw:-rw]
    if bh > 0: return tensor[..., th:-bh, lw:]
    if rw > 0: return tensor[..., th:, lw:-rw]
    return tensor[..., th:, lw:]

def to_tensor(image_np, device="cpu"):
    """Enhanced zero-copy conversion with explicit type safety."""
    if image_np.ndim == 3:
        image_np = image_np.squeeze()
    
    # Force float32 to match EfficientNet weights
    x = torch.as_tensor(image_np, dtype=torch.float32, device=device)
    return (x.unsqueeze(0).unsqueeze(0) / 255.0)
