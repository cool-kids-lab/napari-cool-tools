import torch
import gc
import numpy as np
from experimental_model_def import MaskGenerator

@torch.inference_mode()
def process(data, model, return_numpy: bool = True):
    """Generates a uint8 mask and cleans up memory using inference optimization.
    
    Args:
        data: Input signal as a numpy array or torch tensor.
        model: The MaskGenerator model instance.
        return_numpy: If True, returns result as numpy, else torch tensor.
        
    Returns:
        A high-sensitivity uint8 mask moved to CPU.
    """
    input = torch.as_tensor(data).float()
    device = input.device
    
    try:
        model.to(device).eval()
        
        # Standardize input dimensions to (B, C, H, W)
        if input.ndim == 2:
            input = input.unsqueeze(0).unsqueeze(0)
        
        # Optimize memory layout for 2026 compute kernels
        input = input.to(memory_format=torch.channels_last)
        
        mask = model(input)
        
        # Convert to uint8 (0 or 1) and move to CPU
        output = mask.to(torch.uint8).cpu()
        
        if return_numpy:
            output = output.numpy()
            
        return output

    finally:
        # CLEANUP: Explicitly break tensor references for garbage collection
        del input
        if 'mask' in locals(): 
            del mask
            
        gc.collect()
        
        # Clear hardware cache for non-CPU devices (e.g., CUDA or XPU)
        if device.type != "cpu":
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "xpu":
                torch.xpu.empty_cache()

if __name__ == "__main__":
    # 2026 Inference Workflow Example
    raw = np.random.randn(512, 512)
    net = MaskGenerator(factor=4)
    
    # Generate high-sensitivity uint8 mask
    result = process(raw, net, return_numpy=True)
    print(f"Dtype: {result.dtype}") # Output: uint8
