"""
test bed for new function development
"""


import gc
from itertools import product

from typing import Callable, Iterator, NamedTuple
from reloadr import autoreload
import torch

class PatchIndices(NamedTuple):
    """Container for N-dimensional slice objects."""
    slices: tuple[slice, ...]

def get_accelerator_device() -> torch.device:
    """Detects available hardware using the PyTorch 2.9 Accelerator API."""
    if torch.accelerator.is_available():
        return torch.device(torch.accelerator.current_accelerator().type)
    return torch.device("cpu")

# TODO reimplement this once you upgrade pytorch to a newer version likely 2.9 or 3.0
# def reclaim_memory(device: torch.device) -> None:
#     """
#     Performs strict memory reclamation.
#     Triggers hardware cache clearing only for non-CPU devices.
#     """
#     # Force Python GC to destroy unreachable tensor references first
#     gc.collect()
    
#     # Clear hardware cache via the memory submodule in PyTorch 2.6+
#     if device.type != "cpu" and torch.accelerator.is_available():
#         torch.accelerator.memory.empty_cache()

# TODO replace this with above for now use this janky stuff
def reclaim_memory(device: torch.device) -> None:
    """
    Performs strict memory reclamation.
    Uses device-specific calls as fallbacks for PyTorch 2.7.
    """
    # 1. Always run Python GC to destroy unreachable tensor references
    gc.collect()
    
    # 2. Clear hardware cache only for non-CPU devices
    if device.type == "cpu" or not torch.accelerator.is_available():
        return

    # In PyTorch 2.7, the unified memory submodule does not exist. 
    # We must route to the backend-specific clearing function.
    backend_type = device.type
    if backend_type == "cuda":
        torch.cuda.empty_cache()
    elif backend_type == "mps":
        torch.mps.empty_cache()
    elif backend_type == "xpu":
        torch.xpu.empty_cache()
    # Add other 2.7-supported accelerators as needed (e.g., 'hpu')

def generate_patch_indices(data_shape: tuple[int, ...], patch_size: tuple[int, ...]) -> Iterator[PatchIndices]:
    """Pure generator for N-dimensional coordinate slices."""
    dim_ranges = [range(0, data_shape[i], patch_size[i]) for i in range(len(data_shape))]
    for start_coords in product(*dim_ranges):
        slices = tuple(
            slice(start, min(start + patch_size[i], data_shape[i]))
            for i, start in enumerate(start_coords)
        )
        yield PatchIndices(slices=slices)

def create_patch_processor(
    processor: torch.nn.Module | Callable[[torch.Tensor], torch.Tensor],
    patch_size: tuple[int, ...]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Higher-order function creating a memory-aware processing pipeline."""
    device = get_accelerator_device()
    active_processor = processor.to(device).eval() if isinstance(processor, torch.nn.Module) else processor

    def process(data_cpu: torch.Tensor) -> torch.Tensor:
        if len(patch_size) != data_cpu.ndim:
            raise ValueError(f"Expected {len(patch_size)}D data, got {data_cpu.ndim}D")

        output_cpu = torch.zeros_like(data_cpu)
        for indices in generate_patch_indices(data_cpu.shape, patch_size):
            # Move only the current patch to the accelerator
            patch = data_cpu[indices.slices].to(device)
            
            with torch.inference_mode():
                result = active_processor(patch)
                output_cpu[indices.slices] = result.to("cpu")
            
            # Explicitly delete locals before reclaiming memory
            del patch, result
            reclaim_memory(device)
            
        return output_cpu

    return process

