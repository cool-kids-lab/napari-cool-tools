import torch
import pytest
from current_dev import create_patch_processor, generate_patch_indices

# --- Mock Processors ---

def identity_processor(x: torch.Tensor) -> torch.Tensor:
    """Returns the input unchanged for integrity testing."""
    return x

def multiplication_processor(x: torch.Tensor) -> torch.Tensor:
    """Applies a transformation to check logic application."""
    return x * 2

# --- Test Cases ---

def test_generate_patch_indices_coverage():
    """Verify coordinate slices cover the entire tensor, including partial edges."""
    shape = (10, 10)
    patch_size = (4, 4)
    indices = list(generate_patch_indices(shape, patch_size))
    
    # 10x10 with 4x4 patches requires a 3x3 grid (9 patches total)
    assert len(indices) == 9
    
    # Verify the final corner patch is truncated correctly to the tensor boundary
    last_patch_slices = indices[-1].slices
    assert last_patch_slices[0].stop == 10
    assert last_patch_slices[1].stop == 10

def test_2d_data_integrity():
    """Test that 2D data is reconstructed perfectly using identity mapping."""
    data = torch.randn(100, 100)
    process = create_patch_processor(identity_processor, patch_size=(30, 30))
    
    result = process(data)
    
    # PyTorch 2.9 standard for close comparison
    torch.testing.assert_close(result, data)

def test_3d_transformation():
    """Test N-dimensional (3D) processing with non-uniform patches and a function."""
    data = torch.ones(32, 32, 32)
    process = create_patch_processor(multiplication_processor, patch_size=(8, 16, 8))
    
    result = process(data)
    
    assert result.shape == (32, 32, 32)
    assert torch.all(result == 2.0)

def test_nn_module_support():
    """Verify that the processor correctly handles torch.nn.Module inputs."""
    class SimpleModule(torch.nn.Module):
        def forward(self, x): return x + 1
        
    model = SimpleModule()
    process = create_patch_processor(model, patch_size=(5, 5))
    
    data = torch.zeros(10, 10)
    result = process(data)
    
    assert torch.all(result == 1.0)

def test_dimension_mismatch_raises_error():
    """Ensure the processor catches incorrect input dimensions early."""
    data_2d = torch.randn(10, 10)
    # Processor configured for 3D
    process = create_patch_processor(identity_processor, patch_size=(5, 5, 5))
    
    with pytest.raises(ValueError, match="Expected 3D data"):
        process(data_2d)

@pytest.mark.skipif(not torch.accelerator.is_available(), reason="No accelerator found")
def test_accelerator_workflow():
    """Verify hardware-specific paths: execution on device and memory reclaim."""
    data = torch.randn(20, 20)
    process = create_patch_processor(identity_processor, patch_size=(10, 10))
    
    result = process(data)
    
    # Assert result is moved back to CPU host
    assert result.device.type == "cpu"
    # Verify the reclaim logic (empty_cache) doesn't crash on hardware
    # Memory state is difficult to assert, so we verify error-free execution