import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGenerator(nn.Module):
    """Architecture for high-sensitivity, low-resolution signal expansion."""
    
    def __init__(self, factor: int = 4):
        """Initializes components with specified downscale factor.
        
        Args:
            factor: The downsampling and upsampling ratio.
        """
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1).to(memory_format=torch.channels_last)
        self.pool = nn.MaxPool2d(factor, factor)
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes input to generate an expanded signal mask.
        
        Args:
            x: Input feature tensor.
            
        Returns:
            Binary mask tensor.
        """
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = self.up(x)
        return (torch.sigmoid(x) > 0.5).to(torch.float32)
