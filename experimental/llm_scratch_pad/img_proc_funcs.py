"""
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import transforms
import torchvision.transforms.v2 as transforms

def shift_array(arr, dx, dy):
    """
    Shifts a 2D NumPy array by dx pixels horizontally and dy pixels vertically.
    Positive dx shifts right, positive dy shifts down.
    Elements shifted out are replaced with zeros.
    """
    shifted_arr = np.roll(arr, dy, axis=0) # Shift down
    shifted_arr = np.roll(shifted_arr, dx, axis=1) # Shift right

    # Zero out elements that wrapped around
    if dy > 0:
        shifted_arr[:dy, :] = 0
    elif dy < 0:
        shifted_arr[dy:, :] = 0

    if dx > 0:
        shifted_arr[:, :dx] = 0
    elif dx < 0:
        shifted_arr[:, dx:] = 0

    return shifted_arr

class GaussianPyramid(torch.nn.Module):
    """
    Creates a Gaussian pyramid with a specified number of levels.

    The pyramid is a list of tensors, starting with the original image
    and ending with the most downsampled level.

    Args:
        levels (int): The number of pyramid levels to generate, including the
                      original image.
        sigma (float): Standard deviation of the Gaussian kernel for blurring.
        kernel_size (int): Size of the square Gaussian kernel.
    """
    def __init__(self, levels: int, sigma: float = 2.0, kernel_size: int = 5):
        super().__init__()
        if not (isinstance(levels, int) and levels > 0):
            raise ValueError("The number of levels must be a positive integer.")
        self.levels = levels
        self.sigma = sigma
        self.kernel_size = kernel_size
        
        # Kernel size must be odd for Gaussian blur
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): A tensor image of shape (C, H, W).

        Returns:
            list[torch.Tensor]: A list of tensors, where each tensor is a
                                level of the Gaussian pyramid.
        """
        pyramid = [x]
        current_level = x

        for _ in range(self.levels - 1):
            # 1. Blur the image using a Gaussian filter
            blurred_image = transforms.functional.gaussian_blur(
                current_level, 
                kernel_size=self.kernel_size, 
                sigma=self.sigma
            )
            
            # 2. Downsample the image by a factor of 2
            downsampled_image = F.interpolate(
                blurred_image.unsqueeze(0),  # Add a batch dimension for interpolate
                scale_factor=0.5,
                mode='bilinear',
                align_corners=False,
                recompute_scale_factor=False
            ).squeeze(0) # Remove the batch dimension
            
            pyramid.append(downsampled_image)
            current_level = downsampled_image

        return pyramid
    
class MaxPoolUpscaleTransform(nn.Module):
    """
    A PyTorch transform that applies a series of max pooling operations,
    followed by a series of upscaling operations with skip connections.

    This class simulates a simplified U-Net-like architecture without the
    convolutional layers, demonstrating the max-pooling and skip-connection
    logic.

    Args:
        n_levels (int): The number of max pooling and upscaling steps to perform.
                        Must be a positive integer.
        pool_kernel_size (int or tuple): The size of the max pooling window.
                                         Defaults to 2x2.
        upscale_mode (str): The algorithm used for upsampling.
                            Options include 'nearest', 'bilinear', 'bicubic'.
                            Defaults to 'bilinear'.
    """
    def __init__(self, n_levels, pool_kernel_size=2, upscale_mode='bilinear'):
        super().__init__()
        if n_levels < 1:
            raise ValueError("n_levels must be a positive integer.")

        self.n_levels = n_levels
        self.pool_kernel_size = pool_kernel_size
        self.upscale_mode = upscale_mode
        
        self.max_pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size,
                                     stride=self.pool_kernel_size)
        self.upsample = nn.Upsample(scale_factor=self.pool_kernel_size, 
                                    mode=self.upscale_mode, 
                                    align_corners=False if self.upscale_mode != 'nearest' else None)

    def forward(self, x):
        """
        Applies the max-pooling and upscaling transformations.

        Args:
            x (torch.Tensor): The input tensor, with shape [B, C, H, W].

        Returns:
            torch.Tensor: The transformed tensor, with the same dimensions as the input.
        """
        # Store intermediate max-pooling results for skip connections
        skip_connections = [x]
        current_x = x

        # Contracting path (n max pooling operations)
        for _ in range(self.n_levels):
            current_x = self.max_pool(current_x)
            skip_connections.append(current_x)
            
        # Expansive path (n upscaling operations with summation)
        for i in range(self.n_levels):
            # Start from the deepest feature map and work backward
            pooled_result = skip_connections[self.n_levels - i - 1]
            current_x = self.upsample(current_x)
            
            # Pad or crop upscaled tensor to match the size of the pooled tensor
            # The spatial dimensions of the upscaled tensor are doubled, so they should
            # match the corresponding pooled tensor perfectly.
            # Handle potential off-by-one errors from pooling operations
            upscale_H, upscale_W = current_x.shape[2], current_x.shape[3]
            pooled_H, pooled_W = pooled_result.shape[2], pooled_result.shape[3]
            if upscale_H != pooled_H or upscale_W != pooled_W:
                 current_x = F.interpolate(current_x, size=(pooled_H, pooled_W), mode=self.upscale_mode, align_corners=False if self.upscale_mode != 'nearest' else None)

            # Sum with the corresponding skip connection
            current_x = current_x + pooled_result

        return current_x
    
class MaxPoolUpscaleTransformWithThreshold(nn.Module):
    """
    A PyTorch transform performing n max-pooling operations,
    followed by a thresholding operation, and then n upscaling operations
    with skip connections.
    """
    def __init__(self, num_pooling_layers, threshold, kernel_size=2, stride=2, upsample_mode='bilinear'):
        """
        Initializes the MaxPoolUpscaleTransformWithThreshold.

        Args:
            num_pooling_layers (int): The number of max-pooling and upscaling layers.
            threshold (float): The threshold value between 0 and 1 to apply to the
                               image after the last max pool.
            kernel_size (int): The size of the max-pooling window.
            stride (int): The stride of the max-pooling operation.
            upsample_mode (str): The algorithm used for upsampling.
        """
        super().__init__()
        self.num_pooling_layers = num_pooling_layers
        self.threshold = threshold
        
        # Max-pooling layers for the contracting path
        self.max_pool_layers = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=kernel_size, stride=stride) for _ in range(num_pooling_layers)]
        )
        
        # Upsampling layers for the expansive path
        self.upsample_layers = nn.ModuleList(
            [nn.Upsample(scale_factor=stride, mode=upsample_mode, align_corners=False)
             for _ in range(num_pooling_layers)]
        )

    def forward(self, x):
        """
        Defines the forward pass for the transform, including the thresholding step.

        Args:
            x (torch.Tensor): The input tensor, typically an image batch.

        Returns:
            torch.Tensor: The output tensor after all operations.
        """
        # Store intermediate results for skip connections
        skip_connections = []
        
        # Contracting path (max-pooling)
        current_x = x
        for i in range(self.num_pooling_layers):
            skip_connections.append(current_x)
            current_x = self.max_pool_layers[i](current_x)

        # ----- THRESHOLDING STEP -----
        # Apply the threshold to the most downsampled feature map
        # Replace values below the threshold with 0, and others remain unchanged
        current_x = torch.where(current_x > self.threshold, current_x, torch.tensor(0.0, device=current_x.device))
        
        # Expansive path (upscaling with skip connections)
        # We reverse the skip connections to match corresponding layers
        skip_connections = skip_connections[::-1]
        
        upscaled_x = current_x
        for i in range(self.num_pooling_layers):
            upscaled_x = self.upsample_layers[i](upscaled_x)
            
            skip_connection_tensor = skip_connections[i]
            
            # Match dimensions after upsampling if they differ
            diff_h = skip_connection_tensor.size(2) - upscaled_x.size(2)
            diff_w = skip_connection_tensor.size(3) - upscaled_x.size(3)
            
            if diff_h > 0 or diff_w > 0:
                upscaled_x = F.pad(upscaled_x, [diff_w // 2, diff_w - diff_w // 2,
                                                diff_h // 2, diff_h - diff_h // 2])
            elif diff_h < 0 or diff_w < 0:
                upscaled_x = upscaled_x[:, :, :skip_connection_tensor.size(2), :skip_connection_tensor.size(3)]
                
            # Sum the upscaled feature map with the skip connection
            upscaled_x = upscaled_x + skip_connection_tensor
            
        return upscaled_x
    
class MaxPoolUpscaleTransformWithMultipleThresholds(nn.Module):
    """
    A PyTorch transform that performs n max-pooling operations, with thresholding
    before each max-pool, followed by n upscaling operations with skip connections.
    Thresholding also occurs before the first upscaling step.
    """
    def __init__(self, num_pooling_layers, threshold, kernel_size=2, stride=2, upsample_mode='bilinear'):
        """
        Initializes the MaxPoolUpscaleTransformWithMultipleThresholds.

        Args:
            num_pooling_layers (int): The number of max-pooling and upscaling layers.
            threshold (float): The threshold value between 0 and 1.
            kernel_size (int): The size of the max-pooling window.
            stride (int): The stride of the max-pooling operation.
            upsample_mode (str): The algorithm used for upsampling.
        """
        super().__init__()
        self.num_pooling_layers = num_pooling_layers
        
        # Ensure the threshold is within the valid range [0, 1]
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        
        # Max-pooling layers for the contracting path
        self.max_pool_layers = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=kernel_size, stride=stride) for _ in range(num_pooling_layers)]
        )
        
        # Upsampling layers for the expansive path
        self.upsample_layers = nn.ModuleList(
            [nn.Upsample(scale_factor=stride, mode=upsample_mode, align_corners=False)
             for _ in range(num_pooling_layers)]
        )

    def forward(self, x):
        """
        Defines the forward pass for the transform with multiple thresholding steps.

        Args:
            x (torch.Tensor): The input tensor, typically an image batch.

        Returns:
            torch.Tensor: The output tensor after all operations.
        """
        # Store intermediate results for skip connections
        skip_connections = []
        
        # Contracting path (thresholding + max-pooling)
        current_x = x
        for i in range(self.num_pooling_layers):
            # Thresholding before each max-pooling operation
            current_x = (current_x >= self.threshold).float()
            
            skip_connections.append(current_x)
            current_x = self.max_pool_layers[i](current_x)

        # Thresholding before the first upsampling step
        current_x = (current_x >= self.threshold).float()
        
        # Expansive path (upscaling with skip connections)
        # Reverse the skip connections to match corresponding layers
        skip_connections = skip_connections[::-1]
        
        upscaled_x = current_x
        for i in range(self.num_pooling_layers):
            upscaled_x = self.upsample_layers[i](upscaled_x)
            
            skip_connection_tensor = skip_connections[i]
            
            # Match dimensions after upsampling if they differ
            diff_h = skip_connection_tensor.size(2) - upscaled_x.size(2)
            diff_w = skip_connection_tensor.size(3) - upscaled_x.size(3)
            
            if diff_h > 0 or diff_w > 0:
                upscaled_x = F.pad(upscaled_x, [diff_w // 2, diff_w - diff_w // 2,
                                                diff_h // 2, diff_h - diff_h // 2])
            elif diff_h < 0 or diff_w < 0:
                upscaled_x = upscaled_x[:, :, :skip_connection_tensor.size(2), :skip_connection_tensor.size(3)]
                
            # Sum the upscaled feature map with the skip connection
            upscaled_x = upscaled_x + skip_connection_tensor
            
        return upscaled_x


class MaxPoolUpscaleTransformWithSingleThreshold(nn.Module):
    """
    A PyTorch transform performing n max-pooling operations, with a single
    thresholding operation after the final max-pool, followed by n upscaling
    operations with skip connections.
    """
    def __init__(self, num_pooling_layers, threshold, kernel_size=2, stride=2, upsample_mode='bilinear'):
        """
        Initializes the MaxPoolUpscaleTransformWithSingleThreshold.

        Args:
            num_pooling_layers (int): The number of max-pooling and upscaling layers.
            threshold (float): The threshold value between 0 and 1.
            kernel_size (int): The size of the max-pooling window.
            stride (int): The stride of the max-pooling operation.
            upsample_mode (str): The algorithm used for upsampling.
        """
        super().__init__()
        self.num_pooling_layers = num_pooling_layers
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        
        # Max-pooling layers for the contracting path
        self.max_pool_layers = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=kernel_size, stride=stride) for _ in range(num_pooling_layers)]
        )
        
        # Upsampling layers for the expansive path
        self.upsample_layers = nn.ModuleList(
            [nn.Upsample(scale_factor=stride, mode=upsample_mode, align_corners=False)
             for _ in range(num_pooling_layers)]
        )

    def forward(self, x):
        """
        Defines the forward pass for the transform with a single thresholding step.

        Args:
            x (torch.Tensor): The input tensor, typically an image batch.

        Returns:
            torch.Tensor: The output tensor after all operations.
        """
        # Store intermediate results for skip connections
        skip_connections = []
        
        # Contracting path (max-pooling only, no intermediate thresholding)
        current_x = x
        for i in range(self.num_pooling_layers):
            skip_connections.append(current_x)
            current_x = self.max_pool_layers[i](current_x)

        # ----- SINGLE THRESHOLDING STEP -----
        # Binarize the most downsampled feature map
        current_x = (current_x >= self.threshold).float()
        
        # Expansive path (upscaling with skip connections)
        skip_connections = skip_connections[::-1]
        
        upscaled_x = current_x
        for i in range(self.num_pooling_layers):
            upscaled_x = self.upsample_layers[i](upscaled_x)
            
            skip_connection_tensor = skip_connections[i]
            
            # Match dimensions after upsampling if they differ
            diff_h = skip_connection_tensor.size(2) - upscaled_x.size(2)
            diff_w = skip_connection_tensor.size(3) - upscaled_x.size(3)
            
            if diff_h > 0 or diff_w > 0:
                upscaled_x = F.pad(upscaled_x, [diff_w // 2, diff_w - diff_w // 2,
                                                diff_h // 2, diff_h - diff_h // 2])
            elif diff_h < 0 or diff_w < 0:
                upscaled_x = upscaled_x[:, :, :skip_connection_tensor.size(2), :skip_connection_tensor.size(3)]
                
            # Sum the upscaled feature map with the skip connection
            upscaled_x = upscaled_x + skip_connection_tensor
            
        return upscaled_x