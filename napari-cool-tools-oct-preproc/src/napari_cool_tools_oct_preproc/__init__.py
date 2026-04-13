__version__ = "0.0.1"

__all__ = ()

from enum import Enum
import torch

# Enums are a convenient way to get a dropdown menu
class OCTACalc(Enum):
    STD = "Standard Deviation"
    VAR = "Variance"
    VAR2 = "Variance Squared"
    ADA = "Adaptive Decorrelation"
    ADAVAR2 = "Adaptive Decorrelation + Variance Squared"

# Enums are a convenient way to get a dropdown menu
class Operation(Enum):
    """A set of valid arithmetic operations for image_arithmetic."""
    add = torch.add
    subtract = torch.sub
    multiply = torch.mul
    divide = torch.div

# Enums are a convenient way to get a dropdown menu
class ShiftDir(Enum):
    """A set of valid directions for shifting an image."""
    AXIAL = 1
    LATERAL_FAST = 2
    LATERAL_SLOW = 0


# Enums are a convenient way to get a dropdown menu
class SplitMode(Enum):
    DUAL= 2
    QUAD = 4