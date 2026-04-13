__version__ = "0.0.1"

__all__ = ()

from enum import Enum
import numpy as np

# Enums are a convenient way to get a dropdown menu
class cast_dtype(Enum):
    uint16= np.uint16
    uint8 = np.uint8