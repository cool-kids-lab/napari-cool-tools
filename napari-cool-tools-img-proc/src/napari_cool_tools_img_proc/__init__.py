__version__ = "0.0.1"

__all__ = (
    )

from enum import Enum

class DType(Enum):
    NP_FLOAT = 'f4' #np.dtype(np.float64)
    NP_UINT8 = 'u1' #np.dtype(np.uint8)