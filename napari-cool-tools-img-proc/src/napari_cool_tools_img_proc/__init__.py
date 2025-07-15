__version__ = "0.0.1"

__all__ = ()

from enum import Enum


class DType(Enum):
    NP_UINT8 = "u1"
    NP_FLOAT16 = "f2"
    NP_FLOAT32 = "f4"
    NP_FLOAT64 = "f8"

