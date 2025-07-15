__version__ = "0.0.1"

__all__ = ()

from enum import Enum


class ProjectionDir(Enum):
    FAST_AXIS = 0
    EN_FACE = 1
    SLOW_AXIS = 2

class ProjectionType(Enum):
    MAX = 0
    MEAN = 1
    ARGMAX = 2
    MIN = 3
    ARGMIN = 4