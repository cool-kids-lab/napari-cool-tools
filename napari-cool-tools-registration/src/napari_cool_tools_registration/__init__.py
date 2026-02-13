__version__ = "0.0.1"

__all__ = ()
from dataclasses import dataclass
import numpy as np

@dataclass
class CurvCorrectSettings():
    pivot_point:float = 19.2
    imaging_range:float = 12.0
    reference_motor_position:float = 85.0
    imaging_motor_position: float = 85.0
    imaging_motor_position_delta: float = 0.0
    refractive_index: float = 1.33
    scan_angle: float = 100