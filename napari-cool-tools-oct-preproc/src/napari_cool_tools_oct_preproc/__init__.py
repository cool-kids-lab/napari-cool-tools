__version__ = "0.0.1"

__all__ = ()

from enum import Enum


class Preproc(Enum):
    NLCGbBb = "Norm_Log_CLAHE_Gblur_Bblur"
    SNLC = "Stand_Nom_Log_CLAHE"
    SNL = "Stand_Norm_Log"
    SN = "Stand_Norm"
    CCL = "Conditional_CLAHE_Log"
    RRAR = "Random_Resized_Aspect_Ratio"


class Augmentation(Enum):
    RandCropResizeAspectRat = "Random_Crop_Resized_Aspect_Ratio"


class OCTACalc(Enum):
    STD = "Standard Deviation"
    VAR = "Variance"
    VAR2 = "Variance Squared"


class EnfaceAccumulation(Enum):
    MAX = 0
    MEAN = 1
    MIN = 2
