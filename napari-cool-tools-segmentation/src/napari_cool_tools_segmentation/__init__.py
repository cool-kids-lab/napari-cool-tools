__version__ = "0.0.1"

__all__ = ()

from enum import Enum
from pathlib import Path

import napari
import torch

viewer = napari.current_viewer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

this_file_path = Path(__file__)
onnx_folder_parent_path = this_file_path.parents[3]
onnx_bscan_path = onnx_folder_parent_path / "onnx_models/bscan/"
onnx_bscan_retina_seg_path = onnx_bscan_path / "retina"
onnx_bscan_deconjugate_path = onnx_bscan_path / "deconjugate"

onnx_enface_path = onnx_folder_parent_path / "onnx_models/enface"
onnx_enface_vessels_path = onnx_enface_path / "vessels"
onnx_enface_optic_nerve_path = onnx_enface_path / "optic_nerve"
onnx_enface_ridge_path = onnx_enface_path / "ridge"

onnx_bscan_retina_seg = list(onnx_bscan_retina_seg_path.rglob("*.onnx"))[0]
onnx_bscan_deconjugate = list(onnx_bscan_deconjugate_path.rglob("*.onnx"))[0]

onnx_enface_vessels = list(onnx_enface_vessels_path.rglob("*.onnx"))[0]
onnx_enface_optic_nerve = list(onnx_enface_optic_nerve_path.rglob("*.onnx"))[0]
onnx_enface_ridge = list(onnx_enface_ridge_path.rglob("*.onnx"))[0]


class BscanSegmentationType(Enum):
    RETINASEG = onnx_bscan_retina_seg
    DECONJUGATE = onnx_bscan_deconjugate


class EnfaceSegmentationType(Enum):
    VESSEL = onnx_enface_vessels
    OPTICNERVEHEAD = onnx_enface_optic_nerve
    RIDGE = onnx_enface_ridge
