__version__ = "0.0.1"

__all__ = (
    )

import napari
import torch
import kornia

viewer = napari.current_viewer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
