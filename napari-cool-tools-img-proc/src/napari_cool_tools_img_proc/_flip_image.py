from napari_cool_tools_oct_preproc import ShiftDir
import numpy as np
from napari.layers import Layer

def flip_image_plugin(
    img: Layer,
    direction: ShiftDir,
):

    if img.data.ndim < 3 and direction.value == 2:
        return  # No lateral axis to flip

    img.data = np.flip(img.data, axis=direction.value).copy()

    return