""""""

import numpy as np
from napari_cool_tools_oct_preproc import EnfaceAccumulation
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import (
    return_enface_accumulation,
)
from napari_cool_tools_segmentation import EnfaceSegmentationType
from napari_cool_tools_segmentation._segmentation_funcs import enface_onnx_seg_func
from numpy.typing import NDArray


def align_enface_with_bscan(data):
    aligned = data.copy().transpose(-1, -2)[:, None, :]
    return aligned


def find_first_occurrence_1d_from_sentinel(arr: NDArray):
    """ """
    target_value = arr[0]
    search_arr = arr[1:]

    idx = (search_arr == target_value).argmax()
    out = np.zeros_like(search_arr)
    # limit = len(search_arr)
    if idx > 0:
        start = idx
        # stop = limit
        out[start : start + 1] = 1
    # else:
    #    out[0] = 1

    return out.astype(bool)


def correlate_mip_with_volume(volume_data: NDArray, label_val: int = 10, feature_type:EnfaceSegmentationType=EnfaceSegmentationType.VESSEL):
    """"""
    mip = return_enface_accumulation(
        volume_data, accumulation_type=EnfaceAccumulation.MAX
    )
    mip_y = return_enface_accumulation(
        volume_data, accumulation_type=EnfaceAccumulation.ARGMAX
    )
    vessel_mask = enface_onnx_seg_func(
        mip, onnx_path=feature_type.value, DoG=True
    ).astype(bool)
    x_coord, z_coord = vessel_mask.nonzero()
    y_coord = mip_y[(x_coord, z_coord)]
    vessel_indxs = (x_coord, y_coord, z_coord)

    feature_label_data = np.zeros_like(volume_data).astype(np.uint8)
    feature_label_data[vessel_indxs] = label_val

    return feature_label_data
