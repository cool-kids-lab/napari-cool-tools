import os
from pathlib import Path
from typing import List

import numpy as np
from napari.types import FullLayerData
from napari.utils.notifications import show_info
from napari_cool_tools_img_proc._normalization import (
    normalize_data_in_range_func,
)


def float64_file_writer(
    path: str, layer_data: list[FullLayerData]
) -> List[str]:
    """
    Args:
        path(str or list of str): Path to file, or list of paths
    Returns:

    """
    from imageio import imwrite

    if len(layer_data) == 1:
        data, attributes, layer_type = layer_data[0]
        name = attributes["name"]

        show_info(f"path: {path}, shape: {data.shape}, attributes: {name}\n")
        path = Path(path)
        p_dir = Path(path.parent) / path.stem
        ext = path.suffix
        out_path = p_dir / f"{name}{ext}"

        # show_info(f"data type: {data.dtype}\n")

        # normalize data
        norm_data = normalize_data_in_range_func(data, 0.0, 255.0)
        norm_int_data = norm_data.astype(np.uint8)

        # show_info(f"normalized integer data type: {norm_int_data.dtype}\n")

        imwrite(path, norm_int_data, extension=ext)

        show_info(f"Saving {path}")

    else:
        path = Path(path)
        p_dir = Path(path.parent) / path.stem
        ext = path.suffix
        os.makedirs(p_dir, exist_ok=True)

        for layer in layer_data:
            data, attributes, layer_type = layer
            name = attributes["name"]
            out_path = p_dir / f"{name}{ext}"

            show_info(
                f"path: {out_path}, shape: {data.shape}, attributes: {name}\n"
            )

            # show_info(f"data type: {data.dtype}\n")

            # normalize data
            norm_data = normalize_data_in_range_func(data, 0.0, 255.0)
            norm_int_data = norm_data.astype(np.uint8)

            # show_info(f"normalized integer data type: {norm_int_data.dtype}\n")

            imwrite(out_path, norm_int_data, extension=ext)

            show_info(f"Saving {out_path}")

    return [path]
