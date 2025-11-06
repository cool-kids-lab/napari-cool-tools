"""
This module contains code for calculating and manipulating projections of volumetric data.
"""

from napari.types import ImageData
from napari_cool_tools_vol_proc import ProjectionDir, ProjectionType
import numpy as np

def projection(
    data: ImageData,
    projection_type: int = ProjectionType.MAX.value,
    axis: int = ProjectionDir.EN_FACE.value,
    crop: int = 0,
) -> ImageData: # type: ignore
    """Generate projection along selected axis from structural OCT data.

    Args:
        img: Napari Layer containing 3D structural OCT data
        axis: Axis along which to project EN_FACE,FAST_AXIS,SLOW_AXIS = 0,1,2
        projection_type: MAX,MEAN,ARGMAX,MIN,ARGMIN = 0,1,2,3,4

    Returns:
        ImageData containing selected projection

    Raises:
        ValueError: If data dimension is not == 3
    """

    assert data.ndim == 3, ValueError(
        f"Input has {data.ndim} dimensions but this function requires 3 dimensions."
    )

    if crop > 0 and axis == ProjectionDir.EN_FACE.value:
        data = data[:,crop:-crop,:] # type: ignore
        
    if projection_type == ProjectionType.MAX.value:
        return data.max(axis=axis)
    if projection_type == ProjectionType.MEAN.value:
        return data.mean(axis=axis)
    if projection_type == ProjectionType.ARGMAX.value:
        return data.argmax(axis=axis)
    if projection_type == ProjectionType.MIN.value:
        return data.argmin(axis=axis)
    if projection_type == ProjectionType.ARGMIN.value:
        return data.min(axis=axis)
