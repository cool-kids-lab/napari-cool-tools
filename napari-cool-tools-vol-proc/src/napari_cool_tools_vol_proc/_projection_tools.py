"""
This module contains code for calculating and manipulating projections of volumetric data.
"""

from typing import Generator

from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari_cool_tools_io import viewer

from napari_cool_tools_vol_proc import ProjectionDir, ProjectionType


def projection_plugin(
    img: Layer,
    axis: ProjectionDir = ProjectionDir.EN_FACE,
    projection_type: ProjectionType = ProjectionType.MAX,
    crop: int = 10,
):
    """Generate projection along selected axis from structural OCT data.

    Args:
        img: Napari Layer containing 3D structural OCT data
        axis: Axis along which to project EN_FACE,FAST_AXIS,SLOW_AXIS
        projection_type: MAX,MEAN,ARGMAX,MIN,ARGMIN

    Returns:
        None

    Raises:
        ValueError: If data dimension is not == 3
    """

    projection_thread(img=img, axis=axis, projection_type=projection_type, crop=crop)

    return


@thread_worker(connect={"returned": viewer.add_layer})
def projection_thread(
    img: Layer,
    axis: ProjectionDir = ProjectionDir.EN_FACE,
    projection_type: ProjectionType = ProjectionType.MAX,
    crop: int = 0,
):
    """Generate projection along selected axis from structural OCT data.

    Args:
        img: Napari Layer containing 3D structural OCT data
        axis: Axis along which to project EN_FACE,FAST_AXIS,SLOW_AXIS
        projection_type: MAX,MEAN,ARGMAX,MIN,ARGMIN

    Returns:
        Layer containing selected projection

    Raises:
        ValueError: If data dimension is not == 3
    """
    from napari_cool_tools_vol_proc._projection_tools_funcs import projection

    data = img.data
    name = img.name
    axis_suffix = ""
    if axis == ProjectionDir.EN_FACE:
        axis_suffix = "enface"
    elif axis == ProjectionDir.FAST_AXIS:
        axis_suffix = "fast_axis"
    elif axis == ProjectionDir.SLOW_AXIS:
        axis_suffix = "slow_axis"

    add_kwargs = {"name": f"{name}_{axis_suffix}"}
    layer_type = img.as_layer_data_tuple()[2]

    show_info(f"{axis_suffix.capitalize()} Projection thread has started")
    data = projection(data=data, axis=axis.value, projection_type=projection_type.value, crop=crop)
    layer = Layer.create(data, add_kwargs, layer_type)

    show_info(f"{axis_suffix.capitalize()} Projection thread has completed")

    return layer