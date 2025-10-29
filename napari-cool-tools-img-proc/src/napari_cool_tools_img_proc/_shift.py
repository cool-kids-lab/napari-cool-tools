from magicgui import magic_factory
from napari.layers import Image, Layer, Labels
import torch
import numpy as np
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer, device
from napari_cool_tools_oct_preproc import ShiftDir


@magic_factory(
    shift_value=dict(widget_type="SpinBox", value=0, min=-100000, max=100000),
    )
def shift_image_plugin(
    image : Layer,
    shift_value: int,
    axis: ShiftDir,
):
    shift_image_thread(image, shift_value, axis.value)  # type: ignore
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def shift_image_thread(
        image : Layer,
        shift_value: int  = 0,
        axis: int = 0,
    ):
    
    shifted_image = np.roll(image.data,shift=shift_value,axis=axis)

    if isinstance(image, Image):
        name = image.name+"_shifted"
        layer_type = "image"
        add_kwargs = {"name": name}
        yield Layer.create(shifted_image, add_kwargs, layer_type)

    if isinstance(image, Labels):
        name = image.name+"_shifted"
        layer_type = "labels"
        add_kwargs = {"name": name}
        yield Layer.create(shifted_image, add_kwargs, layer_type)
