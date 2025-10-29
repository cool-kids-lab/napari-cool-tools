from napari.layers import Image, Layer, Labels
import numpy as np
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer
from napari.utils.notifications import show_info

def generate_logscaled_image_plugin(
    img: Image,
    log_gain: float = 1.0,
):
    """Generates a logscaled version of the input image.

    Args:
        img (Image): Input image layer.
        log_gain (float): Gain factor for log scaling.

    Returns:
        Logscaled image layer.
    """
    generate_logscaled_image_thread(img=img, log_gain=log_gain)

    return

@thread_worker(connect={"yielded": viewer.add_layer})
def generate_logscaled_image_thread(
    img: Image,
    log_gain: float = 1.0,
):
    show_info("Generating logscaled image")

    name = f"{img.name}_logscaled"

    output_image = log_gain * np.log10(img.data)

    add_kwargs = {"name": name}
    layer_type = "image"
    layer = Layer.create(output_image, add_kwargs, layer_type)
    yield layer

    show_info("Logscaled image thread has completed")

