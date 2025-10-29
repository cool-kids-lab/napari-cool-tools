from napari.layers import Image, Layer, Labels
from napari.qt.threading import thread_worker
from napari_cool_tools_io import viewer, device
from napari.utils.notifications import show_info
from napari_cool_tools_oct_preproc import Operation
import torch

def image_arithmetic_plugin(
    layerA: Image,
    operation: Operation,
    layerB: Image,
):
    """Adds, subtracts, multiplies, or divides two same-shaped image layers."""

    image_arithmetic_thread(layerA, operation, layerB)  # type: ignore

    return

@thread_worker(connect={"yielded": viewer.add_layer})
def image_arithmetic_thread(
    layerA: Image,
    operation: Operation,
    layerB: Image,
):
    """Adds, subtracts, multiplies, or divides two same-shaped image layers."""

    name = "image_operation_"+operation.value.__name__
    layer_type = "image"
    add_kwargs = {"name": name}

    if layerA.data.shape != layerB.data.shape:
        show_info("Image Arithmetic Failed: Input layers must have the same shape.")
        return
    
    dataA = torch.Tensor(layerA.data).to(device)
    dataB = torch.Tensor(layerB.data).to(device)

    out_data = operation.value(dataA, dataB)

    out_image = Layer.create(out_data.cpu().numpy(), add_kwargs, layer_type)

    del dataA, dataB, out_data
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    yield out_image