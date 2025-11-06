from napari_cool_tools_oct_preproc._oct_preproc_func import desine, generate_octa
import torch
from napari_cool_tools_io import viewer, device
from napari.layers import Image, Layer, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari_cool_tools_oct_preproc import OCTACalc

import numpy as np
from magicgui import magic_factory

def unwarp_sine_plugin(
    img: Image,
    transpose: bool = False,
    interpolation_fac: int = 2,
):
    unwarp_sine_thread(img, transpose=transpose, interpolation_fac=interpolation_fac) # type: ignore

    return

@thread_worker(connect={"yielded": viewer.add_layer})
def unwarp_sine_thread(img: Image, transpose: bool = False, interpolation_fac: int = 2):
    """"""

    show_info("Starting sine unwarping...")

    add_kwargs = {"name": f"{img.name}_unwarped"}
    input_data = torch.Tensor(img.data).to(device)
    output_data = desine(input_data, mode="bilinear", transpose=transpose, scale_fac=interpolation_fac)
    output_data = output_data.cpu().numpy()
    layer = Layer.create(output_data, add_kwargs, "image")

    del input_data, output_data
    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished sine unwarping.")

    yield layer


from napari_cool_tools_vol_proc import ProjectionDir, ProjectionType
from napari_cool_tools_vol_proc._projection_tools import projection_thread

def generate_enface_plugin(
    img: Layer,
    axis: ProjectionDir = ProjectionDir.EN_FACE,
    projection_type: ProjectionType = ProjectionType.MAX,
):

    projection_thread(img=img, axis=axis, projection_type=projection_type)

    return



def generate_octa_plugin(
    img: Layer,
    axis: ProjectionDir = ProjectionDir.EN_FACE,
    projection_type: ProjectionType = ProjectionType.MAX,
):

    projection_thread(img=img, axis=axis, projection_type=projection_type)

    return



def generate_octa_plugin(
    img: Image,
    mscans: int = 3,
    calc: OCTACalc = OCTACalc.STD,
):
    """"""
    generate_octa_thread(img=img, mscans=mscans, calc=calc)
    
    return


@thread_worker(connect={"yielded": viewer.add_layer})
def generate_octa_thread(
    img: Image,
    mscans: int = 3,
    calc: OCTACalc = OCTACalc.STD,
):
    """"""

    show_info("OCTA processing thread started")

    name = f"{img.name}_{calc.name}"
    layer_type = "image"
    add_kwargs = {"name": name}

    out_data = generate_octa(img.data, mscans=mscans, calc=calc)
    out_layer = Layer.create(out_data, add_kwargs, layer_type)
    yield out_layer

    show_info("OCTA processing thread completed")