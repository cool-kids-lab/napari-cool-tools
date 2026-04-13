from napari_cool_tools_oct_preproc._oct_preproc_func import auto_contrast, auto_contrast_split
from napari_cool_tools_oct_preproc._oct_preproc_func import auto_contrast_split_quad, desine, generate_octa
import torch
from napari_cool_tools_io import viewer, device
from napari.layers import Image, Layer, Labels
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_error
from napari_cool_tools_oct_preproc import OCTACalc, SplitMode

import numpy as np
from magicgui import magic_factory

def split_normalize_plugin(
    img: Image,
    mode: SplitMode = SplitMode.DUAL,
    double_side: bool = True,
):
    split_normalize_thread(img=img, mode=mode, double_side=double_side) # type: ignore

@thread_worker(connect={"yielded": viewer.add_layer})
def split_normalize_thread(img: Image, mode: SplitMode = SplitMode.DUAL, double_side: bool = True):
    """"""

    show_info("Starting split and normalize...")

    add_kwargs = {"name": f"{img.name}_normalized"}
    input_data = torch.Tensor(img.data).to(device)

    if double_side:
        input_data[:,:,1::2] = torch.flip(input_data[:,:,1::2], dims=[2])# flip for double side image

    if mode == SplitMode.DUAL:
        for i in range(2):
            data_temp = input_data[:,i::2,:]
            ave = data_temp.mean()
            data_temp = data_temp - ave
            sstd = torch.std(data_temp)
            input_data[:,i::2,:] = data_temp/sstd

    elif mode == SplitMode.QUAD:
        for i in range(4):
            data_temp = input_data[:,i::4,:]
            ave = data_temp.mean()
            data_temp = data_temp - ave
            sstd = torch.std(data_temp)
            input_data[:,i::4,:] = data_temp/sstd


    if double_side:
        input_data[:,:,1::2] = torch.flip(input_data[:,:,1::2], dims=[2])# flip for double side image

    output_data = input_data.cpu().numpy()
    layer = Layer.create(output_data, add_kwargs, "image")

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished split and normalize.")

    yield layer

def auto_contrast_split_plugin(
    img: Image,
    lower_percentileA: float = 1.0,
    upper_percentileA: float = 99.0,
    lower_percentileB: float = 1.0,
    upper_percentileB: float = 99.0,
    num_averages: int = 1,
    double_side: bool = True,
):
    
    auto_contrast_split_thread(
        img=img,
        lower_percentileA=lower_percentileA,
        upper_percentileA=upper_percentileA,
        lower_percentileB=lower_percentileB,
        upper_percentileB=upper_percentileB,
        num_averages=num_averages,
        double_side=double_side,
    ) # type: ignore

@thread_worker(connect={"yielded": viewer.add_layer})    
def auto_contrast_split_thread(
    img: Image,
    lower_percentileA: float = 1.0,
    upper_percentileA: float = 99.0,
    lower_percentileB: float = 1.0,
    upper_percentileB: float = 99.0,
    num_averages: int = 1,
    double_side: bool = True,
):
    """"""

    show_info("Starting auto contrast split...")

    add_kwargs = {"name": f"{img.name}_auto_contrast_split"}
    input_data = torch.Tensor(img.data).to(device)

    if double_side:
        input_data[:,:,1::2] = torch.flip(input_data[:,:,1::2], dims=[2])# flip for double side image

    output_data = auto_contrast_split(
        input_data,
        lower_percentileA=lower_percentileA,
        upper_percentileA=upper_percentileA,
        lower_percentileB=lower_percentileB,
        upper_percentileB=upper_percentileB,
        num_averages=num_averages,
    )

    if double_side:
        output_data[:,:,1::2] = torch.flip(output_data[:,:,1::2], dims=[2])# flip for double side image

    output_data_cpu = output_data.cpu().numpy()
    layer = Layer.create(output_data_cpu, add_kwargs, "image")

    auto_contrast(
        layer,
        lower_percentile=lower_percentileA,
        upper_percentile=upper_percentileA,
        num_averages=num_averages,
    )

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished auto contrast split.")

    yield layer


def auto_contrast_split_quad_plugin(
    img: Image,
    lower_percentileA: float = 1.0,
    upper_percentileA: float = 99.0,
    lower_percentileB: float = 1.0,
    upper_percentileB: float = 99.0,
    lower_percentileC: float = 1.0,
    upper_percentileC: float = 99.0,
    lower_percentileD: float = 1.0,
    upper_percentileD: float = 99.0,
    num_averages: int = 1,
    double_side: bool = True,
):
    auto_contrast_split_quad_thread(
        img,
        lower_percentileA=lower_percentileA,
        upper_percentileA=upper_percentileA,
        lower_percentileB=lower_percentileB,
        upper_percentileB=upper_percentileB,
        lower_percentileC=lower_percentileC,
        upper_percentileC=upper_percentileC,
        lower_percentileD=lower_percentileD,
        upper_percentileD=upper_percentileD,
        num_averages=num_averages,
        double_side=double_side,
    ) # type: ignore


@thread_worker(connect={"yielded": viewer.add_layer})    
def auto_contrast_split_quad_thread(
    img: Image,
    lower_percentileA: float = 1.0,
    upper_percentileA: float = 99.0,
    lower_percentileB: float = 1.0,
    upper_percentileB: float = 99.0,
    lower_percentileC: float = 1.0,
    upper_percentileC: float = 99.0,
    lower_percentileD: float = 1.0,
    upper_percentileD: float = 99.0,
    num_averages: int = 1,
    double_side: bool = True,
):
    """"""

    show_info("Starting auto contrast split...")

    add_kwargs = {"name": f"{img.name}_auto_contrast_quad"}
    input_data = torch.Tensor(img.data).to(device)

    if double_side:
        input_data[:,:,1::2] = torch.flip(input_data[:,:,1::2], dims=[2])# flip for double side image

    output_data = auto_contrast_split_quad(
        input_data,
        lower_percentileA=lower_percentileA,
        upper_percentileA=upper_percentileA,
        lower_percentileB=lower_percentileB,
        upper_percentileB=upper_percentileB,
        lower_percentileC=lower_percentileC,
        upper_percentileC=upper_percentileC,
        lower_percentileD=lower_percentileD,
        upper_percentileD=upper_percentileD,
        num_averages=num_averages,
    )

    if double_side:
        output_data[:,:,1::2] = torch.flip(output_data[:,:,1::2], dims=[2])# flip for double side image

    output_data_cpu = output_data.cpu().numpy()
    layer = Layer.create(output_data_cpu, add_kwargs, "image")

    auto_contrast(
        layer,
        lower_percentile=lower_percentileA,
        upper_percentile=upper_percentileA,
        num_averages=num_averages,
    )

    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished auto contrast split.")

    yield layer

def auto_contrast_plugin(
    img: Image,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    num_averages: int = 1,
):
    auto_contrast(
        img,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        num_averages=num_averages,
    )


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
    output_data_cpu = output_data.cpu().numpy()
    layer = Layer.create(output_data_cpu, add_kwargs, "image")

    # del input_data, output_data
    # Clear cache to free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    show_info("Finished sine unwarping.")

    yield layer


from napari_cool_tools_vol_proc import ProjectionDir, ProjectionType
from napari_cool_tools_vol_proc._projection_tools_funcs import projection

def generate_enface_plugin(
    img: Layer,
    axis: ProjectionDir = ProjectionDir.EN_FACE,
    projection_type: ProjectionType = ProjectionType.MAX,
    desine_unwarp: bool = False,
    crop: int = 0,
):
    
    if img.data.ndim != 3:
        show_error("Input volume must be 3D.")
        return

    generate_enface_thread(img=img, axis=axis, projection_type=projection_type, desine_unwarp=desine_unwarp, crop=crop)


@thread_worker(connect={"yielded": viewer.add_layer})
def generate_enface_thread(
    img: Layer,
    axis: ProjectionDir = ProjectionDir.EN_FACE,
    projection_type: ProjectionType = ProjectionType.MAX,
    desine_unwarp: bool = False,
    crop: int = 0
):    
    if desine_unwarp:
        
        input_data = torch.Tensor(img.data).to(device)
        output_data = desine(input_data, mode="bilinear", transpose=False, scale_fac=2)
        output_data_cpu = output_data.cpu().numpy()
        del input_data, output_data
        # Clear cache to free up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        output_data_cpu = projection(data=output_data_cpu, axis=axis.value, projection_type=projection_type.value, crop=crop)

        axis_suffix = ""
        if axis == ProjectionDir.EN_FACE:
            axis_suffix = "enface"
        elif axis == ProjectionDir.FAST_AXIS:
            axis_suffix = "fast_axis"
        elif axis == ProjectionDir.SLOW_AXIS:
            axis_suffix = "slow_axis"

        add_kwargs = {"name": f"{img.name}_{axis_suffix}"}
        layer = Layer.create(output_data_cpu, add_kwargs, "image")
        vmin, vmax = np.percentile(output_data_cpu, (1, 99))
        layer.contrast_limits = (float(vmin), float(vmax))

        yield layer

    else:
        output_data_cpu = projection(data=img.data, axis=axis.value, projection_type=projection_type.value, crop=crop)

        axis_suffix = ""
        if axis == ProjectionDir.EN_FACE:
            axis_suffix = "enface"
        elif axis == ProjectionDir.FAST_AXIS:
            axis_suffix = "fast_axis"
        elif axis == ProjectionDir.SLOW_AXIS:
            axis_suffix = "slow_axis"

        add_kwargs = {"name": f"{img.name}_{axis_suffix}"}
        layer = Layer.create(output_data_cpu, add_kwargs, "image")
        vmin, vmax = np.percentile(output_data_cpu, (1, 99))
        layer.contrast_limits = (float(vmin), float(vmax))

        yield layer

    

def generate_pseudocolor_enface_plugin(
    img: Layer,
    desine_unwarp: bool = False,
    crop: int = 10, blue_blend: bool = True
):      
    if img.data.ndim != 3:
        show_error("Input volume must be 3D.")
        return

    generate_pseudocolor_enface_thread(img=img, desine_unwarp=desine_unwarp, crop=crop, blue_blend=blue_blend)


@thread_worker(connect={"yielded": viewer.add_layer})
def generate_pseudocolor_enface_thread(img: Layer,
    desine_unwarp: bool = False,
    crop: int = 0, blue_blend: bool = True
    ):
    

    data = img.data

    if desine_unwarp:
        input_data = torch.Tensor(data).to(device)
        output_data = desine(input_data, mode="bilinear", transpose=False, scale_fac=2)
        data = output_data.cpu().numpy()

        del input_data, output_data
        # Clear cache to free up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    output_mean = projection(data=data, axis=ProjectionDir.EN_FACE.value, projection_type=ProjectionType.MEAN.value,crop=crop)
    vmin, vmax = np.percentile(output_mean, (5, 99))
    output_mean = np.clip(output_mean, vmin, vmax)
    output_mean = (output_mean - vmin) / (vmax - vmin)


    # output_max = projection(data=data, axis=ProjectionDir.EN_FACE.value, projection_type=ProjectionType.MAX.value,crop=crop)
    # output_max = np.log10(output_max + 1e-6)  # log scale

    #1 (Choose One option to generate enface max) generate enface max projection by iterative max selection
    output_max = np.zeros_like(output_mean)

    display = data.copy()

    H, D, C = display.shape
    row_idx = np.arange(H)[:, None]
    chan_idx = np.arange(C)[None, :]

    for i in range(20):
        indices = np.argmax(display, axis=1)
        output_max += display[row_idx, indices, chan_idx]
        display[row_idx, indices, chan_idx] = -np.inf
    
    output_max = output_max / 20

    vmin, vmax = np.percentile(output_max, (5, 99))
    output_max = np.clip(output_max, vmin, vmax)
    output_max = (output_max - vmin) / (vmax - vmin)

    if blue_blend:
        #generate pseudo color image blend with blue color blend
        B = output_max*0.4
        G = output_max*0.5 + output_mean*0.2
        R = output_mean
    else:
        # generate pseudo color image blend without blue color blend
        B = np.zeros_like(output_max)
        G = output_max*0.5 + output_mean*0.2
        R = output_mean
        
    output_rgb = np.stack([R, G, B], axis=-1)   # (H, W, 3)

    output_rgb = (output_rgb * 255).astype(np.uint8)


    yield Layer.create(
        output_rgb,
        {
            "name": f"{img.name}_pseudocolor_enface",
            "rgb": True,
        },
        "image",
    )

    # yield Layer.create(output_rgb, {"name": f"{img.name}_pseudocolor_enface"}, "image")


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