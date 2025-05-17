"""
This module contains code for OCT data preprocessing.
"""
from typing import List, Generator
import skimage.transform
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari_cool_tools_io import viewer
from napari.layers import Image, Layer
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,memory_stats
from napari_cool_tools_oct_preproc._oct_preproc_utils_funcs import preproc_bscan, data_augmentation, generate_octa, generate_enface, Preproc, Augmentation, OCTACalc
import scipy
import skimage

def resize_image_plugin(
    img:Image,
    xscale:float=1,
    yscale:float=1,
    zscale:float=1,
):
    """resize OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        xscale (float): scale in x direction
        yscale (float): scale in y direction
        zscale (float): scale in z direction

    Returns:
        Resized image

    """


    resize_image_thread(
            img=img,
        xscale=xscale,
        yscale=yscale,
        zscale=zscale,
    )

    return


@thread_worker(connect={"yielded": viewer.add_layer})
def resize_image_thread(
    img:Image,
    xscale:float=1,
    yscale:float=1,
    zscale:float=1,
)-> Generator[Layer,Layer,Layer]:   
    
    show_info("Resizing image")

    if len(img.data.shape) == 3:
        name = f"{img.name}_resized"
        layer_type = "image"

        # output_image = scipy.ndimage.zoom(img.data, [1/xscale, 1/yscale, 1/zscale]);
        output_image = skimage.transform.rescale(img.data, [xscale, yscale, zscale], anti_aliasing=False, order = 1)
        
    #   output = resize_image(
    #       img=img,
    #       xscale=xscale,
    #       yscale=yscale,
    #       zscale=zscale,
    #   )

        add_kwargs = {"name": f"{name}"}
        layer = Layer.create(output_image,add_kwargs,layer_type)
        yield layer

        show_info(f'Resizing image thread has completed')

    else:
        show_info(f'Resizing image thread Failed, Must be a volume!!!')

    

def generate_enface_plugin(
    img:Image,
    sin_correct:bool=True,
    exp:bool=False,
    n:float=2,
    CLAHE:bool=False,
    clahe_clip:float=2.5,
    log_correct:bool=True,
    log_gain:float=1.0,
    band_pass_filter:bool=False,
    bp_low:float=0.5,
    bp_high:float=6.0,
    bp_pt:bool=True,
    debug:bool=False,
):
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Returns:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    generate_enface_thread(
        img=img,
        sin_correct=sin_correct,
        exp=exp,
        n=n,
        CLAHE=CLAHE,
        clahe_clip=clahe_clip,
        log_correct=log_correct,
        log_gain=log_gain,
        band_pass_filter=band_pass_filter,
        bp_low=bp_low,
        bp_high=bp_high,
        bp_pt=bp_pt,
        debug=debug,
    )
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def generate_enface_thread(
    img:Image,
    sin_correct:bool=True,
    exp:bool=False,
    n:float=2,
    CLAHE:bool=False,
    clahe_clip:float=2.5,
    log_correct:bool=True,
    log_gain:float=1.0,
    band_pass_filter:bool=False,
    bp_low:float=0.5,
    bp_high:float=6.0,
    bp_pt:bool=True,
    debug:bool=False,
)-> Generator[Layer,Layer,Layer]:
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Yields:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    show_info(f'Generate enface image thread has started')
    
    name = f"{img.name}_enface"
    layer_type = "image"
    
    output = generate_enface(
        data=img.data,
        sin_correct=sin_correct,
        exp=exp,
        n=n,
        CLAHE=CLAHE,
        clahe_clip=clahe_clip,
        log_correct=log_correct,
        log_gain=log_gain,
        band_pass_filter=band_pass_filter,
        bp_low=bp_low,
        bp_high=bp_high,
        bp_pt=bp_pt,
        debug=debug,
    )

    for output in output:
        out_data = output[0]
        suffix = output[1]

        add_kwargs = {"name": f"{name}_{suffix}"}
        layer = Layer.create(out_data,add_kwargs,layer_type)
        yield layer

    show_info(f'Generate enface image thread has completed')


def generate_enface_image(vol:Image, debug=False, sin_correct=True, log_correct=True, band_pass_filter=True, CLAHE=True):
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Returns:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    generate_enface_image_thread(vol=vol,debug=debug,sin_correct=sin_correct,log_correct=log_correct,band_pass_filter=band_pass_filter,CLAHE=CLAHE)
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def generate_enface_image_thread(vol:Image, debug=False, sin_correct=True, log_correct=True, band_pass_filter=True, CLAHE=True)-> Generator[Layer,Layer,Layer]:
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Yields:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    show_info(f'Generate enface image thread has started')
    layers = generate_enface_image_func(vol=vol,debug=debug,sin_correct=sin_correct,log_correct=log_correct,band_pass_filter=band_pass_filter,CLAHE=CLAHE)
    for layer in layers:
        yield layer
    show_info(f'Generate enface image thread has completed')


def generate_enface_image_func(vol:Image, debug=False, sin_correct=True, log_correct=True, band_pass_filter=True, CLAHE=True)-> List[Layer]:
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Yields:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """

    from napari_cool_tools_registration._registration_tools import a_scan_correction_func, a_scan_reg_subpix_gen, a_scan_reg_calc_settings_func
    from napari_cool_tools_img_proc._normalization import normalize_data_in_range_pt_func
    from napari_cool_tools_img_proc._denoise import diff_of_gaus_func
    from napari_cool_tools_img_proc._equalization import clahe_pt_func
    from napari_cool_tools_img_proc._luminance import adjust_log_pt_func

    layers = []

    #show_info(f'Generate enface image thread has started')
    data = vol.data
    name = f"Enface_{vol.name}"
    layer_type = "image"

    show_info(f'Generating initial enface MIP')
    yx = data.transpose(1,2,0)
    enface_mip = yx.max(0)
    print(f"enface_mip shape: {enface_mip.shape}\n")

    if debug:
        add_kwargs = {"name": f"init_MIP_{name}"}
        layer = Layer.create(enface_mip,add_kwargs,layer_type)
        layers.append(layer)
        #yield layer

    correct_mip = enface_mip.copy()
    correct_mip.shape = (correct_mip.shape[0],1,correct_mip.shape[1])
    correct_mip = correct_mip.transpose(2,1,0)
    print(f"correct_mip shape: {correct_mip.shape}\n")

    add_kwargs = {"name": f"corrected_MIP_{name}"}
    layer = Layer.create(correct_mip,add_kwargs,layer_type)
    
    if debug:
        layers.append(layer)
        #yield layer

    if sin_correct:
        show_info(f'Correcting enface MIP distortion')
        correct_mip_layer = a_scan_correction_func(layer)
    else:
        correct_mip_layer = layer
    
    if debug:
        layers.append(correct_mip_layer)
        #yield correct_mip_layer
    
    show_info(f'Calculating Optimal Subregions for subpixel registration')
    settings = a_scan_reg_calc_settings_func(correct_mip_layer)
    if debug:
        show_info(f"{settings['region_num']}")

    show_info(f'Completing subpixel registration of A-scans')
    outs = list(a_scan_reg_subpix_gen(correct_mip_layer,settings))
    for i,out in enumerate(outs):
        if  i == len(outs) -1:

            out.data = normalize_data_in_range_pt_func(out.data,0,1)

            if log_correct:
                out = adjust_log_pt_func(out,2.5)
            if CLAHE:
                out.data = out.data.squeeze()
                out = clahe_pt_func(out)
                out.data.shape = (out.data.shape[0],1,out.data.shape[1])
            if band_pass_filter:
                out = diff_of_gaus_func(out,0.6,6.0)

            if debug:
                layers.append(out)
                #yield out
            else:
                out.data = out.data.squeeze()
                out.data = out.data.transpose(1,0)
                layers.append(out)
                #yield out
        else:
            if debug:
                layers.append(out)
                #yield out
            else:
                pass
            pass

    #show_info(f'Generate enface image thread has completed')
    return layers

def process_bscan_preset(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False, Med:bool=False):
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)"""
    
    process_bscan_preset_thread(vol=vol,ascan_corr=ascan_corr,Bandpass=Bandpass,CLAHE=CLAHE,Med=Med)
    return 

@thread_worker(connect={"returned": viewer.add_layer})
def process_bscan_preset_thread(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False, Med:bool=False)->Layer:
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)"""
    
    show_info(f"B-scan preset thread started")
    output = process_bscan_preset_func(vol=vol,ascan_corr=ascan_corr,Bandpass=Bandpass,CLAHE=CLAHE,Med=Med)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f"B-scan preset thread completed")
    return output

def process_bscan_preset_func(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False, Med:bool=False)->Layer:
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)
    """
    from napari_cool_tools_img_proc._normalization import normalize_in_range_func, normalize_in_range_pt_func
    from napari_cool_tools_img_proc._denoise_funcs import diff_of_gaus_func
    from napari_cool_tools_img_proc._equalization import clahe_pt_func
    from napari_cool_tools_img_proc._luminance import adjust_log_pt_func
    from napari_cool_tools_img_proc._filters import filter_bilateral_pt_func, filter_median_pt_func
    from napari_cool_tools_vol_proc._averaging_tools import average_per_bscan
    from napari_cool_tools_registration._registration_tools_funcs import a_scan_correction_func    
    
    #out = normalize_in_range_pt_func(vol,0,1) # add flag and refactor function
    out = normalize_in_range_func(vol,0,1)

    #torch.cuda.empty_cache()

    out = adjust_log_pt_func(out,2.5)
    torch.cuda.empty_cache()
    
    if ascan_corr:
        out = a_scan_correction_func(out)
        torch.cuda.empty_cache()
    if Bandpass:
        out = diff_of_gaus_func(out,1.6,20)
        torch.cuda.empty_cache()
    if CLAHE:
        out = clahe_pt_func(out,1)
        torch.cuda.empty_cache()

    out = normalize_in_range_pt_func(out,0,1)
    torch.cuda.empty_cache()

    out = filter_bilateral_pt_func(out)
    torch.cuda.empty_cache()

    if Med:
        out = filter_median_pt_func(out)
        torch.cuda.empty_cache()

    out = adjust_log_pt_func(out,1.5)
    torch.cuda.empty_cache()

    out = average_per_bscan(out)

    name = f"{out.name}_proc"
    layer_type = 'image'
    add_kwargs = {"name":name}
    out_image = Layer.create(out.data,add_kwargs,layer_type)
    return out_image

def data_augmentation_plugin(vol:Layer, 
                       transform:Augmentation=Augmentation.RandCropResizeAspectRat,
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       log_cor:bool=False,
                       vol_proc:bool = False,
                       chunk_shuff:bool = True,
                       debug:bool = False,
                       gpu:bool= True,
                       chunk_size:int = 1,
                       fov:int = 116,
                       log_gain = 2.5, 
                       clahe_clip_limit=1.0,
                       b_blur_ks = (3,3),
                       b_blur_sc = 0.1,
                       b_blur_ss = (1.0,1.0),
                       b_blur_bt = 'reflect',
                       g_blur_ks = (3,3),
                       g_blur_s = (1.0,1.0),
                       g_blur_bt = 'reflect',
                       ):
    """"""
    data_augmentation_thread(
        vol,
        transform=transform,
        ascan_corr=ascan_corr,
        Bandpass=Bandpass,
        log_cor=log_cor,
        vol_proc=vol_proc,
        chunk_shuff=chunk_shuff,
        gpu=gpu,
        debug=debug,
        chunk_size=chunk_size,
        fov=fov,
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
        b_blur_ks=b_blur_ks,
        b_blur_sc=b_blur_sc,
        b_blur_ss=b_blur_ss,
        b_blur_bt=b_blur_bt,
        g_blur_ks=g_blur_ks,
        g_blur_s=g_blur_s,
        g_blur_bt=g_blur_bt
    )

    return

@thread_worker(connect={"returned": viewer.add_layer})
def data_augmentation_thread(vol:Layer, 
                       transform:Augmentation=Augmentation.RandCropResizeAspectRat,
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       log_cor:bool=False,
                       vol_proc:bool = False,
                       chunk_shuff:bool = True,
                       debug:bool = False,
                       gpu:bool=True, 
                       chunk_size:int = 1,
                       fov:int = 116,
                       log_gain = 2.5, 
                       clahe_clip_limit=1.0,
                       b_blur_ks = (3,3),
                       b_blur_sc = 0.1,
                       b_blur_ss = (1.0,1.0),
                       b_blur_bt = 'reflect',
                       g_blur_ks = (3,3),
                       g_blur_s = (1.0,1.0),
                       g_blur_bt = 'reflect',
                       )->ImageData:
    """"""
    from napari_cool_tools_io import device

    show_info(f"B-scan augmentation thread started")

    current_device = 'cpu'
    

    if debug:
        print(f"\ncurrent device is: {current_device}\n")

    print(f"\ngpu_flag is: {gpu}\n")

    #if gpu and not vol_proc:
    #    current_device = device
    #else:
    #    pass

    if gpu:
        current_device = device

    print(f"\ncurrent device is: {current_device}\n")    

    out = data_augmentation(
        vol.data,
        transform=transform,
        ascan_corr=ascan_corr,
        Bandpass=Bandpass,
        log_cor=log_cor,
        vol_proc=vol_proc,
        chunk_shuff=chunk_shuff,
        debug=debug,
        processor=current_device,
        chunk_size=chunk_size,
        fov=fov,
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
        b_blur_ks=b_blur_ks,
        b_blur_sc=b_blur_sc,
        b_blur_ss=b_blur_ss,
        b_blur_bt=b_blur_bt,
        g_blur_ks=g_blur_ks,
        g_blur_s=g_blur_s,
        g_blur_bt=g_blur_bt
    )

    torch.cuda.empty_cache()
    memory_stats()

    if vol_proc:
        name = f"{vol.name}_vol_{transform.name}"
    else:
        if chunk_size > 1:
            name = f"{vol.name}_{transform.name}({chunk_size})"
        else:
            name = f"{vol.name}_{transform.name}"

    layer_type = 'image'
    add_kwargs = {"name":name}
    out_layer = Layer.create(out,add_kwargs,layer_type)

    show_info(f"B-scan augmentaion thread completed")

    return out_layer


def preproc_bscan_plugin(vol:Image,
                       transform:Preproc=Preproc.NLCGbBb,
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       log_cor:bool=False,
                       vol_proc:bool = False,
                       chunk_shuff:bool = True,
                       gpu:bool= True,
                       debug:bool= False,
                       chunk_size:int = 1,
                       fov:int = 116,
                       log_gain = 2.5, 
                       clahe_clip_limit=1.0,
                       b_blur_ks = (3,3),
                       b_blur_sc = 0.1,
                       b_blur_ss = (1.0,1.0),
                       b_blur_bt = 'reflect',
                       g_blur_ks = (3,3),
                       g_blur_s = (1.0,1.0),
                       g_blur_bt = 'reflect',
                       )->Layer:
    """"""
    preproc_bscan_thread(
        vol,
        transform=transform,
        ascan_corr=ascan_corr,
        Bandpass=Bandpass,
        log_cor=log_cor,
        vol_proc=vol_proc,
        chunk_shuff=chunk_shuff,
        gpu=gpu,
        debug=debug,
        chunk_size=chunk_size,
        fov=fov,
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
        b_blur_ks=b_blur_ks,
        b_blur_sc=b_blur_sc,
        b_blur_ss=b_blur_ss,
        b_blur_bt=b_blur_bt,
        g_blur_ks=g_blur_ks,
        g_blur_s=g_blur_s,
        g_blur_bt=g_blur_bt
    )

    return

@thread_worker(connect={"returned": viewer.add_layer})
def preproc_bscan_thread(vol:Image, 
                       transform = Preproc.NLCGbBb,
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       log_cor:bool=False,
                       vol_proc:bool = False,
                       chunk_shuff:bool = True,
                       gpu:bool= True,
                       debug:bool= False, 
                       chunk_size:int = 1,
                       fov:int = 116,
                       log_gain = 2.5, 
                       clahe_clip_limit=1.0,
                       b_blur_ks = (3,3),
                       b_blur_sc = 0.1,
                       b_blur_ss = (1.0,1.0),
                       b_blur_bt = 'reflect',
                       g_blur_ks = (3,3),
                       g_blur_s = (1.0,1.0),
                       g_blur_bt = 'reflect',
                       )->Layer:
    """"""
    from napari_cool_tools_io import device

    show_info(f"B-scan prepocessing thread started")

    current_device = 'cpu'

    print(f"\ncurrent device is: {current_device}\n")

    print(f"\ngpu_flag is: {gpu}\n")

    if gpu and not vol_proc:
        current_device = device
    else:
        pass

    print(f"\ncurrent device is: {current_device}\n")    

    out = preproc_bscan(
        vol.data,
        transform=transform,
        ascan_corr=ascan_corr,
        Bandpass=Bandpass,
        log_cor=log_cor,
        vol_proc=vol_proc,
        chunk_shuff=chunk_shuff,
        debug=debug,
        processor=current_device,
        chunk_size=chunk_size,
        fov=fov,
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
        b_blur_ks=b_blur_ks,
        b_blur_sc=b_blur_sc,
        b_blur_ss=b_blur_ss,
        b_blur_bt=b_blur_bt,
        g_blur_ks=g_blur_ks,
        g_blur_s=g_blur_s,
        g_blur_bt=g_blur_bt
    )

    torch.cuda.empty_cache()
    memory_stats()

    if vol_proc:
        name = f"{vol.name}_vol_{transform.name}"
    else:
        if chunk_size > 1:
            name = f"{vol.name}_{transform.name}({chunk_size})"
        else:
            name = f"{vol.name}_{transform.name}"

    layer_type = 'image'
    add_kwargs = {"name":name}
    out_layer = Layer.create(out,add_kwargs,layer_type)

    show_info(f"B-scan preprocessing thread completed")

    return out_layer

def preproc_bscan_old(vol:ImageData, 
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       Vol_proc:bool = False,
                       Chunk_size:int = 1,
                       processor='cpu', 
                       log_gain = 2.5, 
                       clahe_clip_limit=1.0,
                       b_blur_ks = (5,5),
                       b_blur_sc = 0.1,
                       b_blur_ss = (1.0,1.0),
                       b_blur_bt = 'reflect',
                       g_blur_ks = (5,5),
                       g_blur_s = (1.0,1.0),
                       g_blur_bt = 'reflect',
                       )->ImageData:
    """"""
    from napari_cool_tools_registration._registration_tools import a_scan_correction_func2
    from jj_nn_framework.nn_transforms import BscanPreproc

    bscan_preproc = BscanPreproc(
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
        b_blur_ks=b_blur_ks,
        b_blur_sc=b_blur_sc,
        b_blur_ss=b_blur_ss,
        b_blur_bt=b_blur_bt,
        g_blur_ks=g_blur_ks,
        g_blur_s=g_blur_s,
        g_blur_bt=g_blur_bt
    )

    out = vol

    if ascan_corr:
        out = a_scan_correction_func2(out)
        torch.cuda.empty_cache()

    out_pt = torch.from_numpy(out.copy()).to(device=processor)

    out_pt.to(processor)

    out_pt_batch = out_pt.unsqueeze(0)
    print(f"out_pt shape: {out_pt.shape}\n")
    print(f"out_pt_batch shape: {out_pt_batch.shape}\n")


    if Vol_proc == True:
        out_pt = bscan_preproc((out_pt_batch,))[0]
        out = out_pt.detach().squeeze().cpu().numpy()
        torch.cuda.empty_cache()

    else:
        out_stack =[]

        chunk_size = Chunk_size
        
        num_chunks = int(out_pt_batch.shape[1] / chunk_size)
        
        #for i in tqdm(range(out_pt_batch.shape[1]),desc="Preprocessing B-scans"):
        for i in tqdm(range(num_chunks),desc="Preprocessing B-scans"):

            start = i*chunk_size
            end = start + chunk_size

            out_pt = bscan_preproc((out_pt_batch[:,start:end,:,:]))[0]
            #print(f"\n\nout_pt shape: {out_pt.shape}\n\n")
            out_stack.append(out_pt.detach().squeeze().cpu())
            torch.cuda.empty_cache()

        if chunk_size == 1:
            out = torch.stack(out_stack,dim=0).numpy()
        elif chunk_size > 1:
            out = torch.concat(out_stack,dim=0).numpy()

    return out

def generate_octa_plugin(
    img:Image,
    mscans:int=3,
    calc:OCTACalc=OCTACalc.STD,
    enface_only:bool=False,
    ascan_corr:bool=False,
    avg_dat:bool=True,
    log_corr:bool=False,
    clahe:bool=False,
    log_gain:float=1,
    clahe_clip:float=2.5,
    octa_data_avg:int=5
):
    """"""
    generate_octa_thread(
        img,
        mscans=mscans,
        calc=calc,
        enface_only=enface_only,
        ascan_corr=ascan_corr,
        avg_dat=avg_dat,
        log_corr=log_corr,
        clahe=clahe,
        log_gain=log_gain,
        clahe_clip=clahe_clip,
        octa_data_avg=octa_data_avg
    )


    return

@thread_worker(connect={"yielded": viewer.add_layer})
def generate_octa_thread(
    img:Image,
    mscans:int=3,
    calc:OCTACalc=OCTACalc.STD,
    enface_only:bool=False,
    ascan_corr:bool=False,
    avg_dat:bool=True,
    log_corr:bool=False,
    clahe:bool=False,
    log_gain:float=1,
    clahe_clip:float=2.5,
    octa_data_avg:int=5

) -> Generator[Layer,Layer,Layer]:
    """"""

    show_info(f"OCTA processing thread started")

    if enface_only:
        name = f"{img.name}_{calc.name}_MIP"
    else:
        name = f"{img.name}_{calc.name}"

    

    outputs = generate_octa(
        img.data,
        mscans=mscans,
        calc=calc,
        enface_only=enface_only,
        ascan_corr=ascan_corr,
        avg_dat=avg_dat,
        log_corr=log_corr,
        clahe=clahe,
        log_gain=log_gain,
        clahe_clip=clahe_clip,
        octa_data_avg=octa_data_avg
    )

    for out_data in outputs:
        
        name = f"{img.name}_{calc.name}"
        if out_data[1] != "":
            name = f"{img.name}_{calc.name}_{out_data[1]}"

        layer_type = 'image'
        add_kwargs = {"name":name}

        out_layer = Layer.create(out_data[0],add_kwargs,layer_type)

        yield(out_layer)

    show_info(f"OCTA processing thread completed")

    #return out_layer

def annotation_preset(vol:Image, ascan_corr:bool=True):
    """Do initial preprocessing of OCT B-scan and or enface to prepare them for annotation and analysis.
    Args:
        vol (Image): 3D ndarray representing structural OCT data
        ascan_corr (bool): If true volume and enface image will be corrected for sin wave scanning distortion

    Returns:
        Layers of processed b-scans, processed enface image, b-scan segmentation, enface segmentation
    """
    annotation_preset_thread(vol=vol,ascan_corr=ascan_corr)
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def annotation_preset_thread(vol:Image, ascan_corr:bool=True) -> Generator[Layer,Layer,Layer]:
    """Do initial preprocessing of OCT B-scan and or enface to prepare them for annotation and analysis.
    Args:
        vol (Image): 3D ndarray representing structural OCT data
        ascan_corr (bool): If true volume and enface image will be corrected for sin wave scanning distortion

    Returns:
        Layers of processed b-scans, processed enface image, b-scan segmentation, enface segmentation
    """
    show_info(f"Annotation preset thread started")
    layers = annotation_preset_func(vol=vol,ascan_corr=ascan_corr)
    for layer in layers:
        yield layer
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f"B-scan preset thread completed")
    show_info(f"Annotation preset thread completed")


def annotation_preset_func(vol:Image, ascan_corr:bool=True) -> Layer:
    """Do initial preprocessing of OCT B-scan and or enface to prepare them for annotation and analysis.
    Args:
        vol (Image): 3D ndarray representing structural OCT data
        ascan_corr (bool): If true volume and enface image will be corrected for sin wave scanning distortion

    Returns:
        Layers of processed b-scans, processed enface image, b-scan segmentation, enface segmentation
    """
    from napari_cool_tools_registration._registration_tools import a_scan_correction_func
    from napari_cool_tools_segmentation._segmentation import b_scan_pix2pixHD_seg_func, enface_unet_seg_func

    layers = []

    if ascan_corr:
        init = a_scan_correction_func(vol)
    else:
        init = vol

    layers.append(init)

    out = process_bscan_preset_func(init,ascan_corr=False)
    enface_list = generate_enface_image_func(vol,sin_correct=True,band_pass_filter=False,CLAHE=False)
    enface = enface_list[0]
    bscan_seg = b_scan_pix2pixHD_seg_func(init)
    enface_seg_list = enface_unet_seg_func(enface)
    enface_seg = enface_seg_list[0]
    print(type(enface_seg))

    layers.append(out)
    layers.append(bscan_seg)
    layers.append(enface)
    layers.append(enface_seg)

    return layers