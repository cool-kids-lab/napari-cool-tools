"""
This module contains code for OCT data preprocessing functions.
"""
import sys
import traceback
import logging
import gc
import numpy as np
from enum import Enum
from tqdm import tqdm
from typing import List, Generator
from napari.utils.notifications import show_info
from napari.layers import Layer
from napari.types import ImageData
from napari_cool_tools_img_proc._equalization_funcs import clahe_pt_func
from napari_cool_tools_img_proc._luminance_funcs import adjust_log_pt_func
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_pt_func
from napari_cool_tools_io import torch
from napari_cool_tools_registration._registration_tools_funcs import a_scan_correction_func2
from napari_cool_tools_vol_proc._averaging_tools_funcs import average_per_bscan_pt

class Preproc(Enum):
    NLCGbBb = "Norm_Log_CLAHE_Gblur_Bblur"
    SNLC = "Stand_Nom_Log_CLAHE"
    SNL = "Stand_Norm_Log"
    SN = "Stand_Norm"
    CCL = "Conditional_CLAHE_Log"
    RRAR = "Random_Resized_Aspect_Ratio"

class Augmentation(Enum):
    RandCropResizeAspectRat = "Random_Crop_Resized_Aspect_Ratio"

class OCTACalc(Enum):
    STD = "Standard Deviation"
    VAR = "Variance"
    VAR2 = "Variance Squared"

def data_augmentation(vol:ImageData, 
                       transform:Augmentation=Augmentation.RandCropResizeAspectRat,
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       log_cor:bool=False,
                       vol_proc:bool = False,
                       chunk_shuff:bool = True,
                       debug:bool = False,
                       processor='cpu', 
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
    from jj_nn_framework.nn_transforms import BscanPreproc, NapStandNorm, NapStandNormLog, NapStandNormLogCLAHE, NapCondCLAHELog, NapRandResizeAspectRatio

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

    stand_norm_log = NapStandNormLog(
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
    )

    stand_norm_log_clahe = NapStandNormLogCLAHE(
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
    )

    stand_norm = NapStandNorm()

    cond_clahe_log = NapCondCLAHELog(log_cor=log_cor)

    rand_resized_aspect = NapRandResizeAspectRatio(fov=fov)

    #if vol_proc:
    #    proc = 'cpu'
    #else:
    #    proc = processor

    proc = processor

    out = vol

    if ascan_corr:
        out = a_scan_correction_func2(out)
        torch.cuda.empty_cache()

    #out_pt = torch.from_numpy(out.copy()).to(device=proc)
    out_pt = torch.tensor(out.copy()).to(device=proc)
    out_vol = torch.zeros_like(out_pt.detach().to(device=proc)).to(device=proc)
    #out_vol = torch.tensor(out.copy()).to(device='cpu')

    #out_pt.to(processor)

    out_pt_batch = out_pt.unsqueeze(0)

    if debug:
        print(f"out_pt shape: {out_pt.shape}\n")
        print(f"out_pt_batch shape: {out_pt_batch.shape}\n")


    if vol_proc == True:

        # select preprocessing preset
        if transform == Augmentation.RandCropResizeAspectRat:
            out_pt = rand_resized_aspect((out_pt_batch,))[0]

        out = out_pt.detach().squeeze().cpu().numpy()
        torch.cuda.empty_cache()
    
    else:
        out_stack =[]

        #chunk_size = Chunk_size

        try:
            assert out_pt_batch.shape[1] % chunk_size == 0, f"\nData of volume {out_pt_batch.shape[1]} is not evenly divisible by chunk of size {chunk_size}\n"
        except AssertionError as e:
            #logging.error(f"\nData of volume {out_pt_batch.shape[1]} is not evenly divisible by chunk of size {chunk_size}\n",exc_info=True)
            #print(f'Assertion failed: {str(e)}')
            _, _, tb = sys.exc_info()
            #traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]
            print(f'Assertion failed: {str(e)}\nCheck File {filename}, line {line}, in {func}\n\
                  --> {text}')
            #raise
            exit(1)
        
        num_chunks = int(out_pt_batch.shape[1] / chunk_size)

        if debug:
            print(f"num chunks: {num_chunks}, chunk size: {chunk_size}\n")

        if chunk_shuff:

            rand_shuff = torch.stack([torch.randperm(num_chunks).to(proc) for _ in range(chunk_size)],dim=1)
            multi = torch.arange(0,out_pt_batch.shape[1],num_chunks).to(proc)
            rand_shuff = rand_shuff + multi

            assess = torch.unique(rand_shuff)

        if debug and chunk_shuff:
            print(f"rand_shuff shape: {rand_shuff.shape}\ncol 0: {rand_shuff[:,0]}\nrow 0: {rand_shuff[0]}\n")
            print(f"flatten rand_shuff: {assess}, shape: {assess.shape}\n")
        
        #for i in tqdm(range(out_pt_batch.shape[1]),desc="Preprocessing B-scans"):
        for i in tqdm(range(num_chunks),desc="Preprocessing B-scans"):

            start = i*chunk_size
            end = start + chunk_size

            if chunk_shuff:
                x = torch.arange(len(out_pt_batch)).to(proc)
                y = rand_shuff[i].to(proc)
            else:
                x = torch.arange(len(out_pt_batch)).to(proc)
                y = torch.arange(i,vol.shape[0],num_chunks).to(proc)           


            # select preprocessing preset
            if transform == Augmentation.RandCropResizeAspectRat:
                if chunk_shuff:
                    out_pt = rand_resized_aspect((out_pt_batch[x,y],))[0]
                else:
                    out_pt = rand_resized_aspect((out_pt_batch[:,start:end,:,:]))[0]

                #show_info(f"{Preproc.RRAR.value} has not been implemented for non_volumetric processing\n")
            
            
            breakpoint()

            #print(f"\n\nout_pt shape: {out_pt.shape}\n\n")
            transfer = out_pt.detach().squeeze().cpu()

            if debug:
                print(f"\nout_vol[y] shape, {out_vol[y].shape}, transfer shape:  {transfer.shape}\n")

            if proc == 'cpu':
                out_vol[y] = transfer[:]
            elif proc == processor:
                out_vol[y] = out_pt.squeeze()
            

            #out_stack.append(out_pt.detach().squeeze().cpu())
            out_stack.append(transfer)
            torch.cuda.empty_cache()

        if chunk_shuff:
            out = out_vol.cpu().numpy()
        else:

            if chunk_size == 1:
                out = torch.stack(out_stack,dim=0).numpy()
            elif chunk_size > 1:
                out = torch.concat(out_stack,dim=0).numpy()
    
    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")


    return out
    

def preproc_bscan(vol:ImageData, 
                       transform:Preproc=Preproc.NLCGbBb,
                       ascan_corr:bool=True, 
                       Bandpass:bool=False,
                       log_cor:bool=False,
                       vol_proc:bool = False,
                       chunk_shuff:bool = True,
                       debug:bool = False,
                       processor='cpu', 
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
    from jj_nn_framework.nn_transforms import BscanPreproc, NapStandNorm, NapStandNormLog, NapStandNormLogCLAHE, NapCondCLAHELog, NapRandResizeAspectRatio

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

    stand_norm_log = NapStandNormLog(
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
    )

    stand_norm_log_clahe = NapStandNormLogCLAHE(
        log_gain=log_gain,
        clahe_clip_limit=clahe_clip_limit,
    )

    stand_norm = NapStandNorm()

    cond_clahe_log = NapCondCLAHELog(log_cor=log_cor)

    rand_resized_aspect = NapRandResizeAspectRatio(fov=fov)

    if vol_proc:
        proc = 'cpu'
    else:
        proc = processor

    if vol.ndim < 2:
        raise ValueError(f"Input data has {vol.ndim} dimensinos and this funciton requires a minimum of two dimensions")
    elif vol.ndim == 2:
        vol = vol.reshape(1,*vol.shape[:])
        print(f"Expanded shape: {vol.shape}")

    out = vol

    if ascan_corr:
        out = a_scan_correction_func2(out)
        torch.cuda.empty_cache()

    #out_pt = torch.from_numpy(out.copy()).to(device=proc)
    out_pt = torch.tensor(out.copy()).to(device=proc)
    out_vol = torch.zeros_like(out_pt.detach().to(device='cpu')).to(device='cpu')
    #out_vol = torch.tensor(out.copy()).to(device='cpu')

    #out_pt.to(processor)

    out_pt_batch = out_pt.unsqueeze(0)

    if debug:
        print(f"out_pt shape: {out_pt.shape}\n")
        print(f"out_pt_batch shape: {out_pt_batch.shape}\n")


    if vol_proc == True:

        # select preprocessing preset
        if transform == Preproc.NLCGbBb:
            out_pt = bscan_preproc((out_pt_batch,))[0]
        elif transform == Preproc.SNLC:
            out_pt = stand_norm_log_clahe((out_pt_batch,))[0]
        elif transform == Preproc.SNL:
            out_pt = stand_norm_log((out_pt_batch,))[0]
        elif transform == Preproc.SN:
            out_pt = stand_norm((out_pt_batch,))[0]
        elif transform == Preproc.CCL:
            out_pt = cond_clahe_log((out_pt_batch,))[0]
        elif transform == Preproc.RRAR:
            out_pt = rand_resized_aspect((out_pt_batch,))[0]

        out = out_pt.detach().squeeze().cpu().numpy()
        torch.cuda.empty_cache()

    else:
        out_stack =[]

        #chunk_size = Chunk_size
        
        num_chunks = int(out_pt_batch.shape[1] / chunk_size)

        if debug:
            print(f"num chunks: {num_chunks}, chunk size: {chunk_size}\n")

        if chunk_shuff:

            rand_shuff = torch.stack([torch.randperm(num_chunks).to(proc) for _ in range(chunk_size)],dim=1)
            multi = torch.arange(0,out_pt_batch.shape[1],num_chunks).to(proc)
            rand_shuff = rand_shuff + multi

            assess = torch.unique(rand_shuff)

        if debug:
            print(f"rand_shuff shape: {rand_shuff.shape}\ncol 0: {rand_shuff[:,0]}\nrow 0: {rand_shuff[0]}\n")
            print(f"flatten rand_shuff: {assess}, shape: {assess.shape}\n")
        
        #for i in tqdm(range(out_pt_batch.shape[1]),desc="Preprocessing B-scans"):
        for i in tqdm(range(num_chunks),desc="Preprocessing B-scans"):

            start = i*chunk_size
            end = start + chunk_size

            if chunk_shuff:
                x = torch.arange(len(out_pt_batch)).to(proc)
                y = rand_shuff[i].to(proc)
            else:
                x = torch.arange(len(out_pt_batch)).to(proc)
                y = torch.arange(i,vol.shape[0],num_chunks).to(proc)           


            # select preprocessing preset
            if transform == Preproc.NLCGbBb:
                if chunk_shuff:
                    out_pt = bscan_preproc((out_pt_batch[x,y],))[0]
                else:
                    out_pt = bscan_preproc((out_pt_batch[:,start:end,:,:]))[0]
            elif transform == Preproc.SNLC:
                if chunk_shuff:
                    out_pt = stand_norm_log_clahe((out_pt_batch[x,y],))[0]
                else:
                    out_pt = stand_norm_log_clahe((out_pt_batch[:,start:end,:,:]))[0]
            elif transform == Preproc.SNL:
                if chunk_shuff:
                    out_pt = stand_norm_log((out_pt_batch[x,y],))[0]
                else:
                    out_pt = stand_norm_log((out_pt_batch[:,start:end,:,:]))[0]
            elif transform == Preproc.SN:
                if chunk_shuff:
                    out_pt = stand_norm((out_pt_batch[x,y],))[0]
                else:
                    out_pt = stand_norm((out_pt_batch[:,start:end,:,:]))[0]
            elif transform == Preproc.CCL:
                if chunk_shuff:
                    out_pt = cond_clahe_log((out_pt_batch[x,y],))[0]
                else:
                    out_pt = cond_clahe_log((out_pt_batch[:,start:end,:,:]))[0]
            elif transform == Preproc.RRAR:
                show_info(f"{Preproc.RRAR.value} has not been implemented for non_volumetric processing\n")
            
            
            #print(f"\n\nout_pt shape: {out_pt.shape}\n\n")
            transfer = out_pt.detach().squeeze().cpu()
            out_vol[y] = transfer[:]

            #print(out_vol[y].shape, transfer.shape)

            #out_stack.append(out_pt.detach().squeeze().cpu())
            out_stack.append(transfer)
            torch.cuda.empty_cache()

        if chunk_shuff:
            out = out_vol.numpy()
        else:

            if chunk_size == 1:
                out = torch.stack(out_stack,dim=0).numpy()
            elif chunk_size > 1:
                out = torch.concat(out_stack,dim=0).numpy()

        
    if proc != 'cpu':
        del (
                out_pt, 
                out_pt_batch,
                x,
                y,
            )
        if chunk_shuff:
            del (
                rand_shuff,
                multi,
                assess,
            )
        gc.collect()
        torch.cuda.empty_cache()
    
    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    if not gpu_mem_clear:
        print(f"{torch.cuda.memory_summary()}\n")

    return out

def generate_enface(
    data:ImageData,
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
)-> Generator[ImageData,ImageData,ImageData]:
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Yields:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    from napari_cool_tools_registration._registration_tools_funcs import a_scan_reg_calc_settings, a_scan_subpix_registration
    from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_pt_func
    from napari_cool_tools_img_proc._denoise_funcs import diff_of_gaus_func
    from napari_cool_tools_img_proc._equalization_funcs import clahe_pt_func
    from napari_cool_tools_img_proc._luminance_funcs import adjust_log_pt_func

    #layers = []

    #show_info(f'Generate enface image thread has started')

    show_info(f'Generating initial enface MIP')
    yx = data.transpose(1,2,0)
    enface_mip = yx.max(0)
    print(f"enface_mip shape: {enface_mip.shape}\n")

    if debug:
        suffix = "init_MIP"
        yield (enface_mip,suffix)

    correct_mip = enface_mip.copy()
    correct_mip.shape = (correct_mip.shape[0],1,correct_mip.shape[1])
    correct_mip = correct_mip.transpose(2,1,0)
    print(f"correct_mip shape: {correct_mip.shape}\n")
    
    if debug:
        suffix = "corrected"
        yield (correct_mip,suffix)

    if sin_correct:
        show_info(f'Correcting enface MIP distortion')
        ac_correct_mip = a_scan_correction_func2(correct_mip.copy())

        if ac_correct_mip.ndim == 2:
            correct_mip = ac_correct_mip.reshape((ac_correct_mip.shape[-2],1,ac_correct_mip.shape[-1]))
        elif ac_correct_mip.ndim == 3:
            correct_mip = ac_correct_mip
        else:
            pass # put error handling here or near init

        if debug:
            suffix = "ascan_corrected"
            yield (ac_correct_mip.squeeze(),suffix)
    else:
        pass    
    
    show_info(f'Calculating Optimal Subregions for subpixel registration')
    

    settings = a_scan_reg_calc_settings(correct_mip)
    if debug:
        show_info(f"{settings['region_num']}")

    show_info(f'Completing subpixel registration of A-scans')
    #outs = list(a_scan_reg_subpix_gen(correct_mip_layer,settings))
    outs = list(a_scan_subpix_registration(correct_mip,settings=settings,fill_gaps=True,roll_over_flag=False,debug=debug))

    for i,out in enumerate(outs):
        out = out.squeeze()
        if  i == len(outs) -1:

            out = normalize_data_in_range_pt_func(out,0,1)

            if CLAHE:
                out = out.squeeze()
                out = clahe_pt_func(out,clip_limit=clahe_clip)
            if log_correct:
                out = adjust_log_pt_func(out,log_gain)
            if band_pass_filter:
                out = diff_of_gaus_func(out,low_sigma=bp_low,high_sigma=bp_high,pt=bp_pt)
            if exp:
                out = out**n

            if debug:
                suffix="debug"
                yield (out,suffix)
            else:
                out = out.squeeze()
                out = out.transpose(1,0)
                suffix="enface"
                yield (out,suffix)
        else:
            if debug:
                suffix="debug2"
                yield (out,suffix)
            else:
                pass
            pass

    #show_info(f'Generate enface image thread has completed')
    #return layers

def generate_octa_var(
    data:ImageData,
    mscans:int=3,
    calc:OCTACalc=OCTACalc.STD,
    ascan_corr:bool=False,
    w_wo_ascan:bool=False,
    avg_dat:bool=True,
    octa_data_avg:int=5,    
):
    """
    Generate OCTA variance data
    """

    #outputs = []

    if mscans < 2:
        raise RuntimeError(f"OCTA processing requires at least 2 M-scans to function")
    
    new_shape = (-1,mscans,data.shape[-2],data.shape[-1])
    m_comp = data.reshape(new_shape)

    if calc == OCTACalc.STD:
        out_data = m_comp.std(axis=1)
    elif calc == OCTACalc.VAR or calc == OCTACalc.VAR2:
        out_data = m_comp.var(axis=1)
        if calc == OCTACalc.VAR2:
            out_data = out_data**2
    else:
        raise RuntimeError(f"{calc} is an invalid calculation mode and has not been implemented")
    
    if avg_dat:
        out_data = average_per_bscan_pt(out_data,scans_per_avg=octa_data_avg,ensemble=True)
    
    
    var = out_data.copy()
    #yield (var,"Var")
    if ascan_corr:
        if w_wo_ascan:
            yield (var,"var")
        var = a_scan_correction_func2(var)
        yield (var,"Var(ASC)")
    else:
        yield (var,"var")


def generate_octa(
    data:ImageData,
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

    #outputs = []

    if mscans < 2:
        raise RuntimeError(f"OCTA processing requires at least 2 M-scans to function")
    
    new_shape = (-1,mscans,data.shape[-2],data.shape[-1])
    m_comp = data.reshape(new_shape)

    if calc == OCTACalc.STD:
        out_data = m_comp.std(axis=1)
    elif calc == OCTACalc.VAR or calc == OCTACalc.VAR2:
        out_data = m_comp.var(axis=1)
        if calc == OCTACalc.VAR2:
            out_data = out_data**2
    else:
        raise RuntimeError(f"{calc} is an invalid calculation mode and has not been implemented")
    
    if avg_dat:
        out_data = average_per_bscan_pt(out_data,scans_per_avg=octa_data_avg,ensemble=True)
    
    if not enface_only:
        #outputs.append((out_data,""))
        var = out_data.copy()
        yield (var,"Var")
        if ascan_corr:
            var = a_scan_correction_func2(var)
            yield (var,"Var(ASC)")
    
    #out_data = out_data.max(axis=1)
    #out_data = out_data.transpose(1,0)

    if ascan_corr:
        sin_correct = True

    out_data = next(generate_enface(out_data,sin_correct=ascan_corr,exp=False,CLAHE=True,clahe_clip=2.5,log_correct=True,log_gain=1,band_pass_filter=False))[0]

    #out_data = normalize_data_in_range_pt_func(out_data,0.0,1.0,numpy_out=True)

    if calc == OCTACalc.STD:
        out_data = out_data**2

    if clahe:
        out_data = clahe_pt_func(out_data,clip_limit=clahe_clip)

    if log_corr:
        out_data = adjust_log_pt_func(out_data,gain=log_gain)
    
    #outputs.append((out_data,"MIP"))
    yield (out_data,"MIP")

    #return outputs
