"""
This module contains function code for registering volumetric and image data.
"""
import gc
import time
import cupy as cp
import numpy as np
from typing import Dict, Generator
from enum import Enum
from tqdm import tqdm
from napari.types import ImageData
from torchvision.transforms.functional import InterpolationMode
from napari_cool_tools_io import torch, device, memory_stats
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func, normalize_data_in_range_pt_func
from jj_nn_framework.torch_utils import torch_interp
from jj_nn_framework.image_funcs import rev_ascan_correction

class AspectRatioPreservationMode(Enum):
    Depth = "Depth"
    Height = "Height"
    Width = "Width"
    Volume = "Volume"

class ArrayImplementation(Enum):
    CUPY = "cupy"
    NUMPY = "numpy"
    TORCH = "pytorch"

def a_scan_numpy(data:ImageData,rev:bool=False) -> ImageData:
    """
    """
    d = data.shape[0]
    h = data.shape[1]
    w = data.shape[2]

    data_out = np.empty_like(data)
    Xn = np.arange(w)

    x_org = (w/2)*np.sin(np.pi/w*Xn-np.pi/2) + (w/2)

    for i in tqdm(range(d),desc="A-scan Correction"):
        for j in range(h):
            if rev:
                data_out[i,j,:] = np.interp(x_org,Xn,data[i,j,:])
            else:
                data_out[i,j,:] = np.interp(Xn,x_org,data[i,j,:])

    return data_out

def a_scan_torch(data:ImageData,rev:bool=False,device=device) -> ImageData:
    """
    """
    tensor = torch.tensor(data.copy()).to(device)
    d = tensor.shape[0]
    h = tensor.shape[1]
    w = tensor.shape[2]

    tensor_out = torch.empty_like(tensor).to(device)
    Xn = torch.arange(w).to(device)

    x_org = (w/2)*torch.sin(torch.pi/w*Xn-torch.pi/2) + (w/2)

    for i in tqdm(range(d),desc="A-scan Correction"):
        for j in range(h):
            if rev:
                tensor_out[i,j,:] = torch_interp(x_org,Xn,tensor[i,j,:],device=device)
                #tensor_out = rev_ascan_correction(tensor)
            else:
                tensor_out[i,j,:] = torch_interp(Xn,x_org,tensor[i,j,:],device=device)
    
    numpy_out = tensor_out.squeeze().cpu().numpy()

    del tensor, tensor_out, Xn, x_org
    gc.collect()
    memory_stats()

    return 


def a_scan_cupy(data:ImageData, rev:bool=False) -> ImageData:
    """
    """
    print(f"cupy used memory(func start): {cp.get_default_memory_pool().used_bytes()} bytes\n")
    d = data.shape[0]
    h = data.shape[1]
    w = data.shape[2]

    data_out = cp.empty_like(data)
    Xn = cp.arange(w)

    x_org = (w/2)*cp.sin(cp.pi/w*Xn-cp.pi/2) + (w/2)

    for i in range(d):
    #for i in tqdm(range(d),desc="A-scan Correction"):
        for j in range(h):
            if rev:
                data_out[i,j,:] = cp.interp(x_org,Xn,data[i,j,:])
            else:
                data_out[i,j,:] = cp.interp(Xn,x_org,data[i,j,:])

    print(f"cupy used memory(func pre-return): {cp.get_default_memory_pool().used_bytes()} bytes\n")
    return data_out


def a_scan_correction_func2(data:ImageData,rev:bool=False,norm_out:bool=True,implementation:ArrayImplementation=ArrayImplementation.NUMPY) -> ImageData:
    """
    """

    shape = data.shape

    if data.ndim == 2:
        data = data.reshape(shape[0],-1,shape[1])
    elif data.ndim == 3:
        pass
    else:
        data = data.reshape(-1,shape[-2],shape[-1])

    if implementation == ArrayImplementation.CUPY:
        start = time.time()
        data_cp = cp.array(data,copy=True)
        data_out = cp.asnumpy(a_scan_cupy(data_cp,rev=rev))
        del data_cp
        cp.get_default_memory_pool().free_all_blocks()
        print(f"cupy used memory(after free): {cp.get_default_memory_pool().used_bytes()} bytes\n")
        processing_time = time.time() - start
        print(f"Cupy implementation processing time: {processing_time}\n")
    elif implementation == ArrayImplementation.NUMPY:
        data_out = a_scan_numpy(data,rev=rev)
    elif implementation == ArrayImplementation.TORCH:
        data_out = a_scan_torch(data,rev=rev,device='cpu')
        memory_stats()
    else:
        raise RuntimeError("Invalid implementation selection")
    
    if norm_out:
        data_out = normalize_data_in_range_func(data_out, 0.0, 1.0)

    data_out = data_out.reshape(shape).squeeze()

    return data_out

def a_scan_reg_calc_settings(data:ImageData, min_regs:int=3, max_regs:int=8,debug:bool=False) -> Dict:
    """Calculate optimal number of regions to divide volume into for a-scan registration.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        max_regs (int): maximum number of regions that will be considered for a-scan registration

    Returns:
        int indicating optimal number of regions to use when performing a-scan registration
    """

    from skimage.registration import phase_cross_correlation

    ascan_settings = {"region_num":None,"regions":None,"shifts":None,"avg_phase_diff":None}

    print(f"\n\nascan reg calc data shape: {data.shape}\n\n")

    for sections in tqdm(range(min_regs,max_regs+1),desc="Testing number of regions\n"):

        #sec_len = int(data.shape[0] / sections) # remove after improving code/comments this axes were swapped
        sec_len = int(data.shape[2] / sections)
        
        regions = []
        shifts = []
        phase_diffs = []

        for s in tqdm(range(sections),desc="Creating Regions\n"):
            start = s*sec_len
            end = start + sec_len
            curr = data[:,:,start:end]
            #curr = data[start:end,:,:] # remove after improving code/comments this axes were swapped
            regions.append(curr)

            even = curr[::2,:,:]
            odd = curr[1::2,:,:]
            
            shift, error, diffphase = phase_cross_correlation(even, odd, upsample_factor=100)

            if debug:
                print(f"\n\nshift:{shift}\nerror: {error}\ndiffphase: {diffphase}\n\n")

            shifts.append(shift[2])
            phase_diffs.append(diffphase)

        avg_phase_diff = np.array(phase_diffs)
        avg_phase_diff = np.absolute(avg_phase_diff)
        avg_phase_diff = np.mean(avg_phase_diff,axis=0)
        avg_phase_diff = avg_phase_diff

        if debug:
            print(f"\n\nSection region(s) has/have an avg phase diff of {avg_phase_diff}\n\n")

        if sections == min_regs:
            ascan_settings["region_num"] = sections
            ascan_settings["regions"] = regions
            ascan_settings["shifts"] = shifts
            ascan_settings["avg_phase_diff"] = avg_phase_diff
        else:
            if ascan_settings["avg_phase_diff"] > avg_phase_diff:

                print(f"\nImproved phase diff from {ascan_settings['avg_phase_diff']} to {avg_phase_diff}\n")
                print(f"Optimal number of regions has changed from {ascan_settings['region_num']} to {sections}\n")
                ascan_settings["region_num"] = sections
                ascan_settings["regions"] = regions
                ascan_settings["shifts"] = shifts
                ascan_settings["avg_phase_diff"] = avg_phase_diff
                
            else:
                pass
    print(f"\nThe optimal number of regions was determined to be {ascan_settings['region_num']}\nThis yields an average phase difference of {ascan_settings['avg_phase_diff']}\n")
    return ascan_settings

def a_scan_subpix_registration(data:ImageData, settings:Dict, sub_pixel_threshold:float=0.5, fill_gaps=True,roll_over_flag=True,debug=False) -> Generator[ImageData,ImageData,ImageData]:
    """"""
    from scipy.ndimage import fourier_shift

    print(f"\n\nsubpix reg data shape: {data.shape}\n\n")

    data_out = np.zeros_like(data)

    replace_val = round(data.max()+2)

    regions = settings["regions"]
    shifts = settings["shifts"]

    out_reg = []
    roll_overs = []

    for i,s in tqdm(enumerate(shifts),desc="Shifting Regions\n"):
        shift2 = round(s)
        shift_idx = abs(shift2)

        out = np.empty_like(regions[i])
        out[::2,:,:] = regions[i][::2,:,:]

        print(f"\n\nregion shape: {out.shape}\n\n")

        yield out

        input_ = np.fft.fft2(regions[i][1::2,:,:])
        result = fourier_shift(input_,(0.0,0.0,s),axis=2)
        result = np.fft.ifft2(result)
        odd_out = result.real

        #yield odd_out

        if s < 0:

            if abs(s) >= sub_pixel_threshold:

                if roll_over_flag:
                    roll_over = odd_out[:,:,-shift_idx:].copy()
                    roll_overs.append(roll_over)

                odd_out[:,:,-shift_idx:] = replace_val

                
                if i > 0 and shifts[i-1] < 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                    out_reg_odd = out_reg[i-1][1::2,:,:]

                    axis_0_len,axis_1_len = out_reg_odd.shape[0], out_reg_odd.shape[1]

                    if debug:
                        print(f"\nout_reg_odd shape: {out_reg_odd.shape}\n")

                        print(f"\nout_reg_odd[{i-1}] range: ({out_reg_odd.min()}, {out_reg_odd.max()})\n")

                    gap_idx = (out_reg_odd == replace_val)

                    gap = out_reg_odd[gap_idx]

                    gap.shape = (axis_0_len,axis_1_len,-1)

                    if debug:
                        print(f"\nout_reg gap shape: {gap.shape}\nroll_over shape: {roll_over.shape}\n")


                    if roll_over_flag:
                        gap_roll_diff = gap.shape[2] - roll_over.shape[2]

                        if gap_roll_diff >= 0:
                            out_reg[i-1][1::2,:,-roll_over.shape[2]:] = roll_over
                            if debug:
                                print(f"\npositive or neutral gap_roll_diff s < 0\n")
                        elif gap_roll_diff < 0:
                            out_reg[i-1][1::2,:,-gap.shape[2]:] = roll_over[:,:,gap.shape[2]:]
                            if debug:
                                print(f"\nnegative gap_roll_diff s > 0\n")


            else:
                roll_overs.append(None)

        elif s > 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,:shift_idx].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,:shift_idx] = replace_val

            
                if i > 0 and shifts[i-1] > 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                        if roll_over_flag:

                            prev_roll_over = roll_overs[i-2]

                            if debug:
                                print(f"\nrollovers length: {len(roll_overs)}\n i: {i-2}\n")

                            gap_roll_diff = shift_idx - roll_over.shape[2]

                            if debug:
                                print(f"\nshift_idx: {shift_idx}, roll_over shape[2]: {prev_roll_over.shape[2]}\n")

                            if gap_roll_diff >= 0:
                                odd_out[:,:,:prev_roll_over.shape[2]] = prev_roll_over
                                
                                if debug:
                                    print(f"\npositive or neutral gap_roll_diff s > 0\n")

                            elif gap_roll_diff < 0:
                                odd_out[:,:,:shift_idx] = prev_roll_over[:,:,:shift_idx]

                                if debug:
                                    print(f"\nnegative gap_roll_diff s > 0\n")

            else:
                roll_overs.append(None)

        out[1::2,:,:] = odd_out

        out_reg.append(out)    

    output = np.concatenate(out_reg,axis=2)

    if fill_gaps:

        # find gaps
        gap_idxs = (output == replace_val)
        init_idxs = gap_idxs[1::2,:,:-1]
        final_idxs = gap_idxs[1::2,:,1:]

        # debug

        if debug:
            debug_data = np.empty_like(output)
            debug_data[gap_idxs] = 1
            debug_data = debug_data.astype('uint8')
            yield debug_data

        gap_starts = (init_idxs < final_idxs)
        gap_ends = (init_idxs > final_idxs)

        gap_starts = gap_starts.nonzero()
        gap_ends = gap_ends.nonzero()

        if debug:
            print(f"\ngap starts: {gap_starts}\ngap ends: {gap_ends}\n")

        gap_start_idxs = np.unique(gap_starts[2])
        gap_end_idxs = np.unique(gap_ends[2])

        if debug:
            print(f"\ngap start indicies: {gap_start_idxs}\ngap end indicies: {gap_end_idxs}\n")


        if len(gap_start_idxs) < len(gap_end_idxs):
            loops = len(gap_end_idxs)
        else:
            loops = len(gap_start_idxs)

        for i in tqdm(range(loops),desc="Filling middle gaps\n"):

            if len(gap_start_idxs) < len(gap_end_idxs):
                if i == 0:
                    s_idx = 0
                    e_idx = gap_end_idxs[i] + 1
                    num_tile = e_idx
                    vals = np.tile(output[1::2,:,e_idx],(num_tile,1,1))
                    vals = np.transpose(vals,(1,2,0))
                    output[1::2,:,:e_idx+1] = vals

                else:
                    s_idx = gap_start_idxs[i-1]
                    e_idx = gap_end_idxs[i] + 1

                    if debug:
                        print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")

                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1

                    if debug:
                        print(f"\nnum: {num}\n")

                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
                   
                    if debug:
                        print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")

                    output[1::2,:,s_idx:e_idx+1] = interp_out

            elif  len(gap_start_idxs) >= len(gap_end_idxs):
            
                if i < len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = gap_end_idxs[i] + 1

                    if debug:
                        print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")

                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1

                    if debug:
                        print(f"\nnum: {num}\n")

                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))

                    if debug:
                        print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")

                    output[1::2,:,s_idx:e_idx+1] = interp_out
                elif i >= len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = output.shape[2] - 1
                    num_tile = e_idx-(s_idx)
                    vals = np.tile(output[1::2,:,s_idx],(num_tile,1,1))
                    vals = np.transpose(vals,(1,2,0))
                    output[1::2,:,s_idx+1:] = vals #output[1::2,:,s_idx]
                    pass
                else:
                    print(f"\nWhy are we here near line 950?\n")

    #print("You made it here!!")
    #if debug:
    print(f"out_data shape : {output.shape}, type: {type(output)}\n")

    # If the w dimension of the volume is not evenly divisible by the optimal number of regions calculated there will be a row or two of missing pixel data which we drop as it is typically outside the area of interest
    # data out is the shape of the input data to keep the output consistent with the input 
    data_out[:output.shape[0],:,:output.shape[2]] = output[:,:,:]

    yield data_out #output

def adjust_aspect_ratio(vol:ImageData,fov:float=116.0,interpolation:InterpolationMode=InterpolationMode.BICUBIC,preservation:AspectRatioPreservationMode=AspectRatioPreservationMode.Volume) -> ImageData:
    """"""
    from torchvision.transforms.functional import resize
    vol_t = torch.tensor(vol.copy()).to(device=device)
    theta = fov/2.0
    sigma = 90.0 - theta

    aspect_ratio = sigma/fov
    DtoW_rat = vol.shape[0]/vol.shape[2]

    # can preserve Depth, Height, Width, or Volume
    

    if preservation == AspectRatioPreservationMode.Volume:
        V = vol_t.numel()
        W = (V/aspect_ratio)**(1/3)
        D = W * DtoW_rat
        H = V / (D*W) 

        print(f"V: {V}, D: {D}, W: {W}, H: {H}\n")

        D,W,H = round(D), round(W), round(H)

        ac_t = vol_t.permute(1,2,0)
        ac_t = resize(ac_t,(W,D))
        ac_t = ac_t.permute(2,0,1)
        ac_t = resize(ac_t,(H,W))

        out_vol = ac_t.cpu().numpy()

    elif preservation == AspectRatioPreservationMode.Height:

        H = vol_t.shape[1]
        W = H/aspect_ratio
        D = W * DtoW_rat

        print(f"{D}, W: {W}, H: {H}\n")

        D,W,H = round(D), round(W), round(H)

        ac_t = vol_t.permute(1,2,0)

        ac_t = resize(ac_t,(W,D),interpolation=interpolation)
        ac_t = ac_t.permute(2,0,1)

        out_vol = ac_t.cpu().numpy()

    elif preservation == AspectRatioPreservationMode.Width:

        W = vol_t.shape[2]
        H = W*aspect_ratio
        D = W * DtoW_rat

        print(f"{D}, W: {W}, H: {H}\n")

        D,W,H = round(D), round(W), round(H)

        ac_t = vol_t.permute(2,1,0)

        ac_t = resize(ac_t,(H,D),interpolation=interpolation)
        ac_t = ac_t.permute(2,1,0)

        out_vol = ac_t.cpu().numpy()

    elif preservation == AspectRatioPreservationMode.Depth:

        D = vol_t.shape[0]
        W = D / DtoW_rat
        H = W*aspect_ratio

        print(f"{D}, W: {W}, H: {H}\n")

        D,W,H = round(D), round(W), round(H)

        ac_t = resize(vol_t,(H,W),interpolation=interpolation)

        out_vol = ac_t.cpu().numpy()

    del vol_t, ac_t
    gc.collect()
    torch.cuda.empty_cache()
    
    gpu_mem_clear = (torch.cuda.memory_allocated() == torch.cuda.memory_reserved() == 0)
    print(f"GPU memory is clear: {gpu_mem_clear}\n")

    return out_vol