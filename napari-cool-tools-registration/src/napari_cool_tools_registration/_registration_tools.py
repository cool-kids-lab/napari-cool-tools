"""
This module contains code for registering volumetric and image data.
"""
import numpy as np
from typing import Dict,List
from tqdm import tqdm
from napari.utils.notifications import show_info, show_error
from napari.layers import Image, Layer
from napari.types import ImageData
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch, viewer, memory_stats
from napari_cool_tools_img_proc._normalization_funcs import normalize_data_in_range_func, normalize_data_in_range_pt_func
from napari_cool_tools_registration._registration_tools_funcs import a_scan_correction_func2, adjust_aspect_ratio, a_scan_reg_calc_settings, a_scan_subpix_registration, InterpolationMode, AspectRatioPreservationMode, ArrayImplementation

def adjust_aspect_ratio_plugin(vol:Layer,fov:float=116.0,interpolation:InterpolationMode=InterpolationMode.BICUBIC,preservation:AspectRatioPreservationMode=AspectRatioPreservationMode.Volume):
    """"""
    adjust_aspect_ratio_thread(
        vol=vol,
        fov=fov,
        interpolation=interpolation,
        preservation=preservation
    )

    return

@thread_worker(connect={"returned": viewer.add_layer})
def adjust_aspect_ratio_thread(vol:Layer,fov:float=116.0,interpolation:InterpolationMode=InterpolationMode.BICUBIC,preservation:AspectRatioPreservationMode=AspectRatioPreservationMode.Volume) -> Layer:
    """"""
    
    show_info(f'Aspect Ratio thread has started')
    name = f"{vol.name}({fov})_AR_Corr"
    add_kwargs = {"name":name}
    layer_type = vol.as_layer_data_tuple()[2]

    if layer_type == "labels":
        interpolation = InterpolationMode.NEAREST_EXACT
    else:
        interpolation = interpolation

    vol_out = adjust_aspect_ratio(
        vol=vol.data,
        fov=fov,
        interpolation=interpolation,
        preservation = preservation
    ) 

    layer = Layer.create(vol_out,add_kwargs,layer_type)
    memory_stats()

    show_info(f'Aspect Ratio thread has completed')

    return layer


def a_scan_correction(lay:Layer,rev:bool=False,implementation:ArrayImplementation=ArrayImplementation.NUMPY):
    """"""
    a_scan_correction_thread2(lay=lay,rev=rev,implementation=implementation)

    return

@thread_worker(connect={"returned": viewer.add_layer})
def a_scan_correction_thread2(lay:Layer,rev:bool=False,implementation:ArrayImplementation=ArrayImplementation.NUMPY) -> Layer:
    ''''''
    show_info(f'A-scan correction thread has started')
    name = f"{lay.name}_AS_Corr"
    add_kwargs = {"name":name}
    layer_type = "image" #lay.as_layer_data_tuple()[2]
    if layer_type == "labels":
        lay_out = a_scan_correction_func2(data=lay.data,rev=rev,norm_out=True,implementation=implementation)
        #print(np.unique(lay_out))
        num_labels = 3
        lay_out = lay_out * (num_labels-1)
        lay_out = lay_out.astype(np.uint8)
    else:
        lay_out = a_scan_correction_func2(data=lay.data,rev=rev,norm_out=True,implementation=implementation)
    layer = Layer.create(lay_out,add_kwargs,layer_type)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'A-scan correction thread has completed')

    return layer

def a_scan_reg_calc_settings(img:Image, min_regs:int=3, max_regs:int=8):
    """Calculate optimal number of regions to divide volume into for a-scan registration.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        max_regs (int): maximum number of regions that will be considered for a-scan registration

    Returns:
        int indicating optimal number of regions to use when performing a-scan registration
    """
    def display_settings(settings):
        show_info(f"Optimal number of regions: {settings['region_num']}")
        show_info(f"Avg phase difference: {settings['avg_phase_diff']}")
        return
    
    worker = a_scan_reg_calc_settings_thread(img=img,min_regs=min_regs,max_regs=max_regs)
    worker.returned.connect(display_settings)
    worker.start()

@thread_worker(progress=True)
def a_scan_reg_calc_settings_thread(img:Image, min_regs:int=3, max_regs:int=8) -> Dict:
    """Calculate optimal number of regions to divide volume into for a-scan registration.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        max_regs (int): maximum number of regions that will be considered for a-scan registration

    Returns:
        int indicating optimal number of regions to use when performing a-scan registration
    """
    show_info(f'A-scan region calc thread has started')
    ascan_settings = a_scan_reg_calc_settings(data=img.data,min_regs=min_regs,max_regs=max_regs)
    show_info(f'A-scan region calc thread has completed')

    return ascan_settings

def a_scan_reg_subpix_old(vol:Image, sections:int=4, sub_pixel_threshold:float=0.5, fill_gaps=True, roll_over_flag=True):
    ''''''
    a_scan_reg_subpix_thread(vol=vol,sections=sections,sub_pixel_threshold=sub_pixel_threshold,fill_gaps=fill_gaps,roll_over_flag=roll_over_flag)

    return

def a_scan_reg_subpix(img:Image, sections:int=4, sub_pixel_threshold:float=0.5, fill_gaps:bool=True, roll_over_flag:bool=True,debug:bool=False):
    ''''''
    a_scan_reg_subpix_thread(img=img,sections=sections,sub_pixel_threshold=sub_pixel_threshold,fill_gaps=fill_gaps,roll_over_flag=roll_over_flag,debug=debug)

    return

@thread_worker(connect={"yielded": viewer.add_layer},progress=True)
def a_scan_reg_subpix_thread_old(vol:Image, sections:int=4, sub_pixel_threshold:float=0.5, fill_gaps=True,roll_over_flag=True) -> List[Layer]:
    ''''''

    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift
    show_info(f'A-scan registration thread has started')

    #sections = 4
    data = vol.data
    name = vol.name

    replace_val = round(data.max()+2)

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_ascan_subpix_reg"}

    # optional layer type argument
    layer_type = "image"

    even = data[::2,:,:]
    odd = data[1::2,:,:]

    sec_len = int(data.shape[0] / sections)

    regions = []

    for s in tqdm(range(sections),desc="Creating Regions\n"):
        start = s*sec_len
        end = start + sec_len
        curr = data[:,:,start:end]
        regions.append(curr)

    shifts = []

    for r in tqdm(regions,desc="Calculating Shifts for region\n"):
        even = r[::2,:,:]
        odd = r[1::2,:,:]
        shift, error, diffphase = phase_cross_correlation(even, odd, upsample_factor=100)
        print(f"\n\nshift:{shift}\nerror: {error}\ndiffphase: {diffphase}\n\n")
        shifts.append(shift[2])

    out_reg = []
    roll_overs = []

    for i,s in tqdm(enumerate(shifts),desc="Shifting Regions\n"):
        shift2 = round(s)
        shift_idx = abs(shift2)

        out = np.empty_like(regions[i])
        out[::2,:,:] = regions[i][::2,:,:]

        input_ = np.fft.fft2(regions[i][1::2,:,:])
        result = fourier_shift(input_,(0.0,0.0,s),axis=2)
        result = np.fft.ifft2(result)
        odd_out = result.real

        if s < 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,-shift_idx:].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,-shift_idx:] = replace_val

                
                if i > 0 and shifts[i-1] < 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                    out_reg_odd = out_reg[i-1][1::2,:,:]

                    axis_0_len,axis_1_len = out_reg_odd.shape[0], out_reg_odd.shape[1]

                    print(f"\nout_reg_odd shape: {out_reg_odd.shape}\n")

                    print(f"\nout_reg_odd[{i-1}] range: ({out_reg_odd.min()}, {out_reg_odd.max()})\n")

                    gap_idx = (out_reg_odd == replace_val)

                    gap = out_reg_odd[gap_idx]

                    gap.shape = (axis_0_len,axis_1_len,-1)

                    print(f"\nout_reg gap shape: {gap.shape}\nroll_over shape: {roll_over.shape}\n")

                    gap_roll_diff = gap.shape[2] - roll_over.shape[2]

                    if roll_over_flag:
                        if gap_roll_diff >= 0:
                            out_reg[i-1][1::2,:,-roll_over.shape[2]:] = roll_over
                            print(f"\npositive or neutral gap_roll_diff s < 0\n")
                        elif gap_roll_diff < 0:
                            out_reg[i-1][1::2,:,-gap.shape[2]:] = roll_over[:,:,gap.shape[2]:]
                            print(f"\nnegative gap_roll_diff s > 0\n")


            else:
                roll_overs.append(None)

        elif s > 0:

            if abs(s) >= sub_pixel_threshold:
                roll_over = odd_out[:,:,:shift_idx].copy()
                roll_overs.append(roll_over)
                odd_out[:,:,:shift_idx] = replace_val

            
                if i > 0 and shifts[i-1] > 0 and abs(shifts[i-1]) > sub_pixel_threshold:

                        prev_roll_over = roll_overs[i-2]

                        print(f"\nrollovers length: {len(roll_overs)}\n i: {i-2}\n")

                        gap_roll_diff = shift_idx - roll_over.shape[2]

                        print(f"\nshift_idx: {shift_idx}, roll_over shape[2]: {prev_roll_over.shape[2]}\n")

                        if roll_over_flag:
                            if gap_roll_diff >= 0:
                                odd_out[:,:,:prev_roll_over.shape[2]] = prev_roll_over
                                print(f"\npositive or neutral gap_roll_diff s > 0\n")

                            elif gap_roll_diff < 0:
                                odd_out[:,:,:shift_idx] = prev_roll_over[:,:,:shift_idx]
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

        # optional kwargs for viewer.add_* method
        add_kwargs2 = {"name": f"{name}_ascan_subpix_reg_debug"}

        # optional layer type argument
        layer_type2 = "labels"

        debug = np.empty_like(output)
        debug[gap_idxs] = 1
        debug = debug.astype('uint8')
        
        debug_label = Layer.create(debug,add_kwargs2,layer_type2)
        yield debug_label

        gap_starts = (init_idxs < final_idxs)
        gap_ends = (init_idxs > final_idxs)

        gap_starts = gap_starts.nonzero()
        gap_ends = gap_ends.nonzero()

        print(f"\ngap starts: {gap_starts}\ngap ends: {gap_ends}\n")

        gap_start_idxs = np.unique(gap_starts[2])
        gap_end_idxs = np.unique(gap_ends[2])

        print(f"\ngap starts: {gap_start_idxs}\ngap ends: {gap_end_idxs}\n")


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
                    #print(f"\n\n\n\nCondition ONE!!!!!!!!!!!!!!!\n\n\n\n")
                else:
                    s_idx = gap_start_idxs[i-1]
                    e_idx = gap_end_idxs[i] + 1

                    print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")
                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1
                    print(f"\nnum: {num}\n")
                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
                    print(f"\nln_interp shape: {ln_interp.shape}, interp_out shape: {interp_out.shape}\n")
                    output[1::2,:,s_idx:e_idx+1] = interp_out
                    #print(f"\n\n\n\nCondition TWO!!!!!!!!!!!!!!!\n\n\n\n")

            elif  len(gap_start_idxs) >= len(gap_end_idxs):
            
                if i < len(gap_end_idxs):
                    s_idx = gap_start_idxs[i]
                    e_idx = gap_end_idxs[i] + 1

                    print(f"\ns_idx: {s_idx}, e_idx: {e_idx}\n")
                    start = output[1::2,:,s_idx]
                    end = output[1::2,:,e_idx]
                    num = (e_idx - s_idx) + 1
                    print(f"\nnum: {num}\n")
                    ln_interp = np.linspace(start,end,num=num)
                    interp_out = np.transpose(ln_interp,(1,2,0))
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
    
    layer = Layer.create(output,add_kwargs,layer_type)

    show_info(f'A-scan registration thread has completed')
    yield layer

@thread_worker(connect={"yielded": viewer.add_layer},progress=True)
def a_scan_reg_subpix_thread(img:Image, sections:int=4, sub_pixel_threshold:float=0.5, fill_gaps:bool=True,roll_over_flag:bool=True,debug:bool=False) -> List[Layer]:
    ''''''

    from skimage.registration import phase_cross_correlation

    show_info(f'A-scan registration thread has started')

    #sections = 4
    data = img.data
    name = img.name

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_ascan_subpix_reg"}

    # optional layer type argument
    layer_type = "image"

    even = data[::2,:,:]
    odd = data[1::2,:,:]

    sec_len = int(data.shape[0] / sections)

    regions = []

    for s in tqdm(range(sections),desc="Creating Regions\n"):
        start = s*sec_len
        end = start + sec_len
        curr = data[:,:,start:end]
        regions.append(curr)

    shifts = []

    for r in tqdm(regions,desc="Calculating Shifts for region\n"):
        even = r[::2,:,:]
        odd = r[1::2,:,:]
        shift, error, diffphase = phase_cross_correlation(even, odd, upsample_factor=100)
        print(f"\n\nshift:{shift}\nerror: {error}\ndiffphase: {diffphase}\n\n")
        shifts.append(shift[2])

    settings = {}
    settings["regions"] = regions
    settings["shifts"] = shifts

    output = a_scan_subpix_registration(data,settings,sub_pixel_threshold=sub_pixel_threshold,fill_gaps=fill_gaps,roll_over_flag=roll_over_flag,debug=debug)

    print("You made it here too!!\n")
    print(f"out_data shape : {output}, type: {type(output)}\n")

    for out_data in output:
        
        layer = Layer.create(out_data,add_kwargs,layer_type)
        yield layer

    show_info(f'A-scan registration thread has completed')


def a_scan_reg_subpix_gen(img:Image, settings:Dict, sub_pixel_threshold:float=0.5, fill_gaps=True,roll_over_flag=True,debug=False) -> List[Layer]:
    ''''''

    #from skimage.registration import phase_cross_correlation
    
    show_info(f'A-scan registration thread has started')

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{img.name}_ascan_subpix_reg"}

    output = a_scan_subpix_registration(img.data,settings=settings,sub_pixel_threshold=sub_pixel_threshold,fill_gaps=fill_gaps,roll_over_flag=roll_over_flag,debug=debug)

    # optional layer type argument
    layer_type = "image"    

    layer = Layer.create(output,add_kwargs,layer_type)

    show_info(f'A-scan registration thread has completed')

    yield layer


def optical_flow_registration(img_seq:Image, slices_to_register:str, target_idx_from_list:int, mode:str='ilk'):
    """Register sequence of images using optical flow estimation via either Iterative Lucas-Kanade sover 'ilk' or TV-L1 solver 'tvl1'.

    Args:
        img_seq (Image): Stack of 2D image data to be registered
        slices_to_register (str): comma separated list of index values of the array slices to be included in the registration process
                                  By default (if no str is provided) all images from index 0 are included
        target_idx_from_list (int): indicates which index from the slices_to_register string will be the target against which all other
                                    slices will be registered.  By default index 0 is used so the first index in the slices_to_register list
                                    will be the target of registration unless another index is chosen.  This paramater refers to the slices_to_register
                                    list and not all of the indicies included in the image sequence

                                    If slices_to_register = "5,4,3,9,1,0" and target_idx_from_list = 3 then images at indicies 5,4,3,1,0 will be registerd
                                    to the image at index 9

    Returns:
        Layer containing the averaged image of the images at indicies selected in slices_to_register registered to the image at the index indicated by target_idx_from_list
    """

    try:
        assert mode == 'ilk' or mode == 'tvl1', "Mode must be 'ilk' or 'tvl1' all other inputs are invalid"
    except AssertionError as e:
        raise Exception("An error Occured:", str(e))
    else:
        show_info(f"Mode: {mode} is valid\n")
        optical_flow_registration_thread(img_seq,slices_to_register,target_idx_from_list)

    return


@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def optical_flow_registration_thread(img_seq:Image, slices_to_register:str, target_idx_from_list:int, mode:str='ilk') -> Layer:
    """Register sequence of images using optical flow estimation via either Iterative Lucas-Kanade sover 'ilk' or TV-L1 solver 'tvl1'.

    Args:
        img_seq (Image): Stack of 2D image data to be registered
        slices_to_register (str): comma separated list of index values of the array slices to be included in the registration process
                                  By default (if no str is provided) all images from index 0 are included
        target_idx_from_list (int): indicates which index from the slices_to_register string will be the target against which all other
                                    slices will be registered.  By default index 0 is used so the first index in the slices_to_register list
                                    will be the target of registration unless another index is chosen.  This paramater refers to the slices_to_register
                                    list and not all of the indicies included in the image sequence

                                    If slices_to_register = "5,4,3,9,1,0" and target_idx_from_list = 3 then images at indicies 5,4,3,1,0 will be registerd
                                    to the image at index 9

    Returns:
        Layer containing the averaged image of the images at indicies selected in slices_to_register registered to the image at the index indicated by target_idx_from_list
    """

    show_info(f'optical_flow_registration thread has started')

    name = img_seq.name

    # optional layer type argument
    layer_type = "image"

    img_seq = img_seq.data

    target_slice = target_idx_from_list
    
    if slices_to_register == '':
        a_slices = np.arange(img_seq.shape[0])
    else:
        # check that values in old_vals are integers
        slice_list = slices_to_register.split(',')

        try:
            possible_nums = [int(x.strip()) for x in slice_list]
            valid_nums = all([isinstance(item, int) for item in possible_nums])
            assert valid_nums, f'old_vals input is invalid!! old_vals accepts comma separated list of integer values only.'
        except ValueError as e:
            raise Exception("An error Occured:", str(e))
        else:
            a_slices = np.array([int(x.strip()) for x in slice_list])

    # move target slice to index 0
    temp = a_slices[target_slice].copy()
    a_slices = np.delete(a_slices,target_slice,axis=0)
    a_slices = np.insert(a_slices,0,temp,axis=0)

    # get active portion of image sequence
    print(f'a_slices: {a_slices}')
    active = img_seq[(a_slices)].copy()

    registered = np.empty_like(active)
    registered[0] = active[target_slice].copy()


    for i,s in enumerate(active[1:]):

        # compute optical flow
        flow_warp = opti_flow_internal(active[0],s,mode=mode)

        # collect registered images
        registered[i] = flow_warp

        print(f'Index: {i+1} of {active[1:].shape[0]} registered.')

    reg_out = registered.mean(axis=0)

    # optional kwargs for viewer.add_* method
    add_kwargs = {"name": f"{name}_idx_{a_slices[0]}_OFR"}

    layer = Layer.create(reg_out,add_kwargs,layer_type)

    show_info(f'optical_flow_registration thread has completed')

    return layer


def opti_flow_internal(img:ImageData,img2:ImageData,mode:str = 'ilk'):
    ''''''

    from skimage.registration import optical_flow_tvl1, optical_flow_ilk
    from skimage.transform import warp

    r_shape = img.shape

    # compute optical flow
    if mode == 'ilk':
        v,u = optical_flow_ilk(img,img2)
    else:
        v,u = optical_flow_tvl1(img,img2)

    # register
    row, col = r_shape
    row_coords, col_coords = np.meshgrid(np.arange(row),np.arange(col), indexing='ij')

    flow_warp = warp(img2,np.array([row_coords + v,col_coords + u]),mode='edge')

    return flow_warp

def m_scan_registration(vol:Image,m_scans:int=3,m_idx:int=1,subpixel:bool=True,debug:bool=False):
    """"""
    data = vol.data
    name = f"{vol.name}"

    # case 4D OCTA data
    if data.ndim == 4:
        show_info(f"Not yet implemented {data.ndim}-dimensional data.")
        pass
    # case continuous OCTA data
    elif data.ndim == 3:
        shifts = []
        for b_idx in range(int(data.shape[0]/m_scans)):
        #for b_idx in tqdm(range(int(data.shape[0]/m_scans)),desc="Registering m-scans"):
            curr_idx = b_idx*m_scans
            target_idx = curr_idx + m_idx

            m_idx_set = set(range(m_scans))
            reg_idxs = np.array(list(m_idx_set - {m_idx}))+curr_idx

            if debug:
                #print(f"current idx: {curr_idx}\ntarget idx: {target_idx}\n")
                #print(f"m_idx_set: {m_idx_set}\nreg_idxs: {reg_idxs}\n")
                pass
            else:
                pass

            #if curr_idx == 1200:
            for i in reg_idxs:
                if debug:
                    show_info(f"Registering m-scan{i} to m-scan{target_idx}\n")
                else:
                    pass
                target_frame = data[curr_idx,:,:]
                reg_frame = data[i,:,:]
                registered = register_frame_2d(target_frame,reg_frame,shifts,subpixel=subpixel,debug=debug)
                data[i,:,:] = registered[:,:]
                
        shift_data = np.stack(shifts)
        avg_shift = shift_data.mean(0)
        #v_shifts = len(shift_data[:,0,:].nonzero())
        #h_shifts = len(shift_data[:,1,].nonzero())

        show_info(f"Avg shift: {avg_shift}\n")

    # case < 3D data or > 4D data
    elif data.ndim < 3 or data.ndim > 4:
        show_error(f"Invalid {data.ndim}-dimensional data.")

def register_frame_2d(target_frame:ImageData,
                   registering_frame:ImageData,
                   shifts,
                   subpixel:bool=True,
                   debug:bool=False):
    """"""
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import fourier_shift

    shift, error, diffphase = phase_cross_correlation(
        target_frame, registering_frame, upsample_factor=100
    )

    shifts.append(shift)

    if subpixel:

        if debug:
            show_info(f"shift: {shift}\n")
        else:
            pass

        input_ = np.fft.fft2(registering_frame)
        result = fourier_shift(input_,(shift[0],shift[1]),axis=-1)
        result = np.fft.ifft2(result)
        reg_shift = result.real

        #viewer.add_image(reg_shift)

    else:
        shift_h = round(shift[1])
        shift_w = round(shift[0])
        shift_h_idx = abs(shift_h)
        shift_w_idx = abs(shift_w)

        if debug:
            print(f"shift: {shift}\nshift_w: {shift_w}\nshift_h: {shift_h}\n")
        else:
            pass

        if shift_h > 0:
            reg_shift = np.roll(registering_frame, shift_w, axis=0)
            reg_shift = np.roll(registering_frame, shift_h, axis=1)
            reg_shift[-shift_w_idx:, :] = 0
            reg_shift[:, -shift_h_idx:] = 0
        else:
            reg_shift = registering_frame

    return reg_shift
