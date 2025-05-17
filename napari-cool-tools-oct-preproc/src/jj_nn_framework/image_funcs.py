import torch
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from jj_nn_framework.constants import IMAGE_DATA_FORMAT
from jj_nn_framework.torch_utils import torch_interp

def get_shape(x):
    ''''''
    #print(f"x shape: {x.shape}\nx dim: {x.dim()}\n")
    if x.dim() == 4:
        return x.shape # File in NCHW (Batch,Class/Channels,Height,Width)
    elif x.dim() == 3:
        return x.unsqueeze(1).shape # NHW -> NCHW asuming that if you have multichannels you will have already defined them with 4 dim
    elif x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0).shape # HW -> NCHW
    else:
        return None

def bw_1_to_3ch(img_tensor,data_format='NCHW'):
    ''''''
    if data_format == IMAGE_DATA_FORMAT[0]:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)   # N = 1, C = 1, H, W
    elif data_format == IMAGE_DATA_FORMAT[1]:
        img_tensor = img_tensor.unsqueeze(0)                # N = 1, C,H,W
    elif data_format == IMAGE_DATA_FORMAT[2]:
        img_tensor = img_tensor.unsqueeze(1)                # N, C = 1, H,W
    elif data_format == IMAGE_DATA_FORMAT[3]:
        pass                                                # N,C,H,W
    else:
        print(f"Image shape is not of valid format (H,w), (C,H,W), (N,H,W), or (N,C,H,W)\n")
        assert(True == False)

    try:
        assert(img_tensor.size()[1] == 1)
    except:
        print(f"Input has {img_tensor.size()[1]} channels this function only accepts single channel data!!\n")
        return img_tensor
    else:
        img_3ch = img_tensor.expand(-1,3,-1,-1)          # expand channel dimension to create grayscale color image

    return img_3ch

def normalize_in_range(img: torch.Tensor, min_val:float, max_val:float) -> torch.Tensor:
    ''''''
    norm_tensor = (max_val - min_val) * ((img-img.min())/ (img.max()-img.min())) + min_val
    return norm_tensor

def normalize_per_channel(tensor,min_val:float=0,max_val:float=1):
    """"""
    original_shape = tensor.shape
    tensor = tensor.view(-1,*tensor.shape[-2:])

    for channel_idx in range(tensor.shape[-3]):
        tensor[channel_idx] = normalize_in_range(tensor[channel_idx],min_val,max_val)

    tensor = tensor.view(original_shape)

    return tensor

def normalize_per_channel_debug(tensor,min_val:float=0,max_val:float=1):
    """"""
    nan_mask = torch.isnan(tensor)
    nans_present_start = torch.any(nan_mask)
    min_max_start = (tensor.min(),tensor.max())
    if nans_present_start:
        print(f"nans at start?: {nans_present_start}")

    original_shape = tensor.shape
    tensor = tensor.view(-1,*tensor.shape[-2:])

    nan_mask0 = torch.isnan(tensor)
    nans_present_first_view = torch.any(nan_mask0)
    if nans_present_first_view:
        print(f"nans pre reshape?: {nans_present_first_view}\nwere nans at start?: {nans_present_start}")


    for channel_idx in range(tensor.shape[-3]):
        tensor[channel_idx] = normalize_in_range(tensor[channel_idx],min_val,max_val)

    nan_mask2 = torch.isnan(tensor)
    nans_present_pre_reshape = torch.any(nan_mask2)
    min_max_pre_reshape = (tensor.min(),tensor.max())
    if nans_present_pre_reshape:
        print(f"min/max start vs min/max pre_reshape:\n{min_max_start} vs {min_max_pre_reshape}")
        print(f"nans pre reshape?: {nans_present_pre_reshape}\nwere nans at start?: {nans_present_start}")

    tensor = tensor.view(original_shape)

    nan_mask3 = torch.isnan(tensor)
    nans_present_end = torch.any(nan_mask3)
    min_max_end = (tensor.min(),tensor.max())
    if nans_present_end:
        print(f"min/max start vs min/max pre_reshape:\n{min_max_start} vs {min_max_end}")
        print(f"nans at end?: {nans_present_end}\nwere nans at start?: {nans_present_start}")

    return tensor

def pad_to_target(img_tensor,target_shape,mode='constant',value=None):
    '''
    '''
    pad_v = int((target_shape[-2] - img_tensor.size()[-2])/2)
    pad_h = int((target_shape[-1] - img_tensor.size()[-1])/2)
    img_h = F.pad(img_tensor,(pad_h,pad_h,pad_v,pad_v),mode=mode,value=value)

    return img_h

def pad_to_target_2d(img_tensor,target_shape,device='cpu'):
    ''''''
    
    if len(img_tensor.size()) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) # N = 1, C = 1, H, W
    elif len(img_tensor.size()) == 3:

        if img_tensor.size()[0] == 1 or img_tensor.size()[0] == 3: # assumes index 0 is C for single channel or 3 channel RGB image
            img_tensor = img_tensor.unsqueeze(0)
        else:                                                      # assumes index 0 is N aka batch size
            img_tensor = img_tensor.unsqueeze(1)
    elif len(img_tensor.size()) == 4:
        pass
    else:
        print(f"Image shape is not of valid format (N,C,H,W), (N,H,W), or (C,H,W)")
        assert(True == False)
        
    # pad image to appropriate size in this case 800 by 800
    pad_b = img_tensor.size()[0]
    pad_c = img_tensor.size()[1]
    pad_v = int((target_shape[0] - img_tensor.size()[2])/2)
    pad_h = int((target_shape[1] - img_tensor.size()[3])/2)
    
    print(pad_b,pad_v,pad_h)
    
    add_v = torch.zeros((pad_b,pad_c,pad_v,img_tensor.size()[3]),device=device)
    
    img_v = torch.cat((add_v,img_tensor,add_v),dim=2)
    
    print(img_v.size())
    
    add_h = torch.zeros((pad_b,pad_c,img_v.size()[2],pad_h),device=device)
    img_h = torch.cat((add_h,img_v,add_h),dim=3)
    
    return img_h

def pad_to_targetM_2d(img_tensor,target_shape,data_format=None,mode='constant',value=None):
    ''''''

    num_dims = len(img_tensor.size())

    if data_format == None:
        if  num_dims ==  2:
            data_format = 'HW'
        elif num_dims == 3:
            data_format = 'NHW' # NHW -> NCHW asuming that if you have multichannels you will have already defined them with 4 dim
            '''
            if img_tensor.size()[0] == 1 or img_tensor.size()[0] == 3:
                data_format = 'CHW'
            else:
                data_format = 'NHW'
            '''
        elif num_dims == 4:
            data_format = 'NCHW'
        else:
            print(f"Image shape is not of valid format (H,w), (C,H,W), (N,H,W), or (N,C,H,W)\n")
            assert(True == False)

    
    if data_format == IMAGE_DATA_FORMAT[0]:
        try:
            assert(img_tensor.dim() == 2)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)   # N = 1, C = 1, H, W
    elif data_format == IMAGE_DATA_FORMAT[1]:
        try:
            assert(img_tensor.dim() == 3)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            img_tensor = img_tensor.unsqueeze(0)                # N = 1, C,H,W
    elif data_format == IMAGE_DATA_FORMAT[2]:
        try:
            assert(img_tensor.dim() == 3)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            img_tensor = img_tensor.unsqueeze(1)                # N, C = 1, H,W
    elif data_format == IMAGE_DATA_FORMAT[3]:
        try:
            assert(img_tensor.dim() == 4)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            pass                                                # N,C,H,W
    else:
        print(f"Image shape is not of valid format (H,w), (C,H,W), (N,H,W), or (N,C,H,W)\n")
        assert(True == False)
        
    # pad image to appropriate size in this case 800 by 800
    pad_b = img_tensor.size()[0]
    pad_c = img_tensor.size()[1]
    pad_v = int((target_shape[-2] - img_tensor.size()[-2])/2)
    pad_h = int((target_shape[-1] - img_tensor.size()[-1])/2)
    
    #print(pad_b,pad_v,pad_h)
    #
    #add_v = torch.zeros((pad_b,pad_c,pad_v,img_tensor.size()[3]),device=device)
    #
    #img_v = torch.cat((add_v,img_tensor,add_v),dim=2)
    #
    #print(img_v.size())
    #
    #add_h = torch.zeros((pad_b,pad_c,img_v.size()[2],pad_h),device=device)
    #img_h = torch.cat((add_h,img_v,add_h),dim=3)
    
    img_h = F.pad(img_tensor,(pad_h,pad_h,pad_v,pad_v),mode=mode,value=value)

    return img_h

def resize_for_padding(img_tensor,target_shape:List[int],interpolation:InterpolationMode = InterpolationMode.BICUBIC,debug:bool=True):
    """"""
    from torchvision.transforms.functional import resize

    shape = img_tensor.shape
    active_shape = (shape[-2],shape[-1])

    excess = []

    if debug:
            print(f"output dimensions: {img_tensor.shape}\n")

    # check if either dimension exceed the target dimensions add any long dimensions to excess
    for i,dim in enumerate(target_shape):
        if active_shape[i] > target_shape[i]:
            # calculate ratio of new and target dimension
            dim_rat = active_shape[i]/target_shape[i]
            excess.append((i,active_shape[i],dim_rat))

    if len(excess) > 0:
        # select the dimension with the largest new/target rat
        max_dim = max(excess, key=lambda item: item[2])

        # get ratio and output modified dimensions
        rat = max_dim[2]

        out_shape = []

        for dim in active_shape:
            out_shape.append(dim/rat)

        if debug:
            print(f"output dimensions: {out_shape}\n")

        out_shape = [int(item) for item in out_shape]

        out_t = resize(img_tensor,out_shape,interpolation=interpolation)
    
    else:
        out_t = img_tensor

    if debug:
            print(f"output dimensions: {out_t.shape}\n")

    return out_t

def average_per_bscan_pt(img_tensor, scans_per_avg: int = 5, axis = 0, trim: bool = False, ensemble:bool=True, gauss:bool=False):
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)
        ensemble (bool): Flag indicating that ensemble average should be genearated average is calculated for all 3 major axes
                         and the results are then averaged generating a more accurate result at the cost of speed.

    Returns:
        ImageData volume where values at each index each slice is an average of the surrounding bscans from vol
    """
    
    vol_t = img_tensor
    #vol_t = torch.as_tensor(np.ascontiguousarray(vol)) # compare with copy() # as contiguous is tied to the numpy array in memory and would be good for inplace changes
    buffer = int(scans_per_avg/2)
    # select axis
    #if axis !=0:
    #    vol_t = vol_t.swapaxes(0,axis).contiguous()

    ''' Stubb for implementation
    if gauss:
        g_dist = sp_gauss(scans_per_avg,1)
        w = g_dist[:buffer] # distribution will be symmetrical about 1 at current index take initial buffer sized array
    else:
        w = np.ones((buffer,),dtype=np.uint8) # make the weight 1 in case gaussian is not used
    '''

    if not ensemble:

        if axis != 0:
            vol_t = vol_t.swapaxes(0,axis)

        avg_t = torch.zeros_like(vol_t)
        prev_vol = torch.zeros_like(vol_t)
        next_vol = torch.zeros_like(vol_t)

        # calc indicies
        axis_len = vol_t.shape[0]
        idxs = torch.arange(buffer,axis_len-buffer)
        prev_i = idxs - buffer
        next_i = idxs + buffer

        #print(f"prev_i: {prev_i.shape}\n\nindxs: {idxs.shape}\n\nnext_i: {next_i.shape}\n\n")

        prev_vol[idxs] = vol_t[prev_i]
        next_vol[idxs] = vol_t[next_i]

        # generate data between prev or next and buffer and sum
        for i in range(buffer-1):
            prev_vol[idxs] = prev_vol[idxs] + vol_t[prev_i+i]
            next_vol[idxs] = next_vol[idxs] + vol_t[next_i-i]

        #print(f"prev_vol shape: {prev_vol.shape}, vol_i shape: {vol_t[idxs].shape}, next_vol shape: {next_vol.shape}\n")

        # calculate avg
        vol_t[idxs] = ((prev_vol[idxs] + vol_t[idxs] + next_vol[idxs])/scans_per_avg)

        if axis!=0:
            vol_t = vol_t.swapaxes(0,axis)

        avg_out = vol_t

    else:
        vol_t1 = vol_t.swapaxes(0,1)
        vol_t2 = vol_t.swapaxes(0,2)
        
        #vol_t1 = torch.empty_like(vol_t)
        #vol_t2 = torch.empty_like(vol_t)

        axis_0_len = len(vol_t)
        axis_1_len = vol_t.shape[1]
        axis_2_len = vol_t.shape[2]

        idxs_0 = torch.arange(buffer,axis_0_len-buffer)
        prev_i_0 = idxs_0 - 1
        next_i_0 = idxs_0 + 1

        idxs_1 = torch.arange(buffer,axis_1_len-buffer)
        prev_i_1 = idxs_1 - 1
        next_i_1 = idxs_1 + 1

        idxs_2 = torch.arange(buffer,axis_2_len-buffer)
        prev_i_2 = idxs_2 - 1
        next_i_2 = idxs_2 + 1

        prev_vol = torch.zeros_like(vol_t)
        next_vol = torch.zeros_like(vol_t)
        prev_vol_1 = torch.zeros_like(vol_t1)
        next_vol_1 = torch.zeros_like(vol_t1)
        prev_vol_2 = torch.zeros_like(vol_t2)
        next_vol_2 = torch.zeros_like(vol_t2)

        prev_vol[idxs_0] = vol_t[prev_i_0]
        next_vol[idxs_0] = vol_t[next_i_0]
        prev_vol_1[idxs_1] = vol_t1[prev_i_1]
        next_vol_1[idxs_1] = vol_t1[next_i_1]
        prev_vol_2[idxs_2] = vol_t2[prev_i_2]
        next_vol_2[idxs_2] = vol_t2[next_i_2]

        # generate data between prev or next and buffer and sum
        for i in range(buffer-1):
            prev_vol[idxs_0] = prev_vol[idxs_0] + vol_t[prev_i_0+i]
            next_vol[idxs_0] = next_vol[idxs_0] + vol_t[next_i_0-i]
            prev_vol_1[idxs_1] = prev_vol_1[idxs_1] + vol_t1[prev_i_1+i]
            next_vol_1[idxs_1] = next_vol_1[idxs_1] + vol_t1[next_i_1-i]
            prev_vol_2[idxs_2] = prev_vol_2[idxs_2] + vol_t2[prev_i_2+i]
            next_vol_2[idxs_2] = next_vol_2[idxs_2] + vol_t2[next_i_2-i]

        # calculate avg
        vol_t[idxs_0] = ((prev_vol[idxs_0] + vol_t[idxs_0] + next_vol[idxs_0])/scans_per_avg)
        vol_t1[idxs_1] = ((prev_vol_1[idxs_1] + vol_t1[idxs_1] + next_vol_1[idxs_1])/scans_per_avg)
        vol_t2[idxs_2] = ((prev_vol_2[idxs_2] + vol_t2[idxs_2] + next_vol_2[idxs_2])/scans_per_avg)

        vol_t1 = vol_t1.swapaxes(0,1)
        vol_t2 = vol_t2.swapaxes(0,2)

        avg_out = ((vol_t+vol_t1+vol_t2)/3)

    
    # trim result if necessary (perhaps adjust for 3D trim of ensemble and alternate axes)
    if trim:
        trim_offset = int((scans_per_avg-1)/2)
        avg_out = avg_out[trim_offset:len(img_tensor)-trim_offset]


    return avg_out


def average_per_bscan_pt_old(img_tensor, scans_per_avg: int = 5, axis = 0, trim: bool = False, ensemble:bool=True):
    """Function averaging every scans_per_avg images/B-scans centered around each image/b-scan.
    Args:
        vol (ImageData): vol representing volumetric or image stack data
        scans_per_avg (int): number of consecutive images/B-scans to average together
        trim (bool): Flag indicating that ends should be trimmed if image/B-scan index is less than (scans_per_avg - 1 / 2)
        ensemble (bool): Flag indicating that ensemble average should be genearated average is calculated for all 3 major axes
                         and the results are then averaged generating a more accurate result at the cost of speed.

    Returns:
        ImageData volume where values at each index each slice is an average of the surrounding bscans from vol
    """

    #print(f"scans_per_avg (img_func): {scans_per_avg}\n")
    
    img_t = img_tensor
    #img_t = torch.as_tensor(np.ascontiguousarray(vol)) # compare with copy() # as contiguous is tied to the numpy array in memory and would be good for inplace changes
    buffer = int(scans_per_avg/2)
    # select axis
    #if axis !=0:
    #    img_t = img_t.swapaxes(0,axis).contiguous()

    if not ensemble:
        # calc indicies
        axis_len = img_t.shape[axis]
        #idxs = torch.arange(buffer,len(img_t)-buffer)
        idxs = torch.arange(buffer,axis_len-buffer)
        prev_i = idxs - 1
        next_i = idxs + 1

        #avg_t = torch.empty_like(img_t)
        #img_t[idxs] = img_t[prev_i] + img_t[idxs] + img_t[next_i]

        # select axis v2 seems faster than swap method although less programatically robust could verify with timeit
        if axis == 0:
            img_t[idxs] = img_t[prev_i] + img_t[idxs] + img_t[next_i]
        elif axis == 1:
            img_t[:,idxs,:] = img_t[:,prev_i,:] + img_t[:,idxs,:] + img_t[:,next_i,:]
        elif axis == 2:
            img_t[:,:,idxs] = img_t[:,:,prev_i] + img_t[:,:,idxs] + img_t[:,:,next_i]
        else:
            raise ValueError(f"Only 3 axes 0,1,2 are supported")


        # restore axis
        #if axis !=0:
        #    img_t = img_t.swapaxes(0,axis).contiguous()

        avg_t = img_t

    else:
        
        img_t1 = torch.empty_like(img_t)
        img_t2 = torch.empty_like(img_t)

        axis_0_len = len(img_t)
        axis_1_len = img_t.shape[1]
        axis_2_len = img_t.shape[2]

        idxs_0 = torch.arange(buffer,axis_0_len-buffer)
        prev_i_0 = idxs_0 - 1
        next_i_0 = idxs_0 + 1

        idxs_1 = torch.arange(buffer,axis_1_len-buffer)
        prev_i_1 = idxs_1 - 1
        next_i_1 = idxs_1 + 1

        idxs_2 = torch.arange(buffer,axis_2_len-buffer)
        prev_i_2 = idxs_2 - 1
        next_i_2 = idxs_2 + 1

        img_t[idxs_0] = img_t[prev_i_0] + img_t[idxs_0] + img_t[next_i_0]
        img_t1[:,idxs_1,:] = img_t[:,prev_i_1,:] + img_t[:,idxs_1,:] + img_t[:,next_i_1,:]
        img_t2[:,:,idxs_2] = img_t[:,:,prev_i_2] + img_t[:,:,idxs_2] + img_t[:,:,next_i_2]

        avg_t = (img_t + img_t1 + img_t2)/3

    
    # trim result if necessary
    if trim:
        trim_offset = int((scans_per_avg-1)/2)
        avg_t = avg_t[trim_offset:len(avg_t)-trim_offset]


    return avg_t

def a_scan_correction(tensor,rev=False):
    """
    """
    d = tensor.shape[0]
    h = tensor.shape[1]
    w = tensor.shape[2]

    tensor_out = torch.empty_like(tensor)
    Xn = torch.arange(w)

    x_org = (w/2)*torch.sin(torch.pi/w*Xn-torch.pi/2) + (w/2)

    for i in tqdm(range(d),desc="A-scan Correction"):
        for j in range(h):
            if rev:
                tensor_out[i,j,:] = torch_interp(x_org,Xn,tensor[i,j,:])
            else:
                tensor_out[i,j,:] = torch_interp(Xn,x_org,tensor[i,j,:])

    return tensor_out

def rev_ascan_correction(img_tensor,device,mode='bilinear', align_corners:bool=True):
    """"""

    #if img_tensor.ndim != 4:
    #    raise ValueError("Inputs for this function requre for dimensions of format NCHW")
    if img_tensor.ndim == 4:
        d = img_tensor.shape[0]
    elif img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(1)
        d = img_tensor.shape[0]
    elif img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        d = 1
    else:
        raise ValueError("This function only accepts tensors with 2-4 dimimensions")

    h,w = img_tensor.shape[-2],img_tensor.shape[-1]
    n,m = h,w # this could be parameters they do not have to equal the height and width they are the number of samples in linspace

    # normalize height and width values between -1,1 to work with grid_sample
    h_i = torch.linspace(-1, 1, n)
    w_i = torch.linspace(-1, 1, m)

    # calculate indicies for sin corection
    Wn = torch.arange(w)
    x_org = (w/2)*torch.sin(torch.pi/w*Wn-torch.pi/2) + (w/2)

    # normalize values to -1,1 for compatibility with grid_sample
    nw_i = normalize_in_range(x_org,-1,1)

    # create grid indicies
    meshx, meshy = torch.meshgrid((h_i, nw_i))

    grid = torch.stack((meshy, meshx), 2)
    grid = grid.reshape((1,h,w,2))#grid.unsqueeze(0) # add batch dim

    grid = grid.repeat(d,1,1,1).to(device)

    #print(f"\ngird type: {grid.dtype}, img type: {img_tensor.dtype}\ngrid device: {grid.device}, image device: {img_tensor.device}\n")

    out_tensor = torch.nn.functional.grid_sample(img_tensor, grid,mode=mode, align_corners=align_corners)
    
    return out_tensor


    '''
    h = data.shape[-2] #16
    w = data.shape[-1] #16

    n = h
    m = w

    #input = torch.arange(h*w).view(1, 1, h, w).float()
    input = torch.tensor(data.copy()).unsqueeze(0)
    print(input.shape)

    # Create grid to upsample input
    h_i = torch.linspace(-1, 1, n)
    w_i = torch.linspace(-1, 1, m)

    Wn = torch.arange(w)
    x_org = (w/2)*torch.sin(torch.pi/w*Wn-torch.pi/2) + (w/2)

    #x_org_inv = torch.lerp(Wn.float(),x_org,torch.arange(w).float())
    #print(f"x_org_inv shape: {x_org_inv.shape}\n")

    nw_i = normalize_data_in_range_pt_func(x_org,-1,1,numpy_out=False)
    #nw_i_inv = normalize_data_in_range_pt_func(x_org_inv,-1,1,numpy_out=False)

    #print(nw_i.min(),nw_i.max())
    #print(nw_i.shape, nw_i_inv.shape)#

    #print(nw_i[:20],nw_i_inv[:20])

    meshx, meshy = torch.meshgrid((h_i, nw_i))
    #meshx, meshy = torch.meshgrid((h_i, nw_i_inv))

    grid = torch.stack((meshy, meshx), 2)
    grid = grid.unsqueeze(0) # add batch dim

    output = torch.nn.functional.grid_sample(input, grid,mode='nearest', align_corners=True)
    print(output.shape)

    '''