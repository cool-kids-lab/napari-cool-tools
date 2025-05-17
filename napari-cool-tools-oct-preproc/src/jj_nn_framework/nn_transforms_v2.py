


from typing import Any,Dict

import numpy as np
import torch
import torch.nn as nn
from torchvision import tv_tensors
from torchvision.transforms import v2
from PIL import Image

from jj_nn_framework.image_funcs import normalize_in_range,normalize_per_channel,normalize_per_channel_debug

to_tensor = torch.nn.Sequential(
    v2.ToImage(),
    v2.ToDtype(torch.float32,scale=True)
)

class v2_Normalize(v2.Transform):
    def __init__(self, min_val:float=0,max_val:float=1):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

    def _transform(self, inpt: Any, params: Dict[str,Any]):
        if isinstance(inpt,tv_tensors.Image) or isinstance(inpt, torch.Tensor):
            inpt = normalize_per_channel(inpt,self.min_val,self.max_val)
            #inpt = normalize_per_channel_debug(inpt,self.min_val,self.max_val)
            return inpt
        elif isinstance(inpt,Image.Image):
            inpt = to_tensor(inpt)
            return normalize_per_channel(inpt,self.min_val,self.max_val)
            #return normalize_per_channel_debug(inpt,self.min_val,self.max_val)
        else:
            return inpt
        
class v2_NanControl(v2.Transform):
    def __init__(self,instance:int=0):
        super().__init__()

        self.instance = instance

    def _transform(self,inpt:Any, params: Dict[str,Any]):
        if isinstance(inpt,tv_tensors.Image) or isinstance(inpt, torch.Tensor):
            nan_mask = torch.isnan(inpt)
            nans_present = torch.any(nan_mask)

            if nans_present:
                #print(f"There are nan values present in this image at:\n{nan_mask.nonzero().flatten()}\n")
                print(f"NanControl_{self.instance}:\nThere are nan values present in this image ({inpt.shape}) at:\n{nan_mask.nonzero()}\n")
                #inpt = torch.nan_to_num(inpt,nan=0.5,posinf=1.0,neginf=0.0,out=inpt)
                #inpt = torch.nan_to_num(inpt,nan=0.0,posinf=1.0,neginf=0.0,out=inpt)
                inpt = inpt.nan_to_num(nan=0.5,posinf=1.0,neginf=0.0)
                print(f"Post nan_to_num (min,mean,max): ({inpt.min()},{inpt.mean()},{inpt.max()}), shape: {inpt.shape} type: {inpt.dtype}") # troubleshoot later because often shows up as only value
            
            return inpt
        
class v2_InspectImage(v2.Transform):
    def __init__(self,instance:int=0):
        super().__init__()

        self.instance = instance

    def _transform(self,inpt:Any, params: Dict[str,Any]):
        print(f"InspectIMage_{self.instance} Input type: {type(inpt)}\n")
        if isinstance(inpt,tv_tensors.Image) or isinstance(inpt, torch.Tensor):
            nan_mask = torch.isnan(inpt)
            nans_present = torch.any(nan_mask)

            if nans_present:
                print(f"InspectImage_{self.instance}:\nThere are nan values present in this image ({inpt.shape}) at:\n{nan_mask.nonzero()}\n")

            if isinstance(inpt,torch.Tensor):
                #print(f"Image statistics: shape,dtype,min,mean,max:\n\n{inpt.shape},{inpt.dtype},{inpt.min()}{inpt.mean()}{inpt.max()}")
                print(f"Image statistics: type,shape,dtype,min,mean,max:\n{type(inpt)}, {inpt.shape}, {inpt.min()}, {inpt.median()}, {inpt.max()}\n")

        if isinstance(inpt,Image.Image):
            numpy_inpt= np.array(inpt)
            print(f"Image statistics: type,shape,dtype,min,mean,max:\n{type(inpt)}, {numpy_inpt.shape}, {numpy_inpt.dtype}, {numpy_inpt.min()}, {numpy_inpt.mean()}, {numpy_inpt.max()}\n")
            
        return inpt