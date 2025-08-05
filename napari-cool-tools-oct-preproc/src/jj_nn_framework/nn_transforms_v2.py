from typing import Any, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision import tv_tensors
from torchvision.transforms import v2
from PIL import Image

from jj_nn_framework.image_funcs import (
    normalize_in_range,
    normalize_per_channel,
    normalize_per_channel_debug,
)
from jj_nn_framework.mod_utils import v2_get_crops

to_tensor = torch.nn.Sequential(v2.ToImage(), v2.ToDtype(torch.float32, scale=True))

to_tensor_v2 = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


class v2_Normalize(v2.Transform):
    def __init__(self, min_val: float = 0, max_val: float = 1):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        if isinstance(inpt, tv_tensors.Image) or isinstance(inpt, torch.Tensor):
            inpt = normalize_per_channel(inpt, self.min_val, self.max_val)
            # inpt = normalize_per_channel_debug(inpt,self.min_val,self.max_val)
            return inpt
        elif isinstance(inpt, Image.Image):
            inpt = to_tensor(inpt)
            return normalize_per_channel(inpt, self.min_val, self.max_val)
            # return normalize_per_channel_debug(inpt,self.min_val,self.max_val)
        else:
            return inpt


class v2_CardinalPermutations(v2.Transform):
    _transformed_types: tuple[type | Callable[[Any], bool], ...] = (
        Image.Image,
        tv_tensors.Image,
        torch.Tensor,
        tv_tensors.Mask,
        str,
    )

    def __init__(self):
        super().__init__()

        self.batch_size = 0
        self.image_queue = []
        self.label_quue = []

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        if isinstance(inpt, tv_tensors.Image) or isinstance(inpt, Image.Image):
            if inpt.ndim < 3:
                raise ValueError(
                    f"Input has {inpt.ndim} dimensions, 3 or 4 dimensions are required"
                )
            if inpt.ndim < 4:
                inpt.unsqueeze()

            images = []
            images.append(inpt)
            images.append(v2.functional.horizontal_flip(inpt))
            images.append(v2.functional.vertical_flip(inpt))
            images.append(
                v2.functional.horizontal_flip(v2.functional.vertical_flip(inpt))
            )
            print(f"Processing: {type(inpt)} of shape: {inpt.shape}\n")

            # return torch.stack(images)
            return tv_tensors.Image(torch.cat(images))
        # elif isinstance(inpt, torch.Tensor):
        elif isinstance(inpt, tv_tensors.Mask):
            print(f"Processing: {type(inpt)} of shape {inpt.shape}\n")
            # return inpt.repeat(4)
            return tv_tensors.Mask(inpt.repeat_interleave(4))
        elif isinstance(inpt, str):
            print(f"Processing: {type(inpt)}, {inpt}\n")
            string_multiples = [inpt] * 4
            out_string = ",".join(string_multiples)
            return out_string


class v2_ImageCrops(v2.Transform):
    _transformed_types: tuple[type | Callable[[Any], bool], ...] = (
        Image.Image,
        tv_tensors.Image,
        torch.Tensor,
        tv_tensors.Mask,
        str,
    )

    def __init__(self, h, w, shuffle=True, device="cpu", verbose=False, sanity=False):
        super().__init__()

        self.crop_dim = (h, w)
        self.shuffle = shuffle
        self.device = device
        self.verbose = verbose
        self.sanity = sanity
        self.multiples = 9

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        if isinstance(inpt, tv_tensors.Image) or isinstance(inpt, Image.Image):
            h, w = inpt.shape[-2], inpt.shape[-1]
            kh, kw = self.crop_dim[0], self.crop_dim[1]

            # print(
            #    f"h: {h}, w: {w}, kh: {kh}, kw: {kw}\n"
            # )

            h_div = h / kh
            w_div = w / kw
            h_diff = h - kh
            w_diff = w - kw
            kh_h_div_diff = int((h - (kh * int(h_div))) / 2)
            kw_w_div_diff = int((w - (kw * int(w_div))) / 2)

            h_start, h_end = 0, h_diff
            w_start, w_end = 0, w_diff

            mid_h_start, mid_h_end = kh_h_div_diff, (h - kh_h_div_diff) - kh
            mid_w_start, mid_w_end = kw_w_div_diff, (w - kw_w_div_diff) - kw

            if mid_h_start == 0 and mid_h_end == h - kh:
                mid_h_start, mid_h_end = kh, h - kh
            else:
                pass

            if mid_w_start == 0 and mid_w_end == w - kw:
                mid_w_start, mid_w_end = kw, w - kw
            else:
                pass

            # print(
            #    f"mid_h_start: {mid_h_start}\n"
            #    f"mid_h_end: {mid_h_end}\n"
            #    f"kh/2: {int(kh/2)}"
            # )

            mid_h_indicies = torch.arange(
                mid_h_start, mid_h_end + 1, int(kh / 2), device=self.device
            )
            mid_w_indicies = torch.arange(
                mid_w_start, mid_w_end + 1, int(kw / 2), device=self.device
            )

            start_h = torch.tensor((h_start,), device=self.device)
            end_h = torch.tensor((h_end,), device=self.device)
            h_indicies = torch.cat((start_h, mid_h_indicies, end_h)).unique()

            start_w = torch.tensor((w_start,), device=self.device)
            end_w = torch.tensor((w_end,), device=self.device)
            w_indicies = torch.cat((start_w, mid_w_indicies, end_w)).unique()

            if self.verbose:
                print(f"h_indicies:\n{h_indicies}\nw_indicies:\n{w_indicies}\n")

            crops_inpt = v2_get_crops(
                inpt,
                kh,
                kw,
                h_indicies,
                w_indicies,
                shuffle=self.shuffle,
                device=self.device,
                verbose=self.verbose,
            )

            if self.verbose:
                print(
                    f"num_crops = {len(crops_inpt)}, shapes per crop x,y: {crops_inpt[0].shape},{crops_inpt[0].shape}\n"
                )

            # if self.sanity:
            #    check_rand_sample_per_crop(crops_x,crops_y,h_indicies,w_indicies,device=self.device,verbose=self.verbose)
            assert len(crops_inpt) / len(inpt) % 1 == 0
            self.multiples = int(len(crops_inpt) / len(inpt))
            return tv_tensors.Image(crops_inpt)

        # elif isinstance(inpt, torch.Tensor):
        elif isinstance(inpt, tv_tensors.Mask):
            print(f"Processing: {type(inpt)} of shape {inpt.shape}\n")
            # return inpt.repeat(self.multiples)
            return tv_tensors.Mask(inpt.repeat_interleave(self.multiples))
        elif isinstance(inpt, str):
            print(f"Processing: {type(inpt)}, {inpt}\n")
            string_multiples = [inpt] * self.multiples
            out_string = ",".join(string_multiples)
            return out_string


class v2_NanControl(v2.Transform):
    def __init__(self, instance: int = 0):
        super().__init__()

        self.instance = instance

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        if isinstance(inpt, tv_tensors.Image) or isinstance(inpt, torch.Tensor):
            nan_mask = torch.isnan(inpt)
            nans_present = torch.any(nan_mask)

            if nans_present:
                # print(f"There are nan values present in this image at:\n{nan_mask.nonzero().flatten()}\n")
                print(
                    f"NanControl_{self.instance}:\nThere are nan values present in this image ({inpt.shape}) at:\n{nan_mask.nonzero()}\n"
                )
                # inpt = torch.nan_to_num(inpt,nan=0.5,posinf=1.0,neginf=0.0,out=inpt)
                # inpt = torch.nan_to_num(inpt,nan=0.0,posinf=1.0,neginf=0.0,out=inpt)
                inpt = inpt.nan_to_num(nan=0.5, posinf=1.0, neginf=0.0)
                print(
                    f"Post nan_to_num (min,mean,max): ({inpt.min()},{inpt.mean()},{inpt.max()}), shape: {inpt.shape} type: {inpt.dtype}"
                )  # troubleshoot later because often shows up as only value

            return inpt


class v2_InspectImage(v2.Transform):
    def __init__(self, instance: int = 0):
        super().__init__()

        self.instance = instance

    def _transform(self, inpt: Any, params: Dict[str, Any]):
        print(f"InspectIMage_{self.instance} Input type: {type(inpt)}\n")
        if isinstance(inpt, tv_tensors.Image) or isinstance(inpt, torch.Tensor):
            nan_mask = torch.isnan(inpt)
            nans_present = torch.any(nan_mask)

            if nans_present:
                print(
                    f"InspectImage_{self.instance}:\nThere are nan values present in this image ({inpt.shape}) at:\n{nan_mask.nonzero()}\n"
                )

            if isinstance(inpt, torch.Tensor):
                # print(f"Image statistics: shape,dtype,min,mean,max:\n\n{inpt.shape},{inpt.dtype},{inpt.min()}{inpt.mean()}{inpt.max()}")
                print(
                    f"Image statistics: type,shape,dtype,min,mean,max:\n{type(inpt)}, {inpt.shape}, {inpt.min()}, {inpt.median()}, {inpt.max()}\n"
                )

        if isinstance(inpt, Image.Image):
            numpy_inpt = np.array(inpt)
            print(
                f"Image statistics: type,shape,dtype,min,mean,max:\n{type(inpt)}, {numpy_inpt.shape}, {numpy_inpt.dtype}, {numpy_inpt.min()}, {numpy_inpt.mean()}, {numpy_inpt.max()}\n"
            )

        return inpt
