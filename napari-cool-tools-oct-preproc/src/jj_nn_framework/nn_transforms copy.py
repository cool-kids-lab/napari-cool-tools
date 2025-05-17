"""
"""
import random
from typing import List
from enum import Enum

import torch
import kornia.augmentation as K
import torch.nn as nn
import torchvision.transforms.v2.functional as TF #import torchvision.transforms.v2.functional as TF
from torchvision.transforms import v2 as transforms # from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from kornia.constants import Resample, SamplePadding
from kornia.enhance import equalize_clahe, adjust_log
from kornia.filters import gaussian_blur2d, bilateral_blur, GaussianBlur2d, BilateralBlur

from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target, pad_to_targetM_2d, bw_1_to_3ch, resize_for_padding, average_per_bscan_pt, rev_ascan_correction, get_shape
from jj_nn_framework.mod_utils import check_rand_sample_per_crop, get_crops

class Application(Enum):
    PYTORCH = "pytorch"
    NAPARI = "napari"


# custom transforms

class SqzOutput(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.get_shape = get_shape

    def forward(self, x):

        x = x.squeeze()

        return x

class ResizeToFit(nn.Module):
    def __init__(self, target_shape:tuple[int,int]):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, t):
        x = t[0]
        l = list(t)

        if x.shape[-2] > self.target_shape[-2] or x.shape[-1] > self.target_shape[-1]:
            x = TF.resize(x,self.target_shape)
            l[0] = x

        return tuple(l)

class KorniaRandomBrightness2(nn.Module):
    def __init__(self,rbr_params):
        super().__init__()
        self.RBr = K.RandomBrightness(**rbr_params)
       
    def forward(self, t):
        x=t[0]
        l = list(t)

        #print(f"\nKorniaRandomBrightness x shape: {x.shape}\n")

        x = self.RBr(x.unsqueeze(1)).squeeze()

        l[0] = x

        return tuple(l)

class TorchRandAffine(nn.Module):
    def __init__(self,params):
        super().__init__()
    
        self.params = params
        #self.rand_aff = transforms.RandomAffine(degrees=degrees,translate=translate,scale=scale,shear=shear)

    def forward(self, t):
        x=t[0]
        y=t[1]

        l = list(t)        

        #params = transforms.RandomCrop.get_params(x,output_size=(H,W3))

        params = transforms.RandomAffine.get_params(**self.params,img_size=(x.shape[-2],x.shape[-1]))

        if random.random() > 0.5:
            x = TF.affine(x,*params)
            y = TF.affine(y,*params)
    
        l[0] = x
        l[1] = y

        #print(x.shape)
        return tuple(l)
    
class TorchRandContrast(nn.Module):
    def __init__(self,contrast=[0,10]):
        super().__init__()
    
        self.ct = contrast

    def forward(self, t):
        x=t[0]

        l = list(t)        

        contrast_factor = random.uniform(self.ct[0],self.ct[1])
        x = TF.adjust_contrast(x.unsqueeze(1),contrast_factor=contrast_factor).squeeze()
    
        l[0] = x

        #print(x.shape)
        return tuple(l)
    
class TorchRandBright(nn.Module):
    def __init__(self,brightness=[0,4]):
        super().__init__()
    
        self.br = brightness

    def forward(self, t):
        x=t[0]

        l = list(t)        

        bightness_factor = random.uniform(self.br[0],self.br[1])
        x = TF.adjust_brightness(x.unsqueeze(1),brightness_factor=bightness_factor).squeeze()
    
        l[0] = x

        #print(x.shape)
        return tuple(l)

class TorchRandVFlip(nn.Module):
    def __init__(self,p:float=0.5):
        super().__init__()
    
        self.rvf = transforms.RandomVerticalFlip(p=p)

    def forward(self, t):
        x=t[0]
        y=t[1]

        l = list(t)        

        if random.random() > 0.5:
            x = TF.vertical_flip(x)
            y = TF.vertical_flip(y)
    
        l[0] = x
        l[1] = y

        #print(x.shape)
        return tuple(l)

class TorchRandHFlip(nn.Module):
    def __init__(self,p:float=0.5):
        super().__init__()
        
        self.rhf = transforms.RandomHorizontalFlip(p=p)

    def forward(self, t):
        x=t[0]
        y=t[1]

        l = list(t)

        if random.random() > 0.5:     
            x = TF.horizontal_flip(x)
            y = TF.horizontal_flip(y)
    
        l[0] = x
        l[1] = y

        #print(x.shape)
        return tuple(l)

class RescaleTargetInt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
       
    def forward(self, t):
        y = t[1]
        l = list(t)
        #print(f"\nymin/ymax (before): {y.min()}/{y.max()}\n")
        y = (y * self.num_classes).to(torch.uint8)
        #print(f"\nymin/ymax (post conversion): {y.min()}/{y.max()}\n")

        l[1] = y
            
        return tuple(l)

class TargetToBinary(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        y = t[1]
        l = list(t)
        y = torch.where(y>0,1,0).to(torch.float32)

        l[1] = y
            
        return tuple(l)

class KorniaBBlur(nn.Module):
    def __init__(self,kernel_size,sigma_color,sigma_space):
        super().__init__()
        
        self.kernel_size = tuple(kernel_size)
        self.sigma_color = sigma_color
        self.sigma_space = tuple(sigma_space)
       
    def forward(self, t):
        x = t[0]
        y = t[1]

        x = bilateral_blur(x,self.kernel_size,self.sigma_color,self.sigma_space)

        x = normalize_in_range(x,0,1)

        return(x,y)

class KorniaGBlur(nn.Module):
    def __init__(self,kernel_size,sigma):
        super().__init__()
        
        self.kernel_size = tuple(kernel_size)
        self.sigma = tuple(sigma)
       
    def forward(self, t):
        x = t[0]
        y = t[1]

        x = gaussian_blur2d(x,kernel_size=self.kernel_size,sigma=self.sigma,border_type='reflect')

        x = normalize_in_range(x,0,1)

        return(x,y)

class CLAHE(nn.Module):
    def __init__(self, clahe_clip_limit=2.5):
        super().__init__()
            
        self.clahe_clip = clahe_clip_limit

    def forward(self, t):
        x=t[0]
        l = list(t)

        # Normalize image to 0-1
        x = normalize_in_range(x,0,1)

        # Equalization
        x = equalize_clahe(x,clip_limit=self.clahe_clip)
    
        l[0] = x

        return tuple(l)

class LogCorr(nn.Module):
    def __init__(self, log_gain = 2.5):
        super().__init__()
            
        self.log_gain = log_gain

    def forward(self, t):
        x=t[0]
        l = list(t)

        # Log adjustment
        x = adjust_log(x,gain=self.log_gain)

        # Renormalize to 0-1
        x = normalize_in_range(x,0,1)
    
        l[0] = x

        return tuple(l)

class Normalize(nn.Module):
    def __init__(self, min_val:float=0,max_val:float=1):
        super().__init__()

        self.min_val = min_val
        self.max_val = max_val

    def forward(self, t):
        x=t[0]
        l = list(t)
        
        # Renormalize to 0-1
        x = normalize_in_range(x,self.min_val,self.max_val)
    
        l[0] = x

        return tuple(l)

class Standardize(nn.Module):
    def __init__(self):
        super().__init__()

        def get_dims(x):
            if x.dim() == 4:
                return [0,2,3]
            elif x.dim() == 3:
                return [1,2]
            elif x.dim() == 2:
                return [0,1]
            else:
                return None
        self.get_dims = get_dims

    def forward(self, t):
        x=t[0]

        l = list(t)

        
        # Standardize image
        dims = self.get_dims(x)
        mean,std = x.mean(dims),x.std(dims)
        x = TF.normalize(x,mean,std,inplace=False)
    
        l[0] = x

        return tuple(l)

class RevAscanCorrect(nn.Module):
    def __init__(self,device):

        super().__init__()

        self.device = device

    def forward(self,t):
        x=t[0]
        y=t[1]
        l=list(t)

        x = rev_ascan_correction(x,self.device) 
        y = rev_ascan_correction(y.float(),self.device)

        # rev_ascan_correction has added dimension at dim=1 probably due to how grid sample works consider refactoring
        # refactor may prevent need for squeeze here
        l[0]=x.squeeze()
        l[1]=y.squeeze()

        return tuple(l)

class SinToLinearWarping(nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self,t):
        x=t[0]
        l=list(t)



        l[0]=x

        return tuple(l)

class TypeConversion(nn.Module): # expand in future beyond uint8
    def __init__(self):
        super().__init__()

    def forward(self,t):
        x=t[0]
        l=list(t)

        x = normalize_in_range(x,0,255)
        x =  x.to(torch.uint8)

        l[0] = x

        return tuple(l)

class NapariDisplay(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,t):
        x=t[0]
        l=list(t)

        if x.ndim == 4:
            x = x.permute(1,0,2,3)
        else:
            pass

        l[0] = x

        return tuple(l)

class ConcatDoG(nn.Module):
    def __init__(self,low_sigma, high_sigma, truncate=4.0, gamma=1.0, gain=1.0):
        super().__init__()

        radius_low = round(truncate * low_sigma)
        radius_high = round(truncate * high_sigma)
        self.kernel_low = 2 * radius_low + 1
        self.kernel_high = 2 * radius_high + 1
        self.gamma = gamma
        self.gain = gain

    def forward(self,t):
        x=t[0]
        l=list(t)

        #Code DiffOGaus
        blur_low = TF.gaussian_blur(x,self.kernel_low)
        blur_high = TF.gaussian_blur(x,self.kernel_high)
        x2 = blur_low - blur_high
        x2 = normalize_in_range(x2,0.0,1.0)
        x2 = TF.adjust_gamma(x2,self.gamma,self.gain)

        #print(f"\nx.ndim, x.shape: {x.ndim}, {x.shape}\n")

        if x.ndim == 3:
            x = torch.stack((x,x2),dim=1)
        elif x.ndim == 4:
            x = torch.cat((x,x2),dim=1)

        l[0] = x

        return tuple(l)
    
class InverseIntensity(nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self,t):
        x=t[0]
        l=list(t)

        x = x.max()-x

        l[0]=x

        return tuple(l)

class FlipInverse(nn.Module):
    def __init__(self):

        super().__init__()

    def forward(self,t):
        x=t[0]
        l=list(t)

        x2 = x[:,0:2:,:]
        x3 = x[:,2:,:,:]
        #x2,x3 = torch.split(x,2,dim=1)
        x = torch.concat((x3,x2),dim=1)

        l[0]=x

        return tuple(l)

class ConcatInverse(nn.Module):
    def __init__(self,concat_inv:bool=True):
        super().__init__()
        self.concat_inv = concat_inv

    def forward(self,t):
        x=t[0]
        l=list(t)

        #print(f"in Concat Inverse, x.shape init {x.shape}\n")

        x2 = x.max()-x

        if x.ndim == 3:
            if self.concat_inv:
                x3 = torch.stack((x2,x),dim=1)
                x4 = torch.stack((x,x2),dim=1)
                x_out = torch.cat((x4,x3),dim=0)
            else:
                x_out = torch.stack((x,x2),dim=1)
        elif x.ndim == 4:
            if self.concat_inv:
                x3 = torch.cat((x2,x),dim=1)
                x4 = torch.cat((x,x2),dim=1)
                x_out = torch.cat((x4,x3),dim=0)
            else:
                x_out = torch.cat((x,x2),dim=1)

        #print(f"in Concat Inverse, x.shape out {x_out.shape}\n")
        l[0] = x_out

        return tuple(l)


class GenerateSlidingAverage(nn.Module):
    def __init__(self,scans_per_avg:int=5):
        super().__init__()

        self.scans_per_avg = scans_per_avg

    def forward(self,t):
        x=t[0]
        l=list(t)

        #print(f"scans_per_avg (transform): {self.scans_per_avg}\n")

        #x = average_per_bscan_pt(x,scans_per_avg=self.scans_per_avg,trim=False,ensemble=True)
        x = average_per_bscan_pt(x,scans_per_avg=self.scans_per_avg,trim=False,ensemble=True)

        l[0] = x

        return tuple(l)
    
class StackSlidingAverage(nn.Module):
    def __init__(self,scans_per_avg:int=5):
        super().__init__()

        self.scans_per_avg = scans_per_avg

    def forward(self,t):
        x=t[0]
        l=list(t)
        

        #x = average_per_bscan_pt(x,scans_per_avg=self.scans_per_avg,trim=False,ensemble=True)
        x2 = average_per_bscan_pt(x,scans_per_avg=self.scans_per_avg,trim=False,ensemble=True)
        x = torch.stack((x,x2),dim=1)

        l[0] = x

        return tuple(l)
    
class SqzFormat(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.get_shape = get_shape

    def forward(self, t):
        x=t[0]
        y=t[1]

        l=list(t)

        x = x.squeeze()
        y = y.squeeze()

        l[0] = x
        l[1] = y

        return tuple(l)

class NCHWFormat(nn.Module):
    def __init__(self):
        super().__init__()
            
        self.get_shape = get_shape

    def forward(self, t):
        x=t[0]
        y=t[1]

        l=list(t)

        #if not torch.equal(torch.Tensor(list(x.shape)),torch.tensor(list(y.shape))):
        #if len(x) != len(y):
        #    raise RuntimeError(f"The number of Images and Labels do not match.")
        
        x_shape = self.get_shape(x)
        y_shape = self.get_shape(y)
        x = x.reshape(x_shape)
        y = y.reshape(y_shape)

        l[0] = x
        l[1] = y

        return tuple(l)


class ResizeAndPad(nn.Module):
    def __init__(self,
            target_shape:List[int],
            debug:bool=False
        ):
        super().__init__()

        self.target_shape = target_shape
        self.debug = debug

    def forward(self,t):
        x=t[0]
        y=t[1]
        l = list(t)
        
        # Resize image
        x = resize_for_padding(x,self.target_shape,interpolation=InterpolationMode.BICUBIC,debug=self.debug)
        y = resize_for_padding(y,self.target_shape,interpolation=InterpolationMode.NEAREST_EXACT,debug=self.debug)

        l[0] = x
        l[1] = y

        return tuple(l)
    
class ResizeAspectRatio(nn.Module):
    def __init__(self,fov:int=116):
        super().__init__()
        
        self.fov = fov
        self.rc = transforms.RandomCrop

    def forward(self, t):
        x=t[0]
        y=t[1]

        l = list(t)        

        H = x.shape[-2]
        W = x.shape[-1]
        #H2 = y.shape[-2]
        #W2 = y.shape[-1]

        aspect = (90-(self.fov/2))/self.fov
        W2 = round(H/aspect) # consider using int here
        WW2_rat = W/W2
        W3 = int(W*WW2_rat) # consider using round here
        #print(W3)
        
        #y = self.RHF(y,params=self.RHF._params)
        
        params = transforms.RandomCrop.get_params(x,output_size=(H,W3))
        x = TF.crop(x,*params)
        y = TF.crop(y,*params)
        x = TF.resize(x,(H,W),interpolation=InterpolationMode.BILINEAR)
        y = TF.resize(y,(H,W),interpolation=InterpolationMode.NEAREST_EXACT)
    
        l[0] = x
        l[1] = y

        #print(x.shape)
        return tuple(l)    


class AaronUnetTrainAug(nn.Module):
    def __init__(self,
        hfp,vfp,bp,cp,degrees=90,scale=(0.9,1.1),bf=(0.8,1.2),cf=(0.8,1.2),crs=(224,224),
        device='cpu'):
        super().__init__()
        
        self.device=device
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        self.RCr = K.RandomCrop(size=crs)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        
    def forward(self, t):
        x=t[0]
        l = list(t)
        shape = self.get_shape(x)

        x = normalize_in_range(x,0,1)
        x = adjust_log(x,gain=self.log_gain)
        x = equalize_clahe(x,clip_limit=self.clahe_clip)
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
        #x = bilateral_blur(x,kernel_size=5, sigma_color=(1.0,1.0),border_type='reflect')
        x = self.b_blur(x)
        #x = gaussian_blur2d(x,kernel_size=5,sigma=(1.0,1.0),border_type='reflect')
        x = self.g_blur(x)
        x = normalize_in_range(x,0,1)
        l[0] = x

        return tuple(l)

class JohnUnetTrainAug(nn.Module):
    def __init__(self,
        hfp,vfp,bp,cp,degrees=90,scale=(0.9,1.1),bf=(0.8,1.2),cf=(0.8,1.2),crs=(224,224),cr_shuff=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x,y = self.CLAHE((x,y))

        x = self.RBr(x)
        x = self.RCo(x)

        x,y = self.RCr((x,y))
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)
            
        return x,y
    
class AaronUnetTrainAug2(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        degrees=90,scale=(0.9,1.1),
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),
        bt_flag=True,
        device='cpu'):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        self.RCr = K.RandomCrop(size=crs)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        x,y = self.PAD((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x,y = self.CLAHE((x,y))

        x = self.RBr(x)
        x = self.RCo(x)

        x = self.RCr(x)
        y = self.RCr(y,params=self.RCr._params)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y
    
class AaronUnetTrainAug3(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        log_gain = 2.5, 
        clahe_clip_limit=1.0,
        b_blur_ks = (5,5),
        b_blur_sc = 0.1,
        b_blur_ss = (1.0,1.0),
        b_blur_bt = 'reflect',
        g_blur_ks = (5,5),
        g_blur_s = (1.0,1.0),
        g_blur_bt = 'reflect',
        degrees=90,scale=(0.9,1.1),
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),
        crops_flag = True,
        bt_flag=True,
        device='cpu'):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.BsPreproc = BscanPreproc(log_gain=log_gain,clahe_clip_limit=clahe_clip_limit,
                                      b_blur_ks=b_blur_ks,b_blur_sc=b_blur_sc,b_blur_ss=b_blur_ss,b_blur_bt=b_blur_bt,
                                      g_blur_ks=g_blur_ks,g_blur_s=g_blur_s,g_blur_bt=g_blur_bt)
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        self.RCr = K.RandomCrop(size=crs)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.Crops_flag = crops_flag
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        x,y = self.PAD((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x,y = self.BsPreproc((x,y))

        x = self.RBr(x)
        x = self.RCo(x)

        if self.Crops_flag:
            x = self.RCr(x)
            y = self.RCr(y,params=self.RCr._params)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y

class JohnUnetTrainAug2(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),cr_shuff=True,
        degrees=90,scale=(0.9,1.1),
        bt_flag=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        #print(f"Initial input:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.PAD((x,y))
        #print(f"Post padding:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.CLAHE((x,y))
        #print(f"Post CLAHE:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.RCr((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x = self.RBr(x)
        x = self.RCo(x)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y

class JohnUnetTrainAug4(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),cr_shuff=True,
        degrees=90,scale=(0.9,1.1),
        bt_flag=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHEBlur()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        #print(f"Initial input:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.PAD((x,y))
        #print(f"Post padding:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.CLAHE((x,y))
        #print(f"Post CLAHE:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.RCr((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x = self.RBr(x)
        x = self.RCo(x)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y

class JohnUnetTrainAug5(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),cr_shuff=True,
        degrees=90,scale=(0.9,1.1),
        bt_flag=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        #print(f"Initial input:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.PAD((x,y))
        #print(f"Post padding:\nx shape: {x.shape}, y shape: {y.shape}\n")

        x = self.RBr(x)
        x = self.RCo(x)

        x,y = self.CLAHE((x,y))
        #print(f"Post CLAHE:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.RCr((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y
    
class JohnUnetTrainAug3(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        low_sigma = 1.0,high_sigma = 6.0,truncate = 4.0,gamma = 1.2,gain = 1.0,
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),cr_shuff=True,
        degrees=90,scale=(0.9,1.1),
        bt_flag=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.DoG = DiffOfGaus(low_sigma=low_sigma,high_sigma=high_sigma,truncate=truncate,gamma=gamma,gain=gain)
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        #print(f"Initial input:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.PAD((x,y))
        #print(f"Post padding:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.CLAHE((x,y))
        #print(f"Post CLAHE:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.DoG((x,y))
        x,y = self.RCr((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x = self.RBr(x)
        x = self.RCo(x)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y

class ImageCrops(nn.Module):
    def __init__(self,h,w,shuffle=True,device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.crop_dim = (h,w)
        self.shuffle = shuffle
        self.device = device
        self.verbose = verbose
        self.sanity = sanity

    def forward(self, t):
        
        x = t[0]
        if len(t) > 1:
            y = t[1]
        else:
            y = None
        l = list(t)
        
        #print(f"\n\nImageCrops x/y shapes: {x.shape}/{y.shape}\n\n")

        h,w = x.shape[-2],x.shape[-1]
        kh,kw = self.crop_dim[0], self.crop_dim[1]

        #print(
        #    f"h: {h}, w: {w}, kh: {kh}, kw: {kw}\n"
        #)
        
        h_div = h/kh
        w_div = w/kw
        h_diff = h - kh
        w_diff = w - kw
        kh_h_div_diff = int((h - (kh*int(h_div))) / 2)
        kw_w_div_diff = int((w - (kw*int(w_div))) / 2)
        
        h_start,h_end = 0, h_diff
        w_start,w_end = 0, w_diff
        
        mid_h_start,mid_h_end = kh_h_div_diff,(h - kh_h_div_diff) - kh
        mid_w_start,mid_w_end = kw_w_div_diff,(w - kw_w_div_diff) - kw
        
        if mid_h_start == 0 and mid_h_end == h - kh:
            mid_h_start,mid_h_end = kh,h - kh
        else:
            pass
        
        if mid_w_start == 0 and mid_w_end == w - kw:
            mid_w_start,mid_w_end = kw,w - kw
        else:
            pass

        #print(
        #    f"mid_h_start: {mid_h_start}\n"
        #    f"mid_h_end: {mid_h_end}\n"
        #    f"kh/2: {int(kh/2)}"
        #)

        mid_h_indicies = torch.arange(mid_h_start,mid_h_end+1,int(kh/2),device=self.device)
        mid_w_indicies = torch.arange(mid_w_start,mid_w_end+1,int(kw/2),device=self.device)
        
        start_h = torch.tensor((h_start,),device=self.device)
        end_h = torch.tensor((h_end,),device=self.device) 
        h_indicies = torch.cat((start_h,mid_h_indicies,end_h)).unique()
        
        start_w = torch.tensor((w_start,),device=self.device)
        end_w = torch.tensor((w_end,),device=self.device) 
        w_indicies = torch.cat((start_w,mid_w_indicies,end_w)).unique()
        
        if self.verbose:
            print(f"h_indicies:\n{h_indicies}\nw_indicies:\n{w_indicies}\n")
        
        crops_x,crops_y = get_crops(x,y,kh,kw,h_indicies,w_indicies,shuffle=self.shuffle,device=self.device,verbose=self.verbose)
        
        if self.verbose:
            print(f"num_crops = {len(crops_x)}, shapes per crop x,y: {crops_x[0].shape},{crops_y[0].shape}\n")
        
        if self.sanity:
            check_rand_sample_per_crop(crops_x,crops_y,h_indicies,w_indicies,device=self.device,verbose=self.verbose)

        l[0] = crops_x
        if len(t) > 1:
            l[1] = crops_y
        
        return tuple(l)

class BscanPreproc(nn.Module):
    def __init__(self, 
                 log_gain = 2.5, 
                 clahe_clip_limit=1.0,
                 b_blur_ks = (5,5),
                 b_blur_sc = 0.1,
                 b_blur_ss = (1.0,1.0),
                 b_blur_bt = 'reflect',
                 g_blur_ks = (5,5),
                 g_blur_s = (1.0,1.0),
                 g_blur_bt = 'reflect',
                 ):
        super().__init__()
            
        self.log_gain = log_gain
        self.clahe_clip = clahe_clip_limit
        self.b_blur = BilateralBlur(kernel_size=b_blur_ks,sigma_color=b_blur_sc,sigma_space=b_blur_ss,border_type=b_blur_bt)
        self.g_blur = GaussianBlur2d(kernel_size=g_blur_ks,sigma=g_blur_s,border_type=g_blur_bt)
        self.get_shape = get_shape

    def forward(self, t):
        x=t[0]
        #y=t[1]
        l = list(t)
        shape = self.get_shape(x)

        x = normalize_in_range(x,0,1)
        x = adjust_log(x,gain=self.log_gain)
        x = equalize_clahe(x,clip_limit=self.clahe_clip)
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
        #x = bilateral_blur(x,kernel_size=5, sigma_color=(1.0,1.0),border_type='reflect')
        x = self.b_blur(x)
        #x = gaussian_blur2d(x,kernel_size=5,sigma=(1.0,1.0),border_type='reflect')
        x = self.g_blur(x)
        x = normalize_in_range(x,0,1)
        l[0] = x

        return tuple(l)
    
class BscanPreproc2(nn.Module):
    def __init__(self, 
                 log_gain = 2.5, 
                 clahe_clip_limit=1.0,
                 b_blur_ks = (5,5),
                 b_blur_sc = 0.1,
                 b_blur_ss = (1.0,1.0),
                 b_blur_bt = 'reflect',
                 g_blur_ks = (5,5),
                 g_blur_s = (1.0,1.0),
                 g_blur_bt = 'reflect',
                 ):
        super().__init__()
            
        self.log_gain = log_gain
        self.clahe_clip = clahe_clip_limit
        self.b_blur = BilateralBlur(kernel_size=b_blur_ks,sigma_color=b_blur_sc,sigma_space=b_blur_ss,border_type=b_blur_bt)
        self.g_blur = GaussianBlur2d(kernel_size=g_blur_ks,sigma=g_blur_s,border_type=g_blur_bt)
        self.get_shape = get_shape

    def forward(self, t):
        x=t[0]
        #y=t[1]
        l = list(t)
        shape = self.get_shape(x)

        x = normalize_in_range(x,0,1)
        x = adjust_log(x,gain=self.log_gain)
        x = equalize_clahe(x,clip_limit=self.clahe_clip)
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
        #x = bilateral_blur(x,kernel_size=5, sigma_color=(1.0,1.0),border_type='reflect')
        x = self.b_blur(x)
        #x = gaussian_blur2d(x,kernel_size=5,sigma=(1.0,1.0),border_type='reflect')
        x = self.g_blur(x)
        x = normalize_in_range(x,0,1)
        l[0] = x

        return tuple(l)

class NormalizeCLAHE(nn.Module):
    def __init__(self):
        super().__init__()
        #self.norm_rang = normalize_in_range
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        #x = (x-x.min())/(x.max()-x.min())
        x = normalize_in_range(x,0,1)
        mean,std = x.mean([0,2,3]),x.std([0,2,3])
        #print(x.shape,y.shape)
        #print(mean,std)
        x = TF.normalize(x,mean,std,inplace=False)
        #x = (x-x.min())/(x.max()-x.min())
        x = normalize_in_range(x,0,1)

        x = equalize_clahe(x,clip_limit=40.0)
            
        return (x,y)
    
class NormalizeCLAHE2(nn.Module):
    def __init__(self, 
                 log_gain = 2.5, 
                 clahe_clip_limit=1.0,
                ):
        super().__init__()
            
        self.log_gain = log_gain
        self.clahe_clip = clahe_clip_limit
        self.get_shape = get_shape

    def forward(self, t):
        x=t[0]
        #y=t[1]
        l = list(t)
        shape = self.get_shape(x)

        x = normalize_in_range(x,0,1)
        x = adjust_log(x,gain=self.log_gain)
        x = equalize_clahe(x,clip_limit=self.clahe_clip)
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
    
        l[0] = x

        return tuple(l)
    
class StandNormLog(nn.Module):
    def __init__(self, 
                 log_gain = 2.5, 
                 clahe_clip_limit=1.0,
                ):
        super().__init__()
            
        self.log_gain = log_gain
        self.clahe_clip = clahe_clip_limit

    def forward(self, t):
        x=t[0]
        #y=t[1]
        l = list(t)
        
        # Standardize image
        mean,std = x.mean([0,2,3]),x.std([0,2,3])

        x = TF.normalize(x,mean,std,inplace=False)

        # Normalize image to 0-1
        x = normalize_in_range(x,0,1)

        # Log adjustment
        x = adjust_log(x,gain=self.log_gain)

        # Renormalize to 0-1
        x = normalize_in_range(x,0,1)
    
        l[0] = x

        return tuple(l)

class NapRandResizeAspectRatio(nn.Module):
    def __init__(self,fov:int=116):
        super().__init__()
        
        self.fov = fov
        self.get_shape = get_shape

    def forward(self, t):
        x=t[0]
        #y=t[1]

        l = list(t)
        

        H = x.shape[-2]
        W = x.shape[-1]
        aspect = (90-(self.fov/2))/self.fov
        W2 = round(H/aspect) # consider using int here
        WW2_rat = W/W2
        W3 = int(W*WW2_rat) # consider using round here
        #print(W3)

        rc = transforms.RandomCrop((H,W3),)
        x = rc(x)
        x = TF.resize(x,(H,W))
        
        shape = self.get_shape(x)
        x = x.reshape(shape)
    
        l[0] = x

        #print(x.shape)
        return tuple(l)        

class NapStandNorm(nn.Module):
    def __init__(self):
        super().__init__()
            
        def get_dims(x):
            if x.dim() == 4:
                return [0,2,3]
            elif x.dim() == 3:
                return [1,2]
            elif x.dim() == 2:
                return [0,1]
            else:
                return None
            
        self.get_shape = get_shape
        self.get_dims = get_dims

    def forward(self, t):
        x=t[0]
        #y=t[1]

        #print(x.shape)

        l = list(t)
        shape = self.get_shape(x)
        
        # Standardize image
        dims = self.get_dims(x)
        #mean,std = x.mean([0,1,2]),x.std([0,1,2])
        mean,std = x.mean(dims),x.std(dims)
        x = TF.normalize(x,mean,std,inplace=False)
        #x = (x-mean)/std

        # Normalize image to 0-1
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
    
        l[0] = x

        return tuple(l)
    
class NapCondCLAHELog(nn.Module):
    def __init__(self,log_cor:bool=False):
        super().__init__()
            
        def get_dims(x):
            if x.dim() == 4:
                return [0,2,3]
            elif x.dim() == 3:
                return [0,1,2]
            elif x.dim() == 2:
                return [0,1]
            else:
                return None
        
        self.log_cor = log_cor
        self.get_shape = get_shape
        self.get_dims = get_dims

    def forward(self, t):
        x=t[0]
        #y=t[1]

        #print(x.shape)

        l = list(t)
        shape = self.get_shape(x)
        
        # Standardize image
        dims = self.get_dims(x)
        #mean,std = x.mean([0,1,2]),x.std([0,1,2])
        mean,std = x.mean(dims),x.std(dims)
        x = TF.normalize(x,mean,std,inplace=False)
        #x = (x-mean)/std

        # Normalize image to 0-1
        x = normalize_in_range(x,0,1)

        ct_proxy = x.std([0,1,2])
        print(f"\nct_proxy: {ct_proxy}\n")

        if ct_proxy < 0.05:
            beta = 10 * ct_proxy
            cl = (1 / (1 - beta)) + beta
            print(f"beta: {beta}, clip limit: {cl}\n")
            print(cl)
            x = equalize_clahe(x,clip_limit=cl.item())

        if self.log_cor:
            x = x = adjust_log(x,gain= 1)
        

        x = x.reshape(shape)
    
        l[0] = x

        return tuple(l)

class NapStandNormLog(nn.Module):
    def __init__(self, 
                 log_gain = 2.5, 
                 clahe_clip_limit=1.0,
                ):
        super().__init__()
            
        def get_dims(x):
            if x.dim() == 4:
                return [0,2,3]
            elif x.dim() == 3:
                return [0,1,2]
            elif x.dim() == 2:
                return [0,1]
            else:
                return None
            
        self.log_gain = log_gain
        self.clahe_clip = clahe_clip_limit
        self.get_shape = get_shape
        self.get_dims = get_dims

    def forward(self, t):
        x=t[0]
        #y=t[1]

        #print(x.shape)

        l = list(t)
        shape = self.get_shape(x)
        
        # Standardize image
        dims = self.get_dims(x)
        #mean,std = x.mean([0,1,2]),x.std([0,1,2])
        mean,std = x.mean(dims),x.std(dims)
        x = TF.normalize(x,mean,std,inplace=False)

        # Normalize image to 0-1
        x = normalize_in_range(x,0,1)

        # Log adjustment
        x = adjust_log(x,gain=self.log_gain)

        # Renormalize to 0-1
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
    
        l[0] = x

        return tuple(l)

class NapStandNormLogCLAHE(nn.Module):
    def __init__(self, 
                 log_gain = 2.5, 
                 clahe_clip_limit=1.0,
                ):
        super().__init__()
            
        def get_dims(x):
            if x.dim() == 4:
                return [0,2,3]
            elif x.dim() == 3:
                return [0,1,2]
            elif x.dim() == 2:
                return [0,1]
            else:
                return None
            
        self.log_gain = log_gain
        self.clahe_clip = clahe_clip_limit
        self.get_shape = get_shape
        self.get_dims = get_dims

    def forward(self, t):
        x=t[0]
        #y=t[1]
        l = list(t)
        shape = self.get_shape(x)

        
        # Standardize image
        dims = self.get_dims(x)
        #mean,std = x.mean([0,1,2]),x.std([0,1,2])
        mean,std = x.mean(dims),x.std(dims)
        x = TF.normalize(x,mean,std,inplace=False)

        # Normalize image to 0-1
        x = normalize_in_range(x,0,1)

        # Log adjustment
        x = adjust_log(x,gain=self.log_gain)

        # Equalization
        x = equalize_clahe(x,clip_limit=self.clahe_clip)

        # Renormalize to 0-1
        x = normalize_in_range(x,0,1)

        x = x.reshape(shape)
    
        l[0] = x

        return tuple(l)

class NormalizeCLAHEBlur(nn.Module):
    def __init__(self):
        super().__init__()
        #self.norm_rang = normalize_in_range
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        #x = (x-x.min())/(x.max()-x.min())
        x = normalize_in_range(x,0,1)
        x = adjust_log(x,gain=1)
        mean,std = x.mean([0,2,3]),x.std([0,2,3])
        #print(x.shape,y.shape)
        #print(mean,std)
        norm = transforms.Normalize(mean,std)
        x = norm(x)
        #x = (x-x.min())/(x.max()-x.min())
        x = normalize_in_range(x,0,1)

        x = equalize_clahe(x,clip_limit=3.0)

        x = gaussian_blur2d(x,kernel_size=3,sigma=(1.0,1.0),border_type='reflect')

        x = normalize_in_range(x,0,1)
            
        return (x,y)

class KorniaRandomVerticalFlip(nn.Module):
    def __init__(self,rvf_params):
        super().__init__()
        self.RVF = K.RandomVerticalFlip(**rvf_params)
       
    def forward(self, t):
        x=t[0]
        y=t[1]
        l = list(t)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        l[0] = x
        l[1] = y

        return tuple(l)
    
class KorniaRandomHorizontalFlip(nn.Module):
    def __init__(self,rvf_params):
        super().__init__()
        self.RVF = K.RandomHorizontalFlip(**rvf_params)
       
    def forward(self, t):
        x=t[0]
        y=t[1]
        l = list(t)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        l[0] = x
        l[1] = y

        return tuple(l)
    
class KorniaRandomAffineTransform(nn.Module):
    def __init__(self,raff_params):
        super().__init__()
        self.RAT = K.RandomAffine(**raff_params)
       
    def forward(self, t):
        x=t[0]
        y=t[1]
        l = list(t)

        x = self.RAT(x)
        y = self.RAT(y.to(torch.float16),params=self.RAT._params)

        l[0] = x
        l[1] = y.to(torch.uint8)

        return tuple(l)
    
class KorniaRandomBrightness(nn.Module):
    def __init__(self,rbr_params):
        super().__init__()
        self.RBr = K.RandomBrightness(**rbr_params)
       
    def forward(self, t):
        x=t[0]
        l = list(t)

        #print(f"\nKorniaRandomBrightness x shape: {x.shape}\n")

        x = self.RBr(x)

        l[0] = x

        return tuple(l)
    
class KorniaRandomContrast(nn.Module):
    def __init__(self,ctr_params):
        super().__init__()
        self.RCo = K.RandomContrast(**ctr_params)
       
    def forward(self, t):
        x=t[0]
        l = list(t)

        x = self.RCo(x)

        l[0] = x

        return tuple(l)

class PadToTarget(nn.Module):
    def __init__(self,h,w,device='cpu'):
        super().__init__()
        self.h = h
        self.w = w
        self.device = device
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        x = pad_to_target(x,(self.h,self.w)) #,device=self.device)#pad_to_target_2d(x,(self.h,self.w),device=self.device)
        y = pad_to_target(y,(self.h,self.w)) #,device=self.device)#pad_to_target_2d(y,(self.h,self.w),device=self.device)
            
        return (x,y) 

class PadToTargetM(nn.Module):
    def __init__(self,h,w,
                 X_data_format=None,y_data_format=None,
                 mode='constant',value=None,
                 pad_gt=True,
                 device='cpu'):
        super().__init__()
        self.h = h
        self.w = w
        self.X_data_format=X_data_format
        self.y_data_format=y_data_format
        self.mode = mode
        self.value = value
        self.pad_gt = pad_gt
       
    def forward(self, t):
        t = list(t)

        x = t[0]
        x = normalize_in_range(x,0,1)

        #print(f'x info: {x.shape}, {self.X_data_format}')

        x = pad_to_targetM_2d(x,(self.h,self.w),data_format=self.X_data_format,mode=self.mode,value=self.value)
        t[0] = x

        if self.pad_gt:
            y = t[1]

            #print(f'y info: {y.shape}, {self.y_data_format}')

            y = normalize_in_range(y,0,1)     
            y = pad_to_targetM_2d(y,(self.h,self.w),data_format=self.y_data_format,mode=self.mode,value=self.value)
            t[1] = y
        
            
        return tuple(t)
    
class DiffOfGaus(nn.Module):
    def __init__(self, low_sigma, high_sigma, truncate=4.0, gamma=1.0, gain=1.0):
        super().__init__()

        #self.l_sig = low_sigma
        #self.h_sig = high_sigma
        #self.trunc = truncate

        radius_low = round(truncate * low_sigma)
        radius_high = round(truncate * high_sigma)
        self.kernel_low = 2 * radius_low + 1
        self.kernel_high = 2 * radius_high + 1
        self.gamma = gamma
        self.gain = gain
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        #Code DiffOfGaus
        blur_low = TF.gaussian_blur(x,self.kernel_low)
        blur_high = TF.gaussian_blur(x,self.kernel_high)
        x = blur_low - blur_high
        x = normalize_in_range(x,0.0,1.0)
        x = TF.adjust_gamma(x,self.gamma,self.gain)
            
        return (x,y)
    
class DiffOfGausPred(nn.Module):
    def __init__(self, low_sigma, high_sigma, truncate=4.0, gamma=1.0, gain=1.0):
        super().__init__()

        #self.l_sig = low_sigma
        #self.h_sig = high_sigma
        #self.trunc = truncate

        radius_low = round(truncate * low_sigma)
        radius_high = round(truncate * high_sigma)
        self.kernel_low = 2 * radius_low + 1
        self.kernel_high = 2 * radius_high + 1
        self.gamma = gamma
        self.gain = gain
       
    def forward(self, x):
        
        #Code DiffOGaus
        blur_low = TF.gaussian_blur(x,self.kernel_low)
        blur_high = TF.gaussian_blur(x,self.kernel_high)
        x = blur_low - blur_high
        x = normalize_in_range(x,0.0,1.0)
        x = TF.adjust_gamma(x,self.gamma,self.gain)
            
        return x

class IntTarget(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        y = (y * self.num_classes).to(torch.uint8)
            
        return (x,y)

class BinaryTarget(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        y = torch.where(y>0,1,0).to(torch.float32)
            
        return (x,y)
    
class BWOneToThreeCh_Basic(nn.Module):
    def __init__(self,data_format='CHW'):
        super().__init__()

        self.data_format = data_format
       
    def forward(self, x):

        #print(f"t type: {type(x)}, shape: {x.shape}\n")
        
        x = bw_1_to_3ch(x,self.data_format)
            
        return x

class BWOneToThreeCh(nn.Module):
    def __init__(self,
        data_format='NCHW',
        include_label=False
    ):
        super().__init__()

        self.data_format = data_format
        self.include_label = include_label
       
    def forward(self, t):

        x = t[0]
        y = t[1]

        l = list(t)
        
        x = bw_1_to_3ch(x,self.data_format)

        if self.include_label:
            y = bw_1_to_3ch(y,self.data_format)
            l[1] = y

        l[0] = x
            
        return tuple(l)

class RcsToTensor(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        
        self.device = device
       
    def forward(self, x):
        x.image = TF.to_tensor(x.image)  #.to(self.device)
        x.label = torch.as_tensor(x.label,dtype=torch.int64).permute(2,0,1)       #device=self.device).permute(2,0,1)
            
        return x 

class RcsChannelFirst(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        
        self.device = device

    def forward(self, t):
        x = t[0]
        y = t[1]

        #remove alpha channel from mask
        y = y[:,:,:,:-1]

        x = x.permute(0,3,1,2)
        y = y.permute(0,3,1,2)
        #x.image = TF.rgb_to_grayscale(x.image)
        #x.label = TF.rgb_to_grayscale(x.label)
            
        return x,y

class RcsToGrayscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        x = t[0]
        y = t[1]
        x = TF.rgb_to_grayscale(x)
        y = TF.rgb_to_grayscale(y)
        #x.image = TF.rgb_to_grayscale(x.image)
        #x.label = TF.rgb_to_grayscale(x.label)
            
        return x,y

class RcsResize(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        
        self.new_dim = (h,w)
       
    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = TF.resize(x,self.new_dim)
        y = TF.resize(y,self.new_dim)
        #x.image = TF.resize(x.image,self.new_dim)
        #x.label = TF.resize(x.label,self.new_dim)
            
        return (x,y) 

class RcsRandCrop(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        
        self.crop_dim = (h,w)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        #print(x.shape,y.shape)
        #print(self.crop_dim)
        
        v_diff = x.shape[2] - self.crop_dim[0]
        h_diff = y.shape[3] - self.crop_dim[1]
        #print(v_diff,h_diff)
        
        v = torch.randint(0,v_diff,(1,)).item()
        h = torch.randint(0,h_diff,(1,)).item()
        
        x = TF.crop(x,v,h,self.crop_dim[0],self.crop_dim[1])
        y = TF.crop(y,v,h,self.crop_dim[0],self.crop_dim[1])
        
        return (x,y)

class RcsRandStitchCrop(nn.Module):
    def __init__(self,kh,kw,device):
        super().__init__()
        
        self.kh = kh
        self.kw = kw
        self.device = device

    def forward(self, t):
        # expecting N,C,H,W format
        x = t[0]
        y = t[1]

        kh,kw = self.kh,self.kw
        
        h_idx = int(x.shape[2] / self.kh)
        w_idx = int(x.shape[3] / self.kw)
        
        #print(h_idx,w_idx)
        
        ph = torch.randint(0,h_idx,(1,),device=self.device)
        pw = torch.randint(0,w_idx,(1,),device=self.device)
        
        #print(ph,pw)
        
        x = x[:,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        y = y[:,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        
        return (x,y)

class RcsFullStitchCrop(nn.Module):
    def __init__(self,kh,kw,device):
        super().__init__()
        
        self.kh = kh
        self.kw = kw
        self.device = device

    def forward(self, t):
        # expecting N,C,H,W format
        x = t[0]
        y = t[1]

        kh,kw = self.kh,self.kw
        
        h_idx = int(x.shape[2] / self.kh)
        w_idx = int(x.shape[3] / self.kw)

        perm = torch.randperm(h_idx*w_idx,device=self.device)
        perm2 = perm.view(h_idx,w_idx)

        #print(perm.shape,perm2.shape)
        
        #print(h_idx,w_idx)
        
        ph = torch.randint(0,h_idx,(1,),device=self.device)
        pw = torch.randint(0,w_idx,(1,),device=self.device)

        out_x = torch.empty(len(perm),1,x.shape[1],kh,kw,device=self.device)
        out_y = torch.empty(len(perm),1,y.shape[1],kh,kw,device=self.device)
        
        for i in range(len(perm)):
            #print((perm2==i).nonzero().squeeze())
            ph,pw = (perm2 == i).nonzero().squeeze()
            out_x[i] = x[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
            out_y[i] = y[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        
        #print(ph,pw)
        
        #x = x[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        #y = y[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        
        return (out_x.squeeze(1),out_y.squeeze(1))

class RcsFullStitchCrop2(nn.Module):
    def __init__(self,kh,kw,device):
        super().__init__()
        
        self.kh = kh
        self.kw = kw
        self.device = device

    def forward(self, t):
        # expecting N,C,H,W format
        x = t[0]
        y = t[1]

        kh,kw = self.kh,self.kw
        sh,sw = int(self.kh/2),int(self.kw/2)
        
        h_idx = int(x.shape[2] / self.kh)
        w_idx = int(x.shape[3] / self.kw)

        perm = torch.randperm(h_idx*w_idx,device=self.device)
        perm2 = perm.view(h_idx,w_idx)

        #print(perm.shape,perm2.shape)
        
        #print(h_idx,w_idx)
        
        ph = torch.randint(0,h_idx,(1,),device=self.device)
        pw = torch.randint(0,w_idx,(1,),device=self.device)

        out_x = torch.empty(len(perm),1,x.shape[1],kh,kw,device=self.device)
        out_y = torch.empty(len(perm),1,y.shape[1],kh,kw,device=self.device)
        
        for i in range(len(perm)):
            #print((perm2==i).nonzero().squeeze())
            ph,pw = (perm2 == i).nonzero().squeeze()

            #if i%2 == 0:
            out_x[i] = x[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
            out_y[i] = y[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        
        #print(ph,pw)
        
        #x = x[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        #y = y[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        
        return (out_x.squeeze(1),out_y.squeeze(1))

class RcsNormalize(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        x = (x-x.min())/(x.max()-x.min())
        mean,std = x.mean([0,2,3]),x.std([0,2,3])
        #print(x.shape,y.shape)
        #print(mean,std)
        norm = transforms.Normalize(mean,std)
        x = norm(x)
        x = (x-x.min())/(x.max()-x.min())
            
        return (x,y) 

class RcsBinaryTarget(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        y = torch.where(y>0,1,0).to(torch.float32)
            
        return (x,y) 


class RcsFlipsRot(nn.Module):
    def __init__(self,
        hfp,vfp,degrees,
        device='cpu'):
        super().__init__()
        
        self.device=device
        self.RHF = transforms.RandomHorizontalFlip(p=hfp)
        self.RVF = transforms.RandomVerticalFlip(p=vfp)
        self.Rot = transforms.RandomRotation(degrees=degrees)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.RHF(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.RHF(y)
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.RVF(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.RVF(y)
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.Rot(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.Rot(y)
        
            
        return x,y


class RcsNonRotAff(nn.Module):
    def __init__(self,
        translate,scale,shear,
        device='cpu'):
        super().__init__()
        
        self.RA = transforms.RandomAffine(degrees=0,translate=translate,scale=scale,shear=shear)
        self.device=device

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.RA(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.RA(y)        
            
        return x,y

class RcsGBlur(nn.Module):
    def __init__(self,
        kernel_size,sigma=(0.1,0.2),
        device='cpu'):
        super().__init__()
        
        self.Gblur = transforms.GaussianBlur(kernel_size,sigma=sigma)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.Gblur(x)
           
        return x,y

class RcsSharpCon(nn.Module):
    def __init__(self,
        sf,sp=0.5,acp=0.5,
        device='cpu'):
        super().__init__()
        
        self.RS = transforms.RandomAdjustSharpness(sf,sp)
        self.AC = transforms.RandomAutocontrast(p=acp)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.AC(x)
        x = self.RS(x)
           
        return x,y