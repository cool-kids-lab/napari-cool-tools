# code based on code from https://www.learnpytorch.io/05_pytorch_going_modular/
"""
Contains PyTorch model code to instantiate a No Need for Routing Between Capsules model.
"""
import torch
import math
import torch.nn.functional as F
from torch import nn

# custom models

class UnetJJ(nn.Module):
    def __init__(self, input_shape, init_filters, depth=5, kernel_size=3, padding=1, device='cpu'):
        super().__init__()

        B,C,H,W = input_shape

        self.depth = depth
        self.in_chs = [C if i == 0 else init_filters * pow(2,i-1) for i in range(depth)]
        self.out_chs = [init_filters if i == 0 else 2* init_filters * pow(2,i-1) for i in range(depth)]
        self.Contracting = nn.ModuleList(
            UnetEncoderBlock(self.in_chs[i],self.out_chs[i],kernel_size=kernel_size, device=device  
            )\
            for i in range(depth-1)
        )
        self.Base = UnetConvBlock(self.in_chs[-1],self.out_chs[-1],kernel_size=kernel_size, device=device)
        
    
        def build_decoder():
            self.Expanding = nn.ModuleList(
                UnetDecoderBlock(self.Contracting[-(i+1)].skip,self.out_chs[-(i+1)],self.in_chs[-(i+1)],kernel_size=kernel_size, device=device
                )\
                for i in range(depth-1)
            )
            
        self.build_decoder = build_decoder
        self.FinalConv = nn.Conv2d(
            self.out_chs[0],self.in_chs[0],kernel_size=kernel_size,padding=padding,device=device
        )
        self.FinalActivation = nn.Sigmoid()
    
    def forward(self,x):
        
        for c in self.Contracting:
            x = c(x)
        x = self.Base(x)
        self.build_decoder()
        for e in self.Expanding:
            x = e(x)
        x = self.FinalConv(x)
        x = self.FinalActivation(x)
        return x

# custom blocks

class UnetEncoderBlock(nn.Module):
    def __init__(self,
                 input_chs, num_filters, kernel_size, padding=1, padding_mode='zeros',
                 p_kernel_size=2, stride=2, device='cpu'
                ):
        super().__init__()
        self.ConvBlock = UnetConvBlock(input_chs,num_filters,kernel_size,padding,padding_mode,device=device)
        self.MaxPool2d = nn.MaxPool2d(p_kernel_size,stride=stride)
    
    def forward(self, x):

        x = self.ConvBlock(x)
        self.skip = x
        x = self.MaxPool2d(x)
        self.pool = x
        return x

class UnetDecoderBlock(nn.Module):
    def __init__(self, skip, input_chs, num_filters, kernel_size, merge_mode='concat', device='cpu'):
        super().__init__()
        self.UpConv2d = UpConv2d(input_chs,num_filters,device=device)
        self.MergeMode = merge_mode
        self.skip = skip
        self.ConvBlock = UnetConvBlock(input_chs,num_filters,kernel_size,device=device)

    def forward(self,x):
        x = self.UpConv2d(x)
        x = torch.cat((x,self.skip),1)
        x = self.ConvBlock(x)
        return x

class UnetConvBlock(nn.Module):
    def __init__(self, input_chs, num_filters, kernel_size, padding=1, padding_mode='zeros', device='cpu'):
        super().__init__()
        self.relu  = nn.ReLU()
        self.batch_n_2d = nn.BatchNorm2d(num_filters,device=device)
        self.Conv2d = nn.Conv2d(
            input_chs, 
            num_filters, 
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
            device=device) #bias optimization
        self.Conv2d2 = nn.Conv2d(
            num_filters,
            num_filters,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
            device=device) #bias optimization
    
    def forward(self, x):

        x = self.Conv2d(x)
        #_,C,H,W = x.shape
        x = self.relu(self.batch_n_2d(x)) # added batch normalization
        #reshape = (H,W,1,C)
        #x = x.view(reshape)
        x = self.Conv2d2(x)
        x = self.relu(self.batch_n_2d(x))
        return x
    
    # custom modules

class UpConv2d(nn.Module):
    def __init__(self, input_chs, num_filters, up_mode='resize', device='cpu'):
        super().__init__()
        self.Upsample = nn.Upsample(mode='nearest',scale_factor=2)
        self.Conv2x2 = nn.Conv2d(
            input_chs, num_filters, kernel_size=1, padding=0, dilation=1, padding_mode='zeros', bias=False, device=device
        )

    def forward(self, x):
        
        x = self.Upsample(x)
        x = self.Conv2x2(x)
        return x