# code based on code from https://www.learnpytorch.io/05_pytorch_going_modular/
"""
Contains PyTorch model code to instantiate a No Need for Routing Between Capsules model.
"""
import torch
import math
import torch.nn.functional as F
from torch import nn

# custom models

class nrnbc_model2(nn.Module):
    def __init__(self, input_chs, h, w, init_filters, kernel_size, filter_inc, num_classes=2,num_conv=[9],cap_dims=[8],verbose=False,soft_out=True):
        super().__init__()
        
        try:
            assert(len(num_conv) == len(cap_dims))
        except:
            print(f"\nnum_conv, and cap_dims arguments must have the same number of elements!!\n"
                 f"num_conv has {len(num_conv)} and cap_dims has {len(cap_dims)}"
                 )
        else:
            self.in_filters = [1 if i == 0 else init_filters + (num_conv[i-1]-1)*filter_inc for i in range(len(num_conv))]
            self.out_filters = [init_filters if i == 0 else init_filters + (num_conv[i-1])*filter_inc for i in range(len(num_conv))]
            self.num_filters = [init_filters + (i-1)*filter_inc for i in num_conv]
            self.dims = [(h - 2*i, w - 2*i) for i in num_conv]
            self.cap_counts = [dim[0]*dim[1]*self.num_filters[j]//cap_dims[j] for j,dim in enumerate(self.dims)]          
            self.branches = nn.ModuleList(
                [
                    branch_block(
                        self.in_filters[i],self.out_filters[i],kernel_size,filter_inc,
                        self.cap_counts[i],int(num_conv[i] - i*3),cap_dims[i],num_classes, # need to error check num_conv and divisor
                        verbose = False
                    )\
                    for i in range(len(num_conv))
                    
                ]
            )
            self.verbose = verbose
            self.soft_out=True
            
            def merge_branches(branch_results):
        
                x = torch.stack(branch_results,1)
                x = torch.sum(x,1)
                if verbose:
                    print(f"\n pre softmax output:\n{x}\n")
                if self.soft_out:
                    x = F.softmax(x,1)
                return(x)
            
            self.merge_branches = merge_branches
                   
    def forward(self,x):
        branch_results = []

        if self.verbose:
            print(f"num branches in model: {len(self.branches)}")
        #branch_results = [branch(x) if i == 0 else branch(branch_results[i-1]) for i,branch in enumerate(self.branches)]
        
        
        for i,branch in enumerate(self.branches):
            if self.verbose:
                print(f"\nloop {i}\n")
            if i == 0:
                x = branch(x)
                branch_results.append(x)
                
            else:
                temp_result = self.branches[i-1].conv_result
                if self.verbose:
                    print(f"\ntemp result {i}: {temp_result.shape}\n")
                x = branch(temp_result)
                branch_results.append(x)
            if self.verbose:
                print(f"\nbranch_results for branch {i} are...\n{x}\n")
        
        '''
        x = self.branches[0](x)
        branch_results.append(x)
        temp_result = self.branches[0].conv_result
        print(f"\ntemp result 0: {temp_result.shape}\n")
        x = self.branches[1](temp_result)
        branch_results.append(x)
        temp_result = self.branches[1].conv_result
        print(f"\ntemp result 1: {temp_result.shape}\n")
        x = self.branches[2](temp_result)
        branch_results.append(x)
        '''

        if self.verbose:
            print(f"\nThere are {len(branch_results)} results:\n{branch_results}\n")

        merge_results = self.merge_branches(branch_results)
        return merge_results

class nrnbc_model(nn.Module):
    def __init__(self, input_chs, h, w, init_filters, kernel_size, filter_inc, num_classes=2,num_conv=[9],cap_dims=[8]):
        super().__init__()
        
        try:
            assert(len(num_conv) == len(cap_dims))
        except:
            print(f"\nnum_conv, and cap_dims arguments must have the same number of elements!!\n"
                 f"num_conv has {len(num_conv)} and cap_dims has {len(cap_dims)}"
                 )
        else:
            self.num_filters = [init_filters + (i-1)*filter_inc for i in num_conv]
            self.dims = [(h - 2*i, w - 2*i) for i in num_conv]
            self.cap_counts = [dim[0]*dim[1]*self.num_filters[j]//cap_dims[j] for j,dim in enumerate(self.dims)]          
            self.branches = nn.ModuleList(
                [
                    branch_block(
                        input_chs,init_filters,kernel_size,filter_inc,
                        self.cap_counts[i],num_conv[i],cap_dims[i],num_classes
                    )\
                    for i in range(len(num_conv))
                    
                ]
            )
            
            def merge_branches(branch_results):
        
                x = torch.stack(branch_results,1)
                x = torch.sum(x,1)
                x = F.softmax(x,1)
                return(x)
            
            self.merge_branches = merge_branches
                   
    def forward(self,x):
        branch_results = [
            branch(x) for branch in self.branches
        ]
        merge_results = self.merge_branches(branch_results)
        return merge_results

# custom blocks

class branch_block(nn.Module):
    def __init__(self, input_chs, num_filters, kernel_size, filter_inc, cap_count, num_conv=9, cap_dims=8, num_classes=10, verbose=False):
        super().__init__()
        self.conv_block = conv_block(input_chs,num_filters,kernel_size,filter_inc,num_conv)
        self.zxy_caps = caps_from_conv_zxy(cap_count,cap_dims)
        self.hvc_caps = hvc_from_zxy(num_classes, cap_count, cap_dims)
        self.conv_result = None
        self.verbose = verbose
        
    def forward(self,x):
        x = self.conv_block(x)
        self.conv_result = x
        if self.verbose:
            print(f"\nOutput for branches convolution block:\n{self.conv_result.shape}")
        x = self.zxy_caps(x)
        x = self.hvc_caps(x)
        x = torch.sum(x,2)
        return x

class conv_block(nn.Module):
    def __init__(self, input_chs, num_filters, kernel_size, filter_inc, num_conv):
        super().__init__()
        self.relu  = nn.ReLU()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(input_chs,num_filters,kernel_size,bias=False) # bias optimization
            ]
        ).extend(
            [
                nn.Conv2d(num_filters + i*filter_inc,num_filters + (i+1)*filter_inc,kernel_size,bias=False) for i in range(num_conv-1) #bias optimization
            ]
        )
        self.batch_layers = nn.ModuleList(
            [
                nn.BatchNorm2d(num_filters + i*filter_inc) for i in range(num_conv)
            ]
        )
    
    def forward(self, x):

        for i,block in enumerate(self.conv_layers):
            if i < len(self.conv_layers) - 1:
                x = self.relu(self.batch_layers[i](block(x)))
            else:
                x = block(x)        
        return x

# custom layers

class caps_from_conv_zxy(nn.Module):
    def __init__(self, cap_count, cap_dims):
        super().__init__()
        self.cap_count = cap_count
        self.cap_dims = cap_dims
        
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = x.reshape(-1,1,self.cap_count,self.cap_dims)   # condsider using Tensor.view instead     
        return x
    
class hvc_from_zxy(nn.Module):
    def __init__(self, num_classes, cap_count, cap_dims):
        super().__init__()
        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(num_classes)
        
        # paper uses random normal initialization --> nn.Parameter(torch.randn(num_classes,cap_count,cap_dims))
        self.zxy_weights = nn.Parameter(torch.randn(num_classes,cap_count,cap_dims) / math.sqrt(num_classes))
        # try builtin pytorch initialization here!!
        # for example --> nn.Parameter(nn.init.kaiming_uniform_(torch.empty(num_classes,cap_count,cap_dims),nonlinearity='relu'))

    def forward(self,x):
        x = torch.sum(torch.mul(x,self.zxy_weights),2)
        x = self.relu(self.batch(x))
        return x
    
