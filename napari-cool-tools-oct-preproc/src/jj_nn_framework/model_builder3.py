"""

"""
import torch
import math
import torch.nn.functional as F
from torch import nn
from jj_nn_framework.model_utils import update_routing

# custom models

class BasicSegCaps(nn.Module):
    def __init__(self, params_list):
        super().__init__()
        self.ConvBlock = SC_ConvBlock(**params_list[0])
        self.PrimaryCaps = SC_ConvCapsuleLayer(**params_list[1])
        self.SegCaps = SC_ConvCapsuleLayer(**params_list[2])
        self.Length = SC_Length(**params_list[3])
        
    def forward(self,x):
        x = self.ConvBlock(x)
        x = self.PrimaryCaps(x)
        x = self.SegCaps(x)
        x = self.Length(x)
        return x

# custom blocks

class SC_ConvBlock(nn.Module):
    def __init__(self, input_chs, num_filters, kernel_size, padding=2, padding_mode='zeros', device='cpu'):
        super().__init__()
        self.device = device
        self.relu  = nn.ReLU()
        self.batch_n_2d = nn.BatchNorm2d(num_filters,device=self.device)
        self.conv2d = nn.Conv2d(
            input_chs, 
            num_filters, 
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,bias=False,
            device=self.device) #bias optimization
    
    def forward(self, x):

        # input N,C,H,W
        x = self.conv2d(x)
        N,C,H,W = x.shape
        x = self.relu(self.batch_n_2d(x)) # added batch normalization
        #reshape = (H,W,1,C)
        #x = x.view(reshape)
        x = x.unsqueeze(2) # N,S,Ct,H,W
        return x

# custom layers

class SC_Length(nn.Module):
    def __init__(self, num_classes, seg=True, device='cpu', **kwargs):
        super(SC_Length, self).__init__(**kwargs)
        if num_classes == 2:
            self.num_classes = 1
        else:
            self.num_classes = num_classes
        self.seg = seg
        self.device = device

    def forward(self, x, **kwargs):
        # original input N,H,W,Ct,S vs my input N,S,Ct,H,W
        if x.ndim == 5:
            assert x.shape[2] == 1, 'Error: Must have num_capsules = 1 going into Length'
            x = x.squeeze(dim=2)
            x = torch.linalg.norm(x,dim=1).unsqueeze(dim=-1)
        if self.seg:
            classes = torch.ones((1,self.num_classes),device=self.device)
            x = x @ classes
            x = x.permute(0,3,1,2)
        else:
            pass
            
        return x #K.expand_dims(tf.norm(inputs, axis=-1), axis=-1)


class SC_ConvCapsuleLayer(nn.Module):
    def __init__(self,
                 input_shape, kernel_size, num_capsule, num_atoms, 
                 strides=1, padding='same', routings=3, device='cpu',
                 verbose=False, **kwargs):
        
        super(SC_ConvCapsuleLayer, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_capsule = num_capsule
        self.num_atoms = num_atoms
        self.strides = strides
        self.padding = padding
        self.routings = routings
        self.device = device
        self.verbose = verbose
        self.update_routing = update_routing
        self.build()
        
    def build(self):
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]
        self.input_num_capsule = self.input_shape[2]
        self.input_num_atoms = self.input_shape[3]
        
        #print(f"input_shape: {self.input_shape}\nh:{self.input_height},w:{self.input_width},nc:{self.input_num_capsule},"
        #        f"na:{self.input_num_atoms}\n"
        #     )
        
        w_shape = (self.num_capsule*self.num_atoms, self.input_num_atoms, self.kernel_size, self.kernel_size)
        #w_shape = (self.input_num_atoms, self.num_capsule*self.num_atoms, self.kernel_size, self.kernel_size) 
        # had to
        # change shape order to accomodate pytorch F.conv2D function in forward pass
        # weights are (out_channels*atoms,input_channel_atoms,kH,kW)
        w = torch.empty(w_shape,device=self.device)
        w = nn.init.kaiming_uniform_(w,mode='fan_in',nonlinearity='relu')
        self.W = nn.Parameter(w)
        #print(self.W.shape,self.W.min(),self.W.mean(),self.W.max())
        
        b_shape = (1,1,self.num_capsule,self.num_atoms)
        b = torch.empty(b_shape,device=self.device)
        b = nn.init.constant_(b,0.1)
        self.b = nn.Parameter(b)
        #print(self.b.shape)
        
    def forward(self,x):

        #print(f"my_input shape: {x.shape}")

        # input  my input N,S_i,Ct_i,H_i,W_i vs original N,H_i,W_i,Ct_i,S_i
        if self.verbose:
            print(f"x from conv block:\n{x.shape}\n")
            
        #x = x.view(self.input_num_capsule,-1,self.input_height,self.input_width,self.input_num_atoms)
 
        input_transposed = x.permute(2,0,3,4,1) #x.permute(3,0,1,2,4) # -> Ct_i,N,H_i,W_i,S_i -> my permute(2,0,3,4,1)
        input_shape = input_transposed.shape

        #print(f"inputT shape = {input_shape}")

        input_tensor_reshaped = input_transposed.reshape(
            # Ct_i * N, H_i, W_i, S_i
            (input_shape[0] * input_shape[1], self.input_height, self.input_width, self.input_num_atoms)
        ) # may need to use reshape --> changed input_tensor_reshaped below as well for 1x1 convolution
        input_tensor_reshaped = input_tensor_reshaped.reshape(-1, self.input_num_atoms, self.input_height, self.input_width) # had
        # to change order to represent N(minibatch),C(input_channels),H,W for pytorch F.conv2D
        
        if self.verbose:
            print(
                f"expanded to batch size:\n{x.shape}\n"
                f"input_shape:\n{input_shape}\n"
                f"input_transposed:\n{input_transposed.shape}\n"
                f"input_tensor_reshaped:\n{input_tensor_reshaped.shape}\n"
            )
        
        conv = F.conv2d(input_tensor_reshaped,self.W,stride=(self.strides,self.strides),padding=self.padding,groups=1)
        
        #print(f"conv.shape: {conv.shape}\n")
        
        votes_shape = conv.shape
        _,_,conv_height,conv_width = conv.shape # had to change vote shape for pytorch N,C,H,W
        
        votes = conv.reshape(
            # N, Ct_i, H_i, W_i, Ct_o, S_o
            (input_shape[1], input_shape[0], votes_shape[2], votes_shape[3], self.num_capsule, self.num_atoms)
        ) # had to change vote shape for pytorch N,C,H,W
        
        # N, Ct_i, H_o?, W_o?, Ct_o, S_o
        votes = votes.view(-1,self.input_num_capsule, conv_height, conv_width, self.num_capsule, self.num_atoms)
        
        if self.verbose:
            print(
                f"conv:\n{conv.shape}\n"
                f"votes_shape:\n{votes_shape}\n"
                f"conv_height:\n{conv_height}\nconv_width:\n{conv_width}\n"
                f"votes:\n{votes.shape}\n"
            )
        
        
        #print(f"Stackables:\n",input_shape[1],input_shape[0],votes_shape[1],votes_shape[2],self.num_capsule)
        #print(f"Stackables typing:\n",type(input_shape[1]),type(input_shape[0]),type(votes_shape[1]),
              #type(votes_shape[2]),type(self.num_capsule))
        
        #logit_shape = torch.stack(
        #    (input_shape[1], input_shape[0], votes_shape[1], votes_shape[2], self.num_capsule)
        #)
        logit_shape = torch.tensor(
            # N, Ct_i, H_i, W_i, Ct_o
            (input_shape[1], input_shape[0], votes_shape[2], votes_shape[3], self.num_capsule)
        )# had to change vote shape for pytorch N,C,H,W
        # H_o, W_o, 1, 1
        biases_replicated = torch.tile(self.b, [conv_height, conv_width, 1, 1])
        
        if self.verbose:
            print(
                f"logit_shape:\n{logit_shape}\n"
                f"biases_replicated:\n{biases_replicated.shape}\n"
                f"place holder:\n{'paste here'}\n"
            )
        
        activations = self.update_routing(
            votes=votes,
            biases=biases_replicated,
            logit_shape=logit_shape,
            num_dims=6,
            input_dim=self.input_num_capsule,
            output_dim=self.num_capsule,
            num_routing=self.routings,
            device=self.device
        )
        
        if self.verbose:
            print(
                f"activations:\n{activations.shape}\nactivations type:\n{type(activations)}\n"
                f"activation 0:\n{activations[0].shape}\n"
                f"place holder:\n{'paste here'}\n"
            )
        
        # activations -> N,H,W,Ct,S
        # reshape activations for my output shape
        return activations.permute(0,4,3,1,2)