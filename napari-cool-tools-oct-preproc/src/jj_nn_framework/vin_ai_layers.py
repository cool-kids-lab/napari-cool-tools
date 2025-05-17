# modified from library file from VinAIResearch @ https://github.com/VinAIResearch/3D-UCaps/blob/main/layers.py

import torch
import torch.nn.functional as F
from torch import nn

class DepthwiseCon3d(nn.Module):
    def __init__(
            self, kernel_size , cap_in, cap_out, atom_in, atom_out,
            stride=2, dilation=1, padding=0, num_routing=3, share_weight=True,device='cpu'
        ):
        super().__init__()
        self.Cti = cap_in
        self.Cto = cap_out
        self.Ai = atom_in
        self.Ao = atom_out
        self.share_weight = share_weight
        self.kH = kernel_size
        self.kW = kernel_size
        self.nR = num_routing
        self.device = device

        self.weights = nn.ConvTranspose2d(self.Ai, self.Cto * self.Ao, (self.kH,self.kW), stride, padding)
        torch.nn.init.normal_(self.weights.weight,std=0.1)

        self.biases = nn.Parameter(torch.nn.init.constant_(torch.empty(self.Cto,self.Ao,1,1),0.1))


    def forward(self, x):
        x_shape = x.shape
        #print(x.shape)

        #print(x_shape[0],self.Cti,self.Ai,x_shape[-2],x_shape[-1])
        x_reshaped = x.view(
                x_shape[0] * self.Cti, self.Ai, x_shape[-2], x_shape[-1]
            )

        conv = self.weights(x_reshaped)
        conv_shape = conv.shape

        #conv_reshaped 
        votes = conv.view(
            x_shape[0], self.Cti,self.Cto,self.Ao,conv_shape[-2],conv_shape[-1]
        )

        x = _update_routing(votes,self.biases,self.nR)

        return x


#From library file from VinAIResearch @ https://github.com/VinAIResearch/3D-UCaps/blob/main/layers.py
def _update_routing(votes, biases, num_routing):
    """
    Sums over scaled votes and applies squash to compute the activations.
    Iteratively updates routing logits (scales) based on the similarity between
    the activation of this layer and the votes of the layer below.
    Args:
        votes: tensor, The transformed outputs of the layer below.
        biases: tensor, Bias variable.
        num_dims: scalar, number of dimmensions in votes. For fully connected
        capsule it is 4, for convolutional 2D it is 6, for convolutional 3D it is 7.
        num_routing: scalar, Number of routing iterations.
    Returns:
        The activation tensor of the output layer after num_routing iterations.
    """
    votes_shape = votes.size()

    logits_shape = list(votes_shape)
    logits_shape[3] = 1
    logits = torch.zeros(logits_shape, requires_grad=False, device=votes.device)

    for i in range(num_routing):
        route = F.softmax(logits, dim=2)
        preactivate = torch.sum(votes * route, dim=1) + biases[None, ...]

        if i + 1 < num_routing:
            distances = F.cosine_similarity(preactivate[:, None, ...], votes, dim=3)
            logits = logits + distances[:, :, :, None, ...]
        else:
            activation = _squash(preactivate)
    return activation


#From library file from VinAIResearch @ https://github.com/VinAIResearch/3D-UCaps/blob/main/layers.py
def _squash(input_tensor, dim=2):
    """
    Applies norm nonlinearity (squash) to a capsule layer.
    Args:
    input_tensor: Input tensor. Shape is [batch, num_channels, num_atoms] for a
        fully connected capsule layer or
        [batch, num_channels, num_atoms, height, width] or
        [batch, num_channels, num_atoms, height, width, depth] for a convolutional
        capsule layer.
    Returns:
    A tensor with same shape as input for output of this layer.
    """
    epsilon = 1e-12
    norm = torch.linalg.norm(input_tensor, dim=dim, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / (norm + epsilon)) * (norm_squared / (1 + norm_squared))