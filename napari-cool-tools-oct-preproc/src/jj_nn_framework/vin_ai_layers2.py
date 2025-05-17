# modified from library file from VinAIResearch @ https://github.com/VinAIResearch/3D-UCaps/blob/main/layers.py

import torch
import torch.nn.functional as F
from torch import nn

class ConvSlimCapsule2D(nn.Module):
    """
    Builds a slim convolutional capsule layer.
    This layer performs 2D convolution given 5D input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]`. Then refines
    the votes with routing and applies Squash non linearity for each capsule.
    Each capsule in this layer is a convolutional unit and shares its kernel over
    the position grid and different capsules of layer below. Therefore, number
    of trainable variables in this layer is:
        kernel: [kernel_size, kernel_size, input_atoms, output_dim * output_atoms]
        bias: [output_dim, output_atoms]
    Output of a conv2d layer is a single capsule with channel number of atoms.
    Therefore conv_slim_capsule is suitable to be added on top of a conv2d layer
    with num_routing=1, input_dim=1 and input_atoms=conv_channels.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        num_routing: scalar, number of routing iterations.
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        Tensor of activations for this layer of shape
        `[batch, output_dim, output_atoms, out_height, out_width]`
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        num_routing=3,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_atoms = output_atoms
        self.num_routing = num_routing
        self.biases = nn.Parameter(torch.nn.init.constant_(torch.empty(output_dim, output_atoms, 1, 1), 0.1))
        self.depthwise_conv3d = DepthwiseConv3d(
            kernel_size=kernel_size,
            input_dim=input_dim,
            output_dim=output_dim,
            input_atoms=input_atoms,
            output_atoms=output_atoms,
            stride=stride,
            dilation=dilation,
            padding=padding,
            share_weight=share_weight,
        )

    def forward(self, input_tensor):
        votes = self.depthwise_conv3d(input_tensor)
        return _update_routing(votes, self.biases, self.num_routing)


#From library file from VinAIResearch @ https://github.com/VinAIResearch/3D-UCaps/blob/main/layers.py
class DepthwiseConv3d(nn.Module):
    """
    Performs 2D convolution given a 5D input tensor.
    This layer given an input tensor of shape
    `[batch, input_dim, input_atoms, input_height, input_width]` squeezes the
    first two dimmensions to get a 4D tensor as the input of torch.nn.Conv2d. Then
    splits the first dimmension and the second dimmension and returns the 6D
    convolution output.
    Args:
        kernel_size: scalar or tuple, convolutional kernels are [kernel_size, kernel_size].
        input_dim: scalar, number of capsules in the layer below.
        output_dim: scalar, number of capsules in this layer.
        input_atoms: scalar, number of units in each capsule of input layer.
        output_atoms: scalar, number of units in each capsule of output layer.
        stride: scalar or tuple, stride of the convolutional kernel.
        padding: scalar or tuple, zero-padding added to both sides of the input
        dilation: scalar or tuple, spacing between kernel elements
        share_weight: share transformation weight matrices between capsules in lower layer or not
    Returns:
        6D Tensor output of a 2D convolution with shape
        `[batch, input_dim, output_dim, output_atoms, out_height, out_width]`.
    """

    def __init__(
        self,
        kernel_size,
        input_dim,
        output_dim,
        input_atoms=8,
        output_atoms=8,
        stride=2,
        dilation=1,
        padding=0,
        share_weight=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_atoms = input_atoms
        self.output_atoms = output_atoms
        self.share_weight = share_weight

        if self.share_weight:
            self.conv2d = nn.Conv2d(
                input_atoms, output_dim * output_atoms, kernel_size, stride=stride, dilation=dilation, padding=padding
            )
        else:
            self.conv2d = nn.Conv2d(
                input_dim * input_atoms,
                input_dim * output_dim * output_atoms,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=input_dim,
            )
        torch.nn.init.normal_(self.conv2d.weight, std=0.1)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()

        if self.share_weight:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0] * self.input_dim, self.input_atoms, input_shape[-2], input_shape[-1]
            )
        else:
            input_tensor_reshaped = input_tensor.view(
                input_shape[0], self.input_dim * self.input_atoms, input_shape[-2], input_shape[-1]
            )

        conv = self.conv2d(input_tensor_reshaped)
        conv_shape = conv.size()

        conv_reshaped = conv.view(
            input_shape[0], self.input_dim, self.output_dim, self.output_atoms, conv_shape[-2], conv_shape[-1]
        )
        return conv_reshaped


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