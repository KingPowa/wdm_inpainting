import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from collections.abc import Collection
from datetime import datetime

def get_timestamp():
    return datetime.timestamp(datetime.now())

def calculate_output_shape(input_shape: Collection[int], 
                           kernel_size: int | Tuple[int, int], 
                           stride: int | Tuple[int, int]=1, 
                           padding: int | Tuple[int, int]=0, 
                           dilation: int =1, out_channels: int =None):
    """
    Calculate the output shape after a convolutional layer.
    
    Args:
        input_shape (tuple): The shape of the input image (batch_size, channels, height, width) or (channels, height, width).
        kernel_size (int or tuple): The size of the convolutional kernel.
        stride (int or tuple): The stride of the convolution. Default is 1.
        padding (int or tuple): The padding applied to the input. Default is 0.
        dilation (int): The dilation rate of the kernel. Default is 1.
        out_channels (int): The number of output channels (number of filters). Required for the number of channels in the output.
    
    Returns:
        tuple: The shape of the output image.
    """
    
    if len(input_shape) == 4:  # Batch size present
        batch_size, in_channels, height, width = input_shape
    elif len(input_shape) == 3:  # No batch size
        in_channels, height, width = input_shape
        batch_size = None
    else:
        raise ValueError("Invalid input shape. Must be of length 3 or 4.")
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Output height and width calculation
    output_height = (height + 2 * padding[0] - dilation * (kernel_size[0] - 1) - 1) // stride[0] + 1
    output_width = (width + 2 * padding[1] - dilation * (kernel_size[1] - 1) - 1) // stride[1] + 1
    
    if out_channels is None:
        out_channels = in_channels  # If not provided, keep the same number of channels
    
    # Return the output shape
    if batch_size is not None:
        return (batch_size, out_channels, output_height, output_width)
    else:
        return (out_channels, output_height, output_width)
    

def calculate_transpose_conv_output_shape(input_shape, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, out_channels=None):
    """
    Calculate the output shape after a transposed convolutional (ConvTranspose) layer.
    
    Args:
        input_shape (tuple): The shape of the input image (batch_size, channels, height, width) or (channels, height, width).
        kernel_size (int or tuple): The size of the convolutional kernel.
        stride (int or tuple): The stride of the convolution. Default is 1.
        padding (int or tuple): The padding applied to the input. Default is 0.
        output_padding (int or tuple): Additional size added to the output shape. Default is 0.
        dilation (int): The dilation rate of the kernel. Default is 1.
        out_channels (int): The number of output channels (number of filters). Required for the number of channels in the output.
    
    Returns:
        tuple: The shape of the output image.
    """
    
    if len(input_shape) == 4:  # Batch size present
        batch_size, in_channels, height, width = input_shape
    elif len(input_shape) == 3:  # No batch size
        in_channels, height, width = input_shape
        batch_size = None
    else:
        raise ValueError("Invalid input shape. Must be of length 3 or 4.")
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    
    # Output height and width calculation for ConvTranspose
    output_height = (height - 1) * stride[0] - 2 * padding[0] + dilation * (kernel_size[0] - 1) + 1 + output_padding[0]
    output_width = (width - 1) * stride[1] - 2 * padding[1] + dilation * (kernel_size[1] - 1) + 1 + output_padding[1]
    
    if out_channels is None:
        out_channels = in_channels  # If not provided, keep the same number of channels
    
    # Return the output shape
    if batch_size is not None:
        return (batch_size, out_channels, output_height, output_width)
    else:
        return (out_channels, output_height, output_width)
    
def reparametrize(mu: torch.Tensor, var: torch.Tensor, log=False):

    std = var.mul(0.5).exp_() if log else var.mul(0.5)
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)

def normalize_image(image: np.ndarray, clip_percentiles=False, pmin=1, pmax=99):
    """
    Function to normalize the images between [0,1]. If percentiles is set to True it clips the intensities at
     percentile 1 and 99
    :param image: numpy array containing the image
    :param clip_percentiles: set to True to clip intensities. (default: False)
    :param pmin: lower percentile to clip
    :param pmax: upper percentile to clip
    :return: normalized image [0,1]
    """
    if clip_percentiles is True:
        pmin = np.percentile(image, pmin)
        pmax = np.percentile(image, pmax)
        v = np.clip(image, pmin, pmax)
    else:
        v = image

    v_min = v.min(axis=(0, 1, 2), keepdims=True)
    v_max = v.max(axis=(0, 1, 2), keepdims=True)

    return (v - v_min) / (v_max - v_min)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

#  In case someone wonders... https://discuss.pytorch.org/t/is-there-any-different-between-torch-sigmoid-and-torch-nn-functional-sigmoid/995
class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)
      
class Sigmoid(nn.Module):
    def forward(self, x):
        return F.sigmoid(x)
    
class Relu(nn.Module):
    def forward(self, x):
        return F.relu(x)
    
class Tanh(nn.Module):
    def forward(self, x):
        return F.tanh(x)

def get_activation(activation: str) -> nn.Module:
    if activation is None: return nn.Identity()
    if activation == "relu": return Relu()
    if activation == "sigmoid": return Sigmoid()

def is_iterable(the_element, cls=None):
    try:
        iter(the_element)
    except TypeError:
        return False
    else:
        return True if cls is None else not isinstance(the_element, cls)