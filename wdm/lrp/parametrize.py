import torch.nn as nn
import torch
import copy

from typing import TypeVar, Any

TorchModel = TypeVar('T', bound=nn.Module)
Conv1Class = TypeVar('C1', bound=nn.Conv1d)
Conv2Class = TypeVar('C2', bound=nn.Conv2d)
Conv3Class = TypeVar('C3', bound=nn.Conv3d)

from .utils import is_batchnorm, expand_model, is_dense, is_dense_or_fc, reconstruct_model, get_conv_args

class BNParametrizer:

    # https://arxiv.org/pdf/2002.11018

    def __init__(self, keep_original: bool = True):
        self.keep_original = keep_original
    
    @staticmethod
    def fuse_bn_with_linear(fc: nn.Linear, bn: nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d, is_before: bool = True):
        """
        Fuse a BatchNorm layer into a fully connected (nn.Linear) layer.
        """
        # Extract BN parameters
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        fc_bias = fc.bias if fc.bias else torch.zeros_like(mean, device=mean.device)

        # Rescale weights and bias of Linear layer
        scale = gamma / torch.sqrt(var + eps)
        weight = scale * fc.weight

        if is_before:
            bias = fc_bias + fc.weight * (beta - scale * mean) 
        else:
            bias = beta + scale * (fc_bias - mean) 

        # Apply new weights and bias
        new_fc = nn.Linear(fc.in_features, fc.out_features)
        new_fc.weight.data = weight
        new_fc.bias.data = bias
        return new_fc

    @staticmethod
    def fuse_bn_with_conv(conv: Conv1Class | Conv2Class | Conv3Class, bn: nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d, is_before: bool = True):
        """
        Fuse a BatchNorm layer into a convolutional layer.
        """
        # Extract BN parameters
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(mean, device=mean.device)

        # Rescale weights and bias of Conv layer
        shps = [1 for _ in conv.weight.shape[1:]]

        scale = gamma / torch.sqrt(var + eps)
        weight = conv.weight * scale.view(-1, *shps)

        if is_before:
            bias_corr = (conv.weight * (beta - scale * mean).view(-1, *shps))
            bias = conv_bias + bias_corr.sum(dim=[i+1 for i, _ in enumerate(conv.weight.shape[1:])])
        else:
            bias = beta + scale * (conv_bias - mean) 

        old_conv_args = get_conv_args(conv)
        old_conv_args.update({"bias": True})

        # Apply new weights and bias
        new_conv = conv.__class__(**old_conv_args)
        new_conv.weight.data = weight
        new_conv.bias.data = bias
        return new_conv

    def reparametrize(self, model: TorchModel) -> TorchModel:
        """
        Traverse the model layers and absorb BatchNorm layers.
        """
        new_modules = []
        layers = expand_model(model.named_children()) # This expand_model keeps the information about each functional block!
        skip_next = False

        for idx, (name, layer) in enumerate(layers):
            if skip_next:  # Skip the next layer because it has been fused
                skip_next = False
                continue

            elif is_batchnorm(layer):
                if idx + 1 < len(layers) and is_dense_or_fc(layers[idx + 1][1]):
                    # Fuse with batchnorm behind
                    linear_layer = layers[idx + 1][1]
                    if is_dense(layer):
                        fused_layer = self.fuse_bn_with_linear(linear_layer, layer, is_before = True)
                    else:
                        fused_layer = self.fuse_bn_with_conv(linear_layer, layer, is_before = True)
                    new_modules.append((name, fused_layer))
                    skip_next = True  # Skip Linear layer, already fused
                else:
                    new_modules.append((name, layer))

            elif is_dense_or_fc(layer):
                if idx + 1 < len(layers) and is_batchnorm(layers[idx + 1][1]):
                    # Fuse with batchnorm after
                    bn_layer = layers[idx + 1][1]
                    if isinstance(layer, nn.Linear):
                        fused_layer = self.fuse_bn_with_linear(layer, bn_layer, is_before = False)
                    else:
                        fused_layer = self.fuse_bn_with_conv(layer, bn_layer, is_before = False)
                    new_modules.append((name, fused_layer))
                    skip_next = True  # Skip BN layer
                else:
                    new_modules.append((name, layer))
            else:
                new_modules.append((name, layer))

        reconstructed_modules = reconstruct_model(new_modules)
        
        new_model: TorchModel = model if self.keep_original else copy.deepcopy(model)

        for key, _ in new_model.named_children():
            if hasattr(new_model, key):
                setattr(new_model, key, copy.deepcopy(reconstructed_modules[key]))
        return new_model
