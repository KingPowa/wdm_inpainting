import torch.nn as nn
from typing import Iterator, Tuple, List, Dict, Union, TypeVar, Any
from collections import OrderedDict
import inspect
from copy import deepcopy

SKIPPABLE = (nn.Sequential,)

Conv1Class = TypeVar('C1', bound=nn.Conv1d)
Conv2Class = TypeVar('C2', bound=nn.Conv2d)
Conv3Class = TypeVar('C3', bound=nn.Conv3d)

def is_skippable(module: nn.Module) -> bool:
    return isinstance(module, SKIPPABLE)

def is_batchnorm(module: nn.Module) -> bool:
    return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

def is_dense(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear)

def is_fc(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))

def is_dense_or_fc(module: nn.Module) -> bool:
    return is_dense(module) or is_fc(module)

def is_activation(module: nn.Module) -> bool:
    return isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6, nn.LeakyReLU, nn.CELU, nn.LogSigmoid, nn.SELU, nn.Softmax))

def get_layer_keys_before_lrp(model: nn.Module) -> List[str]:
    layers = expand_model(model.named_children())
    eff = []
    for name, module in layers:
        if not is_batchnorm(module) and not is_activation(module) and not isinstance(module, nn.Dropout):
            eff.append(name)
    return eff

def expand_model(named_modules: Iterator[Tuple[str, nn.Module]], opt_prefix = None) -> List[Tuple[str, nn.Module]]:
    modules = []
    for key, module in named_modules:
        if isinstance(module, nn.Sequential):
            modules.extend(expand_model(module.named_children(), key))
        else:
            modules.append((((opt_prefix + ".") if opt_prefix else "") + key, module))
    return modules

def reconstruct_model(modules: List[Tuple[str, nn.Module]]) -> dict[str, nn.Module]:
    def add_to_tree(tree: Dict, path: List[str], module: nn.Module):
        """ Recursively add module to the nested dictionary tree based on path. """
        if len(path) == 1:  # Leaf node
            tree[path[0]] = module
        else:
            if path[0] not in tree:  # Create subtree if not exists
                tree[path[0]] = {}
            add_to_tree(tree[path[0]], path[1:], module)

    def build_sequential(tree: Union[Dict, nn.Module]) -> nn.Module:
        """ Recursively build Sequential or return leaf nodes. """
        if isinstance(tree, nn.Module):
            return tree  # Leaf module, return as is
        sequential = nn.Sequential()
        for name, sub_tree in tree.items():
            sequential.add_module(name, build_sequential(sub_tree))
        return sequential

    # Step 1: Build the hierarchical tree
    tree = {}
    for name, module in modules:
        path = name.split('.')
        add_to_tree(tree, path, module)
    
    # Step 2: Reconstruct functional units as Sequential
    reconstructed = OrderedDict()
    for key, subtree in tree.items():
        reconstructed[key] = build_sequential(subtree)
    
    return reconstructed

def get_layer_copy(module: nn.Module) -> nn.Module:
    new_module = deepcopy(module)
    new_module._forward_hooks.clear()
    new_module._backward_hooks.clear()
    new_module._forward_pre_hooks.clear()
    new_module._backward_pre_hooks.clear()
    new_module.zero_grad()
    return new_module

def get_conv_args(conv: Conv1Class | Conv2Class | Conv3Class) -> Dict[str, Any]:
    conv_class = conv.__class__  
    init_signature = inspect.signature(conv_class.__init__)
    init_params = init_signature.parameters

    conv_args = {}

    # Extract existing arguments from conv
    for param in init_params:
        if param != "self":  # Skip 'self'
            # Check if conv has an attribute corresponding to this argument
            if hasattr(conv, param):
                conv_args[param] = getattr(conv, param)

    return conv_args

def contains_bn(model: nn.Module) -> bool:
    modules = expand_model(model.named_children())
    for _, module in modules:
        if is_batchnorm(module): return True
    return False

