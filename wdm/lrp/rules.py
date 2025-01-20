import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from captum.attr._utils.lrp_rules import PropagationRule
from model.utils.nn_utils import MaxPool3dIndices

class WinnerTakesAll(PropagationRule):
    """
    Sets all weights and inputs to ones, similar to the "flat" mode in the TF implementation.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def _manipulate_weights(self, module: MaxPool3dIndices, inputs, outputs):
        pass
        
    def _create_backward_hook_input(self, inputs):
        def _backward_hook_input(grad):
            print("Inputs", self.name, grad[grad != 0])
            relevance = grad * inputs
            device = grad.device
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                self.relevance_input[device].append(relevance.data)

            # replace_out is needed since two hooks are set on the same tensor
            # The output of this hook is needed in backward_hook_activation
            grad.replace_out = relevance
            return relevance

        return _backward_hook_input

    def _create_backward_hook_output(self, outputs):
        def _backward_hook_output(grad):
            print("Outputs", self.name, grad[grad != 0])
            self.relevance_output[grad.device] = grad.data
            return grad

        return _backward_hook_output

class AllOnesRule(PropagationRule):
    """
    Sets all weights and inputs to ones, similar to the "flat" mode in the TF implementation.
    """
    def __init__(self):
        super().__init__()
        self.original_input = None

    def _manipulate_weights(self, module: nn.Module, inputs, outputs):
        # Set weights to 1 if the module has them
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = torch.ones_like(module.weight.data)

        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data = torch.zeros_like(module.bias.data)

    def forward_pre_hook_activations(self, module, inputs):
        """Pass initial activations to graph generation pass"""
        device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        for input, activation in zip(inputs, module.activations[device]):
            input.data = torch.ones_like(input, device=input.device, requires_grad=True)
        return inputs


    
        # # Manip input
        # if isinstance(inputs, tuple):
        #     self.original_input = (input.clone() for input in inputs)
        #     # Modify input data
        #     for input in inputs:
        #         input.data = torch.ones_like(input, device=input.device, requires_grad=True)
        # else:
        #     self.original_input = inputs.clone()
        #     inputs.data = torch.ones_like(inputs, device=inputs.device, requires_grad=True)
        