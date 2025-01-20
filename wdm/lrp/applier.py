from .parametrize import BNParametrizer
from captum.attr._utils.lrp_rules import PropagationRule, EpsilonRule
from captum.attr._core.lrp import SUPPORTED_LAYERS_WITH_RULES, LRP
import torch.nn as nn
import torch
from copy import deepcopy
from typing import List, TypeVar, Tuple
from collections import OrderedDict

from .utils import contains_bn, expand_model, is_activation, is_batchnorm

TorchModel = TypeVar('T', bound=nn.Module)

class LRPApplier:

    # https://iphome.hhi.de/samek/pdf/LetSPM22.pdf

    def __init__(self, model: TorchModel, rules: OrderedDict[str, PropagationRule] | List[PropagationRule]):
        self.model = self.__prepare_model(model)
        self.rules = self.__prepare_rules(rules)
        self.apply_rules()

    def __complete_rules(self, rules: OrderedDict[str, PropagationRule]) -> OrderedDict[str, PropagationRule]:
        for key in self.layers.keys():
            if key not in rules:
                # rules[key] = SUPPORTED_LAYERS_WITH_RULES[type(self.layers[key])]() if type(self.layers[key]) in SUPPORTED_LAYERS_WITH_RULES else EpsilonRule()
                continue
        return rules

    def __validate_rules(self, rules: OrderedDict[str, PropagationRule] | List[PropagationRule]) -> OrderedDict[str, PropagationRule]:
        keys = list(self.layers.keys())

        if rules is None: return OrderedDict()
        if isinstance(rules, (OrderedDict, dict)):
            fltr = []
            for key in rules.keys():
                if key not in keys:
                    print(f"Warning - {key} in rules is not a model layer. Filtering.")
                    fltr.append(key)
            for k in fltr: rules.pop(k, None)
            return rules
        elif isinstance(rules, List):
            rl = OrderedDict()
            for i, key in enumerate(keys):
                if i > len(rules): break
                rl[key] = rules[i]
            return rl
        raise ValueError("Invalid value for rules")
    
    def __prepare_model(self, model) -> TorchModel:

        if contains_bn(model):
            new_model = BNParametrizer(False).reparametrize(model) # https://arxiv.org/pdf/2002.11018
        else: 
            new_model = deepcopy(model)

        return new_model
    
    def __prepare_rules(self, rules: OrderedDict[str, PropagationRule] | List[PropagationRule]) -> OrderedDict[str, PropagationRule]:
        self.layers = {elem[0]: elem[1] for elem in expand_model(self.model.named_children()) if not (is_activation(elem[1]) or isinstance(elem[1], nn.Dropout))}
        rules = self.__validate_rules(rules)
        rules = self.__complete_rules(rules)
        return rules
    
    def set_rules(self, rules: OrderedDict[str, PropagationRule] | List[PropagationRule]):

        self.rules = self.__prepare_rules(rules)
    
    def apply_rules(self):

        rules_to_apply = deepcopy(self.rules)
        for key, layer in self.layers.items():
            layer.rule = rules_to_apply[key]
    
    def attribute(self, input: torch.Tensor):

        self.apply_rules()
        lrp_model = LRP(self.model)
        return lrp_model.attribute(input)

        
        