import torch
from abc import abstractmethod

class Noise:

    @abstractmethod
    def __call__(self, tensor: torch.Tensor):
        pass

class NormalNoise:

    def __call__(self, tensor: torch.Tensor):
        return torch.randn_like(tensor)