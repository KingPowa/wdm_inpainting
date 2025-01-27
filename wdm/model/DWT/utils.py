import torch
from .DWT_IDWT_layer import IDWT_3D, DWT_3D

def convert_to_idwt(x: torch.Tensor, idwt: IDWT_3D, lfc_mul: float = 3.) -> torch.Tensor:

    B, _, H, W, D = x.size()
    x_idwt = idwt(x[:, 0, :, :, :].view(B, 1, H, W, D) * lfc_mul,
                    x[:, 1, :, :, :].view(B, 1, H, W, D),
                    x[:, 2, :, :, :].view(B, 1, H, W, D),
                    x[:, 3, :, :, :].view(B, 1, H, W, D),
                    x[:, 4, :, :, :].view(B, 1, H, W, D),
                    x[:, 5, :, :, :].view(B, 1, H, W, D),
                    x[:, 6, :, :, :].view(B, 1, H, W, D),
                    x[:, 7, :, :, :].view(B, 1, H, W, D))
    return x_idwt

def convert_to_dwt(x: torch.Tensor, dwt: DWT_3D, lfc_mul: float = 3.) -> torch.Tensor:

    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(x)
    x_dwt = torch.cat([LLL / lfc_mul, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
    return x_dwt