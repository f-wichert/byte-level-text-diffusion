import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dimensions: int, eps: float, device: torch.device, dtype: torch.dtype = torch.bfloat16, norm_in_fp32: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dimensions, dtype=dtype).to(device))
        self.norm_in_fp32 = norm_in_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        if self.norm_in_fp32:
            x = x.float()

        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if out.dtype != original_dtype:
            out = out.to(original_dtype)

        return out * self.weight
