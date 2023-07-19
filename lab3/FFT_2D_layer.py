import torch
import math
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .lazy import LazyModuleMixin
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes

from ..common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

# img is the image and filter is the kernel
def fft_conv(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    # take fft of image and shift it
    f = np.fft.fft2(input)
    fshift = np.fft.fftshift(f)
    # take fft of kernel 
    g = np.fft.fft2(weight, fshift.shape)
    F_gaussian = np.fft.fftshift(g)
    # multiply them in time domain
    F_filtered_img = fshift*F_gaussian
    # take inverse fft and normalize it
    filtered_img = np.fft.ifft2(np.fft.ifftshift(F_filtered_img)).real
    filtered_img = (filtered_img - np.min(filtered_img))/(np.max(filtered_img)-np.min(filtered_img))
    return filtered_img


class FFT_Conv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)





