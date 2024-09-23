import torch
from numpy.fft import rfft
from torch.fx.experimental.meta_tracer import nn_layernorm_override
from torch.nn import Module
from torch.fft import rfft, irfft
from torch.nn import Conv1d
import torch.nn.functional as F
import torch.nn as nn
from torch import einsum
from torch.nn.utils.parametrizations import spectral_norm


class SpectralConv1d(Module):
    def __init__(self,
                 n_in,
                 n_out,
                 modes,
                 ):
        super().__init__()
        self.modes = modes
        self.in_channels = n_in
        self.out_channels = n_out
        self.real_weights = nn.Parameter(torch.randn(n_in, n_out, modes))
        self.imag_weights = nn.Parameter(torch.randn(n_in, n_out, modes))

    def comp_mult(self, x, w):
        return einsum("iM,ioM->oM", x, w)

    def forward(self, x):

        channels, spatial_points = x.shape
        x_hat = rfft(x)

        x_hat_under_modes = x_hat[:, :self.modes]

        weights = self.real_weights + 1j*self.imag_weights
        out_hat_under_modes = self.comp_mult(x_hat_under_modes, weights)
        out_hat = torch.zeros(size=(self.out_channels, spatial_points), dtype=x_hat.dtype)

        target_shape = (self.out_channels, spatial_points)
        pad = (0, target_shape[1] - out_hat_under_modes.shape[1], 0, target_shape[0] - out_hat_under_modes.shape[0])
        out_hat = F.pad(out_hat_under_modes, pad, "constant", 0)

        out = irfft(out_hat, n=spatial_points)

        return out

class FNOBlock1d(Module):
    def __init__(self,
                 n_in,
                 n_out,
                 modes,
                 activation
                 ):
        super().__init__()
        self.spectral_conv = SpectralConv1d(n_in, n_out, modes)
        self.conv_layer = Conv1d(n_in, n_out, 1)
        self.activation = activation
    def forward(self, x):
        return self.activation(
            self.spectral_conv(x) + self.conv_layer(x))

class FNO1d(Module):
    def __init__(self,
                 n_in,
                 n_out,
                 modes,
                 width,
                 activation,
                 n_blocks=4
                 ):
        super().__init__()
        self.lifting = Conv1d(n_in, width, 1)

        self.FNOBlocks = []
        for i in range(n_blocks):
            self.FNOBlocks.append(
                FNOBlock1d(width, width, modes, activation
                           ))

        self.projection = Conv1d(width, n_out, 1)

    def forward(self, x):

        x = self.lifting(x)

        for FNOBlock in self.FNOBlocks:
            x = FNOBlock(x)

        x = self.projection(x)

        return x