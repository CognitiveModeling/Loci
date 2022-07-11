import torch.nn as nn
import torch as th
import numpy as np
import nn as nn_modules
from einops import rearrange, repeat, reduce
from utils.utils import LambdaModule

from typing import Union, Tuple

__author__ = "Manuel Traub"

class DynamicLayerNorm(nn.Module):

    def __init__(self, eps: float = 1e-5):
        super(DynamicLayerNorm, self).__init__()
        self.eps = eps

    def forward(self, input: th.Tensor) -> th.Tensor:
        return nn.functional.layer_norm(input, input.shape[2:], None, None, self.eps)


class SkipConnection(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale_factor: float = 1.0
        ):
        super(SkipConnection, self).__init__()
        assert scale_factor == 1 or int(scale_factor) > 1 or int(1 / scale_factor) > 1, f'invalid scale factor in SpikeFunction: {scale_factor}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

    def channel_skip(self, input: th.Tensor):
        in_channels  = self.in_channels
        out_channels = self.out_channels
        
        if in_channels == out_channels:
            return input

        if in_channels % out_channels == 0 or out_channels % in_channels == 0:

            if in_channels > out_channels:
                return reduce(input, 'b (c n) h w -> b c h w', 'mean', n = in_channels // out_channels)

            if out_channels > in_channels:
                return repeat(input, 'b c h w -> b (c n) h w', n = out_channels // in_channels)

        mean_channels = np.gcd(in_channels, out_channels)
        input = reduce(input, 'b (c n) h w -> b c h w', 'mean', n = in_channels // mean_channels)
        return repeat(input, 'b c h w -> b (c n) h w', n = out_channels // mean_channels)

    def scale_skip(self, input: th.Tensor):
        scale_factor = self.scale_factor

        if scale_factor == 1:
            return input

        if scale_factor > 1:
            return repeat(
                input, 
                'b c h w -> b c (h h2) (w w2)', 
                h2 = int(scale_factor),
                w2 = int(scale_factor)
            )

        height = input.shape[2]
        width  = input.shape[3]

        # scale factor < 1
        scale_factor = int(1 / scale_factor)

        if width % scale_factor == 0 and height % scale_factor == 0:
            return reduce(
                input, 
                'b c (h h2) (w w2) -> b c h w', 
                'mean', 
                h2 = scale_factor,
                w2 = scale_factor
            )

        if width >= scale_factor and height >= scale_factor:
            return nn.functional.avg_pool2d(
                input, 
                kernel_size = scale_factor,
                stride = scale_factor
            )

        assert width > 1 or height > 1
        return reduce(input, 'b c h w -> b c 1 1', 'mean')


    def forward(self, input: th.Tensor):

        if self.scale_factor > 1:
            return self.scale_skip(self.channel_skip(input))

        return self.channel_skip(self.scale_skip(input))

class DownScale(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale_factor: int,
            groups: int = 1,
            bias: bool = True
        ):                                                                   

        super(DownScale, self).__init__()

        assert(in_channels % groups == 0)
        assert(out_channels % groups == 0)

        self.groups = groups
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(th.empty((out_channels, in_channels // groups, scale_factor, scale_factor)))
        self.bias = nn.Parameter(th.empty((out_channels,))) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: th.Tensor):
        height = input.shape[2]
        width  = input.shape[3]
        assert height > 1 or width > 1, "trying to dowscale 1x1"
        
        scale_factor = self.scale_factor
        padding = [0, 0]

        if height < scale_factor:
            padding[0] = scale_factor - height

        if width < scale_factor:
            padding[1] = scale_factor - width
        
        return nn.functional.conv2d(
            input, 
            self.weight, 
            bias=self.bias,
            stride=scale_factor,
            padding=padding,
            groups=self.groups
        )


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = (3, 3),
            scale_factor: int = 1,
            groups: Union[int, Tuple[int, int]] = (1, 1),
            bias: bool = True,
            layer_norm: bool = False,
            leaky_relu: bool = False,
            residual: bool = True,
            alpha_residual: bool = False,
            input_nonlinearity = True
        ):

        super(ResidualBlock, self).__init__()
        self.residual       = residual
        self.alpha_residual = alpha_residual
        self.skip           = False
        self.in_channels    = in_channels
        self.out_channels   = out_channels

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        if isinstance(groups, int):
            groups = [groups, groups]

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        _layers = list()
        if layer_norm:
            _layers.append(DynamicLayerNorm())

        if input_nonlinearity:
            if leaky_relu:
                _layers.append(nn.LeakyReLU())
            else:
                _layers.append(nn.ReLU())

        if scale_factor > 1:
            _layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                    groups=groups[0],
                    bias=bias
                )
            )
        elif scale_factor < 1:
            _layers.append(
                DownScale(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    scale_factor=int(1.0/scale_factor),
                    groups=groups[0],
                    bias=bias
                )
            )
        else:
            _layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=groups[0],
                    bias=bias
                )
            )

        if layer_norm:
            _layers.append(DynamicLayerNorm())
        if leaky_relu:
            _layers.append(nn.LeakyReLU())
        else:
            _layers.append(nn.ReLU())
        _layers.append(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups[1],
                bias=bias
            )
        )
        self.layers = nn.Sequential(*_layers)
        
        if self.residual:
            self.skip_connection = SkipConnection(
                in_channels=in_channels,
                out_channels=out_channels,
                scale_factor=scale_factor
            )

        if self.alpha_residual:
            self.alpha = nn.Parameter(th.zeros(1) + 1e-12)

    def set_mode(self, **kwargs):
        if 'skip' in kwargs:
            self.skip = kwargs['skip']

        if 'residual' in kwargs:
            self.residual = kwargs['residual']

    def forward(self, input: th.Tensor) -> th.Tensor:
        if self.skip:
            return self.skip_connection(input)

        if not self.residual:
            return self.layers(input)

        if self.alpha_residual:
            return self.alpha * self.layers(input) + self.skip_connection(input)

        return self.layers(input) + self.skip_connection(input)

class LinearSkip(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(LinearSkip, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if num_inputs % num_outputs != 0 and num_outputs % num_inputs != 0:
            mean_channels = np.gcd(num_inputs, num_outputs)
            print(f"[WW] gcd skip: {num_inputs} -> {mean_channels} -> {num_outputs}")
            assert(False)

    def forward(self, input: th.Tensor):
        num_inputs  = self.num_inputs
        num_outputs = self.num_outputs
        
        if num_inputs == num_outputs:
            return input

        if num_inputs % num_outputs == 0 or num_outputs % num_inputs == 0:

            if num_inputs > num_outputs:
                return reduce(input, 'b (c n) -> b c', 'mean', n = num_inputs // num_outputs)

            if num_outputs > num_inputs:
                return repeat(input, 'b c -> b (c n)', n = num_outputs // num_inputs)

        mean_channels = np.gcd(num_inputs, num_outputs)
        input = reduce(input, 'b (c n) -> b c', 'mean', n = num_inputs // mean_channels)
        return repeat(input, 'b c -> b (c n)', n = num_outputs // mean_channels)

class LinearResidual(nn.Module):
    def __init__(
        self, 
        num_inputs: int, 
        num_outputs: int, 
        num_hidden: int = None,
        residual: bool = True, 
        alpha_residual: bool = False,
        input_relu: bool = True
    ):
        super(LinearResidual, self).__init__()
        
        self.residual       = residual
        self.alpha_residual = alpha_residual

        if num_hidden is None:
            num_hidden = num_outputs

        _layers = []
        if input_relu:
            _layers.append(nn.ReLU())
        _layers.append(nn.Linear(num_inputs, num_hidden))
        _layers.append(nn.ReLU())
        _layers.append(nn.Linear(num_hidden, num_outputs))

        self.layers = nn.Sequential(*_layers)

        if residual:
            self.skip = LinearSkip(num_inputs, num_outputs)
        
        if alpha_residual:
            self.alpha = nn.Parameter(th.zeros(1)+1e-16)

    def forward(self, input: th.Tensor):
        if not self.residual:
            return self.layers(input)
        
        if not self.alpha_residual:
            return self.skip(input) + self.layers(input)

        return self.skip(input) + self.alpha * self.layers(input)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels: int, channel_factor = 4, space_factor = 5):
        super(ResidualAttentionBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.gate2d = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=space_factor,
                padding=0,
                stride=space_factor
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1,
                out_channels=channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=space_factor,
                padding=0,
                stride=space_factor
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=channels,
                out_channels=1,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid()
        )
        #self.gate2d[-2].bias.data = th.ones_like(self.gate2d[-2].bias.data) + 4

        self.channel_gate = nn.Sequential(
            nn.ReLU(),
            LambdaModule(lambda x: reduce(x, 'b c h w -> b c', 'mean')),
            nn.Linear(channels, channels // channel_factor),
            nn.ReLU(),
            nn.Linear(channels // channel_factor, channels),
            nn.Sigmoid(),
            LambdaModule(lambda x: rearrange(x, 'b c -> b c 1 1'))
        )
        #self.channel_gate[-3].bias.data = th.ones_like(self.channel_gate[-3].bias.data) + 4
        

    def forward(self, input: th.Tensor) -> th.Tensor:
        #print(f"channel_gate: {th.mean(self.channel_gate[-3].weight.data).item():.9e} +- {th.std(self.channel_gate[-3].weight.data).item():.2e}")
        #print(f"gate2d: {th.mean(self.gate2d[-2].weight.data).item():.9e} +- {th.std(self.gate2d[-2].weight.data).item():.9e}")
        return input + self.layers(input) * self.gate2d(input) * self.channel_gate(input)
