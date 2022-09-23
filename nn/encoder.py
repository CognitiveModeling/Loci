import torch.nn as nn
import torch as th
import numpy as np
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, Prioritize, LambdaModule, ForcedAlpha
from nn.eprop_lstm import EpropLSTM
from nn.residual import ResidualBlock, SkipConnection
from nn.eprop_gate_l0rd import EpropGateL0rd
from torch.autograd import Function
from einops import rearrange, repeat, reduce

from typing import Tuple, Union, List
import utils
import cv2

__author__ = "Manuel Traub"

class NeighbourChannels(nn.Module):
    def __init__(self, channels):
        super(NeighbourChannels, self).__init__()

        self.register_buffer("weights", th.ones(channels, channels, 1, 1), persistent=False)

        for i in range(channels):
            self.weights[i,i,0,0] = 0

    def forward(self, input: th.Tensor):
        return nn.functional.conv2d(input, self.weights)

class ObjectTracker(nn.Module):
    def __init__(self, num_objects: int, size: Union[int, Tuple[int, int]]): 
        super(ObjectTracker, self).__init__()
        self.num_objects = num_objects
        self.neighbours  = NeighbourChannels(num_objects)
        self.prioritize  = Prioritize(num_objects)
        self.gaus2d      = Gaus2D(size)
        self.to_batch    = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared   = BatchToSharedObjects(num_objects)

    def forward(
        self, 
        input: th.Tensor, 
        mask: th.Tensor,
        object: th.Tensor,
        position: th.Tensor,
        priority: th.Tensor
    ):
        batch_size = input.shape[0]
        size       = input.shape[2:]

        mask     = mask.detach()
        position = position.detach()
        priority = priority.detach()
        
        bg_mask     = repeat(mask[:,-1:], 'b 1 h w -> b c h w', c = self.num_objects)
        mask        = mask[:,:-1]
        mask_others = self.neighbours(mask)

        own_gaus2d    = self.to_shared(self.gaus2d(self.to_batch(position)))
        others_gaus2d = self.neighbours(self.prioritize(own_gaus2d, priority))
        
        input         = repeat(input,            'b c h w -> b o c h w', o = self.num_objects)
        bg_mask       = rearrange(bg_mask,       'b o h w -> b o 1 h w')
        mask_others   = rearrange(mask_others,   'b o h w -> b o 1 h w')
        mask          = rearrange(mask,          'b o h w -> b o 1 h w')
        object        = rearrange(object,        'b (o c) h w -> b o c h w', o = self.num_objects)
        own_gaus2d    = rearrange(own_gaus2d,    'b o h w -> b o 1 h w')
        others_gaus2d = rearrange(others_gaus2d, 'b o h w -> b o 1 h w')
        
        output = th.cat((input, mask, mask_others, bg_mask, object, own_gaus2d, others_gaus2d), dim=2) 
        output = rearrange(output, 'b o c h w -> (b o) c h w')

        return output

class AggressiveDownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AggressiveDownConv, self).__init__()
        assert out_channels % in_channels == 0
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = out_channels, 
                kernel_size  = 11,
                stride       = 4,
                padding      = 5
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = out_channels, 
                out_channels = out_channels, 
                kernel_size  = 3,
                padding      = 1
            )
        )
        self.alpha = nn.Parameter(th.zeros(1) + 1e-12)
        self.size_factor = 4
        self.channels_factor = out_channels // in_channels


    def forward(self, input: th.Tensor):
        s = self.size_factor
        c = self.channels_factor
        skip = reduce(input, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=s, w2=s)
        skip = repeat(skip, 'b c h w -> b (c n) h w', n=c)
        return skip + self.alpha * self.layers(input)

class AggressiveConvTo1x1(nn.Module):
    def __init__(self, in_channels, out_channels, size: Union[int, Tuple[int, int]]):
        super(AggressiveConvTo1x1, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = in_channels, 
                kernel_size  = 5,
                stride       = 3,
                padding      = 3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = out_channels, 
                kernel_size  = ((size[0] + 1)//3 + 1, (size[1] + 1)//3 + 1)
            )
        )
        self.alpha  = nn.Parameter(th.zeros(1) + 1e-12)
        self.size   = size
        self.factor = out_channels // in_channels


    def forward(self, input: th.Tensor):
        skip = reduce(input, 'b c h w -> b c 1 1', 'mean')
        skip = repeat(skip,  'b c 1 1 -> b (c n) 1 1', n = self.factor)
        return skip + self.alpha * self.layers(input)

class PixelToPosition(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]]):
        super(PixelToPosition, self).__init__()

        self.register_buffer("grid_x", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_y", th.arange(size[1]), persistent=False)

        self.grid_x = (self.grid_x / (size[0]-1)) * 2 - 1
        self.grid_y = (self.grid_y / (size[1]-1)) * 2 - 1

        self.grid_x = self.grid_x.view(1, 1, -1, 1).expand(1, 1, *size).clone()
        self.grid_y = self.grid_y.view(1, 1, 1, -1).expand(1, 1, *size).clone()

        self.size = size

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1

        input = rearrange(input, 'b c h w -> b c (h w)')
        input = th.softmax(input, dim=2)
        input = rearrange(input, 'b c (h w) -> b c h w', h = self.size[0], w = self.size[1])

        x = th.sum(input * self.grid_x, dim=(2,3))
        y = th.sum(input * self.grid_y, dim=(2,3))

        return th.cat((x,y),dim=1)

class PixelToSTD(nn.Module):
    def __init__(self):
        super(PixelToSTD, self).__init__()
        self.alpha = ForcedAlpha()

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1
        return self.alpha(reduce(th.sigmoid(input - 10), 'b c h w -> b c', 'mean'))

class PixelToPriority(nn.Module):
    def __init__(self):
        super(PixelToPriority, self).__init__()

    def forward(self, input: th.Tensor):
        assert input.shape[1] == 1
        return reduce(th.tanh(input), 'b c h w -> b c', 'mean')

class GPEncoder(nn.Module):
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]], 
        latent_size: Union[int, Tuple[int, int]],
        num_objects: int, 
        img_channels: int,
        hidden_channels: int,
        level1_channels: int,
        num_layers: int,
        gestalt_size: int,
        batch_size: int,
        reg_lambda: float
    ):
        super(GPEncoder, self).__init__()

        self.num_objects  = num_objects
        self.gestalt_size = gestalt_size
        self.latent_size  = latent_size
        self.level        = 1

        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = self.num_objects))

        print(f"Level1 channels: {level1_channels}")

        self.tracker = nn.ModuleList([
            ObjectTracker(num_objects, (input_size[0] // 16, input_size[1] // 16)),
            ObjectTracker(num_objects, (input_size[0] //  4, input_size[1] //  4)),
            ObjectTracker(num_objects, (input_size[0], input_size[1]))
        ])

        self.to_channels = nn.ModuleList([
            SkipConnection(img_channels, hidden_channels),
            SkipConnection(img_channels, level1_channels),
            SkipConnection(img_channels, img_channels)
        ])

        _layers2 = []
        _layers2.append(AggressiveDownConv(img_channels, level1_channels))
        for i in range(num_layers):
            _layers2.append(ResidualBlock(level1_channels, level1_channels, alpha_residual=True))

        _layers1 = []
        _layers1.append(AggressiveDownConv(level1_channels, hidden_channels))

        _layers0 = []
        for i in range(num_layers):
            _layers0.append(
                ResidualBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels
                )
            )

        self.layers2 = nn.Sequential(*_layers2)
        self.layers1 = nn.Sequential(*_layers1)
        self.layers0 = nn.Sequential(*_layers0)

        _position_encoder = []
        for i in range(num_layers):
            _position_encoder.append(
                ResidualBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels
                )
            )
        _position_encoder.append(
            ResidualBlock(
                in_channels  = hidden_channels,
                out_channels = 3
            )
        )
        self.position_encoder = nn.Sequential(*_position_encoder)

        self.xy_encoder = PixelToPosition(latent_size)
        self.std_encoder = PixelToSTD()
        self.priority_encoder = PixelToPriority()

        _gestalt_encoder = []
        _gestalt_encoder.append( 
            AggressiveConvTo1x1(
                in_channels = hidden_channels, 
                out_channels = max(hidden_channels, gestalt_size),
                size = latent_size
            )
        )
        for i in range(num_layers):
            _gestalt_encoder.append(
                ResidualBlock(
                    in_channels  = max(hidden_channels, gestalt_size),
                    out_channels = max(hidden_channels, gestalt_size),
                    kernel_size  = 1
                )
            )
        _gestalt_encoder.append(
            ResidualBlock(
                in_channels  = max(hidden_channels, gestalt_size),
                out_channels = gestalt_size, 
                kernel_size  = 1 
            )
        )
        _gestalt_encoder.append(LambdaModule(lambda x: rearrange(x, 'b c 1 1 -> b c')))

        self.gestalt_encoder = nn.Sequential(*_gestalt_encoder)

    def set_level(self, level):
        self.level = level

    def forward(
        self, 
        input: th.Tensor,
        mask: th.Tensor,
        object: th.Tensor,
        position: th.Tensor,
        priority: th.Tensor
    ):
        
        latent = self.tracker[self.level](input, mask, object, position, priority)
        latent = self.to_channels[self.level](latent)

        if self.level >= 2:
            latent = self.layers2(latent)

        if self.level >= 1:
            latent = self.layers1(latent)

        latent  = self.layers0(latent)
        gestalt = self.gestalt_encoder(latent)

        latent   = self.position_encoder(latent)
        std      = self.std_encoder(latent[:,0:1])
        xy       = self.xy_encoder(latent[:,1:2])
        priority = self.priority_encoder(latent[:,2:3])

        position = self.to_shared(th.cat((xy, std), dim=1))
        gestalt  = self.to_shared(gestalt)
        priority = self.to_shared(priority)

        return position, gestalt, priority

