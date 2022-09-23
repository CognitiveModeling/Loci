import torch.nn as nn
import torch as th
import numpy as np
from nn.residual import ResidualBlock, SkipConnection
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, Prioritize, LambdaModule
from torch.autograd import Function
from einops import rearrange, repeat, reduce

from typing import Tuple, Union, List
import utils

__author__ = "Manuel Traub"

class GPMerge(nn.Module):
    def __init__(
        self, 
        latent_size: Union[int, Tuple[int, int]],
        num_objects: int
    ):

        super(GPMerge, self).__init__()
        self.num_objects = num_objects

        self.gaus2d = Gaus2D(size=latent_size)

        self.to_batch  = SharedObjectsToBatch(num_objects)
        self.to_shared = BatchToSharedObjects(num_objects)

        self.prioritize = Prioritize(num_objects)

    def forward(self, position, gestalt, priority):
        
        position = rearrange(position, 'b (o c) -> (b o) c', o = self.num_objects)
        gestalt  = rearrange(gestalt, 'b (o c) -> (b o) c 1 1', o = self.num_objects)

        position = self.gaus2d(position)
        position = self.to_batch(self.prioritize(self.to_shared(position), priority))

        return position * gestalt

class AggressiveUpConv(nn.Module):
    def __init__(self, in_channels, img_channels, alpha = 1e-16):
        super(AggressiveUpConv, self).__init__()
        
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels  = in_channels, 
                out_channels = in_channels, 
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels  = in_channels, 
                out_channels = img_channels, 
                kernel_size  = 12,
                stride       = 4,
                padding      = 4
            )
        )
        self.alpha = nn.Parameter(th.zeros(1) + alpha)
        self.size_factor = 4
        self.channels_factor = in_channels // img_channels


    def forward(self, input: th.Tensor):
        s = self.size_factor
        c = self.channels_factor
        skip = reduce(input, 'b (c n) h w -> b c h w', 'mean', n=c)
        skip = repeat(skip, 'b c h w -> b c (h h2) (w w2)', h2=s, w2=s)
        return skip + self.alpha * self.layers(input)

class GPDecoder(nn.Module):
    def __init__(
        self, 
        latent_size: Union[int, Tuple[int, int]],
        gestalt_size: int,
        num_objects: int, 
        img_channels: int,
        hidden_channels: int,
        level1_channels: int,
        num_layers: int,
    ): 

        super(GPDecoder, self).__init__()
        self.to_batch  = SharedObjectsToBatch(num_objects)
        self.to_shared = BatchToSharedObjects(num_objects)
        self.level     = 1

        assert(level1_channels % img_channels == 0)
        level1_factor   = level1_channels // img_channels
        print(f"Level1 channels: {level1_channels}")

        self.merge = GPMerge(
            latent_size = latent_size,
            num_objects = num_objects,
        )

        _layer0 = []
        _layer0.append(
            ResidualBlock(
                in_channels  = gestalt_size,
                out_channels = hidden_channels,
                input_nonlinearity = False
            )
        )
        for i in range(num_layers-1):
            _layer0.append(
                ResidualBlock(
                    in_channels  = hidden_channels,
                    out_channels = hidden_channels
                )
            )
        self.layer0 = nn.Sequential(*_layer0)

        self.to_mask_level0 = ResidualBlock(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
        )

        self.to_mask_level1 = AggressiveUpConv(
            in_channels  = hidden_channels,
            img_channels = level1_factor,
        )

        self.to_mask_level2 = nn.Sequential(
            ResidualBlock(
                in_channels    = hidden_channels,
                out_channels   = hidden_channels,
            ),
            ResidualBlock(
                in_channels    = hidden_channels,
                out_channels   = hidden_channels,
            ),
            AggressiveUpConv(
                in_channels  = hidden_channels,
                img_channels = 4,
                alpha = 1
            ),
            AggressiveUpConv(
                in_channels  = 4,
                img_channels = 1,
                alpha = 1
            )
        )

        self.to_object_level0 = ResidualBlock(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
        )

        self.to_object_level1 = AggressiveUpConv(
            in_channels  = hidden_channels,
            img_channels = level1_channels,
        )

        self.to_object_level2 = nn.Sequential(
            ResidualBlock(
                in_channels    = hidden_channels,
                out_channels   = hidden_channels,
            ),
            ResidualBlock(
                in_channels    = hidden_channels,
                out_channels   = hidden_channels,
            ),
            AggressiveUpConv(
                in_channels  = hidden_channels,
                img_channels = 12,
                alpha = 1
            ),
            AggressiveUpConv(
                in_channels  = 12,
                img_channels = img_channels,
                alpha = 1
            )
        )

        self.mask_to_pixel = nn.ModuleList([
            SkipConnection(hidden_channels, 1),
            SkipConnection(level1_factor,   1),
            SkipConnection(1,               1),
        ])
        self.object_to_pixel = nn.ModuleList([
            SkipConnection(hidden_channels, img_channels),
            SkipConnection(level1_channels, img_channels),
            SkipConnection(img_channels,    img_channels),
        ])

        self.mask_alpha = nn.Parameter(th.zeros(1)+1e-16)
        self.object_alpha = nn.Parameter(th.zeros(1)+1e-16)

    def set_level(self, level):
        self.level = level

    def forward(self, position, gestalt, priority = None):

        maps = self.layer0(self.merge(position, gestalt, priority))

        mask0   = self.to_mask_level0(maps)
        object0 = self.to_object_level0(maps)

        if self.level > 0:
            mask   = self.to_mask_level1(mask0)
            object = self.to_object_level1(object0)

        if self.level > 1:
            mask   = repeat(mask, 'b c h w -> b c (h h2) (w w2)', h2 = 4, w2 = 4)   + self.to_mask_level2(mask0) * self.mask_alpha
            object = repeat(object, 'b c h w -> b c (h h2) (w w2)', h2 = 4, w2 = 4) + self.to_object_level2(object0) * self.object_alpha

        mask   = self.mask_to_pixel[self.level](mask)
        object = self.object_to_pixel[self.level](object)

        return self.to_shared(mask), self.to_shared(object)
