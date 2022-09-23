import torch.nn as nn
import torch as th
from torch.autograd import Function
import nn as nn_modules
import utils
from nn.residual import ResidualBlock, SkipConnection
from nn.encoder import AggressiveDownConv
from nn.encoder import AggressiveConvTo1x1
from nn.decoder import AggressiveUpConv
from utils.utils import LambdaModule, ForcedAlpha, PrintShape, PushToInf
from nn.predictor import EpropAlphaGateL0rd
from nn.vae import VariationalFunction
from einops import rearrange, repeat, reduce

from typing import Union, Tuple

__author__ = "Manuel Traub"

class BackgroundEnhancer(nn.Module):
    def __init__(
        self, 
        input_size: Tuple[int, int], 
        img_channels: int, 
        level1_channels,
        latent_channels,
        gestalt_size,
        batch_size,
        depth
    ):
        super(BackgroundEnhancer, self).__init__()

        latent_size = [input_size[0] // 16, input_size[1] // 16]
        self.input_size = input_size

        self.register_buffer('init', th.zeros(1).long())
        self.alpha = nn.Parameter(th.zeros(1)+1e-16)

        self.level  = 1
        self.down_level2 = nn.Sequential(
            AggressiveDownConv(img_channels*2+2, level1_channels),
            *[ResidualBlock(level1_channels, level1_channels, alpha_residual = True) for i in range(depth)]
        )

        self.down_level1 = nn.Sequential(
            AggressiveDownConv(level1_channels, latent_channels),
            *[ResidualBlock(latent_channels, latent_channels) for i in range(depth)]
        )

        self.down_level0 = nn.Sequential(
            *[ResidualBlock(latent_channels, latent_channels) for i in range(depth)],
            AggressiveConvTo1x1(latent_channels, latent_size),
            LambdaModule(lambda x: rearrange(x, 'b c 1 1 -> b c')),
            nn.Sigmoid(),
            LambdaModule(lambda x: x + x * (1 - x) * th.randn_like(x)),
        )

        self.bias = nn.Parameter(th.zeros((1, gestalt_size, *latent_size)))

        self.to_grid = nn.Sequential(
            LinearResidual(gestalt_size, gestalt_size, input_relu = False),
            LambdaModule(lambda x: rearrange(x, 'b c -> b c 1 1')),
            LambdaModule(lambda x: x + self.bias),
            *[ResidualBlock(gestalt_size, gestalt_size) for i in range(depth)],
        )            


        self.up_level0 = nn.Sequential(
            ResidualBlock(gestalt_size, latent_channels),
            *[ResidualBlock(latent_channels, latent_channels) for i in range(depth)],
        )

        self.up_level1 = nn.Sequential(
            *[ResidualBlock(latent_channels, latent_channels) for i in range(depth)],
            AggressiveUpConv(latent_channels, level1_channels),
        )

        self.up_level2 = nn.Sequential(
            *[ResidualBlock(level1_channels, level1_channels, alpha_residual = True) for i in range(depth)],
            AggressiveUpConv(level1_channels, img_channels),
        )

        self.to_channels = nn.ModuleList([
            SkipConnection(img_channels*2+2, latent_channels),
            SkipConnection(img_channels*2+2, level1_channels),
            SkipConnection(img_channels*2+2, img_channels*2+2),
        ])

        self.to_img = nn.ModuleList([
            SkipConnection(latent_channels, img_channels),
            SkipConnection(level1_channels, img_channels),
            SkipConnection(img_channels,    img_channels),
        ])

        self.mask   = nn.Parameter(th.ones(1, 1, *input_size) * 10)
        self.object = nn.Parameter(th.ones(1, img_channels, *input_size))

        self.register_buffer('latent', th.zeros((batch_size, gestalt_size)), persistent=False)

    def get_init(self):
        return self.init.item()

    def step_init(self):
        self.init = self.init + 1

    def detach(self):
        self.latent = self.latent.detach()

    def reset_state(self):
        self.latent = th.zeros_like(self.latent)

    def set_level(self, level):
        self.level = level

    def encoder(self, input):
        latent = self.to_channels[self.level](input)

        if self.level >= 2:
            latent = self.down_level2(latent)

        if self.level >= 1:
            latent = self.down_level1(latent)

        return self.down_level0(latent)

    def get_last_latent_gird(self):
        return self.to_grid(self.latent) * self.alpha

    def decoder(self, latent, input):
        grid   = self.to_grid(latent)
        latent = self.up_level0(grid)

        if self.level >= 1:
            latent = self.up_level1(latent)

        if self.level >= 2:
            latent = self.up_level2(latent)

        object = reduce(self.object, '1 c (h h2) (w w2) -> 1 c h w', 'mean', h = input.shape[2], w = input.shape[3])
        object = repeat(object,      '1 c h w -> b c h w', b = input.shape[0])

        return th.sigmoid(object + self.to_img[self.level](latent)), grid

    def forward(self, input: th.Tensor, error: th.Tensor = None, mask: th.Tensor = None):
        
        last_bg = self.decoder(self.latent, input)[0]

        bg_error = th.sqrt(reduce((input - last_bg)**2, 'b c h w -> b 1 h w', 'mean')).detach()
        bg_mask  = (bg_error < th.mean(bg_error) + th.std(bg_error)).float().detach()

        if error is None or self.get_init() < 2:
            error = bg_error

        if mask is None or self.get_init() < 2:
            mask = bg_mask

        self.latent = self.encoder(th.cat((input, last_bg, error, mask), dim=1))

        mask = reduce(self.mask, '1 1 (h h2) (w w2) -> 1 1 h w', 'mean', h = input.shape[2], w = input.shape[3])
        mask = repeat(mask,      '1 1 h w -> b 1 h w', b = input.shape[0]) * 0.1
        
        background, grid = self.decoder(self.latent, input)

        if self.get_init() < 1:
            return mask, background, background

        if self.get_init() < 2:
            return mask, th.zeros_like(background), background

        return mask, background, background

class PrecalculatedBackground(nn.Module):
    def __init__(
        self, 
        input_size: Tuple[int, int],
        img_channels: int, 
    ):
        super(PrecalculatedBackground, self).__init__()

        self.mask       = nn.Parameter(th.ones(1, 1, input_size[0], input_size[1]))
        self.init       = 0
        self.alpha      = nn.Parameter(th.zeros(1))
        self.to_inf     = PushToInf()
        self.local_bias = nn.Parameter(th.zeros(1, img_channels, input_size[0], input_size[1]))
        self.register_buffer("background", th.zeros(1, img_channels, input_size[0], input_size[1]), persistent = False)

    def set_background(self, background):
        self.background = background

    def get_init(self):
        return th.tanh(self.to_inf(self.alpha * 0.5)).item()

    def set_level(self, level):
        self.level = level

    def forward(self, input: th.Tensor, error: th.Tensor = None, mask: th.Tensor = None):

        H, W = input.shape[2:]

        background = self.background
        if W < background.shape[-1]:
            background = reduce(background, 'b c (h h2) (w w2) -> b c h w', 'mean', h = H, w = W)
        
        mask = self.mask
        if W < mask.shape[-1]:
            mask = reduce(mask, 'b c (h h2) (w w2) -> b c h w', 'mean', h = H, w = W)

        mask  = mask.expand(input.shape[0], 1, *input.shape[2:])
        return mask, background * th.tanh(self.to_inf(self.alpha * 0.5)), background
