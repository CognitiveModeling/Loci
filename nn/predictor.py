import torch.nn as nn
import torch as th
import numpy as np
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, LambdaModule
from nn.eprop_lstm import EpropLSTM
from nn.eprop_gate_l0rd import EpropGateL0rd
from nn.residual import ResidualBlock
from nn.tracker import CaterSnitchTracker
from nn.vae import VariationalFunction
from torch.autograd import Function
from einops import rearrange, repeat, reduce

from typing import Tuple, Union, List
import utils
import cv2

__author__ = "Manuel Traub"

class AlphaAttention(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_objects,
        heads,
        dropout = 0.0
    ):
        super(AlphaAttention, self).__init__()

        self.to_sequence = LambdaModule(lambda x: rearrange(x, '(b o) c -> b o c', o = num_objects))
        self.to_batch    = LambdaModule(lambda x: rearrange(x, 'b o c -> (b o) c', o = num_objects))

        self.alpha     = nn.Parameter(th.zeros(1)+1e-12)
        self.attention = nn.MultiheadAttention(
            num_hidden, 
            heads, 
            dropout = dropout, 
            batch_first = True
        )

    def forward(self, x: th.Tensor):
        x = self.to_sequence(x)
        x = x + self.alpha * self.attention(x, x, x, need_weights=False)[0]
        return self.to_batch(x)

class EpropAlphaGateL0rd(nn.Module):
    def __init__(self, num_hidden, batch_size, reg_lambda):
        super(EpropAlphaGateL0rd, self).__init__()
        
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)
        self.l0rd  = EpropGateL0rd(
            num_inputs  = num_hidden, 
            num_hidden  = num_hidden, 
            num_outputs = num_hidden, 
            reg_lambda  = reg_lambda,
            batch_size = batch_size
        )

    def forward(self, input):
        return input + self.alpha * self.l0rd(input)

class InputEmbeding(nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super(InputEmbeding, self).__init__()

        self.embeding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_inputs, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
        )
        self.skip = LambdaModule(
            lambda x: repeat(x, 'b c -> b (n c)', n = num_hidden // num_inputs)
        )
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)

    def forward(self, input: th.Tensor):
        return self.skip(input) + self.alpha * self.embeding(input)

class OutputEmbeding(nn.Module):
    def __init__(self, num_hidden, num_outputs):
        super(OutputEmbeding, self).__init__()

        self.embeding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(num_hidden, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
        )
        self.skip = LambdaModule(
            lambda x: reduce(x, 'b (n c) -> b c', 'mean', n = num_hidden // num_outputs)
        )
        self.alpha = nn.Parameter(th.zeros(1)+1e-12)

    def forward(self, input: th.Tensor):
        return self.skip(input) + self.alpha * self.embeding(input)

class EpropGateL0rdTransformer(nn.Module):
    def __init__(
        self, 
        channels,
        num_objects,
        batch_size,
        heads, 
        deepth,
        reg_lambda,
        dropout=0.0
    ):
        super(EpropGateL0rdTransformer, self).__init__()

        num_inputs  = channels
        num_outputs = channels
        num_hidden  = channels * heads
        
        self.deepth = deepth
        _layers = []
        _layers.append(InputEmbeding(num_inputs, num_hidden))

        for i in range(deepth):
            _layers.append(AlphaAttention(num_hidden, num_objects, heads, dropout))
            _layers.append(EpropAlphaGateL0rd(num_hidden, batch_size * num_objects, reg_lambda))

        _layers.append(OutputEmbeding(num_hidden, num_outputs))
        self.layers = nn.Sequential(*_layers)

    def get_openings(self):
        openings = 0
        for i in range(self.deepth):
            openings += self.layers[2 * (i + 1)].l0rd.openings.item()

        return openings / self.deepth

    def get_hidden(self):
        states = []
        for i in range(self.deepth):
            states.append(self.layers[2 * (i + 1)].l0rd.get_hidden())

        return th.cat(states, dim=1)

    def set_hidden(self, hidden):
        states = th.chunk(hidden, self.deepth, dim=1)
        for i in range(self.deepth):
            self.layers[2 * (i + 1)].l0rd.set_hidden(states[i])

    def forward(self, input: th.Tensor) -> th.Tensor:
        return self.layers(input)

class PriorityEncoder(nn.Module):
    def __init__(self, num_objects, batch_size):
        super(PriorityEncoder, self).__init__()

        self.num_objects = num_objects
        self.register_buffer("indices", repeat(th.arange(num_objects), 'a -> (b a) 1', b=batch_size), persistent=False)

        self.index_factor    = nn.Parameter(th.ones(1))
        self.priority_factor = nn.Parameter(th.ones(1))

    def forward(self, priority: th.Tensor) -> th.Tensor:
        
        priority = priority * self.num_objects + th.randn_like(priority) * 0.1
        priority = priority * self.priority_factor 
        priority = priority + self.indices * self.index_factor
        priority = rearrange(priority, '(b o) 1 -> b o', o=self.num_objects)

        return priority * 25

class LatentEpropPredictor(nn.Module): 
    def __init__(
        self, 
        heads: int, 
        layers: int,
        reg_lambda: float,
        num_objects: int, 
        gestalt_size: int, 
        vae_factor: float, 
        batch_size: int,
        camera_view_matrix = None,
        zero_elevation = None
    ):
        super(LatentEpropPredictor, self).__init__()
        self.num_objects = num_objects
        self.std_alpha   = nn.Parameter(th.zeros(1)+1e-16)

        self.reg_lambda = reg_lambda
        self.predictor  = EpropGateL0rdTransformer(
            channels    = gestalt_size + 4,
            heads       = heads, 
            deepth      = layers,
            num_objects = num_objects,
            reg_lambda  = reg_lambda, 
            batch_size  = batch_size,
        )

        self.tracker = None
        if camera_view_matrix is not None:
            self.tracker = CaterSnitchTracker(
                latent_size        = gestalt_size + 2,
                num_objects        = num_objects,
                camera_view_matrix = camera_view_matrix,
                zero_elevation     = zero_elevation
            )

        self.vae = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b c -> b c 1 1')),
            ResidualBlock(gestalt_size, gestalt_size * 2, kernel_size=1),
            VariationalFunction(factor = vae_factor),
            nn.Sigmoid(),
            LambdaModule(lambda x: rearrange(x, '(b o) c 1 1 -> b (o c)', o=num_objects))
        )

        self.priority_encoder = PriorityEncoder(num_objects, batch_size)
                
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o=num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o=num_objects))

    def get_openings(self):
        return self.predictor.get_openings()

    def get_hidden(self):
        return self.predictor.get_hidden()

    def set_hidden(self, hidden):
        self.predictor.set_hidden(hidden)

    def forward(
        self, 
        position: th.Tensor, 
        gestalt: th.Tensor, 
        priority: th.Tensor,
    ):

        position = self.to_batch(position)
        gestalt  = self.to_batch(gestalt)
        priority = self.to_batch(priority)

        input  = th.cat((position, gestalt, priority), dim=1)
        output = self.predictor(input)

        xy       = output[:,:2]
        std      = output[:,2:3]
        gestalt  = output[:,3:-1]
        priority = output[:,-1:]

        snitch_position = None
        if self.tracker is not None:
            snitch_position = self.tracker(xy, output[:,2:])

        position = th.cat((xy, std * self.std_alpha), dim=1)
        
        position = self.to_shared(position)
        gestalt  = self.vae(gestalt)
        priority = self.priority_encoder(priority)

        return position, gestalt, priority, snitch_position
