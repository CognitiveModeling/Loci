import torch.nn as nn
import torch as th
import numpy as np
import nn as nn_modules
from torch.autograd import Function
from einops import rearrange, repeat, reduce

from typing import Tuple, Union, List
import utils

__author__ = "Manuel Traub"

class PrintShape(nn.Module):
    def __init__(self, msg = ""):
        super(PrintShape, self).__init__()
        self.msg = msg

    def forward(self, input: th.Tensor):
        if self.msg != "":
            print(self.msg, input.shape)
        else:
            print(input.shape)
        return input

class PrintStats(nn.Module):
    def __init__(self):
        super(PrintStats, self).__init__()

    def forward(self, input: th.Tensor):
        print(
            "min: ", th.min(input).detach().cpu().numpy(),
            ", mean: ", th.mean(input).detach().cpu().numpy(),
            ", max: ", th.max(input).detach().cpu().numpy()
        )
        return input

class PushToInfFunction(Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.saved_tensors[0]
        grad_input = -th.ones_like(grad_output)
        return grad_input

class PushToInf(nn.Module):
    def __init__(self):
        super(PushToInf, self).__init__()
        
        self.fcn = PushToInfFunction.apply

    def forward(self, input: th.Tensor):
        return self.fcn(input)

class ForcedAlpha(nn.Module):
    def __init__(self, speed = 1):
        super(ForcedAlpha, self).__init__()

        self.init   = nn.Parameter(th.zeros(1))
        self.speed  = speed
        self.to_inf = PushToInf()

    def item(self):
        return th.tanh(self.to_inf(self.init * self.speed)).item()

    def forward(self, input: th.Tensor):
        return input * th.tanh(self.to_inf(self.init * self.speed))

class AlphaThreshold(nn.Module):
    def __init__(self, max_value = 1):
        super(AlphaThreshold, self).__init__()

        self.init      = nn.Parameter(th.zeros(1))
        self.to_inf    = PushToInf()
        self.max_value = max_value

    def forward(self):
        return th.tanh(self.to_inf(self.init)) * self.max_value

class InitialLatentStates(nn.Module):
    def __init__(
            self, 
            gestalt_size: int, 
            num_objects: int, 
            size: Tuple[int, int],
            object_permanence_strength: int 
        ):
        super(InitialLatentStates, self).__init__()

        self.num_objects   = num_objects
        self.gestalt_size  = gestalt_size
        self.gestalt_mean  = nn.Parameter(th.zeros(1, gestalt_size))
        self.gestalt_std   = nn.Parameter(th.ones(1, gestalt_size))
        self.std           = nn.Parameter(th.zeros(1))
        self.object_permanence_strength = object_permanence_strength

        self.register_buffer('init', th.zeros(1).long())
        self.register_buffer('priority', th.arange(num_objects).float() * 25, persistent=False)
        self.register_buffer('threshold', th.ones(1) * 0.5)
        self.last_mask = None

        self.gaus2d = nn.Sequential(
            Gaus2D((size[0] // 16, size[1] // 16)),
            Gaus2D((size[0] //  4, size[1] //  4)),
            Gaus2D(size)
        )

        self.level = 1

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_objects))

    def reset_state(self):
        self.last_mask = None

    def set_level(self, level):
        self.level = level

    def get_init(self):
        return self.init.item()

    def step_init(self):
        self.init = self.init + 1

    def forward(
        self, 
        error: th.Tensor, 
        mask: th.Tensor = None, 
        position: th.Tensor = None,
        gestalt: th.Tensor = None,
        priority: th.Tensor = None
    ):

        batch_size = error.shape[0]
        device     = error.device

        if self.last_mask is None:
            self.last_mask = th.zeros((batch_size * self.num_objects, 1), device = device)
        
        if mask is not None:
            mask           = reduce(mask[:,:-1], 'b c h w -> (b c) 1' , 'max').detach()

            if self.get_init() < 2 or self.object_permanence_strength == 0:
                self.last_mask = mask - self.threshold
            elif self.object_permanence_strength <= 1:
                self.last_mask = self.last_mask + mask - self.threshold
            else:
                self.last_mask = self.last_mask + th.maximum(mask - self.threshold, th.zeros_like(mask))
  
        std  = repeat(self.std, '1 -> (b o) 1', b = batch_size, o=self.num_objects)
        mask = (self.last_mask > 0).float().detach()

        gestalt_rand = th.randn((batch_size * self.num_objects, self.gestalt_size), device = device)
        gestalt_new  = th.sigmoid(gestalt_rand * self.gestalt_std + self.gestalt_mean)

        if gestalt is None:
            gestalt = gestalt_new
        else:
            gestalt = self.to_batch(gestalt) * mask + gestalt_new * (1 - mask)

        if priority is None:
            priority = repeat(self.priority, 'o -> (b o) 1', b = batch_size)
        else:
            priority = self.to_batch(priority) * mask + repeat(self.priority, 'o -> (b o) 1', b = batch_size) * (1 - mask)

        xy_rand_new  = th.rand((batch_size * self.num_objects * 10, 2), device = device) * 2 - 1 
        std_new      = th.zeros((batch_size * self.num_objects * 10, 1), device = device)
        position_new = th.cat((xy_rand_new, std_new), dim=1) 

        position2d = self.gaus2d[self.level](position_new)
        position2d = rearrange(position2d, '(b o) 1 h w -> b o h w', b = batch_size)

        rand_error = reduce(position2d * error, 'b o h w -> (b o) 1', 'sum')

        xy_rand_new = rearrange(xy_rand_new, '(b r) c -> r b c', r = 10)
        rand_error  = rearrange(rand_error,  '(b r) c -> r b c', r = 10)

        max_error = th.argmax(rand_error, dim=0, keepdim=True)
        x, y = th.chunk(xy_rand_new, 2, dim=2)
        x = th.gather(x, dim=0, index=max_error).detach().squeeze(dim=0)
        y = th.gather(y, dim=0, index=max_error).detach().squeeze(dim=0)

        if position is None:
            position = th.cat((x, y, std), dim=1) 
        else:
            position = self.to_batch(position) * mask + th.cat((x, y, std), dim=1) * (1 - mask)

        return self.to_shared(position), self.to_shared(gestalt), self.to_shared(priority)

class Gaus2D(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super(Gaus2D, self).__init__()

        self.size = size

        self.register_buffer("grid_x", th.arange(size[0]), persistent=False)
        self.register_buffer("grid_y", th.arange(size[1]), persistent=False)

        self.grid_x = (self.grid_x / (size[0]-1)) * 2 - 1
        self.grid_y = (self.grid_y / (size[1]-1)) * 2 - 1

        self.grid_x = self.grid_x.view(1, 1, -1, 1).expand(1, 1, *size).clone()
        self.grid_y = self.grid_y.view(1, 1, 1, -1).expand(1, 1, *size).clone()

    def forward(self, input: th.Tensor):

        x   = rearrange(input[:,0:1], 'b c -> b c 1 1')
        y   = rearrange(input[:,1:2], 'b c -> b c 1 1')
        std = rearrange(input[:,2:3], 'b c -> b c 1 1')

        x   = th.clip(x, -1, 1)
        y   = th.clip(y, -1, 1)
        std = th.clip(std, 0, 1)
            
        max_size = max(self.size)
        std_x = (1 + max_size * std) / self.size[0]
        std_y = (1 + max_size * std) / self.size[1]

        return th.exp(-1 * ((self.grid_x - x)**2/(2 * std_x**2) + (self.grid_y - y)**2/(2 * std_y**2)))

class SharedObjectsToBatch(nn.Module):
    def __init__(self, num_objects):
        super(SharedObjectsToBatch, self).__init__()

        self.num_objects = num_objects

    def forward(self, input: th.Tensor):
        return rearrange(input, 'b (o c) h w -> (b o) c h w', o=self.num_objects)

class BatchToSharedObjects(nn.Module):
    def __init__(self, num_objects):
        super(BatchToSharedObjects, self).__init__()

        self.num_objects = num_objects

    def forward(self, input: th.Tensor):
        return rearrange(input, '(b o) c h w -> b (o c) h w', o=self.num_objects)

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class PrintGradientFunction(Function):
    @staticmethod
    def forward(ctx, tensor, msg):
        ctx.msg = msg
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        print(f"{ctx.msg}: {th.mean(grad_output).item()} +- {th.std(grad_output).item()}")
        return grad_input, None

class PrintGradient(nn.Module):
    def __init__(self, msg = "PrintGradient"):
        super(PrintGradient, self).__init__()

        self.fcn = PrintGradientFunction.apply
        self.msg = msg

    def forward(self, input: th.Tensor):
        return self.fcn(input, self.msg)

class Prioritize(nn.Module):
    def __init__(self, num_objects):
        super(Prioritize, self).__init__()

        self.num_objects = num_objects
        self.to_batch    = SharedObjectsToBatch(num_objects)

    def forward(self, input: th.Tensor, priority: th.Tensor):
        
        if priority is None:
            return input

        batch_size = input.shape[0]
        weights    = th.zeros((batch_size, self.num_objects, self.num_objects, 1, 1), device=input.device)

        for o in range(self.num_objects):
            weights[:,o,:,0,0] = th.sigmoid(priority[:,:] - priority[:,o:o+1])
            weights[:,o,o,0,0] = weights[:,o,o,0,0] * 0

        input  = rearrange(input, 'b c h w -> 1 (b c) h w')
        weights = rearrange(weights, 'b o i 1 1 -> (b o) i 1 1')

        output = th.relu(input - nn.functional.conv2d(input, weights, groups=batch_size))
        output = rearrange(output, '1 (b c) h w -> b c h w ', b=batch_size)

        return output
