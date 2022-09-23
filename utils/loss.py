import torch as th
import torchvision as tv
from torch import nn
from utils.utils import BatchToSharedObjects, SharedObjectsToBatch, LambdaModule
from pytorch_msssim import ms_ssim as msssim, ssim
from einops import rearrange, repeat, reduce

__author__ = "Manuel Traub"

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, output: th.Tensor, target: th.Tensor):
        return -ssim(output, target)

class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()

    def forward(self, output: th.Tensor, target: th.Tensor):
        return -msssim(output, target) #, normalize="relu")

class PositionLoss(nn.Module):
    def __init__(self, num_objects: int, teacher_forcing: int):
        super(PositionLoss, self).__init__()

        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.last_mask = None
        self.t = 0 
        self.teacher_forcing = teacher_forcing

    def reset_state(self):
        self.last_mask = None
        self.t = 0

    def forward(self, position, position_last, mask):
        
        mask = th.max(th.max(mask, dim=3)[0], dim=2)[0]
        mask = self.to_batch(mask).detach()
        self.t = self.t + 1

        if self.last_mask is None or self.t <= self.teacher_forcing:
            self.last_mask = mask.detach()
            return th.zeros(1, device=mask.device)

        self.last_mask = th.maximum(self.last_mask, mask)

        position      = self.to_batch(position)
        position_last = self.to_batch(position_last).detach()

        return 0.01 * th.mean(self.last_mask * (position - position_last)**2)


class MaskModulatedObjectLoss(nn.Module):
    def __init__(self, num_objects: int, teacher_forcing: int):
        super(MaskModulatedObjectLoss, self).__init__()

        self.to_batch  = SharedObjectsToBatch(num_objects)
        self.last_mask = None
        self.t = 0 
        self.teacher_forcing = teacher_forcing

    def reset_state(self):
        self.last_mask = None
        self.t = 0
    
    def forward(
        self, 
        object_output,
        object_target,
        mask: th.Tensor
    ):
        mask = self.to_batch(mask).detach()
        mask = th.max(th.max(mask, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        self.t = self.t + 1

        if self.last_mask is None or self.t <= self.teacher_forcing:
            self.last_mask = mask.detach()
            return th.zeros(1, device=mask.device)

        self.last_mask = th.maximum(self.last_mask, mask).detach()

        object_output = th.sigmoid(self.to_batch(object_output) - 2.5)
        object_target = th.sigmoid(self.to_batch(object_target) - 2.5)

        return th.mean((1 - mask) * self.last_mask * (object_output - object_target)**2)

class ObjectModulator(nn.Module):
    def __init__(self, num_objects: int): 
        super(ObjectModulator, self).__init__()
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b (o c)', o = num_objects))
        self.position  = None
        self.gestalt   = None

    def reset_state(self):
        self.position = None
        self.gestalt  = None

    def forward(self, position: th.Tensor, gestalt: th.Tensor, mask: th.Tensor):

        position = self.to_batch(position)
        gestalt  = self.to_batch(gestalt)

        if self.position is None or self.gestalt is None:
            self.position = position.detach()
            self.gestalt  = gestalt.detach()
            return self.to_shared(position), self.to_shared(gestalt)

        mask = th.max(th.max(mask, dim=3)[0], dim=2)[0]
        mask = self.to_batch(mask.detach())

        _position = mask * position + (1 - mask) * self.position
        position  = th.cat((position[:,:-1], _position[:,-1:]), dim=1)
        gestalt   = mask * gestalt  + (1 - mask) * self.gestalt

        self.gestalt = gestalt.detach()
        self.position = position.detach()
        return self.to_shared(position), self.to_shared(gestalt)

class MoveToCenter(nn.Module):
    def __init__(self, num_objects: int):
        super(MoveToCenter, self).__init__()

        self.to_batch2d = SharedObjectsToBatch(num_objects)
        self.to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o = num_objects))
    
    def forward(self, input: th.Tensor, position: th.Tensor):
        
        input    = self.to_batch2d(input)
        position = self.to_batch(position).detach()
        position = th.stack((position[:,1], position[:,0]), dim=1)

        theta = th.tensor([1, 0, 0, 1], dtype=th.float, device=input.device).view(1,2,2)
        theta = repeat(theta, '1 a b -> n a b', n=input.shape[0])

        position = rearrange(position, 'b c -> b c 1')
        theta    = th.cat((theta, position), dim=2)

        grid   = nn.functional.affine_grid(theta, input.shape, align_corners=False)
        output = nn.functional.grid_sample(input, grid, align_corners=False)

        return output

class TranslationInvariantObjectLoss(nn.Module):
    def __init__(self, num_objects: int, teacher_forcing: int):
        super(TranslationInvariantObjectLoss, self).__init__()

        self.move_to_center  = MoveToCenter(num_objects)
        self.to_batch        = SharedObjectsToBatch(num_objects)
        self.last_mask       = None
        self.t               = 0 
        self.teacher_forcing = teacher_forcing

    def reset_state(self):
        self.last_mask = None
        self.t = 0
    
    def forward(
        self, 
        mask: th.Tensor,
        object1: th.Tensor, 
        position1: th.Tensor,
        object2: th.Tensor, 
        position2: th.Tensor,
    ):
        mask = self.to_batch(mask).detach()
        mask = th.max(th.max(mask, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0]
        self.t = self.t + 1

        if self.last_mask is None or self.t <= self.teacher_forcing:
            self.last_mask = mask.detach()
            return th.zeros(1, device=mask.device)

        self.last_mask = th.maximum(self.last_mask, mask).detach()

        object1 = self.move_to_center(th.sigmoid(object1 - 2.5), position1)
        object2 = self.move_to_center(th.sigmoid(object2 - 2.5), position2)

        return th.mean(self.last_mask * (object1 - object2)**2)

