import torch.nn as nn
import torch as th
import numpy as np
from torch.autograd import Function
from einops import rearrange, repeat, reduce
from typing import Tuple, Union, List
from utils.utils import LambdaModule
from nn.residual import LinearResidual

__author__ = "Manuel Traub"

class GridClassifier(nn.Module):
    def __init__(self):
        super(GridClassifier, self).__init__()

        x = (th.arange(36) % 6) - 3 + 0.5
        y = th.floor(th.arange(36) / 6) - 3 + 0.5
        self.register_buffer("grid_centers", th.stack((x,y), dim=1).unsqueeze(dim=0), persistent=False)

    def forward(self, position: th.Tensor):
        grid_distances = th.exp(-th.sum((self.grid_centers - position[:,:2].unsqueeze(dim=1))**2, dim=2))
        return th.softmax(grid_distances, dim=1)

class L1GridDistance(nn.Module):
    def __init__(self, camera_view_matrix, zero_elevation):
        super(L1GridDistance, self).__init__()

        self.to_world = ScreenToWorld(camera_view_matrix, zero_elevation)
        self.to_grid = GridClassifier()

    def forward(self, position, target_label):

        probabilities = self.to_grid(self.to_world(position))
        label = th.argmax(probabilities, dim=1).long()
        target_label = target_label.long()

        x = label % 6
        y = label / 6

        target_x = target_label % 6
        target_y = target_label / 6

        l1   = th.mean((th.abs(x - target_x) + th.abs(y - target_y)).float())
        top1 = th.mean((label == target_label).float())
        top5 = th.mean(th.sum((th.topk(probabilities, 5, dim=1)[1] == target_label.unsqueeze(dim=1)).float(), dim=1))

        return l1, top1, top5, label

class L2TrackingDistance(nn.Module):
    def __init__(self, camera_view_matrix, zero_elevation):
        super(L2TrackingDistance, self).__init__()

        self.to_world = ScreenToWorld(camera_view_matrix, zero_elevation)

    def forward(self, position, target_position):

        position = self.to_world(position)
        return th.sqrt(th.sum((position - target_position)**2, dim=1)), position


class TrackingLoss(nn.Module):
    def __init__(self, camera_view_matrix, zero_elevation):
        super(TrackingLoss, self).__init__()

        self.to_screen = WolrdToScreen(camera_view_matrix, zero_elevation)

    def forward(self, position, target_position):

        target_position = self.to_screen(target_position)
        return th.mean((position - target_position)**2)

class WolrdToScreen(nn.Module):
    def __init__(self, camera_view_matrix, zero_elevation):
        super(WolrdToScreen, self).__init__()

        self.register_buffer('cam', th.tensor(camera_view_matrix).float())

    def forward(self, world_xyz):
        world_xyzw  = th.cat((world_xyz, th.ones((world_xyz.shape[0], 1), device=world_xyz.device)), dim=1)
        screen_yxzw = world_xyzw.mm(self.cam.t())

        screen_yx = th.stack((
            screen_yxzw[:,1] / -screen_yxzw[:,-1],
            screen_yxzw[:,0] / screen_yxzw[:,-1]
        ), dim=1)

        return screen_yx

class ScreenToWorld(nn.Module):
    def __init__(self, camera_view_matrix, zero_elevation):
        super(ScreenToWorld, self).__init__()

        self.register_buffer('cam', th.tensor(camera_view_matrix).float())
        self.z = zero_elevation

    def forward(self, xy, z = None):
        a,b,c,d = th.unbind(self.cam[0])
        e,f,g,h = th.unbind(self.cam[1])
        m,n,o,p = th.unbind(self.cam[3])

        X = xy[:,1]
        Y = xy[:,0]

        if z is None:
            z = th.zeros_like(X)
        else:
            z = z[:,0]
        
        z = z + self.z

        x = (b*h - d*f + Y*b*p - Y*d*n + X*f*p - X*h*n + b*g*z - c*f*z + Y*b*o*z - Y*c*n*z + X*f*o*z - X*g*n*z)/(a*f - b*e + Y*a*n - Y*b*m + X*e*n - X*f*m)
        y = -(a*h - d*e + Y*a*p - Y*d*m + X*e*p - X*h*m + a*g*z - c*e*z + Y*a*o*z - Y*c*m*z + X*e*o*z - X*g*m*z)/(a*f - b*e + Y*a*n - Y*b*m + X*e*n - X*f*m)

        return th.stack((x,y,z), dim=1)


class CaterSnitchTracker(nn.Module):
    def __init__(
        self, 
        latent_size,
        num_objects,
        camera_view_matrix,
        zero_elevation
    ):
        super(CaterSnitchTracker, self).__init__()

        self.to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b o c', o = num_objects))

        self.gate = nn.Sequential(
            LinearResidual(latent_size, latent_size),
            LinearResidual(latent_size, latent_size),
            LinearResidual(latent_size, 1),
            LambdaModule(lambda x: rearrange(x, '(b o) 1 -> b o 1', o = num_objects)),
            nn.Softmax(dim=1),
            LambdaModule(lambda x: x + x * (1 - x) * th.randn_like(x)),
        )

    def forward(self, position, latent_state):
        return th.sum(self.to_shared(position) * self.gate(latent_state), dim=1)


if __name__ == "__main__":
    cam = np.array([
        (1.4503, 1.6376,  0.0000, -0.0251),
        (-1.0346, 0.9163,  2.5685,  0.0095),
        (-0.6606, 0.5850, -0.4748, 10.5666),
        (-0.6592, 0.5839, -0.4738, 10.7452)
    ])

    z = 0.3421497941017151

    to_screen = WolrdToScreen(cam, z)
    to_world  = ScreenToWorld(cam, z)

    xyz = th.rand(1000, 3) * 6 - 3

    _xyz = to_world(to_screen(to_world(to_screen(xyz), xyz[:,2:] - z)), xyz[:,2:] - z)

    print(th.mean((xyz - _xyz)**2).item())
