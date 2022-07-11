import torch.nn as nn
import torch as th
import numpy as np
import nn as nn_modules
from utils.utils import Gaus2D, SharedObjectsToBatch, BatchToSharedObjects, LambdaModule, PrintShape
from nn.residual import LinearResidual
from nn.eprop_lstm import EpropLSTM
from nn.eprop_gate_l0rd import EpropGateL0rd
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
        heads = 1,
        alpha = 1e-16,
        dropout = 0.5
    ):
        super(AlphaAttention, self).__init__()

        self.alpha     = nn.Parameter(th.zeros(1)+alpha)
        self.attention = nn.MultiheadAttention(
            num_hidden, 
            heads, 
            dropout = dropout, 
            batch_first = True
        )

    def forward(self, x: th.Tensor):
        return x + self.alpha * self.attention(x, x, x, need_weights=False)[0]

class Conv1dAlphaResidual(nn.Module):
    def __init__(self, num_hidden: int, initial_relu = True, alpha = 1e-16):
        super(Conv1dAlphaResidual, self).__init__()
        
        if initial_relu:
            self.layers = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
            )
        self.alpha = nn.Parameter(th.zeros(1)+alpha)

    def forward(self, input: th.Tensor):
        return input + self.alpha * self.layers(input)


class AlternatingTimeObjectAttention(nn.Module):
    def __init__(
        self,
        num_objects,
        num_timesteps,
        num_hidden,
        heads = 1,
        dropout = 0.0
    ):
        super(AlternatingTimeObjectAttention, self).__init__()
        self.attention = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t) o c')),
            AlphaAttention(num_hidden, heads),
            LambdaModule(lambda x: rearrange(x, '(b t) o c -> (b t o) c ', t = num_timesteps)),
            LinearAlphaResidual(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> (b o) t c', t = num_timesteps, o = num_objects)),
            AlphaAttention(num_hidden, heads),
            LambdaModule(lambda x: rearrange(x, '(b o) t c -> (b t o) c', o = num_objects)),
            LinearAlphaResidual(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects)),
        )

    def forward(self, input: th.Tensor):
        return self.attention(input)

class ObjectVisible(nn.Module):
    def __init__(
        self, 
        net_width,
        num_objects,
        num_timesteps
    ):
        super(ObjectVisible, self).__init__()

    def forward(self, input: th.Tensor):
        mask = input[:,:,:,-1:]
        return th.ceil((mask - 0.25))

class GridClassifier(nn.Module):
    def __init__(self):
        super(GridClassifier, self).__init__()

        x = (th.arange(36) % 6) - 3 + 0.5
        y = th.floor(th.arange(36) / 6) - 3 + 0.5
        self.register_buffer("grid_centers", th.stack((x,y), dim=1).unsqueeze(dim=0), persistent=False)

    def forward(self, position: th.Tensor):
        grid_distances = th.exp(-th.sum((self.grid_centers - position[:,:2].unsqueeze(dim=1))**2, dim=2))
        return th.softmax(grid_distances, dim=1)

class ScreenTo3D(nn.Module):
    def __init__(self, camera_view_matrix, zero_elevation):
        super(ScreenTo3D, self).__init__()

        self.cam = nn.Parameter(th.tensor(camera_view_matrix))
        self.z   = nn.Parameter(th.tensor(zero_elevation))

    def forward(self, xy: th.Tensor):
        a,b,c,d = th.unbind(self.cam[0])
        e,f,g,h = th.unbind(self.cam[1])
        m,n,o,p = th.unbind(self.cam[3])

        X = xy[:,:,1]
        Y = xy[:,:,0]

        z = self.z.expand(X.shape) 

        x = (b*h - d*f + Y*b*p - Y*d*n + X*f*p - X*h*n + b*g*z - c*f*z + Y*b*o*z - Y*c*n*z + X*f*o*z - X*g*n*z)/(a*f - b*e + Y*a*n - Y*b*m + X*e*n - X*f*m)
        y = -(a*h - d*e + Y*a*p - Y*d*m + X*e*p - X*h*m + a*g*z - c*e*z + Y*a*o*z - Y*c*m*z + X*e*o*z - X*g*m*z)/(a*f - b*e + Y*a*n - Y*b*m + X*e*n - X*f*m)

        return th.stack((x,y,z), dim=2)

class ObjectSelector(nn.Module):
    """ should select the object from which the position is taken"""
    def __init__(
        self, 
        gestalt_size,
        net_width,
        num_objects,
        num_timesteps
    ):
        super(ObjectSelector, self).__init__()

        self.to_batch = LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c'))

        self.object_visible = ObjectVisible(
            net_width      = net_width,
            num_objects    = num_objects,
            num_timesteps  = num_timesteps
        )

        num_inputs = gestalt_size + 2
        num_hidden = num_inputs * net_width 
        identity_channels = 13

        self.object_identity = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            LinearResidual(num_inputs, num_hidden, input_relu = False),
            LinearResidual(num_hidden, identity_channels),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )

        num_inputs = 3 + identity_channels
        num_hidden = num_inputs * net_width 

        self.object_selection_visible = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t) o c')),
            AlphaAttention(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b t) o c -> (b o) c t', t = num_timesteps)),
            Conv1dAlphaResidual(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b o) c t -> (b t) o c', o = num_objects)),
            AlphaAttention(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b t) o c -> (b o) t c', t = num_timesteps)),
            AlphaAttention(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b o) t c -> (b t o) c', o = num_objects)),
            LinearResidual(num_hidden, 1, alpha_residual=True),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects)),
            nn.Softmax(dim=2),
            LambdaModule(lambda x: x + x * (1 - x) * th.randn_like(x)),
        )

        self.object_selection_hidden = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t) o c')),
            AlphaAttention(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b t) o c -> (b o) c t', t = num_timesteps)),
            Conv1dAlphaResidual(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b o) c t -> (b t) o c', o = num_objects)),
            AlphaAttention(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b t) o c -> (b o) t c', t = num_timesteps)),
            AlphaAttention(num_hidden),
            LambdaModule(lambda x: rearrange(x, '(b o) t c -> (b t o) c', o = num_objects)),
            LinearResidual(num_hidden, 1, alpha_residual=True),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects)),
            nn.Softmax(dim=2),
            LambdaModule(lambda x: x + x * (1 - x) * th.randn_like(x)),
        )

    def forward(self, input: th.Tensor):
        object_visible   = self.object_visible(input)
        object_identity  = self.object_identity(input[:,:,:,2:-1])

        position = input[:,:,:,:2]
        mask     = input[:,:,:,-1:]
        
        selector_input = th.cat((position, mask, object_identity), dim=3)
        
        return self.object_selection_visible(selector_input) * object_visible, self.object_selection_hidden(selector_input) * object_visible
        

class CaterLocalizer(nn.Module):
    def __init__(
        self, 
        gestalt_size,
        net_width,
        num_objects,
        num_timesteps,
        camera_view_matrix,
        zero_elevation
    ):
        super(CaterLocalizer, self).__init__()

        self.to_batch = LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c'))

        self.gate = ObjectSelector(
            gestalt_size  = gestalt_size,
            net_width     = net_width,
            num_objects   = num_objects,
            num_timesteps = num_timesteps
        )

        num_inputs = gestalt_size + 5
        num_hidden = num_inputs * net_width 

        self.correction_visible = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 2),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )
        self.correction_hidden = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 2),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )
        self.to_3d = ScreenTo3D(camera_view_matrix, zero_elevation)

        self.labels_visible = GridClassifier()
        self.labels_hidden  = GridClassifier()

    def forward(self, input: th.Tensor):

        position = input[:,:,:,:2]

        position_visible = position + self.correction_visible(input)
        position_hidden  = position + self.correction_hidden(input)

        gate_visible, gate_hidden = self.gate(input)

        visible_3d = self.to_3d(th.sum(position_visible * gate_visible, dim=2)) 
        hidden_3d  = self.to_3d(th.sum(position_hidden * gate_hidden, dim=2))

        return (
            visible_3d,
            hidden_3d,
            self.labels_visible(visible_3d[:,-1]),
            self.labels_hidden(hidden_3d[:,-1]),
        )

class CaterObjectBehavior(nn.Module):
    def __init__(
        self, 
        gestalt_size,
        net_width,
        num_objects,
        num_timesteps,
    ):
        super(CaterObjectBehavior, self).__init__()

        self.to_batch = LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c'))

        num_inputs = gestalt_size + 2
        num_hidden = num_inputs * net_width 

        self.object_identitiy = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 5),
            nn.Softmax(dim=1),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )

        self.object_material = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 2),
            nn.Softmax(dim=1),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )

        self.object_size = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 3),
            nn.Softmax(dim=1),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )

        self.object_color = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b t o) c')),
            nn.Linear(num_inputs, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, 8),
            nn.Softmax(dim=1),
            LambdaModule(lambda x: rearrange(x, '(b t o) c -> b t o c', t = num_timesteps, o = num_objects))
        )


        num_inputs = 2
        num_hidden = num_inputs * net_width * 2

        self.detect_movement = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b o) c t')),
            nn.Conv1d(num_inputs, num_hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_hidden, num_hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_hidden, 1, 5, padding=2),
            nn.Sigmoid(),
            LambdaModule(lambda x: rearrange(x, '(b o) c t -> b t o c', o = num_objects)),
        )


        num_inputs = gestalt_size + 2
        num_hidden = num_inputs * net_width 

        self.detect_rotation = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b o) c t')),
            nn.Conv1d(num_inputs, num_hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_hidden, num_hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_hidden, 1, 5, padding=2),
            nn.Sigmoid(),
            LambdaModule(lambda x: rearrange(x, '(b o) c t -> b t o c', o = num_objects))
        )


        num_inputs = 2
        num_hidden = num_inputs * net_width * 2

        self.detect_pick_place = nn.Sequential(
            LambdaModule(lambda x: rearrange(x, 'b t o c -> (b o) c t')),
            nn.Conv1d(num_inputs, num_hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(num_hidden, num_hidden, 5, padding=10, dilation=5),
            nn.Tanh(),
            nn.Conv1d(num_hidden, 1, 5, padding=2),
            nn.Sigmoid(),
            LambdaModule(lambda x: rearrange(x, '(b o) c t -> b t o c', t = num_timesteps, o = num_objects))
        )


    def forward(self, input: th.Tensor):
        
        visible  = input[:,:,:,-1:]
        hidden   = (1 - visible)

        # mask object slots that never contain an object
        input = input * th.cummax(visible, dim=1)[0]

        position = input[:,:,:, :2]
        gestalt  = input[:,:,:,2:-1]

        identity   = self.object_identitiy(gestalt)
        material   = self.object_material(gestalt)
        size       = self.object_size(gestalt)
        color      = self.object_color(gestalt)
        movement   = self.detect_movement(position)
        rotation   = self.detect_rotation(gestalt)
        pick_place = self.detect_pick_place(position)

        sphere, snitch, cylinder, cube, cone = th.chunk(identity, 5, dim=3)
        rubber, metal = th.chunk(material, 2, dim=3)
        small, medium, large = th.chunk(size, 3, dim=3)
        red, purple, yellow, brown, gray, blue, cyan, green = th.chunk(color, 8, dim=3)

        # moving without pick place => sliding 
        # sliding ins the only move posible when contained
        slide = (1 - pick_place) * movement

        # only moving objects can perform pick and place
        pick_place = pick_place * movement

        # sphere and cones can not rotate
        # only stationary objects can rotate
        rotation = rotation * (1 - sphere) * (1 - cone) * (1 - movement)

        # construct the different action classes
        sphere_slide        = sphere * slide
        sphere_pick_place   = sphere * pick_place
        snitch_slide        = snitch * slide
        snitch_pick_place   = snitch * pick_place
        snitch_rotate       = snitch * rotation
        cylinder_slide      = cylinder * slide      
        cylinder_pick_place = cylinder * pick_place
        cylinder_rotate     = cylinder * rotation   
        cube_slide          = cube * slide      
        cube_pick_place     = cube * pick_place
        cube_rotate         = cube * rotation   
        cone_slide          = cone * slide      
        cone_pick_place     = cone * pick_place
        
        object_actions = th.cat((
            sphere_slide,       
            sphere_pick_place,  
            snitch_slide,       
            snitch_pick_place,  
            snitch_rotate,      
            cylinder_slide,     
            cylinder_pick_place,
            cylinder_rotate,    
            cube_slide,         
            cube_pick_place,    
            cube_rotate,        
            cone_slide,         
            cone_pick_place
        ), dim = 3)

        # construct different material classes
        sphere_rubber    = sphere * rubber
        sphere_metal     = sphere * metal
        cylinder_rubber  = cylinder * rubber
        cylinder_metal   = cylinder * metal
        cube_rubber      = cube * rubber
        cube_metal       = cube * metal
        cone_rubber      = cone * rubber
        cone_metal       = cone * metal

        object_materials = th.cat((
            sphere_rubber,
            sphere_metal,
            cylinder_rubber,
            cylinder_metal,
            cube_rubber,
            cube_metal,
            cone_rubber,
            cone_metal
        ), dim=3)

        # construct different size classes
        sphere_small    = sphere * small
        sphere_medium   = sphere * medium
        sphere_large    = sphere * large
        cylinder_small  = cylinder * small
        cylinder_medium = cylinder * medium
        cylinder_large  = cylinder * large
        cube_small      = cube * small
        cube_medium     = cube * medium
        cube_large      = cube * large
        cone_small      = cone * small
        cone_medium     = cone * medium
        cone_large      = cone * large

        object_sizes = th.cat((
            sphere_small,
            sphere_medium,
            sphere_large,
            cylinder_small,
            cylinder_medium,
            cylinder_large,
            cube_small,
            cube_medium,
            cube_large,
            cone_small,
            cone_medium,
            cone_large
        ), dim=3)

        # construct different color classes
        sphere_red      = sphere * red
        sphere_purple   = sphere * purple
        sphere_yellow   = sphere * yellow
        sphere_brown    = sphere * brown
        sphere_gray     = sphere * gray
        sphere_blue     = sphere * blue
        sphere_cyan     = sphere * cyan
        sphere_green    = sphere * green
        cylinder_red    = cylinder * red
        cylinder_purple = cylinder * purple
        cylinder_yellow = cylinder * yellow
        cylinder_brown  = cylinder * brown
        cylinder_gray   = cylinder * gray
        cylinder_blue   = cylinder * blue
        cylinder_cyan   = cylinder * cyan
        cylinder_green  = cylinder * green
        cube_red        = cube * red
        cube_purple     = cube * purple
        cube_yellow     = cube * yellow
        cube_brown      = cube * brown
        cube_gray       = cube * gray
        cube_blue       = cube * blue
        cube_cyan       = cube * cyan
        cube_green      = cube * green
        cone_red        = cone * red
        cone_purple     = cone * purple
        cone_yellow     = cone * yellow
        cone_brown      = cone * brown
        cone_gray       = cone * gray
        cone_blue       = cone * blue
        cone_cyan       = cone * cyan
        cone_green      = cone * green

        object_colors = th.cat((
            sphere_red,
            sphere_purple,
            sphere_yellow,
            sphere_brown,
            sphere_gray,
            sphere_blue,
            sphere_cyan,
            sphere_green,
            cylinder_red,
            cylinder_purple,
            cylinder_yellow,
            cylinder_brown,
            cylinder_gray,
            cylinder_blue,
            cylinder_cyan,
            cylinder_green,
            cube_red,
            cube_purple,
            cube_yellow,
            cube_brown,
            cube_gray,
            cube_blue,
            cube_cyan,
            cube_green,
            cone_red,
            cone_purple,
            cone_yellow,
            cone_brown,
            cone_gray,
            cone_blue,
            cone_cyan,
            cone_green
        ), dim=3)

        actions_visible   = th.max(object_actions * visible, dim=2)[0]
        actions_hidden    = th.max(object_actions * hidden,  dim=2)[0]
        materials_visible = th.max(object_materials * visible,  dim=2)[0]
        materials_hidden  = th.max(object_materials * hidden,  dim=2)[0]
        sizes_visible     = th.max(object_sizes * visible,  dim=2)[0]
        sizes_hidden      = th.max(object_sizes * hidden,  dim=2)[0]
        colors_visible    = th.max(object_colors * visible,  dim=2)[0]
        colors_hidden     = th.max(object_colors * hidden,  dim=2)[0]

        output = th.cat((
            actions_visible,
            actions_hidden,
            materials_visible,
            materials_hidden,
            sizes_visible,
            sizes_hidden,
            colors_visible,
            colors_hidden
        ), dim=2)

        properties = None
        with th.no_grad():
            properties = {
                "all_visible": th.cat((
                    actions_visible,
                    materials_visible,
                    sizes_visible,
                    colors_visible,
                ), dim=2),
                "all_hidden": th.cat((
                    actions_hidden,
                    materials_hidden,
                    sizes_hidden,
                    colors_hidden
                ), dim=2),
                "sphere_visible": th.cat((
                    actions_visible[:,:,:2],
                    materials_visible[:,:,:2],
                    sizes_visible[:,:,:3],
                    colors_visible[:,:,:8],
                ), dim=2),
                "sphere_hidden": th.cat((
                    actions_hidden[:,:,:2],
                    materials_hidden[:,:,:2],
                    sizes_hidden[:,:,:3],
                    colors_hidden[:,:,:8],
                ), dim=2),
                "snitch_visible": actions_visible[:,:,2:5],
                "snitch_hidden": actions_hidden[:,:,2:5],
                "cylinder_visible": th.cat((
                    actions_visible[:,:,2:4],
                    materials_visible[:,:,2:4],
                    sizes_visible[:,:,3:6],
                    colors_visible[:,:,8:16],
                ), dim=2),
                "cylinder_hidden": th.cat((
                    actions_hidden[:,:,2:4],
                    materials_hidden[:,:,2:4],
                    sizes_hidden[:,:,3:6],
                    colors_hidden[:,:,8:16],
                ), dim=2),
                "cube_visible": th.cat((
                    actions_visible[:,:,4:6],
                    materials_visible[:,:,4:6],
                    sizes_visible[:,:,6:9],
                    colors_visible[:,:,16:24],
                ), dim=2),
                "cone_visible": th.cat((
                    actions_visible[:,:,6:8],
                    materials_visible[:,:,6:8],
                    sizes_visible[:,:,9:12],
                    colors_visible[:,:,24:32],
                ), dim=2),
                "cone_hidden": th.cat((
                    actions_hidden[:,:,6:8],
                    materials_hidden[:,:,6:8],
                    sizes_hidden[:,:,9:12],
                    colors_hidden[:,:,24:32],
                ), dim=2),
                "small_visible": th.stack((
                    sizes_visible[:,:,0],
                    sizes_visible[:,:,3],
                    sizes_visible[:,:,6],
                    sizes_visible[:,:,9]
                ), dim=2),
                "small_hidden": th.stack((
                    sizes_hidden[:,:,0],
                    sizes_hidden[:,:,3],
                    sizes_hidden[:,:,6],
                    sizes_hidden[:,:,9]
                ), dim=2),
                "medium_visible": th.stack((
                    sizes_visible[:,:,1],
                    sizes_visible[:,:,4],
                    sizes_visible[:,:,7],
                    sizes_visible[:,:,10]
                ), dim=2),
                "medium_hidden": th.stack((
                    sizes_hidden[:,:,1],
                    sizes_hidden[:,:,4],
                    sizes_hidden[:,:,7],
                    sizes_hidden[:,:,10]
                ), dim=2),
                "large_visible": th.stack((
                    sizes_visible[:,:,2],
                    sizes_visible[:,:,5],
                    sizes_visible[:,:,8],
                    sizes_visible[:,:,11]
                ), dim=2),
                "slide_visible": th.stack((
                    actions_visible[:,:,0],
                    actions_visible[:,:,2],
                    actions_visible[:,:,5],
                    actions_visible[:,:,8],
                    actions_visible[:,:,11]
                ), dim=2),
                "slide_hidden": th.stack((
                    actions_hidden[:,:,0],
                    actions_hidden[:,:,2],
                    actions_hidden[:,:,5],
                    actions_hidden[:,:,8],
                    actions_hidden[:,:,11]
                ), dim=2),
                "pick_place": th.stack((
                    actions_visible[:,:,1],
                    actions_visible[:,:,3],
                    actions_visible[:,:,6],
                    actions_visible[:,:,9],
                    actions_visible[:,:,12]
                ), dim=2),
                "rotate": th.stack((
                    actions_visible[:,:,4],
                    actions_visible[:,:,7],
                    actions_visible[:,:,10],
                ), dim=2),
                "materials_visible": materials_visible,
                "materials_hidden": materials_hidden,
                "colors_visible": colors_visible,
                "colors_hidden": colors_hidden,
            }

        return output, properties
