import argparse
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import torch as th
import torch.nn as nn
import cv2
import pickle
from utils.utils import Gaus2D
from einops import rearrange, repeat, reduce
from utils.configuration import Configuration
from nn.background import BackgroundEnhancer
from nn.decoder import GPDecoder

class LociPlayground:
    
    def __init__(self, cfg, device, file, background, gestalt = None, position = None):

        device = th.device(device)

        self.cfg      = cfg
        self.gestalt  = th.zeros((1, cfg.gestalt_size)).to(device)
        self.position = th.tensor([[0,0,0.05]]).to(device)
        self.size     = cfg.input_size
        self.gaus2d   = Gaus2D(cfg.input_size).to(device)
        self.gestalt_gridcell_size   = 25
        self.gestalt_gridcell_margin = 5
        self.gestalt_grid_width      = 16
        self.gestalt_grid_height     = 6

        if gestalt is not None:
            self.gestalt = gestalt.to(device)

        if position is not None:
            self.position = position.to(device)

        self.decoder = GPDecoder(
            latent_size      = cfg.latent_size,
            num_objects      = 1,
            gestalt_size     = cfg.gestalt_size,
            img_channels     = cfg.img_channels,
            hidden_channels  = cfg.decoder.channels,
            level1_channels  = cfg.decoder.level1_channels,
            num_layers       = cfg.decoder.num_layers,
        ).to(device)
        self.decoder.set_level(2)

        print(f'loading to device {self.decoder.mask_alpha.device}...', end='', flush=True)
        state = th.load(file, map_location=device)

        # backward compatibility
        model = {}
        for key, value in state["model"].items():
            model[key.replace(".module.", ".")] = value

        decoder_state = {}
        for k, v in model.items():
            if k.startswith('decoder.'):
                decoder_state[k.replace('decoder.', '')] = v

        self.decoder.load_state_dict(decoder_state)

        self.bg_mask    = model['background.mask']
        self.background = th.from_numpy(cv2.imread(background)).to(device) / 255
        self.background = rearrange(self.background, 'h w c -> 1 c h w')
        print('done', flush=True)

        self.fig = plt.figure(figsize=(6,6))

        self.ax_gestalt  = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        self.ax_position = plt.subplot2grid((3, 3), (0, 2))
        self.ax_output1  = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        self.ax_output2  = plt.subplot2grid((3, 3), (1, 2))
        self.ax_output3  = plt.subplot2grid((3, 3), (2, 2))

        self.outputs = [self.ax_output1, self.ax_output2, self.ax_output3]
        self.indices = [0, 1, 2]

        self.connections = ()


        self.add_image(self.ax_gestalt, self.create_gestalt_image())
        self.add_image(self.ax_position, self.create_position_image())

        self.update_outputs()

        plt.tight_layout()

    def update_outputs(self):
        mask, object, output = None, None, None
        with th.no_grad():
            mask, object        = self.decoder(self.position, self.gestalt)
            bg_mask, background = self.bg_mask, self.background

            # we have to somehow correct for not having 30 objects
            mask   = th.softmax(th.cat((mask*100, bg_mask), dim=1) * 0.01, dim=1)
            object = th.cat((th.sigmoid(object - 2.5), background), dim=1)

            mask   = rearrange(mask, 'b n h w -> b n 1 h w')
            object = rearrange(object, 'b (n c) h w -> b n c h w', n = 2)

            output = th.sum(mask * object, dim = 1)
            output = rearrange(output[0], 'c h w -> h w c').cpu().numpy()

            object = rearrange(object[0,0], 'c h w -> h w c').cpu().numpy()
            mask   = rearrange(mask[0,0],   'c h w -> h w c')
            mask   = th.cat((mask, mask, mask * 0.6 + 0.4), dim=2).cpu().numpy()

        self.add_image(self.outputs[self.indices[0]], output[:,:,::-1])
        self.add_image(self.outputs[self.indices[1]], object[:,:,::-1])
        self.add_image(self.outputs[self.indices[2]], mask)

    def __enter__(self):
        self.connections = (
            self.fig.canvas.mpl_connect('button_press_event', self.onclick),
            self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        )
        return self

    def __exit__(self, *args, **kwargs):
        for connection in self.connections:
            self.fig.canvas.mpl_disconnect(connection)

    def create_gestalt_image(self):

        gestalt = self.gestalt[0].cpu().numpy()
        size    = self.gestalt_gridcell_size
        margin  = self.gestalt_gridcell_margin
            
        width  = self.gestalt_grid_width  * (margin + size) + margin
        height = self.gestalt_grid_height * (margin + size) + margin
        img = np.zeros((height, width, 3)) + 0.3
        
        for i in range(gestalt.shape[0]):
            h = i // self.gestalt_grid_width
            w = i  % self.gestalt_grid_width
            
            img[h*size+(h+1)*margin:(h+1)*(margin+size),w*size+(w+1)*margin:(w+1)*(margin+size),0] = (1 - gestalt[i]) * 0.8 + gestalt[i] * 0.2
            img[h*size+(h+1)*margin:(h+1)*(margin+size),w*size+(w+1)*margin:(w+1)*(margin+size),1] = gestalt[i] * 0.8 + (1 - gestalt[i]) * 0.2
            img[h*size+(h+1)*margin:(h+1)*(margin+size),w*size+(w+1)*margin:(w+1)*(margin+size),2] = 0.2


        return img

    def create_position_image(self):
        
        img = self.gaus2d(self.position)
        img = rearrange(img[0], 'c h w -> h w c')

        return th.cat((img, img, img * 0.6 + 0.4), dim=2).cpu().numpy()


    def add_image(self, ax, img):
        ax.clear()
        ax.imshow(img)
        ax.axis('off')



    def onclick(self, event):
        x, y = event.xdata, event.ydata

        if self.ax_gestalt == event.inaxes:

            size    = self.gestalt_gridcell_size
            margin  = self.gestalt_gridcell_margin

            w = int(x / (margin + size))
            h = int(y / (margin + size))

            i = h * self.gestalt_grid_width + w
            self.gestalt[0,i] = 1 - self.gestalt[0,i]

            self.add_image(self.ax_gestalt, self.create_gestalt_image())
            self.update_outputs()
            self.fig.canvas.draw()

        if self.ax_position == event.inaxes:

            x = (x / self.size[1]) * 2 - 1
            y = (y / self.size[0]) * 2 - 1

            self.position[0,0] = y
            self.position[0,1] = x

            self.add_image(self.ax_position, self.create_position_image())
            self.update_outputs()
            self.fig.canvas.draw()

        if self.ax_output2 == event.inaxes:
            ax_tmp = self.indices[0]
            self.indices[0] = self.indices[1]
            self.indices[1] = ax_tmp
            self.update_outputs()
            self.fig.canvas.draw()

        if self.ax_output3 == event.inaxes:
            ax_tmp = self.indices[0]
            self.indices[0] = self.indices[2]
            self.indices[2] = ax_tmp
            self.update_outputs()
            self.fig.canvas.draw()
            
    def onscroll(self, event):
        if self.ax_position == event.inaxes:
            std = max(self.position[0,2], 0.0)
            if event.button == 'down':
                self.position[0,2] = max(std - std * (1 - std) * 0.1, 0.0)
                
            elif event.button == 'up':
                self.position[0,2] = std + max(std * (1 - std) * 0.1, 0.001)

            self.add_image(self.ax_position, self.create_position_image())
            self.update_outputs()
            self.fig.canvas.draw()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", required=True, type=str)
    parser.add_argument("-load", required=True, type=str)
    parser.add_argument("-background", required=True, type=str)
    parser.add_argument("-latent", default="", type=str)
    parser.add_argument("-device", default=0, type=int)

    args = parser.parse_args(sys.argv[1:])
    cfg  = Configuration(args.cfg)

    gestalt  = None
    position = None
    if args.latent != "":
        with open(args.latent, 'rb') as infile:
            state = pickle.load(infile)
            gestalt  = state["gestalt"]
            position = state["position"]

    with LociPlayground(cfg.model, args.device, args.load, args.background, gestalt, position):
        plt.show()
