import torch as th
import torch.nn as nn
import numpy as np
from typing import Tuple
from einops import rearrange, repeat, reduce
from utils.optimizers import SDAMSGrad
from utils.scheduled_sampling import ScheduledSampler
from nn.decoder import GPDecoder
from nn.encoder import GPEncoder
from nn.predictor import LatentEpropPredictor
from utils.utils import PrintGradient, InitialLatentStates
from utils.loss import MaskModulatedObjectLoss, ObjectModulator, TranslationInvariantObjectLoss, PositionLoss
from nn.background import BackgroundEnhancer

class Loci(nn.Module):
    def __init__(
        self,
        cfg,
        camera_view_matrix = None,
        zero_elevation = None,
        sampler: ScheduledSampler = None,
        closed_loop=False,
        teacher_forcing=1
    ):
        super(Loci, self).__init__()

        self.closed_loop = closed_loop
        self.teacher_forcing = teacher_forcing
        self.cfg = cfg

        self.sampler = sampler

        self.encoder = GPEncoder(
            input_size       = cfg.input_size,
            latent_size      = cfg.latent_size,
            num_objects      = cfg.num_objects,
            img_channels     = cfg.img_channels * 2 + 6,
            hidden_channels  = cfg.encoder.channels,
            level1_channels  = cfg.encoder.level1_channels,
            num_layers       = cfg.encoder.num_layers,
            gestalt_size     = cfg.gestalt_size,
            batch_size       = cfg.batch_size,
            reg_lambda       = cfg.encoder.reg_lambda,
        )

        self.predictor = LatentEpropPredictor(
            num_objects        = cfg.num_objects,
            gestalt_size       = cfg.gestalt_size,
            heads              = cfg.predictor.heads,
            layers             = cfg.predictor.layers,
            reg_lambda         = cfg.predictor.reg_lambda,
            vae_factor         = cfg.vae_factor,
            batch_size         = cfg.batch_size,
            camera_view_matrix = camera_view_matrix,
            zero_elevation     = zero_elevation
        )

        self.decoder = GPDecoder(
            latent_size      = cfg.latent_size,
            num_objects      = cfg.num_objects,
            gestalt_size     = cfg.gestalt_size,
            img_channels     = cfg.img_channels,
            hidden_channels  = cfg.decoder.channels,
            level1_channels  = cfg.decoder.level1_channels,
            num_layers       = cfg.decoder.num_layers
        )

        self.background = BackgroundEnhancer(
            input_size        = cfg.input_size,
            gestalt_size      = cfg.gestalt_size,
            img_channels      = cfg.img_channels,
            deepth            = cfg.background.num_layers, 
            latent_channels   = cfg.background.latent_channels,
            level1_channels   = cfg.background.level1_channels,
            reg_lambda        = cfg.background.reg_lambda,
            vae_factor         = cfg.vae_factor,
            batch_size        = cfg.batch_size,
        )

        self.initial_states = InitialLatentStates(
            gestalt_size               = cfg.gestalt_size,
            num_objects                = cfg.num_objects,
            size                       = cfg.input_size,
            object_permanence_strength = cfg.object_permanence_strength
        )

        self.background_layer_offset = int(np.log2(cfg.latent_size[0] // cfg.patch_grid_size[0])) - 1

        self.translation_invariant_object_loss = TranslationInvariantObjectLoss(cfg.num_objects, teacher_forcing)
        self.mask_modulated_object_loss        = MaskModulatedObjectLoss(cfg.num_objects, teacher_forcing)
        self.position_loss                     = PositionLoss(cfg.num_objects, teacher_forcing)
        self.modulator                         = ObjectModulator(cfg.num_objects)
        self.print_gradient                    = PrintGradient("GPNet")

        self.background.set_level(cfg.level)
        self.encoder.set_level(cfg.level)
        self.decoder.set_level(cfg.level)
        self.initial_states.set_level(cfg.level)

    def get_init_status(self):
        init = []
        for module in self.modules():
            if callable(getattr(module, "get_init", None)): 
                init.append(module.get_init())

        assert len(set(init)) == 1
        return init[0]

    def inc_init_level(self):
        for module in self.modules():
            if callable(getattr(module, "step_init", None)):
                module.step_init()

    def get_openings(self):
        return self.predictor.get_openings()

    def detach(self):
        for module in self.modules():
            if module != self and callable(getattr(module, "detach", None)):
                module.detach()

    def reset_state(self):
        for module in self.modules():
            if module != self and callable(getattr(module, "reset_state", None)):
                module.reset_state()

    def forward(self, *input, reset=True, detach=True, mode='end2end', evaluate=False, train_background=False):

        if detach:
            self.detach()

        if reset:
            self.reset_state()

        if train_background or self.get_init_status() < 1:
            return self.background(*input)[1]

        return self.run_end2end(*input, evaluate=evaluate)

    def run_encoder(
        self, 
        input: th.Tensor, 
        error: th.Tensor,
        mask: th.Tensor = None,
        object: th.Tensor = None,
        position: th.Tensor = None,
        priority: th.Tensor = None
    ):
        return self.encoder(th.cat((input, error), dim=1), mask, object, position, priority)

    def run_decoder(
        self, 
        position: th.Tensor, 
        gestalt: th.Tensor,
        priority: th.Tensor,
        bg_mask: th.Tensor,
        background: th.Tensor,
        latent_bg: th.Tensor
    ):
        mask, object = self.decoder(latent_bg, position, gestalt, priority)

        mask   = th.softmax(th.cat((mask, bg_mask), dim=1), dim=1) 
        object = th.cat((th.sigmoid(object - 2.5), background), dim=1)

        _mask   = mask.unsqueeze(dim=2)
        _object = object.view(
            mask.shape[0], 
            self.cfg.num_objects + 1,
            self.cfg.img_channels,
            *mask.shape[2:]
        )

        output  = th.sum(_mask * _object, dim=1)
        return output, mask, object

    def run_end2end(
        self, 
        input: th.Tensor,
        error: th.Tensor = None,
        mask: th.Tensor = None,
        position: th.Tensor = None,
        gestalt: th.Tensor = None,
        priority: th.Tensor = None,
        evaluate = False
    ):
        output_sequence     = list()
        position_sequence   = list()
        gestalt_sequence    = list()
        priority_sequence   = list()
        mask_sequence       = list()
        object_sequence     = list()
        background_sequence = list()
        error_sequence      = list()

        position_loss = th.tensor(0, device=input.device)
        object_loss   = th.tensor(0, device=input.device)
        time_loss     = th.tensor(0, device=input.device)
        latent_bg     = None
        bg_mask       = None

        if error is None or mask is None:
            bg_mask, background, latent_bg, _ = self.background(input)
            error    = th.sqrt(reduce((input - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()
        else:
            latent_bg = self.background.get_last_latent_gird()

        position, gestalt, priority = self.initial_states(error, mask, position, gestalt, priority)

        if mask is None:
            mask     = self.decoder(latent_bg, position, gestalt, priority)[0]
            mask     = th.softmax(th.cat((mask, bg_mask), dim=1), dim=1) 

        
        output      = None
        bg_mask     = None
        object_last = None

        position_last = position
        object_last   = self.decoder(latent_bg, position_last, gestalt)[-1]

        # background and cores ponding mask for the next time point
        bg_mask, background, latent_bg, raw_background = self.background(input, error, mask[:,-1:])

        # position and gestalt for the current time point
        position, gestalt, priority = self.run_encoder(input, error, mask, object_last, position, priority)

        # position and gestalt for the next time point
        position, gestalt, priority, snitch_position = self.predictor(position, gestalt, priority) 

        # combinded background and objects (masks) for next timepoint
        output, mask, object = self.run_decoder(position, gestalt, priority, bg_mask, background, latent_bg)

        if not evaluate:

            #regularize to small possition chananges over time
            position_loss = position_loss + self.position_loss(position, position_last.detach(), mask[:,:-1].detach())

            # regularize to encode last visible object
            object_cur       = self.decoder(latent_bg, position, gestalt)[-1]
            object_modulated = self.decoder(latent_bg, *self.modulator(position, gestalt, mask[:,:-1]))[-1]
            object_loss      = object_loss + self.mask_modulated_object_loss(
                object_cur, 
                object_modulated.detach(), 
                mask[:,:-1].detach()
            )

            # regularize to prduce consistent object codes over time
            time_loss = time_loss + 0.1 * self.translation_invariant_object_loss(
                mask[:,:-1].detach(),
                object_last.detach(), 
                position_last.detach(),
                object_cur,
                position.detach(),
            )

        if evaluate:
            object = self.run_decoder(position, gestalt, None, bg_mask, background, latent_bg)[-1]

        return (
            output, 
            position, 
            gestalt, 
            priority, 
            mask, 
            object, 
            raw_background, 
            position_loss,
            object_loss,
            time_loss,
            snitch_position
        )
