import torch as th
from torch.utils.data import Dataset, DataLoader
from torch import nn

import h5py
import os
from utils.configuration import Configuration
from utils.io import model_path
from model.loci import Loci
from utils.utils import LambdaModule, Gaus2D, Prioritize
from utils.loss import SSIMLoss, MSSSIMLoss
from utils.optimizers import SDAdam, SDAMSGrad
from utils.io import Timer
import numpy as np
import cv2
from pathlib import Path
import shutil
import pickle
from einops import rearrange, repeat, reduce
from model.scripts.training import eval_net

def preprocess(tensor, scale=1, normalize=False, mean_std_normalize=False):

    if normalize:
        min_ = th.min(tensor)
        max_ = th.max(tensor)
        tensor = (tensor - min_) / (max_ - min_)

    if mean_std_normalize:
        mean = th.mean(tensor)
        std = th.std(tensor)
        tensor = th.clip((tensor - mean) / (2 * std), -1, 1) * 0.5 + 0.5

    if scale > 1:
        upsample = nn.Upsample(scale_factor=scale).to(tensor[0].device)
        tensor = upsample(tensor)

    return tensor

def color_mask(mask):

    colors = th.tensor([
	[ 255,   0,   0 ],
	[   0,   0, 255 ],
	[ 255, 255,   0 ],
	[ 255,   0, 255 ],
	[   0, 255, 255 ],
	[   0, 255,   0 ],
	[ 255, 128,   0 ],
	[ 128, 255,   0 ],
	[ 128,   0, 255 ],
	[ 255,   0, 128 ],
	[   0, 255, 128 ],
	[   0, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 255 ],
	[ 128, 255, 255 ],
	[ 128, 255, 255 ],
	[ 255, 255, 128 ],
	[ 255, 255, 128 ],
	[ 255, 128, 255 ],
	[ 128,   0,   0 ],
	[   0,   0, 128 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
    ], device = mask.device) / 255.0

    colors = colors.view(1, -1, 3, 1, 1)
    mask = mask.unsqueeze(dim=2)

    return th.sum(colors[:,:mask.shape[1]] * mask, dim=1)


def priority_to_img(priority, h, w):

    imgs = []

    for p in range(priority.shape[2]):

        img = np.zeros((h,w,3), np.uint8)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        text_position          = (h // 6, w //2)
        font_scale             = w / 256
        font_color             = (255,255,255)
        thickness              = 2
        lineType               = 2

        cv2.putText(img,f'{priority[0,0,p].item():.2e}',
            text_position,
            font,
            font_scale,
            font_color,
            thickness,
            lineType)

        imgs.append(rearrange(th.tensor(img, device=priority.device), 'h w c -> 1 1 c h w'))

    return imgs

def to_rgb_object(tensor, o):
    colors = th.tensor([
	[ 255,   0,   0 ],
	[   0,   0, 255 ],
	[ 255, 255,   0 ],
	[ 255,   0, 255 ],
	[   0, 255, 255 ],
	[   0, 255,   0 ],
	[ 255, 128,   0 ],
	[ 128, 255,   0 ],
	[ 128,   0, 255 ],
	[ 255,   0, 128 ],
	[   0, 255, 128 ],
	[   0, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 128 ],
	[ 128, 255, 128 ],
	[ 128, 128, 255 ],
	[ 255, 128, 255 ],
	[ 128, 255, 255 ],
	[ 128, 255, 255 ],
	[ 255, 255, 128 ],
	[ 255, 255, 128 ],
	[ 255, 128, 255 ],
	[ 128,   0,   0 ],
	[   0,   0, 128 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128, 128,   0 ],
	[ 128,   0, 128 ],
	[ 128,   0, 128 ],
	[   0, 128, 128 ],
	[   0, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
	[ 128, 128, 128 ],
    ], device = tensor.device) / 255.0

    colors = colors.view(48,3,1,1)
    return colors[o] * tensor

def to_rgb(tensor: th.Tensor):
    return th.cat((
        tensor * 0.6 + 0.4,
        tensor, 
        tensor
    ), dim=1)

def eval_mnist(cfg: Configuration, dataset: Dataset, file, active_layer, activity, init, size):

    #assert(cfg.sequence_len == 2)
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 10

    dataloader = DataLoader(
        dataset, 
        num_workers = 1, 
        pin_memory = False, 
        batch_size = cfg_net.batch_size,
        shuffle = False
    )

    # create model 
    net = Loci(
        cfg_net,
        teacher_forcing = 1000000
    )

    # load model
    if file != '':
        state = th.load(file, map_location=device)
        net.load_state_dict(state["model"])
        print(f"loaded: {file}")
    
    net = net.to(device=device)
    net.eval()
    
    mseloss = nn.MSELoss()

    mask_last         = None
    position_last     = None
    gestalt_last      = None
    priority_last     = None
    object_last       = None
    mask_cur          = None

    prediction_mse = 0
    prediction_ssim = 0
    counter = 0

    ssim = SSIMLoss()

    with th.no_grad():
        for i, input in enumerate(dataloader):

            tensor = input[0].float().to(device)

            input  = tensor[:,0]
            target = th.clip(tensor[:,0], 0, 1)
            error  = None

            mse_loss  = 0
            ssim_loss = 0

            for t in range(-cfg.teacher_forcing, _tensor.shape[1]-1):
                if t >= 10 and cfg.closed_loop:
                    input  = output_next
                    target = th.clip(tensor[:,t+1], 0, 1)
                    error  = th.zeros_like(net_error)
                elif t >= 0:
                    input  = tensor[:,t]
                    target = th.clip(tensor[:,t+1], 0, 1)

                (
                    output_next, 
                    position_next, 
                    gestalt_next, 
                    priority_next, 
                    mask_next, 
                    object_next, 
                    background_next, 
                    _,_,_,_
                ) = net(
                    input, 
                    error,
                    mask_last, 
                    position_last, 
                    gestalt_last,
                    priority_last,
                    reset = (t == -cfg.teacher_forcing),
                    evaluate=True
                )
                mask_last     = mask_next.clone()
                position_last = position_next.clone()
                gestalt_last  = gestalt_next.clone()
                priority_last = priority_next.clone()

                bg_error  = th.sqrt(reduce((target - background_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error     = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error     = th.sqrt(error) * bg_error

                error_mask = th.gt(patch_error, 0.1).float().detach()

                if t >= 9:
                    target = reduce(target, 'b c h w -> b 1 h w', 'sum')
                    output = reduce(output_next, 'b c h w -> b 1 h w', 'sum')
                    mse_loss  = mse_loss + 0.1 * th.mean(reduce((target - output)**2, 'b 1 h w -> b', 'sum')).item()
                    ssim_loss = ssim_loss + 0.1 * np.abs(ssim(target, output).item())

            prediction_mse += mse_loss
            prediction_ssim += ssim_loss
            counter += 1
            print(f'Evaluating [{i * cfg_net.batch_size / 100:.2f}%] MSE: {prediction_mse / counter}, SSIM: {prediction_ssim / counter}')


def save(cfg: Configuration, dataset: Dataset, file, active_layer, size, object_view, nice_view, individual_views):

    #assert(cfg.sequence_len == 2)
    cfg_net = cfg.model
    device = th.device(cfg.device)
    cfg_net.batch_size = 1

    dataloader = DataLoader(
        dataset, 
        num_workers = 1, 
        pin_memory = False, 
        batch_size = 1,
        shuffle = False
    )

    # create model 
    net = Loci(
        cfg_net,
        camera_view_matrix = dataset.cam if cfg.datatype == 'cater' else None,
        zero_elevation     = dataset.z if cfg.datatype == 'cater' else None,
        teacher_forcing    = 1000000
    )

    # load model
    if file != '':
        print(f"load {file} to device {device}")
        state = th.load(file, map_location=device)

        # backward compatibility
        model = {}
        for k, v in state["model"].items():
            k = k.replace(".module.", ".")
            k = k.replace("predictor.vae","predictor.bottleneck")
            model[k] = v

        net.load_state_dict(model)

    net = net.to(device=device)
    net.eval()
    
    gaus2d = Gaus2D(size).to(device)
    scale  = size[0] // (cfg_net.latent_size[0] * 2**active_layer)

    init = net.get_init_status()
    print(f"Init status: {init}")
    mseloss = nn.MSELoss()

    prioritize = Prioritize(cfg_net.num_objects).to(device)

    with th.no_grad():
        for i, input in enumerate(dataloader):

            tensor         = input[0].float().to(device)
            old_background = input[1].to(device)

            if cfg.datatype == 'cater' and i == 0:
                net.background.set_background(old_background[:,0])

            input  = tensor[:,0]
            target = th.clip(tensor[:,0], 0, 1)
            error  = None

            mse_loss       = 0
            ssim_loss      = 0
            mask_cur       = None
            mask_last      = None
            position_last  = None
            gestalt_last   = None
            priority_last  = None

            for t in range(-cfg.teacher_forcing, tensor.shape[1]-1):
                if t >= 10 and cfg.closed_loop:
                    input  = output_next
                    target = th.clip(tensor[:,t+1], 0, 1)
                    error  = th.zeros_like(net_error)
                elif t >= 0:
                    input  = tensor[:,t]
                    target = th.clip(tensor[:,t+1], 0, 1)
                elif t < 0:
                    input  = tensor[:,0]
                    target = th.clip(tensor[:,0], 0, 1)

                (
                    output_next, 
                    position_next, 
                    gestalt_next, 
                    priority_next, 
                    mask_next, 
                    object_next, 
                    background_next, 
                    _,_,_,_
                ) = net(
                    input, 
                    error,
                    mask_last, 
                    position_last, 
                    gestalt_last,
                    priority_last,
                    reset = (t == -cfg.teacher_forcing),
                    evaluate=True
                )
                mask_last     = mask_next.clone()
                position_last = position_next.clone()
                gestalt_last  = gestalt_next.clone()
                priority_last = priority_next.clone()

                if t == 0:
                    _gestalt = rearrange(gestalt_last, 'b (n c) -> b n c', n = cfg_net.num_objects)
                    _position    = rearrange(position_last,    'b (n c) -> b n c', n = cfg_net.num_objects)
                    for n in range(0, cfg_net.num_objects):
                        with open(f'latent-{i:04d}-{n:02d}.pickle', "wb") as outfile:
                            state = {
                                "gestalt":  th.round(th.clip(_gestalt[0:1,n], 0, 1)),
                                "position": _position[0:1,n],
                            }
                            pickle.dump(state, outfile)

                bg_error  = th.sqrt(reduce((target - background_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error     = th.sqrt(reduce((target - output_next)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error     = th.sqrt(error) * bg_error

                error_mask = th.gt(bg_error, 0.1).float().detach()
                error_next = error.clone()

                print(f'Saving[{(t+1+cfg.teacher_forcing)/3:.2f}%/{i+1}/{len(dataloader)+cfg.teacher_forcing}]: {(i*300+t+1) / (len(dataloader)*3):.3f}%, Loss: {mseloss(output_next, target).item()}')

                output = output_next.clone()

                output = th.clip(output, 0, 1) + background_next * (1 - error_mask) * (1 - init)
                target = target * error_mask * (1 - init) + target * init

                highlited_input = input
                if mask_cur is not None:
                    grayscale        = input[:,0:1] * 0.299 + input[:,1:2] * 0.587 + input[:,2:3] * 0.114
                    object_mask_cur  = th.sum(mask_cur[:,:-1], dim=1).unsqueeze(dim=1)
                    highlited_input  = grayscale * (1 - object_mask_cur) 
                    highlited_input += grayscale * object_mask_cur * 0.3333333 
                    highlited_input  = highlited_input + color_mask(mask_cur[:,:-1]) * 0.6666666

                mask_cur = mask_next.clone()

                input           = preprocess(input, scale)
                output          = preprocess(th.clip(output, 0, 1), scale)
                highlited_input = preprocess(highlited_input, scale)

                position_next2d = gaus2d(rearrange(position_next, 'b (o c) -> (b o) c', o=cfg_net.num_objects))
                position_next2d = rearrange(position_next2d, '(b o) c h w -> b (o c) h w', o=cfg_net.num_objects)
                position_next3d = prioritize(position_next2d, priority_next)
                position_next3d = to_rgb(th.max(position_next3d, dim=1, keepdim=True)[0])
                                 
                object_next      = preprocess(object_next, scale)
                mask_next        = preprocess(mask_next, scale)
                background_next  = preprocess(background_next, scale)
                error_next       = to_rgb(preprocess(error_next, scale))
                bg_error         = to_rgb(preprocess(bg_error, scale))

                object_next     = rearrange(object_next, 'b (o c) h w -> b o c h w', c = cfg_net.img_channels)
                mask_next       = rearrange(mask_next, 'b (o 1) h w -> b o 1 h w')

                if nice_view:
                    img = th.ones((3, size[0] * 2 + 10*3, size[1] * 2 +  10*3), device = object_next.device) * 0.2
                    img[:,10:size[0]+10, 10:size[1]+10]                                = input[0]
                    img[:,10+size[0]+10:(size[0]+10)*2, 10:size[1]+10]                 = output[0]
                    img[:,10:size[0]+10, 10+size[1]+10:(size[1]+10)*2]                 = highlited_input[0]
                    img[:,10+size[0]+10:(size[0]+10)*2, 10+size[1]+10:(size[1]+10)*2]  = position_next3d[0]
                    
                    img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
                    cv2.imwrite(f'loci-nice-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', img)


                if object_view:
                    height = size[0] * 4 + 18*3
                    width  = size[1] * 4 + 18*2 + size[1]*int(np.ceil(cfg_net.num_objects/2)) + 6*(int(np.ceil(cfg_net.num_objects/2))+1)
                    img = th.ones((3, height, width), device = object_next.device) * 0.2

                    img[:,18:size[0]*2+18, 18:size[1]*2+18]      = preprocess(input, 2)[0]
                    img[:,size[0]*2+18*2:-18, 18:size[1]*2+18]   = preprocess(highlited_input, 2)[0]
                    img[:,18:size[0]*2+18, -size[1]*2-18:-18]    = preprocess(output, 2)[0]
                    img[:,size[0]*2+18*2:-18, -size[1]*2-18:-18] = preprocess(background_next, 2)[0]

                    for o in range(int(np.ceil(cfg_net.num_objects / 2))):

                        img[:,18:18+size[0],18+size[1]*2+6+o*(6+size[1]):18+size[1]*2+(o+1)*(6+size[1])]               = preprocess(object_next[:,o*2], normalize=True)[0]
                        img[:,18+size[0]+6:18+size[0]*2+6,18+size[1]*2+6+o*(6+size[1]):18+size[1]*2+(o+1)*(6+size[1])] = to_rgb_object(mask_next[0,o*2], o*2)

                        if o * 2 + 1 < cfg_net.num_objects:
                            img[:,18+2*(size[0]+6):18+size[0]*3+12,18+size[1]*2+6+o*(6+size[1]):18+size[1]*2+(o+1)*(6+size[1])] = preprocess(object_next[:,o*2+1], normalize=True)[0]
                            img[:,18+3*(size[0]+6):18+size[0]*4+18,18+size[1]*2+6+o*(6+size[1]):18+size[1]*2+(o+1)*(6+size[1])] = to_rgb_object(mask_next[0,o*2+1], o*2+1)

                    img = rearrange(img * 255, 'c h w -> h w c').cpu().numpy()
                    cv2.imwrite(f'loci-objects-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', img)

                if individual_views:
                    cv2.imwrite(f'loci-input-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(input[0] * 255, 'c h w -> h w c').cpu().numpy())
                    cv2.imwrite(f'loci-background-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(background_next[0] * 255, 'c h w -> h w c').cpu().numpy())
                    cv2.imwrite(f'loci-bg-error-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(bg_error[0] * 255, 'c h w -> h w c').cpu().numpy())
                    cv2.imwrite(f'loci-position-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(position_next3d[0] * 255, 'c h w -> h w c').cpu().numpy())
                    cv2.imwrite(f'loci-output-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(output[0] * 255, 'c h w -> h w c').cpu().numpy())
                    cv2.imwrite(f'loci-highlited-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(highlited_input[0] * 255, 'c h w -> h w c').cpu().numpy())
                    cv2.imwrite(f'loci-error-{i:04d}-{t+cfg.teacher_forcing:03d}.jpg', rearrange(error_next[0] * 255, 'c h w -> h w c').cpu().numpy())

                    for o in range(cfg_net.num_objects + 1):
                        cv2.imwrite(f'loci-mask-{i:04d}-{t+cfg.teacher_forcing:03d}-{o:02d}.jpg', rearrange(mask_next[0,o] * 255, 'c h w -> h w c').cpu().numpy())
                        cv2.imwrite(f'loci-object-{i:04d}-{t+cfg.teacher_forcing:03d}-{o:02d}.jpg', rearrange(object_next[0,o] * 255, 'c h w -> h w c').cpu().numpy())

def export_dataset(cfg: Configuration, trainset: Dataset, testset: Dataset, net_file, hdf5_path):

    hdf5_file = h5py.File(f'{hdf5_path}.hdf5', "w")
    group_train = hdf5_file.create_group("train")
    group_test  = hdf5_file.create_group("test")

    export_latent(cfg, testset, net_file, hdf5_file, group_test)
    export_latent(cfg, trainset, net_file, hdf5_file, group_train)

    hdf5_file.flush()
    hdf5_file.close()

def export_latent(cfg: Configuration, dataset: Dataset, net_file, hdf5_file, hdf5_group):

    #assert(cfg.sequence_len == 2)
    cfg_net = cfg.model
    device = th.device(cfg.device)
    print(f"export_latent device: {device}")

    dataloader = DataLoader(
        dataset, 
        num_workers = 1, 
        pin_memory = False, 
        batch_size = cfg_net.batch_size,
        shuffle = False,
        drop_last = True,
    )

    # create model 
    net = Loci(
        cfg_net,
        teacher_forcing = 1000000
    )

    # load model
    if net_file != '':
        state = th.load(net_file, map_location=device)
        net.load_state_dict(state["model"])
    
    net = net.to(device=device)
    net.eval()

    to_batch  = LambdaModule(lambda x: rearrange(x, 'b (o c) -> (b o) c', o=cfg_net.num_objects))
    to_shared = LambdaModule(lambda x: rearrange(x, '(b o) c -> b o c', o=cfg_net.num_objects))

    init = net.get_init_status()
    
    avg_object_loss = 0
    avg_openings = 0
    avgsum  = 1e-30
    mseloss = nn.MSELoss()
    timer   = Timer()
    with th.no_grad():
        for batch_index, input in enumerate(dataloader):
            
            tensor            = input[0].float().to(device)
            background        = input[1].float().to(device)
            snitch_positions  = input[2].float()
            snitch_label      = input[3].float()
            snitch_contained  = input[4].float().unsqueeze(dim=2)
            actions_visible   = input[5].float()
            actions_hidden    = input[6].float()
            materials_visible = input[7].float()
            materials_hidden  = input[8].float()
            sizes_visible     = input[9].float()
            sizes_hidden      = input[10].float()
            colors_visible    = input[11].float()
            colors_hidden     = input[12].float()

            if background.shape[1] == 1:
                background = background.expand(tensor.shape)


            # run complete forward pass to get a position estimation
            position    = None
            gestalt     = None
            priority    = None
            mask        = None
            object      = None
            output      = None
            loss        = th.tensor(0)

            states_over_time = []

            input     = tensor[:,0:1]
            target    = th.clip(tensor[:,0], 0, 1)
            bg_input  = background[:,0:1]
            bg_target = background[:,0]
            net_error = th.sqrt(reduce((tensor[:,0] - background[:,0])**2, 'b c h w -> b 1 h w', 'mean'))

            for t in range(-cfg.teacher_forcing, tensor.shape[1]-1):
                if t >= 0:
                    input     = tensor[:,t:t+1]
                    target    = th.clip(tensor[:,t+1], 0, 1)
                    bg_input  = background[:,t:t+1]
                    bg_target = background[:,t+1]

                output, position, gestalt, priority, mask, _, _, _, _, _, _ = net(
                    input, 
                    net_error,
                    bg_input,
                    mask,
                    position,
                    gestalt,
                    priority,
                    reset = (t == -cfg.teacher_forcing),
                    evaluate=True
                )

                bg_error  = th.sqrt(reduce((target[:,0] - background[:,0])**2, 'b c h w -> b 1 h w', 'mean')).detach()
                net_error = th.sqrt(reduce((target[:,0] - output_next[:,0])**2, 'b c h w -> b 1 h w', 'mean')).detach()
                net_error = th.sqrt(net_error) * bg_error

                mask     = mask[:,0]
                position = position[:,0]
                gestalt  = gestalt[:,0]
                priority = priority[:,0]
                
                if t >= 0:
                    max_mask = th.max(th.max(mask[:,:-1], dim=3)[0], dim=2)[0]
                    state = th.cat((to_batch(position), to_batch(gestalt), to_batch(priority), to_batch(max_mask)), dim=1)
                    states_over_time.append(to_shared(state))

                    avg_openings = avg_openings * 0.99 + net.get_openings() 
                    avgsum       = avgsum * 0.99 + 1
                    
                    loss = mseloss(output[:,0] * bg_error, target * bg_error)
                    avg_object_loss        = avg_object_loss * 0.99 + loss.item()

            states_over_time = th.stack(states_over_time, dim=1)
            shape    = list(states_over_time.shape)
            shape[0] = 1

            snitch_positions  = snitch_positions[:,1:]
            snitch_contained  = snitch_contained[:,1:]
            actions_visible   = actions_visible[:,1:]
            actions_hidden    = actions_hidden[:,1:]
            materials_visible = materials_visible[:,1:]
            materials_hidden  = materials_hidden[:,1:]
            sizes_visible     = sizes_visible[:,1:]
            sizes_hidden      = sizes_hidden[:,1:]
            colors_visible    = colors_visible[:,1:]
            colors_hidden     = colors_hidden[:,1:]

            if batch_index == 0:
                hdf5_group.create_dataset(
                    name="object_states",
                    data=states_over_time.cpu().numpy(), 
                    compression='gzip', 
                    chunks=tuple(shape),
                    maxshape=(None,shape[1], shape[2], shape[3])
                )
                hdf5_group.create_dataset(
                    name="snitch_positions",
                    data=snitch_positions.numpy(), 
                    chunks=(1,300,3),
                    compression='gzip', 
                    maxshape=(None,300,3)
                )
                hdf5_group.create_dataset(
                    name="snitch_label",
                    data=snitch_label.numpy(), 
                    chunks=(1,),
                    compression='gzip', 
                    maxshape=(None,)
                )
                hdf5_group.create_dataset(
                    name="snitch_contained",
                    data=snitch_contained.numpy(), 
                    chunks=(1,300,1),
                    compression='gzip', 
                    maxshape=(None,300,1)
                )
                hdf5_group.create_dataset(
                    name="actions_visible",
                    data=actions_visible.numpy(), 
                    chunks=(1,300,14),
                    compression='gzip', 
                    maxshape=(None,300,14)
                )
                hdf5_group.create_dataset(
                    name="actions_hidden",
                    data=actions_hidden.numpy(), 
                    chunks=(1,300,14),
                    compression='gzip', 
                    maxshape=(None,300,14)
                )
                hdf5_group.create_dataset(
                    name="materials_visible",
                    data=materials_visible.numpy(), 
                    chunks=(1,300,8),
                    compression='gzip', 
                    maxshape=(None,300,8)
                )
                hdf5_group.create_dataset(
                    name="materials_hidden",
                    data=materials_hidden.numpy(), 
                    chunks=(1,300,8),
                    compression='gzip', 
                    maxshape=(None,300,8)
                )
                hdf5_group.create_dataset(
                    name="sizes_visible",
                    data=sizes_visible.numpy(), 
                    chunks=(1,300,12),
                    compression='gzip', 
                    maxshape=(None,300,12)
                )
                hdf5_group.create_dataset(
                    name="sizes_hidden",
                    data=sizes_hidden.numpy(), 
                    chunks=(1,300,12),
                    compression='gzip', 
                    maxshape=(None,300,12)
                )
                hdf5_group.create_dataset(
                    name="colors_visible",
                    data=colors_visible.numpy(), 
                    chunks=(1,300,32),
                    compression='gzip', 
                    maxshape=(None,300,32)
                )
                hdf5_group.create_dataset(
                    name="colors_hidden",
                    data=colors_hidden.numpy(), 
                    chunks=(1,300,32),
                    compression='gzip', 
                    maxshape=(None,300,32)
                )
            else:
                hdf5_group['object_states'].resize(
                    (hdf5_group['object_states'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['object_states'][-cfg_net.batch_size:] = states_over_time.cpu().numpy()
                hdf5_group['snitch_positions'].resize(
                    (hdf5_group['snitch_positions'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['snitch_positions'][-cfg_net.batch_size:] = snitch_positions.numpy()
                hdf5_group['snitch_label'].resize(
                    (hdf5_group['snitch_label'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['snitch_label'][-cfg_net.batch_size:] = snitch_label.numpy()
                hdf5_group['snitch_contained'].resize(
                    (hdf5_group['snitch_contained'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['snitch_contained'][-cfg_net.batch_size:] = snitch_contained.numpy()
                hdf5_group['actions_visible'].resize(
                    (hdf5_group['actions_visible'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['actions_visible'][-cfg_net.batch_size:] = actions_visible.numpy()
                hdf5_group['actions_hidden'].resize(
                    (hdf5_group['actions_hidden'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['actions_hidden'][-cfg_net.batch_size:] = actions_hidden.numpy()
                hdf5_group['materials_visible'].resize(
                    (hdf5_group['materials_visible'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['materials_visible'][-cfg_net.batch_size:] = materials_visible.numpy()
                hdf5_group['materials_hidden'].resize(
                    (hdf5_group['materials_hidden'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['materials_hidden'][-cfg_net.batch_size:] = materials_hidden.numpy()
                hdf5_group['sizes_visible'].resize(
                    (hdf5_group['sizes_visible'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['sizes_visible'][-cfg_net.batch_size:] = sizes_visible.numpy()
                hdf5_group['sizes_hidden'].resize(
                    (hdf5_group['sizes_hidden'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['sizes_hidden'][-cfg_net.batch_size:] = sizes_hidden.numpy()
                hdf5_group['colors_visible'].resize(
                    (hdf5_group['colors_visible'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['colors_visible'][-cfg_net.batch_size:] = colors_visible.numpy()
                hdf5_group['colors_hidden'].resize(
                    (hdf5_group['colors_hidden'].shape[0] + cfg_net.batch_size), axis=0
                )
                hdf5_group['colors_hidden'][-cfg_net.batch_size:] = colors_hidden.numpy()

            hdf5_file.flush()

            print("Exporting[{}/{}]: {}, {:.2f}%, Loss: {:.2e}, init: {:.4f}, openings: {:.2e}".format(
                batch_index,
                len(dataloader),
                str(timer),
                batch_index * 100 / len(dataloader),
                avg_object_loss/avgsum,
                net.get_init_status(),
                avg_openings / avgsum
            ), flush=True)

def evaluate(cfg: Configuration, num_gpus: int, dataset: Dataset, file, active_layer):

    rank = cfg.device
    world_size = -1

    print(f'rank {rank} online', flush=True)
    device = th.device(rank)

    if world_size < 0:
        rank = 0

    path = None
    if rank == 0:
        path = model_path(cfg, overwrite=False)
        cfg.save(path)

    cfg_net = cfg.model

    background = None
    if world_size > 0:
        dataset = DatasetPartition(dataset, rank, world_size)

    net = Loci(
        cfg = cfg_net,
        camera_view_matrix = dataset.cam if cfg.datatype == 'cater' else None,
        zero_elevation     = dataset.z if cfg.datatype == 'cater' else None,
        teacher_forcing    = cfg.teacher_forcing
    )

    net = net.to(device=device)

    if rank == 0:
        print(f'Loaded model with {sum([param.numel() for param in net.parameters()]):7d} parameters', flush=True)
        print(f'  States:     {sum([param.numel() for param in net.initial_states.parameters()]):7d} parameters', flush=True)
        print(f'  Encoder:    {sum([param.numel() for param in net.encoder.parameters()]):7d} parameters', flush=True)
        print(f'  Decoder:    {sum([param.numel() for param in net.decoder.parameters()]):7d} parameters', flush=True)
        print(f'  predictor:  {sum([param.numel() for param in net.predictor.parameters()]):7d} parameters', flush=True)
        print(f'  background: {sum([param.numel() for param in net.background.parameters()]):7d} parameters', flush=True)
        print("\n")
    
    # load model
    if file != '':
        print(f"load {file} to device {device}")
        state = th.load(file, map_location=device)

        # backward compatibility
        model = {}
        for key, value in state["model"].items():
            model[key.replace(".module.", ".")] = value

        net.load_state_dict(model)

    dataloader = DataLoader(
        dataset, 
        pin_memory = True, 
        num_workers = cfg.num_workers, 
        batch_size = cfg_net.batch_size, 
        shuffle = False,
        drop_last = True, 
        prefetch_factor = cfg.prefetch_factor, 
        persistent_workers = True
    )

    th.backends.cudnn.benchmark = True
    eval_net(net, 'Test', dataset, dataloader, device, cfg, 0)

