import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ms_ssim as msssim, ssim
import cv2

from torch.nn.parallel import DistributedDataParallel
import numpy as np
import os
from typing import Tuple, Union, List
from einops import rearrange, repeat, reduce

from utils.configuration import Configuration
from utils.parallel import run_parallel, DatasetPartition
from utils.scheduled_sampling import ExponentialSampler
from utils.io import model_path
from utils.optimizers import SDAdam, SDAMSGrad, RAdam
from model.loci import Loci
import torch.distributed as dist
import time
import random
import nn as nn_modules
from utils.io import Timer, BinaryStatistics, UEMA
from utils.data import DeviceSideDataset
from utils.loss import SSIMLoss, MSSSIMLoss
from nn.latent_classifier import CaterLocalizer, CaterObjectBehavior
from nn.tracker import TrackingLoss, L1GridDistance, L2TrackingDistance
from einops import rearrange, repeat, reduce

def save_model(
    file,
    net, 
    optimizer_init, 
    optimizer_encoder, 
    optimizer_decoder, 
    optimizer_predictor,
    optimizer_background
):

    state = { }

    state['optimizer_init'] = optimizer_init.state_dict()
    state['optimizer_encoder'] = optimizer_encoder.state_dict()
    state['optimizer_decoder'] = optimizer_decoder.state_dict()
    state['optimizer_predictor'] = optimizer_predictor.state_dict()
    state['optimizer_background'] = optimizer_background.state_dict()

    state["model"] = net.state_dict()
    th.save(state, file)

def load_model(
    file,
    cfg,
    net, 
    optimizer_init, 
    optimizer_encoder, 
    optimizer_decoder, 
    optimizer_predictor,
    optimizer_background,
    load_optimizers = True
):
    device = th.device(cfg.device)
    state = th.load(file, map_location=device)
    print(f"loaded {file} to {device}")
    print(f"load optimizers: {load_optimizers}")

    if load_optimizers:
        optimizer_init.load_state_dict(state[f'optimizer_init'])
        for n in range(len(optimizer_init.param_groups)):
            optimizer_init.param_groups[n]['lr'] = cfg.learning_rate

        optimizer_encoder.load_state_dict(state[f'optimizer_encoder'])
        for n in range(len(optimizer_encoder.param_groups)):
            optimizer_encoder.param_groups[n]['lr'] = cfg.learning_rate

        optimizer_decoder.load_state_dict(state[f'optimizer_decoder'])
        for n in range(len(optimizer_decoder.param_groups)):
            optimizer_decoder.param_groups[n]['lr'] = cfg.learning_rate

        optimizer_predictor.load_state_dict(state['optimizer_predictor'])
        for n in range(len(optimizer_predictor.param_groups)):
            optimizer_predictor.param_groups[n]['lr'] = cfg.learning_rate

        optimizer_background.load_state_dict(state['optimizer_background'])
        for n in range(len(optimizer_background.param_groups)):
            optimizer_background.param_groups[n]['lr'] = cfg.model.background.learning_rate

    # backward compatibility
    model = {}
    for k, v in state['model'].items():
        model[k.replace("module.","")] = v

    net.load_state_dict(model)

def run(cfg: Configuration, num_gpus: int, trainset: Dataset, valset: Dataset, testset: Dataset, file, active_layer):
    print("run training", flush=True)
    if num_gpus == 1:
        train_eprop(cfg.device, -1, cfg, trainset, valset, testset, file, active_layer)
    else:
        run_parallel(train_eprop, num_gpus, cfg, trainset, valset, testset, file, active_layer)

def train_eprop(rank: int, world_size: int, cfg: Configuration, trainset: Dataset, valset: Dataset, testset: Dataset, file, active_layer):

    print(f'rank {rank} online', flush=True)
    device = th.device(rank)

    if world_size < 0:
        rank = 0

    path = None
    if rank == 0:
        path = model_path(cfg, overwrite=False)
        cfg.save(path)

    cfg_net = cfg.model
    sampler = ExponentialSampler(0.9999)

    background = None
    if world_size > 0:
        dataset = DatasetPartition(dataset, rank, world_size)

    net = Loci(
        cfg = cfg_net,
        camera_view_matrix = trainset.cam if cfg.datatype == 'cater' else None,
        zero_elevation     = trainset.z if cfg.datatype == 'cater' else None,
        sampler = sampler if cfg.scheduled_sampling else None,
        teacher_forcing = cfg.teacher_forcing
    )

    net = net.to(device=device)

    l1distance = L1GridDistance(trainset.cam, trainset.z).to(device) if cfg.datatype == 'cater' else None
    l2distance = L2TrackingDistance(trainset.cam, trainset.z).to(device) if cfg.datatype == 'cater' else None
    l2loss     = TrackingLoss(trainset.cam, trainset.z).to(device) if cfg.datatype == 'cater' else None

    if rank == 0:
        print(f'Loaded model with {sum([param.numel() for param in net.parameters()]):7d} parameters', flush=True)
        print(f'  States:     {sum([param.numel() for param in net.initial_states.parameters()]):7d} parameters', flush=True)
        print(f'  Encoder:    {sum([param.numel() for param in net.encoder.parameters()]):7d} parameters', flush=True)
        print(f'  Decoder:    {sum([param.numel() for param in net.decoder.parameters()]):7d} parameters', flush=True)
        print(f'  predictor:  {sum([param.numel() for param in net.predictor.parameters()]):7d} parameters', flush=True)
        print(f'  background: {sum([param.numel() for param in net.background.parameters()]):7d} parameters', flush=True)
        print("\n")
    
    bceloss    = nn.BCELoss()
    mseloss    = nn.MSELoss()
    msssimloss = MSSSIMLoss()
    l1loss     = nn.L1Loss()

    optimizer_init = th.optim.Adam(net.initial_states.parameters(), lr = cfg.learning_rate * 30, betas=(0.9,0.99), eps=1e-16)
    optimizer_encoder = RAdam(net.encoder.parameters(), lr = cfg.learning_rate)
    optimizer_decoder = RAdam(net.decoder.parameters(), lr = cfg.learning_rate)
    optimizer_predictor = RAdam(net.predictor.parameters(), lr = cfg.learning_rate)
    optimizer_background = RAdam(net.background.parameters(), lr = cfg_net.background.learning_rate)

    if file != "":
        load_model(
            file,
            cfg,
            net,
            optimizer_init, 
            optimizer_encoder, 
            optimizer_decoder, 
            optimizer_predictor,
            optimizer_background,
            cfg.load_optimizers
        )
        print(f'loaded[{rank}] {file}', flush=True)

            
    trainloader = DataLoader(
        trainset, 
        pin_memory = True, 
        num_workers = cfg.num_workers, 
        batch_size = cfg_net.batch_size, 
        shuffle = True,
        drop_last = True, 
        prefetch_factor = cfg.prefetch_factor, 
        persistent_workers = True
    )

    valloader = DataLoader(
        valset, 
        pin_memory = True, 
        num_workers = cfg.num_workers, 
        batch_size = cfg_net.batch_size, 
        shuffle = False,
        drop_last = True, 
        prefetch_factor = cfg.prefetch_factor, 
        persistent_workers = True
    )

    testloader = DataLoader(
        testset, 
        pin_memory = True, 
        num_workers = cfg.num_workers, 
        batch_size = cfg_net.batch_size, 
        shuffle = False,
        drop_last = True, 
        prefetch_factor = cfg.prefetch_factor, 
        persistent_workers = True
    )

    net_parallel = None
    if world_size > 0:
        net_parallel = DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)
    else:
        net_parallel = net
    
    if rank == 0:
        save_model(
            os.path.join(path, 'net0.pt'),
            net, 
            optimizer_init, 
            optimizer_encoder, 
            optimizer_decoder, 
            optimizer_predictor,
            optimizer_background
        )

    #th.autograd.set_detect_anomaly(True)

    num_updates = 0
    num_time_steps = 0
    lr = cfg.learning_rate

    avgloss                  = UEMA()
    avg_position_loss        = UEMA()
    avg_time_loss            = UEMA()
    avg_object_loss          = UEMA()
    avg_mse_object_loss      = UEMA()
    avg_long_mse_object_loss = UEMA(100 * (cfg.sequence_len - 1 + cfg.teacher_forcing))
    avg_num_objects          = UEMA()
    avg_openings             = UEMA()
    avg_l1_distance          = UEMA()
    avg_l2_distance          = UEMA()
    avg_top1_accuracy        = UEMA()
    avg_tracking_loss        = UEMA()

    th.backends.cudnn.benchmark = True
    timer = Timer()

    for epoch in range(cfg.epochs):
        #if epoch > 0:
        #    eval_net(net_parallel, 'Val',  valset,  valloader, device, cfg, epoch)
        #    eval_net(net_parallel, 'Test', testset, testloader, device, cfg, epoch)
        for batch_index, input in enumerate(trainloader):
            
            tensor          = input[0]
            old_background  = input[1].to(device)
            target_position = input[2].float().to(device) if cfg.datatype == 'cater' else None
            target_label    = input[3].to(device) if cfg.datatype == 'cater' else None

            if cfg.datatype == 'cater' and epoch == 0 and batch_index == 0:
                net_parallel.background.set_background(old_background[:,0])

            # run complete forward pass to get a position estimation
            position    = None
            gestalt     = None
            priority    = None
            mask        = None
            object      = None
            output      = None
            loss        = th.tensor(0)

            sequence_len = (tensor.shape[1]-1) 

            input      = tensor[:,0].to(device)
            input_next = input
            target     = th.clip(input, 0, 1).detach()
            pos_target = target_position[:,0] if cfg.datatype == 'cater' else None
            error      = None

            for t in range(-cfg.teacher_forcing, sequence_len):
                if t >= 0:
                    num_time_steps += 1
                
                if t >= 0:
                    input      = input_next
                    input_next = tensor[:,t+1].to(device)
                    target     = th.clip(input_next, 0, 1)
                    pos_target = target_position[:,t+1] if cfg.datatype == 'cater' else None

                if num_updates <= cfg.background_pretraining_steps and net.get_init_status() < 0.001:
                    output = net_parallel(input, train_background=True, detach=True, reset=(t == -cfg.teacher_forcing))

                    loss = l1loss(output, target)
                    optimizer_background.zero_grad()
                    loss.backward()
                    optimizer_background.step()
                else:

                    output, position, gestalt, priority, mask, object, background, position_loss, object_loss, time_loss, snitch_position = net_parallel(
                        input, 
                        error,
                        mask,
                        position,
                        gestalt,
                        priority,
                        reset = (t == -cfg.teacher_forcing),
                    )

                    object_loss   = object_loss   * cfg_net.object_regularizer
                    position_loss = position_loss * cfg_net.position_regularizer
                    time_loss     = time_loss     * cfg_net.time_regularizer
                    tracking_loss = l2loss(snitch_position, pos_target) * cfg_net.supervision_factor if cfg.datatype == 'cater' else 0

                    bg_error = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                    error    = th.sqrt(reduce((target - output)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                    error    = th.sqrt(error) * bg_error

                    mask     = mask.detach()
                    position = position.detach()
                    gestalt  = gestalt.detach()
                    priority = priority.detach()

                    avg_openings.update(net.get_openings())

                    if t == sequence_len - 1 and cfg.datatype == 'cater':
                        l1, top1, top5, _ = l1distance(snitch_position, target_label)
                        l2                = th.mean(l2distance(snitch_position, target_position[:,-1])[0])

                        avg_l1_distance.update(l1.item())
                        avg_l2_distance.update(l2.item())
                        avg_top1_accuracy.update(top1.item())
                    
                    if t >= cfg.statistics_offset:
                        old_bg_error = th.sqrt(reduce((target - old_background[:,0])**2, 'b c h w -> b 1 h w', 'mean')).detach()
                        loss = mseloss(output * old_bg_error, target * old_bg_error)

                        avg_position_loss.update(position_loss.item())
                        avg_object_loss.update(object_loss.item())
                        avg_time_loss.update(time_loss.item())
                        avg_mse_object_loss.update(loss.item())
                        avg_long_mse_object_loss.update(loss.item())

                        if cfg.datatype == 'cater': 
                            avg_tracking_loss.update(tracking_loss.item())

                        avg_num_objects.update(th.mean(reduce((reduce(mask[:,:-1], 'b c h w -> b c', 'max') > 0.5).float(), 'b c -> b', 'sum')).item())


                    init          = max(0, min(1, net.get_init_status()))
                    fg_mask       = th.gt(bg_error, 0.1).float().detach()

                    cliped_output = th.clip(output, 0, 1)
                    target        = th.clip(target * fg_mask * (1 - init) + target * init, 0, 1)
                    
                    loss = None
                    if tensor.shape[-2] >= 64 and cfg.msssim:
                        loss = msssimloss(cliped_output, target) + position_loss + object_loss + time_loss + tracking_loss
                    else:
                        loss = bceloss(cliped_output, target) + position_loss + object_loss + time_loss + tracking_loss

                    if t == -cfg.teacher_forcing:
                        optimizer_init.zero_grad()

                    optimizer_encoder.zero_grad()
                    optimizer_decoder.zero_grad()
                    optimizer_predictor.zero_grad()

                    if num_updates >= cfg.entity_pretraining_steps or net.get_init_status() > 0.001:
                        optimizer_background.zero_grad()

                    loss.backward()

                    if t == -cfg.teacher_forcing:
                        optimizer_init.step()
                    else:
                        optimizer_encoder.step()
                        optimizer_decoder.step()
                        optimizer_predictor.step()

                        if num_updates >= cfg.entity_pretraining_steps or net.get_init_status() > 0.001:
                            optimizer_background.step()

                avgloss.update(loss.item())

                num_updates += 1
                print("Epoch[{}/{}/{}/{}]: {}, {}, Loss: {:.2e}|{:.2e}|{:.2e}, reg: {:.2e}|{:.2e}|{:.2e}, snitch:, {:.2e}|{:.2f}|{:.2f}|{:.2f}, i: {:.2f}, obj: {:.1f}, openings: {:.2e}".format(
                    num_updates,
                    num_time_steps,
                    sequence_len,
                    1,
                    str(timer),
                    epoch + 1,
                    np.abs(float(avgloss)),
                    float(avg_mse_object_loss),
                    float(avg_long_mse_object_loss),
                    float(avg_object_loss),
                    float(avg_time_loss),
                    float(avg_position_loss),
                    float(avg_tracking_loss),
                    float(avg_top1_accuracy),
                    float(avg_l1_distance),
                    float(avg_l2_distance),
                    net.get_init_status(),
                    float(avg_num_objects),
                    float(avg_openings),
                ), flush=True)

                if num_updates > cfg.updates:
                    if rank == 0:
                        save_model(
                            os.path.join(path, 'net.pt'),
                            net, 
                            optimizer_init, 
                            optimizer_encoder, 
                            optimizer_decoder, 
                            optimizer_predictor,
                            optimizer_background
                        )
                    return

            if batch_index % 10 == 0 or num_updates < 3000:
                if not np.isinf(loss.item()) and not np.isnan(loss.item()):
                    if rank == 0:
                        save_model(
                            os.path.join(path, 'net{}.{}.pt'.format(epoch, batch_index // 100)),
                            net, 
                            optimizer_init, 
                            optimizer_encoder, 
                            optimizer_decoder, 
                            optimizer_predictor,
                            optimizer_background
                        )


        if (epoch + 1) % 1 == 0 or (epoch < 10 and batch_index % 100 == 0):
            if not np.isinf(loss.item()) and not np.isnan(loss.item()):
                if rank == 0:
                    save_model(
                        os.path.join(path, 'net{}.pt'.format(epoch+1)),
                        net, 
                        optimizer_init, 
                        optimizer_encoder, 
                        optimizer_decoder, 
                        optimizer_predictor,
                        optimizer_background
                    )


def eval_net(net_parallel, prefix, dataset, dataloader, device, cfg, epoch):

    net = net_parallel

    timer = Timer()

    bceloss    = nn.BCELoss()
    mseloss    = nn.MSELoss()
    msssimloss = MSSSIMLoss()

    l1distance = L1GridDistance(dataset.cam, dataset.z).to(device)
    l2distance = L2TrackingDistance(dataset.cam, dataset.z).to(device)
    l2loss     = TrackingLoss(dataset.cam, dataset.z).to(device)

    avgloss = 0
    avg_position_loss = 0
    avg_position_sum = 1e-30
    avg_time_loss = 0
    avg_time_sum = 1e-30
    avg_object_loss = 0
    avg_object_sum = 1e-30
    avg_object_loss2 = 0
    avg_object_sum2 = 1e-30
    avg_object_loss3 = 0
    avg_object_sum3 = 1e-30
    avg_num_objects = 0
    avg_num_objects_sum = 1e-30
    avg_openings = 0
    avg_openings_sum = 1e-30
    avg_l1_distance = 0
    avg_l2_distance = 0
    avg_top1_accuracy = 0
    avg_top5_accuracy = 0
    avg_tracking_sum = 1e-30
    avg_tracking_loss = 0
    avgsum = 1e-30
    min_loss = 100000
    num_updates = 0
    num_time_steps = 0
    l2_contained = []
    for t in range(cfg.sequence_len):
        l2_contained.append([])

    with th.no_grad():
        for batch_index, input in enumerate(dataloader):
            
            tensor           = input[0]
            old_background   = input[1].to(device)
            target_position  = input[2].float().to(device)
            target_label     = input[3].to(device)
            snitch_contained = input[4].to(device)
            snitch_contained_time = th.zeros_like(snitch_contained)

            if batch_index == 0:
                net_parallel.background.set_background(old_background[:,0])
            
            snitch_contained_time[:,0] = snitch_contained[:,0]
            for t in range(1, snitch_contained.shape[1]):
                snitch_contained_time[:,t] = snitch_contained_time[:,t-1] * snitch_contained[:,t] + snitch_contained[:,t]

            snitch_contained_time = snitch_contained_time.long()
                

            # run complete forward pass to get a position estimation
            position    = None
            gestalt     = None
            priority    = None
            mask        = None
            object      = None
            net.sampler = None
            output      = None
            loss        = th.tensor(0)

            sequence_len = (tensor.shape[1]-1) 

            input      = tensor[:,0].to(device)
            input_next = input
            target     = th.clip(input, 0, 1).detach()
            pos_target = target_position[:,0] if cfg.datatype == 'cater' else None
            error      = None

            for t in range(-cfg.teacher_forcing, sequence_len):
                if t >= 0:
                    num_time_steps += 1
                
                if t >= 0:
                    input      = input_next
                    input_next = tensor[:,t+1].to(device)
                    target     = th.clip(input_next, 0, 1)
                    pos_target = target_position[:,t+1] if cfg.datatype == 'cater' else None

                output, position, gestalt, priority, mask, object, background, position_loss, object_loss, time_loss, snitch_position = net_parallel(
                    input, 
                    error,
                    mask,
                    position,
                    gestalt,
                    priority,
                    reset = (t == -cfg.teacher_forcing),
                    test = True
                )

                init = max(0, min(1, net.get_init_status()))

                bg_error = th.sqrt(reduce((target - background)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error    = th.sqrt(reduce((target - output)**2, 'b c h w -> b 1 h w', 'mean')).detach()
                error    = th.sqrt(error) * bg_error

                mask     = mask.detach()
                position = position.detach()
                gestalt  = gestalt.detach()
                priority = priority.detach()

                if t >= 0:

                    l2, _ = l2distance(snitch_position, target_position[:,t+1])
                    for b in range(snitch_contained_time.shape[0]):
                        c = snitch_contained_time[b,t].item()
                        if c > 0.5:
                            l2_contained[c].append(l2[b].detach().item())


                if t == sequence_len - 1:
                    l1, top1, top5, label = l1distance(snitch_position, target_label)
                    l2, world_position    = l2distance(snitch_position, target_position[:,-1])
                    l2                    = th.mean(l2)

                    batch_size = label.shape[0]
                    for n in range(batch_size):
                        print(f"Sample {batch_size * batch_index + n:04d}: ", end="")
                        print(f"Label: {label[n].item():02d}, Target: {target_label[n].long().item():02d}, ", end="")
                        print(f"World-Position: {world_position[n,0].item():.5f}|{world_position[n,1].item():.5f}|{world_position[n,2].item():.5f}, ", end="")
                        print(f"Target-Position: {target_position[n,-1,0].item():.5f}|{target_position[n,-1,1].item():.5f}|{target_position[n,-1,2].item():.5f}, ", flush=True)

                    avg_l1_distance = avg_l1_distance + l1.item()
                    avg_l2_distance = avg_l2_distance + l2.item()
                    avg_top1_accuracy = avg_top1_accuracy + top1.item()
                    avg_top5_accuracy = avg_top5_accuracy + top5.item()
                    avg_tracking_sum  = avg_tracking_sum + 1
                
                if t >= cfg.statistics_offset:
                    loss = mseloss(output * bg_error, target * bg_error)
                    avg_position_sum  = avg_position_sum + 1
                    avg_object_sum3   = avg_object_sum3 + 1
                    avg_time_sum      = avg_time_sum + 1
                    avg_object_loss   = avg_object_loss + loss.item()
                    avg_object_sum    = avg_object_sum + 1
                    avg_object_loss2  = avg_object_loss2 + loss.item()
                    avg_object_sum2   = avg_object_sum2 + 1

                    n_objects = th.mean(reduce((reduce(mask, 'b c h w -> b c', 'max') > 0.5).float()[:,1:], 'b c -> b', 'sum')).item()
                    avg_num_objects = avg_num_objects + n_objects
                    avg_num_objects_sum = avg_num_objects_sum + 1

                fg_mask = th.gt(bg_error, 0.1).float().detach()

                cliped_output = th.clip(output, 0, 1)
                target        = th.clip(target * fg_mask * (1 - init) + target * init, 0, 1)

                loss = None
                if tensor.shape[-2] >= 64 and cfg.msssim:
                    loss = msssimloss(cliped_output, target)
                else:
                    loss = bceloss(cliped_output, target)

                avgloss  = avgloss + loss.item()
                avgsum   = avgsum + 1

                num_updates += 1
                print("{}[{}/{}/{}]: {}, Loss: {:.2e}|{:.2e}, snitch:, {:.2e}, top1: {:.4f}, top5: {:.4f}, L1:, {:.6f}, L2: {:.6f}, i: {:.2f}, obj: {:.1f}, openings: {:.2e}".format(
                    prefix,
                    num_updates,
                    num_time_steps,
                    sequence_len * len(dataloader),
                    str(timer),
                    np.abs(avgloss/avgsum),
                    avg_object_loss/avg_object_sum,
                    avg_tracking_loss / avg_time_sum,
                    avg_top1_accuracy / avg_tracking_sum * 100,
                    avg_top5_accuracy / avg_tracking_sum * 100,
                    avg_l1_distance / avg_tracking_sum,
                    avg_l2_distance  / avg_tracking_sum,
                    net.get_init_status(),
                    avg_num_objects / avg_num_objects_sum,
                    avg_openings / avg_openings_sum,
                ), flush=True)

    print("\\addplot+[mark=none,name path=quantil9,trialcolorb,opacity=0.1,forget plot] plot coordinates {")
    for t in range(cfg.sequence_len):
        if len(l2_contained[t]) > 0:
            data = np.array(l2_contained[t])
            print(f"({t},{np.quantile(data, 0.9):0.4f})")
    print("};")

    print("\\addplot+[mark=none,name path=quantil75,trialcolorb,opacity=0.3,forget plot] plot coordinates {")
    for t in range(cfg.sequence_len):
        if len(l2_contained[t]) > 0:
            data = np.array(l2_contained[t])
            print(f"({t},{np.quantile(data, 0.75):0.4f})")
    print("};")

    print("\\addplot+[mark=none,trialcolorb,thick,forget plot] plot coordinates {")
    for t in range(cfg.sequence_len):
        if len(l2_contained[t]) > 0:
            data = np.array(l2_contained[t])
            print(f"({t},{np.quantile(data, 0.5):0.4f})")
    print("};")
    print("\\addplot+[mark=none,trialcolorb,thick,dotted,forget plot] plot coordinates {")
    for t in range(cfg.sequence_len):
        if len(l2_contained[t]) > 0:
            data = np.array(l2_contained[t])
            print(f"({t},{np.mean(data):0.4f})")
    print("};")

    print("\\addplot+[mark=none,name path=quantil25,trialcolorb,opacity=0.3,forget plot] plot coordinates {")
    for t in range(cfg.sequence_len):
        if len(l2_contained[t]) > 0:
            data = np.array(l2_contained[t])
            print(f"({t},{np.quantile(data, 0.25):0.4f})")
    print("};")

    print("\\addplot+[mark=none,name path=quantil1,trialcolorb,opacity=0.1,forget plot] plot coordinates {")
    for t in range(cfg.sequence_len):
        if len(l2_contained[t]) > 0:
            data = np.array(l2_contained[t])
            print(f"({t},{np.quantile(data, 0.1):0.4f})")
    print("};")
    print("\\addplot[trialcolorb,opacity=0.1] fill between[of=quantil9 and quantil1];")
    print("\\addplot[trialcolorb,opacity=0.2] fill between[of=quantil75 and quantil25];")

class L1GridDistanceTracker(nn.Module):
    def __init__(self):
        super(L1GridDistanceTracker, self).__init__()

    def forward(self, label, target_label):

        label        = label.long()
        target_label = target_label.long()

        x = label % 6
        y = label / 6

        target_x = target_label % 6
        target_y = target_label / 6
        
        return th.sum((th.abs(x - target_x) + th.abs(y - target_y)).float())

def train_latent_tracker(cfg: Configuration, trainset: Dataset, valset: Dataset, testset: Dataset, file):

    print(f'device {cfg.device} online', flush=True)
    device = th.device(cfg.device)

    path = model_path(cfg, overwrite=False)
    cfg.save(path)

    cfg_net = cfg.model

    net = CaterLocalizer(
        gestalt_size       = cfg_net.gestalt_size,
        net_width          = 1,
        num_objects        = cfg_net.num_objects,
        num_timesteps      = cfg.sequence_len,
        camera_view_matrix = trainset.cam,
        zero_elevation     = trainset.z
    )

    net = net.to(device=device)

    print(f'Loaded model with {sum([param.numel() for param in net.parameters()]):6d} parameters', flush=True)

    mseloss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    l1grid = L1GridDistanceTracker()

    optimizer = th.optim.Adam(
        net.parameters(), 
        lr = cfg.learning_rate,
        amsgrad=True
    )

    l2_contained = []
    for t in range(cfg.sequence_len):
        l2_contained.append([])

    cam = np.array([
        (1.4503, 1.6376,  0.0000, -0.0251),
        (-1.0346, 0.9163,  2.5685,  0.0095),
        (-0.6606, 0.5850, -0.4748, 10.5666),
        (-0.6592, 0.5839, -0.4738, 10.7452)
    ])

    z = 0.3421497941017151

    l2distance = L2TrackingDistance(cam, z).to(device)

    if file != "":
        state = th.load(file)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        print(f'loaded {file}', flush=True)

            
    trainloader = DeviceSideDataset(trainset, device, cfg_net.batch_size)
    testloader  = DeviceSideDataset(testset, device, cfg_net.batch_size)
    valloader   = DeviceSideDataset(valset, device, cfg_net.batch_size)

    #trainloader = DataLoader(
    #    trainset, 
    #    pin_memory = True, 
    #    shuffle = True,
    #    drop_last = True, 
    #    num_workers = cfg.num_workers, 
    #    batch_size = cfg_net.batch_size, 
    #    prefetch_factor = cfg.prefetch_factor, 
    #    persistent_workers = True
    #)
    #testloader = DataLoader(
    #    testset, 
    #    pin_memory = True, 
    #    shuffle = True,
    #    drop_last = True, 
    #    num_workers = cfg.num_workers, 
    #    batch_size = cfg_net.batch_size, 
    #    prefetch_factor = cfg.prefetch_factor, 
    #    persistent_workers = True
    #)
    #valloader = DataLoader(
    #    valset, 
    #    pin_memory = True, 
    #    shuffle = True,
    #    drop_last = True, 
    #    num_workers = cfg.num_workers, 
    #    batch_size = cfg_net.batch_size, 
    #    prefetch_factor = cfg.prefetch_factor, 
    #    persistent_workers = True
    #)

    state = { }
    state['optimizer'] = optimizer.state_dict()
    state["model"] = net.state_dict()
    #th.save(state, os.path.join(path, 'net0.pt'))

    lr = cfg.learning_rate

    th.backends.cudnn.benchmark = True
    timer = Timer()

    #time_weight = th.arange(300, device=device).view(1,-1,1) / 300

    for epoch in range(cfg.epochs):

        avg_train_loss = 0
        avg_test_loss  = 0
        avg_val_loss  = 0

        avg_train_l2_distance = 0
        avg_test_l2_distance  = 0
        avg_val_l2_distance  = 0

        avg_train_l1_distance = 0
        avg_test_l1_distance  = 0
        avg_val_l1_distance  = 0

        avg_train_sum = 0
        avg_test_sum = 0
        avg_val_sum = 0

        avg_train_top1 = 0
        avg_test_top1 = 0
        avg_val_top1 = 0

        avg_train_top5 = 0
        avg_test_top5 = 0
        avg_val_top5 = 0

        net.train()
        for batch_index, data in enumerate(trainloader):

            tensor, positions, labels, contained, actions = data[:5]
            #tensor    = tensor.to(device)
            #positions = positions.to(device)
            #labels    = labels.to(device)
            #contained = contained.to(device)
            #actions   = actions.to(device)
                

            contained   = contained.squeeze(dim=2)
            ground_mask = 1 - actions[:,:,3]
            labels      = labels.int()

            out_visible, out_hidden, label_visible, label_hidden = net(tensor)
            output = out_visible * (1 - contained).unsqueeze(dim=2) + out_hidden * contained.unsqueeze(dim=2)
            out_label = label_visible * (1 - contained[:,-1:]) + label_hidden * contained[:,-1:]

            loss = th.mean((output - positions)**2) 

            avg_train_top1 += th.mean((th.argmax(out_label.detach(), dim=1) == labels).float()) / len(trainloader)
            avg_train_top5 += th.mean(th.sum((th.topk(out_label.detach(), 5, dim=1)[1] == labels.unsqueeze(dim=1)).float(), dim=1)) / len(trainloader)

            avg_train_loss        += loss.item()
            avg_train_l2_distance += th.sum(th.sqrt(th.sum((output[:,-1] - positions[:,-1])**2, dim=1)), dim=0).detach().item()

            avg_train_l1_distance += l1grid(th.argmax(out_label.detach(), dim=1), labels).item()
            avg_train_sum         += output.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with th.no_grad():
            net.eval()
            for batch_index, data in enumerate(testloader):

                tensor, positions, labels, contained, actions = data[:5]
                #tensor    = tensor.to(device)
                #positions = positions.to(device)
                #labels    = labels.to(device)
                #contained = contained.to(device)
                #actions   = actions.to(device)

                contained_time = th.zeros_like(contained)
                contained_time[:,0] = contained[:,0]
                for t in range(1, cfg.sequence_len):
                    contained_time[:,t] = contained_time[:,t-1] * contained[:,t] + contained[:,t]

                contained_time = contained_time.long()

                contained   = contained.squeeze(dim=2)
                ground_mask = 1 - actions[:,:,3]

                out_visible, out_hidden, label_visible, label_hidden = net(tensor)
                output = out_visible * (1 - contained).unsqueeze(dim=2) + out_hidden * contained.unsqueeze(dim=2)
                out_label = label_visible * (1 - contained[:,-1:]) + label_hidden * contained[:,-1:]

                loss = th.mean((output - positions)**2) 

                avg_test_top1 += th.mean((th.argmax(out_label, dim=1) == labels).float()) / len(testloader)
                avg_test_top5 += th.mean(th.sum((th.topk(out_label, 5, dim=1)[1] == labels.unsqueeze(dim=1)).float(), dim=1)) / len(testloader)

                avg_test_loss        += loss.item()
                avg_test_l2_distance += th.sum(th.sqrt(th.sum((output[:,-1] - positions[:,-1])**2, dim=1)), dim=0).detach().item()

                avg_test_l1_distance += l1grid(th.argmax(out_label.detach(), dim=1), labels).item()

                avg_test_sum         += output.shape[0] 

                for t in range(1, cfg.sequence_len):
                    l2 = th.sqrt(th.sum((output[:,t] - positions[:,t])**2, dim=1))
                    for b in range(contained_time.shape[0]):
                        c = contained_time[b,t].item()
                        if c > 0.5:
                            l2_contained[c].append(l2[b].detach().item())

            print("\\addplot+[mark=none,name path=quantil9,trialcolor,opacity=0.1,forget plot] plot coordinates {")
            for t in range(cfg.sequence_len):
                if len(l2_contained[t]) > 0:
                    data = np.array(l2_contained[t])
                    print(f"({t},{np.quantile(data, 0.9):0.4f})")
            print("};")

            print("\\addplot+[mark=none,name path=quantil75,trialcolor,opacity=0.3,forget plot] plot coordinates {")
            for t in range(cfg.sequence_len):
                if len(l2_contained[t]) > 0:
                    data = np.array(l2_contained[t])
                    print(f"({t},{np.quantile(data, 0.75):0.4f})")
            print("};")

            print("\\addplot+[mark=none,trialcolor,thick,forget plot] plot coordinates {")
            for t in range(cfg.sequence_len):
                if len(l2_contained[t]) > 0:
                    data = np.array(l2_contained[t])
                    print(f"({t},{np.quantile(data, 0.5):0.4f})")
            print("};")
            print("\\addplot+[mark=none,trialcolor,thick,dotted,forget plot] plot coordinates {")
            for t in range(cfg.sequence_len):
                if len(l2_contained[t]) > 0:
                    data = np.array(l2_contained[t])
                    print(f"({t},{np.mean(data):0.4f})")
            print("};")

            print("\\addplot+[mark=none,name path=quantil25,trialcolor,opacity=0.3,forget plot] plot coordinates {")
            for t in range(cfg.sequence_len):
                if len(l2_contained[t]) > 0:
                    data = np.array(l2_contained[t])
                    print(f"({t},{np.quantile(data, 0.25):0.4f})")
            print("};")

            print("\\addplot+[mark=none,name path=quantil1,trialcolor,opacity=0.1,forget plot] plot coordinates {")
            for t in range(cfg.sequence_len):
                if len(l2_contained[t]) > 0:
                    data = np.array(l2_contained[t])
                    print(f"({t},{np.quantile(data, 0.1):0.4f})")
            print("};")
            print("\\addplot[trialcolor,opacity=0.1] fill between[of=quantil9 and quantil1];")
            print("\\addplot[trialcolor,opacity=0.2] fill between[of=quantil75 and quantil25];")


                    #print(f"Contained[{t:03d}] {len(data):04d}: {np.quantile(data, 0.25):0.4f}|{np.quantile(data, 0.5):.4f}|{np.quantile(data, 0.75):.4f}")
            return

            for batch_index, data in enumerate(valloader):

                tensor, positions, labels, contained, actions = data[:5]
                #tensor    = tensor.to(device)
                #positions = positions.to(device)
                #labels    = labels.to(device)
                #contained = contained.to(device)
                #actions   = actions.to(device)
                
                contained   = contained.squeeze(dim=2)
                ground_mask = 1 - actions[:,:,3]

                out_visible, out_hidden, label_visible, label_hidden = net(tensor)
                output = out_visible * (1 - contained).unsqueeze(dim=2) + out_hidden * contained.unsqueeze(dim=2)
                out_label = label_visible * (1 - contained[:,-1:]) + label_hidden * contained[:,-1:]

                loss = th.mean((output - positions)**2)

                avg_val_top1 += th.mean((th.argmax(out_label, dim=1) == labels).float()) / len(valloader)
                avg_val_top5 += th.mean(th.sum((th.topk(out_label, 5, dim=1)[1] == labels.unsqueeze(dim=1)).float(), dim=1)) / len(valloader)

                avg_val_loss        += loss.item()
                avg_val_l2_distance += th.sum(th.sqrt(th.sum((output[:,-1] - positions[:,-1])**2, dim=1)), dim=0).detach().item()

                avg_val_l1_distance += l1grid(th.argmax(out_label.detach(), dim=1), labels).item()

                avg_val_sum         += output.shape[0] 

        np.set_printoptions(threshold=10000)
        np.set_printoptions(linewidth=np.inf)
        
        
        print("Epoch[{}/{}]: {}, Loss: {:.2e}|{:.2e}|{:.2e}, L2: {:.4f}|{:.4f}|{:.4f}, L1: {:.4f}|{:.4f}|{:.4f}, lr: {:.2e}, Top-1: {:.4f}%|{:.4f}%|{:.4f}%, Top-5: {:.4f}%|{:.4f}%|{:.4f}%".format(
            epoch + 1,
            cfg.epochs,
            str(timer),
            avg_train_loss/len(trainloader),
            avg_val_loss/len(valloader),
            avg_test_loss/len(testloader),
            avg_train_l2_distance / avg_train_sum,
            avg_val_l2_distance / avg_val_sum,
            avg_test_l2_distance / avg_test_sum,
            avg_train_l1_distance / avg_train_sum,
            avg_val_l1_distance / avg_val_sum,
            avg_test_l1_distance / avg_test_sum,
            lr,
            avg_train_top1 * 100,
            avg_val_top1 * 100,
            avg_test_top1 * 100,
            avg_train_top5 * 100,
            avg_val_top5 * 100,
            avg_test_top5 * 100,
        ), flush=True)

    state = { }
    state['optimizer'] = optimizer.state_dict()
    state["model"] = net.state_dict()
    th.save(state, os.path.join(path, 'net.pt'))
