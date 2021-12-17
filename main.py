import os
import sys
import json
import subprocess
import random

import torch
from torch import nn
from torch.nn import BCELoss, CrossEntropyLoss
from torch.backends import cudnn
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.transforms import transforms

import pandas as pd
import numpy as np
from opts import parse_opts
from model import generate_model, make_data_parallel, load_pretrained_model, resume_model, get_fine_tuning_parameters
from dataset import VideoFrameDataset
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
# from mean import get_mean
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)


def get_train_utils(opt, model_parameters):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    # normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
    #                                  opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(transforms.ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    # spatial_transform.append(ScaleValue(opt.value_scale))
    # spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    # assert opt.train_t_crop in ['random', 'center']
    # temporal_transform = []
    # if opt.sample_t_stride > 1:
    #     temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    # if opt.train_t_crop == 'random':
    #     temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    # elif opt.train_t_crop == 'center':
    #     temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    # temporal_transform = TemporalCompose(temporal_transform)

    train_data = VideoFrameDataset(
        root_path=opt.root_path,
        annotationfile_path=opt.annotation_path,
        num_segments=1,
        frames_per_segment=opt.t_dim_frames,
        imagefile_template=opt.imagefile_template,
        transform=spatial_transform,
        random_shift=True,
        test_mode=False
        )
    

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn
                                               )

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(model_parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)

    return (train_loader, train_sampler, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    spatial_transform = [
        transforms.Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        transforms.ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    #spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)
    val_data = VideoFrameDataset(
        root_path=opt.root_path,
        annotationfile_path=opt.val_annotation_path,
        num_segments=1,
        frames_per_segment=opt.t_dim_frames,
        imagefile_template=opt.imagefile_template,
        transform=spatial_transform,
        random_shift=False,
        test_mode=False
        )
    
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn
                                             )

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger

#------ Check point save ------------#
def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)



if __name__=="__main__":
    opt = parse_opts()
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    opt.imagefile_template='{:010d}.jpg'
    #opt.mean = get_mean()
    opt.sample_size = (320,240)
    opt.t_dim_frames = 200
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    # opt.n_classes = 400
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    model = generate_model(opt)
    # if opt.resume_path is not None:
    #     model = resume_model(opt.resume_path, opt.arch, model)
    model = make_data_parallel(model, opt.device)
    if not opt.verbose:
        print(model)

    if opt.n_classes > 1:
        criterion = CrossEntropyLoss().to(opt.device)
    else:
        criterion = BCELoss().to(opt.device)
    
    parameters = model.parameters()
    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger,
         optimizer, scheduler) = get_train_utils(opt, parameters)
        # if opt.resume_path is not None:
        #     opt.begin_epoch, optimizer, scheduler = resume_train_utils(
        #         opt.resume_path, opt.begin_epoch, optimizer, scheduler)
        #     if opt.overwrite_milestones:
        #         scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None
    prev_val_loss = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer)

            if i % opt.checkpoint == 0 and opt.is_master_node:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer,)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)

    # class_names = []
    # with open('class_names_list') as f:
    #     for row in f:
    #         class_names.append(row[:-1])

    # ffmpeg_loglevel = 'quiet'
    # if opt.verbose:
    #     ffmpeg_loglevel = 'info'

    # if os.path.exists('tmp'):
    #     subprocess.call('rm -rf tmp', shell=True)

    # outputs = []
    # for input_file in input_files:
    #     video_path = os.path.join(opt.video_root, input_file)
    #     if os.path.exists(video_path):
    #         print(video_path)
    #         subprocess.call('mkdir tmp', shell=True)
    #         subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
    #                         shell=True)

    #         result = classify_video('tmp', input_file, class_names, model, opt)
    #         outputs.append(result)

    #         subprocess.call('rm -rf tmp', shell=True)
    #     else:
    #         print('{} does not exist'.format(input_file))

    # if os.path.exists('tmp'):
    #     subprocess.call('rm -rf tmp', shell=True)

    # with open(opt.output, 'w') as f:
    #     json.dump(outputs, f)