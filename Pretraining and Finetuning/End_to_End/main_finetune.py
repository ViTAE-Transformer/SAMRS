#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import os, random, time
from tqdm import tqdm

import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import torch.distributed as dist

import logging
from sync_batchnorm import patch_replication_callback
from datasets import ISPRSDataset, ISAIDDataset
from models import SemsegFinetuneFramework
from utils import AverageMeter, intersectionAndUnionGPU

import cv2

import subprocess

#EPOCHES = 120
#BATCH_SIZE = 16
#IMAGE_SIZE = 512

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
parser.add_argument('--backbone', type=str, default=None, choices=['resnet50','swint','vitaev2_s','vit_b_rvsa','internimage_t'], help='backbone name')
parser.add_argument('--decoder', type=str, default=None, choices=['unet','unetpp','upernet','mask2former'], help='decoder name')
parser.add_argument('--dataset', type=str, default=None, choices=['potsdam', 'vaihingen', 'isaid'], help='used dataset')

# epoch
parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')

# batch size
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--batch_size_val', type=int, default=8, help='input batch size for validation')
parser.add_argument('--workers', type=int, default=0, help='workers num')

# distributed
parser.add_argument('--distributed', type=str, default='True', choices=['True', 'False'], help='distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', type=int, default=0)

# ft: continue training
parser.add_argument('--ft', type=str, default='False', choices=['True', 'False'], help='finetune model')

# must have resume
parser.add_argument('--resume', type=str, default=None, help='resume name')

# save
parser.add_argument('--save_path', type=str, default=None, help='path of saving model')

# ignored
parser.add_argument('--ignore_label', type=int, default=255, help='ignore index of loss')

# interval
parser.add_argument('--interval', default=5, type=int, help='valid interval')

# init_backbone

parser.add_argument('--init_backbone', type=str, default=None, choices=['none','imp', 'rsp', 'beit', 'mae', 'samrs-mae-expand'], help='init model')

# optim
parser.add_argument('--optim', type=str, default=None, choices=['adamw', 'sgd'], help='optim')

# input img size
parser.add_argument('--image_size', type=int, default=-1, help='image size')

# port 
parser.add_argument('--port', type=str, default=None, help='master ports')

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

logger_name = "main-logger"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'), mode='a')
log_format = '%(asctime)s %(message)s'
fh.setFormatter(logging.Formatter(log_format))
logger.addHandler(fh)

handler = logging.StreamHandler()
fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
handler.setFormatter(logging.Formatter(fmt))
logger.addHandler(handler)

def main_process(args):
    return not args.distributed == 'True' or (args.distributed == 'True' and args.rank % args.world_size == 0)

def set_seeds(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()

################################################### setting ###################################################

if args.distributed == 'True':

    #if 'MASTER_ADDR' not in os.environ:
    if 'SLURM_NTASKS' in os.environ.keys():
        logger.info('#################### srun for DDP! ############################')
        #torch.multiprocessing.set_start_method('spawn')
        args.world_size = int(os.environ['SLURM_NTASKS'])
        #args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        args.rank = int(os.environ['SLURM_PROCID']) #if 'RANK' in os.environ else 0
        #args.rank = int(os.environ["RANK"])
        #args.rank = dist.get_rank()
        LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
        #LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        #LOCAL_RANK = args.rank % torch.cuda.device_count()
        #IP = os.environ['SLURM_STEP_NODELIST']
        #DIST_URL = 'tcp://' + IP + ':' + str(port)
        torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
        #os.environ['MASTER_PORT'] = args.port
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
        #os.environ['MASTER_ADDR'] = addr
        dist_url = 'tcp://%s:%s' % (addr, args.port)
        dist.init_process_group(backend='nccl', init_method=dist_url, world_size=args.world_size, rank=args.rank)#分布式TCP初始化

    else:
        logger.info('#################### Launch for DDP! ############################')
    #     args.world_size = int(os.environ['SLURM_NTASKS'])
        args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    #     #args.rank = int(os.environ['SLURM_PROCID']) #if 'RANK' in os.environ else 0
        args.rank = int(os.environ["RANK"])
    #     #args.rank = dist.get_rank()
    #     #LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    #     #LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        LOCAL_RANK = args.rank % torch.cuda.device_count()
    #     #IP = os.environ['SLURM_STEP_NODELIST']
    #     #DIST_URL = 'tcp://' + IP + ':' + str(port)
        torch.cuda.set_device(LOCAL_RANK)  # 设置节点等级为GPU数
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)#分布式TCP初始化

    assert torch.distributed.is_initialized()

if main_process(args):
    logger.info('<<<<<<<<<<<<<<<<< args <<<<<<<<<<<<<<<<<')
    logger.info(args)

##################################################### augmentation #######################################################

if args.image_size == -1:

    if args.dataset == 'potsdam':
        IMAGE_SIZE = 512
        root = '/dataset/potsdam_rgb_dataset'
    elif args.dataset == 'vaihingen':
        IMAGE_SIZE = 512
        root = ''
    elif args.dataset == 'isaid':
        IMAGE_SIZE = 896
        root = ''
    else:
        raise NotImplementedError

    args.image_size = IMAGE_SIZE

else:
    IMAGE_SIZE = args.image_size

train_trfm = A.Compose([
    #A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.5, 2.0), p=1),
    A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_label, p=1.0),
    A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    #A.RandomCrop(NEW_SIZE*3, NEW_SIZE*3),
    #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90()
#    A.OneOf([
#        A.RandomContrast(),
#        A.RandomGamma(),
#        A.RandomBrightness(),
#        A.ColorJitter(brightness=0.07, contrast=0.07,
#                   saturation=0.1, hue=0.1, always_apply=False, p=0.3),
#        ], p=0.3),
])
val_trfm = A.Compose([
    #A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomRotate90()
])

########################################################## dataset ####################################################

if args.dataset == 'potsdam':
    classes = 5
    root = '/work/share/achk2o1zg1/diwang22/dataset/potsdam_rgb_dataset/'
    trn_dataset = ISPRSDataset(args, img_size = IMAGE_SIZE, split='train', data_root=root, transform=train_trfm)
    val_dataset = ISPRSDataset(args, img_size = IMAGE_SIZE, split='val', data_root=root, transform=val_trfm)
elif args.dataset == 'vaihingen':
    classes = 5
    root = ''
    trn_dataset = ISPRSDataset(args, img_size = IMAGE_SIZE, split='train', data_root=root, transform=train_trfm)
    val_dataset = ISPRSDataset(args, img_size = IMAGE_SIZE, split='val', data_root=root, transform=val_trfm)
elif args.dataset == 'isaid':
    classes = 16
    root = '/work/share/achk2o1zg1/diwang22/dataset/isaid_patches/'
    trn_dataset = ISAIDDataset(args, img_size = IMAGE_SIZE, split='train', data_root=root, transform=train_trfm)
    val_dataset = ISAIDDataset(args, img_size = IMAGE_SIZE, split='val', data_root=root, transform=val_trfm)
else:
    raise NotImplementedError

################################## sampler

if args.distributed=='True':
    train_sampler = D.distributed.DistributedSampler(trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
else:
    train_sampler = None

if args.distributed=='True':
    val_sampler = D.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
else:
    val_sampler = None

################################### batch

batch_size = args.batch_size
batch_size_val = args.batch_size_val
workers = args.workers

################################### dataloaders
if args.decoder == 'mask2former':
    from utils import maskformer_collate

    train_loader = D.DataLoader(
        trn_dataset, batch_size=batch_size, shuffle=(train_sampler is None), 
        num_workers=workers, pin_memory=True, collate_fn=maskformer_collate,
        sampler=train_sampler, drop_last=True)

    valid_loader = D.DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=False, 
        num_workers=workers, pin_memory=True, collate_fn=maskformer_collate,
        sampler=val_sampler)

##################################################### model #####################################################

################################## loss

criterion = nn.CrossEntropyLoss(ignore_index = args.ignore_label)

################################## model

model = SemsegFinetuneFramework(args, classes=classes)

#################### load checkpoint (backbone or whole model)

# if args.load == 'backbone':
#     if args.backbone == 'resnet50':
#         #backbone_ckpt = torch.load(args.resume, map_location='cpu')
#         model.encoder.init_weights(args.resume)
#         print('Use resnet50!')
#     elif args.backbone == 'vitaev2':
#         backbone_ckpt = torch.load(args.resume, map_location='cpu')
#         model.encoder.init_weights(backbone_ckpt)
#         print('Use vitaev2 with dpr=0.1!')
#     elif args.backbone == 'vitaev2_3':
#         backbone_ckpt = torch.load(args.resume, map_location='cpu')
#         model.encoder.init_weights(backbone_ckpt)    
#         print('Use vitaev2 with dpr=0.3!')
#     elif args.backbone == 'vitae_rvsa':
#         backbone_ckpt = torch.load(args.resume, map_location='cpu')
#         model.encoder.init_weights(backbone_ckpt)
#         print('Use vitae_rvsa!')
#     else:
#         raise NotImplementedError
# else:
if args.load == 'network':
    if os.path.isfile(args.resume):
        if main_process(args):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        if main_process(args):
            logger.info("=> loading ft model...")
        ckpt_dict = checkpoint['state_dict']

        if 'vit_b' in args.backbone:

            if list(ckpt_dict.keys())[0].startswith('module.'):
                ckpt_dict = {k[7:]: v for k, v in ckpt_dict.items()}

            from mmengine.dist import get_dist_info

            rank, _ = get_dist_info()
            if 'encoder.pos_embed' in ckpt_dict:
                pos_embed_checkpoint = ckpt_dict['encoder.pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                H, W = model.encoder.patch_embed.patch_shape
                num_patches = model.encoder.patch_embed.num_patches
                num_extra_tokens = 0
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    if rank == 0:
                        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
                    # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    ckpt_dict['encoder.pos_embed'] = new_pos_embed
    
                else:
                    ckpt_dict['encoder.pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

        if 'rvsa' in args.backbone:

            if list(ckpt_dict.keys())[0].startswith('module.'):
                ckpt_dict = {k[7:]: v for k, v in ckpt_dict.items()}

            #model_dict = model.state_dict()

            # import warnings

            # for k,v in ckpt_dict.items():
            #     if 'rel_pos_h' in k or 'rel_pos_w' in k:
            #         L1, C1 = ckpt_dict[k].shape
            #         L2, C2 = model_dict[k].shape

            #         if C1 != C2:
            #             warnings.warn(f'Error in loading {k}, pass')
            #         elif L1 != L2:
            #             table_pretrained_resized = F.interpolate(
            #                 ckpt_dict[k].permute(1, 0).reshape(1, 1, C1, L1).permute(0, 2, 3, 1),
            #                 size=(L2, 1),
            #                 mode='bicubic').permute(0,3,1,2)
            #             ckpt_dict[k] = table_pretrained_resized.view(
            #                 C2, L2).permute(1, 0).contiguous()

        msg = model.load_state_dict(ckpt_dict, strict=False)
        print('&&&&&&&&&&&&&&&&&&&', msg)
        if main_process(args):
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        if main_process(args):
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


model.cuda()

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

if main_process(args):
    logger.info('number of finetune params (M): %.2f' % (n_parameters / 1.e6))

################################## optimizer and scheduler

if 'resnet' in args.backbone:
    # if args.optim == 'adamw':
    #     optimizer = torch.optim.AdamW(model.parameters(),  lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=5e-6, last_epoch=-1)
    # elif args.optim == 'sgd':
    #     optimizer = torch.optim.SGD(model.parameters(),  lr=1e-3, momentum=0.9, weight_decay=1e-4)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    # else:
    #     raise NotImplementedError
    from mmengine.optim import OptimWrapper
    from mmengine.utils import is_list_of
    from mmengine.optim import build_optim_wrapper

    #print(dist.get_world_size())

    # optimizer = torch.optim.AdamW(model.parameters(),  lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=5e-6, last_epoch=-1)

    embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
    optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(
        type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
        clip_grad=dict(max_norm=0.01, norm_type=2),
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(lr_mult=0.1, decay_mult=1.0),
                'query_embed': embed_multi,
                'query_feat': embed_multi,
                'level_embed': embed_multi,
            },
            norm_decay_mult=0.0))
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)


elif 'swint' in args.backbone or 'vitaev2' in args.backbone:
    from mmengine.optim import build_optim_wrapper
    # AdamW optimizer, no weight decay for position embedding & layer norm
    # in backbone
    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
        paramwise_cfg=dict(
            custom_keys={
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'norm': dict(decay_mult=0.)
            }))
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)

elif 'vit_' in args.backbone:
    from mmengine.optim import build_optim_wrapper
    from mmcv_custom.layer_decay_optimizer_constructor_vit import *
    # AdamW optimizer, no weight decay for position embedding & layer norm in backbone
    
    optim_wrapper = dict(
            optimizer=dict(
            type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
            constructor='LayerDecayOptimizerConstructor_ViT', 
            paramwise_cfg=dict(
                num_layers=12, 
                layer_decay_rate=0.9,
                )
                )
    
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)

elif 'internimage' in args.backbone:
    from mmengine.optim import build_optim_wrapper
    from mmcv_custom.custom_layer_decay_optimizer_constructor import *
    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
        constructor='CustomLayerDecayOptimizerConstructor_InternImage',
        paramwise_cfg=dict(num_layers=30, 
                        layer_decay_rate=1.0,
                        depths=[4, 4, 18, 4]
                        )
                        )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.epochs, eta_min=0, last_epoch=-1)

################################## distributed
if args.distributed == 'True':
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],find_unused_parameters=True)

    if main_process(args):
        logger.info("Implementing distributed training!")
    seed = 2023 + LOCAL_RANK
    set_seeds(seed)
else:
    model = torch.nn.DataParallel(model)#普通的单机多卡
    patch_replication_callback(model)
    if main_process(args):
        logger.info("Implementing parallel training!")

################################# ft & continue train

losses = []

if args.ft=='True':
    if os.path.isfile(args.resume):
        if main_process(args):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losses = checkpoint['loss_finetune'].tolist()
        if main_process(args):
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        if main_process(args):
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


##################################################### validation #####################################################

@torch.no_grad()
def validation(args, model, valid_loader):

    intersection_meter = AverageMeter()
    union_meter  = AverageMeter()
    target_meter  = AverageMeter()
    predict_meter = AverageMeter()

    model.eval()

    for (x, y) in valid_loader:

        if args.decoder == 'mask2former':
            x, y = x.to(DEVICE), y
        else:
            x, y = x.to(DEVICE), y.long().to(DEVICE)

        o = model.forward(x, y=y)

        o = o.max(1)[1]
        if args.decoder == 'mask2former':
            y = torch.cat([data_sample.gt_sem_seg.data for data_sample in y]).long().cuda()
        intersection, union, target, predict = intersectionAndUnionGPU(o, y, classes, args.ignore_label)
        if args.distributed=='True':
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(predict)
        intersection, union, target, predict = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), predict.cpu().numpy(),
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target), predict_meter.update(predict)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)

    if args.dataset == 'isaid':
        mIoU = np.mean(iou_class[1:])
        mAcc = np.mean(accuracy_class[1:])
        mF1 = np.mean(F1_class[1:])
        allAcc = sum(intersection_meter.sum[1:]) / (sum(target_meter.sum[1:]) + 1e-10)
    else:
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        mF1 = np.mean(F1_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc


##################################################### training #####################################################

best_acc = 0

for epoch in range(args.start_epoch, args.epochs):

    if args.distributed == 'True':
        train_sampler.set_epoch(epoch)

    start_time = time.time()
    model.train()

    for i, (x, y) in enumerate(train_loader):

        # we use a per iteration (instead of per epoch) lr scheduler

        if args.decoder == 'mask2former':
            x, y = x.to(DEVICE), y
        else:
            x, y = x.to(DEVICE), y.long().to(DEVICE)

        o = model.forward(x, y=y)

        if args.decoder == 'mask2former':

            log_vars = []

            for loss_name, loss_value in o.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars.append([loss_name, loss_value.mean()])
                elif is_list_of(loss_value, torch.Tensor):
                    log_vars.append(
                        [loss_name,
                        sum(_loss.mean() for _loss in loss_value)])
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            loss = sum(value for key, value in log_vars if 'loss' in key)

        else:
            loss = criterion(o, y)

        optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        if main_process(args):
            logger.info('Training epoch [{}/{}] iter [{}/{}]: loss {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), loss.item()))

        losses.append(loss.item())

    if (epoch - args.start_epoch) % args.interval == 0:

        if main_process(args):

            logger.info('>>>>>>>>>>>>>>>> Start Evaluation of Finetune >>>>>>>>>>>>>>>>')

        start_time = time.time()
        mIoU, mAcc, mF1, allAcc = validation(args, model, valid_loader)
        end_time = time.time()

        if main_process(args):
            logger.info('Validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs'.format(epoch+1, args.epochs, mIoU, mAcc, mF1, allAcc, end_time-start_time))
            logger.info('<<<<<<<<<<<<<<<<< End Evaluation of Finetune <<<<<<<<<<<<<<<<<')

        if mIoU > best_acc:
            best_acc = mIoU
            if main_process(args):
                filename = args.save_path + '/best_{}_{}_finetune_model.pth'.format(args.backbone, args.decoder)
                logger.info('Saving epoch {} checkpoint to: {}'.format(epoch,filename))
                torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_finetune':np.array(losses)},
                            filename)
                filename = args.save_path + '/best_{}_{}_finetune_model_encoder.pth'.format(args.backbone, args.decoder)
                torch.save({'epoch': epoch, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_finetune':np.array(losses)},
                            filename)            
            if main_process(args):
                print("best acc is {}".format(best_acc))

    scheduler.step()

########### last validation

if main_process(args):

    logger.info('>>>>>>>>>>>>>>>> Start Evaluation of Finetune >>>>>>>>>>>>>>>>')

start_time = time.time()
mIoU, mAcc, mF1, allAcc = validation(args, model, valid_loader)
end_time = time.time()

if main_process(args):
    logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs'.format(epoch+1, args.epochs, mIoU, mAcc, mF1, allAcc, end_time-start_time))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation of Finetune <<<<<<<<<<<<<<<<<')

if main_process(args):
    print("last acc is {}".format(mIoU))    

if main_process(args):
    filename = args.save_path + '/last_{}_{}_finetune_model.pth'.format(args.backbone, args.decoder)
    torch.save({'epoch': epoch, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_finetune':np.array(losses)},
                        filename)
    filename = args.save_path + '/last_{}_{}_finetune_model_encoder.pth'.format(args.backbone, args.decoder)
    torch.save({'epoch': epoch, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_finetune':np.array(losses)},
                            filename)

logger.info('################# Fine tune model save finished! ######################')