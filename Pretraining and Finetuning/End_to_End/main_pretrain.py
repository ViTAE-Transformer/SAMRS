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
from datasets import SegmentationDataset
from models import SemsegPretrnFramework
from utils import AverageMeter, intersectionAndUnionGPU

import cv2
import subprocess


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
parser.add_argument('--backbone', type=str, default=None, choices=['resnet50','swint','vitaev2_s','vit_b_rvsa','internimage_t'], help='backbone name')
parser.add_argument('--decoder', type=str, default=None, choices=['unet','unetpp','upernet','mask2former'], help='decoder name')
parser.add_argument('--datasets', type=str, nargs='+',help='used dataset')
# epoch
parser.add_argument('--start_epoch', type=int, default=0, help='index of start epoch')
parser.add_argument('--start_iter', type=int, default=0, help='index of start iteration')
parser.add_argument('--end_iter', type=int, default=5, help='number of epochs to train')

# batch size
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--batch_size_val', type=int, default=8, help='input batch size for validation')
parser.add_argument('--workers', type=int, default=0, help='workers num')

# learning rate
parser.add_argument('--lr', type=float, default=None, help='actual learning rate')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

# distributed
parser.add_argument('--distributed', type=str, default='True', choices=['True', 'False'], help='distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', type=int, default=0)

# ft
parser.add_argument('--ft', type=str, default='False', choices=['True', 'False'], help='finetune model')
parser.add_argument('--resume', type=str, default=None, help='dataset name')

# save
parser.add_argument('--save_path', type=str, default=None, help='path of saving model')

# ignored
parser.add_argument('--ignore_label', type=int, default=255, help='ignore index of loss')

# interval
parser.add_argument('--interval', default=5, type=int, help='valid interval')

# init_backbone
parser.add_argument('--init_backbone', type=str, default=None, choices=['imp', 'rsp', 'none', 'mae'], help='init model')

# input img size
parser.add_argument('--image_size', type=int, default=None, help='image size')

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

if main_process(args):
    logger.info('<<<<<<<<<<<<<<<<< args <<<<<<<<<<<<<<<<<')
    logger.info(args)

##################################################### augmentation #######################################################

IMAGE_SIZE = args.image_size

train_trfm = A.Compose([
    #A.RandomResizedCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(1.0, 1.0), p=1),
    A.RandomScale(scale_limit=(-0.5, 1.0), p=0.5),
    A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_label, p=1.0),
    A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
    #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(),
    A.OneOf([
        A.RandomContrast(),
        A.RandomGamma(),
        A.RandomBrightness(),
        A.ColorJitter(brightness=0.07, contrast=0.07,
                   saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        ], p=0.3)
])
val_trfm = A.Compose([
    A.CenterCrop(args.image_size, args.image_size),
    #A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomRotate90()
])

########################################################## dataset ####################################################

classes1 = 18
classes2 = 20
classes3 = 37

root = '/dataset/samrs/dotav2_1024/trainval/'
image_path = '/dataset/samrs/dotav2_1024/trainval/images/'
label_path = '/dataset/samrs/dotav2_1024/trainval/hbox_segs_init/gray/'
sota_trn_dataset = SegmentationDataset(args, IMAGE_SIZE, root, image_path, label_path, ext_img='.png', ext_lbl='.png', flag='trn', transform=train_trfm)
sota_val_dataset = SegmentationDataset(args, IMAGE_SIZE, root, image_path, label_path, ext_img='.png', ext_lbl='.png', flag='val', transform=val_trfm)

root = '/dataset/samrs/dior/'
image_path = '/dataset/samrs/dior/JPEGImages-trainval/'
label_path = '/dataset/samrs/dior/hbox_segs_trainvaltest_init/gray/'
sior_trn_dataset = SegmentationDataset(args, IMAGE_SIZE, root, image_path, label_path, ext_img='.jpg', ext_lbl='.png', flag='trn', transform=train_trfm)
sior_val_dataset = SegmentationDataset(args, IMAGE_SIZE, root, image_path, label_path, ext_img='.jpg', ext_lbl='.png', flag='val', transform=val_trfm)

root = '/dataset/samrs/fair1m_1024/trainval/'
image_path = '/dataset/samrs/fair1m_1024/trainval/images/'
label_path = '/dataset/samrs/fair1m_1024/trainval/rhbox_segs_init/gray/'
fast_trn_dataset = SegmentationDataset(args, IMAGE_SIZE, root, image_path, label_path, ext_img='.png', ext_lbl='.png', flag='trn', transform=train_trfm)
fast_val_dataset = SegmentationDataset(args, IMAGE_SIZE, root, image_path, label_path, ext_img='.png', ext_lbl='.png', flag='val', transform=val_trfm)

################################## sampler

if args.distributed=='True':
    train_sampler_sota = D.distributed.DistributedSampler(sota_trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
    train_sampler_sior = D.distributed.DistributedSampler(sior_trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
    train_sampler_fast = D.distributed.DistributedSampler(fast_trn_dataset, num_replicas=args.world_size,rank=args.rank)#分布式采样器
else:
    train_sampler_sota = None
    train_sampler_sior = None
    train_sampler_fast = None

if args.distributed=='True':
    val_sampler_sota = D.distributed.DistributedSampler(sota_val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
    val_sampler_sior = D.distributed.DistributedSampler(sior_val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
    val_sampler_fast = D.distributed.DistributedSampler(fast_val_dataset, num_replicas=args.world_size,rank=args.rank) # 分布式采样器
else:
    val_sampler_sota = None
    val_sampler_sior = None
    val_sampler_fast = None

############################### batch size 

batch_size = args.batch_size
batch_size_val = args.batch_size_val
workers = args.workers

all_img_num = 0
if 'sota' in args.datasets:
    all_img_num += 17480
    logger.info('Using SOTA dataset!')
if 'sior' in args.datasets:
    all_img_num += 11725
    logger.info('Using SIOR dataset!')
if 'fast' in args.datasets:
    all_img_num += 64147
    logger.info('Using FAST dataset!')

if 'sota' in args.datasets:
    batch_size_sota = int(batch_size * 17480 * 1.0 / all_img_num)
    batch_size_val_sota = int(batch_size_val * 17480 * 1.0 / all_img_num)
    workers_sota = int(workers * 17480 * 1.0 / all_img_num)
else:
    batch_size_sota = args.world_size
    batch_size_val_sota = args.world_size
    workers_sota = args.world_size

if 'sior' in args.datasets:
    batch_size_sior = int(batch_size * 11725 * 1.0 / all_img_num)
    batch_size_val_sior = int(batch_size_val * 11725 * 1.0 / all_img_num)
    workers_sior = int(workers * 11725 * 1.0 / all_img_num)
else:
    batch_size_sior = args.world_size
    batch_size_val_sior = args.world_size
    workers_sior = args.world_size

if 'fast' in args.datasets:
    batch_size_fast = int(batch_size * 64147 * 1.0 / all_img_num)
    batch_size_val_fast = int(batch_size_val * 64147 * 1.0 / all_img_num)
    workers_fast = int(workers * 64147 * 1.0 / all_img_num)
else:
    batch_size_fast = args.world_size
    batch_size_val_fast = args.world_size
    workers_fast = args.world_size

trn_loader_length = np.min([sota_trn_dataset.length / (batch_size_sota * args.world_size),
                          sior_trn_dataset.length / (batch_size_sior * args.world_size),
                          fast_trn_dataset.length / (batch_size_fast * args.world_size)])

val_loader_length = np.min([sota_val_dataset.length / (batch_size_val_sota),
                          sior_val_dataset.length / (batch_size_val_sior),
                          fast_val_dataset.length / (batch_size_val_fast)])

if main_process(args):
    logger.info('train data length: {}, {}, {}'.format(sota_trn_dataset.length, sior_trn_dataset.length, fast_trn_dataset.length))
    logger.info('train batch size: {}, {}, {}'.format(batch_size_sota*args.world_size, batch_size_sior*args.world_size, batch_size_fast*args.world_size))
    logger.info('train loader length: {}'.format(trn_loader_length))

    logger.info('valid data length: {}, {}, {}'.format(sota_val_dataset.length, sior_val_dataset.length, fast_val_dataset.length))
    logger.info('valid batch size: {}, {}, {}'.format(batch_size_val_sota, batch_size_val_sior, batch_size_val_fast))
    logger.info('valid loader length: {}'.format(val_loader_length))

################################### dataloaders

if args.decoder == 'mask2former':
    from utils import maskformer_collate
    train_loader_sota = D.DataLoader(
        sota_trn_dataset, batch_size=batch_size_sota, shuffle=(train_sampler_sota is None), 
        num_workers=workers_sota, pin_memory=True, collate_fn=maskformer_collate,
        sampler=train_sampler_sota, drop_last=True)
    train_loader_sior = D.DataLoader(
        sior_trn_dataset, batch_size=batch_size_sior, shuffle=(train_sampler_sior is None), 
        num_workers=workers_sior, pin_memory=True, collate_fn=maskformer_collate,
        sampler=train_sampler_sior, drop_last=True)
    train_loader_fast = D.DataLoader(
        fast_trn_dataset, batch_size=batch_size_fast, shuffle=(train_sampler_fast is None), 
        num_workers=workers_fast, pin_memory=True, collate_fn=maskformer_collate,
        sampler=train_sampler_fast, drop_last=True)

    valid_loader_sota = D.DataLoader(
        sota_val_dataset, batch_size=batch_size_val_sota, shuffle=False, 
        num_workers=workers_sota, pin_memory=True, collate_fn=maskformer_collate,
        sampler=val_sampler_sota)

    valid_loader_sior = D.DataLoader(
        sior_val_dataset, batch_size=batch_size_val_sior, shuffle=False, 
        num_workers=workers_sior, pin_memory=True, collate_fn=maskformer_collate,
        sampler=val_sampler_sior)

    valid_loader_fast = D.DataLoader(
        fast_val_dataset, batch_size=batch_size_val_fast, shuffle=False, 
        num_workers=workers_fast, pin_memory=True, collate_fn=maskformer_collate,
        sampler=val_sampler_fast)

##################################################### model #####################################################

################################## loss

criterion1 = nn.CrossEntropyLoss(ignore_index = args.ignore_label)
criterion2 = nn.CrossEntropyLoss(ignore_index = args.ignore_label)
criterion3 = nn.CrossEntropyLoss(ignore_index = args.ignore_label)

################################## model

model = SemsegPretrnFramework(args, classes1=classes1, classes2=classes2, classes3=classes3)

model.cuda(LOCAL_RANK)

losses = []

#####################################################  optimizer #####################################################

if 'resnet' in args.backbone:

    # if args.decoder == 'mask2former':

    #     from mmengine.optim import build_optim_wrapper

        

    #     embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
    #     optim_wrapper = dict(
    #         optimizer=dict(
    #         type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    #         clip_grad=dict(max_norm=0.01, norm_type=2),
    #         paramwise_cfg=dict(
    #             custom_keys={
    #                 'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    #                 'query_embed': embed_multi,
    #                 'query_feat': embed_multi,
    #                 'level_embed': embed_multi,
    #             },
    #             norm_decay_mult=0.0))
    #     optimizer = build_optim_wrapper(model, optim_wrapper)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.end_iter, eta_min=0, last_epoch=-1)
    #     args.blr = 1e-3

    # else:

    from mmengine.optim import OptimWrapper
    from mmengine.utils import is_list_of

    print(dist.get_world_size())

    args.blr = 1e-3

    if args.lr is None:  # only base_lr is specified
        if args.distributed == 'True':
            args.lr = args.blr * (args.batch_size * dist.get_world_size() / 96) # 累积iter, lr会增加
        else:
            args.lr = args.blr * (args.batch_size / 96) # 累积iter, lr会增加
    else:
        raise NotImplementedError
    
    optimizer = torch.optim.AdamW(model.parameters(),  lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_iter, eta_min=5e-6, last_epoch=-1)

        
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.end_iter, eta_min=0, last_epoch=-1)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.end_iter, eta_min=0, last_epoch=-1)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.end_iter, eta_min=0, last_epoch=-1)


## ft
if args.ft=='True':
    if os.path.isfile(args.resume):
        if main_process(args):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        # checkpoint = torch.load(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        if main_process(args):
            logger.info("=> loading ft model...")
        args.start_epoch = checkpoint['epoch']
        args.start_iter = checkpoint['iteration']
        ckpt_dict = checkpoint['state_dict']
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in ckpt_dict.items():
            if k in state_dict: #and k!='module.GraphReason.conv.0.block1.0.weight':
                model_dict[k] = v
        state_dict.update(model_dict)
        msg = model.load_state_dict(state_dict,strict=False)
        print(msg)
        #model.load_state_dict(checkpoint['state_dict'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        losses = checkpoint['loss_pretrain'].tolist()
        if main_process(args):
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        if main_process(args):
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

### 分布式
if args.distributed == 'True':
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],find_unused_parameters=True)
    if main_process(args):
        logger.info("Implementing distributed training!")
    seed = 2023 + LOCAL_RANK
    set_seeds(seed)
else:
    model.encoder = torch.nn.DataParallel(model.encoder)
    model.decoder = torch.nn.DataParallel(model.decoder)
    model.semseghead_1 = torch.nn.DataParallel(model.semseghead_1)
    model.semseghead_2 = torch.nn.DataParallel(model.semseghead_2)
    model.semseghead_3 = torch.nn.DataParallel(model.semseghead_3)
    model = torch.nn.DataParallel(model)#普通的单机多卡
    patch_replication_callback(model)
    if main_process(args):
        logger.info("Implementing parallel training!")

##################################################### validation #####################################################

@torch.no_grad()
def validation(args, logger, epoch, model, valid_loader_sota, valid_loader_sior, valid_loader_fast):

    intersection_meter1, intersection_meter2, intersection_meter3 = AverageMeter(), AverageMeter(), AverageMeter()
    union_meter1, union_meter2, union_meter3 = AverageMeter(), AverageMeter(), AverageMeter()
    target_meter1, target_meter2, target_meter3 = AverageMeter(), AverageMeter(), AverageMeter()
    predict_meter1, predict_meter2, predict_meter3 = AverageMeter(), AverageMeter(), AverageMeter()

    model.eval()

    for (i, ((x1, y1),(x2,y2),(x3,y3))) in enumerate(zip(valid_loader_sota, valid_loader_sior, valid_loader_fast)):

        if args.decoder == 'mask2former':
            x1, y1, x2, y2, x3, y3 = x1.cuda(LOCAL_RANK), y1, x2.cuda(LOCAL_RANK), y2, x3.cuda(LOCAL_RANK), y3
        else:
            x1, y1, x2, y2, x3, y3 = x1.cuda(LOCAL_RANK), y1.long().cuda(LOCAL_RANK), x2.cuda(LOCAL_RANK), y2.long().cuda(LOCAL_RANK), x3.cuda(LOCAL_RANK), y3.long().cuda(LOCAL_RANK)

        output = model.forward(x1, x2, x3, y1=y1, y2=y2, y3=y3)

        o1, o2, o3 = output

        if 'sota' in args.datasets:

            o1 = o1.max(1)[1]
            if args.decoder == 'mask2former':
                y1 = torch.cat([data_sample.gt_sem_seg.data for data_sample in y1]).long().cuda()
            intersection, union, target, predict = intersectionAndUnionGPU(o1, y1, classes1, args.ignore_label)
            if args.distributed=='True':
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(predict)
            intersection, union, target, predict = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), predict.cpu().numpy(),
            intersection_meter1.update(intersection), union_meter1.update(union), target_meter1.update(target), predict_meter1.update(predict)
        
        if 'sior' in args.datasets:

            o2 = o2.max(1)[1]
            if args.decoder == 'mask2former':
                y2 = torch.cat([data_sample.gt_sem_seg.data for data_sample in y2]).long().cuda()
            intersection, union, target, predict = intersectionAndUnionGPU(o2, y2, classes2, args.ignore_label)
            if args.distributed=='True':
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(predict)
            intersection, union, target, predict = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), predict.cpu().numpy(),
            intersection_meter2.update(intersection), union_meter2.update(union), target_meter2.update(target), predict_meter2.update(predict)            
        
        if 'fast' in args.datasets:

            o3 = o3.max(1)[1]
            if args.decoder == 'mask2former':
                y3 = torch.cat([data_sample.gt_sem_seg.data for data_sample in y3]).long().cuda()
            intersection, union, target, predict = intersectionAndUnionGPU(o3, y3, classes3, args.ignore_label)
            if args.distributed=='True':
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target), dist.all_reduce(predict)
            intersection, union, target, predict = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), predict.cpu().numpy(),
            intersection_meter3.update(intersection), union_meter3.update(union), target_meter3.update(target), predict_meter3.update(predict)

        logger.info('Valid epoch {}, sample {}/{}'.format(epoch, i, val_loader_length))

    mIoU1, mIoU2, mIoU3 = -1, -1, -1

    mIoUs = []

    if 'sota' in args.datasets:
        iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)

        # accuracy_class1 = intersection_meter1.sum / (target_meter1.sum + 1e-10)
        # precise_class1 = intersection_meter1.sum / (predict_meter1.sum + 1e-10)
        # F1_class1 = 2*(precise_class1*accuracy_class1) / (precise_class1+accuracy_class1)

        mIoU1 = np.mean(iou_class1)
        mIoUs.append(mIoU1)
        
        if main_process(args):
            logger.info("miou of SOTA validation set: {}".format(mIoU1))


    if 'sior' in args.datasets:
        iou_class2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
        # accuracy_class2 = intersection_meter2.sum / (target_meter2.sum + 1e-10)
        # precise_class2 = intersection_meter2.sum / (predict_meter2.sum + 1e-10)
        # F1_class2 = 2*(precise_class2*accuracy_class2) / (precise_class2+accuracy_class2)

        mIoU2 = np.mean(iou_class2)
        mIoUs.append(mIoU2)
        
        if main_process(args):
            logger.info("miou of SIOR validation set: {}".format(mIoU2))

    if 'fast' in args.datasets:
        iou_class3 = intersection_meter3.sum / (union_meter3.sum + 1e-10)
        # accuracy_class3 = intersection_meter3.sum / (target_meter3.sum + 1e-10)
        # precise_class3 = intersection_meter3.sum / (predict_meter3.sum + 1e-10)
        # F1_class3 = 2*(precise_class3*accuracy_class3) / (precise_class3+accuracy_class3)

        mIoU3 = np.mean(iou_class3)
        mIoUs.append(mIoU3)
        
        if main_process(args):
            logger.info("miou of FAST validation set: {}".format(mIoU3))

    mIoU = np.mean(np.array(mIoUs))

    model.train()

    return mIoU


##################################################### training #####################################################

best_acc = 0

iter = args.start_iter
epoch = args.start_epoch


while True:

    if args.distributed == 'True':
        train_sampler_sota.set_epoch(epoch)
        train_sampler_sior.set_epoch(epoch)
        train_sampler_fast.set_epoch(epoch)

    start_time = time.time()
    model.train()

    optimizer.zero_grad()
    for ((x1, y1), (x2, y2), (x3, y3)) in zip(train_loader_sota, train_loader_sior, train_loader_fast):

        if args.decoder == 'mask2former':
            x1, y1, x2, y2, x3, y3 = x1.cuda(LOCAL_RANK), y1, x2.cuda(LOCAL_RANK), y2, x3.cuda(LOCAL_RANK), y3
        else:
            x1, y1, x2, y2, x3, y3 = x1.cuda(LOCAL_RANK), y1.long().cuda(LOCAL_RANK), x2.cuda(LOCAL_RANK), y2.long().cuda(LOCAL_RANK), x3.cuda(LOCAL_RANK), y3.long().cuda(LOCAL_RANK)

        output = model.forward(x1, x2, x3, y1=y1, y2=y2, y3=y3)

        o1, o2, o3 = output

        loss = 0
        loss_ = ''

        if 'sota' in args.datasets:
            if args.decoder == 'mask2former':

                log_vars = []
                
                for loss_name, loss_value in o1.items():
                    if isinstance(loss_value, torch.Tensor):
                        log_vars.append([loss_name, loss_value.mean()])
                    elif is_list_of(loss_value, torch.Tensor):
                        log_vars.append(
                            [loss_name,
                            sum(_loss.mean() for _loss in loss_value)])
                    else:
                        raise TypeError(
                            f'{loss_name} is not a tensor or list of tensors')

                loss1 = sum(value for key, value in log_vars if 'loss' in key)

            else:
                loss1 = criterion1(o1, y1)
            loss +=  loss1
            loss_ += str(loss1.item())[:4] + ' '

        if 'sior' in args.datasets:
            if args.decoder == 'mask2former':

                log_vars = []
                
                for loss_name, loss_value in o2.items():
                    if isinstance(loss_value, torch.Tensor):
                        log_vars.append([loss_name, loss_value.mean()])
                    elif is_list_of(loss_value, torch.Tensor):
                        log_vars.append(
                            [loss_name,
                            sum(_loss.mean() for _loss in loss_value)])
                    else:
                        raise TypeError(
                            f'{loss_name} is not a tensor or list of tensors')

                loss2 = sum(value for key, value in log_vars if 'loss' in key)

            else:
                loss2 = criterion2(o2, y2)
            loss +=  loss2
            loss_ += str(loss2.item())[:4] + ' '

        if 'fast' in args.datasets:
            if args.decoder == 'mask2former':

                log_vars = []
                
                for loss_name, loss_value in o3.items():
                    if isinstance(loss_value, torch.Tensor):
                        log_vars.append([loss_name, loss_value.mean()])
                    elif is_list_of(loss_value, torch.Tensor):
                        log_vars.append(
                            [loss_name,
                            sum(_loss.mean() for _loss in loss_value)])
                    else:
                        raise TypeError(
                            f'{loss_name} is not a tensor or list of tensors')

                loss3 = sum(value for key, value in log_vars if 'loss' in key)

            else:
                loss3 = criterion2(o3, y3)

            loss +=  loss3
            loss_ += str(loss3.item())[:6]

        iter +=1

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        torch.cuda.synchronize()
        
        losses.append(loss.item())
        # print(loss.item())

        if main_process(args):
            logger.info('Train epoch {} iter {}/{}, lr: {:.7f} loss: {} sum: {:.2f}'.format(epoch+1, iter, args.end_iter, optimizer.param_groups[0]["lr"], loss_, loss.item()))

    #if (epoch - args.start_epoch) % args.interval == 0:

        if (iter > 10000 and iter % 5000 ==0) or (iter < 10000 and iter % 1000 ==0):

            logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

            start_time = time.time()
            vacc = validation(args, logger, epoch, model, valid_loader_sota, valid_loader_sior, valid_loader_fast)
            end_time = time.time()

            logger.info('Validation epoch {}, iter [{}/{}]: Average mIoU {:.4f}. Cost {:.4f} secs'.format(epoch+1, iter, args.end_iter, vacc, end_time-start_time))

            logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

            if vacc > best_acc:
                best_acc = vacc
                if main_process(args):
                    filename = args.save_path + '/best_{}_{}_pretrn_model.pth'.format(args.backbone, args.decoder)
                    logger.info('Saving epoch {} checkpoint to: {}'.format(epoch,filename))
                    torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                                filename)
                    filename = args.save_path + '/best_{}_{}_pretrn_model_encoder.pth'.format(args.backbone, args.decoder)
                    torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                                filename)            
                if main_process(args):
                    print("best acc is {}".format(best_acc))
                    
        scheduler.step()

        if iter >= args.end_iter:
            break

    if iter >= args.end_iter:
        break

    epoch +=1

########### last validation

logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

start_time = time.time()
vacc = validation(args, logger, epoch, model, valid_loader_sota, valid_loader_sior, valid_loader_fast)
end_time = time.time()

logger.info('Last: validation epoch {}, iter [{}/{}]: Average mIoU {:.4f}. Cost {:.4f} secs'.format(epoch+1, iter, args.end_iter, vacc, end_time-start_time))

logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

if main_process(args):
    print("last acc is {}".format(vacc))

if main_process(args):            
    filename = args.save_path + '/last_{}_{}_pretrn_model.pth'.format(args.backbone, args.decoder)
    torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                        filename)
    filename = args.save_path + '/last_{}_{}_pretrn_model_encoder.pth'.format(args.backbone, args.decoder)
    torch.save({'epoch': epoch, 'iteration': iter, 'state_dict': model.module.encoder.state_dict(), 'optimizer': optimizer.state_dict(),'scheduler':scheduler.state_dict(), 'loss_pretrain': np.array(losses)},
                            filename)

logger.info('################# Pretrain model save finished! ######################')