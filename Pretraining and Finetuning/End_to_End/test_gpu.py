import os
import argparse

import numpy as np
import cv2
from tqdm import tqdm

import time, datetime
import torch
import logging

import torch.utils.data as D
import torch.nn.functional as F
from models import SemsegFinetuneFramework
from utils import AverageMeter, intersectionAndUnion, colorize
from datasets import ISPRSDataset, ISAIDDataset

from PIL import Image

#IMAGE_SIZE = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
parser.add_argument('--backbone', type=str, default=None, choices=['resnet50','vitaev2','vitaev2_3', 'vit_b_rvsa'], help='backbone name')
parser.add_argument('--decoder', type=str, default=None, choices=['unet','unetpp','upernet','mask2former'], help='decoder name')
parser.add_argument('--dataset', type=str, default=None, choices=['potsdam', 'vaihingen', 'isaid'], help='used dataset')

parser.add_argument('--workers', type=int, default=2, help='workers num')
parser.add_argument('--ms', type=str, default='False', choices=['True','False'], help='multiscale')

parser.add_argument('--resume', type=str, default=None, help='resume name')
parser.add_argument('--save_path', type=str, default=None, help='path of saving model')
parser.add_argument('--load', type=str, default=None, choices=['backbone','network'], help='loaded model part')

# mode
parser.add_argument('--mode', type=str, default=None, choices=['val','test','inf'], help='testing mode')

# ignored
parser.add_argument('--ignore_label', type=int, default=255, help='ignore index of loss')

# init_backbone
parser.add_argument('--init_backbone', type=str, default=None, choices=['imp', 'rsp'], help='init model')

# input img size
parser.add_argument('--image_size', type=int, default=-1, help='image size')

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

############################ scale

if args.ms == 'True':
    scales=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    #args.scales = [0.75, 1.0, 1.25]
else:
    scales = [1.0]

################################################### dataset ############################################

############################ dataset

if args.image_size == -1:

    if args.dataset == 'potsdam':
        IMAGE_SIZE = 512
    elif args.dataset == 'vaihingen':
        IMAGE_SIZE = 512
    elif args.dataset == 'isaid':
        IMAGE_SIZE = 896
    else:
        raise NotImplementedError
    
    args.image_size = IMAGE_SIZE

else:
    IMAGE_SIZE = args.image_size

args.crop_h = IMAGE_SIZE
args.crop_w = IMAGE_SIZE


if args.dataset == 'potsdam':
    classes = 5
    crop_h = IMAGE_SIZE
    crop_w = IMAGE_SIZE
    root = '/work/share/achk2o1zg1/diwang22/dataset/potsdam_rgb_dataset/'
    tes_dataset = ISPRSDataset(args, img_size = None, split='test', data_root=root, transform=None)
elif args.dataset == 'vaihingen':
    classes = 5
    crop_h = IMAGE_SIZE
    crop_w = IMAGE_SIZE
    root = ''
    tes_dataset = ISPRSDataset(args, img_size = None, split='test', data_root=root, transform=None)
elif args.dataset == 'isaid':
    classes = 16
    crop_h = IMAGE_SIZE #896
    crop_w = IMAGE_SIZE #896
    root = '/work/share/achk2o1zg1/diwang22/dataset/isaid_patches/'
    tes_dataset = ISAIDDataset(args, img_size = None, split='test', data_root=root, transform=None)
else:
    raise NotImplementedError


if args.decoder == 'mask2former':
    from utils import maskformer_collate
    test_loader = D.DataLoader(tes_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=maskformer_collate)

############################ save_path or color

gray_folder = os.path.join(args.save_path, 'gray')
color_folder = os.path.join(args.save_path, 'color')

os.makedirs(gray_folder, exist_ok=True)
os.makedirs(color_folder, exist_ok=True)

if args.dataset == 'potsdam' or args.dataset == 'vaihingen':
    from utils import ISPRS_CLASSES, ISPRS_PALETTE
    colors = ISPRS_PALETTE
    names = ISPRS_CLASSES
elif args.dataset == 'isaid':
    from utils import ISAID_CLASSES, ISAID_PALETTE
    colors = ISAID_PALETTE
    names = ISAID_CLASSES

#colors = np.array(colors)
############################################## model ###############################################

model = SemsegFinetuneFramework(args, classes=classes)
############################ resume

if os.path.isfile(args.resume):
    checkpoint = torch.load(args.resume)
    # print('############# model ###########')
    # print(model.state_dict().keys())
    # print('############ ckpt #############')
    # print(checkpoint['state_dict'].keys())
    logger.info("=> loaded checkpoint '{}'".format(args.resume))
    msg = model.load_state_dict(checkpoint['state_dict'])
    print('##########',msg)
else:
    raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

model = torch.nn.DataParallel(model.to(DEVICE))
############################################## testing ###############################################

def net_process(model, image, meta, flip=True):

    if flip:
        input = torch.cat([image, image.flip(3)], 0)
        input_meta = meta + meta
    with torch.no_grad():
        output = model(input, y=input_meta)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        raise NotImplementedError
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.permute(1, 2, 0)
    return output


def scale_process(model, image, meta, classes, crop_h, crop_w, h, w, stride_rate=2/3):
    _, _, ori_h, ori_w = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    if pad_h > 0 or pad_w > 0:
        #print(mean)

        y=torch.zeros([image.shape[0],3, ori_h+pad_h, ori_w+pad_w]).cuda()
        for k in range(3):
            y[:,[k],:,:]=F.pad(image[:,[k],:,:], (0, pad_w, 0, pad_h), mode='constant', value=0)

        image = y

    _, _, new_h, new_w = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = torch.zeros([new_h, new_w, classes], dtype=torch.float).cuda()
    count_crop = torch.zeros([new_h, new_w], dtype=float).cuda()
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].clone()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, meta)
    prediction_crop /= torch.unsqueeze(count_crop, 2)
    prediction_crop = prediction_crop[:ori_h, :ori_w].permute(2,0,1).unsqueeze(0)
    prediction = F.interpolate(prediction_crop, (h, w), mode='bilinear')
    prediction = prediction.squeeze(0).permute(1, 2, 0)
    return prediction


logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

intersection_meter = AverageMeter()
union_meter = AverageMeter()
target_meter = AverageMeter()
predict_meter = AverageMeter()

model.eval()

model = model.module

data_list = tes_dataset.data_list

start_time = time.time()

for i, (image, meta) in enumerate(test_loader):
    image = image.cuda()
    _, _, h, w = image.shape
    prediction = torch.zeros([h, w, classes], dtype=torch.float).cuda()
    for scale in scales:
        image_scale = F.interpolate(image, scale_factor=scale, mode = 'bilinear')
        prediction += scale_process(model, image_scale, meta, classes, crop_h, crop_w, h, w)
    prediction /= len(scales)
    prediction = torch.max(prediction, dim=2)[1].cpu().numpy()
    gray = np.uint8(prediction)

    image_path = data_list[0][i]
    image_name = image_path.split('/')[-1].split('.')[0]

    if args.mode != 'inf':

        target = cv2.imread(data_list[1][i], cv2.IMREAD_GRAYSCALE)
        target = np.array(target, dtype=np.int32)
        target = tes_dataset.tes_class_to_trainid(target) # gt 转为 trainid

        intersection, union, target, predict = intersectionAndUnion(gray, target, classes, ignore_index=args.ignore_label)

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        predict_meter.update(predict)

        if args.dataset == 'isaid':
            accuracy = sum(intersection_meter.val[1:]) / (sum(target_meter.val[1:]) + 1e-10)
        else:
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info(
            'Evaluating {}/{} on image {}, accuracy {:.4f}.'.format(i + 1, len(data_list[0]), image_name + '.png',
                                                                        accuracy))
                                                                        
    else:

        logger.info(
            'Inference {}/{} on image {}.'.format(i + 1, len(data_list[0]), image_name + '.png'))

    #color = colorize(gray, colors)#保存的是train_id对应的颜色图

    color = np.zeros([h, w, 3], dtype=np.uint8)

    for k,v in colors.items():
        m = gray==k
        color[m] = v
    
    color = Image.fromarray(color)

    gray_path = os.path.join(gray_folder, image_name + '.png')
    color_path = os.path.join(color_folder, image_name + '.png')

    #color.save(color_path)

    gray = tes_dataset._trainid_to_class(gray) # trainid to classes
    if args.mode == 'inf':
        cv2.imwrite(gray_path, np.uint8(gray))


end_time = time.time()

logger.info('Testing time: {}'.format(str(datetime.timedelta(seconds=int(end_time-start_time)))))

if args.mode != 'inf':
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)

    if args.dataset == 'isaid':
        mIoU = np.nanmean(iou_class[1:])
        mAcc = np.nanmean(accuracy_class[1:])
        mF1 = np.nanmean(F1_class[1:])
        allAcc = sum(intersection_meter.sum[1:]) / (sum(target_meter.sum[1:]) + 1e-10)
    else:
        mIoU = np.nanmean(iou_class)
        mAcc = np.nanmean(accuracy_class)
        mF1 = np.nanmean(F1_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc, mF1))

    for i in range(classes):
        logger.info(
            'Class_{} result: iou/f1/accuracy {:.4f}/{:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], F1_class[i], accuracy_class[i],
                                                                                names[i]))

logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')