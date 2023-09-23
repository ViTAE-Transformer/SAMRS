import os
import argparse
import json
import pickle
import torch
import numpy as np
#from utils import show_box, show_mask, show_hbox_mask
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from loaddata import load_hrsc, load_dota, load_dior
from PIL import Image
from instance_to_json import binary_to_coco_gt_hrsc, binary_to_coco_pre_hrsc
import cv2
from mapping import MAPPING, DOTA2_0, DIOR

from pycocotools import mask as maskUtils

def show_hbox_mask(point, box, mask, ax, random_color='None', color=None):
    if random_color == 'True':
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif random_color == 'False':
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    else:
        c = np.concatenate([color / 255,np.array([0.4])],axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * c.reshape(1, 1, -1)
    ax.imshow(mask_image)
    c = np.concatenate([color / 255,np.array([1])],axis=0)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=c,
                               facecolor=(0, 0, 0, 0), lw=2))
    #ax.scatter(point[0],point[1],marker='*',s=200, color=np.array([1,1,1,1]), edgecolor=np.array([1,1,1,1]), linewidth=2)
    #ax.scatter(point[0],point[1],marker='*',s=200, color=np.array([1,0,0,1]), edgecolor=np.array([1,0,0,1]), linewidth=2)

parser = argparse.ArgumentParser(description="det2seg")
## dataset setting

parser.add_argument('--dataset', type=str, default='dior',
                    choices=['dota','hrsc','dior'],
                    help='detection annotation type')
parser.add_argument('--instance', type=str, default='False',
                     choices=['True','False'],
                     help='visualization')
parser.add_argument('--semantic', type=str, default='True',
                    choices=['True','False'],
                    help='visualization')
parser.add_argument('--show', type=str, default='False',
                    choices=['True','False'],
                    help='visualization')

args = parser.parse_args()

if args.dataset == 'dota':
    lbl2cls = {k:v for k,v in enumerate(DOTA2_0)}
elif args.dataset == 'dior':
    lbl2cls = {k:v for k,v in enumerate(DIOR)}


EXE_TYPE = ".jpg"

file_dir = "/root/dataset/dior/JPEGImages-test/"

img_dir = "/root/dataset/dior/JPEGImages-test/"
ann_dir = "/root/dataset/dior/Annotations/Horizontal Bounding Boxes/"

# for ins
labeld_dir = "/root/dataset/HRSC2016/FullDataSet/LandMask/"
json_dir = "/root/dw/samrs/work_dir/hrsc/json/"

# for seg
save_dir = "/root/dataset/dior/hbox_segs_test_init/"

os.makedirs(os.path.join(save_dir,'gray'), exist_ok=True)
os.makedirs(os.path.join(save_dir,'color'), exist_ok=True)
os.makedirs(os.path.join(save_dir,'ins'), exist_ok=True)

# for vis
vis_dir = "/root/dw/samrs/test/dota_demo/vis/"

os.makedirs(vis_dir, exist_ok=True)

sam_checkpoint = "/root/dw/pretrn/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device=device)
predictor = SamPredictor(sam)

batch_size = 20

if __name__ == '__main__':

    files = os.listdir(file_dir)

    print('{} dataset contains {} images'.format(args.dataset, len(files)))

    all_labels = []
    all_masks = []
    all_colors = []
    all_names = []
    all_probs = []
    all_points = []
    all_hboxes = []
    all_hboxmasks = []

    cnt = 0

    for file in files:

        filename = os.path.basename(file)
        img_name = filename[:-4]
        img = np.array(Image.open(os.path.join(img_dir,img_name + EXE_TYPE)))
        hbox_masks = []

        if args.dataset == 'hrsc':
            gt_hboxes, _, gt_colors, gt_points, gt_labels, error = load_hrsc(img_name, ann_dir)
            # gt_colors: b,3
        if args.dataset == 'dota':
            gt_hboxes, _, gt_points, gt_labels, error = load_dota(img_name, ann_dir)

        if args.dataset == 'dior':
            gt_hboxes, gt_points, gt_labels, error = load_dior(img_name, ann_dir)

        if error==1: # must have box
            sam_masks = np.zeros(img.shape[:2])
            seg_mask = 255*np.ones(img.shape[:2], dtype=np.uint8)
            continue

        # obtain box region

        # for i in range(len(gt_hboxes)):
        #     canvas = np.zeros([*img.shape[:2], 3], dtype=np.uint8)
        #     draws = cv2.rectangle(canvas, (int(gt_hboxes[i][0]), int(gt_hboxes[i][1])), (int(gt_hboxes[i][2]), int(gt_hboxes[i][3])), (255, 255, 255), -1)
        #     box_mask = np.zeros([*img.shape[:2]], dtype=np.uint8)
        #     posi = np.all(draws == np.array([255,255,255]).reshape(1, 1, 3), axis=2)
        #     # mask failed
        #     # box_mask[posi] = 1000
        #     # box_mask = cv2.resize(box_mask,(256,256), interpolation=cv2.INTER_NEAREST)
        #     # rbox_mask_prompts.append(torch.tensor(box_mask).float())
            
        #     # for vis
        #     box_mask = np.zeros([*img.shape[:2]], dtype=np.uint8)
        #     box_mask[posi] = 1
        #     hbox_masks.append(box_mask)

        gt_hboxes = np.stack(gt_hboxes,axis=0) # b,4
        gt_hboxes = torch.from_numpy(gt_hboxes).cuda() # b,4
        #gt_points = np.stack(gt_points,axis=0)
        #gt_points = torch.from_numpy(gt_points).cuda() # b, 2
        #input_labels = torch.ones(gt_points.shape[0]).cuda() # b
        #hbox_masks = np.stack(hbox_masks,axis=0) #b, h, w

        predictor.set_image(img)

        part_num = len(gt_labels) // batch_size + 1

        start = 0
        end = np.min([len(gt_labels), start+batch_size])

        seg_mask = 255*np.ones(img.shape[:2], dtype=np.uint8)
        seg_color = 255*np.ones([*img.shape[:2],3], dtype=np.uint8)

        image_info = []

        for i in range(0, part_num):

            if start < end:

                batch_hboxes = gt_hboxes[start:end]
                batch_labels = gt_labels[start:end]

                transformed_boxes = predictor.transform.apply_boxes_torch(batch_hboxes, img.shape[:2])
                # use point and box to predict mask
                sam_masks, _, _ = predictor.predict_torch(
                    point_coords=None, #gt_points[:,None,:],
                    point_labels=None, #input_labels[:,None],
                    boxes=transformed_boxes,
                    mask_input=None,
                    multimask_output=False) # 只输出一个mask
        
                sam_masks = sam_masks.squeeze(1) # b, h, w
                #qualities = qualities.squeeze(-1) # 每个box生成mask的质量

                #gt_points = gt_points.squeeze(-1)
                #gt_points = gt_points.cpu().numpy()
            
                sam_masks = sam_masks.cpu().numpy()
                #gt_hboxes = gt_hboxes.cpu().numpy()
                #if args.semantic == 'True':

                batch_hboxes = batch_hboxes.cpu().numpy()

                for j in range(len(batch_labels)):
                    # seg
                    mask_points_row, mask_points_col = np.array(np.nonzero(sam_masks[j]))
                    seg_mask[mask_points_row, mask_points_col] = batch_labels[j]
                    seg_color[mask_points_row, mask_points_col] = MAPPING[batch_labels[j]]
                    # ins
                    rle = maskUtils.encode(np.asfortranarray(sam_masks[j].astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    bbox = batch_hboxes[j]
                    area = np.sum(sam_masks[j])
                    ins_info = {'mask':rle, 'bbox':bbox, 'category':lbl2cls[batch_labels[j]], 'label':batch_labels[j], 'size':area}
                    image_info.append(ins_info)

            start = end
            end = np.min([len(gt_labels), start+batch_size])

        # save gray, rgb. ins infos
        seg_mask = Image.fromarray(seg_mask)
        seg_color = Image.fromarray(seg_color)
        seg_mask.save(os.path.join(save_dir,'gray',img_name+'.png'))
        seg_color.save(os.path.join(save_dir,'color',img_name+'.png'))
        pickle.dump(image_info, open(os.path.join(save_dir,'ins',img_name+'.pkl'), 'wb'))
                
        print('Predict {} batches ({} boxes) to generate mask for image {}: {}.'.format(part_num, len(gt_labels), cnt, img_name))
        cnt += 1

        #del predictor

    #     all_labels.append(gt_labels)
    #     all_masks.append(sam_masks)
    #     all_names.append(img_name)
    #     all_points.append(gt_points)
    #     all_hboxes.append(gt_hboxes)
    
    #     if args.instance == 'True':

    #         all_probs.append(qualities)
    #         all_hboxmasks.append(hbox_masks)

    # if args.show == 'True':

    #     ## sam mask

    #     for i in range(len(all_labels)):
    #         img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
    #         plt.figure(i,figsize=(10, 10))
    #         plt.imshow(img)
    #         for j in range(len(all_labels[i])):
    #             color = np.array(MAPPING[all_labels[i][j]])
    #             show_hbox_mask(all_points[i][j], all_hboxes[i][j], all_masks[i][j], plt.gca(), random_color='box', color=color)
    #         plt.axis('off')
    #         plt.savefig(vis_dir + 'out_hbox_sam_mask_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
    #         plt.close()

    #         print('Save the mask of image {}: {}.'.format(i, all_names[i]))

    #     print('Sam semantic mask of Hbox visualization finished!')

        # ## box_mask

        # for i in range(len(all_labels)):
        #     img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
        #     plt.figure(i,figsize=(10, 10))
        #     plt.imshow(img)
        #     for j in range(len(all_labels[i])):
        #         color = all_colors[i][j]
        #         show_hbox_mask(all_points[i][j], all_hboxes[i][j], all_hboxmasks[i][j], plt.gca(), random_color='box', color=color)
        #     plt.axis('off')
        #     plt.savefig(vis_dir + 'out_horizontal_box_mask_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
        #     plt.close()

        #     print('Save the mask of image {}: {}.'.format(i, all_names[i]))

        # print('Horizontal box visualization finished!')
        