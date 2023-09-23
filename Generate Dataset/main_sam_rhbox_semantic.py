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
from mapping import MAPPING, FAIR1M

from pycocotools import mask as maskUtils

def show_rhbox_mask(point, rbox, rhbox, mask, ax, random_color='None', color=None):
    if random_color == 'True':
        color = np.concatenate([np.random.random(3), np.array([0.4])], axis=0)
    elif random_color == 'False':
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4])
    else:
        color = np.concatenate([color / 255,np.array([0.4])],axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.plot([rbox[0,0],rbox[1,0],rbox[2,0],rbox[3,0],rbox[0,0]],[rbox[0,1],rbox[1,1],rbox[2,1],rbox[3,1],rbox[0,1]], linestyle='--',lw=1.5,color=np.array([1,1,1,1]))
    x0, y0 = rhbox[0], rhbox[1]
    w, h = rhbox[2] - rhbox[0], rhbox[3] - rhbox[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color,
                              facecolor=(0, 0, 0, 0), lw=2))
    ax.scatter(point[0],point[1],marker='*',s=200, color=np.array([1,1,1,1]), edgecolor=np.array([1,1,1,1]), linewidth=2)

parser = argparse.ArgumentParser(description="det2seg")
## dataset setting

parser.add_argument('--dataset', type=str, default='fair1m',
                    choices=['fair1m'],
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

EXE_TYPE = ".png"

file_dir = "/root/dataset/fair1m_1024/trainval/images/"

img_dir = "/root/dataset/fair1m_1024/trainval/images/"
ann_dir = "/root/dataset/fair1m_1024/trainval/rbbtxts/"

# for ins
labeld_dir = "/root/dataset/HRSC2016/FullDataSet/LandMask/"
json_dir = "/root/dw/samrs/work_dir/hrsc/json/"

# for seg
save_dir = "/root/dataset/fair1m_1024/trainval/rhbox_segs_init/"

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

lbl2cls = {k:v for k,v in enumerate(FAIR1M)}

if __name__ == '__main__':

    files = os.listdir(file_dir)

    print('{} dataset contains {} images'.format(args.dataset, len(files)))

    all_labels = []
    all_masks = []
    all_colors = []
    all_names = []
    all_probs = []
    all_points = []
    all_rboxes = []
    all_rhboxes = []
    all_rhboxmasks = []

    cnt = 0
    for file in files:
        filename = os.path.basename(file)
        img_name = filename[:-4]
        img = np.array(Image.open(os.path.join(img_dir,img_name + EXE_TYPE)))
        rhbox_masks = []

        if args.dataset == 'fair1m':
            _, gt_rboxes, gt_points, gt_labels, error = load_dota(img_name, ann_dir)

        if error==1: # must have box
            sam_masks = np.zeros(img.shape[:2])
            seg_mask = 255*np.ones(img.shape[:2], dtype=np.uint8)
            continue

        # obtain box region
        # rbox -> hbox
        # (b,4,2) -> (b,2,2)

        gt_rboxes = np.stack(gt_rboxes, axis=0) # b,4,2

        xmin = np.min(gt_rboxes[:,:,0], axis=1)
        ymin = np.min(gt_rboxes[:,:,1], axis=1)
        xmax = np.max(gt_rboxes[:,:,0], axis=1)
        ymax = np.max(gt_rboxes[:,:,1], axis=1)
        
        gt_rhboxes = np.stack([xmin, ymin, xmax, ymax], axis=1) # b,4

        # obtain box region

        # for i in range(gt_rhboxes.shape[0]):
        #     canvas = np.zeros([*img.shape[:2], 3], dtype=np.uint8)
        #     draws = cv2.rectangle(canvas, (int(gt_rhboxes[i,0]), int(gt_rhboxes[i,1])), (int(gt_rhboxes[i,2]), int(gt_rhboxes[i,3])), (255, 255, 255), -1)
        #     box_mask = np.zeros([*img.shape[:2]], dtype=np.uint8)
        #     posi = np.all(draws == np.array([255,255,255]).reshape(1, 1, 3), axis=2)
        #     # mask failed
        #     # box_mask[posi] = 1000
        #     # box_mask = cv2.resize(box_mask,(256,256), interpolation=cv2.INTER_NEAREST)
        #     # rbox_mask_prompts.append(torch.tensor(box_mask).float())
            
        #     # for vis
        #     box_mask = np.zeros([*img.shape[:2]], dtype=np.uint8)
        #     box_mask[posi] = 1
        #     rhbox_masks.append(box_mask)

        gt_rhboxes = torch.from_numpy(gt_rhboxes).cuda() # b,4
        # gt_points = np.stack(gt_points,axis=0)
        # gt_points = torch.from_numpy(gt_points).cuda() # b, 2
        # input_labels = torch.ones(gt_points.shape[0]).cuda() # b
        # rhbox_masks = np.stack(rhbox_masks,axis=0) #b, h, w

        predictor.set_image(img)

        part_num = len(gt_labels) // batch_size + 1

        start = 0
        end = np.min([len(gt_labels), start+batch_size])

        seg_mask = 255*np.ones(img.shape[:2], dtype=np.uint8)
        seg_color = 255*np.ones([*img.shape[:2],3], dtype=np.uint8)

        image_info = []

        for i in range(0, part_num):

            if start < end:
                
                batch_rhboxes = gt_rhboxes[start:end]
                batch_labels = gt_labels[start:end]

                transformed_boxes = predictor.transform.apply_boxes_torch(batch_rhboxes, img.shape[:2])
                # use point and box to predict mask
                sam_masks, _, _ = predictor.predict_torch(
                    point_coords = None, #gt_points[:,None,:],
                    point_labels= None, #input_labels[:, None],
                    boxes=transformed_boxes,
                    mask_input=None,
                    multimask_output=False) # 只输出一个mask
        
                sam_masks = sam_masks.squeeze(1) # b, h, w
                #qualities = qualities.squeeze(-1) # 每个box生成mask的质量

                # gt_points = gt_points.squeeze(-1)
                # gt_points = gt_points.cpu().numpy()

                sam_masks = sam_masks.cpu().numpy()
                #gt_rhboxes = gt_rhboxes.cpu().numpy()
        
                # if args.semantic == 'True':
                batch_rhboxes = batch_rhboxes.cpu().numpy()
                # also save rbox
                batch_rboxes =  gt_rboxes[start:end]

                for j in range(len(batch_labels)):
                    # seg
                    mask_points_row, mask_points_col = np.array(np.nonzero(sam_masks[j]))
                    seg_mask[mask_points_row, mask_points_col] = batch_labels[j]
                    seg_color[mask_points_row, mask_points_col] = MAPPING[batch_labels[j]]
                    # ins
                    rle = maskUtils.encode(np.asfortranarray(sam_masks[j].astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    rhbox = batch_rhboxes[j]
                    rbox = batch_rboxes[j]
                    area = np.sum(sam_masks[j])
                    ins_info = {'mask': rle, 'rbox':rbox, 'rhbox': rhbox, 'category': lbl2cls[batch_labels[j]],'label':batch_labels[j], 'size': area}
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
        
        # if args.instance == 'True':

        #     all_labels.append(gt_labels)
        #     all_masks.append(sam_masks)
        #     all_colors.append(gt_colors)
        #     all_names.append(img_name)
        #     all_probs.append(qualities)
        #     all_points.append(gt_points)
        #     all_rboxes.append(gt_rboxes)
        #     all_rhboxes.append(gt_rhboxes)
        #     all_rhboxmasks.append(rhbox_masks)
    
    # if args.show == 'True':

    #     ## sam mask

    #     for i in range(len(all_labels)):
    #         img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
    #         plt.figure(i,figsize=(10, 10))
    #         plt.imshow(img)
    #         for j in range(len(all_labels[i])):
    #             color = all_colors[i][j]
    #             show_rhbox_mask(all_points[i][j], all_rboxes[i][j], all_rhboxes[i][j], all_masks[i][j], plt.gca(), random_color='box', color=color)
    #         plt.axis('off')
    #         plt.savefig(vis_dir + 'out_rotate_honrizontal_box_sam_mask_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
    #         plt.close()

    #         print('Save the mask of image {}: {}.'.format(i, all_names[i]))

    #     print('Sam mask of Rbox visualization finished!')

        ## box_mask

        # for i in range(len(all_labels)):
        #     img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
        #     plt.figure(i,figsize=(10, 10))
        #     plt.imshow(img)
        #     for j in range(len(all_labels[i])):
        #         color = all_colors[i][j]
        #         show_rbox_mask(all_points[i][j], all_rboxes[i][j], all_rboxmasks[i][j], plt.gca(), random_color='box', color=color)
        #     plt.axis('off')
        #     plt.savefig(vis_dir + 'out_rotate_box_mask_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
        #     plt.close()

        #     print('Save the mask of image {}: {}.'.format(i, all_names[i]))

        # print('Rotated box visualization finished!')


        #### 可视化标签，以检查分割标注颜色和xml是否对齐

        # for i in range(len(all_labels)):
        #     img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
        #     plt.figure(i,figsize=(10, 10))
        #     plt.imshow(img)
        #     for j in range(len(all_labels[i])):
        #         color = all_colors[i][j]
        #         show_rbox_mask(all_points[i][j], all_rboxes[i][j], all_gt_masks[i][j], plt.gca(), random_color='box', color=color)
        #     plt.axis('off')
        #     plt.savefig(vis_dir + 'out_ground_truth_mask_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
        #     plt.close()

        #     print('Save the mask of image {}: {}.'.format(i, all_names[i]))

        # print('Ground Truth visualization finished!')

        
