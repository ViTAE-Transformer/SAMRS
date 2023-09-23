import os
import argparse
import json
import torch
import numpy as np
#from utils import show_box, show_mask, show_hbox_mask
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from loaddata import load_hrsc, load_dota
from PIL import Image
from instance_to_json import binary_to_coco_gt_hrsc, binary_to_coco_pre_hrsc
import cv2
from segment_anything.utils.transforms import ResizeLongestSide

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
    # ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=c,
    #                            facecolor=(0, 0, 0, 0), lw=2))
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=(1,1,1,1),
                                facecolor=(0, 0, 0, 0), lw=2, linestyle='--'))
    
    #ax.scatter(point[0],point[1],marker='*',s=200, color=c, edgecolor=c, linewidth=2)
    #ax.scatter(point[0],point[1],marker='*',s=200, color=np.array([1,0,0,1]), edgecolor=np.array([1,0,0,1]), linewidth=2)

parser = argparse.ArgumentParser(description="det2seg")
## dataset setting
parser.add_argument('--type', type=str, default='center',
                    choices=['corner','center'],
                    help='detection annotation type')
parser.add_argument('--dataset', type=str, default='hrsc',
                    choices=['dota','hrsc'],
                    help='detection annotation type')
parser.add_argument('--instance', type=str, default='True',
                    choices=['True','False'],
                    help='visualization')
# parser.add_argument('--semantic', type=str, default='False',
#                     choices=['True','False'],
#                     help='visualization')
parser.add_argument('--show', type=str, default='True',
                    choices=['True','False'],
                    help='visualization')

args = parser.parse_args()

EXE_TYPE = ".bmp"

file_dir = "/root/dataset/HRSC2016/Test/AllImages/"

img_dir = "/root/dataset/HRSC2016/Test/AllImages/"
ann_dir = "/root/dataset/HRSC2016/Test/Annotations/"

# for ins
labeld_dir = "/root/dataset/HRSC2016/FullDataSet/LandMask/"
json_dir = "/root/dw/samrs/work_dir/hrsc/json/"

# for seg
save_dir = ""

# for vis
vis_dir = "/root/dw/samrs/work_dir/hrsc/Vis_Test_Seg_HBox_Mask_Prompt/"

os.makedirs(vis_dir, exist_ok=True)


sam_checkpoint = "/root/dw/pretrn/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device=device)
predictor = SamPredictor(sam)

if __name__ == '__main__':

    files = os.listdir(file_dir)

    all_labels = []
    all_masks = []
    all_colors = []
    all_names = []
    all_probs = []
    all_points = []
    all_hboxes = []
    all_hboxmasks = []

    all_masks_1 = []
    all_masks_2 = []
    all_masks_3 = []

    cnt = 0

    for file in files:
        filename = os.path.basename(file)
        img_name = filename[:-4]
        img = np.array(Image.open(os.path.join(img_dir,img_name + EXE_TYPE)))
        hbox_masks = []
        hbox_mask_prompts = []

        if args.type == 'center':
            
            if args.dataset == 'hrsc':
                gt_hboxes, _, gt_colors, gt_points, gt_labels, error = load_hrsc(img_name, ann_dir)
                # gt_colors: b,3

        else:

            if args.dataset == 'dota':
                gt_hboxes, _, gt_points, gt_labels = load_dota(img_name, ann_dir)

        if error==1: # must have box
            sam_masks = np.zeros(img.shape[:2])
            #seg_mask = np.zeros(img.shape[:2])
            continue

        # obtain box region

        for i in range(len(gt_hboxes)):
            canvas = np.zeros([*img.shape[:2], 3], dtype=np.uint8)
            draws = cv2.rectangle(canvas, (int(gt_hboxes[i][0]), int(gt_hboxes[i][1])), (int(gt_hboxes[i][2]), int(gt_hboxes[i][3])), (255, 255, 255), -1)
            box_mask = -1000*np.ones([*img.shape[:2]])
            posi = np.all(draws == np.array([255,255,255]).reshape(1, 1, 3), axis=2)
            # mask
            box_mask[posi] = 1000
             # 长边缩放为网络设置值
            target_size = ResizeLongestSide.get_preprocess_shape(box_mask.shape[0], box_mask.shape[1], sam.image_encoder.img_size)
            box_mask = cv2.resize(box_mask,target_size[::-1], interpolation=cv2.INTER_LINEAR)
            padh = sam.image_encoder.img_size - box_mask.shape[0]
            padw = sam.image_encoder.img_size - box_mask.shape[1]
            box_mask = cv2.copyMakeBorder(box_mask, 0, padh, 0, padw, cv2.BORDER_CONSTANT, value=-1000) # 短边填充
            box_mask = cv2.resize(box_mask,(256,256), interpolation=cv2.INTER_LINEAR)
            hbox_mask_prompts.append(torch.tensor(box_mask).float())
            
            # for vis
            box_mask = np.zeros([*img.shape[:2]], dtype=np.uint8)
            box_mask[posi] = 1
            hbox_masks.append(box_mask)

        gt_hboxes = np.stack(gt_hboxes,axis=0) # b,4
        gt_hboxes = torch.from_numpy(gt_hboxes).cuda() # b,4
        gt_points = np.stack(gt_points,axis=0)
        gt_points = torch.from_numpy(gt_points).cuda() # b, 2
        input_labels = torch.ones(gt_points.shape[0]).cuda() # b
        hbox_masks = np.stack(hbox_masks,axis=0) #b, h, w
        hbox_mask_prompts= torch.stack(hbox_mask_prompts,dim=0).cuda()

        predictor.set_image(img)
        transformed_boxes = predictor.transform.apply_boxes_torch(gt_hboxes, img.shape[:2])
        # use point and box to predict mask
        sam_masks, qualities, lr_logits = predictor.predict_torch(
            point_coords=gt_points[:,None,:],
            point_labels=input_labels[:,None],
            boxes=None,#transformed_boxes,
            mask_input=None,#hbox_mask_prompts[:,None,:,:],
            multimask_output=False) # 只输出一个mask

        # mask prompt

        # lr_logits = lr_logits.squeeze(1)
        # lr_logits_1 = lr_logits.clone()
        # lr_logits_2 = lr_logits.clone()

        # for k in range(lr_logits.shape[0]):
        #     lr_logits_1[k] = lr_logits[k] / (torch.max(torch.abs(lr_logits[k]))) * 300.0
        #     lr_logits_2[k] = lr_logits[k] / (torch.max(torch.abs(lr_logits[k]))) * 500.0

        # lr_logits_3 = lr_logits.clone()
        # lr_logits_3[lr_logits>0] = 300.0
        # lr_logits_3[lr_logits<0] = -300.0

        # sam_masks_1, qualities_1, lr_logits1 = predictor.predict_torch(
        #     point_coords=None, #gt_points[:,None,:],
        #     point_labels=None, #input_labels[:,None],
        #     boxes=None,
        #     mask_input=lr_logits_1[:,None,:,:],
        #     multimask_output=False) # 只输出一个mask

        # sam_masks_2, qualities_2, lr_logits2 = predictor.predict_torch(
        #     point_coords=None, #gt_points[:,None,:],
        #     point_labels=None, #input_labels[:,None],
        #     boxes=None,
        #     mask_input=lr_logits_2[:,None,:,:],
        #     multimask_output=False) # 只输出一个mask

        # sam_masks_3, qualities_3, lr_logits3 = predictor.predict_torch(
        #     point_coords=None, #gt_points[:,None,:],
        #     point_labels=None, #input_labels[:,None],
        #     boxes=None,
        #     mask_input=lr_logits_3[:,None,:,:],
        #     multimask_output=False) # 只输出一个mask           
    
        sam_masks = sam_masks.squeeze(1) # b, h, w
        qualities = qualities.squeeze(-1) # 每个box生成mask的质量

        gt_points = gt_points.squeeze(-1)
        gt_points = gt_points.cpu().numpy()
    
        sam_masks = sam_masks.cpu().numpy()
        gt_hboxes = gt_hboxes.cpu().numpy()

        # sam_masks_1 = sam_masks_1.squeeze(1) # b, h, w
        # sam_masks_1 = sam_masks_1.cpu().numpy()

        # sam_masks_2 = sam_masks_2.squeeze(1) # b, h, w
        # sam_masks_2 = sam_masks_2.cpu().numpy()

        # sam_masks_3 = sam_masks_3.squeeze(1) # b, h, w
        # sam_masks_3 = sam_masks_3.cpu().numpy()

        # if args.semantic == 'True':

        #     seg_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        #     for i in range(len(gt_labels)):
        #         mask_points_row, mask_points_col = np.array(np.nonzero(sam_masks[i]))
        #         seg_mask[mask_points_row, mask_points_col] = gt_labels[i]
                
        print('Predict the mask of image {}: {}.'.format(cnt, img_name))
        cnt += 1
        
        if args.instance == 'True':

            all_labels.append(gt_labels)
            all_masks.append(sam_masks)
            all_colors.append(gt_colors)
            all_names.append(img_name)
            all_probs.append(qualities)
            all_points.append(gt_points)
            all_hboxes.append(gt_hboxes)
            all_hboxmasks.append(hbox_masks)

            # all_masks_1.append(sam_masks_1)
            # all_masks_2.append(sam_masks_2)
            # all_masks_3.append(sam_masks_3)


    if args.instance == 'True':

        all_gt_masks = []

        for i in range(len(all_labels)):
            sam_masks = all_masks[i] == 1
            gt_ins = []
            for j in range(len(all_labels[i])):
                ins_mask = np.zeros([*sam_masks.shape[1:]], dtype=np.uint8) # official labeled
                ins_color = all_colors[i][j]
                labeld_img = np.array(Image.open(os.path.join(labeld_dir,all_names[i]+'.png')), dtype=np.uint8)
                ins_posi = np.all(labeld_img == ins_color.reshape(1, 1, 3), axis=2)
                ins_mask[ins_posi] = 1
                gt_ins.append(ins_mask)
            gt_ins = np.stack(gt_ins, axis=0)
            all_gt_masks.append(gt_ins)

        ## Compute mIoU
        
        avg_ious = []
        area_intersects = []
        area_unions = []
        for i in range(len(all_masks)):
            for j in range(all_masks[i].shape[0]):
                
                single_box_gt  = all_gt_masks[i][j].reshape(-1).astype(float)
                single_box_pre = all_masks[i][j].reshape(-1).astype(float)
                
                intersect = float(np.sum(single_box_gt * single_box_pre))
                
                addition = single_box_gt + single_box_pre
                union = float(np.sum(np.array(addition>0)))
                
                if union > 0: # boxes may have no seg labels
                
                    area_intersects.append(intersect)
                    area_unions.append(union)
                    avg_ious.append(intersect * 1.0 / union)
                
        miou_avg = np.mean(np.array(avg_ious))
        miou_area = np.sum(np.array(area_intersects)) / np.sum(np.array(area_unions))
        
        print('Average mIOU: ',miou_avg, 'Area mIOU: ', miou_area)
        
        ## Obtain COCO format for AP calculation
            
        gt_coco_dict= binary_to_coco_gt_hrsc(all_gt_masks, all_names)
        sam_coco_dict = binary_to_coco_pre_hrsc(all_masks, all_names, all_probs=all_probs)
        
        with open(os.path.join(json_dir, 'sam_ins_hbox.json'), "w") as f:
            json.dump(sam_coco_dict, f)

        with open(os.path.join(json_dir, 'gt_ins_hbox.json'), "w") as f:
            json.dump(gt_coco_dict, f)

        print('Instance masks saved!')


    if args.show == 'True':

        ## sam mask

        # for i in range(len(all_labels)):
        #     img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
        #     plt.figure(i,figsize=(10, 10))
        #     plt.imshow(img)
        #     for j in range(len(all_labels[i])):
        #         color = all_colors[i][j]
        #         show_hbox_mask(all_points[i][j], all_hboxes[i][j], all_masks[i][j], plt.gca(), random_color='box', color=color)
        #     plt.axis('off')
        #     plt.savefig(vis_dir + 'out_hbox_prompt_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
        #     plt.close()

        #     print('Save the mask of image {}: {}.'.format(i, all_names[i]))

        # print('Sam mask of Hbox visualization finished!')

        # ## box_mask

        for i in range(len(all_labels)):
            img = np.array(Image.open(os.path.join(img_dir,all_names[i] + EXE_TYPE)))
            plt.figure(i,figsize=(10, 10))
            plt.imshow(img)
            for j in range(len(all_labels[i])):
                color = all_colors[i][j]
                show_hbox_mask(all_points[i][j], all_hboxes[i][j], all_hboxmasks[i][j], plt.gca(), random_color='box', color=color)
            plt.axis('off')
            plt.savefig(vis_dir + 'out_horizontal_box_mask_'+all_names[i]+'.png',bbox_inches='tight', pad_inches = 0)
            plt.close()

            print('Save the mask of image {}: {}.'.format(i, all_names[i]))

        print('Horizontal box visualization finished!')
        