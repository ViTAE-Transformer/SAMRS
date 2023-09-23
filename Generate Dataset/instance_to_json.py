import json
import numpy as np
from pycocotools import mask as maskUtils

def binary_to_coco_gt_hrsc(binary_list, img_name_list):
    N = len(binary_list)

    # Define the COCO JSON format skeleton
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "ship",
                "supercategory": "None"
            }
        ]
    }

    # Add image information
    for n in range(N):
        channels, H, W = binary_list[n].shape
        image_info = {
            "id": int(n),
            "width": int(W),
            "height": int(H),
            "file_name": "{}.png".format(img_name_list[n])
        }
        coco_format["images"].append(image_info)

    # Add annotations
    for n in range(N):
        binary_array = binary_list[n]
        channels, H, W = binary_array.shape
        annotation_id = 0
        for c in range(channels):
            # Find the instance pixels
            instance_mask = binary_array[c, :, :]
            # if len(instance_indices) == 0:
            #     continue

            # Convert instance mask to COCO RLE format
            instance_rle = maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
            instance_rle['counts'] = instance_rle['counts'].decode('ascii')

#            if flag == 'gt':
            # Add annotation information
            annotation_info = {
                "id": annotation_id,
                "image_id": n,
                "category_id": 0,
                "area": int(np.sum(instance_mask)),
                "iscrowd": 0,
                "segmentation": instance_rle,
                "attributes": {}
            }
            # else:
            #     annotation_info = {
            #         "id": annotation_id,
            #         "image_id": n,
            #         "category_id": 0,
            #         "bbox": [float(y_min), float(x_min), float(y_max-y_min), float(x_max-x_min)],
            #         "area": int(np.sum(instance_mask)),
            #         "iscrowd": 0,
            #         "segmentation": instance_rle,
            #         "attributes": {},
            #         "score": float(all_probs[n][c])
            #     }
            coco_format["annotations"].append(annotation_info)
            annotation_id += 1

        print('Transform truth image {}: {} to coco format.'.format(n, img_name_list[n]))

    return coco_format


def binary_to_coco_pre_hrsc(binary_list, img_name_list, all_probs=None):
    N = len(binary_list)

    # Define the COCO JSON format skeleton
    coco_format = []

    # Add annotations
    for n in range(N):
        binary_array = binary_list[n]
        channels, H, W = binary_array.shape
        for c in range(channels):
            sample_info={}
            sample_info["image_id"] = int(n)
            # Find the instance pixels
            instance_mask = binary_array[c, :, :]
            # if len(instance_indices) == 0:
            #     continue

            # Convert instance mask to COCO RLE format
            instance_rle = maskUtils.encode(np.asfortranarray(instance_mask.astype(np.uint8)))
            instance_rle['counts'] = instance_rle['counts'].decode('ascii')

            sample_info["category_id"] = 0
            sample_info["segmentation"] = instance_rle

            sample_info["score"] = float(all_probs[n][c])

            coco_format.append(sample_info)

        print('Transform result image {}: {} to coco format.'.format(n, img_name_list[n]))

    return coco_format