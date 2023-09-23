import os
import numpy as np
import argparse
from PIL import Image

MAPPING_VISUALIZE = {
    255: (255, 255, 255),
    6: (0, 0, 63),
    9: (0, 191, 127),
    1: (0, 63, 0),
    7: (0, 63, 127),
    8: (0, 63, 191),
    3: (0, 63, 255),
    2: (0, 127, 63),
    5: (0, 127, 127),
    4: (0, 0, 127),
    14: (0, 0, 191),
    13: (0, 0, 255),
    11: (0, 63, 63),
    10: (0, 127, 191),
    0: (0, 127, 255),
    12: (0, 100, 155),
    15: (64, 191, 127),
    16: (64, 0, 191),
    17: (128, 63, 63),
    18: (128, 0, 63),
    19: (191, 63, 0),
    20: (255, 127, 0),
    21: (63, 0, 0),
    22: (127, 63, 0),
    23: (63, 255, 0),
    24: (0, 127, 0),
    25: (127, 127, 0),
    26: (63, 0, 63),
    27: (63, 127, 0),
    28: (63, 191, 0),
    29: (191, 127, 0),
    30: (127, 191, 0),
    31: (63, 63, 0),
    32: (100, 155, 0),
    33: (0, 255, 0),
    34: (0, 191, 0),
    35: (191, 127, 64),
    36: (0, 191, 64)
    }


parser = argparse.ArgumentParser(description="det2seg")
## dataset setting

parser.add_argument('--dataset', type=str, default='fast',
                    choices=['sota','sior_1', 'sior_2','fast'],
                    help='detection annotation type')
args = parser.parse_args()


if args.dataset == 'sota':
    img_dir = '/root/dataset/dotav2_1024/trainval/images/'
    label_dir = '/root/dataset/dotav2_1024/trainval/hbox_segs_init/gray/'
    vis_dir = '/root/dataset/dotav2_1024/trainval/hbox_segs_init/vis/'
elif args.dataset == 'sior_1':
    img_dir = '/root/dataset/dior/JPEGImages-trainval/'
    label_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/gray/'
    vis_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/vis/'
elif args.dataset == 'sior_2':
    img_dir = '/root/dataset/dior/JPEGImages-test/'
    label_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/gray/'
    vis_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/vis/'
elif args.dataset == 'fast':
    img_dir = '/root/dataset/fair1m_1024/trainval/images/'
    label_dir = '/root/dataset/fair1m_1024/trainval/rhbox_segs_init/gray/'
    vis_dir = '/root/dataset/fair1m_1024/trainval/rhbox_segs_init/vis/'
else:
    raise NotImplementedError

os.makedirs(vis_dir, exist_ok=True)

files = os.listdir(img_dir)

cnt = 0
for file in files:
    filename = os.path.basename(file)
    img_name = filename[:-4]

    image = np.array(Image.open(os.path.join(img_dir,file)))
    label = np.array(Image.open(os.path.join(label_dir,img_name+'.png')))

    seg_color = np.zeros([*image.shape[:2],3], dtype=np.uint8)

    for k, v in MAPPING_VISUALIZE.items():
        m = label == k
        seg_color[m] = v

    image = Image.fromarray(image)
    seg_color = Image.fromarray(seg_color)
    vis = Image.blend(image, seg_color, 0.4)
    
    vis.save(os.path.join(vis_dir, filename))

    print('Dataset {}: generate image {}: {}'.format(args.dataset, cnt, filename))
    cnt += 1

