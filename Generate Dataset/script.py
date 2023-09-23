import os
import pickle
import argparse
import numpy as np
from glob import glob
from mapping import DOTA2_0, DIOR

parser = argparse.ArgumentParser(description="det2seg")
## dataset setting

parser.add_argument('--dataset', type=str, default='sior',
                    choices=['sota','sior','fast'],
                    help='detection annotation type')
args = parser.parse_args()

if args.dataset == 'sota':
    ins_dir = '/root/dataset/dotav2_1024/trainval/hbox_segs_init/ins/'
    save_dir = '/root/dataset/dotav2_1024/trainval/hbox_segs_init/ins_new/'
    lbl2cls = {k:v for k,v in enumerate(DOTA2_0)}
elif args.dataset == 'sior':
    ins_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/ins/'
    save_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/ins_new/'
    lbl2cls = {k:v for k,v in enumerate(DIOR)}

os.makedirs(save_dir, exist_ok=True)

files = glob(os.path.join(ins_dir,'*.pkl'))

for i in range(len(files)):
    file = files[i]
    filename = os.path.basename(file)
    f = open(file, 'rb')
    image_info = pickle.load(f)
    f.close()
    new_image_info = []
    for j in range(len(image_info)):
        ins_info = image_info[j]
        label = ins_info['category']
        category = lbl2cls[label]
        ins_info['label'] = label
        ins_info['category'] = category
        new_image_info.append(ins_info)

    pickle.dump(new_image_info, open(os.path.join(save_dir, filename), 'wb'))

    print('process file {}: {}'.format(i, file))
