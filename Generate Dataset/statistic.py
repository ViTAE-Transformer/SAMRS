import os
import pickle
import argparse
import numpy as np
from glob import glob

# number of pixel and instance for each category
def obtain_class_pixel_ins_num(args, files, class_pixel_num_array, class_instance_num_array, save_dir):

    for i in range(len(files)):
        file = files[i]
        f = open(file, 'rb')
        image_info = pickle.load(f)
        f.close()
        for j in range(len(image_info)):
            ins_info = image_info[j]
            label = ins_info['label']
            area = ins_info['size']
            if area > 0:
                class_pixel_num_array[label] += area
                class_instance_num_array[label] += 1

        print('memory pixel and instance nums for file {}: {}'.format(i, file))

    print('{} dataset class v.s. pixel num'.format(args.dataset))
    print(class_pixel_num_array)
    print('{} dataset class v.s. instance num'.format(args.dataset))
    print(class_instance_num_array)

    pickle.dump(class_pixel_num_array, open(os.path.join(save_dir, 'class_pixel_num_'+ str(args.dataset) +'.pkl'), 'wb'))
    pickle.dump(class_instance_num_array, open(os.path.join(save_dir, 'class_instance_num_'+ str(args.dataset) +'.pkl'), 'wb'))

# mask size of each instance
def obtain_instance_mask_size(args, files, save_dir):

    instance_mask_size_list = []

    for i in range(len(files)):
        file = files[i]
        f = open(file, 'rb')
        image_info = pickle.load(f)
        f.close()
        image_mask_sizes = []
        for j in range(len(image_info)):
            ins_info = image_info[j]
            area = ins_info['size']
            if area > 0:
                image_mask_sizes.append(area)
        instance_mask_size_list += image_mask_sizes

        print('memory mask sizes for file {}: {}'.format(i, file))

    print('{} dataset has {} instances'.format(args.dataset, len(instance_mask_size_list)))

    #pickle.dump(instance_mask_size_list, open(os.path.join(save_dir, 'instance_mask_size_'+ str(args.dataset) +'.pkl'), 'wb'))

    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description="det2seg")
    ## dataset setting

    parser.add_argument('--dataset', type=str, default='fast',
                        choices=['sota','sior','fast'],
                        help='detection annotation type')
    args = parser.parse_args()


    if args.dataset == 'sota':
        class_num = 18
        label_dir = '/root/dataset/dotav2_1024/trainval/hbox_segs_init/ins/'
        save_dir = '/root/dataset/dotav2_1024/trainval/hbox_segs_init/statistic'
    elif args.dataset == 'sior':
        class_num = 20
        label_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/ins/'
        save_dir = '/root/dataset/dior/hbox_segs_trainvaltest_init/statistic'
    elif args.dataset == 'fast':
        class_num = 37
        label_dir = '/root/dataset/fair1m_1024/trainval/rhbox_segs_init/ins/'
        save_dir =  '/root/dataset/fair1m_1024/trainval/rhbox_segs_init/statistic'

    #os.makedirs(save_dir, exist_ok=True)

    class_pixel_num_array = dict()
    class_instance_num_array = dict()

    for i in range(class_num):
        class_pixel_num_array[i] = 0
        class_instance_num_array[i] = 0

    files = glob(os.path.join(label_dir,'*.pkl'))

    #obtain_class_pixel_ins_num(args, files, class_pixel_num_array, class_instance_num_array, save_dir)

    obtain_instance_mask_size(args, files, save_dir)
