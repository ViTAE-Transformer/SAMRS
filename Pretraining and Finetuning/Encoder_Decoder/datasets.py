import os
import cv2
import torch
import numpy as np
import torch.utils.data as D
from torchvision import transforms as T
from PIL import Image
from glob import glob
from skimage import io

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class SegmentationDataset(D.Dataset):

    def __init__(self, args, img_size, root, image_path, label_path, ext_img = '.png', ext_lbl = '.png', flag = None, transform=None):

        self.args = args

        if flag=='trn' or flag=='val' or flag=='tes':

            with open(os.path.join(root, 'train.txt'), mode='r') as f:
                train_infos = f.readlines()
            f.close()

            trn_files = []
            trn_targets = []

            for item in train_infos:
                fname = item.strip()
                #print(fname)
                trn_files.append(os.path.join(image_path, fname + ext_img))
                trn_targets.append(os.path.join(label_path, fname + ext_lbl))


            with open(os.path.join(root, 'valid.txt'), mode='r') as f:
                valid_infos = f.readlines()
            f.close()

            val_files = []
            val_targets = []

            for item in valid_infos:
                fname = item.strip()
                val_files.append(os.path.join(image_path, fname + ext_img))
                val_targets.append(os.path.join(label_path, fname + ext_lbl))
        
        else:
            raise NotImplementedError

        if flag=='trn':
            self.files = trn_files
            self.targets = trn_targets
        elif flag=='val':
            self.files = val_files[-500:]
            self.targets = val_targets[-500:]
        elif flag == 'tes':
            self.files = val_files
            self.targets = val_targets

        self.length = len(self.targets)

        self.flag = flag
        self.transform = transform

        self.to_tensor = T.Compose([
            T.ToPILImage(),
            #T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

    def _background_to_trainid(self, label):

        label_copy = label.copy()
        label_copy += 1
        label_copy[label_copy==256] = 0

        return label_copy

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]
        lbl_path = self.targets[i]

        image = np.array(Image.open(img_path))
        label = np.array(Image.open(lbl_path))

        if self.args.background == 'True':
            label = np.array(label, dtype=np.int32)
            label = self._background_to_trainid(label)

        augments = self.transform(image=image, mask=label)

        aug_img = self.to_tensor(np.uint8(augments['image']))
        aug_lbl = torch.from_numpy(augments['mask'])

        return aug_img, aug_lbl
    

class ISPRSDataset(D.Dataset):
    def __init__(self, img_size = None, split=None, data_root=None, transform=None):
        self.split = split#每个元素：[图像路径，标记路径]
        if split == 'train':
            self.image_list = glob(os.path.join(data_root, 'img_dir', 'train', '*.png'))
        elif split == 'val':
            self.image_list = glob(os.path.join(data_root, 'img_dir', 'val', '*.png'))
            self.image_list = self.image_list[:500]
        elif split == 'test':
            self.image_list = glob(os.path.join(data_root, 'img_dir', 'val', '*.png'))
        else:
            raise NotImplementedError

        self.label_list = []

        for i in range(len(self.image_list)):
            path, image_name = os.path.split(self.image_list[i])
            folder_name = path.split('/')[-1]
            self.label_list.append(os.path.join(data_root, 'ann_dir', folder_name, image_name))

        self.data_list = [self.image_list, self.label_list]

        print("Totally {} samples in {} set.".format(len(self.label_list), split))

        self.transform = transform

        if self.split == 'test':
            self.to_tensor = T.Compose([
                T.ToTensor(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

        else:
            self.to_tensor = T.Compose([
                T.ToPILImage(),
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

    def _trainid_to_class(self, label):

        return label

    def tes_class_to_trainid(self, label):
   
        return label

    def _class_to_trainid(self, label):

        return label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        # cv2.setNumThreads(0)
        # cv2.ocl.setUseOpenCL(False)

        image_path = self.image_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3

        image = np.float32(image)
        image = image[:, :, ::-1].copy()  ## BGR -> RGB

        if self.split != 'test':
            
            label_path = self.label_list[index]

            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

            label = np.array(label, dtype=np.int32)
            label = self._class_to_trainid(label)

            augments = self.transform(image=image, mask=label)

            aug_img = self.to_tensor(np.uint8(augments['image']))
            aug_lbl = torch.from_numpy(augments['mask'])

            return aug_img, aug_lbl
        
        else:
        
            return self.to_tensor(np.uint8(image))


class ISAIDDataset(D.Dataset):
    def __init__(self, img_size = None, split=None, data_root=None, transform=None):
        self.split = split#每个元素：[图像路径，标记路径]
        if split == 'train':
            self.image_list = glob(os.path.join(data_root, 'train', 'images', '*.png'))
        elif split == 'val':
            self.image_list = glob(os.path.join(data_root, 'val', 'images', '*.png'))
            self.image_list = self.image_list[:500]
        elif split == 'test':
            self.image_list = glob(os.path.join(data_root, 'val', 'images', '*.png'))
        else:
            raise NotImplementedError

        self.label_list = []

        if split == 'val' or split == 'test':
            path_split = 'val'
        else:
            path_split = 'train'

        for i in range(len(self.image_list)):
            _, image_name = os.path.split(self.image_list[i])
            base_name = image_name[:-4]
            self.label_list.append(os.path.join(data_root, path_split, 'labels', base_name + '_instance_color_RGB.png'))

        self.data_list = [self.image_list, self.label_list]

        print("Totally {} samples in {} set.".format(len(self.label_list), split))

        self.transform = transform

        if self.split == 'test':
            self.to_tensor = T.Compose([
                T.ToTensor(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

        else:
            self.to_tensor = T.Compose([
                T.ToPILImage(),
                #T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

    def _trainid_to_class(self, label):

        return label

    def tes_class_to_trainid(self, label):

        return label

    def _class_to_trainid(self, label):

        return label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        image_path, label_path = self.image_list[index], self.label_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3

        image = np.float32(image)
        image = image[:, :, ::-1].copy()  ## BGR -> RGB

        if self.split != 'test':
            
            label_path = self.label_list[index]

            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

            label = np.array(label, dtype=np.int32)
            label = self._class_to_trainid(label)

            augments = self.transform(image=image, mask=label)

            aug_img = self.to_tensor(np.uint8(augments['image']))
            aug_lbl = torch.from_numpy(augments['mask'])

            return aug_img, aug_lbl
        
        else:
        
            return self.to_tensor(np.uint8(image))