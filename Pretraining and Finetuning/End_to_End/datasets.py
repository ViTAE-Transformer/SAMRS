import os
import cv2
import torch
import numpy as np
import torch.utils.data as D
from torchvision import transforms as T
from PIL import Image
from glob import glob
from skimage import io

from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

#from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample

from mmengine.structures import BaseDataElement, PixelData

# Copyright (c) OpenMMLab. All rights reserved.
class SegDataSample(BaseDataElement):
    """A data structure interface of MMSegmentation. They are used as
    interfaces between different components.

    The attributes in ``SegDataSample`` are divided into several parts:

        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
        - ``seg_logits``(PixelData): Predicted logits of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import PixelData
         >>> from mmseg.structures import SegDataSample

         >>> data_sample = SegDataSample()
         >>> img_meta = dict(img_shape=(4, 4, 3),
         ...                 pad_shape=(4, 4, 3))
         >>> gt_segmentations = PixelData(metainfo=img_meta)
         >>> gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
         >>> data_sample.gt_sem_seg = gt_segmentations
         >>> assert 'img_shape' in data_sample.gt_sem_seg.metainfo_keys()
         >>> data_sample.gt_sem_seg.shape
         (4, 4)
         >>> print(data_sample)
        <SegDataSample(

            META INFORMATION

            DATA FIELDS
            gt_sem_seg: <PixelData(

                    META INFORMATION
                    img_shape: (4, 4, 3)
                    pad_shape: (4, 4, 3)

                    DATA FIELDS
                    data: tensor([[[1, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 0, 1]]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = SegDataSample()
        >>> gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> data_sample.gt_sem_seg = gt_sem_seg
        >>> assert 'gt_sem_seg' in data_sample
        >>> assert 'sem_seg' in data_sample.gt_sem_seg
    """

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> PixelData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_seg_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits


#@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results):
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = results['gt_seg_map'][None,:,:]
            else:
                raise NotImplementedError
            
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        return data_sample

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

        if 'mask2former' in self.args.decoder:
            self.packinputs = PackSegInputs()

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

        if 'mask2former' in self.args.decoder:
            data_info = dict()
            
            data_info['img_path'] = img_path
            data_info['seg_map_path'] = lbl_path
            data_info['ori_shape'] = tuple(image.shape)
            data_info['img_shape'] = tuple(aug_img.permute(1,2,0).shape)

            data_info['gt_seg_map'] = aug_lbl

            data_sample = self.packinputs.transform(data_info)

            return aug_img, data_sample
        else:
            return aug_img, aug_lbl
    

class ISPRSDataset(D.Dataset):
    def __init__(self, args, img_size = None, split=None, data_root=None, transform=None):

        self.args = args

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

        if 'mask2former' in self.args.decoder:
            self.packinputs = PackSegInputs()

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


            if 'mask2former' in self.args.decoder:
                data_info = dict()
                
                data_info['img_path'] = image_path
                data_info['seg_map_path'] = label_path
                data_info['ori_shape'] = tuple(image.shape)
                data_info['img_shape'] = tuple(aug_img.permute(1,2,0).shape)
                data_info['gt_seg_map'] = aug_lbl

                data_sample = self.packinputs.transform(data_info)

                return aug_img, data_sample

            return aug_img, aug_lbl
        
        else:

            if 'mask2former' in self.args.decoder:
                data_info = dict()
                
                data_info['img_path'] = image_path
                data_info['ori_shape'] = tuple(image.shape)
                data_info['img_shape'] = (self.args.crop_h, self.args.crop_w, 3)

                data_sample = self.packinputs.transform(data_info)

                return self.to_tensor(np.uint8(image)), data_sample
        
            return self.to_tensor(np.uint8(image))


class ISAIDDataset(D.Dataset):
    def __init__(self, args, img_size = None, split=None, data_root=None, transform=None):

        self.args = args

        self.split = split#每个元素：[图像路径，标记路径]
        if split == 'train':
            self.image_list = glob(os.path.join(data_root, 'train', 'images', '*.png'))
        elif split == 'val':
            self.image_list = glob(os.path.join(data_root, 'val', 'images', '*.png'))
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
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])

        if 'mask2former' in self.args.decoder:
            self.packinputs = PackSegInputs()

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


            if 'mask2former' in self.args.decoder:
                data_info = dict()
                
                data_info['img_path'] = image_path
                data_info['seg_map_path'] = label_path
                data_info['ori_shape'] = tuple(image.shape)
                data_info['img_shape'] = tuple(aug_img.permute(1,2,0).shape)
                data_info['gt_seg_map'] = aug_lbl

                data_sample = self.packinputs.transform(data_info)

                return aug_img, data_sample

            return aug_img, aug_lbl
        
        else:

            if 'mask2former' in self.args.decoder:
                data_info = dict()
                
                data_info['img_path'] = image_path
                data_info['ori_shape'] = tuple(image.shape)
                data_info['img_shape'] = (self.args.crop_h, self.args.crop_w, 3)

                data_sample = self.packinputs.transform(data_info)

                return self.to_tensor(np.uint8(image)), data_sample
        
            return self.to_tensor(np.uint8(image))
