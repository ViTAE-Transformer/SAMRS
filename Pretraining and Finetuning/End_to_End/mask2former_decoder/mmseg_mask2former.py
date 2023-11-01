# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

# try:
#     from mmdet.models.dense_heads import \
#         Mask2FormerHead as MMDET_Mask2FormerHead
# except ModuleNotFoundError:
#     MMDET_Mask2FormerHead = BaseModule

from .mmdet_mask2former import Decoupled_MMDET_Mask2FormerDecoder, Decoupled_MMDET_Mask2FormerHead

from mmengine.structures import InstanceData
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.utils import ConfigType, SampleList


class Decoupled_MMSEG_Mask2FormerDecoder(Decoupled_MMDET_Mask2FormerDecoder):
    """Implements the Mask2Former head.

    See `Mask2Former: Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/abs/2112.01527>`_ for details.

    Args:
        num_classes (int): Number of classes. Default: 150.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        ignore_index (int): The label index to be ignored. Default: 255.
    """

    def __init__(self,
                 align_corners=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.align_corners = align_corners

    def decode(self, x, batch_data_samples) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        x = x[1:]

        # forward
        decoder_out_list, all_mask_preds = self(x, batch_data_samples)

        return decoder_out_list, all_mask_preds

    
class Decoupled_MMSEG_Mask2FormerHead(Decoupled_MMDET_Mask2FormerHead):

    def __init__(self,
                 args,
                 num_classes = None,
                 **kwargs):
        super().__init__(args, **kwargs)

        self.num_classes = num_classes
        self.out_channels = num_classes
        self.ignore_index = args.ignore_label
        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

    
    def _seg_data_to_instance_data(self, batch_data_samples: SampleList):
        """Perform forward propagation to convert paradigm from MMSegmentation
        to MMDetection to ensure ``MMDET_Mask2FormerHead`` could be called
        normally. Specifically, ``batch_gt_instances`` would be added.

        Args:
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two lists.

                - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                    gt_instance. It usually includes ``labels``, each is
                    unique ground truth label id of images, with
                    shape (num_gt, ) and ``masks``, each is ground truth
                    masks of each instances of a image, shape (num_gt, h, w).
                - batch_img_metas (list[dict]): List of image meta information.
        """
        batch_img_metas = []
        batch_gt_instances = []

        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            gt_sem_seg = data_sample.gt_sem_seg.data.long().cuda()
            classes = torch.unique(
                gt_sem_seg,
                sorted=False,
                return_inverse=False,
                return_counts=False)

            #print('classes', classes)

            # remove ignored region
            gt_labels = classes[classes != self.ignore_index]

            #print('classes', classes)

            masks = []

            #print('gt_labels', gt_labels)

            for class_id in gt_labels:
                masks.append(gt_sem_seg == class_id)

            if len(masks) == 0:
                gt_masks = torch.zeros(
                    (0, gt_sem_seg.shape[-2],
                     gt_sem_seg.shape[-1])).to(gt_sem_seg).long().cuda()
            else:
                gt_masks = torch.stack(masks).squeeze(1).long().cuda()

            instance_data = InstanceData(labels=gt_labels, masks=gt_masks)
            batch_gt_instances.append(instance_data)
        return batch_gt_instances, batch_img_metas
    
    def _forward_cls_head(self, decoder_out_list: List) -> Tuple[Tensor]:

        all_cls_preds = []

        for x in decoder_out_list:
        
            cls_pred = self.cls_embed(x)

            all_cls_preds.append(cls_pred)

        return all_cls_preds

    def loss(self, batch_data_samples, decoder_out_list, all_mask_preds) -> dict:

         # batch SegDataSample to InstanceDataSample
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
             batch_data_samples)

        all_cls_preds = self._forward_cls_head(decoder_out_list)

        #print('cls_pred', cls_pred.shape)

        # loss
        losses = self.loss_by_feat(all_cls_preds, all_mask_preds,
                                   batch_gt_instances, batch_img_metas)
        
        #print('##############',losses)

        return losses
    
    def predict(self, batch_img_metas, decode_out_list, all_mask_preds) -> Tuple[Tensor]:
        # x = x[1:]

        # batch_data_samples = [
        #     SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        # ]

        # all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        mask_cls_results = self.cls_embed(decode_out_list[-1])
        mask_pred_results = all_mask_preds[-1]
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']
        # upsample mask
        ##################### changed: size[:-1]
        mask_pred_results = F.interpolate(
            mask_pred_results, size=size[:-1], mode='bilinear', align_corners=False)
        cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)
        return seg_logits
    


