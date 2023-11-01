import torch
import torch.nn as nn
from typing import Optional, Union, List
#from backbone.resnet import Our_ResNet
from backbone.resnet_mmseg_10 import ResNet
from backbone.swin import swin
from backbone.ViTAE_Window_NoShift.base_model import vitaev2_s
from backbone.vit_win_rvsa_v3_wsz7 import vit_b_rvsa
from backbone.intern_image import InternImage
from mask2former_decoder.mmseg_mask2former import Decoupled_MMSEG_Mask2FormerDecoder, Decoupled_MMSEG_Mask2FormerHead
from modules import Activation

from mmengine.config import Config

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SemsegPretrnFramework(torch.nn.Module):
    def __init__(self, 
                  args, 
                  decoder_use_batchnorm: bool = True,
                  decoder_channels: List[int] = (512, 256, 128, 64), #(256, 128, 64, 32, 16),
                  decoder_attention_type: Optional[str] = None,
                  classes1: int = 1,
                  classes2: int = 1,
                  classes3: int = 1,
                  activation: Optional[Union[str, callable]] = None,
                  aux_params: Optional[dict] = None):
        super(SemsegPretrnFramework, self).__init__()

        self.args = args

        # encoder

        if args.backbone == 'resnet50':
            #self.encoder = Our_ResNet()
            self.encoder = ResNet(50, out_indices=(0, 1, 2, 3), norm_eval=False)
            print('################# Using ResNet-50 as backbone! ###################')
        elif args.backbone == 'swint':
            self.encoder = swin(embed_dim=96, 
                                           depths=[2, 2, 6, 2],
                                           num_heads=[3, 6, 12, 24],
                                           window_size=7,
                                           ape=False,
                                           drop_path_rate=0.3,
                                           patch_norm=True
                                           )
            print('################# Using Swin-T as backbone! ###################')
        elif args.backbone == 'vitaev2_s':
            self.encoder = vitaev2_s()
            print('################# Using ViTAEv2 of dpr=0.1 as backbone! ###################')
        elif args.backbone == 'vit_b_rvsa':
            self.encoder = vit_b_rvsa(args)
            print('################# Using ViT-B + RVSA as backbone! ###################')
        elif args.backbone == 'internimage_t':
            self.encoder = InternImage(core_op='DCNv3',
                            channels=64,
                            depths=[4, 4, 18, 4],
                            groups=[4, 8, 16, 32],
                            mlp_ratio=4.,
                            drop_path_rate=0.2,
                            norm_layer='LN',
                            layer_scale=1.0,
                            offset_scale=1.0,
                            post_norm=False,
                            with_cp=False,
                            out_indices=(0, 1, 2, 3)
                            )
            print('################# Using InternImage-T as backbone! ###################')
        
        # decoder
        if args.decoder == 'mask2former':

            self.decoder = Decoupled_MMSEG_Mask2FormerDecoder(
                        in_channels=self.encoder.out_channels[1:],
                        strides=[4, 8, 16, 32],
                        feat_channels=256,
                        out_channels=256,
                        num_queries=100,
                        num_transformer_feat_level=3,
                        align_corners=False,
            )

            num_classes1 = classes1

            self.semseghead_1=Decoupled_MMSEG_Mask2FormerHead(
                args,
                num_classes = num_classes1,
                feat_channels=256,
                num_queries = 100,
                loss_cls=Config(dict(
                        type='mmdet.CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=2.0,
                        reduction='mean',
                        ignore_index = args.ignore_label,
                        class_weight=[1.0] * num_classes1 + [0.1])),
                
            )

            num_classes2 = classes2

            self.semseghead_2=Decoupled_MMSEG_Mask2FormerHead(
                args,
                num_classes = num_classes2,
                feat_channels=256,
                num_queries = 100,
                loss_cls=Config( dict(
                        type='mmdet.CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=2.0,
                        reduction='mean',
                        ignore_index = args.ignore_label,
                        class_weight=[1.0] * num_classes2 + [0.1])),
                
            )

            num_classes3 = classes3

            self.semseghead_3=Decoupled_MMSEG_Mask2FormerHead(
                args,
                num_classes = num_classes3,
                feat_channels=256,
                num_queries = 100,
                loss_cls=Config(dict(
                        type='mmdet.CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=2.0,
                        reduction='mean',
                        ignore_index = args.ignore_label,
                        class_weight=[1.0] * num_classes3 + [0.1])),
                
            )

            print('################# Using Mask2Former for Pretraining! ######################')

        if args.backbone == 'resnet50':
            if args.init_backbone == 'rsp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/rsp-resnet-50-ckpt.pth')
                print('################# Initing ResNet-50 RSP pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'imp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/resnet50-19c8e357.pth')
                print('################# Initing ResNet-50 IMP pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ResNet-50 SEP finetuning! ###################')
            else:
                raise NotImplementedError
        elif args.backbone == 'swint':
            if args.init_backbone == 'rsp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/rsp-swin-t-ckpt.pth')
                print('################# Initing Swin-T RSP pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'imp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/swin_tiny_patch4_window7_224.pth')
                print('################# Initing Swin-T IMP pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure Swin-T SEP finetuning! ###################')
            else:
                raise NotImplementedError
        elif 'vitaev2' in args.backbone:
            if args.init_backbone == 'rsp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/rsp-vitaev2-s-ckpt.pth')
                print('################# Initing ViTAEv2 RSP pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'imp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/ViTAEv2-S.pth.tar')
                print('################# Initing VITAEv2 IMP pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ViTAEv2 SEP finetuning! ###################')
            else:
                raise NotImplementedError
        elif args.backbone == 'vit_b_rvsa':
            if args.init_backbone == 'mae':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/vit-b-checkpoint-1599.pth')
                print('################# Initing ViT-B + RVSA pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ViT-B + RVSA SEP finetuning! ###################')
            else:
                raise NotImplementedError

        elif args.backbone == 'internimage_t':
            if args.init_backbone == 'imp':
                self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/internimage_t_1k_224.pth')
                print('################# Initing InterImage-T pretrained weights for Finetuning! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure InterImage-T SEP finetuning! ###################')
            else:
                raise NotImplementedError

        self.initialize()

    def forward(self, x1, x2, x3, y1=None, y2=None, y3=None):

        output = []

        o1 = o2 = o3 = 0

        if 'sota' in self.args.datasets:

            x1 = self.encoder(x1)

            d1, a1 = self.decoder.decode(x1, y1)

            if self.training:

                o1 = self.semseghead_1.loss(y1, d1, a1)
            else:
                # in testing, y doesn't contain ground truth map
                batch_img_metas = []
                for data_sample in y1:
                    batch_img_metas.append(data_sample.metainfo)
                o1 = self.semseghead_1.predict(batch_img_metas, d1, a1)
        output.append(o1)

        if 'sior' in self.args.datasets:

            x2 = self.encoder(x2)

            d2, a2 = self.decoder.decode(x2, y2)

            if self.training:

                o2 = self.semseghead_2.loss(y2, d2, a2)
            else:
                # in testing, y doesn't contain ground truth map
                batch_img_metas = []
                for data_sample in y2:
                    batch_img_metas.append(data_sample.metainfo)
                o2 = self.semseghead_2.predict(batch_img_metas, d2, a2)
        output.append(o2)
        
        if 'fast' in self.args.datasets:

            x3 = self.encoder(x3)

            d3, a3 = self.decoder.decode(x3, y3)

            if self.training:

                o3 = self.semseghead_3.loss(y3, d3, a3)
            else:
                # in testing, y doesn't contain ground truth map
                batch_img_metas = []
                for data_sample in y3:
                    batch_img_metas.append(data_sample.metainfo)
                o3 = self.semseghead_3.predict(batch_img_metas, d3, a3)
        output.append(o3)

        return output
        
    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.semseghead_1)
        initialize_head(self.semseghead_2)
        initialize_head(self.semseghead_3)


class SemsegFinetuneFramework(torch.nn.Module):
    
    def __init__(self, 
                  args, 
                  inchannels: int = 3, 
                  decoder_use_batchnorm: bool = True,
                  decoder_attention_type: Optional[str] = None,
                  classes: int = 1,
                  activation: Optional[Union[str, callable]] = None,
                  aux_params: Optional[dict] = None):
        super(SemsegFinetuneFramework, self).__init__()

        # encoder

        self.args = args
        if args.backbone == 'resnet50':
            #self.encoder = Our_ResNet()
            self.encoder = ResNet(50, out_indices=(0, 1, 2, 3), norm_eval=False)
            print('################# Using ResNet-50 as backbone! ###################')
        elif args.backbone == 'swint':
            self.encoder = swin(embed_dim=96, 
                                           depths=[2, 2, 6, 2],
                                           num_heads=[3, 6, 12, 24],
                                           window_size=7,
                                           ape=False,
                                           drop_path_rate=0.3,
                                           patch_norm=True
                                           )
            print('################# Using Swin-T as backbone! ###################')
        elif args.backbone == 'vitaev2_s':
            self.encoder = vitaev2_s()
            print('################# Using ViTAEv2 of dpr=0.1 as backbone! ###################')
        elif args.backbone == 'vit_b_rvsa':
            self.encoder = vit_b_rvsa(args, inchannels=inchannels)
            print('################# Using ViT-B + RVSA as backbone! ###################')
        elif args.backbone == 'internimage_t':
            self.encoder = InternImage(core_op='DCNv3',
                            channels=64,
                            depths=[4, 4, 18, 4],
                            groups=[4, 8, 16, 32],
                            mlp_ratio=4.,
                            drop_path_rate=0.2,
                            norm_layer='LN',
                            layer_scale=1.0,
                            offset_scale=1.0,
                            post_norm=False,
                            with_cp=False,
                            out_indices=(0, 1, 2, 3)
                            )
            print('################# Using InternImage-T as backbone! ###################')
        # decoder
        if args.decoder == 'mask2former':

            self.decoder = Decoupled_MMSEG_Mask2FormerDecoder(
                        in_channels=self.encoder.out_channels[1:],
                        strides=[4, 8, 16, 32],
                        feat_channels=256,
                        out_channels=256,
                        num_queries=100,
                        num_transformer_feat_level=3,
                        align_corners=False,
            )

            self.semseghead = Decoupled_MMSEG_Mask2FormerHead(
                args,
                num_classes = classes,
                feat_channels=256,
                num_queries = 100,
                loss_cls=Config(dict(
                        type='mmdet.CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=2.0,
                        reduction='mean',
                        ignore_index = args.ignore_label,
                        class_weight=[1.0] * classes + [0.1])),
                
            )

            print('################# Using Mask2Former for Finetuning! ######################')

        if args.load == 'backbone':

            if args.backbone == 'resnet50':
                if args.init_backbone == 'rsp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/rsp-resnet-50-ckpt.pth')
                    print('################# Initing ResNet-50 RSP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'imp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/resnet50-19c8e357.pth')
                    print('################# Initing ResNet-50 IMP pretrained weights for Finetuning! ###################')
                else:
                    raise NotImplementedError
            elif args.backbone == 'swint':
                if args.init_backbone == 'rsp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/rsp-swin-t-ckpt.pth')
                    print('################# Initing Swin-T RSP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'imp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/swin_tiny_patch4_window7_224.pth')
                    print('################# Initing Swin-T IMP pretrained weights for Finetuning! ###################')
                else:
                    raise NotImplementedError
            elif 'vitaev2' in args.backbone:
                if args.init_backbone == 'rsp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/rsp-vitaev2-s-ckpt.pth')
                    print('################# Initing ViTAEv2 RSP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'imp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/ViTAEv2-S.pth.tar')
                    print('################# Initing VITAEv2 IMP pretrained weights for Finetuning! ###################')
                else:
                    raise NotImplementedError
            elif args.backbone == 'vit_b_rvsa':
                if args.init_backbone == 'mae':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/vit-b-checkpoint-1599.pth')
                    print('################# Initing ViT-B + RVSA pretrained weights for Finetuning! ###################')
                else:
                    raise NotImplementedError
            elif args.backbone == 'internimage_t':
                if args.init_backbone == 'imp':
                    self.encoder.init_weights('/work/home/acdtzwus6v/dw/pretrn/internimage_t_1k_224.pth')
                    print('################# Initing InterImage-T pretrained weights for Finetuning! ###################')
                else:
                    raise NotImplementedError
        else:
            print('################# Load network for finetuning! ###################')
            pass

        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.semseghead)

    def forward(self, x, y=None):

        features = self.encoder(x)
        d, a = self.decoder.decode(features, y)

        if self.training:
            output = self.semseghead.loss(y, d, a)
        else:
            # in testing, y doesn't contain ground truth map
            batch_img_metas = []
            for data_sample in y:
                batch_img_metas.append(data_sample.metainfo)
            output = self.semseghead.predict(batch_img_metas, d, a)
        return output









