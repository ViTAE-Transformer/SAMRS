import torch
import torch.nn as nn
from typing import Optional, Union, List
#from backbone.resnet import Our_ResNet
from backbone.resnet_mmseg_10 import ResNet
from backbone.swin import swin
from backbone.ViTAE_Window_NoShift.base_model import vitaev2_s
from backbone.vit_win_rvsa_v3_wsz7 import vit_b_rvsa
from backbone.intern_image import InternImage
from backbone.vit_adapter import ViTAdapter
from backbone.vit import ViT
from unet import UnetDecoder
from unetpp import UnetPlusPlusDecoder
from modules import Activation

from upernet_mmseg_30 import UPerHead

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
        
        elif args.backbone == 'vitadapter_b':
            self.encoder = ViTAdapter(
                            patch_size=16,
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            mlp_ratio=4,
                            drop_path_rate=0.3,
                            conv_inplane=64,
                            n_points=4,
                            deform_num_heads=12,
                            cffn_ratio=0.25,
                            deform_ratio=0.5,
                            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                            window_attn=[False] * 12,
                            window_size=[None] * 12
                            )
            print('################# Using ViT-Adapter-B as backbone! ###################')

        elif args.backbone == 'vit_b':

            self.encoder = ViT(
                            img_size=args.image_size,
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            mlp_ratio=4,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            drop_path_rate=0.15,
                            use_abs_pos_emb=True,
                            )
            print('################# Using ViT-B as backbone! ###################')

        # decoder
        if args.decoder == 'unet':

            self.decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels, # (3, 64, 256, 512, 1024, 2048), (3, 64, 128, 256, 512)
                decoder_channels=decoder_channels,
                n_blocks=4,
                use_batchnorm=decoder_use_batchnorm,
                center=False,
                attention_type=decoder_attention_type,
            )

            self.semseghead_1 = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes1,
                activation=activation,
                kernel_size=3,
            )
            self.semseghead_2 = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes2,
                activation=activation,
                kernel_size=3,
            )
            self.semseghead_3 = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes3,
                activation=activation,
                kernel_size=3,
            )

            print('################# Using UNet for Pretraining! ######################')

        elif args.decoder == 'upernet':

            self.decoder = UPerHead(
                in_channels = self.encoder.out_channels[1:],
                channels = self.encoder.out_channels[2],
                in_index = (0, 1, 2, 3),
                dropout_ratio = 0.1,
                norm_cfg = dict(type='SyncBN', requires_grad=True)
            )

            self.semseghead_1 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(self.encoder.out_channels[2], classes1, kernel_size=1)
            )

            self.semseghead_2 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(self.encoder.out_channels[2], classes2, kernel_size=1)
            )

            self.semseghead_3 = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(self.encoder.out_channels[2], classes3, kernel_size=1)
            )

            print('################# Using UperNet for Pretraining! ######################')

        if args.backbone == 'resnet50':
            if args.init_backbone == 'rsp':
                self.encoder.init_weights('/pretrn/rsp-resnet-50-ckpt.pth')
                print('################# Initing ResNet-50 RSP pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'imp':
                self.encoder.init_weights('/pretrn/resnet50-19c8e357.pth')
                print('################# Initing ResNet-50 IMP pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ResNet-50 SEP Pretraining! ###################')
            else:
                raise NotImplementedError
        elif args.backbone == 'swint':
            if args.init_backbone == 'rsp':
                self.encoder.init_weights('/pretrn/rsp-swin-t-ckpt.pth')
                print('################# Initing Swin-T RSP pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'imp':
                self.encoder.init_weights('/pretrn/swin_tiny_patch4_window7_224.pth')
                print('################# Initing Swin-T IMP pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure Swin-T SEP Pretraining! ###################')
            else:
                raise NotImplementedError
        elif 'vitaev2' in args.backbone:
            if args.init_backbone == 'rsp':
                self.encoder.init_weights('/pretrn/rsp-vitaev2-s-ckpt.pth')
                print('################# Initing ViTAEv2 RSP pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'imp':
                self.encoder.init_weights('/pretrn/ViTAEv2-S.pth.tar')
                print('################# Initing VITAEv2 IMP pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ViTAEv2 SEP Pretraining! ###################')
            else:
                raise NotImplementedError
        elif args.backbone == 'vit_b_rvsa':
            if args.init_backbone == 'mae':
                self.encoder.init_weights('/pretrn/vit-b-checkpoint-1599.pth')
                print('################# Initing ViT-B + RVSA pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'samrs-mae-expand':
                self.encoder.init_weights('/work_dir/mae_pretrain/vit_base_norm_pix_expand/samrs_224/pretrn_weights/checkpoint-1599.pth')
                print('################# Initing ViT-B + RVSA pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure ViT-B + RVSA SEP Pretraining! ###################')
            else:
                raise NotImplementedError

        elif args.backbone == 'internimage_t':
            if args.init_backbone == 'imp':
                self.encoder.init_weights('/pretrn/internimage_t_1k_224.pth')
                print('################# Initing InterImage-T pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'none':
                print('################# Pure InterImage-T SEP Pretraining! ###################')
            else:
                raise NotImplementedError
            
        elif args.backbone == 'vitadapter_b' or args.backbone == 'vit_b':
            if args.init_backbone == 'imp':
                self.encoder.init_weights('/pretrn/deit_base_patch16_224-b5f2ef4d.pth')
                print('################# Initing ViT-Adapter-B pretrained weights for Pretraining! ###################')
            elif args.init_backbone == 'beit':
                self.encoder.init_weights('/pretrn/beit_base_patch16_224_pt22k_ft22k.pth')
                print('################# Initing ViT-Adapter-B pretrained weights for Pretraining! ###################') 
            elif args.init_backbone == 'none':
                print('################# Pure ViT-Adapter-B SEP Pretraining! ###################')
            else:
                raise NotImplementedError

        self.initialize()

    def forward(self, x1, x2, x3):

        output = []

        o1 = o2 = o3 = 0

        if 'sota' in self.args.datasets:

            x1 = self.encoder(x1)

            x1 = self.decoder(*x1)
            o1 = self.semseghead_1(x1)

        output.append(o1)
        
        if 'sior' in self.args.datasets:

            x2 = self.encoder(x2)

            x2 = self.decoder(*x2)
            o2 = self.semseghead_2(x2)

        output.append(o2)
        
        if 'fast' in self.args.datasets:

            x3 = self.encoder(x3)

            x3 = self.decoder(*x3)
            o3 = self.semseghead_3(x3)

        output.append(o3)

        #features = self.encoder(x)

        #img, o0, o1, o2, o3 = self.encoder(x)

        #output = self.decoder(*features)

        #output = self.decoder(img, o0, o1, o2, o3)

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

        elif args.backbone == 'vitadapter_b':
            self.encoder = ViTAdapter(
                            patch_size=16,
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            mlp_ratio=4,
                            drop_path_rate=0.3,
                            conv_inplane=64,
                            n_points=4,
                            deform_num_heads=12,
                            cffn_ratio=0.25,
                            deform_ratio=0.5,
                            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
                            window_attn=[False] * 12,
                            window_size=[None] * 12
                            )
            print('################# Using ViT-Adapter-B as backbone! ###################')

        elif args.backbone == 'vit_b':

            self.encoder = ViT(
                            img_size=args.image_size,
                            embed_dim=768,
                            depth=12,
                            num_heads=12,
                            mlp_ratio=4,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            drop_path_rate=0.15,
                            use_abs_pos_emb=True,
                            )
            print('################# Using ViT-B as backbone! ###################')

        # decoder

        if args.decoder == 'unet':

            decoder_channels = (512, 256, 128, 64)

            self.decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels, # (3, 64, 256, 512, 1024, 2048), (3, 64, 128, 256, 512)
                decoder_channels=decoder_channels,
                n_blocks=4,
                use_batchnorm=decoder_use_batchnorm,
                center=False,
                attention_type=decoder_attention_type,
            )

            self.semseghead = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )

            print('################# Using UNet for Finetuning! ######################')

        elif args.decoder == 'upernet':

            self.decoder = UPerHead(
                in_channels = self.encoder.out_channels[1:],
                channels = self.encoder.out_channels[2],
                in_index = (0, 1, 2, 3),
                dropout_ratio = 0.1,
                norm_cfg = dict(type='SyncBN', requires_grad=True)
            )

            self.semseghead = nn.Sequential(
                nn.Dropout2d(0.1),
                nn.Conv2d(self.encoder.out_channels[2], classes, kernel_size=1)
            )

            print('################# Using UperNet for Finetuning! ###################')

        if args.load == 'backbone':

            if args.backbone == 'resnet50':
                if args.init_backbone == 'rsp':
                    self.encoder.init_weights('/pretrn/rsp-resnet-50-ckpt.pth')
                    print('################# Initing ResNet-50 RSP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'imp':
                    self.encoder.init_weights('/pretrn/resnet50-19c8e357.pth')
                    print('################# Initing ResNet-50 IMP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'none':
                    pass 
                else:
                    raise NotImplementedError
            elif args.backbone == 'swint':
                if args.init_backbone == 'rsp':
                    self.encoder.init_weights('/pretrn/rsp-swin-t-ckpt.pth')
                    print('################# Initing Swin-T RSP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'imp':
                    self.encoder.init_weights('/pretrn/swin_tiny_patch4_window7_224.pth')
                    print('################# Initing Swin-T IMP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'none':
                    pass 
                else:
                    raise NotImplementedError
            elif 'vitaev2' in args.backbone:
                if args.init_backbone == 'rsp':
                    self.encoder.init_weights('/pretrn/rsp-vitaev2-s-ckpt.pth')
                    print('################# Initing ViTAEv2 RSP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'imp':
                    self.encoder.init_weights('/pretrn/ViTAEv2-S.pth.tar')
                    print('################# Initing VITAEv2 IMP pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'none':
                    pass
                else:
                    raise NotImplementedError
            elif args.backbone == 'vit_b_rvsa':
                if args.init_backbone == 'mae':
                    self.encoder.init_weights('/pretrn/vit-b-checkpoint-1599.pth')
                    print('################# Initing ViT-B + RVSA pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'samrs-mae-expand':
                    self.encoder.init_weights('/work_dir/mae_pretrain/vit_base_norm_pix_expand/samrs_224/pretrn_weights/checkpoint-1599.pth')
                    print('################# Initing ViT-B + RVSA pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'none':
                    pass
                else:
                    raise NotImplementedError

            elif args.backbone == 'internimage_t':
                if args.init_backbone == 'imp':
                    self.encoder.init_weights('/pretrn/internimage_t_1k_224.pth')
                    print('################# Initing InterImage-T pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'none':
                    pass
                else:
                    raise NotImplementedError

            elif args.backbone == 'vitadapter_b' or args.backbone == 'vit_b':
                if args.init_backbone == 'imp':
                    self.encoder.init_weights('/pretrn/deit_base_patch16_224-b5f2ef4d.pth')
                    print('################# Initing ViT-Adapter-B pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'beit':
                    self.encoder.init_weights('/pretrn/beit_base_patch16_224_pt22k_ft22k.pth')
                    print('################# Initing ViT-Adapter-B pretrained weights for Finetuning! ###################')
                elif args.init_backbone == 'none':
                    pass 
                else:
                    raise NotImplementedError
        else:
            print('################# Load network for finetuning! ###################')
            pass

        self.initialize()

    def initialize(self):
        initialize_decoder(self.decoder)
        initialize_head(self.semseghead)

    def forward(self, x):

        features = self.encoder(x)
        output = self.decoder(*features)
        output = self.semseghead(output)

        return output










