
<h1 align="center"> SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model </h1>

# Results

## Segmentation Pretrained Models

| Pretrain | Backbone | Decoder | Weights |
| :----- | :----- | :----- | :-----:|
| IMP + SEP | ResNet-50 | UNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclq6p_ic9dg8QhMtg?e=dRmqdy) |
| IMP + SEP | ResNet-50 | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclrvwsXrhGR0kN0tg?e=fbE39F) | 
| IMP + SEP | Swin-T | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclvnpQmjnm8feMr3Q?e=cvNLgb) | 
| IMP + SEP | ViTAEv2-S | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcluCxAmKWYwjEi44g?e=3yXpFZ) |
| IMP + SEP | InternImage-T | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclttXw6eAw_xI8y9Q?e=odCb5A) |
| IMP + SEP | ViT-B | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclxUGEZ4aIY5oR3gg?e=p1S93m) |
| IMP + SEP | ViT-Adapter-B | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclyZSfoCA0BMnFiQw?e=TLt1qI) | 
| RSP + SEP | ResNet-50 | UNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclpSeyfreAN0hO66Q?e=6r7tjp)  | 
| RSP + SEP | ResNet-50 | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclsik11TTIT7L_VWw?e=nF8ws4) | 
| BEiT + SEP | ViT-B | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclw8bK-fkLZTQq_sA?e=WpDKbd) |
| MAE  + SEP | ViT-B + RVSA | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcl0gojng2CzeHVohQ?e=Ynr6Aq)  | 
| SAMRS-MAE + SEP | ViT-B + RVSA | UperNet | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcl1XGhIPz9NIfBSqg?e=9sovVo) | 
| IMP + SEP | ResNet-50 | Mask2Former | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgclzhjljnG4Q2S2xKQ?e=I8jcOw) |

## SAMRS-MAE Pretrained Model

| Pretrain | Backbone | Clipping | Weights |
| :----- | :----- | :----- | :-----:|
| MAE | ViT-B |   | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcl2bCi1g_0oI0A9Qw?e=D5uqI4) | 
| MAE | ViT-B | √  | [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcl318C5gq17UQgzOA?e=IDkLIh) | 
# Usages

## Environment:
| Package | Version | Package | Version | Package | Version |
| ----- | :-----: | ----- | :-----: | ----- | :-----: |
| Python | 3.8.17 | timm | 0.9.5 | MMEngine | 0.8.4 |
| Pytorch | 1.10.0 | OpenCV | 4.8.0 | MMSegmentation |1.0.0 |
| Torchvision | 0.10.0 | MMCV | 2.0.0 | MMDetection | 3.1.0 |

## Pretraining and Finetuning

Performing the **segmentation pretraining (SEP)** with different segmention networks on the SAMRS dataset. Here is an example of using ImageNet pretrained UperNet-ResNet-50:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=10001 --master_addr = [server ip] main_pretrain.py \
    --backbone 'resnet50' --decoder 'upernet' \
    --datasets 'sota' 'sior' 'fast' \
    --batch_size 12 --batch_size_val 12 --workers 8 \
    --save_path '[SEP model save path]' \
    --distributed 'True' --end_iter 80000 \
    --image_size 224 --init_backbone 'imp'
```

**Finetuning** the above segmentation pretrained model on different datasets, examplified by ISPRS Potsdam. Here, we manully set the epoch to satisfy 80000 iters.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=10002 --master_addr = [server ip] main_finetune.py \
    --backbone 'resnet50' --decoder 'upernet' \
    --dataset 'potsdam' \
    --batch_size 1 --batch_size_val 1 --workers 8 \
    --save_path [funetuned model save path] \
    --distributed 'True' --epochs 75 --interval 15 \
    --load 'network' --init_backbone 'none' \
    --optim 'adamw' \
    --ft 'False' --resume [SEP model save path]
```

Predicting the fintuned model on the testing set:

```
CUDA_VISIBLE_DEVICES=0 python test_gpu.py \
    --backbone 'resnet50' --decoder 'upernet' \
    --dataset 'potsdam' \
    --ms 'False' --mode 'test' \
    --resume [funetuned model save path] \
    --save_path [prediction save path]
```

## Thanks

[MMsegmentation](https://github.com/open-mmlab/mmsegmentation), [MMDetection](https://github.com/open-mmlab/mmdetection), [segmentation_models.pytorch
](https://github.com/qubvel/segmentation_models.pytorch), [InternImage](https://github.com/OpenGVLab/InternImage), [ViT-Adapter](https://github.com/czczup/ViT-Adapter)