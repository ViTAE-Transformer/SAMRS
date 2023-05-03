<h1 align="center"> Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model <a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a> </h1>
<p align="center">
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/">Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model</a>.</h4>
<h5 align="center"><em>Di Wang, Jing Zhang, Bo Du, Dacheng Tao, Liangpei Zhang</em></h5>
<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#method">Method</a> |
  <a href="#usage">Usage</a> |
  <a href="#results">Results</a> |
  <a href="#statement">Statement</a>
</p>







# News

***02/05/2023***

- The paper is post on arxiv! The code will be made public available once cleaned up.

- Relevant Project: 

  > [**An Empirical Study of Remote Sensing Pretraining** ](https://arxiv.org/abs/2204.02825) | [Code](https://github.com/ViTAE-Transformer/RSP)
  >
  > Di Wang, Jing Zhang, Bo Du, Gui-Song Xia and Dacheng Tao
  >
  > [**Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model** ](https://arxiv.org/abs/2208.03987) | [Code](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA)
  >
  > Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao and Liangpei Zhang

  Other applications of [ViTAE](https://github.com/ViTAE-Transformer/ViTAE-Transformer) inlcude: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) | [Remote Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing) | [Matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting) | [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA) | [Video Object Segmentation](https://github.com/ViTAE-Transformer/VOS-LLB)

# Abstract

<p align="left">The success of the <a href="https://arxiv.org/abs/2304.02643"> Segment Anything Model (SAM) </a> demonstrates the significance of data-centric machine learning. However, due to the difficulties and high costs associated with annotating Remote Sensing (RS) images, a large amount of valuable RS data remains unlabeled, particularly at the pixel level. In this study, we leverage SAM and existing RS object detection datasets to develop an efficient pipeline for generating a large-scale RS segmentation dataset, dubbed SAMRS. SAMRS surpasses existing high-resolution RS segmentation datasets in size by several orders of magnitude, and provides object category, location, and instance information that can be used for semantic segmentation, instance segmentation, and object detection, either individually or in combination. We also provide a comprehensive analysis of SAMRS from various aspects. We hope it could facilitate research in RS segmentation, particularly in large model pre-training.


# Usage
The code and dataset will be released soon.


# Results
# The Quality of Generated Masks

<figure>
<img src="figs/figure3.png">
<figcaption align = "center"><b>Figure 3: The distribution of IoU between the generated
masks and ground truth masks in the COCOText
training dataset:  <a href="https://arxiv.org/abs/1601.07140">COCO_Text V2</a>  
 </b></figcaption>
</figure>
We present the comparison of our SAMRS dataset with existing high-resolution RS segmentation datasets in table. Based on the available high-resolution RSI object detection datasets, we can efficiently annotate 10,5090 images, which is more than ten times the capacity of existing datasets. Additionally, SAMRS inherits the categories of the original detection datasets, which makes them more diverse than other high-resolution RS segmentation collections. It is worth noting that RS object datasets usually have more diverse categories than RS segmentation datasets due to the difficulty of tagging pixels in RSIs, and thus our SAMRS reduces this gap. 



# Visualization of Generated Masks



<figure>
<img src="figs/figure2.jpg">
<figcaption align = "center"><b>Figure 2: Some visualization results of the generated masks in five datasets using the SAMText
pipeline. The top row shows the scene text frames while the bottom row shows the generated masks.</a>  
 </b></figcaption>
</figure>

In figure, we visualize some segmentation annotations from the three subsets in our SAMRS dataset. As can be seen, SOTA exhibits a greater number of instances for tiny cars, whereas FAST provides a more fine-grained annotation of existing categories in SOTA such as car, ship, and plane. SIOR on the other hand, offers annotations for more diverse ground objects, such as *dam*. Hence, our SAMRS dataset encompasses a wide range of categories with varying sizes and distributions, thereby presenting a new challenge for RS semantic segmentation.







## Dataset Statistics and Analysis
### The size distribution.

<figure>
<img src="figs/figure4.png">
<figcaption align = "center"><b>Figure 4: (a) The mask size distributions of the ICDAR15, RoadText-1k, LSVDT, and DSText datasets.
Masks exceeding 10,000 pixels are excluded from the statistics. (b) The mask size distributions of
the BOVText datasets. Masks exceeding 80,000 pixels are excluded from the statistics.</a>  
 </b></figcaption>
</figure>



### The IoU and COV distribution.

<figure>
<img src="figs/figure5.png">
<figcaption align = "center"><b>Figure 5: (a) The distribution of IoU between the generated masks and ground truth bounding boxes
in each dataset. (b) The CoV distribution of mask size changes for the same individual in consecutive
frames in all five datasets, excluding the CoV scores exceeding 1.0 from the statistics.</a>  
 </b></figcaption>
</figure>



### The spatial distribution.

<figure>
<img src="figs/figure6.png">
<figcaption align = "center"><b>Figure 6: Visualization of the heatmaps that depict the spatial distribution of the generated masks in
the five video text spotting datasets employed to establish SAMText-9M.</a>  
 </b></figcaption>
</figure>



# Statement

This project is for research purpose only. For any other questions please contact [haibinhe@whu.edu.cn](mailto:haibinhe@whu.edu.cn).



## Citation

If you find SAMText helpful, please consider giving this repo a star:star: and citing:

```
@inproceedings{SAMText,
  title={Scalable Mask Annotation for Video Text Spotting},
  author={Haibin He, Jing Zhang, Mengyang Xu, Juhua Liu, Bo Du, Dacheng Tao},
  booktitle={arxiv},
  year={arXiv preprint arXiv:2305.01443}
}
```


