<h1 align="center"> Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model </h1>
<p align="center">
<a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>
<h5 align="center"><em>Di Wang, Jing Zhang, Bo Du, Dacheng Tao, Liangpei Zhang</em></h5>
<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#usage">Usage</a> |
  <a href="#results">Results</a> |
  <a href="#statement">Statement</a>
</p>




# News

***04/05/2023***

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

This is the official repository of the paper <a href="https://arxiv.org/abs/"> Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model </a>

<figure>
<img src="Figs/example.png">
<figcaption align = "center"><b>Figure 1: Some examples of SAM segmentation results on remote sensing images. 
 </b></figcaption>
</figure>

<p>

<p align="left">The success of the <a href="https://arxiv.org/abs/2304.02643"> Segment Anything Model (SAM) </a> demonstrates the significance of data-centric machine learning. However, due to the difficulties and high costs associated with annotating Remote Sensing (RS) images, a large amount of valuable RS data remains unlabeled, particularly at the pixel level. In this study, we leverage SAM and existing RS object detection datasets to develop an efficient pipeline for generating a large-scale RS segmentation dataset, dubbed SAMRS. SAMRS surpasses existing high-resolution RS segmentation datasets in size by several orders of magnitude, and provides object category, location, and instance information that can be used for semantic segmentation, instance segmentation, and object detection, either individually or in combination. We also provide a comprehensive analysis of SAMRS from various aspects. We hope it could facilitate research in RS segmentation, particularly in large model pre-training.


# Usage
The code and dataset will be released soon.


# Results
## The basic information of generated datasets

<figure>
<img src="Figs/compare.png">
<figcaption align = "center"><b>Figure 2: Comparisons of different high-resolution RS segmentation datasets. 
 </b></figcaption>
</figure>

<p>

We present the comparison of our SAMRS dataset with existing high-resolution RS segmentation datasets in table. Based on the available high-resolution RSI object detection datasets, we can efficiently annotate 10,5090 images, which is more than ten times the capacity of existing datasets. Additionally, SAMRS inherits the categories of the original detection datasets, which makes them more diverse than other high-resolution RS segmentation collections. It is worth noting that RS object datasets usually have more diverse categories than RS segmentation datasets due to the difficulty of tagging pixels in RSIs, and thus our SAMRS reduces this gap. 



## Visualization of Generated Masks



<figure>
<img src="Figs/vis.jpg">
<figcaption align = "center"><b>Figure 3: Some visual examples from the three subsets of our SAMRS dataset.  
 </b></figcaption>
</figure>

<p>

In figure, we visualize some segmentation annotations from the three subsets in our SAMRS dataset. As can be seen, SOTA exhibits a greater number of instances for tiny cars, whereas FAST provides a more fine-grained annotation of existing categories in SOTA such as car, ship, and plane. SIOR on the other hand, offers annotations for more diverse ground objects, such as *dam*. Hence, our SAMRS dataset encompasses a wide range of categories with varying sizes and distributions, thereby presenting a new challenge for RS semantic segmentation.



## Dataset Statistics and Analysis
### The class distribution.

<figure>
<img src="Figs/class.png">
<figcaption align = "center"><b>Figure 4: Statistics of the number of pixels and instances for each category in the SAMRS database. The histograms for the subsets SOTA, SIOR, and FAST are shown in the first, second, and third columns, respectively. The first row presents histograms on a per-pixel basis, while the second row presents histograms on a per-instance basis.</a>  
 </b></figcaption>
</figure>



### The mask size distribution.

<figure>
<img src="Figs/mask_size.png">
<figcaption align = "center"><b>Figure 5: Statistics of the mask sizes in different subsets of the SAMRS database. (a) SOTA. (b) SIOR. (c) FAST.</a>  
 </b></figcaption>
</figure>




# Statement

This project is for research purpose only. For any other questions please contact [d_wang@whu.edu.cn](mailto:d_wang@whu.edu.cn).



## Citation

If you find SAMRS helpful, please consider giving this repo a star:star: and citing:

```

```


