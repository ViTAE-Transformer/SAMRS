<h1 align="center"> SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model </h1>
<p align="center">
<a href="https://arxiv.org/abs/2305.02034"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>
<h5 align="center"><em>Di Wang, Jing Zhang, Bo Du, Minqiang Xu, Lin Liu, Dacheng Tao, Liangpei Zhang</em></h5>
<p align="center">
  <a href="#usage">Usage</a> |
  <a href="#usage">Results</a> |
</p>

# Usage

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
<img src="Figs/vis.png">
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
