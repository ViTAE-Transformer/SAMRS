<h1 align="center"> SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model </h1>

# Usage

## Environment:

* Python 3.8.5
* Pytorch 1.9.0+cu111
* torchvision 0.10.0+cu111

## Main Function

- `main_sam_hbox_mask_instance.py/main_sam_hbox_mask_instance.py/main_sam_hbox_mask_instance.py`: Evaluating different prompts with corresponding boxes. They can additionally save the instance information or visualize related masks.

- `main_sam_hbox_semantic.py/main_sam_rhbox_semantic.py`: Generating semantic labels with corresponding boxes for obtaining segmentation datasets.

- `loaddata.py`: Loading annotation files of object detection datasets.

- `ann_transform.py`: Several functions for transforming annotataions.

- `visualize.py`: Visualizing the generated labels (Figure 4)

- `statistic.py`: Dataset analyses (Figure 5-6).

- `instance_to_json.py`: Saving the binary map containing instance mask to json format.

- `script.py`: Editing the pkl files of instances.


## Step

1. Downloading corresponding datasets (HRSC2016, DOTA, DIOR, FAIR1M) and preparing pretrained SAM model.

2. Evaluating the performances of different prompts or combinations on the HRSC2016 dataset by `main_sam_hbox_mask_instance.py/main_sam_hbox_mask_instance.py/main_sam_hbox_mask_instance.py`.

3. Preprocessing dataset:

  - DOTA: Clipping the image to patches with BBoxToolkit, here the labels will be saved in a pkl file. Then, transforming the pkl file to multiple *.txt files that contains the annotations of corresponding patches through `ann_transform.py`. 

  - FAIR1M: Firstly, transforming *.xml to *.txt of the DOTA format. Then, since FAIR1M training and validation sets have the same filenames, we rename them for merging together. These processes can be finished by `ann_transform.py`. The next steps are the same as DOTA.

4. Generating segmentation datasets with `main_sam_hbox_semantic.py/main_sam_rhbox_semantic.py`.
