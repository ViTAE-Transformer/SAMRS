import torch
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET
from utils.transform import obb2poly_np
from mapping import DIOR

dior_cls2lbl = {k:v for v,k in enumerate(DIOR)}

def load_dior(img_name, ann_path):

    error = 0
    # hbox
    hbox_xml_path = osp.join(ann_path,f'{img_name}.xml')
    hbox_tree = ET.parse(hbox_xml_path)
    hbox_root = hbox_tree.getroot()

    gt_hboxes = []
    gt_points = []
    gt_labels = []

    for obj in hbox_root.findall('object'):
        category = str(obj.find('name').text.lower())
        bndbox = obj.find('bndbox')
        if not bndbox:
            bndbox = obj.find('robndbox') # some xml use robndbox
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        hbox = np.array([xmin, ymin, xmax, ymax],dtype=np.float32)
        gt_hboxes.append(hbox)
        gt_points.append(np.array([(xmin+xmax)/2, (ymin+ymax)/2]))
        gt_labels.append(int(dior_cls2lbl[category]))
    
    if len(gt_hboxes) == 0:
        error = 1

    return gt_hboxes, gt_points, gt_labels, error
    
def load_hrsc(img_name, ann_path):

    error = 0

    xml_path = osp.join(ann_path,f'{img_name}.xml')
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_hboxes = []
    gt_rboxes = []
    gt_colors = []
    gt_points = []

    for obj in root.findall('HRSC_Objects/HRSC_Object'):

        hbox = np.array([
            float(obj.find('box_xmin').text),
            float(obj.find('box_ymin').text),
            float(obj.find('box_xmax').text),
            float(obj.find('box_ymax').text)],
                        dtype=np.float32)

        rbox = np.array([[
            float(obj.find('mbox_cx').text),
            float(obj.find('mbox_cy').text),
            float(obj.find('mbox_w').text),
            float(obj.find('mbox_h').text),
            float(obj.find('mbox_ang').text), 0
        ]],
                        dtype=np.float32)

        # cx,cy,w,h -> dota format 
        polygon = obb2poly_np(rbox, 'le90')[0, :-1].astype(np.float32)
        polygon = polygon.reshape(-1, 2)

        color_list = obj.find('seg_color').text.split(',')

        if len(color_list) != 3:
            error = 1
            color = np.array([0,0,0],dtype=np.uint8)
        else:
            r,g,b = color_list
            color = np.array([int(r),int(g),int(b)],dtype=np.uint8)

        cpoint = np.array([
            float(obj.find('mbox_cx').text),
            float(obj.find('mbox_cy').text)],
            dtype = np.float32)

        gt_hboxes.append(hbox)
        gt_rboxes.append(polygon)
        gt_colors.append(color)
        gt_points.append(cpoint)

    # single class

    gt_labels = [0 for i in range(len(gt_rboxes))]

    if len(gt_hboxes) == 0 or len(gt_rboxes) == 0:
        error = 1

    return gt_hboxes, gt_rboxes, gt_colors, gt_points, gt_labels, error

def load_dota(img_name, ann_path):

    error = 0

    f = open(osp.join(ann_path,img_name+'.txt'),'r')

    all_infos = f.readlines()

    f.close()

    gt_hboxes = []
    gt_rboxes = []
    gt_points = []
    gt_classes = []
    gt_labels = []

    for box_info in all_infos:
        x1, y1, x2, y2, x3, y3, x4, y4, class_name, class_index = box_info.strip().split()
        x1, y1, x2, y2, x3, y3, x4, y4 = float(x1),float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)
        gt_hboxes.append(np.array([x1, y1, x3, y3]))
        gt_rboxes.append(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        gt_points.append(np.array([(x1+x3)/2, (y1+y3)/2]))
        gt_classes.append(class_name)
        gt_labels.append(int(class_index))
    
    if len(gt_hboxes) == 0 or len(gt_rboxes) == 0:
        error = 1

    return gt_hboxes, gt_rboxes, gt_points, gt_labels, error

