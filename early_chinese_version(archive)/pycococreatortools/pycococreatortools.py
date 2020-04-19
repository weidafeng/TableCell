#!/usr/bin/env python3

import os
import re
import datetime
import numpy as np
from itertools import groupby
from skimage import measure
from PIL import Image
from pycocotools import mask

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

def resize_binary_mask(array, new_size):
    ''' 从[0,1】 resize成【0,255】
    :param array:
    :param new_size:
    :return:
    '''
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
                counts.append(0)
        counts.append(len(list(elements)))

    return rle

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info

# 汇总 mask、bounding-box等信息
def mask_create_annotation_info(annotation_id, image_id, area, category_id, image_size=None, bounding_box=None,segmentation= None):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0, # 0或1，指定为0，表示“单个的对象（不存在多个对象重叠）”.只要是iscrowd=0那么segmentation就是polygon格式
        "area": area,  # area是area of encoded masks，是标注区域的面积。如果是矩形框，那就是高乘宽； 浮点数，需大于0，因icdar数据没有segmentation，所以本项人为指定为10
        "bbox": bounding_box,
        "segmentation": segmentation, #polygon格式.这些数按照相邻的顺序两两组成一个点的xy坐标，如果有n个数（必定是偶数），那么就是n/2个点坐标。
        # 注意这里，必须是list 包含list，底层的list中必须有至少6个元素，否则coco api会过滤掉这个annotations,也就是说你必须用至少三个点来表达一块。
        # 外层的list的长度取决于一个完整的物体是否被分割成了数块，比如一个物体苹果没有任何的遮挡，则外部的List长度就为1
        # 按照给出各个坐标的顺序描点（顺时针、逆时针都行）,eg:
        # gemfield_polygons1 = [[0,0,10,0,10,20,0,10]] # 逆时针
        # gemfield_polygons2 = [[0,0,0,10,10,20,10,0]] # 顺时针
        # gemfield_polygons3 = [[10,0,0,10,0,0,10,20]] # 注意次序，此时不是四边形，而是两个三角形

        "width": image_size[0],
        "height": image_size[1],
    }

    return annotation_info
