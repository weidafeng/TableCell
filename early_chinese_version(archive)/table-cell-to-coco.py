# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         table-cell-to-coco.py
# Author:       wdf
# Date:         2019/7/20
# IDE：         PyCharm 
# Parameters:
#     @param:
#     @param:
# Return： 
#       
# Description:  把左上角、右下角坐标转换为coco格式
# Usage：
#-------------------------------------------------------------------------------

import datetime
import json
from pathlib import Path
import re
from PIL import Image
import numpy as np
import progressbar  #用于Python的文本进度条库。文本进度条通常用于显示长时间运行的操作的进度，提供正在进行处理的可视化提示。
from multiprocessing import pool
from pycococreatortools import pycococreatortools

ROOT_DIR = Path('/media/tristan/Files/dataset/TableBank/table-cell')
IMAGE_DIR = ROOT_DIR / Path('images')
ANNOTATION_DIR = ROOT_DIR / Path('labels')

INFO = {
    "description": "TABLE-CELL Dataset",
    "url": "https://github.com/weidafeng",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "DafengWei",
    "date_created": datetime.datetime.utcnow().isoformat(' ')  # 显示此刻时间，格式：'2019-04-30 02:17:49.040415'
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'cell',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'background',
        'supercategory': 'shape',
    }
    # {
    #     'id': 3,
    #     'name': 'ignore',
    #     'supercategory': 'shape',
    # }
]


# 获取 bounding-box， segmentation 信息
def get_info(content):
    # 输入content是list格式，
    # 获得bounding-box的坐标和内容, 以及segmentation信息：有序的四个点的坐标（bounding-box坐标）
    left, top  = float(content[1][0]), float(content[1][1])  # 左上角
    right, down = float(content[0][0]), float(content[0][1])  # 右下角

    height = down - top
    width = right-left
    #     word = content['word'] # 不考虑
    segmentation = [left,top, left + width, top, left + width, top + height, left, top + height] # 浮点形式
    return [left, top, width, height], [segmentation] # bounding-box信息， coco格式： x,y,w,h）；segmentation为[[1,2,3,4,5,6，7,8]]格式

def main():
    # coco lable文件（如training2017.json）需要存储的信息
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # 初始化id（以后依次加一）
    image_id = 1
    annotation_id = 1

    # 加载图片信息
    im_files = [f for f in IMAGE_DIR.iterdir()]
    im_files.sort(key=lambda f: f.stem,reverse=True)  # 以文件名排序，防止顺序错乱、数据和标签不对应
    # print("im-length:",len(im_files),"\n im_files：",im_files)

    # 加载annotation信息
    an_files = [f for f in ANNOTATION_DIR.iterdir()]
    an_files.sort(key=lambda f: f.stem,reverse=True)  # 以文件名排序，防止顺序错乱、数据和标签不对应
    # print("an-length:",len(an_files),"\n an_files：",an_files)

    assert len(an_files)==len(im_files), "图片数与lablel文件数目不匹配，请运行diff_two_folder.py，删除不匹配的文件" # 确保每个图片与label文件相匹配

    for im_file, an_file in zip(im_files, an_files):
        # 以coco格式，写入图片信息（id、图片名、图片大小）,其中id从1开始
        image = Image.open(im_file)
        im_info = pycococreatortools.create_image_info( image_id, im_file.name, image.size) # 图片信息
        coco_output['images'].append(im_info) # 存储图片信息（id、图片名、大小）
        myPool = pool.Pool(processes=16)  # 并行化处理

        # 开始处理label 信息
        annotation_info_list = []  # 存储单张图片的所有标注信息

        with open(an_file, 'r') as f:
            datas = json.load(f)
            for i in range(len(datas)):
                data = datas[i]
                # print(data)
                bounding_box = get_info(data)[0]
                segmentation = get_info(data)[1]  # 有序的四个点的坐标（bounding-box坐标）
                # print(bounding_box)
                # print(segmentation)

                class_id = 1  # label 数字形式

                # 显示日志
                print(bounding_box, segmentation)
                area = bounding_box[-1] * bounding_box[-2]  # 当前bounding-box的面积,宽×高
                # an_infos = pycococreatortools.mask_create_annotation_info(annotation_id=annotation_id, image_id=image_id, category_id=class_id, area=area, image_size=image.size, bounding_box=bounding_box,segmentation = segmentation)
                # annotation_info_list.append(an_infos)
                myPool.apply_async(func=pycococreatortools.mask_create_annotation_info,
                					args= (annotation_id, image_id, category_id, area, image.size, bounding_box, segmentation),
                					callbacks=annotation_info_list.append)
                annotation_id += 1

        myPool.close()
        myPool.join()
        # 上面得到单张图片的所有bounding-box信息，接下来每单张图片存储一次
        for annotation_info in annotation_info_list:
            if annotation_info is not None:
                coco_output['annotations'].append(annotation_info)
        image_id += 1

    # 保存成json格式
    print("保存annotations文件")
    output_json = Path(f'table_coco.json')
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(coco_output, f)
    print("Annotations JSON file saved in：", str(output_json))

if __name__ == "__main__":
    main()
